//! Wyoming event handler for TTS requests.
//!
//! Provides [`handle_connection`], an async function that runs the event
//! loop for a single Wyoming client connection: it reads events from an
//! [`AsyncBufRead`] source, dispatches `synthesize` requests to a
//! [`ModelRuntime`], and writes `audio-start`/`audio-chunk`/`audio-stop`
//! responses to an [`AsyncWrite`] sink.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use anyhow::Result;
use candle_core::DType;
use crane::audio::tts::pcm_f32_to_i16;
use crane_core::generation::SpeechOptions;
use crane_engine::{ModelRuntime, TtsGenerateRequest, TtsHandle};
use tokio::io::{AsyncBufRead, AsyncWrite};
use tokio::sync::oneshot;

use crate::event::{
    AudioChunkData, AudioStartData, AudioStopData, ErrorData, Event, InfoData, PingData, PongData,
    SynthesizeData,
};
use crate::wire::{read_event, write_event};

/// Maps voice names to TTS model registration names.
///
/// Built once at startup by scanning all registered TTS models' voices.
/// When two models define the same voice name, the model listed first in
/// `model_names` wins and the duplicate is logged as a warning (matching
/// the "first model on the command line wins" rule for Wyoming service
/// discovery).
pub struct VoiceMap {
    /// voice name -> model registration name
    map: HashMap<String, String>,
    /// Registration names of all configured models that were found in the
    /// runtime (used to distinguish "no voices left after dedup" from
    /// "not a configured model" in service discovery).
    model_names: HashSet<String>,
    /// Name of the default model, used when a `synthesize` event specifies
    /// no voice.
    default_model: Option<String>,
}

impl VoiceMap {
    /// Build a voice-to-model mapping from a [`ModelRuntime`].
    ///
    /// `model_names` gives model registration names in priority order
    /// (typically command-line `--model` order); the first model to claim
    /// a voice name wins. Names not found in `runtime` are skipped.
    #[must_use]
    pub fn new(model_names: &[String], runtime: &ModelRuntime) -> Self {
        let mut map = HashMap::new();
        let mut found_names = HashSet::new();
        for name in model_names {
            let Some(handle) = runtime.tts_handle(name) else {
                continue;
            };
            found_names.insert(name.clone());
            for voice in handle.voices() {
                match map.entry(voice.name.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(name.clone());
                    },
                    Entry::Occupied(entry) => {
                        tracing::warn!(
                            voice = %voice.name,
                            first_model = %entry.get(),
                            duplicate_model = %name,
                            "Duplicate voice name; using first model",
                        );
                    },
                }
            }
        }
        let default_model = runtime.default_tts_name().map(String::from);
        Self {
            map,
            model_names: found_names,
            default_model,
        }
    }

    /// Returns the model registration name for `voice_name`, if known.
    #[must_use]
    pub fn model_for_voice(&self, voice_name: &str) -> Option<&str> {
        self.map.get(voice_name).map(String::as_str)
    }

    /// Returns `true` if `name` is a configured model registration name.
    #[must_use]
    pub fn has_model(&self, name: &str) -> bool {
        self.model_names.contains(name)
    }

    /// Returns the default model's registration name, if any TTS model is loaded.
    #[must_use]
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }
}

/// Outcome of resolving a `synthesize` event's voice to a model.
enum VoiceResolution<'m, 'd> {
    /// A model was resolved. `voice_name` is `None` when the client did not
    /// request a specific voice (the model's default voice is used).
    Found {
        model_name: &'m str,
        voice_name: Option<&'d str>,
    },
    /// The client requested a voice name with no matching model.
    NotFound(&'d str),
    /// No voice was requested and no TTS model is loaded.
    NoModel,
}

/// Resolve which model (and voice) a `synthesize` event should use.
fn resolve_voice<'m, 'd>(
    voice_map: &'m VoiceMap,
    data: &'d SynthesizeData,
) -> VoiceResolution<'m, 'd> {
    if let Some(voice_name) = data.voice.as_ref().and_then(|v| v.name.as_deref()) {
        return match voice_map.model_for_voice(voice_name) {
            Some(model_name) => VoiceResolution::Found {
                model_name,
                voice_name: Some(voice_name),
            },
            None => VoiceResolution::NotFound(voice_name),
        };
    }
    match voice_map.default_model() {
        Some(model_name) => VoiceResolution::Found {
            model_name,
            voice_name: None,
        },
        None => VoiceResolution::NoModel,
    }
}

/// Run the Wyoming event loop for a single client connection.
///
/// Reads events from `reader` and dispatches them: `synthesize` requests
/// generate speech through `runtime`, `ping` is answered with `pong`,
/// `describe` gets an `info` response with available TTS model metadata, and unrecognized
/// event types get an `error` response. The loop continues until the
/// client disconnects cleanly (EOF) or the wire protocol desyncs.
///
/// Application-level failures (unknown voice, generation error) are
/// reported as Wyoming `error` events and do not terminate the
/// connection; only I/O errors and malformed wire data do.
///
/// # Cancellation
///
/// If this future is dropped (e.g. the TCP connection resets) while a
/// TTS request is in flight, the request's oneshot response channel is
/// dropped. The TTS thread checks for this before and after generation
/// and discards the result instead of blocking or erroring.
///
/// # Errors
///
/// Returns an error if reading or writing the wire protocol fails.
pub async fn handle_connection<R, W>(
    reader: &mut R,
    writer: &mut W,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
) -> Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    loop {
        let event = match read_event(reader).await {
            Ok(Some(event)) => event,
            Ok(None) => {
                tracing::debug!("Client disconnected");
                return Ok(());
            },
            Err(e) => {
                tracing::warn!(error = %e, "Failed to read event, disconnecting");
                return Err(e);
            },
        };

        match event {
            Event::Synthesize(data) => handle_synthesize(writer, runtime, voice_map, data).await?,
            Event::Ping(data) => handle_ping(writer, data).await?,
            Event::Describe => handle_describe(writer, runtime, voice_map).await?,
            Event::Unknown { event_type, .. } => {
                tracing::warn!(event_type = %event_type, "Unknown event type");
                send_error(
                    writer,
                    &format!("Unknown event type: {event_type}"),
                    Some("unknown-event"),
                )
                .await?;
            },
            other => {
                tracing::debug!(event_type = %other.event_type(), "Ignoring event");
            },
        }
    }
}

/// Handle a `synthesize` event: generate TTS audio and stream it back.
async fn handle_synthesize<W>(
    writer: &mut W,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
    data: SynthesizeData,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let (model_name, voice_name) = match resolve_voice(voice_map, &data) {
        VoiceResolution::Found {
            model_name,
            voice_name,
        } => (model_name, voice_name.map(String::from)),
        VoiceResolution::NotFound(voice_name) => {
            tracing::warn!(voice = %voice_name, "Voice not found");
            return send_error(
                writer,
                &format!("Unknown voice: {voice_name}"),
                Some("voice-not-found"),
            )
            .await;
        },
        VoiceResolution::NoModel => {
            return send_error(writer, "No TTS model loaded", None).await;
        },
    };

    let Some(handle) = runtime.tts_handle(model_name) else {
        return send_error(
            writer,
            &format!("TTS model '{model_name}' unavailable"),
            None,
        )
        .await;
    };
    let audio_info = handle.audio_info();

    let language = data
        .voice
        .and_then(|v| v.language)
        .unwrap_or_else(|| "auto".into());

    let (tx, rx) = oneshot::channel();
    let req = TtsGenerateRequest {
        text: data.text,
        language,
        voice: voice_name,
        opts: SpeechOptions::default(),
        reference_audio: None,
        reference_text: None,
        response_tx: tx,
    };

    if let Err(e) = runtime.generate_speech(model_name, req) {
        tracing::error!(error = %e, "Failed to dispatch TTS request");
        return send_error(writer, &format!("TTS engine unavailable: {e}"), None).await;
    }

    let tensor = match rx.await {
        Ok(Ok(tensor)) => tensor,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "TTS generation failed");
            return send_error(writer, &format!("TTS generation failed: {e}"), None).await;
        },
        Err(_) => {
            tracing::error!("TTS thread dropped the response channel");
            return send_error(writer, "TTS engine did not respond", None).await;
        },
    };

    let samples: Vec<f32> = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let pcm_bytes = pcm_f32_to_i16(&samples);
    debug_assert_eq!(
        audio_info.bits_per_sample, 16,
        "pcm_f32_to_i16 only produces 16-bit PCM"
    );
    let width = 2;

    write_event(
        writer,
        &Event::AudioStart(AudioStartData {
            rate: audio_info.sample_rate,
            width,
            channels: audio_info.channels,
            timestamp: None,
        }),
    )
    .await?;

    write_event(
        writer,
        &Event::AudioChunk {
            data: AudioChunkData {
                rate: audio_info.sample_rate,
                width,
                channels: audio_info.channels,
                timestamp: None,
            },
            audio: pcm_bytes,
        },
    )
    .await?;

    write_event(writer, &Event::AudioStop(AudioStopData { timestamp: None })).await?;

    Ok(())
}

/// Answer a `ping` event with a `pong` echoing the same text.
async fn handle_ping<W>(writer: &mut W, data: PingData) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event(writer, &Event::Pong(PongData { text: data.text })).await
}

/// Answer a `describe` event with service discovery info.
///
/// Lists every TTS model registered in `runtime` as a `tts` program
/// descriptor, including its voices. ASR and wake-word lists are always
/// empty (Crane does not yet serve those over Wyoming).
async fn handle_describe<W>(
    writer: &mut W,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event(writer, &Event::Info(build_info(runtime, voice_map))).await
}

/// Returns the Crane project attribution object used in Wyoming `info` responses.
fn crane_attribution() -> serde_json::Value {
    serde_json::json!({
        "name": "Crane",
        "url": "https://github.com/crane-ai/crane",
    })
}

/// Build the `info` event data from registered TTS models.
///
/// Each model becomes a `TtsProgram`-shaped JSON value (see the Wyoming
/// `wyoming/info.py` schema), named after its registration name so two
/// loaded models of the same architecture don't collide. Voices claimed by
/// an earlier-priority model (per `voice_map`'s "first model wins" rule)
/// are excluded from a later model's voice list so clients don't see the
/// same voice name twice under different programs. Models present in
/// `runtime` but not part of `voice_map`'s configured set are skipped
/// entirely, since they would otherwise show up with an empty voice list.
fn build_info(runtime: &ModelRuntime, voice_map: &VoiceMap) -> InfoData {
    let mut models: Vec<(&str, &TtsHandle)> = runtime.tts_handles().collect();
    models.sort_by_key(|(name, _)| *name);

    let tts = models
        .into_iter()
        .filter(|(model_name, _)| {
            let configured = voice_map.has_model(model_name);
            if !configured {
                tracing::warn!(
                    model = %model_name,
                    "Model in runtime but not in voice map config; excluding from describe response",
                );
            }
            configured
        })
        .map(|(model_name, handle)| {
            let voices: Vec<serde_json::Value> = handle
                .voices()
                .iter()
                .filter(|voice| voice_map.model_for_voice(&voice.name) == Some(model_name))
                .map(|voice| {
                    serde_json::json!({
                        "name": voice.name,
                        "attribution": crane_attribution(),
                        "installed": true,
                        "description": null,
                        "version": null,
                        "languages": voice.languages,
                        "speakers": null,
                    })
                })
                .collect();

            serde_json::json!({
                "name": model_name,
                "attribution": crane_attribution(),
                "installed": true,
                "description": null,
                "version": null,
                "voices": voices,
                "supports_synthesize_streaming": false,
            })
        })
        .collect();

    InfoData {
        tts,
        asr: vec![],
        wake: vec![],
    }
}

/// Send an `error` event with an optional machine-readable code.
async fn send_error<W>(writer: &mut W, text: &str, code: Option<&str>) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event(
        writer,
        &Event::Error(ErrorData {
            text: text.to_string(),
            code: code.map(String::from),
        }),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::SynthesizeVoice;
    use candle_core::{Device, Tensor};
    use crane::audio::tts::{AudioInfo, Tts, VoiceInfo};
    use crane_engine::model_factory::ModelType;
    use std::io::Cursor as SyncCursor;
    use tokio::io::BufReader;

    struct MockTts {
        audio_info: AudioInfo,
        voices: Vec<VoiceInfo>,
    }

    impl MockTts {
        fn new(sample_rate: u32, voices: Vec<VoiceInfo>) -> Self {
            Self {
                audio_info: AudioInfo {
                    sample_rate,
                    channels: 1,
                    bits_per_sample: 16,
                },
                voices,
            }
        }
    }

    impl Tts for MockTts {
        fn audio_info(&self) -> AudioInfo {
            self.audio_info
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            self.voices.clone()
        }

        fn generate_speech(
            &mut self,
            text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<candle_core::Tensor> {
            let n = text.chars().count().max(1);
            Tensor::new(vec![0.5f32; n], &Device::Cpu).map_err(Into::into)
        }
    }

    struct FailingTts;

    impl Tts for FailingTts {
        fn audio_info(&self) -> AudioInfo {
            AudioInfo {
                sample_rate: 24000,
                channels: 1,
                bits_per_sample: 16,
            }
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            vec![]
        }

        fn generate_speech(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<candle_core::Tensor> {
            anyhow::bail!("mock generation failure")
        }
    }

    fn test_runtime() -> ModelRuntime {
        let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        ModelRuntime::new(
            "test-model".into(),
            ModelType::Qwen3TTS,
            "F32".into(),
            "Cpu".into(),
            tokenizer,
            vec![2],
        )
    }

    fn voices(names: &[&str]) -> Vec<VoiceInfo> {
        names
            .iter()
            .map(|n| VoiceInfo {
                name: (*n).to_string(),
                languages: vec!["en".into()],
            })
            .collect()
    }

    async fn run_events(
        runtime: &ModelRuntime,
        voice_map: &VoiceMap,
        events: Vec<Event>,
    ) -> Vec<Event> {
        let mut input = Vec::new();
        for event in &events {
            write_event(&mut input, event).await.unwrap();
        }
        let mut reader = BufReader::new(SyncCursor::new(input));
        let mut output = Vec::new();
        handle_connection(&mut reader, &mut output, runtime, voice_map)
            .await
            .unwrap();

        let mut out_reader = BufReader::new(SyncCursor::new(output));
        let mut results = Vec::new();
        while let Some(event) = read_event(&mut out_reader).await.unwrap() {
            results.push(event);
        }
        results
    }

    #[tokio::test]
    async fn test_synthesize_default_voice() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Synthesize(SynthesizeData {
                text: "hello".into(),
                voice: None,
            })],
        )
        .await;

        assert_eq!(results.len(), 3);
        match &results[0] {
            Event::AudioStart(data) => {
                assert_eq!(data.rate, 24000);
                assert_eq!(data.width, 2);
                assert_eq!(data.channels, 1);
            },
            other => panic!("expected AudioStart, got {other:?}"),
        }
        match &results[1] {
            Event::AudioChunk { audio, .. } => {
                assert_eq!(*audio, pcm_f32_to_i16(&[0.5f32; 5]));
            },
            other => panic!("expected AudioChunk, got {other:?}"),
        }
        assert!(matches!(results[2], Event::AudioStop(_)));
    }

    #[tokio::test]
    async fn test_synthesize_named_voice() {
        let mut rt = test_runtime();
        rt.register_tts(
            "a".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        rt.register_tts(
            "b".into(),
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Synthesize(SynthesizeData {
                text: "hi".into(),
                voice: Some(SynthesizeVoice {
                    name: Some("bob".into()),
                    language: None,
                    speaker: None,
                }),
            })],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.rate, 16000),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_unknown_voice() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Synthesize(SynthesizeData {
                text: "hi".into(),
                voice: Some(SynthesizeVoice {
                    name: Some("ghost".into()),
                    language: None,
                    speaker: None,
                }),
            })],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => {
                assert_eq!(data.code.as_deref(), Some("voice-not-found"));
            },
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_no_model_loaded() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Synthesize(SynthesizeData {
                text: "hi".into(),
                voice: None,
            })],
        )
        .await;

        assert_eq!(results.len(), 1);
        assert!(matches!(&results[0], Event::Error(_)));
    }

    #[tokio::test]
    async fn test_synthesize_generation_failure() {
        let mut rt = test_runtime();
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(FailingTts))
            .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Synthesize(SynthesizeData {
                text: "hi".into(),
                voice: None,
            })],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert!(data.text.contains("mock generation failure")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_ping_pong() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Ping(PingData {
                text: Some("hi".into()),
            })],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Pong(PongData {
                text: Some("hi".into())
            })]
        );
    }

    #[tokio::test]
    async fn test_ping_pong_no_text() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);

        let results = run_events(&rt, &vm, vec![Event::Ping(PingData { text: None })]).await;

        assert_eq!(results, vec![Event::Pong(PongData { text: None })]);
    }

    #[tokio::test]
    async fn test_describe_no_models() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);

        let results = run_events(&rt, &vm, vec![Event::Describe]).await;

        assert_eq!(results, vec![Event::Info(InfoData::default())]);
    }

    #[tokio::test]
    async fn test_describe_single_model() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(&rt, &vm, vec![Event::Describe]).await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts.len(), 1);
                assert!(data.asr.is_empty());
                assert!(data.wake.is_empty());
                let program = &data.tts[0];
                assert_eq!(program["name"], "m1");
                assert_eq!(program["supports_synthesize_streaming"], false);
                let voices = program["voices"].as_array().unwrap();
                assert_eq!(voices.len(), 1);
                assert_eq!(voices[0]["name"], "alice");
                assert_eq!(voices[0]["languages"], serde_json::json!(["en"]));
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_multiple_models() {
        let mut rt = test_runtime();
        rt.register_tts(
            "a".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        rt.register_tts(
            "b".into(),
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        let results = run_events(&rt, &vm, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts.len(), 2);
                let names: Vec<&str> = data
                    .tts
                    .iter()
                    .map(|p| p["name"].as_str().unwrap())
                    .collect();
                assert_eq!(names, vec!["a", "b"]);
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_voice_conflict_excludes_duplicate() {
        let mut rt = test_runtime();
        rt.register_tts(
            "first".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        rt.register_tts(
            "second".into(),
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["first".to_string(), "second".to_string()], &rt);

        let results = run_events(&rt, &vm, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                let by_name = |name: &str| {
                    data.tts
                        .iter()
                        .find(|p| p["name"] == name)
                        .unwrap_or_else(|| panic!("missing program {name}"))
                };
                let first_voices = by_name("first")["voices"].as_array().unwrap();
                assert_eq!(first_voices.len(), 1);
                let second_voices = by_name("second")["voices"].as_array().unwrap();
                assert!(second_voices.is_empty());
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_excludes_model_not_in_voice_map() {
        let mut rt = test_runtime();
        rt.register_tts(
            "configured".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        rt.register_tts(
            "unconfigured".into(),
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["configured".to_string()], &rt);

        let results = run_events(&rt, &vm, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts.len(), 1);
                assert_eq!(data.tts[0]["name"], "configured");
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_streaming_is_false() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(&rt, &vm, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts[0]["supports_synthesize_streaming"], false);
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_unknown_event() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Unknown {
                event_type: "future-event".into(),
                data: serde_json::json!({}),
                payload: None,
            }],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert_eq!(data.code.as_deref(), Some("unknown-event")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_eof_returns_ok() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);
        let mut reader = BufReader::new(SyncCursor::new(Vec::<u8>::new()));
        let mut writer = Vec::new();
        let result = handle_connection(&mut reader, &mut writer, &rt, &vm).await;
        assert!(result.is_ok());
        assert!(writer.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_events_sequential() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![
                Event::Ping(PingData {
                    text: Some("a".into()),
                }),
                Event::Synthesize(SynthesizeData {
                    text: "hi".into(),
                    voice: None,
                }),
                Event::Ping(PingData {
                    text: Some("b".into()),
                }),
            ],
        )
        .await;

        assert_eq!(results.len(), 5);
        assert_eq!(
            results[0],
            Event::Pong(PongData {
                text: Some("a".into())
            })
        );
        assert!(matches!(results[1], Event::AudioStart(_)));
        assert!(matches!(results[2], Event::AudioChunk { .. }));
        assert!(matches!(results[3], Event::AudioStop(_)));
        assert_eq!(
            results[4],
            Event::Pong(PongData {
                text: Some("b".into())
            })
        );
    }

    #[test]
    fn test_voice_map_first_wins() {
        let mut rt = test_runtime();
        rt.register_tts(
            "first".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        rt.register_tts(
            "second".into(),
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["alice"]))),
        )
        .unwrap();

        let vm = VoiceMap::new(&["first".to_string(), "second".to_string()], &rt);
        assert_eq!(vm.model_for_voice("alice"), Some("first"));
    }

    #[test]
    fn test_voice_map_separate_voices() {
        let mut rt = test_runtime();
        rt.register_tts(
            "a".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        rt.register_tts(
            "b".into(),
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        )
        .unwrap();

        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);
        assert_eq!(vm.model_for_voice("alice"), Some("a"));
        assert_eq!(vm.model_for_voice("bob"), Some("b"));
        assert_eq!(vm.model_for_voice("ghost"), None);
    }

    #[test]
    fn test_voice_map_default_model() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);
        assert_eq!(vm.default_model(), Some("m1"));
    }

    #[test]
    fn test_voice_map_no_models() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);
        assert_eq!(vm.default_model(), None);
        assert_eq!(vm.model_for_voice("anything"), None);
    }

    #[tokio::test]
    async fn test_audio_start_width_is_bytes() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            vec![Event::Synthesize(SynthesizeData {
                text: "hi".into(),
                voice: None,
            })],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.width, 2),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }
}
