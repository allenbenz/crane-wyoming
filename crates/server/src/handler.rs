// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>

//! Wyoming event handler for TTS and ASR requests.
//!
//! Provides [`handle_connection`], an async function that runs the event
//! loop for a single Wyoming client connection: it reads events from an
//! [`AsyncBufRead`] source, dispatches `synthesize` requests to a
//! [`ModelRuntime`], writes `audio-start`/`audio-chunk`/`audio-stop`
//! responses to an [`AsyncWrite`] sink, and handles `transcribe` requests
//! by collecting audio from the client and replying with a `transcript`.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::Result;
use candle_core::{DType, Tensor};
use crane::audio::{AudioInfo, pcm_f32_to_i16, pcm_i16_to_f32};
use crane_core::generation::SpeechOptions;

use crate::engine::{AsrHandle, AsrTranscribeRequest, ModelRuntime, TtsGenerateRequest, TtsHandle};
use tokio::io::{AsyncBufRead, AsyncWrite};
use tokio::sync::oneshot;

use wyoming_protocol::event::{
    AudioChunkData, AudioFormat, AudioStartData, AudioStopData, ErrorData, Event, InfoData,
    PingData, PongData, SynthesizeData, SynthesizeStartData, SynthesizeVoice, TranscribeData,
    TranscriptChunkData, TranscriptData, TranscriptStartData,
};
use wyoming_protocol::wire::{read_event, write_event};

/// Extracts the base subtag of a language tag, lowercased, e.g. `"de-DE"`,
/// `"de_DE"`, and `"DE"` all become `"de"`.
fn base_language_subtag(language: &str) -> String {
    language
        .split(['-', '_'])
        .next()
        .unwrap_or(language)
        .to_ascii_lowercase()
}

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
    /// lowercased base language subtag (e.g. "de" for "de" or "de-DE") ->
    /// (model registration name, voice name) of the first voice offering
    /// that language.
    by_language: HashMap<String, (String, String)>,
    /// Registration names of all configured models that were found in the
    /// runtime (used to distinguish "no voices left after dedup" from
    /// "not a configured model" in service discovery).
    model_names: HashSet<String>,
    /// Name of the default model, used when a `synthesize` event specifies
    /// no voice and no language matching any known voice.
    default_model: Option<String>,
}

impl VoiceMap {
    /// Build a voice-to-model mapping from a [`ModelRuntime`].
    ///
    /// `model_names` gives model registration names in priority order
    /// (typically command-line `--model-tts` order); the first model to
    /// claim a voice name, or a language, wins. Names not found in
    /// `runtime` are skipped.
    #[must_use]
    pub fn new(model_names: &[String], runtime: &ModelRuntime) -> Self {
        let mut map = HashMap::new();
        let mut by_language = HashMap::new();
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
                for language in &voice.languages {
                    let base = base_language_subtag(language);
                    by_language
                        .entry(base)
                        .or_insert_with(|| (name.clone(), voice.name.clone()));
                }
            }
        }
        let default_model = runtime.default_tts_name().map(String::from);
        Self {
            map,
            by_language,
            model_names: found_names,
            default_model,
        }
    }

    /// Returns the model registration name for `voice_name`, if known.
    #[must_use]
    pub fn model_for_voice(&self, voice_name: &str) -> Option<&str> {
        self.map.get(voice_name).map(String::as_str)
    }

    /// Returns the `(model registration name, voice name)` of the first
    /// registered voice offering `language`, if any.
    ///
    /// Matching compares only the base subtag (the part before a `-` or
    /// `_`), case-insensitively, so a request for `"de"` matches a voice
    /// tagged `"de"`, `"de-DE"`, or `"de_DE"`.
    #[must_use]
    pub fn model_for_language(&self, language: &str) -> Option<(&str, &str)> {
        self.by_language
            .get(&base_language_subtag(language))
            .map(|(model, voice)| (model.as_str(), voice.as_str()))
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

/// Maps ASR model names and languages to model registration names.
///
/// Built once at startup from a [`ModelRuntime`]'s registered ASR models.
/// Unlike [`VoiceMap`], there is no per-voice granularity -- ASR models
/// advertise supported languages directly via [`AsrHandle::languages`]. A
/// `transcribe` request selects a model by explicit name, by language hint
/// (matched against the per-model language lists), or falls back to the
/// default model.
pub struct AsrModelMap {
    /// Registration names of all configured ASR models that were found in
    /// the runtime.
    model_names: HashSet<String>,
    /// Lowercased base language subtag (e.g. `"de"` for `"de"` or
    /// `"de-DE"`) -> model registration name of the first model claiming
    /// that language.
    by_language: HashMap<String, String>,
    /// Name of the default ASR model, used when a `transcribe` event
    /// specifies no model name and no language matching any known model.
    default_model: Option<String>,
}

impl AsrModelMap {
    /// Build an ASR model map from a [`ModelRuntime`].
    ///
    /// `model_names` lists model registration names to include; names not
    /// found in `runtime` are skipped.
    #[must_use]
    pub fn new(model_names: &[String], runtime: &ModelRuntime) -> Self {
        let mut found_names = HashSet::new();
        let mut by_language = HashMap::new();
        for name in model_names {
            let Some(handle) = runtime.asr_handle(name) else {
                continue;
            };
            found_names.insert(name.clone());
            for language in handle.languages() {
                by_language
                    .entry(base_language_subtag(language))
                    .or_insert_with(|| name.clone());
            }
        }
        let default_model = runtime.default_asr_name().map(String::from);
        Self {
            model_names: found_names,
            by_language,
            default_model,
        }
    }

    /// Returns `true` if `name` is a configured model registration name.
    #[must_use]
    pub fn has_model(&self, name: &str) -> bool {
        self.model_names.contains(name)
    }

    /// Returns the registration name matching `name`, if it's configured.
    ///
    /// Unlike [`has_model`](Self::has_model), returns the map's own copy of
    /// the name so callers can propagate it with a lifetime tied to this
    /// `AsrModelMap` rather than to their (possibly shorter-lived) input.
    #[must_use]
    pub fn model_name(&self, name: &str) -> Option<&str> {
        self.model_names.get(name).map(String::as_str)
    }

    /// Returns the registration name of the first model claiming to
    /// support `language`, matched on the base language subtag (see
    /// [`base_language_subtag`]).
    #[must_use]
    pub fn model_for_language(&self, language: &str) -> Option<&str> {
        self.by_language
            .get(&base_language_subtag(language))
            .map(String::as_str)
    }

    /// Returns the default model's registration name, if any ASR model is loaded.
    #[must_use]
    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }
}

/// Outcome of resolving a `synthesize` event's voice to a model.
enum VoiceResolution<'m> {
    /// A model was resolved. `voice_name` is `None` when the client did not
    /// request a specific voice or language matching any known voice (the
    /// model's default voice is used).
    ///
    /// `voice_name` is owned rather than borrowed because a language match
    /// yields a name borrowed from `VoiceMap` (lifetime `'m`), which can't
    /// unify with a borrow of `data`'s lifetime without a `Cow` or an extra
    /// lifetime parameter on this enum.
    Found {
        model_name: &'m str,
        voice_name: Option<String>,
    },
    /// The client requested a voice name with no matching model.
    NotFound(String),
    /// No voice was requested and no TTS model is loaded.
    NoModel,
}

/// Resolve which model (and voice) a `synthesize` event should use.
///
/// An explicit voice name always wins. Otherwise, if the request names a
/// language, it is matched against known voices' languages -- see
/// [`VoiceMap::model_for_language`] -- so e.g. `language: "de"` with no
/// voice picks a German voice instead of silently falling back to
/// whichever voice happens to be the configured default. Only when neither
/// resolves does the default model apply.
fn resolve_voice<'m>(voice_map: &'m VoiceMap, data: &SynthesizeData) -> VoiceResolution<'m> {
    if let Some(voice_name) = data.voice.as_ref().and_then(|v| v.name.as_deref()) {
        return match voice_map.model_for_voice(voice_name) {
            Some(model_name) => VoiceResolution::Found {
                model_name,
                voice_name: Some(voice_name.to_string()),
            },
            None => VoiceResolution::NotFound(voice_name.to_string()),
        };
    }
    if let Some(language) = data.voice.as_ref().and_then(|v| v.language.as_deref())
        && let Some((model_name, voice_name)) = voice_map.model_for_language(language)
    {
        return VoiceResolution::Found {
            model_name,
            voice_name: Some(voice_name.to_string()),
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

/// Outcome of resolving a `transcribe` event's target model.
enum AsrResolution<'m> {
    /// A model was resolved.
    Found(&'m str),
    /// The client requested a model name with no matching model.
    NotFound(String),
    /// No model was requested and no ASR model is loaded.
    NoModel,
}

/// Resolve which ASR model a `transcribe` event should use.
///
/// An explicit model name always wins. Otherwise, if the request names a
/// language, it is matched against known models' languages -- see
/// [`AsrModelMap::model_for_language`] -- so e.g. `language: "de"` with no
/// model name picks a model claiming German instead of silently falling
/// back to whichever model happens to be the configured default. Only when
/// neither resolves does the default model apply.
fn resolve_asr_model<'m>(asr_map: &'m AsrModelMap, data: &TranscribeData) -> AsrResolution<'m> {
    if let Some(name) = data.name.as_deref() {
        return match asr_map.model_name(name) {
            Some(model_name) => AsrResolution::Found(model_name),
            None => AsrResolution::NotFound(name.to_string()),
        };
    }
    if let Some(language) = data.language.as_deref()
        && let Some(model_name) = asr_map.model_for_language(language)
    {
        return AsrResolution::Found(model_name);
    }
    match asr_map.default_model() {
        Some(model_name) => AsrResolution::Found(model_name),
        None => AsrResolution::NoModel,
    }
}

/// How long to wait for the next event before disconnecting an idle client.
const IDLE_TIMEOUT: Duration = Duration::from_mins(1);

/// How long to wait for a single write to the client during streaming
/// synthesis or streaming transcription before giving up.
///
/// The streaming path holds a model's dedicated worker thread hostage for
/// as long as writes to the client take (see the backpressure docs on
/// [`ModelRuntime::generate_speech_stream`] and
/// [`ModelRuntime::transcribe_stream`]): the worker blocks in
/// `blocking_send` once the bounded chunk channel fills, which happens as
/// soon as this handler stops draining it. An unresponsive client (slow
/// reader, congested link, or one that simply stopped reading) would
/// otherwise stall every other request queued for the same model
/// indefinitely. Timing out tears down this connection and drops the
/// chunk receiver, unblocking the worker.
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

/// Sample width, in bytes, that ASR models expect (16-bit PCM).
const EXPECTED_SAMPLE_WIDTH: u16 = 2;

/// Channel count that ASR models expect (mono).
const EXPECTED_CHANNELS: u16 = 1;

/// Maximum total size, in bytes, of the PCM audio collected for a single
/// `transcribe` request.
///
/// Bounds memory use against a client that keeps sending `audio-chunk`
/// events indefinitely; 1 MiB is about 30 seconds of 16-bit mono audio at
/// 16 kHz, comfortably more than a single voice-assistant utterance.
const MAX_TRANSCRIBE_AUDIO_BYTES: usize = 1024 * 1024;

/// Run the Wyoming event loop for a single client connection.
///
/// Reads events from `reader` and dispatches them: `synthesize` requests
/// generate speech through `runtime`, `transcribe` requests collect audio
/// and reply with a transcript, `ping` is answered with `pong`, `describe`
/// gets an `info` response with available TTS/ASR model metadata, and
/// unrecognized event types get an `error` response. The loop continues
/// until the client disconnects cleanly (EOF), the wire protocol desyncs,
/// or the client goes idle for longer than [`IDLE_TIMEOUT`] between events.
///
/// Application-level failures (unknown voice, generation error) are
/// reported as Wyoming `error` events and do not terminate the
/// connection; only I/O errors and malformed wire data do.
///
/// # Cancellation
///
/// If this future is dropped (e.g. the TCP connection resets) while a
/// TTS request is in flight, the request's response channel is dropped --
/// the oneshot sender for [`handle_synthesize_blob`], or the chunk sender
/// for [`handle_synthesize_streaming`]. The TTS thread checks for this
/// before generation (and, for streaming, after each chunk) and stops
/// instead of blocking or erroring. The same applies to a `transcribe`
/// request in flight: the ASR thread checks its oneshot sender before
/// transcribing in [`handle_transcribe_batch`], or its chunk sender after
/// each chunk in [`handle_transcribe_streaming`].
///
/// # Errors
///
/// Returns an error if reading or writing the wire protocol fails.
pub async fn handle_connection<R, W>(
    reader: &mut R,
    writer: &mut W,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
    asr_map: &AsrModelMap,
) -> Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    loop {
        let event = match tokio::time::timeout(IDLE_TIMEOUT, read_event(reader)).await {
            Ok(Ok(Some(event))) => event,
            Ok(Ok(None)) => {
                tracing::debug!("Client disconnected");
                return Ok(());
            },
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "Failed to read event, disconnecting");
                // Protocol errors are intentionally erased to anyhow here:
                // the server logs them above and does not need to match on
                // a specific ProtocolError variant. The typed error exists
                // for downstream library consumers of wyoming-protocol.
                return Err(e.into());
            },
            Err(_) => {
                tracing::info!("Client idle for {IDLE_TIMEOUT:?}, disconnecting");
                return Ok(());
            },
        };

        match event {
            Event::Synthesize(data) => handle_synthesize(writer, runtime, voice_map, data).await?,
            Event::SynthesizeStart(data) => {
                handle_synthesize_streamed_text(reader, writer, runtime, voice_map, data).await?;
            },
            Event::Transcribe(data) => {
                handle_transcribe(reader, writer, runtime, asr_map, data).await?;
            },
            Event::Ping(data) => handle_ping(writer, data).await?,
            Event::Describe => handle_describe(writer, runtime, voice_map, asr_map).await?,
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

/// Convert a raw f32 PCM tensor into an `audio-chunk` event.
fn tensor_to_audio_chunk(tensor: &Tensor, audio_info: AudioInfo) -> Result<Event> {
    let samples: Vec<f32> = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    Ok(Event::AudioChunk {
        data: AudioChunkData::new(audio_format(audio_info)),
        audio: pcm_f32_to_i16(&samples),
    })
}

/// Build the `audio-start` event announcing the format of the audio that
/// will follow.
fn audio_start_event(audio_info: AudioInfo) -> Event {
    Event::AudioStart(AudioStartData::new(audio_format(audio_info)))
}

/// Convert Crane's [`AudioInfo`] into the protocol's [`AudioFormat`].
fn audio_format(audio_info: AudioInfo) -> AudioFormat {
    AudioFormat {
        rate: audio_info.sample_rate,
        width: audio_info.bits_per_sample / 8,
        channels: audio_info.channels,
    }
}

/// Write an event to `writer`, giving up after [`WRITE_TIMEOUT`].
///
/// Used on the streaming synthesis and streaming transcription paths, where
/// a stalled write blocks the model's shared worker thread -- see
/// [`WRITE_TIMEOUT`].
async fn write_event_timeout<W>(writer: &mut W, event: &Event) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    match tokio::time::timeout(WRITE_TIMEOUT, write_event(writer, event)).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e.into()),
        Err(_elapsed) => Err(anyhow::anyhow!("Write timed out after {WRITE_TIMEOUT:?}")),
    }
}

/// Build an `error` event's data from a message and optional code.
fn error_data(text: &str, code: Option<&str>) -> ErrorData {
    match code {
        Some(code) => ErrorData::new(text).with_code(code),
        None => ErrorData::new(text),
    }
}

/// Send an `error` event, giving up after [`WRITE_TIMEOUT`].
async fn send_error_timeout<W>(writer: &mut W, text: &str, code: Option<&str>) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event_timeout(writer, &Event::Error(error_data(text, code))).await
}

/// Close an in-progress audio stream with `audio-stop`, then report `text`
/// as an `error` event -- crane-wyoming's own mid-stream error convention.
/// The Wyoming protocol spec does not define `error`'s relationship to
/// `audio-stop` at all, so this ordering is not something a generic
/// Wyoming client can assume; [`wyoming_protocol::client::Client`] is the
/// one client expected to rely on it.
async fn stop_stream_with_error<W>(writer: &mut W, text: &str) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event_timeout(writer, &Event::AudioStop(AudioStopData::new())).await?;
    send_error_timeout(writer, text, None).await
}

/// Close an in-progress transcript stream with `transcript-stop`, then
/// report `text` as an `error` event -- mirrors [`stop_stream_with_error`]'s
/// mid-stream error convention for the ASR streaming path.
async fn stop_transcript_stream_with_error<W>(writer: &mut W, text: &str) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event_timeout(writer, &Event::TranscriptStop).await?;
    send_error_timeout(writer, text, None).await
}

/// Parameters for a resolved `synthesize` request, shared by the streaming
/// and blob dispatch paths.
struct SynthesizeParams {
    /// Text to synthesize.
    text: String,
    /// Target language (e.g. "english", "auto").
    language: String,
    /// Voice name from the resolved model's voices, or `None` for the
    /// model default.
    voice_name: Option<String>,
    /// Audio format the resolved model produces.
    audio_info: AudioInfo,
}

/// Handle a `synthesize` event: generate TTS audio and send it back.
///
/// Resolves the requested voice to a model, then dispatches to
/// [`handle_synthesize_streaming`] or [`handle_synthesize_blob`] depending
/// on [`ModelRuntime::streaming_enabled`] -- see that method's doc for why
/// streaming is conditional.
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
        } => {
            tracing::debug!(
                requested_voice = ?data.voice.as_ref().and_then(|v| v.name.as_deref()),
                requested_language = ?data.voice.as_ref().and_then(|v| v.language.as_deref()),
                resolved_model = %model_name,
                resolved_voice = ?voice_name,
                "Resolved synthesize voice",
            );
            (model_name, voice_name)
        },
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

    // Every Tts impl in the crane crate accepts ISO 639-1 codes (or "auto") and
    // converts to whatever format the underlying model needs internally.
    let language = data
        .voice
        .and_then(|v| v.language)
        .map_or_else(|| "auto".into(), |code| base_language_subtag(&code));

    let params = SynthesizeParams {
        text: data.text,
        language,
        voice_name,
        audio_info,
    };

    if runtime.streaming_enabled() {
        handle_synthesize_streaming(writer, runtime, model_name, params).await
    } else {
        handle_synthesize_blob(writer, runtime, model_name, params).await
    }
}

/// Handle a `synthesize-start` event: read the `synthesize-chunk`*/
/// `synthesize-stop` sequence that follows, then dispatch the concatenated
/// text through the same path as a one-shot `synthesize` event, and
/// terminate the exchange with `synthesize-stopped`.
///
/// The chunks are read here rather than by the caller's main event loop,
/// since they belong to this one synthesize exchange -- the same convention
/// [`handle_transcribe`] uses for its `audio-*` sequence.
///
/// The start event's `language` and `voice` play the same role as the voice
/// specification of a one-shot `synthesize` (see [`resolve_voice`]): a voice
/// name wins, then a language, then the default model. When both the start
/// event's `language` and its `voice.language` are set, the voice's own
/// language wins, since it is the more specific choice.
///
/// `synthesize-stopped` (the Wyoming protocol's "end of streaming
/// response" event, `wyoming.tts.SynthesizeStopped`) is sent after the
/// audio stream -- including after an `error` response, where it merely
/// confirms the exchange is over. Home Assistant's streaming TTS reader
/// ends on `synthesize-stopped`, not `audio-stop`; without it the client
/// hangs until the idle timeout closes the connection and never plays the
/// audio. Reference Wyoming TTS servers (e.g. wyoming-piper) send it too.
async fn handle_synthesize_streamed_text<R, W>(
    reader: &mut R,
    writer: &mut W,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
    start: SynthesizeStartData,
) -> Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut text = String::new();
    loop {
        let Some(event) = read_event_or_idle(reader).await? else {
            return Ok(());
        };
        match event {
            Event::SynthesizeChunk(chunk) => text.push_str(&chunk.text),
            Event::SynthesizeStop => break,
            Event::Synthesize(_) => (),
            other => {
                tracing::warn!(
                    event_type = %other.event_type(),
                    "Expected synthesize-chunk or synthesize-stop during streamed synthesize",
                );
                send_error(
                    writer,
                    "Expected synthesize-chunk or synthesize-stop during streamed synthesize",
                    None,
                )
                .await?;
                return Ok(());
            },
        }
    }

    // Fold the start event's fields into the one-shot synthesize shape:
    // the resolver looks for name and language on the voice, so a bare
    // top-level `language` is carried by a synthesized voice spec.
    let voice = match start.voice {
        Some(mut voice) => {
            if voice.language.is_none() {
                voice.language = start.language;
            }
            Some(voice)
        },
        None => start.language.map(|language| {
            let mut voice = SynthesizeVoice::new();
            voice.language = Some(language);
            voice
        }),
    };

    let data = match voice {
        Some(voice) => SynthesizeData::new(text).with_voice(voice),
        None => SynthesizeData::new(text),
    };
    handle_synthesize(writer, runtime, voice_map, data).await?;
    write_event_timeout(writer, &Event::SynthesizeStopped).await
}

/// Generate TTS audio incrementally and send it back as it's produced.
///
/// Audio is sent as it's generated: `audio-start` is delayed until the
/// first chunk arrives, so a generation failure before any audio is
/// produced surfaces as a plain `error` event with no `audio-start`/
/// `audio-stop` wrapper. A stream that completes with no chunks at all is
/// not an error -- it is reported as an empty `audio-start`/`audio-stop`
/// pair. A failure after the first chunk (including a chunk that fails to
/// encode) closes the in-progress stream with `audio-stop` before sending
/// the `error` event, per crane-wyoming's own mid-stream error convention
/// (see [`stop_stream_with_error`]) -- not something the Wyoming protocol
/// spec itself defines. Every write to the client uses [`WRITE_TIMEOUT`],
/// since a stalled write here blocks the model's shared worker thread.
async fn handle_synthesize_streaming<W>(
    writer: &mut W,
    runtime: &ModelRuntime,
    model_name: &str,
    params: SynthesizeParams,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut rx = match runtime.generate_speech_stream(
        model_name,
        params.text,
        params.language,
        params.voice_name,
        SpeechOptions::default(),
    ) {
        Ok(rx) => rx,
        Err(e) => {
            tracing::error!(error = %e, "Failed to dispatch TTS request");
            return send_error_timeout(writer, &format!("TTS engine unavailable: {e}"), None).await;
        },
    };

    let first_chunk = match rx.recv().await {
        Some(Ok(tensor)) => tensor,
        Some(Err(e)) => {
            tracing::error!(error = %e, "TTS generation failed");
            return send_error_timeout(writer, &format!("TTS generation failed: {e}"), None).await;
        },
        None => {
            tracing::debug!("TTS stream produced no audio chunks");
            write_event_timeout(writer, &audio_start_event(params.audio_info)).await?;
            return write_event_timeout(writer, &Event::AudioStop(AudioStopData::new())).await;
        },
    };

    write_event_timeout(writer, &audio_start_event(params.audio_info)).await?;
    match tensor_to_audio_chunk(&first_chunk, params.audio_info) {
        Ok(event) => write_event_timeout(writer, &event).await?,
        Err(e) => {
            tracing::error!(error = %e, "Failed to encode audio chunk");
            return stop_stream_with_error(writer, &format!("Audio encoding failed: {e}")).await;
        },
    }

    loop {
        match rx.recv().await {
            Some(Ok(tensor)) => match tensor_to_audio_chunk(&tensor, params.audio_info) {
                Ok(event) => write_event_timeout(writer, &event).await?,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to encode audio chunk");
                    return stop_stream_with_error(writer, &format!("Audio encoding failed: {e}"))
                        .await;
                },
            },
            Some(Err(e)) => {
                tracing::error!(error = %e, "TTS generation failed mid-stream");
                return stop_stream_with_error(writer, &format!("TTS generation failed: {e}"))
                    .await;
            },
            None => {
                return write_event_timeout(writer, &Event::AudioStop(AudioStopData::new())).await;
            },
        }
    }
}

/// Generate the complete TTS waveform, then send it back as a single chunk.
///
/// Used when [`ModelRuntime::streaming_enabled`] is `false` (hardware too
/// slow to keep up with real-time incremental generation). Dispatches
/// through [`ModelRuntime::generate_speech`], which is cache-eligible --
/// unlike the streaming path, so repeated phrases on slow hardware benefit
/// from [`crane::engine::TtsCache`]. A failure to encode the generated
/// tensor after `audio-start` has been sent closes the stream with
/// `audio-stop` before reporting the `error` event, matching the
/// mid-stream error convention documented in WYOMING.md.
async fn handle_synthesize_blob<W>(
    writer: &mut W,
    runtime: &ModelRuntime,
    model_name: &str,
    params: SynthesizeParams,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let (tx, rx) = oneshot::channel();
    let req = TtsGenerateRequest {
        text: params.text,
        language: params.language,
        voice: params.voice_name,
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

    write_event(writer, &audio_start_event(params.audio_info)).await?;
    match tensor_to_audio_chunk(&tensor, params.audio_info) {
        Ok(event) => write_event(writer, &event).await?,
        Err(e) => {
            tracing::error!(error = %e, "Failed to encode audio chunk");
            write_event(writer, &Event::AudioStop(AudioStopData::new())).await?;
            return send_error(writer, &format!("Audio encoding failed: {e}"), None).await;
        },
    }
    write_event(writer, &Event::AudioStop(AudioStopData::new())).await?;

    Ok(())
}

/// Wait for the next event during a multi-event exchange (audio collection
/// for `transcribe`, text collection for streamed `synthesize`), giving up
/// after [`IDLE_TIMEOUT`].
///
/// Returns `Ok(None)` if the client disconnected cleanly or went idle --
/// both end the connection the same way the main loop in
/// [`handle_connection`] does, so callers should return `Ok(())` in that
/// case rather than treating it as an error.
async fn read_event_or_idle<R>(reader: &mut R) -> Result<Option<Event>>
where
    R: AsyncBufRead + Unpin,
{
    match tokio::time::timeout(IDLE_TIMEOUT, read_event(reader)).await {
        Ok(Ok(Some(event))) => Ok(Some(event)),
        Ok(Ok(None)) => {
            tracing::debug!("Client disconnected during event collection");
            Ok(None)
        },
        Ok(Err(e)) => Err(e.into()),
        Err(_) => {
            tracing::info!(
                "Client idle for {IDLE_TIMEOUT:?} during event collection, disconnecting",
            );
            Ok(None)
        },
    }
}

/// Builds the error message for an `audio-start`/`audio-chunk` whose format
/// doesn't match what the resolved ASR model expects.
fn unsupported_audio_format_message(
    rate: u32,
    width: u16,
    channels: u16,
    expected_rate: u32,
) -> String {
    format!(
        "Unsupported audio format: rate={rate}, width={width}, channels={channels} \
         (expected rate={expected_rate}, width={EXPECTED_SAMPLE_WIDTH}, channels={EXPECTED_CHANNELS})",
    )
}

/// Reads the `audio-start`/`audio-chunk`*/`audio-stop` sequence for one
/// `transcribe` exchange and returns the collected PCM bytes.
///
/// `audio-start`, and every subsequent `audio-chunk`, must declare the
/// exact format the model expects (16-bit mono PCM at `expected_rate`);
/// resampling is not implemented, so a mismatched format is rejected with
/// an `error` event rather than silently misinterpreted. The total
/// collected size is capped at [`MAX_TRANSCRIBE_AUDIO_BYTES`].
///
/// Returns `Ok(None)` if the client disconnected, went idle, sent an
/// unexpected event, or exceeded a limit above -- in all of these cases an
/// `error` event has already been sent (if applicable) and the caller
/// should end the `transcribe` exchange by returning `Ok(())`.
async fn collect_transcribe_audio<R, W>(
    reader: &mut R,
    writer: &mut W,
    expected_rate: u32,
) -> Result<Option<Vec<u8>>>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let Some(event) = read_event_or_idle(reader).await? else {
        return Ok(None);
    };
    let Event::AudioStart(start) = event else {
        tracing::warn!(
            event_type = %event.event_type(),
            "Expected audio-start after transcribe",
        );
        send_error(writer, "Expected audio-start after transcribe", None).await?;
        return Ok(None);
    };
    if start.rate != expected_rate
        || start.width != EXPECTED_SAMPLE_WIDTH
        || start.channels != EXPECTED_CHANNELS
    {
        send_error(
            writer,
            &unsupported_audio_format_message(
                start.rate,
                start.width,
                start.channels,
                expected_rate,
            ),
            Some("unsupported-audio-format"),
        )
        .await?;
        return Ok(None);
    }

    let mut pcm_bytes = Vec::new();
    loop {
        let Some(event) = read_event_or_idle(reader).await? else {
            return Ok(None);
        };
        match event {
            Event::AudioChunk { data, audio } => {
                if data.rate != expected_rate
                    || data.width != EXPECTED_SAMPLE_WIDTH
                    || data.channels != EXPECTED_CHANNELS
                {
                    send_error(
                        writer,
                        &unsupported_audio_format_message(
                            data.rate,
                            data.width,
                            data.channels,
                            expected_rate,
                        ),
                        Some("unsupported-audio-format"),
                    )
                    .await?;
                    return Ok(None);
                }
                if pcm_bytes.len() + audio.len() > MAX_TRANSCRIBE_AUDIO_BYTES {
                    send_error(
                        writer,
                        &format!(
                            "Audio exceeds maximum size of \
                             {MAX_TRANSCRIBE_AUDIO_BYTES} bytes for a single transcribe request",
                        ),
                        Some("audio-too-large"),
                    )
                    .await?;
                    return Ok(None);
                }
                pcm_bytes.extend_from_slice(&audio);
            },
            Event::AudioStop(_) => break,
            other => {
                tracing::warn!(
                    event_type = %other.event_type(),
                    "Expected audio-chunk or audio-stop during transcribe",
                );
                send_error(
                    writer,
                    "Expected audio-chunk or audio-stop during transcribe",
                    None,
                )
                .await?;
                return Ok(None);
            },
        }
    }

    Ok(Some(pcm_bytes))
}

/// Handle a `transcribe` event: collect the client's audio and reply with a
/// transcript.
///
/// Resolves the requested ASR model, then reads the following
/// `audio-start`/`audio-chunk`*/`audio-stop` sequence via
/// [`collect_transcribe_audio`] -- these are read directly here rather than
/// by the caller's main event loop, since they belong to this one
/// `transcribe` exchange. If a VAD model is loaded and the ASR model's
/// sample rate is VAD-compatible (see [`vad_decision`]), the
/// collected audio is pre-filtered through
/// [`ModelRuntime::vad_filter_audio`] using `data.vad_sensitivity` before
/// dispatch; otherwise the audio is passed through unfiltered. Dispatches
/// to [`handle_transcribe_streaming`] or [`handle_transcribe_batch`]
/// depending on [`ModelRuntime::streaming_enabled`], the same flag
/// [`handle_synthesize`] uses for TTS -- see that method's doc for why
/// streaming is conditional (in short: CPU hardware can't keep up).
async fn handle_transcribe<R, W>(
    reader: &mut R,
    writer: &mut W,
    runtime: &ModelRuntime,
    asr_map: &AsrModelMap,
    data: TranscribeData,
) -> Result<()>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let model_name = match resolve_asr_model(asr_map, &data) {
        AsrResolution::Found(model_name) => model_name,
        AsrResolution::NotFound(name) => {
            tracing::warn!(model = %name, "ASR model not found");
            return send_error(
                writer,
                &format!("Unknown ASR model: {name}"),
                Some("asr-model-not-found"),
            )
            .await;
        },
        AsrResolution::NoModel => {
            return send_error(writer, "No ASR model loaded", None).await;
        },
    };

    let Some(handle) = runtime.asr_handle(model_name) else {
        return send_error(
            writer,
            &format!("ASR model '{model_name}' unavailable"),
            None,
        )
        .await;
    };
    let expected_rate = handle.input_sample_rate();

    let Some(pcm_bytes) = collect_transcribe_audio(reader, writer, expected_rate).await? else {
        return Ok(());
    };
    let audio = pcm_i16_to_f32(&pcm_bytes);

    let audio = match vad_decision(runtime, expected_rate) {
        VadDecision::Skip => audio,
        VadDecision::IncompatibleRate => {
            tracing::warn!(
                sample_rate = expected_rate,
                "ASR sample rate unsupported by VAD (requires 8000 or 16000 Hz); skipping pre-filtering",
            );
            audio
        },
        VadDecision::Apply => match runtime.vad_filter_audio(audio, data.vad_sensitivity).await {
            Ok(filtered) => filtered,
            Err(e) => {
                tracing::error!(error = %e, "VAD filtering failed");
                return send_error(writer, &format!("VAD filtering failed: {e}"), None).await;
            },
        },
    };

    if runtime.streaming_enabled() {
        handle_transcribe_streaming(writer, runtime, model_name, audio, data.language).await
    } else {
        handle_transcribe_batch(writer, runtime, model_name, audio, data.language).await
    }
}

/// Whether [`handle_transcribe`] should pre-filter collected audio through
/// VAD before ASR dispatch, as returned by [`vad_decision`].
enum VadDecision {
    /// No VAD model is loaded; pass audio through unfiltered.
    Skip,
    /// A VAD model is loaded, but the ASR model's sample rate isn't one
    /// Silero VAD supports; pass audio through unfiltered (with a warning).
    IncompatibleRate,
    /// A VAD model is loaded and `sample_rate` is VAD-compatible; filter
    /// the audio through it.
    Apply,
}

/// Decides whether [`handle_transcribe`] should filter audio through VAD,
/// based on whether a VAD model is loaded and whether `sample_rate` is one
/// Silero VAD supports (8 kHz or 16 kHz). Every ASR model in Crane today
/// reports one of these two rates, but
/// [`crane::audio::Asr::input_sample_rate`] does not constrain callers to
/// them, so a future model reporting e.g. 24000 or 44100 must skip VAD
/// pre-filtering rather than feed an unsupported rate into
/// [`ModelRuntime::vad_filter_audio`].
fn vad_decision(runtime: &ModelRuntime, sample_rate: u32) -> VadDecision {
    if !runtime.has_vad() {
        VadDecision::Skip
    } else if matches!(sample_rate, 8000 | 16000) {
        VadDecision::Apply
    } else {
        VadDecision::IncompatibleRate
    }
}

/// Transcribe the complete audio and send back a single `transcript`.
///
/// Used when [`ModelRuntime::streaming_enabled`] is `false`. Dispatches
/// through [`ModelRuntime::transcribe`] and waits for the one-shot result.
async fn handle_transcribe_batch<W>(
    writer: &mut W,
    runtime: &ModelRuntime,
    model_name: &str,
    audio: Vec<f32>,
    language: Option<String>,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let (tx, rx) = oneshot::channel();
    let req = AsrTranscribeRequest {
        audio,
        language,
        response_tx: tx,
    };

    if let Err(e) = runtime.transcribe(model_name, req) {
        tracing::error!(error = %e, "Failed to dispatch ASR request");
        return send_error(writer, &format!("ASR engine unavailable: {e}"), None).await;
    }

    let transcript = match rx.await {
        Ok(Ok(transcript)) => transcript,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "ASR transcription failed");
            return send_error(writer, &format!("ASR transcription failed: {e}"), None).await;
        },
        Err(_) => {
            tracing::error!("ASR thread dropped the response channel");
            return send_error(writer, "ASR engine did not respond", None).await;
        },
    };

    let mut result = TranscriptData::new(transcript.text);
    if let Some(language) = transcript.language {
        result = result.with_language(language);
    }
    write_event(writer, &Event::Transcript(result)).await?;

    Ok(())
}

/// Transcribe the audio and deliver the transcript incrementally.
///
/// The audio has already been fully collected by the time this is called --
/// only the *transcript* is delivered incrementally, via
/// `transcript-start`/`transcript-chunk`*/`transcript`/`transcript-stop`.
/// See [`ModelRuntime::transcribe_stream`]'s docs for why: Crane's `Asr`
/// trait takes a complete audio slice, so "streaming" here only applies to
/// the output.
///
/// `transcript-start` is delayed until the first chunk arrives, so a
/// failure before any output surfaces as a plain `error` event with no
/// `transcript-start`/`transcript-stop` wrapper. A stream that completes
/// with no chunks at all is not an error -- it is reported as an empty
/// `transcript-start`/`transcript`/`transcript-stop` sequence. A failure
/// after the first chunk closes the in-progress stream with
/// `transcript-stop` before sending the `error` event, via
/// [`stop_transcript_stream_with_error`] -- crane-wyoming's own mid-stream
/// error convention, mirroring [`stop_stream_with_error`] for TTS. Every
/// write to the client uses [`WRITE_TIMEOUT`], for the same reason
/// documented there: a stalled write here blocks the model's shared worker
/// thread.
///
/// Unlike the TTS streaming loop, which stops on channel closure, this loop
/// stops as soon as it sees a chunk with `is_final: true` and returns
/// without waiting on `rx` again. This relies on `Asr::transcribe_stream`'s
/// contract that the final chunk is the last one produced; a non-conformant
/// implementation that kept sending after `is_final: true` would find its
/// next `blocking_send` fail once this function has returned and dropped
/// `rx`, which the worker thread logs as a caller disconnect rather than
/// the model bug it would actually be.
async fn handle_transcribe_streaming<W>(
    writer: &mut W,
    runtime: &ModelRuntime,
    model_name: &str,
    audio: Vec<f32>,
    language: Option<String>,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut rx = match runtime.transcribe_stream(model_name, audio, language) {
        Ok(rx) => rx,
        Err(e) => {
            tracing::error!(error = %e, "Failed to dispatch ASR stream request");
            return send_error_timeout(writer, &format!("ASR engine unavailable: {e}"), None).await;
        },
    };

    let first = match rx.recv().await {
        Some(Ok(transcript)) => transcript,
        Some(Err(e)) => {
            tracing::error!(error = %e, "ASR streaming failed");
            return send_error_timeout(writer, &format!("ASR transcription failed: {e}"), None)
                .await;
        },
        None => {
            tracing::debug!("ASR stream produced no transcript chunks");
            write_event_timeout(writer, &Event::TranscriptStart(TranscriptStartData::new()))
                .await?;
            write_event_timeout(writer, &Event::Transcript(TranscriptData::new(""))).await?;
            return write_event_timeout(writer, &Event::TranscriptStop).await;
        },
    };

    let start = match &first.language {
        Some(language) => TranscriptStartData::new().with_language(language.clone()),
        None => TranscriptStartData::new(),
    };
    write_event_timeout(writer, &Event::TranscriptStart(start)).await?;

    let mut current = first;
    loop {
        if current.is_final {
            let mut result = TranscriptData::new(current.text);
            if let Some(language) = current.language {
                result = result.with_language(language);
            }
            write_event_timeout(writer, &Event::Transcript(result)).await?;
            return write_event_timeout(writer, &Event::TranscriptStop).await;
        }
        write_event_timeout(
            writer,
            &Event::TranscriptChunk(TranscriptChunkData::new(current.text)),
        )
        .await?;

        current = match rx.recv().await {
            Some(Ok(transcript)) => transcript,
            Some(Err(e)) => {
                tracing::error!(error = %e, "ASR streaming failed mid-stream");
                return stop_transcript_stream_with_error(
                    writer,
                    &format!("ASR transcription failed: {e}"),
                )
                .await;
            },
            None => {
                tracing::warn!("ASR stream ended without a final transcript");
                write_event_timeout(writer, &Event::Transcript(TranscriptData::new(""))).await?;
                return write_event_timeout(writer, &Event::TranscriptStop).await;
            },
        };
    }
}

/// Answer a `ping` event with a `pong` echoing the same text.
async fn handle_ping<W>(writer: &mut W, data: PingData) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut pong = PongData::new();
    pong.text = data.text;
    write_event(writer, &Event::Pong(pong)).await?;
    Ok(())
}

/// Answer a `describe` event with service discovery info.
///
/// Lists every TTS model registered in `runtime` as a `tts` program
/// descriptor, including its voices, and every ASR model as an `asr`
/// program descriptor. The `wake` list is always empty: wake-word models
/// aren't served over Wyoming at all.
async fn handle_describe<W>(
    writer: &mut W,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
    asr_map: &AsrModelMap,
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event(
        writer,
        &Event::Info(build_info(runtime, voice_map, asr_map)),
    )
    .await?;
    Ok(())
}

/// Returns the Crane project attribution object used in Wyoming `info` responses.
fn crane_attribution() -> serde_json::Value {
    serde_json::json!({
        "name": "Crane",
        "url": "https://github.com/crane-ai/crane",
    })
}

/// Build the `info` event data from registered TTS and ASR models.
///
/// Each TTS model becomes a `TtsProgram`-shaped JSON value (see the Wyoming
/// `wyoming/info.py` schema), named after its registration name so two
/// loaded models of the same architecture don't collide. Voices claimed by
/// an earlier-priority model (per `voice_map`'s "first model wins" rule)
/// are excluded from a later model's voice list so clients don't see the
/// same voice name twice under different programs. Models present in
/// `runtime` but not part of `voice_map`'s configured set are skipped
/// entirely, since they would otherwise show up with an empty voice list.
///
/// Each ASR model becomes an `AsrProgram`-shaped JSON value, similarly
/// named after its registration name and skipped if not part of
/// `asr_map`'s configured set. `languages` comes from the model's
/// [`AsrHandle::languages`] (i.e. `Asr::supported_languages()`).
fn build_info(runtime: &ModelRuntime, voice_map: &VoiceMap, asr_map: &AsrModelMap) -> InfoData {
    let streaming_enabled = runtime.streaming_enabled();
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
                "supports_synthesize_streaming": streaming_enabled,
            })
        })
        .collect();

    let mut asr_models: Vec<(&str, &AsrHandle)> = runtime.asr_handles().collect();
    asr_models.sort_by_key(|(name, _)| *name);

    let asr = asr_models
        .into_iter()
        .filter(|(model_name, _)| {
            let configured = asr_map.has_model(model_name);
            if !configured {
                tracing::warn!(
                    model = %model_name,
                    "ASR model in runtime but not in ASR model map config; excluding from describe response",
                );
            }
            configured
        })
        .map(|(model_name, handle)| {
            serde_json::json!({
                "name": model_name,
                "models": [{
                    "name": model_name,
                    "languages": handle.languages(),
                    "attribution": crane_attribution(),
                    "installed": true,
                }],
                "attribution": crane_attribution(),
                "installed": true,
                "supports_transcript_streaming": streaming_enabled,
                "requires_external_vad": !runtime.has_vad(),
            })
        })
        .collect();

    InfoData::new().with_tts(tts).with_asr(asr)
}

/// Send an `error` event with an optional machine-readable code.
async fn send_error<W>(writer: &mut W, text: &str, code: Option<&str>) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_event(writer, &Event::Error(error_data(text, code))).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use candle_core::{Device, Tensor};
    use crane::audio::AudioInfo;
    use crane::audio::tts::{Tts, TtsStream, VoiceInfo};
    use crane::audio::{Asr, TranscribeOptions, Transcript};
    use crane_core::models::silero_vad::{Vad, VadConfig};
    use std::io::Cursor as SyncCursor;
    use tokio::io::BufReader;
    use wyoming_protocol::event::SynthesizeChunkData;

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

    /// A `Tts` that records the `language` it was called with, for
    /// asserting what the handler actually passed to the model.
    struct LanguageCapturingTts {
        voices: Vec<VoiceInfo>,
        captured_language: Arc<Mutex<Option<String>>>,
    }

    impl Tts for LanguageCapturingTts {
        fn audio_info(&self) -> AudioInfo {
            AudioInfo {
                sample_rate: 24000,
                channels: 1,
                bits_per_sample: 16,
            }
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            self.voices.clone()
        }

        fn generate_speech(
            &mut self,
            _text: &str,
            language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<candle_core::Tensor> {
            *self.captured_language.lock().unwrap() = Some(language.to_string());
            Tensor::new(vec![0.5f32; 1], &Device::Cpu).map_err(Into::into)
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

    /// A `Tts` that yields its samples as multiple separate chunks from
    /// `generate_speech_stream`, one f32 sample per chunk.
    struct StreamingMockTts {
        audio_info: AudioInfo,
        chunks: Vec<f32>,
    }

    impl StreamingMockTts {
        fn new(sample_rate: u32, chunks: Vec<f32>) -> Self {
            Self {
                audio_info: AudioInfo {
                    sample_rate,
                    channels: 1,
                    bits_per_sample: 16,
                },
                chunks,
            }
        }
    }

    impl Tts for StreamingMockTts {
        fn audio_info(&self) -> AudioInfo {
            self.audio_info
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            voices(&["alice"])
        }

        fn generate_speech(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<candle_core::Tensor> {
            Tensor::new(self.chunks.clone(), &Device::Cpu).map_err(Into::into)
        }

        fn generate_speech_stream(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<TtsStream<'_>> {
            let chunks: Vec<Result<candle_core::Tensor>> = self
                .chunks
                .iter()
                .map(|&v| Tensor::new(vec![v], &Device::Cpu).map_err(Into::into))
                .collect();
            Ok(TtsStream::new(self.audio_info, chunks.into_iter()))
        }
    }

    /// A `Tts` whose stream yields one good chunk, then fails.
    struct MidStreamFailingTts {
        audio_info: AudioInfo,
    }

    impl Tts for MidStreamFailingTts {
        fn audio_info(&self) -> AudioInfo {
            self.audio_info
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
            anyhow::bail!("should not be called")
        }

        fn generate_speech_stream(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<TtsStream<'_>> {
            let good: Result<candle_core::Tensor> =
                Tensor::new(vec![0.5f32], &Device::Cpu).map_err(Into::into);
            let chunks: Vec<Result<candle_core::Tensor>> =
                vec![good, Err(anyhow::anyhow!("mid-stream failure"))];
            Ok(TtsStream::new(self.audio_info, chunks.into_iter()))
        }
    }

    fn test_runtime() -> ModelRuntime {
        ModelRuntime::new()
    }

    /// Registers `tts` on `Device::Cpu` -- the device is irrelevant to these
    /// tests since `with_context` is a cheap no-op wrapper on CPU either way.
    fn register_test_tts(
        rt: &mut ModelRuntime,
        name: &str,
        model_type_name: &'static str,
        tts: Box<dyn Tts + Send>,
    ) {
        rt.register_tts(name.into(), model_type_name, tts, &Device::Cpu)
            .unwrap();
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
        asr_map: &AsrModelMap,
        events: Vec<Event>,
    ) -> Vec<Event> {
        let mut input = Vec::new();
        for event in &events {
            write_event(&mut input, event).await.unwrap();
        }
        let mut reader = BufReader::new(SyncCursor::new(input));
        let mut output = Vec::new();
        handle_connection(&mut reader, &mut output, runtime, voice_map, asr_map)
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
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new("hello"))],
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
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        );
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hi").with_voice(SynthesizeVoice::with_name("bob")),
            )],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.rate, 16000),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_language_fallback_voice() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "casual".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new(
                16000,
                vec![VoiceInfo {
                    name: "de_female".into(),
                    languages: vec!["de".into()],
                }],
            )),
        );
        // "a" is the default model (registered first), but a request for
        // German with no explicit voice must resolve to "b"'s German voice
        // instead of silently falling back to the (English) default.
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        let mut voice = SynthesizeVoice::new();
        voice.language = Some("de-DE".to_string());
        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hi").with_voice(voice),
            )],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.rate, 16000),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_named_voice_overrides_language() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "casual".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new(
                16000,
                vec![VoiceInfo {
                    name: "de_female".into(),
                    languages: vec!["de".into()],
                }],
            )),
        );
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        // Explicit voice name ("casual", model "a") conflicts with the
        // language ("de", which matches model "b"); the name must win.
        let mut voice = SynthesizeVoice::with_name("casual");
        voice.language = Some("de".to_string());
        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hi").with_voice(voice),
            )],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.rate, 24000),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_language_dispatch_by_model_type() {
        // Qwen3-TTS's codec_language_id table is keyed by full English
        // names, but Voxtral takes ISO 639-1 codes directly; requesting
        // "de-DE" must reach each model in its own expected format.
        let qwen3_language = Arc::new(Mutex::new(None));
        let voxtral_language = Arc::new(Mutex::new(None));

        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_tts(
            &mut rt,
            "q",
            "qwen3_tts",
            Box::new(LanguageCapturingTts {
                voices: voices(&["q_voice"]),
                captured_language: Arc::clone(&qwen3_language),
            }),
        );
        register_test_tts(
            &mut rt,
            "v",
            "voxtral_tts",
            Box::new(LanguageCapturingTts {
                voices: voices(&["v_voice"]),
                captured_language: Arc::clone(&voxtral_language),
            }),
        );
        let vm = VoiceMap::new(&["q".to_string(), "v".to_string()], &rt);

        let mut voice = SynthesizeVoice::with_name("q_voice");
        voice.language = Some("de-DE".to_string());
        run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hi").with_voice(voice),
            )],
        )
        .await;

        let mut voice = SynthesizeVoice::with_name("v_voice");
        voice.language = Some("de-DE".to_string());
        run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hi").with_voice(voice),
            )],
        )
        .await;

        assert_eq!(qwen3_language.lock().unwrap().as_deref(), Some("german"));
        assert_eq!(voxtral_language.lock().unwrap().as_deref(), Some("de"));
    }

    #[tokio::test]
    async fn test_synthesize_unmatched_language_falls_back_to_default() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "casual".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new(
                16000,
                vec![VoiceInfo {
                    name: "de_female".into(),
                    languages: vec!["de".into()],
                }],
            )),
        );
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        // "fr" matches no registered voice, so this must fall back to the
        // default model "a", not error and not pick "b".
        let mut voice = SynthesizeVoice::new();
        voice.language = Some("fr".to_string());
        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hi").with_voice(voice),
            )],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.rate, 24000),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_unknown_voice() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hi").with_voice(SynthesizeVoice::with_name("ghost")),
            )],
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
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new("hi"))],
        )
        .await;

        assert_eq!(results.len(), 1);
        assert!(matches!(&results[0], Event::Error(_)));
    }

    #[tokio::test]
    async fn test_synthesize_generation_failure() {
        let mut rt = test_runtime();
        register_test_tts(&mut rt, "m1", "qwen3_tts", Box::new(FailingTts));
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new("hi"))],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert!(data.text.contains("mock generation failure")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_streams_multiple_chunks() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(StreamingMockTts::new(24000, vec![0.1, 0.2, 0.3])),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new("hello"))],
        )
        .await;

        assert_eq!(
            results.len(),
            5,
            "expected AudioStart + 3 AudioChunk + AudioStop"
        );
        assert!(matches!(results[0], Event::AudioStart(_)));
        for (i, expected) in [0.1f32, 0.2, 0.3].into_iter().enumerate() {
            match &results[i + 1] {
                Event::AudioChunk { audio, .. } => {
                    assert_eq!(*audio, pcm_f32_to_i16(&[expected]));
                },
                other => panic!("expected AudioChunk, got {other:?}"),
            }
        }
        assert!(matches!(results[4], Event::AudioStop(_)));
    }

    #[tokio::test]
    async fn test_synthesize_streaming_empty_stream() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(StreamingMockTts::new(24000, vec![])),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new(String::new()))],
        )
        .await;

        assert_eq!(
            results.len(),
            2,
            "expected AudioStart + AudioStop, not an Error"
        );
        assert!(matches!(results[0], Event::AudioStart(_)));
        assert!(matches!(results[1], Event::AudioStop(_)));
    }

    #[tokio::test]
    async fn test_synthesize_blob_fallback_when_streaming_disabled() {
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(StreamingMockTts::new(24000, vec![0.1, 0.2, 0.3])),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new("hello"))],
        )
        .await;

        assert_eq!(
            results.len(),
            3,
            "expected AudioStart + one AudioChunk (blob) + AudioStop, not per-chunk streaming"
        );
        assert!(matches!(results[0], Event::AudioStart(_)));
        match &results[1] {
            Event::AudioChunk { audio, .. } => {
                assert_eq!(*audio, pcm_f32_to_i16(&[0.1, 0.2, 0.3]));
            },
            other => panic!("expected AudioChunk, got {other:?}"),
        }
        assert!(matches!(results[2], Event::AudioStop(_)));
    }

    #[tokio::test]
    async fn test_synthesize_mid_stream_failure_sends_stop_then_error() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MidStreamFailingTts {
                audio_info: AudioInfo {
                    sample_rate: 24000,
                    channels: 1,
                    bits_per_sample: 16,
                },
            }),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new("hi"))],
        )
        .await;

        assert_eq!(
            results.len(),
            4,
            "expected AudioStart + AudioChunk + AudioStop + Error"
        );
        assert!(matches!(results[0], Event::AudioStart(_)));
        assert!(matches!(results[1], Event::AudioChunk { .. }));
        assert!(matches!(results[2], Event::AudioStop(_)));
        match &results[3] {
            Event::Error(data) => assert!(data.text.contains("mid-stream failure")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_streamed_text_joins_chunks() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![
                Event::SynthesizeStart(
                    SynthesizeStartData::new().with_voice(SynthesizeVoice::with_name("alice")),
                ),
                Event::SynthesizeChunk(SynthesizeChunkData::new("hel")),
                Event::SynthesizeChunk(SynthesizeChunkData::new("lo")),
                Event::SynthesizeStop,
            ],
        )
        .await;

        // Same output as a one-shot `synthesize` of the joined text, plus
        // the terminating `synthesize-stopped` of the streamed exchange:
        // MockTts emits one sample per character, so 5 samples prove the
        // chunks were concatenated into "hello".
        assert_eq!(results.len(), 4);
        assert!(matches!(results[0], Event::AudioStart(_)));
        match &results[1] {
            Event::AudioChunk { audio, .. } => {
                assert_eq!(*audio, pcm_f32_to_i16(&[0.5f32; 5]));
            },
            other => panic!("expected AudioChunk, got {other:?}"),
        }
        assert!(matches!(results[2], Event::AudioStop(_)));
        // Home Assistant's streaming TTS reader waits for this event.
        assert_eq!(results[3], Event::SynthesizeStopped);
    }

    #[tokio::test]
    async fn test_synthesize_streamed_text_language_selects_model() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "casual".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new(
                16000,
                vec![VoiceInfo {
                    name: "de_female".into(),
                    languages: vec!["de".into()],
                }],
            )),
        );
        // "a" is the default model (registered first), but a streamed
        // request for German with no explicit voice must resolve to "b"'s
        // German voice, like a one-shot synthesize does.
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![
                Event::SynthesizeStart(SynthesizeStartData::new().with_language("de-DE")),
                Event::SynthesizeChunk(SynthesizeChunkData::new("hallo")),
                Event::SynthesizeStop,
            ],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.rate, 16000),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_synthesize_no_synthesize_stopped_for_one_shot() {
        // `synthesize-stopped` terminates streamed-text exchanges only; a
        // one-shot `synthesize` ends at `audio-stop` (matching reference
        // Wyoming servers and HA's non-streaming reader).
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(
                SynthesizeData::new("hello").with_voice(SynthesizeVoice::with_name("alice")),
            )],
        )
        .await;

        assert_eq!(results.len(), 3);
        assert!(matches!(results[0], Event::AudioStart(_)));
        assert!(matches!(results[1], Event::AudioChunk { .. }));
        assert!(matches!(results[2], Event::AudioStop(_)));
    }

    #[tokio::test]
    async fn test_synthesize_streamed_text_unexpected_event() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![
                Event::SynthesizeStart(SynthesizeStartData::new()),
                Event::SynthesizeChunk(SynthesizeChunkData::new("hi")),
                Event::Ping(PingData::new()),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => {
                assert!(data.text.contains("synthesize-chunk"));
            },
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
            &AsrModelMap::new(&[], &rt),
            vec![Event::Ping(PingData::with_text("hi"))],
        )
        .await;

        assert_eq!(results, vec![Event::Pong(PongData::with_text("hi"))]);
    }

    #[tokio::test]
    async fn test_ping_pong_no_text() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Ping(PingData::new())],
        )
        .await;

        assert_eq!(results, vec![Event::Pong(PongData::new())]);
    }

    #[tokio::test]
    async fn test_describe_no_models() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);

        let results =
            run_events(&rt, &vm, &AsrModelMap::new(&[], &rt), vec![Event::Describe]).await;

        assert_eq!(results, vec![Event::Info(InfoData::default())]);
    }

    #[tokio::test]
    async fn test_describe_single_model() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results =
            run_events(&rt, &vm, &AsrModelMap::new(&[], &rt), vec![Event::Describe]).await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts.len(), 1);
                assert!(data.asr.is_empty());
                assert!(data.wake.is_empty());
                let program = &data.tts[0];
                assert_eq!(program["name"], "m1");
                assert_eq!(program["supports_synthesize_streaming"], true);
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
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        );
        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);

        let results =
            run_events(&rt, &vm, &AsrModelMap::new(&[], &rt), vec![Event::Describe]).await;

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
        register_test_tts(
            &mut rt,
            "first",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        register_test_tts(
            &mut rt,
            "second",
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["first".to_string(), "second".to_string()], &rt);

        let results =
            run_events(&rt, &vm, &AsrModelMap::new(&[], &rt), vec![Event::Describe]).await;

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
        register_test_tts(
            &mut rt,
            "configured",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        register_test_tts(
            &mut rt,
            "unconfigured",
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        );
        let vm = VoiceMap::new(&["configured".to_string()], &rt);

        let results =
            run_events(&rt, &vm, &AsrModelMap::new(&[], &rt), vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts.len(), 1);
                assert_eq!(data.tts[0]["name"], "configured");
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_streaming_is_true() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results =
            run_events(&rt, &vm, &AsrModelMap::new(&[], &rt), vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts[0]["supports_synthesize_streaming"], true);
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_streaming_disabled() {
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results =
            run_events(&rt, &vm, &AsrModelMap::new(&[], &rt), vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts[0]["supports_synthesize_streaming"], false);
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_asr_streaming_is_true() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(&rt, &VoiceMap::new(&[], &rt), &am, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.asr[0]["supports_transcript_streaming"], true);
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_asr_streaming_disabled() {
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(&rt, &VoiceMap::new(&[], &rt), &am, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.asr[0]["supports_transcript_streaming"], false);
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_single_asr_model() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(&rt, &vm, &am, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert!(data.tts.is_empty());
                assert_eq!(data.asr.len(), 1);
                let program = &data.asr[0];
                assert_eq!(program["name"], "m1");
                assert_eq!(program["supports_transcript_streaming"], true);
                let models = program["models"].as_array().unwrap();
                assert_eq!(models.len(), 1);
                assert_eq!(models[0]["name"], "m1");
                assert_eq!(models[0]["languages"], serde_json::json!([]));
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_asr_model_languages_populated() {
        let mut rt = test_runtime();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "m1",
                languages: vec!["de", "en"],
            }),
        );
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(&rt, &vm, &am, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                let models = data.asr[0]["models"].as_array().unwrap();
                assert_eq!(models[0]["languages"], serde_json::json!(["de", "en"]));
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_excludes_asr_model_not_in_asr_map() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "configured", "qwen3_asr", Box::new(MockAsr));
        register_test_asr(&mut rt, "unconfigured", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["configured".to_string()], &rt);

        let results = run_events(&rt, &vm, &am, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.asr.len(), 1);
                assert_eq!(data.asr[0]["name"], "configured");
            },
            other => panic!("expected Info, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_describe_tts_and_asr_together() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "t1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        register_test_asr(&mut rt, "a1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&["t1".to_string()], &rt);
        let am = AsrModelMap::new(&["a1".to_string()], &rt);

        let results = run_events(&rt, &vm, &am, vec![Event::Describe]).await;

        match &results[0] {
            Event::Info(data) => {
                assert_eq!(data.tts.len(), 1);
                assert_eq!(data.tts[0]["name"], "t1");
                assert_eq!(data.asr.len(), 1);
                assert_eq!(data.asr[0]["name"], "a1");
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
            &AsrModelMap::new(&[], &rt),
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
        let am = AsrModelMap::new(&[], &rt);
        let mut reader = BufReader::new(SyncCursor::new(Vec::<u8>::new()));
        let mut writer = Vec::new();
        let result = handle_connection(&mut reader, &mut writer, &rt, &vm, &am).await;
        assert!(result.is_ok());
        assert!(writer.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_events_sequential() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![
                Event::Ping(PingData::with_text("a")),
                Event::Synthesize(SynthesizeData::new("hi")),
                Event::Ping(PingData::with_text("b")),
            ],
        )
        .await;

        assert_eq!(results.len(), 5);
        assert_eq!(results[0], Event::Pong(PongData::with_text("a")));
        assert!(matches!(results[1], Event::AudioStart(_)));
        assert!(matches!(results[2], Event::AudioChunk { .. }));
        assert!(matches!(results[3], Event::AudioStop(_)));
        assert_eq!(results[4], Event::Pong(PongData::with_text("b")));
    }

    #[test]
    fn test_voice_map_first_wins() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "first",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        register_test_tts(
            &mut rt,
            "second",
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["alice"]))),
        );

        let vm = VoiceMap::new(&["first".to_string(), "second".to_string()], &rt);
        assert_eq!(vm.model_for_voice("alice"), Some("first"));
    }

    #[test]
    fn test_voice_map_separate_voices() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new(16000, voices(&["bob"]))),
        );

        let vm = VoiceMap::new(&["a".to_string(), "b".to_string()], &rt);
        assert_eq!(vm.model_for_voice("alice"), Some("a"));
        assert_eq!(vm.model_for_voice("bob"), Some("b"));
        assert_eq!(vm.model_for_voice("ghost"), None);
    }

    #[test]
    fn test_voice_map_default_model() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
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

    #[test]
    fn test_voice_map_language_lookup() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![
                    VoiceInfo {
                        name: "de_female".into(),
                        languages: vec!["de".into()],
                    },
                    VoiceInfo {
                        name: "casual".into(),
                        languages: vec!["en".into()],
                    },
                ],
            )),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        assert_eq!(vm.model_for_language("de"), Some(("m1", "de_female")));
        // Matching is on the base subtag only, case-insensitively, and
        // accepts both `-` and `_` as the subtag separator.
        assert_eq!(vm.model_for_language("DE-DE"), Some(("m1", "de_female")));
        assert_eq!(vm.model_for_language("de_DE"), Some(("m1", "de_female")));
        assert_eq!(vm.model_for_language("fr"), None);
    }

    #[test]
    fn test_voice_map_language_first_model_wins() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "first",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["de".into()],
                }],
            )),
        );
        register_test_tts(
            &mut rt,
            "second",
            "voxtral_tts",
            Box::new(MockTts::new(
                16000,
                vec![VoiceInfo {
                    name: "bob".into(),
                    languages: vec!["de".into()],
                }],
            )),
        );

        let vm = VoiceMap::new(&["first".to_string(), "second".to_string()], &rt);
        assert_eq!(vm.model_for_language("de"), Some(("first", "alice")));
    }

    /// Shared `transcribe` body for [`MockAsr`] and [`MockAsrWithRate`],
    /// which differ only in `input_sample_rate`.
    fn mock_transcribe(audio: &[f32], opts: &TranscribeOptions) -> Transcript {
        Transcript {
            text: format!("heard {} samples", audio.len()),
            language: opts.language.clone(),
            is_final: true,
        }
    }

    struct MockAsr;

    impl Asr for MockAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(mock_transcribe(audio, opts))
        }
    }

    /// An ASR model reporting a sample rate Silero VAD does not support
    /// (only 8000/16000 are valid), for testing that VAD pre-filtering is
    /// skipped rather than fed an incompatible rate.
    struct MockAsrWithRate {
        rate: u32,
    }

    impl Asr for MockAsrWithRate {
        fn input_sample_rate(&self) -> u32 {
            self.rate
        }

        fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(mock_transcribe(audio, opts))
        }
    }

    /// An ASR model that claims to support a fixed set of languages, for
    /// testing [`AsrModelMap`]'s language-based resolution. `tag` is
    /// embedded in the transcript text so tests can tell which model
    /// instance actually handled a request.
    struct MockAsrWithLanguages {
        tag: &'static str,
        languages: Vec<&'static str>,
    }

    impl Asr for MockAsrWithLanguages {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(Transcript {
                text: format!("{}: heard {} samples", self.tag, audio.len()),
                language: opts.language.clone(),
                is_final: true,
            })
        }

        fn supported_languages(&self) -> Vec<String> {
            self.languages.iter().map(|s| (*s).to_string()).collect()
        }
    }

    struct FailingAsr;

    impl Asr for FailingAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, _audio: &[f32], _opts: &TranscribeOptions) -> Result<Transcript> {
            anyhow::bail!("mock transcription failure")
        }
    }

    /// An ASR model with true incremental decoding: `transcribe_stream`
    /// yields one chunk per string in `chunks`, marking only the last as
    /// final.
    struct MockStreamingAsr {
        chunks: Vec<&'static str>,
    }

    impl Asr for MockStreamingAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(Transcript {
                text: format!("heard {} samples", audio.len()),
                language: opts.language.clone(),
                is_final: true,
            })
        }

        fn transcribe_stream(
            &mut self,
            _audio: &[f32],
            opts: &TranscribeOptions,
        ) -> Result<crane::audio::AsrStream<'_>> {
            let last = self.chunks.len().saturating_sub(1);
            let language = opts.language.clone();
            let chunks: Vec<Result<Transcript>> = self
                .chunks
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    Ok(Transcript {
                        text: (*text).to_string(),
                        language: if i == last { language.clone() } else { None },
                        is_final: i == last,
                    })
                })
                .collect();
            Ok(crane::audio::AsrStream::new(chunks.into_iter()))
        }
    }

    /// An ASR model whose streaming transcription fails after producing one
    /// partial chunk, used to test the mid-stream error convention.
    struct MidStreamFailingAsr;

    impl Asr for MidStreamFailingAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, _audio: &[f32], _opts: &TranscribeOptions) -> Result<Transcript> {
            anyhow::bail!("mock transcription failure")
        }

        fn transcribe_stream(
            &mut self,
            _audio: &[f32],
            _opts: &TranscribeOptions,
        ) -> Result<crane::audio::AsrStream<'_>> {
            let chunks: Vec<Result<Transcript>> = vec![
                Ok(Transcript {
                    text: "partial".to_string(),
                    language: None,
                    is_final: false,
                }),
                Err(anyhow::anyhow!("mock mid-stream failure")),
            ];
            Ok(crane::audio::AsrStream::new(chunks.into_iter()))
        }
    }

    /// Registers `asr` on `Device::Cpu` -- the device is irrelevant to these
    /// tests since `with_context` is a cheap no-op wrapper on CPU either way.
    fn register_test_asr(
        rt: &mut ModelRuntime,
        name: &str,
        model_type_name: &'static str,
        asr: Box<dyn Asr + Send>,
    ) {
        rt.register_asr(name.into(), model_type_name, asr, &Device::Cpu)
            .unwrap();
    }

    #[test]
    fn test_asr_model_map_single_model() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));

        let am = AsrModelMap::new(&["m1".to_string()], &rt);
        assert!(am.has_model("m1"));
    }

    #[test]
    fn test_asr_model_map_multiple_models() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "a", "qwen3_asr", Box::new(MockAsr));
        register_test_asr(&mut rt, "b", "qwen3_asr", Box::new(MockAsr));

        let am = AsrModelMap::new(&["a".to_string(), "b".to_string()], &rt);
        assert!(am.has_model("a"));
        assert!(am.has_model("b"));
        assert!(!am.has_model("ghost"));
    }

    #[test]
    fn test_asr_model_map_default_model() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));

        let am = AsrModelMap::new(&["m1".to_string()], &rt);
        assert_eq!(am.default_model(), Some("m1"));
    }

    #[test]
    fn test_asr_model_map_no_models() {
        let rt = test_runtime();
        let am = AsrModelMap::new(&[], &rt);
        assert_eq!(am.default_model(), None);
        assert!(!am.has_model("anything"));
    }

    #[test]
    fn test_asr_model_map_skips_names_not_in_runtime() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));

        // "ghost" is listed in the configured names but was never
        // registered on the runtime, so it must not appear in the map.
        let am = AsrModelMap::new(&["m1".to_string(), "ghost".to_string()], &rt);
        assert!(am.has_model("m1"));
        assert!(!am.has_model("ghost"));
    }

    #[test]
    fn test_asr_model_map_language_lookup() {
        let mut rt = test_runtime();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "m1",
                languages: vec!["de", "en"],
            }),
        );
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        assert_eq!(am.model_for_language("de"), Some("m1"));
        // Matching is on the base subtag only, case-insensitively, and
        // accepts both `-` and `_` as the subtag separator.
        assert_eq!(am.model_for_language("DE-DE"), Some("m1"));
        assert_eq!(am.model_for_language("de_DE"), Some("m1"));
        assert_eq!(am.model_for_language("fr"), None);
    }

    #[test]
    fn test_asr_model_map_language_first_model_wins() {
        let mut rt = test_runtime();
        register_test_asr(
            &mut rt,
            "first",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "first",
                languages: vec!["de"],
            }),
        );
        register_test_asr(
            &mut rt,
            "second",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "second",
                languages: vec!["de"],
            }),
        );

        let am = AsrModelMap::new(&["first".to_string(), "second".to_string()], &rt);
        assert_eq!(am.model_for_language("de"), Some("first"));
    }

    /// 16kHz mono 16-bit PCM format, matching `MockAsr::input_sample_rate`.
    fn mock_asr_format() -> AudioFormat {
        AudioFormat {
            rate: 16000,
            width: 2,
            channels: 1,
        }
    }

    #[tokio::test]
    async fn test_transcribe_default_model() {
        // Batch mode: streaming disabled, so a single `transcript` is
        // expected with no transcript-start/stop wrapper.
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(TranscriptData::new("heard 4 samples"))]
        );
    }

    #[tokio::test]
    async fn test_transcribe_named_model() {
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(&mut rt, "a", "qwen3_asr", Box::new(MockAsr));
        register_test_asr(&mut rt, "b", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["a".to_string(), "b".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new().with_name("b")),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 2]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(TranscriptData::new("heard 2 samples"))]
        );
    }

    #[tokio::test]
    async fn test_transcribe_language_hint_forwarded() {
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new().with_language("de")),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 1]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(
                TranscriptData::new("heard 1 samples").with_language("de")
            )]
        );
    }

    #[tokio::test]
    async fn test_transcribe_language_selects_model() {
        // "default" is registered first (making it the runtime's default)
        // and would fail if selected, so a passing test proves the "de"
        // language hint actually routed to "german" instead of falling
        // through to the default.
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(&mut rt, "default", "qwen3_asr", Box::new(FailingAsr));
        register_test_asr(
            &mut rt,
            "german",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "german",
                languages: vec!["de"],
            }),
        );
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["default".to_string(), "german".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new().with_language("de")),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 1]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(
                TranscriptData::new("german: heard 1 samples").with_language("de")
            )]
        );
    }

    #[tokio::test]
    async fn test_transcribe_name_overrides_language() {
        // "german" claims "de" but is requested by an explicit name that
        // doesn't match its language, while "named" is requested by name
        // and claims no language at all. The explicit name must win.
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(
            &mut rt,
            "german",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "german",
                languages: vec!["de"],
            }),
        );
        register_test_asr(
            &mut rt,
            "named",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "named",
                languages: vec![],
            }),
        );
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["german".to_string(), "named".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new().with_name("named").with_language("de")),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 1]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(
                TranscriptData::new("named: heard 1 samples").with_language("de")
            )]
        );
    }

    #[tokio::test]
    async fn test_transcribe_unmatched_language_falls_back_to_default() {
        // "default" is registered first (making it the runtime's default)
        // and claims "en", while "german" claims "de". Neither matches the
        // requested "fr", so this must fall back to "default" rather than
        // erroneously picking "german".
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(
            &mut rt,
            "default",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "default",
                languages: vec!["en"],
            }),
        );
        register_test_asr(
            &mut rt,
            "german",
            "qwen3_asr",
            Box::new(MockAsrWithLanguages {
                tag: "german",
                languages: vec!["de"],
            }),
        );
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["default".to_string(), "german".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new().with_language("fr")),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 1]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(
                TranscriptData::new("default: heard 1 samples").with_language("fr")
            )]
        );
    }

    #[tokio::test]
    async fn test_transcribe_no_asr_model_loaded() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&[], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![Event::Transcribe(TranscribeData::new())],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert_eq!(data.code, None),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_unknown_model() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![Event::Transcribe(TranscribeData::new().with_name("ghost"))],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert_eq!(data.code.as_deref(), Some("asr-model-not-found")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_wrong_sample_rate() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(AudioFormat {
                    rate: 8000,
                    ..mock_asr_format()
                })),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => {
                assert_eq!(data.code.as_deref(), Some("unsupported-audio-format"));
            },
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_wrong_channels() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(AudioFormat {
                    channels: 2,
                    ..mock_asr_format()
                })),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => {
                assert_eq!(data.code.as_deref(), Some("unsupported-audio-format"));
            },
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_wrong_width() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(AudioFormat {
                    width: 4,
                    ..mock_asr_format()
                })),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => {
                assert_eq!(data.code.as_deref(), Some("unsupported-audio-format"));
            },
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_chunk_format_mismatch() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(AudioFormat {
                        rate: 8000,
                        ..mock_asr_format()
                    }),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => {
                assert_eq!(data.code.as_deref(), Some("unsupported-audio-format"));
            },
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_multi_chunk() {
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 2]),
                },
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 3]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(TranscriptData::new("heard 5 samples"))]
        );
    }

    #[tokio::test]
    async fn test_transcribe_audio_too_large() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        // One f32 sample becomes 2 PCM bytes; send one more sample than fits
        // in MAX_TRANSCRIBE_AUDIO_BYTES.
        let too_many_samples = MAX_TRANSCRIBE_AUDIO_BYTES / 2 + 1;
        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&vec![0.0; too_many_samples]),
                },
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => {
                assert_eq!(data.code.as_deref(), Some("audio-too-large"));
            },
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_unexpected_event_during_chunks() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::Ping(PingData::new()),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        assert!(matches!(&results[0], Event::Error(_)));
    }

    #[tokio::test]
    async fn test_transcribe_transcription_failure() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(FailingAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert!(data.text.contains("mock transcription failure")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_vad_skipped_for_incompatible_sample_rate() {
        // The loaded `Vad` has no ONNX model (`Vad::load` was never
        // called), so if the handler tried to filter through it the
        // request would fail. A successful transcript here proves the
        // 24 kHz ASR rate caused VAD pre-filtering to be skipped rather
        // than attempted.
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(MockAsrWithRate { rate: 24000 }),
        );
        rt.set_vad(Vad::new(VadConfig::default()).unwrap());
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(format)),
                Event::AudioChunk {
                    data: AudioChunkData::new(format),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![Event::Transcript(TranscriptData::new("heard 4 samples"))]
        );
    }

    #[tokio::test]
    async fn test_transcribe_vad_filter_failure_sends_error() {
        // The loaded `Vad` has no ONNX model, so filtering non-trivial
        // audio at a VAD-compatible rate (16 kHz) fails; the handler must
        // surface that as an `error` event rather than propagating a hard
        // connection error.
        let mut rt = test_runtime();
        rt.set_streaming_enabled(false);
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        rt.set_vad(Vad::new(VadConfig::default()).unwrap());
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 1024]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert!(data.text.contains("VAD filtering failed")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_streaming_multi_chunk() {
        // Streaming enabled (the test_runtime default): a model with true
        // incremental decoding delivers transcript-start, one
        // transcript-chunk per partial result, a final transcript, then
        // transcript-stop.
        let mut rt = test_runtime();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(MockStreamingAsr {
                chunks: vec!["one", "one two", "one two three"],
            }),
        );
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![
                Event::TranscriptStart(TranscriptStartData::new()),
                Event::TranscriptChunk(TranscriptChunkData::new("one")),
                Event::TranscriptChunk(TranscriptChunkData::new("one two")),
                Event::Transcript(TranscriptData::new("one two three")),
                Event::TranscriptStop,
            ]
        );
    }

    #[tokio::test]
    async fn test_transcribe_streaming_single_chunk_default_impl() {
        // MockAsr does not override transcribe_stream, so the default impl
        // wraps transcribe() in a single final chunk: transcript-start,
        // transcript (no transcript-chunk), transcript-stop.
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![
                Event::TranscriptStart(TranscriptStartData::new()),
                Event::Transcript(TranscriptData::new("heard 4 samples")),
                Event::TranscriptStop,
            ]
        );
    }

    #[tokio::test]
    async fn test_transcribe_streaming_language_forwarded() {
        let mut rt = test_runtime();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(MockStreamingAsr {
                chunks: vec!["hallo"],
            }),
        );
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new().with_language("de")),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 1]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![
                Event::TranscriptStart(TranscriptStartData::new().with_language("de")),
                Event::Transcript(TranscriptData::new("hallo").with_language("de")),
                Event::TranscriptStop,
            ]
        );
    }

    #[tokio::test]
    async fn test_transcribe_streaming_failure_before_start() {
        // A failure while setting up the stream (transcribe_stream itself
        // errors) surfaces as a plain error -- no transcript-start has been
        // sent yet.
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(FailingAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        match &results[0] {
            Event::Error(data) => assert!(data.text.contains("mock transcription failure")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_streaming_mid_stream_failure_sends_stop_then_error() {
        // A failure after the first (partial) chunk closes the stream with
        // transcript-stop before reporting the error, per crane-wyoming's
        // mid-stream error convention.
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MidStreamFailingAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(results.len(), 4);
        assert_eq!(
            results[0],
            Event::TranscriptStart(TranscriptStartData::new())
        );
        assert_eq!(
            results[1],
            Event::TranscriptChunk(TranscriptChunkData::new("partial"))
        );
        assert_eq!(results[2], Event::TranscriptStop);
        match &results[3] {
            Event::Error(data) => assert!(data.text.contains("mock mid-stream failure")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_transcribe_streaming_empty_stream() {
        let mut rt = test_runtime();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(MockStreamingAsr { chunks: vec![] }),
        );
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::AudioStart(AudioStartData::new(mock_asr_format())),
                Event::AudioChunk {
                    data: AudioChunkData::new(mock_asr_format()),
                    audio: pcm_f32_to_i16(&[0.0; 4]),
                },
                Event::AudioStop(AudioStopData::new()),
            ],
        )
        .await;

        assert_eq!(
            results,
            vec![
                Event::TranscriptStart(TranscriptStartData::new()),
                Event::Transcript(TranscriptData::new("")),
                Event::TranscriptStop,
            ]
        );
    }

    #[tokio::test]
    async fn test_transcribe_missing_audio_start() {
        let mut rt = test_runtime();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr));
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &am,
            vec![
                Event::Transcribe(TranscribeData::new()),
                Event::Ping(PingData::new()),
            ],
        )
        .await;

        assert_eq!(results.len(), 1);
        assert!(matches!(&results[0], Event::Error(_)));
    }

    #[tokio::test]
    async fn test_audio_start_width_is_bytes() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(24000, voices(&["alice"]))),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);

        let results = run_events(
            &rt,
            &vm,
            &AsrModelMap::new(&[], &rt),
            vec![Event::Synthesize(SynthesizeData::new("hi"))],
        )
        .await;

        match &results[0] {
            Event::AudioStart(data) => assert_eq!(data.width, 2),
            other => panic!("expected AudioStart, got {other:?}"),
        }
    }
}
