//! Protocol-independent TTS model runtime.
//!
//! [`ModelRuntime`] owns every loaded TTS model, keyed by registration name.
//! Consumers load models via [`ModelRuntime::load_tts`], then send
//! generation requests through [`ModelRuntime::generate_speech`] /
//! [`ModelRuntime::generate_speech_stream`]. Each TTS model runs on its own
//! dedicated OS thread and is addressed through channels, so [`ModelRuntime`]
//! can be shared via `Arc` across async tasks without locking.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tokio::sync::{mpsc, oneshot};

use crane::audio::tts::{AudioInfo, Tts, VoiceInfo};
use crane_core::generation::SpeechOptions;

use crate::engine::model_factory::{self, ModelType};

/// A request to generate speech, sent to a TTS model's dedicated thread.
///
/// This type is transport-agnostic: it carries no HTTP- or Wyoming-specific
/// fields. The response is a raw f32 PCM [`Tensor`]; encoding to WAV, PCM
/// bytes, or any other wire format is the caller's responsibility.
pub struct TtsGenerateRequest {
    /// Text to synthesize.
    pub text: String,
    /// Target language (e.g. "english", "auto").
    pub language: String,
    /// Voice name from [`TtsHandle::voices`], or `None` for the model default.
    pub voice: Option<String>,
    /// Generation parameters (temperature, max tokens, etc.).
    pub opts: SpeechOptions,
    /// Reference audio path for voice cloning. `None` for a predefined voice.
    pub reference_audio: Option<String>,
    /// Transcript of the reference audio (required by some models).
    pub reference_text: Option<String>,
    /// Channel to send back the generated audio tensor.
    pub response_tx: oneshot::Sender<Result<Tensor>>,
}

/// A request to generate speech incrementally, sent to a TTS model's
/// dedicated thread.
///
/// Unlike [`TtsGenerateRequest`], the response is delivered as a series of
/// chunks over `chunk_tx` rather than a single tensor -- one send per chunk
/// yielded by [`Tts::generate_speech_stream`], followed by the sender being
/// dropped to signal completion. An `Err` chunk ends the stream immediately.
///
/// # Backpressure
///
/// `chunk_tx` is bounded, and the model's worker thread sends on it with a
/// blocking call. A consumer that stops draining the matching receiver
/// without dropping it will stall the worker thread indefinitely -- which
/// also blocks every other request (streaming or blob) queued for the same
/// model, since they share one thread. Callers must keep draining or drop
/// the receiver promptly.
pub(crate) struct TtsStreamRequest {
    /// Text to synthesize.
    pub(crate) text: String,
    /// Target language (e.g. "english", "auto").
    pub(crate) language: String,
    /// Voice name from [`TtsHandle::voices`], or `None` for the model default.
    pub(crate) voice: Option<String>,
    /// Generation parameters (temperature, max tokens, etc.).
    pub(crate) opts: SpeechOptions,
    /// Channel to send generated audio chunks back as they're produced.
    pub(crate) chunk_tx: mpsc::Sender<Result<Tensor>>,
}

/// A request queued on a TTS model's dedicated thread: either a blob
/// [`TtsGenerateRequest`] or an incremental [`TtsStreamRequest`].
///
/// Both variants share one channel/thread so streaming and non-streaming
/// requests for the same model are processed in a single FIFO queue,
/// matching the "one request at a time" concurrency model documented on
/// [`TtsHandle`].
enum TtsRequest {
    /// Generate the complete waveform and return it in one response.
    Generate(TtsGenerateRequest),
    /// Generate the waveform incrementally, streaming chunks as produced.
    Stream(TtsStreamRequest),
}

/// Handle to a TTS model running on its dedicated thread.
///
/// Cloneable metadata (audio format, voices) is queried once at load time,
/// before the model is moved to its thread, so it can be read without
/// blocking on the generation queue.
pub struct TtsHandle {
    tx: mpsc::UnboundedSender<TtsRequest>,
    audio_info: AudioInfo,
    voices: Vec<VoiceInfo>,
    supports_voice_cloning: bool,
    model_type_name: &'static str,
    pending_count: Arc<AtomicU64>,
}

impl TtsHandle {
    /// Returns the audio format this model produces.
    #[must_use]
    pub fn audio_info(&self) -> AudioInfo {
        self.audio_info
    }

    /// Returns the voices available for service discovery.
    #[must_use]
    pub fn voices(&self) -> &[VoiceInfo] {
        &self.voices
    }

    /// Returns true if this model supports voice cloning from reference audio.
    #[must_use]
    pub fn supports_voice_cloning(&self) -> bool {
        self.supports_voice_cloning
    }

    /// Returns the model type name (e.g. "`qwen3_tts`", "`voxtral_tts`").
    #[must_use]
    pub fn model_type_name(&self) -> &'static str {
        self.model_type_name
    }

    /// Returns the number of requests currently queued or being processed.
    #[must_use]
    pub fn pending_count(&self) -> u64 {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// Send a generation request to the model's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's thread has stopped.
    pub fn send(&self, req: TtsGenerateRequest) -> Result<()> {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.tx.send(TtsRequest::Generate(req)).map_err(|_| {
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            anyhow::anyhow!("TTS thread has stopped")
        })
    }

    /// Send a streaming generation request to the model's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's thread has stopped.
    fn send_stream(&self, req: TtsStreamRequest) -> Result<()> {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.tx.send(TtsRequest::Stream(req)).map_err(|_| {
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            anyhow::anyhow!("TTS thread has stopped")
        })
    }
}

/// Protocol-independent TTS model runtime.
///
/// Owns every loaded TTS model, keyed by name. crane-wyoming builds one
/// `ModelRuntime` at startup and shares it via `Arc`.
pub struct ModelRuntime {
    tts: HashMap<String, TtsHandle>,
    default_tts: Option<String>,
    /// Whether callers should use
    /// [`generate_speech_stream`](Self::generate_speech_stream) for
    /// incremental TTS delivery. Defaults to `true`; see
    /// [`set_streaming_enabled`](Self::set_streaming_enabled).
    streaming_enabled: bool,
}

impl Default for ModelRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRuntime {
    /// Create an empty runtime with no models loaded.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tts: HashMap::new(),
            default_tts: None,
            streaming_enabled: true,
        }
    }

    /// Enable or disable incremental TTS streaming for this runtime.
    ///
    /// Transports whose hardware can't keep up with real-time incremental
    /// generation (e.g. crane-wyoming on a CPU-only device, where chunked
    /// audio arrives slower than it plays back) should call this with
    /// `false` and fall back to the blob
    /// [`generate_speech`](Self::generate_speech) path instead.
    pub fn set_streaming_enabled(&mut self, enabled: bool) {
        self.streaming_enabled = enabled;
    }

    /// Returns whether incremental TTS streaming is enabled for this runtime.
    #[must_use]
    pub fn streaming_enabled(&self) -> bool {
        self.streaming_enabled
    }

    /// Register an already-constructed TTS model under `name`.
    ///
    /// Queries audio format, voices, and voice-cloning support from the
    /// model before moving it to a dedicated thread. This ordering lets
    /// tests inject a mock [`Tts`] implementation without touching disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's dedicated thread fails to spawn.
    pub fn register_tts(
        &mut self,
        name: String,
        model_type_name: &'static str,
        tts: Box<dyn Tts + Send>,
    ) -> Result<()> {
        let audio_info = tts.audio_info();
        let voices = tts.voices();
        let supports_voice_cloning = tts.supports_voice_cloning();
        let pending_count = Arc::new(AtomicU64::new(0));

        let (tx, rx) = mpsc::unbounded_channel::<TtsRequest>();

        let thread_name = format!("tts-{name}");
        let log_name = name.clone();
        let thread_pending_count = Arc::clone(&pending_count);
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || run_tts_thread(rx, tts, &log_name, &thread_pending_count))
            .map_err(|e| anyhow::anyhow!("Failed to spawn TTS thread: {e}"))?;

        if self.default_tts.is_none() {
            self.default_tts = Some(name.clone());
        }

        self.tts.insert(
            name,
            TtsHandle {
                tx,
                audio_info,
                voices,
                supports_voice_cloning,
                model_type_name,
                pending_count,
            },
        );
        Ok(())
    }

    /// Load a TTS model from disk and register it.
    ///
    /// Detects the model type from `model_path`, constructs the model, and
    /// registers it under a name derived from the path's final component.
    /// Returns the registration name.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to load.
    pub fn load_tts(&mut self, model_path: &str, device: &Device, dtype: &DType) -> Result<String> {
        let resolved_type = model_factory::resolve(ModelType::Auto, model_path);
        let tts = model_factory::create_tts(resolved_type, model_path, device, dtype)?;
        let name = extract_model_name(model_path);

        tracing::info!(
            name = %name,
            model_type = %resolved_type.display_name(),
            "TTS model loaded, spawning thread",
        );

        self.register_tts(name.clone(), resolved_type.display_name(), tts)?;
        Ok(name)
    }

    /// Returns the TTS handle registered under `name`, if any.
    #[must_use]
    pub fn tts_handle(&self, name: &str) -> Option<&TtsHandle> {
        self.tts.get(name)
    }

    /// Returns an arbitrary loaded TTS handle.
    ///
    /// Useful when only one TTS model is loaded (the common case for a
    /// Wyoming server started with a single `--model-path` flag).
    #[must_use]
    pub fn default_tts_handle(&self) -> Option<&TtsHandle> {
        self.default_tts
            .as_ref()
            .and_then(|name| self.tts.get(name))
    }

    /// Returns the registration name of the default TTS model, if any.
    #[must_use]
    pub fn default_tts_name(&self) -> Option<&str> {
        self.default_tts.as_deref()
    }

    /// Returns an iterator over all registered TTS model names and their handles.
    ///
    /// Iteration order is unspecified. Callers that need a deterministic
    /// ordering (e.g. for voice name conflict resolution) should collect
    /// and sort, or track registration order separately.
    pub fn tts_handles(&self) -> impl Iterator<Item = (&str, &TtsHandle)> {
        self.tts
            .iter()
            .map(|(name, handle)| (name.as_str(), handle))
    }

    /// Dispatch a TTS request directly to the model's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if `model_name` is not registered or the model's
    /// thread has stopped.
    pub fn generate_speech(&self, model_name: &str, req: TtsGenerateRequest) -> Result<()> {
        let handle = self
            .tts_handle(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown TTS model: {model_name}"))?;
        handle.send(req)
    }

    /// Dispatch a TTS request for incremental generation.
    ///
    /// Unlike [`generate_speech`](Self::generate_speech), the returned
    /// receiver yields audio chunks as the model produces them (see
    /// [`Tts::generate_speech_stream`]) instead of waiting for the complete
    /// waveform. The channel closes once generation finishes; an `Err` item
    /// signals generation failed and ends the stream.
    ///
    /// # Backpressure
    ///
    /// The returned receiver is bounded and the model's worker thread sends
    /// on it with a blocking call. Keep draining it (or drop it) promptly --
    /// a stalled consumer blocks the worker thread and, with it, every other
    /// request queued for the same model.
    ///
    /// # Errors
    ///
    /// Returns an error if `model_name` is not registered or the model's
    /// thread has stopped.
    pub fn generate_speech_stream(
        &self,
        model_name: &str,
        text: String,
        language: String,
        voice: Option<String>,
        opts: SpeechOptions,
    ) -> Result<mpsc::Receiver<Result<Tensor>>> {
        let handle = self
            .tts_handle(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown TTS model: {model_name}"))?;

        // Capacity 2: lets the worker produce one chunk ahead of what the
        // consumer is processing, without buffering the whole waveform.
        let (chunk_tx, chunk_rx) = mpsc::channel(2);
        handle.send_stream(TtsStreamRequest {
            text,
            language,
            voice,
            opts,
            chunk_tx,
        })?;
        Ok(chunk_rx)
    }
}

/// Derive a registration name from a model path's final path component.
fn extract_model_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .map_or_else(|| "tts".to_string(), |n| n.to_string_lossy().into_owned())
}

/// Run the blocking generation loop for one TTS model on its dedicated thread.
///
/// Processes both blob ([`TtsGenerateRequest`]) and incremental
/// ([`TtsStreamRequest`]) requests from a single FIFO queue, so streaming and
/// non-streaming requests for the same model never run concurrently.
fn run_tts_thread(
    mut rx: mpsc::UnboundedReceiver<TtsRequest>,
    mut tts: Box<dyn Tts + Send>,
    model_name: &str,
    pending_count: &AtomicU64,
) {
    tracing::info!(model = %model_name, "TTS thread started");
    while let Some(req) = rx.blocking_recv() {
        match req {
            TtsRequest::Generate(req) => handle_generate_request(&mut *tts, req, model_name),
            TtsRequest::Stream(req) => handle_stream_request(&mut *tts, &req, model_name),
        }
        pending_count.fetch_sub(1, Ordering::Relaxed);
    }
    tracing::info!(model = %model_name, "TTS thread stopped (channel closed)");
}

/// Extract a human-readable message from a caught panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Handle one blob request: generate the complete waveform and send it back
/// via `req.response_tx`.
fn handle_generate_request(tts: &mut dyn Tts, req: TtsGenerateRequest, model_name: &str) {
    let text_len = req.text.chars().count();

    if req.response_tx.is_closed() {
        tracing::warn!(model = %model_name, text_len, "Caller disconnected, skipping");
        return;
    }

    let t0 = std::time::Instant::now();

    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if let Some(ref ref_audio) = req.reference_audio {
            let ref_text = req.reference_text.as_deref().unwrap_or("");
            tts.generate_voice_clone(&req.text, &req.language, ref_audio, ref_text, &req.opts)
        } else {
            tts.generate_speech(&req.text, &req.language, req.voice.as_deref(), &req.opts)
        }
    }));

    let result = match panic_result {
        Ok(result) => result,
        Err(panic_payload) => {
            let msg = panic_message(&*panic_payload);
            tracing::error!(model = %model_name, text_len, panic = %msg, "TTS generation panicked");
            Err(anyhow::anyhow!("TTS generation panicked: {msg}"))
        },
    };

    let elapsed_ms = t0.elapsed().as_millis();
    match &result {
        Ok(tensor) => {
            tracing::info!(
                model = %model_name,
                text_len,
                samples = tensor.elem_count(),
                elapsed_ms,
                "TTS generation complete",
            );
        },
        Err(e) => {
            tracing::error!(model = %model_name, text_len, elapsed_ms, error = %e, "TTS generation failed");
        },
    }

    let _ = req.response_tx.send(result);
}

/// Handle one streaming request: generate the waveform incrementally,
/// sending each chunk over `req.chunk_tx` as it's produced.
///
/// Stops early (without treating it as an error) if the receiver is
/// dropped, i.e. the caller disconnected mid-stream.
fn handle_stream_request(tts: &mut dyn Tts, req: &TtsStreamRequest, model_name: &str) {
    let text_len = req.text.chars().count();

    if req.chunk_tx.is_closed() {
        tracing::warn!(model = %model_name, text_len, "Caller disconnected, skipping");
        return;
    }

    let t0 = std::time::Instant::now();
    let mut chunk_count = 0usize;

    // `Ok(true)` means the caller disconnected mid-stream (not an error);
    // `Ok(false)` means the stream ran to completion.
    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<bool> {
        let mut stream =
            tts.generate_speech_stream(&req.text, &req.language, req.voice.as_deref(), &req.opts)?;
        while let Some(chunk) = stream.next_chunk()? {
            chunk_count += 1;
            if req.chunk_tx.blocking_send(Ok(chunk)).is_err() {
                return Ok(true);
            }
        }
        Ok(false)
    }));

    let result = match panic_result {
        Ok(result) => result,
        Err(panic_payload) => {
            let msg = panic_message(&*panic_payload);
            tracing::error!(model = %model_name, text_len, panic = %msg, "TTS streaming panicked");
            Err(anyhow::anyhow!("TTS generation panicked: {msg}"))
        },
    };

    let elapsed_ms = t0.elapsed().as_millis();
    match result {
        Ok(true) => {
            tracing::warn!(model = %model_name, text_len, chunk_count, elapsed_ms, "Caller disconnected mid-stream");
        },
        Ok(false) => {
            tracing::info!(model = %model_name, text_len, chunk_count, elapsed_ms, "TTS streaming complete");
        },
        Err(e) => {
            tracing::error!(model = %model_name, text_len, chunk_count, elapsed_ms, error = %e, "TTS streaming failed");
            let _ = req.chunk_tx.blocking_send(Err(e));
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    struct MockTts {
        audio_info: AudioInfo,
        voices: Vec<VoiceInfo>,
        supports_cloning: bool,
    }

    impl MockTts {
        fn new() -> Self {
            Self {
                audio_info: AudioInfo {
                    sample_rate: 24000,
                    channels: 1,
                    bits_per_sample: 16,
                },
                voices: vec![
                    VoiceInfo {
                        name: "alice".into(),
                        languages: vec!["en".into()],
                    },
                    VoiceInfo {
                        name: "bob".into(),
                        languages: vec!["en".into(), "fr".into()],
                    },
                ],
                supports_cloning: false,
            }
        }

        fn with_cloning(mut self) -> Self {
            self.supports_cloning = true;
            self
        }

        fn with_sample_rate(mut self, sample_rate: u32) -> Self {
            self.audio_info.sample_rate = sample_rate;
            self
        }
    }

    impl Tts for MockTts {
        fn audio_info(&self) -> AudioInfo {
            self.audio_info
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            self.voices.clone()
        }

        fn supports_voice_cloning(&self) -> bool {
            self.supports_cloning
        }

        fn generate_speech(
            &mut self,
            text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            let n = text.chars().count().max(1);
            Tensor::new(vec![0.5f32; n], &Device::Cpu).map_err(Into::into)
        }

        fn generate_voice_clone(
            &mut self,
            text: &str,
            _language: &str,
            _ref_audio: &str,
            _ref_text: &str,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            let n = text.chars().count().max(1);
            Tensor::new(vec![-0.5f32; n], &Device::Cpu).map_err(Into::into)
        }
    }

    #[test]
    fn test_register_tts_stores_handle() {
        let mut rt = ModelRuntime::new();
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();

        let handle = rt.tts_handle("m1").expect("handle should be registered");
        assert_eq!(handle.audio_info().sample_rate, 24000);
        assert_eq!(handle.voices().len(), 2);
        assert!(!handle.supports_voice_cloning());
        assert_eq!(handle.model_type_name(), "qwen3_tts");
    }

    #[test]
    fn test_generate_speech_roundtrip() {
        let mut rt = ModelRuntime::new();
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();
        let handle = rt.tts_handle("m1").unwrap();

        let (tx, rx) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx,
            })
            .unwrap();

        let tensor = rx.blocking_recv().unwrap().unwrap();
        let samples: Vec<f32> = tensor.to_vec1().unwrap();
        assert_eq!(samples, vec![0.5f32; 5]);
    }

    #[test]
    fn test_generate_voice_clone_roundtrip() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new().with_cloning()),
        )
        .unwrap();
        let handle = rt.tts_handle("m1").unwrap();

        let (tx, rx) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "hi".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: Some("/ref.wav".into()),
                reference_text: Some("hi there".into()),
                response_tx: tx,
            })
            .unwrap();

        let tensor = rx.blocking_recv().unwrap().unwrap();
        let samples: Vec<f32> = tensor.to_vec1().unwrap();
        assert_eq!(samples, vec![-0.5f32; 2]);
    }

    #[test]
    fn test_multiple_tts_models() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "a".into(),
            "qwen3_tts",
            Box::new(MockTts::new().with_sample_rate(24000)),
        )
        .unwrap();
        rt.register_tts(
            "b".into(),
            "voxtral_tts",
            Box::new(MockTts::new().with_sample_rate(16000)),
        )
        .unwrap();

        assert_eq!(rt.tts_handle("a").unwrap().audio_info().sample_rate, 24000);
        assert_eq!(rt.tts_handle("b").unwrap().audio_info().sample_rate, 16000);
    }

    #[test]
    fn test_default_tts_handle() {
        let mut rt = ModelRuntime::new();
        assert!(rt.default_tts_handle().is_none());

        rt.register_tts("a".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();
        assert!(rt.default_tts_handle().is_some());
    }

    #[test]
    fn test_tts_handle_not_found() {
        let rt = ModelRuntime::new();
        assert!(rt.tts_handle("nonexistent").is_none());
    }

    #[test]
    fn test_model_runtime_new_metadata() {
        let rt = ModelRuntime::new();
        assert!(rt.default_tts_handle().is_none());
        assert!(rt.default_tts_name().is_none());
        assert!(rt.streaming_enabled());
    }

    #[test]
    fn test_streaming_enabled_roundtrip() {
        let mut rt = ModelRuntime::new();
        assert!(rt.streaming_enabled());
        rt.set_streaming_enabled(false);
        assert!(!rt.streaming_enabled());
        rt.set_streaming_enabled(true);
        assert!(rt.streaming_enabled());
    }

    #[test]
    fn test_default_tts_name() {
        let mut rt = ModelRuntime::new();
        assert!(rt.default_tts_name().is_none());
        rt.register_tts("m1".into(), "qwen3_tts", Box::new(MockTts::new()))
            .unwrap();
        assert_eq!(rt.default_tts_name(), Some("m1"));
    }

    #[test]
    fn test_extract_model_name() {
        assert_eq!(
            extract_model_name("/models/Qwen3-TTS-12Hz-0.6B"),
            "Qwen3-TTS-12Hz-0.6B"
        );
        assert_eq!(extract_model_name("/models/voxtral/"), "voxtral");
        assert_eq!(extract_model_name(""), "tts");
    }

    #[test]
    fn test_send_after_thread_stopped() {
        let (tx, rx) = mpsc::unbounded_channel::<TtsRequest>();
        drop(rx);

        let handle = TtsHandle {
            tx,
            audio_info: AudioInfo {
                sample_rate: 24000,
                channels: 1,
                bits_per_sample: 16,
            },
            voices: vec![],
            supports_voice_cloning: false,
            model_type_name: "qwen3_tts",
            pending_count: Arc::new(AtomicU64::new(0)),
        };

        let (resp_tx, _resp_rx) = oneshot::channel();
        let result = handle.send(TtsGenerateRequest {
            text: "hello".into(),
            language: "en".into(),
            voice: None,
            opts: SpeechOptions::default(),
            reference_audio: None,
            reference_text: None,
            response_tx: resp_tx,
        });

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("stopped"));
        assert_eq!(handle.pending_count(), 0);
    }

    #[test]
    fn test_register_duplicate_name() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "dup".into(),
            "qwen3_tts",
            Box::new(MockTts::new().with_sample_rate(24000)),
        )
        .unwrap();
        rt.register_tts(
            "dup".into(),
            "voxtral_tts",
            Box::new(MockTts::new().with_sample_rate(16000)),
        )
        .unwrap();

        let handle = rt.tts_handle("dup").unwrap();
        assert_eq!(handle.audio_info().sample_rate, 16000);
        assert_eq!(handle.model_type_name(), "voxtral_tts");
    }

    struct PanickingTts {
        call_count: u32,
    }

    impl PanickingTts {
        fn new() -> Self {
            Self { call_count: 0 }
        }
    }

    impl Tts for PanickingTts {
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
            text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            self.call_count += 1;
            assert!(self.call_count != 1, "simulated model panic");
            let n = text.chars().count().max(1);
            Tensor::new(vec![0.1f32; n], &Device::Cpu).map_err(Into::into)
        }
    }

    #[test]
    fn test_panic_in_generate_is_caught() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "panic_model".into(),
            "qwen3_tts",
            Box::new(PanickingTts::new()),
        )
        .unwrap();
        let handle = rt.tts_handle("panic_model").unwrap();

        let (tx1, rx1) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "boom".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx1,
            })
            .unwrap();

        let result1 = rx1.blocking_recv().unwrap();
        assert!(result1.is_err());
        assert!(result1.unwrap_err().to_string().contains("panicked"));

        let (tx2, rx2) = oneshot::channel();
        handle
            .send(TtsGenerateRequest {
                text: "ok".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx2,
            })
            .unwrap();

        let result2 = rx2.blocking_recv().unwrap();
        assert!(result2.is_ok());
        let samples: Vec<f32> = result2.unwrap().to_vec1().unwrap();
        assert_eq!(samples, vec![0.1f32; 2]);
    }

    #[test]
    fn test_generate_speech_unknown_model() {
        let rt = ModelRuntime::new();
        let (tx, _rx) = oneshot::channel();
        let result = rt.generate_speech(
            "nonexistent",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx,
            },
        );
        assert!(result.is_err());
    }

    struct StreamingMockTts {
        audio_info: AudioInfo,
        chunks: Vec<f32>,
    }

    impl StreamingMockTts {
        fn new(chunks: Vec<f32>) -> Self {
            Self {
                audio_info: AudioInfo {
                    sample_rate: 24000,
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
            vec![]
        }

        fn generate_speech(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            Tensor::new(self.chunks.clone(), &Device::Cpu).map_err(Into::into)
        }

        fn generate_speech_stream(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<crane::audio::tts::TtsStream<'_>> {
            let chunks: Vec<Result<Tensor>> = self
                .chunks
                .iter()
                .map(|&v| Tensor::new(vec![v], &Device::Cpu).map_err(Into::into))
                .collect();
            Ok(crane::audio::tts::TtsStream::new(
                self.audio_info,
                chunks.into_iter(),
            ))
        }
    }

    #[test]
    fn test_generate_speech_stream_multiple_chunks() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(StreamingMockTts::new(vec![0.1, 0.2, 0.3])),
        )
        .unwrap();

        let mut rx = rt
            .generate_speech_stream(
                "m1",
                "hello".into(),
                "en".into(),
                None,
                SpeechOptions::default(),
            )
            .unwrap();

        let mut received = Vec::new();
        while let Some(result) = rx.blocking_recv() {
            let tensor = result.unwrap();
            received.push(tensor.to_vec1::<f32>().unwrap()[0]);
        }
        assert_eq!(received, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_generate_speech_stream_unknown_model() {
        let rt = ModelRuntime::new();
        let result = rt.generate_speech_stream(
            "nonexistent",
            "hello".into(),
            "en".into(),
            None,
            SpeechOptions::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_speech_stream_empty() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "empty".into(),
            "qwen3_tts",
            Box::new(StreamingMockTts::new(vec![])),
        )
        .unwrap();

        let mut rx = rt
            .generate_speech_stream(
                "empty",
                "hello".into(),
                "en".into(),
                None,
                SpeechOptions::default(),
            )
            .unwrap();
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn test_generate_speech_stream_receiver_dropped() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(StreamingMockTts::new(vec![0.1, 0.2, 0.3])),
        )
        .unwrap();

        let mut rx = rt
            .generate_speech_stream(
                "m1",
                "hello".into(),
                "en".into(),
                None,
                SpeechOptions::default(),
            )
            .unwrap();
        rx.blocking_recv().unwrap().unwrap();
        drop(rx);

        // The worker thread must notice the dropped receiver and move on --
        // this blob request would hang forever if it didn't.
        let (tx, rx2) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "still alive".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx,
            },
        )
        .unwrap();

        assert!(rx2.blocking_recv().unwrap().is_ok());
    }

    struct PanicOnSecondChunk {
        index: usize,
    }

    impl Iterator for PanicOnSecondChunk {
        type Item = Result<Tensor>;

        fn next(&mut self) -> Option<Self::Item> {
            self.index += 1;
            match self.index {
                1 => Some(Tensor::new(vec![0.1f32], &Device::Cpu).map_err(Into::into)),
                2 => panic!("simulated stream panic"),
                _ => None,
            }
        }
    }

    struct StreamPanickingTts {
        call_count: u32,
    }

    impl StreamPanickingTts {
        fn new() -> Self {
            Self { call_count: 0 }
        }
    }

    impl Tts for StreamPanickingTts {
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
        ) -> Result<Tensor> {
            Tensor::new(vec![0.1f32], &Device::Cpu).map_err(Into::into)
        }

        fn generate_speech_stream(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<crane::audio::tts::TtsStream<'_>> {
            self.call_count += 1;
            let audio_info = self.audio_info();
            if self.call_count == 1 {
                Ok(crane::audio::tts::TtsStream::new(
                    audio_info,
                    PanicOnSecondChunk { index: 0 },
                ))
            } else {
                let chunks: Vec<Result<Tensor>> = vec![0.5f32, 0.6f32]
                    .into_iter()
                    .map(|v| Tensor::new(vec![v], &Device::Cpu).map_err(Into::into))
                    .collect();
                Ok(crane::audio::tts::TtsStream::new(
                    audio_info,
                    chunks.into_iter(),
                ))
            }
        }
    }

    #[test]
    fn test_generate_speech_stream_panic_recovery() {
        let mut rt = ModelRuntime::new();
        rt.register_tts(
            "panic_stream".into(),
            "qwen3_tts",
            Box::new(StreamPanickingTts::new()),
        )
        .unwrap();

        let mut rx = rt
            .generate_speech_stream(
                "panic_stream",
                "boom".into(),
                "en".into(),
                None,
                SpeechOptions::default(),
            )
            .unwrap();

        let first = rx.blocking_recv().unwrap();
        assert!(first.is_ok());

        let second = rx.blocking_recv().unwrap();
        assert!(second.is_err());
        assert!(second.unwrap_err().to_string().contains("panicked"));

        assert!(rx.blocking_recv().is_none());

        // The thread must have recovered: the next stream request succeeds.
        let mut rx2 = rt
            .generate_speech_stream(
                "panic_stream",
                "ok".into(),
                "en".into(),
                None,
                SpeechOptions::default(),
            )
            .unwrap();

        let mut received = Vec::new();
        while let Some(result) = rx2.blocking_recv() {
            received.push(result.unwrap().to_vec1::<f32>().unwrap()[0]);
        }
        assert_eq!(received, vec![0.5, 0.6]);
    }
}
