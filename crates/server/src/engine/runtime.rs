// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>
//
// Based on the `engine` module of Crane's `crane-serve` crate
// (https://github.com/lucasjinreal/Crane), Copyright (c) 2024 Nicholas Jela,
// licensed under the MIT License.

//! Protocol-independent TTS/ASR model runtime.
//!
//! [`ModelRuntime`] owns every loaded TTS and ASR model, keyed by
//! registration name. Consumers load TTS models via
//! [`ModelRuntime::load_tts`], then send generation requests through
//! [`ModelRuntime::generate_speech`] / [`ModelRuntime::generate_speech_stream`];
//! ASR models are loaded via [`ModelRuntime::load_asr`] and dispatched
//! through [`ModelRuntime::transcribe`]. Each model runs on its own
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

use crane::audio::AudioInfo;
use crane::audio::tts::{Tts, VoiceInfo};
use crane::audio::{Asr, TranscribeOptions, Transcript};
use crane_core::generation::SpeechOptions;

use crate::engine::cache::{CacheKey, TtsCache};
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
    /// Identity used in TTS cache keys. Defaults to the registration name,
    /// but [`ModelRuntime::load_tts`] overrides it to the full model path so
    /// two directories that merely share a final path component (different
    /// checkpoint, dtype, or quantization) don't collide in the cache.
    cache_model_id: String,
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

/// A request to transcribe audio, sent to an ASR model's dedicated thread.
///
/// Transport-agnostic like [`TtsGenerateRequest`]: it carries no Wyoming- or
/// HTTP-specific fields. `audio` must already be mono f32 PCM at the
/// model's [`AsrHandle::input_sample_rate`]; converting from wire PCM bytes
/// is the caller's responsibility.
pub struct AsrTranscribeRequest {
    /// Mono f32 PCM audio at the model's expected sample rate.
    pub audio: Vec<f32>,
    /// Language hint (e.g. "en", "zh"), or `None` to let the model
    /// auto-detect.
    pub language: Option<String>,
    /// Channel to send back the transcription result.
    pub response_tx: oneshot::Sender<Result<Transcript>>,
}

/// A request to transcribe audio incrementally, sent to an ASR model's
/// dedicated thread.
///
/// Unlike [`AsrTranscribeRequest`], the response is delivered as a series of
/// [`Transcript`] chunks over `chunk_tx` rather than a single result -- one
/// send per chunk yielded by [`Asr::transcribe_stream`], followed by the
/// sender being dropped to signal completion. An `Err` chunk ends the
/// stream immediately.
///
/// # Backpressure
///
/// `chunk_tx` is bounded, and the model's worker thread sends on it with a
/// blocking call. A consumer that stops draining the matching receiver
/// without dropping it will stall the worker thread indefinitely -- which
/// also blocks every other request (streaming or batch) queued for the same
/// model, since they share one thread. Callers must keep draining or drop
/// the receiver promptly.
pub(crate) struct AsrStreamRequest {
    /// Mono f32 PCM audio at the model's expected sample rate.
    pub(crate) audio: Vec<f32>,
    /// Language hint (e.g. "en", "zh"), or `None` to let the model
    /// auto-detect.
    pub(crate) language: Option<String>,
    /// Channel to send transcript chunks back as they're produced.
    pub(crate) chunk_tx: mpsc::Sender<Result<Transcript>>,
}

/// A request queued on an ASR model's dedicated thread: either a batch
/// [`AsrTranscribeRequest`] or an incremental [`AsrStreamRequest`].
///
/// Both variants share one channel/thread so streaming and non-streaming
/// requests for the same model are processed in a single FIFO queue,
/// matching the "one request at a time" concurrency model documented on
/// [`AsrHandle`].
enum AsrRequest {
    /// Transcribe the complete audio and return a single result.
    Transcribe(AsrTranscribeRequest),
    /// Transcribe the audio incrementally, streaming transcript chunks.
    Stream(AsrStreamRequest),
}

/// Handle to an ASR model running on its dedicated thread.
///
/// The input sample rate is queried once at load time, before the model is
/// moved to its thread, so it can be read without blocking on the
/// transcription queue.
pub struct AsrHandle {
    tx: mpsc::UnboundedSender<AsrRequest>,
    input_sample_rate: u32,
    model_type_name: &'static str,
    pending_count: Arc<AtomicU64>,
}

impl AsrHandle {
    /// Returns the sample rate this model expects input audio at (e.g. 16000).
    #[must_use]
    pub fn input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }

    /// Returns the model type name (e.g. "`qwen3_asr`").
    #[must_use]
    pub fn model_type_name(&self) -> &'static str {
        self.model_type_name
    }

    /// Returns the number of requests currently queued or being processed.
    #[must_use]
    pub fn pending_count(&self) -> u64 {
        self.pending_count.load(Ordering::Relaxed)
    }

    /// Send a transcription request to the model's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's thread has stopped.
    pub fn send(&self, req: AsrTranscribeRequest) -> Result<()> {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.tx.send(AsrRequest::Transcribe(req)).map_err(|_| {
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            anyhow::anyhow!("ASR thread has stopped")
        })
    }

    /// Send a streaming transcription request to the model's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's thread has stopped.
    fn send_stream(&self, req: AsrStreamRequest) -> Result<()> {
        self.pending_count.fetch_add(1, Ordering::Relaxed);
        self.tx.send(AsrRequest::Stream(req)).map_err(|_| {
            self.pending_count.fetch_sub(1, Ordering::Relaxed);
            anyhow::anyhow!("ASR thread has stopped")
        })
    }
}

/// Protocol-independent TTS/ASR model runtime.
///
/// Owns every loaded TTS and ASR model, keyed by name. crane-wyoming builds
/// one `ModelRuntime` at startup and shares it via `Arc`.
pub struct ModelRuntime {
    tts: HashMap<String, TtsHandle>,
    default_tts: Option<String>,
    tts_cache: Option<Arc<TtsCache>>,
    /// Whether callers should use
    /// [`generate_speech_stream`](Self::generate_speech_stream) for
    /// incremental TTS delivery. Defaults to `true`; see
    /// [`set_streaming_enabled`](Self::set_streaming_enabled).
    streaming_enabled: bool,
    asr: HashMap<String, AsrHandle>,
    default_asr: Option<String>,
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
            tts_cache: None,
            streaming_enabled: true,
            asr: HashMap::new(),
            default_asr: None,
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

    /// Enable disk caching for TTS responses.
    ///
    /// Once set, [`generate_speech`](Self::generate_speech) checks the
    /// cache before dispatching to a model's thread and returns a cached
    /// waveform on a hit. Disabled by default. Requests with
    /// `reference_audio` set (voice cloning) always bypass the cache.
    pub fn set_tts_cache(&mut self, cache: TtsCache) {
        self.tts_cache = Some(Arc::new(cache));
    }

    /// Register an already-constructed TTS model under `name`.
    ///
    /// Queries audio format, voices, and voice-cloning support from the
    /// model before moving it to a dedicated thread. This ordering lets
    /// tests inject a mock [`Tts`] implementation without touching disk.
    ///
    /// `device` is the device `tts` was constructed on; the dedicated
    /// thread runs each request's generation inside
    /// [`Device::with_context`](candle_core::Device::with_context) so CPU
    /// inference uses candle's warm, affinity-pinned rayon pool instead of
    /// rayon's ambient global pool. Pass `&Device::Cpu` for mock models in
    /// tests -- it's a cheap no-op wrapper either way.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's dedicated thread fails to spawn.
    pub fn register_tts(
        &mut self,
        name: String,
        model_type_name: &'static str,
        tts: Box<dyn Tts + Send>,
        device: &Device,
    ) -> Result<()> {
        let audio_info = tts.audio_info();
        let voices = tts.voices();
        let supports_voice_cloning = tts.supports_voice_cloning();
        let pending_count = Arc::new(AtomicU64::new(0));

        let (tx, rx) = mpsc::unbounded_channel::<TtsRequest>();

        let thread_name = format!("tts-{name}");
        let log_name = name.clone();
        let thread_pending_count = Arc::clone(&pending_count);
        let thread_device = device.clone();
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                run_tts_thread(rx, tts, &log_name, &thread_pending_count, &thread_device);
            })
            .map_err(|e| anyhow::anyhow!("Failed to spawn TTS thread: {e}"))?;

        if self.default_tts.is_none() {
            self.default_tts = Some(name.clone());
        }

        self.tts.insert(
            name.clone(),
            TtsHandle {
                tx,
                audio_info,
                voices,
                supports_voice_cloning,
                model_type_name,
                pending_count,
                cache_model_id: name,
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
        let resolved_type = model_factory::resolve_tts(ModelType::Auto, model_path)?;
        let tts = model_factory::create_tts(resolved_type, model_path, device, dtype)?;
        let name = extract_model_name(model_path);

        tracing::info!(
            name = %name,
            model_type = %resolved_type.display_name(),
            "TTS model loaded, spawning thread",
        );

        self.register_tts(name.clone(), resolved_type.display_name(), tts, device)?;
        // Cache keys should discriminate by the full on-disk path, not just
        // its final component -- two directories with the same file name
        // (different checkpoint, dtype, or quantization) must not collide.
        if let Some(handle) = self.tts.get_mut(&name) {
            handle.cache_model_id = model_path.to_string();
        }
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

    /// Dispatch a TTS request, consulting the cache first if one is configured.
    ///
    /// This is the preferred entry point for TTS generation over calling
    /// [`TtsHandle::send`] directly:
    ///
    /// 1. If no cache is configured, delegates straight to `TtsHandle::send`.
    /// 2. If `req.reference_audio` is set (voice cloning), delegates
    ///    straight to `TtsHandle::send` -- voice-cloned audio is never
    ///    cached (see [`crate::engine::cache::TtsCache`]).
    /// 3. On a cache hit, sends the cached tensor back immediately without
    ///    running inference.
    /// 4. On a cache miss, tees the model's response through the cache: the
    ///    result is forwarded to the caller and written to disk on a
    ///    spawned task so the write never blocks generation.
    ///
    /// Concurrent requests for the same not-yet-cached text each miss and
    /// run inference independently -- there is no in-flight de-duplication.
    /// Atomic writes (see [`crate::engine::cache::TtsCache::put`]) keep this safe,
    /// just not maximally efficient under that access pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if `model_name` is not registered or the model's
    /// thread has stopped.
    pub fn generate_speech(&self, model_name: &str, req: TtsGenerateRequest) -> Result<()> {
        let handle = self
            .tts_handle(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown TTS model: {model_name}"))?;

        let Some(cache) = &self.tts_cache else {
            return handle.send(req);
        };
        if req.reference_audio.is_some() {
            return handle.send(req);
        }

        let digest = CacheKey::from_request(&handle.cache_model_id, &req).digest();
        if let Some(audio) = cache.get(&digest) {
            tracing::debug!(model = %model_name, "TTS cache hit");
            let _ = req.response_tx.send(Ok(audio));
            return Ok(());
        }

        let (tx, rx) = oneshot::channel();
        let original_tx = req.response_tx;
        let cache = Arc::clone(cache);
        tokio::spawn(async move {
            if let Ok(result) = rx.await {
                if let Ok(ref audio) = result
                    && let Err(e) = cache.put(&digest, audio)
                {
                    tracing::warn!("Failed to write TTS cache entry: {e}");
                }
                let _ = original_tx.send(result);
            }
        });
        handle.send(TtsGenerateRequest {
            response_tx: tx,
            ..req
        })
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

    /// Register an already-constructed ASR model under `name`.
    ///
    /// Queries the input sample rate from the model before moving it to a
    /// dedicated thread. This ordering lets tests inject a mock [`Asr`]
    /// implementation without touching disk.
    ///
    /// See [`register_tts`](Self::register_tts) for why the dedicated
    /// thread runs each request inside
    /// [`Device::with_context`](candle_core::Device::with_context).
    ///
    /// # Errors
    ///
    /// Returns an error if the model's dedicated thread fails to spawn.
    pub fn register_asr(
        &mut self,
        name: String,
        model_type_name: &'static str,
        asr: Box<dyn Asr + Send>,
        device: &Device,
    ) -> Result<()> {
        let input_sample_rate = asr.input_sample_rate();
        let pending_count = Arc::new(AtomicU64::new(0));

        let (tx, rx) = mpsc::unbounded_channel::<AsrRequest>();

        let thread_name = format!("asr-{name}");
        let log_name = name.clone();
        let thread_pending_count = Arc::clone(&pending_count);
        let thread_device = device.clone();
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                run_asr_thread(rx, asr, &log_name, &thread_pending_count, &thread_device);
            })
            .map_err(|e| anyhow::anyhow!("Failed to spawn ASR thread: {e}"))?;

        if self.default_asr.is_none() {
            self.default_asr = Some(name.clone());
        }

        self.asr.insert(
            name,
            AsrHandle {
                tx,
                input_sample_rate,
                model_type_name,
                pending_count,
            },
        );
        Ok(())
    }

    /// Load an ASR model from disk and register it.
    ///
    /// Detects the model type from `model_path`, constructs the model, and
    /// registers it under a name derived from the path's final component.
    /// Returns the registration name.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to load.
    pub fn load_asr(&mut self, model_path: &str, device: &Device, dtype: &DType) -> Result<String> {
        let resolved_type = model_factory::resolve_asr(ModelType::Auto, model_path)?;
        let asr = model_factory::create_asr(resolved_type, model_path, device, dtype)?;
        let name = extract_asr_model_name(model_path);

        tracing::info!(
            name = %name,
            model_type = %resolved_type.display_name(),
            "ASR model loaded, spawning thread",
        );

        self.register_asr(name.clone(), resolved_type.display_name(), asr, device)?;
        Ok(name)
    }

    /// Returns the ASR handle registered under `name`, if any.
    #[must_use]
    pub fn asr_handle(&self, name: &str) -> Option<&AsrHandle> {
        self.asr.get(name)
    }

    /// Returns an arbitrary loaded ASR handle.
    ///
    /// Useful when only one ASR model is loaded (the common case for a
    /// Wyoming server started with a single `--asr-model-path` flag).
    #[must_use]
    pub fn default_asr_handle(&self) -> Option<&AsrHandle> {
        self.default_asr
            .as_ref()
            .and_then(|name| self.asr.get(name))
    }

    /// Returns the registration name of the default ASR model, if any.
    #[must_use]
    pub fn default_asr_name(&self) -> Option<&str> {
        self.default_asr.as_deref()
    }

    /// Returns an iterator over all registered ASR model names and their handles.
    ///
    /// Iteration order is unspecified.
    pub fn asr_handles(&self) -> impl Iterator<Item = (&str, &AsrHandle)> {
        self.asr
            .iter()
            .map(|(name, handle)| (name.as_str(), handle))
    }

    /// Dispatch a transcription request to the named ASR model.
    ///
    /// Unlike [`generate_speech`](Self::generate_speech), there is no cache
    /// equivalent -- ASR input is always unique audio, so caching would
    /// never hit.
    ///
    /// # Errors
    ///
    /// Returns an error if `model_name` is not registered or the model's
    /// thread has stopped.
    pub fn transcribe(&self, model_name: &str, req: AsrTranscribeRequest) -> Result<()> {
        let handle = self
            .asr_handle(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown ASR model: {model_name}"))?;
        handle.send(req)
    }

    /// Dispatch a streaming transcription request to the named ASR model.
    ///
    /// The returned receiver is bounded and the model's worker thread sends
    /// on it with a blocking call -- see [`AsrStreamRequest`]'s backpressure
    /// docs. Keep draining it (or drop it) promptly.
    ///
    /// # Errors
    ///
    /// Returns an error if `model_name` is not registered or the model's
    /// thread has stopped.
    pub fn transcribe_stream(
        &self,
        model_name: &str,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> Result<mpsc::Receiver<Result<Transcript>>> {
        let handle = self
            .asr_handle(model_name)
            .ok_or_else(|| anyhow::anyhow!("unknown ASR model: {model_name}"))?;

        // Capacity 2: lets the worker produce one chunk ahead of what the
        // consumer is processing, matching generate_speech_stream.
        let (chunk_tx, chunk_rx) = mpsc::channel(2);
        handle.send_stream(AsrStreamRequest {
            audio,
            language,
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

/// Derive a registration name from an ASR model path's final path component.
fn extract_asr_model_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .map_or_else(|| "asr".to_string(), |n| n.to_string_lossy().into_owned())
}

/// Run the blocking generation loop for one TTS model on its dedicated thread.
///
/// Processes both blob ([`TtsGenerateRequest`]) and incremental
/// ([`TtsStreamRequest`]) requests from a single FIFO queue, so streaming and
/// non-streaming requests for the same model never run concurrently.
///
/// Each request is handled inside `device`'s
/// [`with_context`](candle_core::Device::with_context) so that, on CPU, its
/// forward pass's matmuls dispatch onto candle's warm, affinity-pinned rayon
/// pool instead of rayon's ambient global pool. It's a no-op on CUDA/Metal.
/// This is scoped per request rather than around the whole loop: `with_context`
/// installs candle's process-wide CPU pool by running the wrapped closure on
/// one of that pool's own worker threads, blocking the caller until it
/// returns -- wrapping the whole (otherwise idle, blocked-on-`recv`) loop
/// would permanently pin one pool worker per loaded CPU model for as long as
/// the model is loaded, starving other models' forward passes and, once the
/// number of loaded CPU models reaches the pool's worker count, deadlocking
/// any further model whose job can never be scheduled onto a worker.
fn run_tts_thread(
    mut rx: mpsc::UnboundedReceiver<TtsRequest>,
    mut tts: Box<dyn Tts + Send>,
    model_name: &str,
    pending_count: &AtomicU64,
    device: &Device,
) {
    tracing::info!(model = %model_name, "TTS thread started");
    while let Some(req) = rx.blocking_recv() {
        device.with_context(|| match req {
            TtsRequest::Generate(req) => handle_generate_request(&mut *tts, req, model_name),
            TtsRequest::Stream(req) => handle_stream_request(&mut *tts, &req, model_name),
        });
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

/// Run the blocking transcription loop for one ASR model on its dedicated
/// thread.
///
/// See [`run_tts_thread`] for why each request runs inside `device`'s
/// [`with_context`](candle_core::Device::with_context).
fn run_asr_thread(
    mut rx: mpsc::UnboundedReceiver<AsrRequest>,
    mut asr: Box<dyn Asr + Send>,
    model_name: &str,
    pending_count: &AtomicU64,
    device: &Device,
) {
    tracing::info!(model = %model_name, "ASR thread started");
    while let Some(req) = rx.blocking_recv() {
        device.with_context(|| match req {
            AsrRequest::Transcribe(req) => handle_transcribe_request(&mut *asr, req, model_name),
            AsrRequest::Stream(req) => handle_stream_transcribe_request(&mut *asr, req, model_name),
        });
        pending_count.fetch_sub(1, Ordering::Relaxed);
    }
    tracing::info!(model = %model_name, "ASR thread stopped (channel closed)");
}

/// Handle one transcription request: transcribe the complete audio and send
/// the result back via `req.response_tx`.
fn handle_transcribe_request(asr: &mut dyn Asr, req: AsrTranscribeRequest, model_name: &str) {
    let sample_count = req.audio.len();

    if req.response_tx.is_closed() {
        tracing::warn!(model = %model_name, sample_count, "Caller disconnected, skipping");
        return;
    }

    let t0 = std::time::Instant::now();
    let opts = TranscribeOptions {
        language: req.language,
        ..TranscribeOptions::default()
    };

    let panic_result =
        std::panic::catch_unwind(AssertUnwindSafe(|| asr.transcribe(&req.audio, &opts)));

    let result = match panic_result {
        Ok(result) => result,
        Err(panic_payload) => {
            let msg = panic_message(&*panic_payload);
            tracing::error!(model = %model_name, sample_count, panic = %msg, "ASR transcription panicked");
            Err(anyhow::anyhow!("ASR transcription panicked: {msg}"))
        },
    };

    let elapsed_ms = t0.elapsed().as_millis();
    match &result {
        Ok(transcript) => {
            tracing::info!(
                model = %model_name,
                sample_count,
                text_len = transcript.text.chars().count(),
                elapsed_ms,
                "ASR transcription complete",
            );
        },
        Err(e) => {
            tracing::error!(model = %model_name, sample_count, elapsed_ms, error = %e, "ASR transcription failed");
        },
    }

    let _ = req.response_tx.send(result);
}

/// Handle one streaming transcription request: transcribe the audio
/// incrementally, sending each chunk over `req.chunk_tx` as it's produced.
///
/// Stops early (without treating it as an error) if the receiver is
/// dropped, i.e. the caller disconnected mid-stream.
fn handle_stream_transcribe_request(asr: &mut dyn Asr, req: AsrStreamRequest, model_name: &str) {
    let sample_count = req.audio.len();

    if req.chunk_tx.is_closed() {
        tracing::warn!(model = %model_name, sample_count, "Caller disconnected, skipping");
        return;
    }

    let t0 = std::time::Instant::now();
    let mut chunk_count = 0usize;
    let opts = TranscribeOptions {
        language: req.language,
        ..TranscribeOptions::default()
    };

    // `Ok(true)` means the caller disconnected mid-stream (not an error);
    // `Ok(false)` means the stream ran to completion.
    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<bool> {
        let mut stream = asr.transcribe_stream(&req.audio, &opts)?;
        while let Some(transcript) = stream.next_chunk()? {
            chunk_count += 1;
            if req.chunk_tx.blocking_send(Ok(transcript)).is_err() {
                return Ok(true);
            }
        }
        Ok(false)
    }));

    let result = match panic_result {
        Ok(result) => result,
        Err(panic_payload) => {
            let msg = panic_message(&*panic_payload);
            tracing::error!(model = %model_name, sample_count, panic = %msg, "ASR streaming panicked");
            Err(anyhow::anyhow!("ASR transcription panicked: {msg}"))
        },
    };

    let elapsed_ms = t0.elapsed().as_millis();
    match result {
        Ok(true) => {
            tracing::warn!(model = %model_name, sample_count, chunk_count, elapsed_ms, "Caller disconnected mid-stream");
        },
        Ok(false) => {
            tracing::info!(model = %model_name, sample_count, chunk_count, elapsed_ms, "ASR streaming complete");
        },
        Err(e) => {
            tracing::error!(model = %model_name, sample_count, chunk_count, elapsed_ms, error = %e, "ASR streaming failed");
            let _ = req.chunk_tx.blocking_send(Err(e));
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::fs;

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
        register_test_tts(&mut rt, "m1", "qwen3_tts", Box::new(MockTts::new()));

        let handle = rt.tts_handle("m1").expect("handle should be registered");
        assert_eq!(handle.audio_info().sample_rate, 24000);
        assert_eq!(handle.voices().len(), 2);
        assert!(!handle.supports_voice_cloning());
        assert_eq!(handle.model_type_name(), "qwen3_tts");
    }

    #[test]
    fn test_generate_speech_roundtrip() {
        let mut rt = ModelRuntime::new();
        register_test_tts(&mut rt, "m1", "qwen3_tts", Box::new(MockTts::new()));
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
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new().with_cloning()),
        );
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
        register_test_tts(
            &mut rt,
            "a",
            "qwen3_tts",
            Box::new(MockTts::new().with_sample_rate(24000)),
        );
        register_test_tts(
            &mut rt,
            "b",
            "voxtral_tts",
            Box::new(MockTts::new().with_sample_rate(16000)),
        );

        assert_eq!(rt.tts_handle("a").unwrap().audio_info().sample_rate, 24000);
        assert_eq!(rt.tts_handle("b").unwrap().audio_info().sample_rate, 16000);
    }

    #[test]
    fn test_default_tts_handle() {
        let mut rt = ModelRuntime::new();
        assert!(rt.default_tts_handle().is_none());

        register_test_tts(&mut rt, "a", "qwen3_tts", Box::new(MockTts::new()));
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
        register_test_tts(&mut rt, "m1", "qwen3_tts", Box::new(MockTts::new()));
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
            cache_model_id: "test-model".into(),
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
        register_test_tts(
            &mut rt,
            "dup",
            "qwen3_tts",
            Box::new(MockTts::new().with_sample_rate(24000)),
        );
        register_test_tts(
            &mut rt,
            "dup",
            "voxtral_tts",
            Box::new(MockTts::new().with_sample_rate(16000)),
        );

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
        register_test_tts(
            &mut rt,
            "panic_model",
            "qwen3_tts",
            Box::new(PanickingTts::new()),
        );
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
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(StreamingMockTts::new(vec![0.1, 0.2, 0.3])),
        );

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
        register_test_tts(
            &mut rt,
            "empty",
            "qwen3_tts",
            Box::new(StreamingMockTts::new(vec![])),
        );

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
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(StreamingMockTts::new(vec![0.1, 0.2, 0.3])),
        );

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
        register_test_tts(
            &mut rt,
            "panic_stream",
            "qwen3_tts",
            Box::new(StreamPanickingTts::new()),
        );

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

    #[tokio::test]
    async fn test_generate_speech_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let mut rt = ModelRuntime::new();
        rt.set_tts_cache(TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap());
        register_test_tts(&mut rt, "m1", "qwen3_tts", Box::new(MockTts::new()));

        // First request: cache miss, generates and caches.
        let (tx1, rx1) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx1,
            },
        )
        .unwrap();
        let first = rx1.await.unwrap().unwrap();
        assert_eq!(first.to_vec1::<f32>().unwrap(), vec![0.5f32; 5]);

        // The tee task writes the cache entry before forwarding the
        // response, so by the time `rx1.await` resolves above the entry
        // is already on disk -- no extra synchronization needed here.

        // Second identical request: cache hit, same result without
        // depending on the (still-registered) model thread.
        let (tx2, rx2) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: None,
                reference_text: None,
                response_tx: tx2,
            },
        )
        .unwrap();
        let second = rx2.await.unwrap().unwrap();
        assert_eq!(second.to_vec1::<f32>().unwrap(), vec![0.5f32; 5]);
    }

    #[tokio::test]
    async fn test_generate_speech_reference_audio_bypasses_cache() {
        let dir = tempfile::tempdir().unwrap();
        let mut rt = ModelRuntime::new();
        rt.set_tts_cache(TtsCache::new(dir.path().to_path_buf(), 10_000_000).unwrap());
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new().with_cloning()),
        );

        let (tx, rx) = oneshot::channel();
        rt.generate_speech(
            "m1",
            TtsGenerateRequest {
                text: "hello".into(),
                language: "en".into(),
                voice: None,
                opts: SpeechOptions::default(),
                reference_audio: Some("/ref.wav".into()),
                reference_text: Some("hi".into()),
                response_tx: tx,
            },
        )
        .unwrap();

        let result = rx.await.unwrap().unwrap();
        assert_eq!(result.to_vec1::<f32>().unwrap(), vec![-0.5f32; 5]);

        // Nothing should have been written to the cache directory.
        let has_entries = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .any(|e| e.path().is_dir());
        assert!(!has_entries);
    }

    /// Registers `asr` on `Device::Cpu` -- the device is irrelevant to
    /// these tests since `with_context` is a cheap no-op wrapper on CPU
    /// either way.
    fn register_test_asr(
        rt: &mut ModelRuntime,
        name: &str,
        model_type_name: &'static str,
        asr: Box<dyn Asr + Send>,
    ) {
        rt.register_asr(name.into(), model_type_name, asr, &Device::Cpu)
            .unwrap();
    }

    struct MockAsr {
        input_sample_rate: u32,
    }

    impl MockAsr {
        fn new() -> Self {
            Self {
                input_sample_rate: 16000,
            }
        }

        fn with_sample_rate(mut self, sample_rate: u32) -> Self {
            self.input_sample_rate = sample_rate;
            self
        }
    }

    impl Asr for MockAsr {
        fn input_sample_rate(&self) -> u32 {
            self.input_sample_rate
        }

        fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(Transcript {
                text: format!("heard {} samples", audio.len()),
                language: opts.language.clone(),
                is_final: true,
            })
        }
    }

    #[test]
    fn test_register_asr_stores_handle() {
        let mut rt = ModelRuntime::new();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr::new()));

        let handle = rt.asr_handle("m1").expect("handle should be registered");
        assert_eq!(handle.input_sample_rate(), 16000);
        assert_eq!(handle.model_type_name(), "qwen3_asr");
        assert_eq!(handle.pending_count(), 0);
    }

    #[test]
    fn test_transcribe_roundtrip() {
        let mut rt = ModelRuntime::new();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr::new()));
        let handle = rt.asr_handle("m1").unwrap();

        let (tx, rx) = oneshot::channel();
        handle
            .send(AsrTranscribeRequest {
                audio: vec![0.0f32; 4],
                language: None,
                response_tx: tx,
            })
            .unwrap();

        let transcript = rx.blocking_recv().unwrap().unwrap();
        assert_eq!(transcript.text, "heard 4 samples");
        assert!(transcript.is_final);
        assert_eq!(transcript.language, None);
    }

    #[test]
    fn test_transcribe_with_language_hint() {
        let mut rt = ModelRuntime::new();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr::new()));

        let (tx, rx) = oneshot::channel();
        rt.transcribe(
            "m1",
            AsrTranscribeRequest {
                audio: vec![0.0f32; 2],
                language: Some("de".into()),
                response_tx: tx,
            },
        )
        .unwrap();

        let transcript = rx.blocking_recv().unwrap().unwrap();
        assert_eq!(transcript.language, Some("de".into()));
    }

    #[test]
    fn test_multiple_asr_models() {
        let mut rt = ModelRuntime::new();
        register_test_asr(
            &mut rt,
            "a",
            "qwen3_asr",
            Box::new(MockAsr::new().with_sample_rate(16000)),
        );
        register_test_asr(
            &mut rt,
            "b",
            "qwen3_asr",
            Box::new(MockAsr::new().with_sample_rate(8000)),
        );

        assert_eq!(rt.asr_handle("a").unwrap().input_sample_rate(), 16000);
        assert_eq!(rt.asr_handle("b").unwrap().input_sample_rate(), 8000);
    }

    #[test]
    fn test_default_asr_handle() {
        let mut rt = ModelRuntime::new();
        assert!(rt.default_asr_handle().is_none());

        register_test_asr(&mut rt, "a", "qwen3_asr", Box::new(MockAsr::new()));
        assert!(rt.default_asr_handle().is_some());
    }

    #[test]
    fn test_default_asr_name() {
        let mut rt = ModelRuntime::new();
        assert!(rt.default_asr_name().is_none());
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr::new()));
        assert_eq!(rt.default_asr_name(), Some("m1"));
    }

    #[test]
    fn test_asr_handle_not_found() {
        let rt = ModelRuntime::new();
        assert!(rt.asr_handle("nonexistent").is_none());
    }

    #[test]
    fn test_asr_handles_iterator() {
        let mut rt = ModelRuntime::new();
        register_test_asr(&mut rt, "a", "qwen3_asr", Box::new(MockAsr::new()));
        register_test_asr(&mut rt, "b", "qwen3_asr", Box::new(MockAsr::new()));

        let mut names: Vec<&str> = rt.asr_handles().map(|(name, _)| name).collect();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_extract_asr_model_name() {
        assert_eq!(
            extract_asr_model_name("/models/Qwen3-ASR-0.6B-hf"),
            "Qwen3-ASR-0.6B-hf"
        );
        assert_eq!(extract_asr_model_name("/models/qwen-asr/"), "qwen-asr");
        assert_eq!(extract_asr_model_name(""), "asr");
    }

    #[test]
    fn test_send_after_asr_thread_stopped() {
        let (tx, rx) = mpsc::unbounded_channel::<AsrRequest>();
        drop(rx);

        let handle = AsrHandle {
            tx,
            input_sample_rate: 16000,
            model_type_name: "qwen3_asr",
            pending_count: Arc::new(AtomicU64::new(0)),
        };

        let (resp_tx, _resp_rx) = oneshot::channel();
        let result = handle.send(AsrTranscribeRequest {
            audio: vec![0.0f32; 4],
            language: None,
            response_tx: resp_tx,
        });

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("stopped"));
        assert_eq!(handle.pending_count(), 0);
    }

    #[test]
    fn test_register_asr_duplicate_name() {
        let mut rt = ModelRuntime::new();
        register_test_asr(
            &mut rt,
            "dup",
            "qwen3_asr",
            Box::new(MockAsr::new().with_sample_rate(16000)),
        );
        register_test_asr(
            &mut rt,
            "dup",
            "qwen3_asr",
            Box::new(MockAsr::new().with_sample_rate(8000)),
        );

        let handle = rt.asr_handle("dup").unwrap();
        assert_eq!(handle.input_sample_rate(), 8000);
    }

    #[test]
    fn test_transcribe_unknown_model() {
        let rt = ModelRuntime::new();
        let (tx, _rx) = oneshot::channel();
        let result = rt.transcribe(
            "nonexistent",
            AsrTranscribeRequest {
                audio: vec![0.0f32; 2],
                language: None,
                response_tx: tx,
            },
        );
        assert!(result.is_err());
    }

    struct PanickingAsr {
        call_count: u32,
    }

    impl PanickingAsr {
        fn new() -> Self {
            Self { call_count: 0 }
        }
    }

    impl Asr for PanickingAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, audio: &[f32], _opts: &TranscribeOptions) -> Result<Transcript> {
            self.call_count += 1;
            assert!(self.call_count != 1, "simulated model panic");
            Ok(Transcript {
                text: format!("ok {} samples", audio.len()),
                language: None,
                is_final: true,
            })
        }
    }

    #[test]
    fn test_panic_in_transcribe_is_caught() {
        let mut rt = ModelRuntime::new();
        register_test_asr(
            &mut rt,
            "panic_model",
            "qwen3_asr",
            Box::new(PanickingAsr::new()),
        );
        let handle = rt.asr_handle("panic_model").unwrap();

        let (tx1, rx1) = oneshot::channel();
        handle
            .send(AsrTranscribeRequest {
                audio: vec![0.0f32; 4],
                language: None,
                response_tx: tx1,
            })
            .unwrap();

        let result1 = rx1.blocking_recv().unwrap();
        assert!(result1.is_err());
        assert!(result1.unwrap_err().to_string().contains("panicked"));

        let (tx2, rx2) = oneshot::channel();
        handle
            .send(AsrTranscribeRequest {
                audio: vec![0.0f32; 2],
                language: None,
                response_tx: tx2,
            })
            .unwrap();

        let result2 = rx2.blocking_recv().unwrap();
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().text, "ok 2 samples");
    }

    struct StreamingMockAsr {
        chunks: Vec<&'static str>,
    }

    impl StreamingMockAsr {
        fn new(chunks: Vec<&'static str>) -> Self {
            Self { chunks }
        }
    }

    impl Asr for StreamingMockAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, audio: &[f32], _opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(Transcript {
                text: format!("heard {} samples", audio.len()),
                language: None,
                is_final: true,
            })
        }

        fn transcribe_stream(
            &mut self,
            _audio: &[f32],
            _opts: &TranscribeOptions,
        ) -> Result<crane::audio::AsrStream<'_>> {
            let last = self.chunks.len().saturating_sub(1);
            let chunks: Vec<Result<Transcript>> = self
                .chunks
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    Ok(Transcript {
                        text: (*text).to_string(),
                        language: None,
                        is_final: i == last,
                    })
                })
                .collect();
            Ok(crane::audio::AsrStream::new(chunks.into_iter()))
        }
    }

    #[test]
    fn test_transcribe_stream_multiple_chunks() {
        let mut rt = ModelRuntime::new();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(StreamingMockAsr::new(vec![
                "one",
                "one two",
                "one two three",
            ])),
        );

        let mut rx = rt.transcribe_stream("m1", vec![0.0f32; 4], None).unwrap();

        let mut received = Vec::new();
        while let Some(result) = rx.blocking_recv() {
            received.push(result.unwrap().text);
        }
        assert_eq!(received, vec!["one", "one two", "one two three"]);
    }

    #[test]
    fn test_transcribe_stream_single_chunk_default_impl() {
        // MockAsr does not override transcribe_stream, so the default
        // impl wraps transcribe() in a single-item stream.
        let mut rt = ModelRuntime::new();
        register_test_asr(&mut rt, "m1", "qwen3_asr", Box::new(MockAsr::new()));

        let mut rx = rt.transcribe_stream("m1", vec![0.0f32; 4], None).unwrap();

        let first = rx.blocking_recv().unwrap().unwrap();
        assert_eq!(first.text, "heard 4 samples");
        assert!(first.is_final);
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn test_transcribe_stream_unknown_model() {
        let rt = ModelRuntime::new();
        let result = rt.transcribe_stream("nonexistent", vec![0.0f32; 2], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcribe_stream_empty() {
        let mut rt = ModelRuntime::new();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(StreamingMockAsr::new(vec![])),
        );

        let mut rx = rt.transcribe_stream("m1", vec![0.0f32; 4], None).unwrap();
        assert!(rx.blocking_recv().is_none());
    }

    #[test]
    fn test_transcribe_stream_receiver_dropped() {
        let mut rt = ModelRuntime::new();
        register_test_asr(
            &mut rt,
            "m1",
            "qwen3_asr",
            Box::new(StreamingMockAsr::new(vec!["one", "two", "three"])),
        );

        let mut rx = rt.transcribe_stream("m1", vec![0.0f32; 4], None).unwrap();
        rx.blocking_recv().unwrap().unwrap();
        drop(rx);

        // The worker thread must notice the dropped receiver and move on --
        // this batch request would hang forever if it didn't.
        let (tx, rx2) = oneshot::channel();
        rt.transcribe(
            "m1",
            AsrTranscribeRequest {
                audio: vec![0.0f32; 2],
                language: None,
                response_tx: tx,
            },
        )
        .unwrap();

        assert!(rx2.blocking_recv().unwrap().is_ok());
    }

    /// A transcript iterator that yields one partial chunk, then panics on
    /// the next pull -- mirrors [`PanicOnSecondChunk`] for TTS streaming.
    struct PanicOnSecondTranscriptChunk {
        index: usize,
    }

    impl Iterator for PanicOnSecondTranscriptChunk {
        type Item = Result<Transcript>;

        fn next(&mut self) -> Option<Self::Item> {
            self.index += 1;
            match self.index {
                1 => Some(Ok(Transcript {
                    text: "partial".to_string(),
                    language: None,
                    is_final: false,
                })),
                2 => panic!("simulated stream panic"),
                _ => None,
            }
        }
    }

    struct StreamPanickingAsr {
        call_count: u32,
    }

    impl StreamPanickingAsr {
        fn new() -> Self {
            Self { call_count: 0 }
        }
    }

    impl Asr for StreamPanickingAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, audio: &[f32], _opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(Transcript {
                text: format!("ok {} samples", audio.len()),
                language: None,
                is_final: true,
            })
        }

        fn transcribe_stream(
            &mut self,
            _audio: &[f32],
            _opts: &TranscribeOptions,
        ) -> Result<crane::audio::AsrStream<'_>> {
            self.call_count += 1;
            if self.call_count == 1 {
                Ok(crane::audio::AsrStream::new(PanicOnSecondTranscriptChunk {
                    index: 0,
                }))
            } else {
                let chunks: Vec<Result<Transcript>> = vec![Ok(Transcript {
                    text: "ok".to_string(),
                    language: None,
                    is_final: true,
                })];
                Ok(crane::audio::AsrStream::new(chunks.into_iter()))
            }
        }
    }

    #[test]
    fn test_transcribe_stream_panic_recovery() {
        let mut rt = ModelRuntime::new();
        register_test_asr(
            &mut rt,
            "panic_stream",
            "qwen3_asr",
            Box::new(StreamPanickingAsr::new()),
        );

        let mut rx = rt
            .transcribe_stream("panic_stream", vec![0.0f32; 4], None)
            .unwrap();

        let first = rx.blocking_recv().unwrap();
        assert!(first.is_ok());

        let second = rx.blocking_recv().unwrap();
        assert!(second.is_err());
        assert!(second.unwrap_err().to_string().contains("panicked"));

        assert!(rx.blocking_recv().is_none());

        // The thread must have recovered: the next stream request succeeds.
        let mut rx2 = rt
            .transcribe_stream("panic_stream", vec![0.0f32; 4], None)
            .unwrap();
        let result = rx2.blocking_recv().unwrap();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().text, "ok");
        assert!(rx2.blocking_recv().is_none());
    }
}
