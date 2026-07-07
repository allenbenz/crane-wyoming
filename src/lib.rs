//! Wyoming protocol wire format for Crane.
//!
//! Implements the JSONL + binary framing used by the
//! [Wyoming protocol](https://github.com/rhasspy/wyoming) for
//! Home Assistant voice integration. This crate handles serialization
//! and deserialization of Wyoming events over async byte streams
//! (`tokio::io::AsyncRead` / `AsyncWrite`).
//!
//! # Wire format
//!
//! Each message consists of three parts:
//!
//! 1. **JSON header line** (newline-terminated) -- type, version,
//!    `data_length`, `payload_length`
//! 2. **Extended data** (`data_length` bytes) -- UTF-8 JSON of the
//!    event's data dict
//! 3. **Binary payload** (`payload_length` bytes) -- raw bytes (PCM
//!    audio for `audio-chunk` events)
//!
//! The protocol is symmetric: both client and server use the same
//! framing.
//!
//! # Module layout
//!
//! | Module    | Responsibility                                       |
//! |-----------|-------------------------------------------------------|
//! | `event`   | Typed event enum and per-event data structs           |
//! | `wire`    | Async read/write functions for the wire protocol      |
//! | `handler` | TTS event handling, dispatching to `ModelRuntime`     |

pub mod event;
pub mod handler;
pub mod wire;

pub use event::Event;
pub use handler::{VoiceMap, handle_connection};
pub use wire::{MAX_DATA_LENGTH, MAX_HEADER_LINE, MAX_PAYLOAD_LENGTH, read_event, write_event};

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use crane_engine::model_factory::ModelType;
use crane_engine::{ModelRuntime, TtsCache};
use tracing::info;

/// Command-line arguments for the Wyoming protocol TTS server.
#[derive(Parser, Debug, Clone)]
#[command(about = "Wyoming protocol TTS server for Home Assistant voice integration")]
pub struct Args {
    /// Path to a TTS model directory. Repeat to load multiple models; the
    /// first `--model-path` becomes the default voice when a client does not
    /// request one by name.
    #[arg(short = 'm', long = "model-path", required = true)]
    pub model_path: Vec<PathBuf>,

    /// TCP port to listen on.
    #[arg(short = 'p', long, default_value_t = 10200)]
    pub port: u16,

    /// Host address to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Force CPU-only inference (disables CUDA/Metal auto-detection).
    #[arg(long)]
    pub cpu: bool,

    /// Maximum number of concurrent client connections.
    #[arg(long, default_value_t = 16)]
    pub max_connections: usize,

    /// Directory for the on-disk TTS response cache. Omit to disable caching.
    #[arg(long)]
    pub tts_cache_dir: Option<PathBuf>,

    /// Maximum size of the TTS cache, e.g. `"500M"` or `"1G"`.
    #[arg(long, default_value = "500M")]
    pub tts_cache_max_size: String,
}

/// Parse a human-readable byte size string (e.g. `"500M"`, `"1G"`, `"1024"`)
/// into a byte count.
///
/// Accepts optional `K`/`M`/`G` suffixes (case-insensitive), with or without
/// a trailing `B` (so `"500M"` and `"500MB"` are equivalent). A bare integer
/// is interpreted as a byte count.
///
/// # Errors
///
/// Returns an error if `s` is empty or not a valid size string.
fn parse_size(s: &str) -> Result<u64> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        anyhow::bail!("empty size string");
    }
    let upper = trimmed.to_ascii_uppercase();
    let (num_part, multiplier) =
        if let Some(prefix) = upper.strip_suffix("GB").or_else(|| upper.strip_suffix('G')) {
            (prefix, 1024u64.pow(3))
        } else if let Some(prefix) = upper.strip_suffix("MB").or_else(|| upper.strip_suffix('M')) {
            (prefix, 1024u64.pow(2))
        } else if let Some(prefix) = upper.strip_suffix("KB").or_else(|| upper.strip_suffix('K')) {
            (prefix, 1024u64)
        } else {
            (upper.as_str(), 1u64)
        };
    let value: f64 = num_part
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid size string: {s}"))?;
    if value < 0.0 {
        anyhow::bail!("negative size is not allowed: {s}");
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let bytes = (value * multiplier as f64) as u64;
    Ok(bytes)
}

/// Load the tokenizer for a TTS-only model, falling back to a stub tokenizer
/// (TTS models don't need one for generation, only for `ModelRuntime`
/// metadata plumbing shared with the LLM code path).
fn load_tts_tokenizer(model_path: &str) -> tokenizers::Tokenizer {
    crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(model_path).unwrap_or_else(
        |e| {
            tracing::warn!("Failed to load HF tokenizer: {e}; creating stub for TTS-only mode");
            tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
        },
    )
}

/// Detect the EOS token ID from a loaded tokenizer.
fn detect_tts_eos(tokenizer: &tokenizers::Tokenizer) -> u32 {
    tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(2)
}

/// Initialize the tracing subscriber with compact formatting.
fn init_logging() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();
}

/// Entry point for the `crane-wyoming` binary.
///
/// Parses command-line arguments and runs the server. See [`Args`] for
/// available flags.
///
/// # Errors
///
/// Returns an error if model loading, TCP binding, or the accept loop fails.
pub async fn cli_main() -> Result<()> {
    init_logging();
    run(Args::parse()).await
}

/// Serve a single Wyoming connection to completion.
///
/// Splits the accepted stream into buffered read/write halves and runs
/// [`handle_connection`]'s event loop against `runtime` and `voice_map`.
async fn serve_connection(
    stream: tokio::net::TcpStream,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
) -> Result<()> {
    let (reader, writer) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(reader);
    let mut writer = tokio::io::BufWriter::new(writer);
    handle_connection(&mut reader, &mut writer, runtime, voice_map).await
}

/// Run the Wyoming protocol TTS server.
///
/// Loads each `--model-path` into a shared [`ModelRuntime`], optionally enables
/// the on-disk TTS cache, then accepts TCP connections on `args.host`:
/// `args.port`. Each connection is handled on its own tokio task via
/// [`handle_connection`]; TTS requests from all connections queue on the
/// target model's dedicated thread and are processed one at a time.
///
/// # Errors
///
/// Returns an error if a model fails to load or the TCP listener cannot bind.
pub async fn run(args: Args) -> Result<()> {
    let device = if args.cpu {
        crane_core::models::Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            crane_core::models::Device::cuda_if_available(0)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            #[cfg(target_os = "macos")]
            {
                crane_core::models::Device::new_metal(0).unwrap_or(crane_core::models::Device::Cpu)
            }
            #[cfg(not(target_os = "macos"))]
            {
                crane_core::models::Device::Cpu
            }
        }
    };

    #[cfg(feature = "cuda")]
    let dtype = if args.cpu {
        crane_core::models::DType::F32
    } else {
        crane_core::models::DType::BF16
    };
    #[cfg(not(feature = "cuda"))]
    let dtype = crane_core::models::DType::F32;

    let device_name = format!("{device:?}");
    let dtype_name = format!("{dtype:?}");
    info!("Device: {device_name}, dtype: {dtype_name}");

    let first_model_path = args
        .model_path
        .first()
        .ok_or_else(|| anyhow::anyhow!("at least one --model-path is required"))?;
    let first_model_str = first_model_path.to_string_lossy();
    let tokenizer = load_tts_tokenizer(&first_model_str);
    let eos_id = detect_tts_eos(&tokenizer);
    let model_name = first_model_path.file_name().map_or_else(
        || "crane-wyoming".to_string(),
        |n| n.to_string_lossy().to_string(),
    );

    let mut runtime = ModelRuntime::new(
        model_name,
        ModelType::Auto,
        dtype_name,
        device_name,
        tokenizer,
        vec![eos_id],
    );

    if let Some(cache_dir) = &args.tts_cache_dir {
        let max_bytes = parse_size(&args.tts_cache_max_size)?;
        runtime.set_tts_cache(TtsCache::new(cache_dir.clone(), max_bytes)?);
        info!(dir = %cache_dir.display(), max = %args.tts_cache_max_size, "TTS cache enabled");
    }

    let mut model_names = Vec::with_capacity(args.model_path.len());
    for model_path in &args.model_path {
        let path_str = model_path.to_string_lossy();
        let name = runtime.load_tts(&path_str, &device, &dtype)?;
        info!(name = %name, path = %path_str, "TTS model loaded");
        model_names.push(name);
    }

    let voice_map = VoiceMap::new(&model_names, &runtime);
    let runtime = Arc::new(runtime);
    let voice_map = Arc::new(voice_map);

    let addr = if args.host.contains(':') {
        format!("[{}]:{}", args.host, args.port)
    } else {
        format!("{}:{}", args.host, args.port)
    };
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let local_addr = listener.local_addr()?;
    info!(
        version = env!("CARGO_PKG_VERSION"),
        listen = %local_addr,
        models = model_names.len(),
        "crane-wyoming ready"
    );

    let semaphore = Arc::new(tokio::sync::Semaphore::new(args.max_connections));
    serve(listener, runtime, voice_map, semaphore).await
}

/// Accept and serve Wyoming connections on `listener` until a shutdown
/// signal (Ctrl-C) is received.
///
/// Each connection is dispatched to its own tokio task via
/// [`serve_connection`]; a connection-level error is logged and does not
/// affect other connections. Accept errors (e.g. transient resource
/// exhaustion) are logged and do not stop the server. Concurrent
/// connections are capped by `semaphore`; once the cap is reached, the
/// accept loop stalls until a slot frees up.
async fn serve(
    listener: tokio::net::TcpListener,
    runtime: Arc<ModelRuntime>,
    voice_map: Arc<VoiceMap>,
    semaphore: Arc<tokio::sync::Semaphore>,
) -> Result<()> {
    loop {
        let (stream, peer_addr) = tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok(conn) => conn,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to accept connection");
                        continue;
                    },
                }
            },
            _ = tokio::signal::ctrl_c() => {
                info!("Received shutdown signal, stopping accept loop");
                break;
            },
        };

        let Ok(permit) = Arc::clone(&semaphore).acquire_owned().await else {
            break;
        };
        info!(peer = %peer_addr, "connection accepted");
        let runtime = Arc::clone(&runtime);
        let voice_map = Arc::clone(&voice_map);
        tokio::spawn(async move {
            if let Err(e) = serve_connection(stream, &runtime, &voice_map).await {
                tracing::warn!(peer = %peer_addr, error = %e, "connection ended with error");
            }
            drop(permit);
        });
    }
    info!("Accept loop stopped; in-flight connections will complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{PingData, SynthesizeData};
    use candle_core::{Device, Tensor};
    use crane::audio::tts::{AudioInfo, Tts, VoiceInfo, pcm_f32_to_i16};
    use crane_core::generation::SpeechOptions;
    use crane_engine::model_factory::ModelType;
    use tokio::net::{TcpListener, TcpStream};

    #[test]
    fn parse_size_bare_bytes() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
    }

    #[test]
    fn parse_size_kilobytes() {
        assert_eq!(parse_size("500K").unwrap(), 500 * 1024);
        assert_eq!(parse_size("500KB").unwrap(), 500 * 1024);
    }

    #[test]
    fn parse_size_megabytes() {
        assert_eq!(parse_size("500M").unwrap(), 500 * 1024 * 1024);
        assert_eq!(parse_size("100MB").unwrap(), 100 * 1024 * 1024);
    }

    #[test]
    fn parse_size_gigabytes() {
        assert_eq!(parse_size("1G").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(
            parse_size("1.5G").unwrap(),
            (1.5 * 1024.0 * 1024.0 * 1024.0) as u64
        );
    }

    #[test]
    fn parse_size_lowercase_suffix() {
        assert_eq!(parse_size("500m").unwrap(), 500 * 1024 * 1024);
    }

    #[test]
    fn parse_size_rejects_empty() {
        assert!(parse_size("").is_err());
        assert!(parse_size("   ").is_err());
    }

    #[test]
    fn parse_size_rejects_invalid() {
        assert!(parse_size("abc").is_err());
        assert!(parse_size("M").is_err());
    }

    #[test]
    fn parse_size_rejects_negative() {
        assert!(parse_size("-500M").is_err());
        assert!(parse_size("-1").is_err());
    }

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
        ) -> Result<Tensor> {
            let n = text.chars().count().max(1);
            Tensor::new(vec![0.5f32; n], &Device::Cpu).map_err(Into::into)
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

    async fn spawn_test_server(runtime: ModelRuntime, voice_map: VoiceMap) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16));
        tokio::spawn(serve(
            listener,
            Arc::new(runtime),
            Arc::new(voice_map),
            semaphore,
        ));
        addr
    }

    #[tokio::test]
    async fn tcp_ping_pong() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let stream = TcpStream::connect(addr).await.unwrap();
        let (reader, mut writer) = stream.into_split();
        let mut reader = tokio::io::BufReader::new(reader);

        write_event(
            &mut writer,
            &Event::Ping(PingData {
                text: Some("hi".into()),
            }),
        )
        .await
        .unwrap();

        let response = read_event(&mut reader).await.unwrap().unwrap();
        match response {
            Event::Pong(data) => assert_eq!(data.text.as_deref(), Some("hi")),
            other => panic!("expected Pong, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tcp_synthesize_round_trip() {
        let mut rt = test_runtime();
        rt.register_tts(
            "m1".into(),
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["en".into()],
                }],
            )),
        )
        .unwrap();
        let vm = VoiceMap::new(&["m1".to_string()], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let stream = TcpStream::connect(addr).await.unwrap();
        let (reader, mut writer) = stream.into_split();
        let mut reader = tokio::io::BufReader::new(reader);

        write_event(
            &mut writer,
            &Event::Synthesize(SynthesizeData {
                text: "hi".into(),
                voice: None,
            }),
        )
        .await
        .unwrap();

        let start = read_event(&mut reader).await.unwrap().unwrap();
        match start {
            Event::AudioStart(data) => {
                assert_eq!(data.rate, 24000);
                assert_eq!(data.width, 2);
                assert_eq!(data.channels, 1);
            },
            other => panic!("expected AudioStart, got {other:?}"),
        }

        let chunk = read_event(&mut reader).await.unwrap().unwrap();
        match chunk {
            Event::AudioChunk { audio, .. } => {
                assert_eq!(audio, pcm_f32_to_i16(&[0.5f32; 2]));
            },
            other => panic!("expected AudioChunk, got {other:?}"),
        }

        let stop = read_event(&mut reader).await.unwrap().unwrap();
        assert!(matches!(stop, Event::AudioStop(_)));
    }
}
