//! Wyoming protocol TTS server for Home Assistant voice integration.
//!
//! # Module layout
//!
//! | Module    | Responsibility                                       |
//! |-----------|-------------------------------------------------------|
//! | `engine`  | `ModelRuntime` -- owns and dispatches to TTS models   |
//! | `handler` | TTS event handling, dispatching to `ModelRuntime`     |
//!
//! The wire protocol types (`Event`, `read_event`, `write_event`) are
//! implemented in and re-exported from the [`wyoming_protocol`] crate.

pub mod engine;
pub mod handler;

pub use handler::{VoiceMap, handle_connection};
pub use wyoming_protocol::error::ProtocolError;
pub use wyoming_protocol::event::Event;
pub use wyoming_protocol::wire::{
    MAX_DATA_LENGTH, MAX_HEADER_LINE, MAX_PAYLOAD_LENGTH, read_event, write_event,
};

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tracing::info;

use crate::engine::ModelRuntime;

/// Command-line arguments for the Wyoming protocol TTS server.
#[derive(Parser, Debug, Clone)]
#[command(about = "Wyoming protocol TTS server for Home Assistant voice integration")]
pub struct Args {
    /// Path to a TTS model directory. Separate multiple paths with `;` to
    /// load multiple models; the first path becomes the default voice when
    /// a client does not request one by name.
    #[arg(
        short = 'm',
        long = "model-path",
        required = true,
        env = "CRANE_WYOMING_MODEL_PATH",
        value_delimiter = ';'
    )]
    pub model_path: Vec<PathBuf>,

    /// TCP port to listen on.
    #[arg(short = 'p', long, default_value_t = 10200, env = "CRANE_WYOMING_PORT")]
    pub port: u16,

    /// Host address to bind to.
    #[arg(long, default_value = "0.0.0.0", env = "CRANE_WYOMING_HOST")]
    pub host: String,

    /// Address to listen on, given as a URI: `tcp://host:port` or
    /// `unix:///path/to/socket`. Overrides `--host`/`--port` when set.
    #[arg(long, env = "CRANE_WYOMING_URI")]
    pub uri: Option<String>,

    /// Force CPU-only inference (disables CUDA/Metal auto-detection).
    #[arg(long, env = "CRANE_WYOMING_CPU")]
    pub cpu: bool,

    /// Maximum number of concurrent client connections.
    #[arg(long, default_value_t = 16, env = "CRANE_WYOMING_MAX_CONNECTIONS")]
    pub max_connections: usize,

    /// Directory for the on-disk TTS response cache. Omit to disable caching.
    #[arg(long, env = "CRANE_WYOMING_TTS_CACHE_DIR")]
    pub tts_cache_dir: Option<PathBuf>,

    /// Maximum size of the TTS cache, e.g. `"500M"` or `"1G"`.
    #[arg(long, default_value = "500M", env = "CRANE_WYOMING_TTS_CACHE_MAX_SIZE")]
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

/// Where to listen, resolved from `--uri` (if set) or `--host`/`--port`.
enum BindTarget {
    /// Bind a TCP listener to this address (`host:port`).
    Tcp(String),
    /// Bind a Unix domain socket listener at this path.
    #[cfg(unix)]
    Unix(PathBuf),
}

/// Resolve `args` into a [`BindTarget`].
///
/// `--uri` takes precedence when set; it must use the `tcp://` or
/// `unix://` scheme. Otherwise falls back to `--host`/`--port`.
///
/// # Errors
///
/// Returns an error if `--uri` is set but uses an unsupported scheme, or
/// requests a `unix://` socket on a platform without Unix socket support.
fn resolve_bind_target(args: &Args) -> Result<BindTarget> {
    if let Some(uri) = &args.uri {
        if let Some(rest) = uri.strip_prefix("tcp://") {
            return Ok(BindTarget::Tcp(rest.to_string()));
        }
        if let Some(path) = uri.strip_prefix("unix://") {
            #[cfg(unix)]
            {
                return Ok(BindTarget::Unix(PathBuf::from(path)));
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                anyhow::bail!("unix:// sockets are not supported on this platform");
            }
        }
        anyhow::bail!("invalid --uri (expected tcp:// or unix://): {uri}");
    }
    let addr = if args.host.contains(':') {
        format!("[{}]:{}", args.host, args.port)
    } else {
        format!("{}:{}", args.host, args.port)
    };
    Ok(BindTarget::Tcp(addr))
}

/// Identifies a Unix domain socket file on disk, so a [`Listener`] can tell
/// on drop whether the file at its bind path is still the same socket it
/// created (as opposed to one a different process rebound in the meantime).
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SocketInode {
    /// Device ID of the filesystem the socket file lives on.
    dev: u64,
    /// Inode number of the socket file.
    ino: u64,
}

/// Read the `(device, inode)` identity of the file at `path`.
///
/// # Errors
///
/// Returns an error if `path` cannot be stat'd.
#[cfg(unix)]
fn socket_inode(path: &std::path::Path) -> Result<SocketInode> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path)?;
    Ok(SocketInode {
        dev: meta.dev(),
        ino: meta.ino(),
    })
}

/// A bound listener, either TCP or (on Unix platforms) a Unix domain socket.
enum Listener {
    /// TCP listener.
    Tcp(tokio::net::TcpListener),
    /// Unix domain socket listener, with its bound path and inode identity
    /// (for cleanup on drop).
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf, SocketInode),
}

impl Listener {
    /// Bind a listener for `target`.
    ///
    /// For `unix://` targets, binding is attempted first; only if that
    /// fails with `AddrInUse` is the existing file removed and the bind
    /// retried, matching the Wyoming reference server's handling of stale
    /// sockets left behind by a previous, uncleanly-terminated run. This
    /// avoids unlinking a socket another process is still actively using.
    async fn bind(target: BindTarget) -> Result<Self> {
        match target {
            BindTarget::Tcp(addr) => Ok(Self::Tcp(tokio::net::TcpListener::bind(&addr).await?)),
            #[cfg(unix)]
            BindTarget::Unix(path) => {
                let listener = match tokio::net::UnixListener::bind(&path) {
                    Ok(listener) => listener,
                    Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                        std::fs::remove_file(&path)?;
                        tokio::net::UnixListener::bind(&path)?
                    },
                    Err(e) => return Err(e.into()),
                };
                let inode = socket_inode(&path)?;
                Ok(Self::Unix(listener, path, inode))
            },
        }
    }

    /// Accept the next connection, returning the stream and a display
    /// string identifying the peer (a socket address for TCP; a fixed
    /// label for Unix sockets, which have no meaningful peer address).
    async fn accept(&self) -> std::io::Result<(Stream, String)> {
        match self {
            Self::Tcp(listener) => {
                let (stream, addr) = listener.accept().await?;
                Ok((Stream::Tcp(stream), addr.to_string()))
            },
            #[cfg(unix)]
            Self::Unix(listener, ..) => {
                let (stream, _addr) = listener.accept().await?;
                Ok((Stream::Unix(stream), "unix-socket-client".to_string()))
            },
        }
    }

    /// Display string for the bound address, used in the startup log line.
    fn display_addr(&self) -> String {
        match self {
            Self::Tcp(listener) => listener
                .local_addr()
                .map_or_else(|_| "unknown".to_string(), |a| a.to_string()),
            #[cfg(unix)]
            Self::Unix(_, path, _) => format!("unix://{}", path.display()),
        }
    }
}

impl Drop for Listener {
    /// Remove the socket file on shutdown so it doesn't linger as a stale
    /// socket for the next run -- but only if the file at `path` is still
    /// the same socket this listener bound (identified by device+inode),
    /// so a socket that another process has since rebound at the same path
    /// is left alone.
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Self::Unix(_, path, inode) = self
            && socket_inode(path).ok().as_ref() == Some(inode)
        {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// An accepted connection, either a TCP stream or (on Unix platforms) a
/// Unix domain socket stream.
enum Stream {
    /// TCP stream.
    Tcp(tokio::net::TcpStream),
    /// Unix domain socket stream.
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
}

/// Serve a single Wyoming connection to completion.
///
/// Splits the accepted stream into buffered read/write halves and runs
/// [`handle_connection`]'s event loop against `runtime` and `voice_map`.
async fn serve_connection(
    stream: Stream,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
) -> Result<()> {
    /// Split a stream into buffered halves and run [`handle_connection`].
    /// A macro (rather than a generic helper) because `TcpStream::into_split`
    /// and `UnixStream::into_split` return unrelated concrete types.
    macro_rules! split_and_handle {
        ($stream:expr) => {{
            let (reader, writer) = $stream.into_split();
            let mut reader = tokio::io::BufReader::new(reader);
            let mut writer = tokio::io::BufWriter::new(writer);
            handle_connection(&mut reader, &mut writer, runtime, voice_map).await
        }};
    }

    match stream {
        Stream::Tcp(stream) => split_and_handle!(stream),
        #[cfg(unix)]
        Stream::Unix(stream) => split_and_handle!(stream),
    }
}

/// Run the Wyoming protocol TTS server.
///
/// Loads each `--model-path` into a shared [`ModelRuntime`], optionally enables
/// the on-disk TTS cache, then accepts connections on the address resolved
/// from `--uri` (or `args.host`:`args.port` if unset) -- either TCP or a
/// Unix domain socket. Each connection is handled on its own tokio task via
/// [`handle_connection`]; TTS requests from all connections queue on the
/// target model's dedicated thread and are processed one at a time.
///
/// # Errors
///
/// Returns an error if a model fails to load or the listener cannot bind.
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

    // Incremental streaming is only worth it on a GPU: both CUDA and Metal
    // have far more memory bandwidth than CPU DDR, which is what
    // autoregressive TTS generation is bottlenecked on. On CPU, chunks
    // arrive slower than they play back, producing bursty audio with dead
    // air -- worse than just waiting once for the full clip.
    let streaming_enabled = device.is_cuda() || device.is_metal();

    let device_name = format!("{device:?}");
    let dtype_name = format!("{dtype:?}");
    info!("Device: {device_name}, dtype: {dtype_name}, streaming: {streaming_enabled}");

    if args.model_path.is_empty() {
        anyhow::bail!("at least one --model-path is required");
    }

    let mut runtime = ModelRuntime::new();
    runtime.set_streaming_enabled(streaming_enabled);

    if let Some(cache_dir) = &args.tts_cache_dir {
        let max_bytes = parse_size(&args.tts_cache_max_size)?;
        runtime.set_tts_cache(crate::engine::TtsCache::new(cache_dir.clone(), max_bytes)?);
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

    let listener = Listener::bind(resolve_bind_target(&args)?).await?;
    info!(
        version = env!("CARGO_PKG_VERSION"),
        listen = %listener.display_addr(),
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
    listener: Listener,
    runtime: Arc<ModelRuntime>,
    voice_map: Arc<VoiceMap>,
    semaphore: Arc<tokio::sync::Semaphore>,
) -> Result<()> {
    let mut conn_id: u64 = 0;
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
        conn_id += 1;
        info!(peer = %peer_addr, conn = conn_id, "connection accepted");
        let runtime = Arc::clone(&runtime);
        let voice_map = Arc::clone(&voice_map);
        tokio::spawn(async move {
            if let Err(e) = serve_connection(stream, &runtime, &voice_map).await {
                tracing::warn!(peer = %peer_addr, conn = conn_id, error = %e, "connection ended with error");
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
    use candle_core::{Device, Tensor};
    use crane::audio::tts::{AudioInfo, Tts, VoiceInfo, pcm_f32_to_i16};
    use crane_core::generation::SpeechOptions;
    use tokio::net::{TcpListener, TcpStream};
    use wyoming_protocol::event::{PingData, SynthesizeData};

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
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
        ModelRuntime::new()
    }

    async fn spawn_test_server(runtime: ModelRuntime, voice_map: VoiceMap) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16));
        tokio::spawn(serve(
            Listener::Tcp(listener),
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

        write_event(&mut writer, &Event::Ping(PingData::with_text("hi")))
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

        write_event(&mut writer, &Event::Synthesize(SynthesizeData::new("hi")))
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

    fn args_with_uri(uri: &str) -> Args {
        Args {
            model_path: vec![],
            port: 10200,
            host: "0.0.0.0".into(),
            uri: Some(uri.to_string()),
            cpu: false,
            max_connections: 16,
            tts_cache_dir: None,
            tts_cache_max_size: "500M".into(),
        }
    }

    #[test]
    fn resolve_bind_target_tcp_uri() {
        let args = args_with_uri("tcp://127.0.0.1:9999");
        match resolve_bind_target(&args).unwrap() {
            BindTarget::Tcp(addr) => assert_eq!(addr, "127.0.0.1:9999"),
            #[cfg(unix)]
            BindTarget::Unix(_) => panic!("expected Tcp"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn resolve_bind_target_unix_uri() {
        let args = args_with_uri("unix:///tmp/test.sock");
        match resolve_bind_target(&args).unwrap() {
            BindTarget::Unix(path) => assert_eq!(path, PathBuf::from("/tmp/test.sock")),
            BindTarget::Tcp(_) => panic!("expected Unix"),
        }
    }

    #[test]
    fn resolve_bind_target_invalid_scheme() {
        let args = args_with_uri("http://foo");
        assert!(resolve_bind_target(&args).is_err());
    }

    #[test]
    fn resolve_bind_target_host_port_fallback() {
        let mut args = args_with_uri("tcp://ignored:1");
        args.uri = None;
        args.host = "127.0.0.1".into();
        args.port = 4242;
        match resolve_bind_target(&args).unwrap() {
            BindTarget::Tcp(addr) => assert_eq!(addr, "127.0.0.1:4242"),
            #[cfg(unix)]
            BindTarget::Unix(_) => panic!("expected Tcp"),
        }
    }

    #[test]
    fn resolve_bind_target_ipv6_brackets() {
        let mut args = args_with_uri("tcp://ignored:1");
        args.uri = None;
        args.host = "::1".into();
        args.port = 4242;
        match resolve_bind_target(&args).unwrap() {
            BindTarget::Tcp(addr) => assert_eq!(addr, "[::1]:4242"),
            #[cfg(unix)]
            BindTarget::Unix(_) => panic!("expected Tcp"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_stale_socket_cleanup() {
        let socket_path = std::env::temp_dir().join(format!(
            "crane-wyoming-test-stale-{}.sock",
            std::process::id()
        ));
        // Simulate a stale socket file left behind by an uncleanly-terminated
        // previous run: a regular file, not an actual bound socket.
        std::fs::write(&socket_path, b"").unwrap();

        let listener = Listener::bind(BindTarget::Unix(socket_path.clone())).await;
        assert!(
            listener.is_ok(),
            "bind should clean up the stale file and retry"
        );
        drop(listener);
        assert!(!socket_path.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_listener_drop_removes_socket() {
        let socket_path = std::env::temp_dir().join(format!(
            "crane-wyoming-test-drop-{}.sock",
            std::process::id()
        ));
        let listener = Listener::bind(BindTarget::Unix(socket_path.clone()))
            .await
            .unwrap();
        assert!(socket_path.exists());
        drop(listener);
        assert!(!socket_path.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_drop_preserves_foreign_socket() {
        let socket_path = std::env::temp_dir().join(format!(
            "crane-wyoming-test-foreign-{}.sock",
            std::process::id()
        ));
        let first = Listener::bind(BindTarget::Unix(socket_path.clone()))
            .await
            .unwrap();
        // Simulate a second instance replacing the socket file at the same
        // path while `first` (e.g. a slow-shutting-down stale process) is
        // still holding it open.
        std::fs::remove_file(&socket_path).unwrap();
        let second = Listener::bind(BindTarget::Unix(socket_path.clone()))
            .await
            .unwrap();

        drop(first);
        assert!(
            socket_path.exists(),
            "dropping the stale listener must not remove the second listener's socket"
        );

        drop(second);
        assert!(!socket_path.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_ping_pong() {
        let socket_path =
            std::env::temp_dir().join(format!("crane-wyoming-test-{}.sock", std::process::id()));
        let listener = Listener::bind(BindTarget::Unix(socket_path.clone()))
            .await
            .unwrap();

        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16));
        tokio::spawn(serve(listener, Arc::new(rt), Arc::new(vm), semaphore));

        let stream = tokio::net::UnixStream::connect(&socket_path).await.unwrap();
        let (reader, mut writer) = stream.into_split();
        let mut reader = tokio::io::BufReader::new(reader);

        write_event(&mut writer, &Event::Ping(PingData::with_text("hi")))
            .await
            .unwrap();

        let response = read_event(&mut reader).await.unwrap().unwrap();
        match response {
            Event::Pong(data) => assert_eq!(data.text.as_deref(), Some("hi")),
            other => panic!("expected Pong, got {other:?}"),
        }
    }
}
