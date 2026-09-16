// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 Andreas Schneider <asn@cryptomilk.org>

//! Wyoming protocol TTS/ASR server for Home Assistant voice integration.
//!
//! # Module layout
//!
//! | Module    | Responsibility                                         |
//! |-----------|----------------------------------------------------------|
//! | `engine`  | `ModelRuntime` -- owns and dispatches to TTS/ASR models |
//! | `handler` | TTS/ASR event handling, dispatching to `ModelRuntime`   |
//!
//! The wire protocol types (`Event`, `read_event`, `write_event`) are
//! implemented in and re-exported from the [`wyoming_protocol`] crate.

pub mod engine;
pub mod handler;

pub use handler::{AsrModelMap, VoiceMap, handle_connection};
pub use wyoming_protocol::error::ProtocolError;
pub use wyoming_protocol::event::Event;
pub use wyoming_protocol::wire::{
    MAX_DATA_LENGTH, MAX_HEADER_LINE, MAX_PAYLOAD_LENGTH, read_event, write_event,
};

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{info, warn};

use crate::engine::ModelRuntime;
use crate::engine::model_factory::{DiscoveredModel, DiscoveredVadModel};

/// Command-line arguments for the Wyoming protocol TTS/ASR server.
#[derive(Parser, Debug, Clone)]
#[command(about = "Wyoming protocol TTS/ASR server for Home Assistant voice integration")]
pub struct Args {
    /// Parent directory containing `tts/`, `asr/`, and/or `vad/`
    /// subdirectories. `tts/` and `asr/` each hold model subdirectories for
    /// that family; `vad/` holds a single Silero VAD model directory
    /// containing `model.onnx`. All recognized models found as immediate
    /// subdirectories of `tts/` are loaded, unless restricted with
    /// `--model-tts`. The first loaded TTS model (alphabetically, or by
    /// `--model-tts` order if given) becomes the default voice when a
    /// client does not request one by name. A missing `tts/`, `asr/`, or
    /// `vad/` subdirectory means zero models of that family, not an error.
    #[arg(
        short = 'm',
        long = "model-path",
        required = true,
        env = "CRANE_WYOMING_MODEL_PATH"
    )]
    pub model_path: PathBuf,

    /// Load only the named model subdirectories from `<model-path>/tts`
    /// (directory name, not full path). Repeatable, or separate multiple
    /// names with `;`; order determines voice priority (first = default).
    /// When omitted, all recognized TTS models are loaded alphabetically.
    #[arg(
        long = "model-tts",
        env = "CRANE_WYOMING_MODEL_TTS",
        value_delimiter = ';'
    )]
    pub model_tts: Vec<String>,

    /// Load only the named model subdirectories from `<model-path>/asr`
    /// (directory name, not full path). Repeatable, or separate multiple
    /// names with `;`; order determines default-model priority (first =
    /// default). When omitted, all recognized ASR models are loaded
    /// alphabetically.
    #[arg(
        long = "model-asr",
        env = "CRANE_WYOMING_MODEL_ASR",
        value_delimiter = ';'
    )]
    pub model_asr: Vec<String>,

    /// List the models recognized under `--model-path` and exit, without
    /// starting the server.
    #[arg(long = "list-models")]
    pub list_models: bool,

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

    /// Disable VAD pre-filtering on ASR input, even when a VAD model is
    /// loaded under `<model-path>/vad/`. Has no effect if no VAD model was
    /// found.
    #[arg(long, env = "CRANE_WYOMING_NO_VAD_FILTER")]
    pub no_vad_filter: bool,

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

/// The fd systemd passes the first (and, for us, only) listening socket on,
/// per the `sd_listen_fds` protocol.
#[cfg(unix)]
const SD_LISTEN_FDS_START: std::os::unix::io::RawFd = 3;

/// Check whether systemd has passed us a pre-bound listening socket via
/// socket activation (`LISTEN_PID`/`LISTEN_FDS` env vars), returning its fd.
///
/// Returns `None` if `LISTEN_PID` doesn't match this process, or `LISTEN_FDS`
/// isn't exactly `1` (we only support a single activated socket). On success,
/// clears both env vars per the `sd_listen_fds` protocol, so a subprocess we
/// spawn later doesn't also try to claim the fd.
#[cfg(unix)]
fn systemd_listen_fd() -> Option<std::os::unix::io::RawFd> {
    let listen_pid: u32 = std::env::var("LISTEN_PID").ok()?.parse().ok()?;
    if listen_pid != std::process::id() {
        return None;
    }
    let listen_fds: u32 = std::env::var("LISTEN_FDS").ok()?.parse().ok()?;
    if listen_fds != 1 {
        warn!(
            listen_fds,
            "Ignoring systemd socket activation: expected exactly 1 LISTEN_FDS"
        );
        return None;
    }

    // SAFETY: `run()` calls this before loading any TTS models, which is
    // the first point it spawns other threads (one worker thread per
    // model). No other thread exists yet to race on these variables.
    unsafe {
        std::env::remove_var("LISTEN_PID");
        std::env::remove_var("LISTEN_FDS");
    }
    Some(SD_LISTEN_FDS_START)
}

/// A bound listener, either TCP or (on Unix platforms) a Unix domain socket.
enum Listener {
    /// TCP listener.
    Tcp(tokio::net::TcpListener),
    /// Unix domain socket listener, with its bound path and inode identity
    /// (for cleanup on drop).
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf, SocketInode),
    /// Unix domain socket listener passed to us by systemd via socket
    /// activation. Unlike `Unix`, the socket file is owned and managed by
    /// systemd, not this process, so it must not be unlinked on drop.
    #[cfg(unix)]
    Systemd(tokio::net::UnixListener),
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

    /// Wrap a systemd-provided listening socket fd (from [`systemd_listen_fd`])
    /// into a [`Listener::Systemd`].
    ///
    /// # Safety
    ///
    /// `fd` must be a valid, open listening Unix domain socket file
    /// descriptor owned by this process (i.e. one obtained from
    /// `systemd_listen_fd`), not already wrapped by another owner.
    ///
    /// # Errors
    ///
    /// Returns an error if the fd cannot be set non-blocking or registered
    /// with the tokio runtime.
    #[cfg(unix)]
    unsafe fn from_systemd_fd(fd: std::os::unix::io::RawFd) -> Result<Self> {
        use std::os::unix::io::FromRawFd;

        // SAFETY: Caller guarantees `fd` is a valid, owned listening socket.
        let std_listener = unsafe { std::os::unix::net::UnixListener::from_raw_fd(fd) };
        std_listener.set_nonblocking(true)?;
        Ok(Self::Systemd(tokio::net::UnixListener::from_std(
            std_listener,
        )?))
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
            Self::Unix(listener, ..) | Self::Systemd(listener) => {
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
            #[cfg(unix)]
            Self::Systemd(listener) => {
                let path = listener
                    .local_addr()
                    .ok()
                    .and_then(|a| a.as_pathname().map(|p| p.display().to_string()));
                match path {
                    Some(p) => format!("unix://{p} (systemd)"),
                    None => "unix://systemd-socket (systemd)".to_string(),
                }
            },
        }
    }
}

impl Drop for Listener {
    /// Remove the socket file on shutdown so it doesn't linger as a stale
    /// socket for the next run -- but only if the file at `path` is still
    /// the same socket this listener bound (identified by device+inode),
    /// so a socket that another process has since rebound at the same path
    /// is left alone. `Systemd` is deliberately excluded: that socket file
    /// is owned and managed by systemd, not this process.
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
/// [`handle_connection`]'s event loop against `runtime`, `voice_map`, and
/// `asr_map`.
async fn serve_connection(
    stream: Stream,
    runtime: &ModelRuntime,
    voice_map: &VoiceMap,
    asr_map: &AsrModelMap,
) -> Result<()> {
    /// Split a stream into buffered halves and run [`handle_connection`].
    /// A macro (rather than a generic helper) because `TcpStream::into_split`
    /// and `UnixStream::into_split` return unrelated concrete types.
    macro_rules! split_and_handle {
        ($stream:expr) => {{
            let (reader, writer) = $stream.into_split();
            let mut reader = tokio::io::BufReader::new(reader);
            let mut writer = tokio::io::BufWriter::new(writer);
            handle_connection(&mut reader, &mut writer, runtime, voice_map, asr_map).await
        }};
    }

    match stream {
        Stream::Tcp(stream) => split_and_handle!(stream),
        #[cfg(unix)]
        Stream::Unix(stream) => split_and_handle!(stream),
    }
}

/// Directory name, under `--model-path`, holding TTS model subdirectories.
const TTS_SUBDIR: &str = "tts";
/// Directory name, under `--model-path`, holding ASR model subdirectories.
const ASR_SUBDIR: &str = "asr";
/// Directory name, under `--model-path`, holding a VAD model subdirectory.
const VAD_SUBDIR: &str = "vad";

/// Scan `dir` for models with `discover`, treating a *missing* `dir` as
/// "zero models" rather than an error.
///
/// `--model-path` is a directory containing `tts/`/`asr/` subdirectories;
/// either one may legitimately be absent (e.g. an ASR-only deployment has
/// no `tts/` subdirectory at all), which shouldn't fail the scan. Other
/// failures to access `dir` (e.g. permission denied, or `dir` existing as a
/// non-directory) are real problems and are propagated rather than
/// silently treated as "zero models".
///
/// # Errors
///
/// Returns an error if `dir` exists but cannot be scanned for models.
fn discover_or_empty(
    dir: &std::path::Path,
    discover: impl Fn(&std::path::Path) -> Result<Vec<DiscoveredModel>>,
) -> Result<Vec<DiscoveredModel>> {
    match std::fs::metadata(dir) {
        Ok(_) => discover(dir),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e).with_context(|| format!("cannot access '{}'", dir.display())),
    }
}

/// Scan `dir` for a VAD model with [`discover_vad_model`], treating a
/// *missing* `dir` as "no VAD model" rather than an error. See
/// [`discover_or_empty`], which does the same for TTS/ASR.
///
/// # Errors
///
/// Returns an error if `dir` exists but cannot be scanned for a model.
fn discover_vad_or_none(dir: &std::path::Path) -> Result<Option<DiscoveredVadModel>> {
    match std::fs::metadata(dir) {
        Ok(_) => engine::model_factory::discover_vad_model(dir),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("cannot access '{}'", dir.display())),
    }
}

/// Print the TTS and ASR models found under `<model_path>/tts` and
/// `<model_path>/asr`, for `--list-models`.
///
/// # Errors
///
/// Returns an error if a `tts`/`asr` subdirectory exists but cannot be read.
fn print_discovered_models(model_path: &std::path::Path) -> Result<()> {
    let tts = discover_or_empty(
        &model_path.join(TTS_SUBDIR),
        engine::model_factory::discover_models,
    )?;
    let asr = discover_or_empty(
        &model_path.join(ASR_SUBDIR),
        engine::model_factory::discover_asr_models,
    )?;
    let vad = discover_vad_or_none(&model_path.join(VAD_SUBDIR))?;

    if tts.is_empty() && asr.is_empty() && vad.is_none() {
        println!("No supported models found in '{}'", model_path.display());
        return Ok(());
    }

    print_model_group("TTS models:", &tts);
    println!();
    print_model_group("ASR models:", &asr);
    println!();
    println!("VAD model:");
    match &vad {
        Some(v) => println!("  {}", v.name),
        None => println!("  (none)"),
    }

    Ok(())
}

/// Print one `--list-models` section: a header followed by one line per
/// model, or `(none)` if `models` is empty.
fn print_model_group(header: &str, models: &[DiscoveredModel]) {
    println!("{header}");
    for m in models {
        println!("  {}  ({})", m.name, m.model_type.display_name());
    }
    if models.is_empty() {
        println!("  (none)");
    }
}

/// Load `vad_model`'s `model.onnx` into `runtime` for ASR pre-filtering,
/// unless `no_vad_filter` is set. Returns whether a VAD model ended up
/// loaded (`false` if `vad_model` is `None` or `no_vad_filter` is set).
///
/// VAD always runs on CPU: the Silero model is tiny (~2 MB) and fast
/// enough that CPU inference adds negligible latency, avoiding competing
/// for GPU memory with TTS/ASR models.
///
/// # Errors
///
/// Returns an error if the ONNX model file fails to load.
fn load_vad_model(
    runtime: &mut ModelRuntime,
    vad_model: Option<&DiscoveredVadModel>,
    no_vad_filter: bool,
) -> Result<bool> {
    let Some(vad_model) = vad_model else {
        return Ok(false);
    };
    if no_vad_filter {
        info!(
            name = %vad_model.name,
            "VAD model found but --no-vad-filter is set; skipping",
        );
        return Ok(false);
    }

    let onnx_path = vad_model.path.join("model.onnx");
    let onnx_str = onnx_path.to_string_lossy();
    let mut vad =
        crane_core::models::silero_vad::Vad::new(crane_core::models::silero_vad::VadConfig {
            use_cpu: true,
            ..crane_core::models::silero_vad::VadConfig::default()
        })?;
    vad.load(&*onnx_str)
        .with_context(|| format!("loading VAD model '{onnx_str}'"))?;
    runtime.set_vad(vad);
    info!(name = %vad_model.name, path = %onnx_str, "VAD model loaded");
    Ok(true)
}

/// Resolve the compute device and floating-point dtype to load models with,
/// from `--cpu` and (on CUDA builds) auto-detected GPU availability.
///
/// # Errors
///
/// Returns an error if CUDA device initialization fails (CUDA builds only).
#[cfg_attr(not(feature = "cuda"), allow(clippy::unnecessary_wraps))]
fn resolve_device_and_dtype(
    args: &Args,
) -> Result<(crane_core::models::Device, crane_core::models::DType)> {
    let device = if args.cpu {
        crane_core::models::Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            // new_cuda_with_stream, NOT cuda_if_available/new_cuda: candle's
            // default CudaDevice::new hands out per-thread default streams
            // (cudaStreamPerThread) while cudarc's cross-stream event guards
            // stay disabled (num_streams == 0), so tensors read from another
            // thread (the tokio handlers) race with in-flight kernels and
            // come back as garbled audio. An explicit stream activates the
            // guards and serializes all device work on one stream.
            crane_core::models::Device::new_cuda_with_stream(0)?
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

    Ok((device, dtype))
}

/// Run the Wyoming protocol TTS/ASR server.
///
/// Discovers TTS models under `<model-path>/tts` (optionally restricted by
/// `--model-tts`), ASR models under `<model-path>/asr` (optionally
/// restricted by `--model-asr`), and an optional VAD model under
/// `<model-path>/vad` (used for ASR pre-filtering unless `--no-vad-filter`
/// is set). If `--list-models` is set, prints the discovered models and
/// returns without starting the server. Otherwise loads them into a
/// shared [`ModelRuntime`], optionally enables the on-disk TTS cache, then
/// accepts connections on the address resolved from `--uri` (or
/// `args.host`:`args.port` if unset) -- either TCP or a Unix domain
/// socket. Each connection is handled on its own tokio task via
/// [`handle_connection`]; TTS/ASR requests from all connections queue on
/// the target model's dedicated thread and are processed one at a time.
///
/// # Errors
///
/// Returns an error if `--model-path` doesn't exist or isn't a directory,
/// no supported TTS or ASR models are found under `<model-path>/tts` or
/// `<model-path>/asr`, a named `--model-tts`/`--model-asr` doesn't match a
/// discovered model or is repeated, a model fails to load, or the listener
/// cannot bind.
pub async fn run(args: Args) -> Result<()> {
    if args.list_models {
        print_discovered_models(&args.model_path)?;
        return Ok(());
    }

    match std::fs::metadata(&args.model_path) {
        Ok(meta) if meta.is_dir() => {},
        Ok(_) => anyhow::bail!(
            "model path '{}' is not a directory",
            args.model_path.display()
        ),
        Err(e) => {
            return Err(e).with_context(|| format!("model path '{}'", args.model_path.display()));
        },
    }

    let tts_dir = args.model_path.join(TTS_SUBDIR);
    let tts_discovered = discover_or_empty(&tts_dir, engine::model_factory::discover_models)?;

    let asr_dir = args.model_path.join(ASR_SUBDIR);
    let asr_discovered = discover_or_empty(&asr_dir, engine::model_factory::discover_asr_models)?;

    let vad_dir = args.model_path.join(VAD_SUBDIR);
    let vad_discovered = discover_vad_or_none(&vad_dir)?;

    if tts_discovered.is_empty() && asr_discovered.is_empty() {
        anyhow::bail!(
            "no supported TTS or ASR models found in '{}'",
            args.model_path.display()
        );
    }

    let (device, dtype) = resolve_device_and_dtype(&args)?;

    // Incremental streaming is only worth it on a GPU: both CUDA and Metal
    // have far more memory bandwidth than CPU DDR, which is what
    // autoregressive TTS generation is bottlenecked on. On CPU, chunks
    // arrive slower than they play back, producing bursty audio with dead
    // air -- worse than just waiting once for the full clip.
    let streaming_enabled = device.is_cuda() || device.is_metal();

    let device_name = format!("{device:?}");
    let dtype_name = format!("{dtype:?}");
    info!("Device: {device_name}, dtype: {dtype_name}, streaming: {streaming_enabled}");

    // When `--model-tts` is given, load exactly those (in the given order,
    // so the first one named becomes the default); otherwise load
    // everything discovered, in alphabetical order.
    let models_to_load: Vec<&DiscoveredModel> =
        engine::model_factory::resolve_models_to_load(&tts_discovered, &args.model_tts)
            .with_context(|| format!("in '{}'", tts_dir.display()))?;

    // Same resolution, for ASR models under `<model-path>/asr`.
    let asr_models_to_load: Vec<&DiscoveredModel> =
        engine::model_factory::resolve_asr_models_to_load(&asr_discovered, &args.model_asr)
            .with_context(|| format!("in '{}'", asr_dir.display()))?;

    // Resolved before model loading so `systemd_listen_fd`'s env-var cleanup
    // (see its doc comment) runs while this process is still single-threaded,
    // i.e. before `load_tts` below spawns per-model worker threads.
    #[cfg(unix)]
    let listener = if let Some(fd) = systemd_listen_fd() {
        info!(fd, "Using systemd-provided listening socket");
        // SAFETY: `systemd_listen_fd` verified `LISTEN_PID`/`LISTEN_FDS`,
        // so `fd` is a listening socket systemd handed to this process.
        unsafe { Listener::from_systemd_fd(fd)? }
    } else {
        Listener::bind(resolve_bind_target(&args)?).await?
    };
    #[cfg(not(unix))]
    let listener = Listener::bind(resolve_bind_target(&args)?).await?;

    let mut runtime = ModelRuntime::new();
    runtime.set_streaming_enabled(streaming_enabled);

    if let Some(cache_dir) = &args.tts_cache_dir {
        let max_bytes = parse_size(&args.tts_cache_max_size)?;
        runtime.set_tts_cache(crate::engine::TtsCache::new(cache_dir.clone(), max_bytes)?);
        info!(dir = %cache_dir.display(), max = %args.tts_cache_max_size, "TTS cache enabled");
    }

    let mut model_names = Vec::with_capacity(models_to_load.len());
    for model in &models_to_load {
        let path_str = model.path.to_string_lossy();
        let name = runtime.load_tts(&path_str, &device, &dtype)?;
        info!(name = %name, path = %path_str, "TTS model loaded");
        model_names.push(name);
    }

    let mut asr_model_names = Vec::with_capacity(asr_models_to_load.len());
    for model in &asr_models_to_load {
        let path_str = model.path.to_string_lossy();
        let name = runtime.load_asr(&path_str, &device, &dtype)?;
        info!(name = %name, path = %path_str, "ASR model loaded");
        asr_model_names.push(name);
    }

    let vad_loaded = load_vad_model(&mut runtime, vad_discovered.as_ref(), args.no_vad_filter)?;

    let voice_map = VoiceMap::new(&model_names, &runtime);
    let asr_map = AsrModelMap::new(&asr_model_names, &runtime);
    let runtime = Arc::new(runtime);
    let voice_map = Arc::new(voice_map);
    let asr_map = Arc::new(asr_map);

    info!(
        version = env!("CARGO_PKG_VERSION"),
        listen = %listener.display_addr(),
        tts_models = model_names.len(),
        asr_models = asr_model_names.len(),
        vad = vad_loaded,
        "crane-wyoming ready"
    );

    let semaphore = Arc::new(tokio::sync::Semaphore::new(args.max_connections));
    serve(listener, runtime, voice_map, asr_map, semaphore).await
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
    asr_map: Arc<AsrModelMap>,
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
        let asr_map = Arc::clone(&asr_map);
        tokio::spawn(async move {
            if let Err(e) = serve_connection(stream, &runtime, &voice_map, &asr_map).await {
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
    use crane::audio::tts::{Tts, VoiceInfo};
    use crane::audio::{AudioInfo, pcm_f32_to_i16};
    use crane_core::generation::SpeechOptions;
    use std::ops::ControlFlow;
    use tokio::net::{TcpListener, TcpStream};
    use wyoming_protocol::client::{Client, ClientError};
    use wyoming_protocol::event::{PingData, SynthesizeData};

    #[test]
    fn discover_or_empty_missing_dir_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("asr");
        let result = discover_or_empty(&missing, engine::model_factory::discover_asr_models);
        assert_eq!(result.unwrap(), vec![]);
    }

    #[test]
    fn discover_or_empty_non_directory_propagates_error() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tts");
        std::fs::write(&file, "not a directory").unwrap();
        let result = discover_or_empty(&file, engine::model_factory::discover_models);
        assert!(result.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn discover_or_empty_permission_denied_propagates_error() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let restricted = dir.path().join("restricted");
        std::fs::create_dir(&restricted).unwrap();
        let target = restricted.join("tts");
        std::fs::create_dir(&target).unwrap();
        // Remove the restricted dir's execute bit so `target` can't be stat'd.
        std::fs::set_permissions(&restricted, std::fs::Permissions::from_mode(0o000)).unwrap();
        // Root ignores permission bits, so skip the assertion in that case
        // rather than failing under a root-run test suite.
        let running_as_root = std::fs::metadata(&target).is_ok();

        let result = discover_or_empty(&target, engine::model_factory::discover_models);

        std::fs::set_permissions(&restricted, std::fs::Permissions::from_mode(0o755)).unwrap();
        if !running_as_root {
            assert!(result.is_err());
        }
    }

    #[test]
    fn discover_or_empty_existing_dir_delegates_to_discover() {
        let dir = tempfile::tempdir().unwrap();
        let tts_dir = dir.path().join("tts");
        std::fs::create_dir(&tts_dir).unwrap();
        std::fs::create_dir(tts_dir.join("model")).unwrap();
        std::fs::write(
            tts_dir.join("model/config.json"),
            r#"{"model_type": "qwen3_tts"}"#,
        )
        .unwrap();

        let result = discover_or_empty(&tts_dir, engine::model_factory::discover_models).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "model");
    }

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

    async fn spawn_test_server(runtime: ModelRuntime, voice_map: VoiceMap) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let asr_map = AsrModelMap::new(&[], &runtime);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16));
        tokio::spawn(serve(
            Listener::Tcp(listener),
            Arc::new(runtime),
            Arc::new(voice_map),
            Arc::new(asr_map),
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
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
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

    #[tokio::test]
    async fn test_client_ping() {
        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let mut client = Client::connect(&format!("tcp://{addr}")).await.unwrap();
        let pong = client.ping(PingData::with_text("hi")).await.unwrap();
        assert_eq!(pong.text.as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn test_client_describe() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let mut client = Client::connect(&format!("tcp://{addr}")).await.unwrap();
        let info = client.describe().await.unwrap();
        assert!(!info.tts.is_empty());
        let voices = info.tts[0].get("voices").unwrap().as_array().unwrap();
        assert!(voices.iter().any(|v| v.get("name").unwrap() == "alice"));
    }

    #[tokio::test]
    async fn test_client_synthesize() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let mut client = Client::connect(&format!("tcp://{addr}")).await.unwrap();
        let response = client.synthesize(SynthesizeData::new("hi")).await.unwrap();
        assert_eq!(response.format.rate, 24000);
        assert_eq!(response.format.width, 2);
        assert_eq!(response.format.channels, 1);
        assert_eq!(response.audio, pcm_f32_to_i16(&[0.5f32; 2]));
    }

    #[tokio::test]
    async fn test_client_synthesize_streaming() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let mut client = Client::connect(&format!("tcp://{addr}")).await.unwrap();
        let mut total_bytes = 0usize;
        let mut chunk_count = 0usize;
        let format = client
            .synthesize_streaming(SynthesizeData::new("hi"), |chunk, _format| {
                total_bytes += chunk.len();
                chunk_count += 1;
                ControlFlow::Continue(())
            })
            .await
            .unwrap();
        assert_eq!(format.rate, 24000);
        assert!(chunk_count > 0);
        assert_eq!(total_bytes, pcm_f32_to_i16(&[0.5f32; 2]).len());
    }

    #[tokio::test]
    async fn test_client_synthesize_error() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let mut client = Client::connect(&format!("tcp://{addr}")).await.unwrap();
        let request = SynthesizeData::new("hi")
            .with_voice(wyoming_protocol::event::SynthesizeVoice::with_name("ghost"));
        let err = client.synthesize(request).await.unwrap_err();
        match err {
            ClientError::ServerError { code, .. } => {
                assert_eq!(code.as_deref(), Some("voice-not-found"));
            },
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_client_synthesize_streaming_break() {
        let mut rt = test_runtime();
        register_test_tts(
            &mut rt,
            "m1",
            "qwen3_tts",
            Box::new(MockTts::new(
                24000,
                vec![VoiceInfo {
                    name: "alice".into(),
                    languages: vec!["en".into()],
                }],
            )),
        );
        let vm = VoiceMap::new(&["m1".to_string()], &rt);
        let addr = spawn_test_server(rt, vm).await;

        let mut client = Client::connect(&format!("tcp://{addr}")).await.unwrap();
        let mut chunk_count = 0usize;
        let result = client
            .synthesize_streaming(SynthesizeData::new("hi"), |_chunk, _format| {
                chunk_count += 1;
                ControlFlow::Break(())
            })
            .await;
        assert!(result.is_ok());
        assert_eq!(chunk_count, 1);

        // The connection must still be usable for a subsequent request.
        let pong = client
            .ping(PingData::with_text("still alive"))
            .await
            .unwrap();
        assert_eq!(pong.text.as_deref(), Some("still alive"));
    }

    fn args_with_uri(uri: &str) -> Args {
        Args {
            model_path: PathBuf::from("/nonexistent"),
            model_tts: vec![],
            model_asr: vec![],
            list_models: false,
            port: 10200,
            host: "0.0.0.0".into(),
            uri: Some(uri.to_string()),
            cpu: false,
            no_vad_filter: false,
            max_connections: 16,
            tts_cache_dir: None,
            tts_cache_max_size: "500M".into(),
        }
    }

    #[tokio::test]
    async fn run_bails_when_model_path_missing() {
        let args = Args {
            model_path: PathBuf::from("/nonexistent/crane-wyoming-test-path"),
            ..args_with_uri("tcp://127.0.0.1:0")
        };
        let err = run(args).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("/nonexistent/crane-wyoming-test-path")
        );
    }

    #[tokio::test]
    async fn run_bails_when_no_models_found() {
        let dir = tempfile::tempdir().unwrap();
        let args = Args {
            model_path: dir.path().to_path_buf(),
            ..args_with_uri("tcp://127.0.0.1:0")
        };
        let err = run(args).await.unwrap_err();
        assert!(err.to_string().contains("no supported TTS or ASR models"));
    }

    #[tokio::test]
    async fn run_allows_asr_only_deployment() {
        // No `tts/` subdirectory at all -- only `asr/` with a discoverable
        // model. `run()` must not hit the "no models found" bail just
        // because TTS is absent.
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("asr").join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(
            model_dir.join("config.json"),
            r#"{"model_type": "qwen3_asr"}"#,
        )
        .unwrap();

        let args = Args {
            model_path: dir.path().to_path_buf(),
            ..args_with_uri("tcp://127.0.0.1:0")
        };
        let err = run(args).await.unwrap_err();
        // Fails later, while actually constructing the ASR model (there
        // are no real model weights in `model_dir`, so parsing the
        // deliberately-incomplete `config.json` above fails on the missing
        // `audio_config` field), not at the "no models found" bail.
        assert!(!err.to_string().contains("no supported TTS or ASR models"));
        assert!(err.to_string().contains("audio_config"));
    }

    #[test]
    fn print_discovered_models_missing_subdirs_ok() {
        let dir = tempfile::tempdir().unwrap();
        assert!(print_discovered_models(dir.path()).is_ok());
    }

    #[test]
    fn print_discovered_models_non_directory_tts_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tts"), "oops").unwrap();
        assert!(print_discovered_models(dir.path()).is_err());
    }

    #[test]
    fn discover_vad_or_none_missing_dir_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(discover_vad_or_none(&dir.path().join("vad")).unwrap(), None);
    }

    #[test]
    fn discover_vad_or_none_finds_model() {
        let dir = tempfile::tempdir().unwrap();
        let vad_dir = dir.path().join("vad").join("silero-vad");
        std::fs::create_dir_all(&vad_dir).unwrap();
        std::fs::write(vad_dir.join("model.onnx"), b"").unwrap();

        let result = discover_vad_or_none(&dir.path().join("vad"))
            .unwrap()
            .unwrap();
        assert_eq!(result.name, "silero-vad");
    }

    #[test]
    fn load_vad_model_no_model_is_noop() {
        let mut rt = test_runtime();
        let loaded = load_vad_model(&mut rt, None, false).unwrap();
        assert!(!loaded);
    }

    #[test]
    fn load_vad_model_skipped_when_no_vad_filter_set() {
        let dir = tempfile::tempdir().unwrap();
        let vad_model = DiscoveredVadModel {
            path: dir.path().to_path_buf(),
            name: "silero-vad".into(),
        };
        let mut rt = test_runtime();
        let loaded = load_vad_model(&mut rt, Some(&vad_model), true).unwrap();
        assert!(!loaded);
    }

    #[test]
    fn load_vad_model_invalid_onnx_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"not a real onnx file").unwrap();
        let vad_model = DiscoveredVadModel {
            path: dir.path().to_path_buf(),
            name: "silero-vad".into(),
        };
        let mut rt = test_runtime();
        let err = load_vad_model(&mut rt, Some(&vad_model), false).unwrap_err();
        assert!(err.to_string().contains("loading VAD model"));
    }

    #[test]
    fn print_discovered_models_lists_vad_model() {
        let dir = tempfile::tempdir().unwrap();
        let vad_dir = dir.path().join("vad").join("silero-vad");
        std::fs::create_dir_all(&vad_dir).unwrap();
        std::fs::write(vad_dir.join("model.onnx"), b"").unwrap();
        assert!(print_discovered_models(dir.path()).is_ok());
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
        let am = AsrModelMap::new(&[], &rt);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16));
        tokio::spawn(serve(
            listener,
            Arc::new(rt),
            Arc::new(vm),
            Arc::new(am),
            semaphore,
        ));

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

    // Guards tests that mutate the process-global LISTEN_PID/LISTEN_FDS env
    // vars, since `cargo test` runs `#[test]` functions concurrently.
    #[cfg(unix)]
    static SYSTEMD_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(unix)]
    #[test]
    fn systemd_listen_fd_detection() {
        let _guard = SYSTEMD_ENV_LOCK.lock().unwrap();

        // No LISTEN_PID/LISTEN_FDS set: not activated by systemd.
        assert!(systemd_listen_fd().is_none());

        // A LISTEN_PID that doesn't match this process must be ignored.
        // SAFETY: guarded by SYSTEMD_ENV_LOCK; no other test reads these vars concurrently.
        unsafe {
            std::env::set_var("LISTEN_PID", "4294967295");
            std::env::set_var("LISTEN_FDS", "1");
        }
        assert!(systemd_listen_fd().is_none());

        // SAFETY: guarded by SYSTEMD_ENV_LOCK; no other test reads these vars concurrently.
        unsafe {
            std::env::remove_var("LISTEN_PID");
            std::env::remove_var("LISTEN_FDS");
        }
    }

    #[cfg(unix)]
    #[test]
    fn systemd_listen_fd_detection_matching_pid() {
        let _guard = SYSTEMD_ENV_LOCK.lock().unwrap();

        // A matching LISTEN_PID and LISTEN_FDS=1 must be detected, and both
        // env vars cleared afterward per the sd_listen_fds protocol.
        // SAFETY: guarded by SYSTEMD_ENV_LOCK; no other test reads these vars concurrently.
        unsafe {
            std::env::set_var("LISTEN_PID", std::process::id().to_string());
            std::env::set_var("LISTEN_FDS", "1");
        }
        assert_eq!(systemd_listen_fd(), Some(SD_LISTEN_FDS_START));
        assert!(std::env::var("LISTEN_PID").is_err());
        assert!(std::env::var("LISTEN_FDS").is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn systemd_listener_from_fd_does_not_unlink_on_drop() {
        use std::os::unix::io::IntoRawFd;

        let socket_path = std::env::temp_dir().join(format!(
            "crane-wyoming-test-systemd-{}.sock",
            std::process::id()
        ));
        let std_listener = std::os::unix::net::UnixListener::bind(&socket_path).unwrap();
        let fd = std_listener.into_raw_fd();

        // SAFETY: `fd` is a listening socket we just created and own.
        let listener = unsafe { Listener::from_systemd_fd(fd).unwrap() };
        assert!(listener.display_addr().contains("systemd"));

        drop(listener);
        assert!(
            socket_path.exists(),
            "dropping a Systemd listener must not unlink the socket file"
        );
        std::fs::remove_file(&socket_path).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn systemd_listener_ping_pong() {
        use std::os::unix::io::IntoRawFd;

        let socket_path = std::env::temp_dir().join(format!(
            "crane-wyoming-test-systemd-accept-{}.sock",
            std::process::id()
        ));
        let std_listener = std::os::unix::net::UnixListener::bind(&socket_path).unwrap();
        let fd = std_listener.into_raw_fd();

        // SAFETY: `fd` is a listening socket we just created and own.
        let listener = unsafe { Listener::from_systemd_fd(fd).unwrap() };

        let rt = test_runtime();
        let vm = VoiceMap::new(&[], &rt);
        let am = AsrModelMap::new(&[], &rt);
        let semaphore = Arc::new(tokio::sync::Semaphore::new(16));
        tokio::spawn(serve(
            listener,
            Arc::new(rt),
            Arc::new(vm),
            Arc::new(am),
            semaphore,
        ));

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

        std::fs::remove_file(&socket_path).unwrap();
    }
}
