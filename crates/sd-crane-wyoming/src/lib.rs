//! speech-dispatcher output module for crane-wyoming.
//!
//! speechd launches this binary as a subprocess and talks to it over
//! stdin/stdout using speechd's own line-oriented, SMTP-style module
//! protocol. Internally it is a Wyoming client to a running
//! `crane-wyoming` server. `INIT` connects, calls `describe`, and caches
//! the resulting voice list; `SET` stores per-utterance settings for a
//! later `SPEAK`; `AUDIO` accepts server-side audio output; `SPEAK`
//! synthesizes text and streams it back as `705 AUDIO` events;
//! `LIST VOICES` answers from the cached list; `STOP`/`PAUSE` abort
//! in-flight synthesis; `QUIT` disconnects; anything else is rejected
//! as unknown.

use std::io::{self, BufRead, Write};
use std::ops::ControlFlow;
use std::os::fd::BorrowedFd;

use anyhow::Context;
use wyoming_protocol::event::{InfoData, SynthesizeData, SynthesizeVoice};
use wyoming_protocol::{AudioFormat, Client, ClientError};

/// A voice offered by the connected Wyoming server, as reported by
/// `describe` and cached at `INIT` time for `LIST VOICES`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Voice {
    /// Voice name, passed back to the server as `synthesis_voice` on
    /// `SPEAK`.
    name: String,
    /// Voice language, or `"none"` if the server didn't report one.
    language: String,
}

/// Extracts the flat voice list from a `describe` response.
///
/// Walks `info.tts[*].voices[*]`, pulling `name` and the first entry of
/// `languages` (falling back to `"none"` if absent or empty). Entries
/// missing a `name` are skipped.
fn extract_voices(info: &InfoData) -> Vec<Voice> {
    let mut voices = Vec::new();
    for program in &info.tts {
        let Some(program_voices) = program.get("voices").and_then(|v| v.as_array()) else {
            continue;
        };
        for voice in program_voices {
            let Some(name) = voice.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            let language = voice
                .get("languages")
                .and_then(|langs| langs.as_array())
                .and_then(|langs| langs.first())
                .and_then(|lang| lang.as_str())
                .unwrap_or("none");
            voices.push(Voice {
                name: name.to_string(),
                language: language.to_string(),
            });
        }
    }
    voices
}

/// The `CraneURI` directive is present but its value isn't a quoted string.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("CraneURI directive value must be a quoted string")]
struct CraneUriError;

/// Parses the `CraneURI "..."` directive from a speechd module config file.
///
/// speechd passes this module's config file path as `argv[1]`; the only
/// directive currently understood is `CraneURI`, the Wyoming server URI
/// to connect to.
///
/// Returns `None` if no `CraneURI` line is present, or
/// `Some(Err(CraneUriError))` if one is present but malformed, so callers
/// can tell the two cases apart.
fn parse_crane_uri(config: &str) -> Option<Result<String, CraneUriError>> {
    for line in config.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("CraneURI") else {
            continue;
        };
        if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
            // e.g. "CraneURITimeout" — not actually the CraneURI directive.
            continue;
        }
        let value = rest
            .trim()
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .map(str::to_string)
            .ok_or(CraneUriError);
        return Some(value);
    }
    None
}

/// Writes a speechd module reply and flushes it immediately.
///
/// speechd reads replies from a blocking pipe a line at a time, so a
/// buffered write it hasn't seen yet would stall the handshake.
fn write_reply(output: &mut impl Write, reply: &str) -> io::Result<()> {
    output.write_all(reply.as_bytes())?;
    output.flush()
}

/// HDLC-escapes PCM bytes for speechd's `705 AUDIO` wire format.
///
/// The `705 AUDIO` line is terminated by `\n`, so any `\n` byte in the
/// raw PCM data must not appear literally; the escape byte itself
/// (`0x7d`) needs the same treatment so decoding is unambiguous. Each
/// occurrence of `0x7d` or `0x0a` is replaced with `0x7d` followed by
/// the byte XOR'd with `0x20`, matching speechd's own
/// `module_tts_output_send_server` in `module_process.c`.
fn hdlc_escape(pcm: &[u8]) -> Vec<u8> {
    let mut escaped = Vec::with_capacity(pcm.len());
    for &byte in pcm {
        if byte == 0x7d || byte == 0x0a {
            escaped.push(0x7d);
            escaped.push(byte ^ 0x20);
        } else {
            escaped.push(byte);
        }
    }
    escaped
}

/// Maximum PCM bytes carried by a single `705 AUDIO` event, matching
/// speechd's own `MAX_CHUNK` in `module_process.c` — large enough for
/// efficient transfer, small enough to stay reactive to `STOP`.
const MAX_CHUNK: usize = 10_000;

/// Writes one `705 AUDIO` event carrying `pcm` (at most [`MAX_CHUNK`]
/// bytes; callers split larger buffers before calling this).
///
/// `format` supplies the header fields; `pcm` is HDLC-escaped and
/// framed as `705-AUDIO\0{escaped}\n705 AUDIO\n`, the wire format
/// speechd's server (`output.c`) expects for server-side audio output.
fn write_audio_event(output: &mut impl Write, format: AudioFormat, pcm: &[u8]) -> io::Result<()> {
    let sample_size = usize::from(format.channels) * usize::from(format.width);
    debug_assert!(sample_size != 0, "zero-width or zero-channel audio format");
    debug_assert!(
        pcm.len().is_multiple_of(sample_size),
        "audio chunk not sample-aligned"
    );
    let num_samples = pcm.len() / sample_size;
    writeln!(output, "705-bits={}", u32::from(format.width) * 8)?;
    writeln!(output, "705-num_channels={}", format.channels)?;
    writeln!(output, "705-sample_rate={}", format.rate)?;
    writeln!(output, "705-num_samples={num_samples}")?;
    writeln!(output, "705-big_endian=0")?;
    output.write_all(b"705-AUDIO\0")?;
    output.write_all(&hdlc_escape(pcm))?;
    output.write_all(b"\n705 AUDIO\n")?;
    output.flush()
}

/// Reads a `SPEAK` message body: lines up to a lone `.` terminator.
///
/// Lines starting with `.` have that leading dot stripped
/// (SMTP-style dot-stuffing, so a line of literal text can start with
/// `.` without being mistaken for the terminator) before being joined
/// with `\n`. Returns `Ok(None)` if the input ends before the
/// terminator is seen.
fn read_text_block(
    lines: &mut impl Iterator<Item = io::Result<String>>,
) -> io::Result<Option<String>> {
    let mut text = String::new();
    let mut first = true;
    for line in lines {
        let line = line?;
        if line.trim() == "." {
            return Ok(Some(text));
        }
        if !first {
            text.push('\n');
        }
        first = false;
        text.push_str(line.strip_prefix('.').unwrap_or(&line));
    }
    Ok(None)
}

/// Connects to the Wyoming server at `uri`, confirms it responds to
/// `describe`, and returns its voice list for `LIST VOICES` to serve
/// later.
async fn init(uri: &str) -> Result<(Client, Vec<Voice>), ClientError> {
    let mut client = Client::connect(uri).await?;
    let info = client.describe().await?;
    Ok((client, extract_voices(&info)))
}

/// A speechd module command line.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// `INIT` — connect to the Wyoming server and confirm it responds.
    Init,
    /// `SET` — receive a multiline block of `key=value` settings.
    Set,
    /// `AUDIO` — receive a multiline block of audio output settings.
    Audio,
    /// `LOGLEVEL` — receive a multiline block setting the module's log
    /// level.
    LogLevel,
    /// `SPEAK` — receive text and synthesize it.
    Speak,
    /// `LIST VOICES` — report the voice list cached at `INIT` time.
    ListVoices,
    /// `STOP` — abort in-flight synthesis.
    Stop,
    /// `PAUSE` — abort in-flight synthesis (same handling as `STOP`).
    Pause,
    /// `QUIT` — disconnect and terminate the module.
    Quit,
    /// Any other command, not yet supported.
    Unknown,
}

impl Command {
    /// Parses a single command line received from speechd.
    fn parse(line: &str) -> Self {
        match line.trim() {
            "INIT" => Self::Init,
            "SET" => Self::Set,
            "AUDIO" => Self::Audio,
            "LOGLEVEL" => Self::LogLevel,
            "SPEAK" => Self::Speak,
            "LIST VOICES" => Self::ListVoices,
            "STOP" => Self::Stop,
            "PAUSE" => Self::Pause,
            "QUIT" => Self::Quit,
            _ => Self::Unknown,
        }
    }
}

/// A mid-synthesis interrupt requested by speechd, detected by polling
/// stdin between audio chunks in [`speak`].
///
/// Neural TTS here has no mid-utterance resume, so `STOP` and `PAUSE`
/// abort synthesis identically — only the speechd reply code differs
/// (`703 STOP` vs. `704 PAUSE`), since speechd's server treats them
/// differently for queue bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Interrupt {
    /// speechd sent `STOP`.
    Stop,
    /// speechd sent `PAUSE`.
    Pause,
}

impl Interrupt {
    /// The speechd reply line for this interrupt.
    const fn reply(self) -> &'static str {
        match self {
            Self::Stop => "703 STOP\n",
            Self::Pause => "704 PAUSE\n",
        }
    }
}

/// Checks `fd` for a pending, complete `STOP`/`PAUSE` line without
/// blocking, consuming it if present.
///
/// Used to poll stdin for an interrupt between audio chunks during
/// `SPEAK`, while the main command loop is blocked driving synthesis.
/// The main loop's `BufRead` line iterator is dormant for the whole
/// `SPEAK` — nothing calls `lines.next()` again until `speak` returns —
/// so its internal buffer can't already hold bytes this function needs;
/// bypassing it and reading `fd` directly here is therefore safe, not
/// just convenient. speechd always writes `STOP`/`PAUSE` as a single,
/// separate pipe write (well under `PIPE_BUF`), so a byte-at-a-time read
/// here can't observe a partial line from a still-in-flight write. Any
/// other text pending on `fd` (there shouldn't be any — speechd doesn't
/// pipeline commands) is silently consumed and ignored. The read is
/// bounded by the byte count `ioctl_fionread` reports up front, so a
/// buggy peer that sends data without a trailing newline can't block
/// this function waiting for one.
fn poll_fd_interrupt(fd: BorrowedFd<'_>) -> Option<Interrupt> {
    let avail = rustix::io::ioctl_fionread(fd).unwrap_or(0);
    if avail == 0 {
        return None;
    }
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    let mut remaining = avail;
    loop {
        if remaining == 0 {
            return None;
        }
        match rustix::io::read(fd, &mut byte[..]) {
            Ok(1) if byte[0] == b'\n' => break,
            Ok(1) => {
                line.push(byte[0]);
                remaining -= 1;
            },
            Err(rustix::io::Errno::INTR) => {},
            _ => return None,
        }
    }
    match String::from_utf8_lossy(&line).trim() {
        "STOP" => Some(Interrupt::Stop),
        "PAUSE" => Some(Interrupt::Pause),
        _ => None,
    }
}

/// Per-utterance settings sent by speechd's `SET` command.
///
/// Populated from the `key=value` block preceding each `SPEAK`; consumed
/// by a later step when `SPEAK` is implemented.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Settings {
    /// Generic voice type (`male1`, `female1`, ...), speechd's own
    /// vocabulary.
    voice: Option<String>,
    /// The actual server-side voice name to request from Wyoming.
    synthesis_voice: Option<String>,
    /// Language code, or `None` for the server's default.
    language: Option<String>,
    /// Speech rate. speechd's range is -100 to 100; not validated here.
    rate: Option<i32>,
    /// Pitch. speechd's range is -100 to 100; not validated here.
    pitch: Option<i32>,
    /// Volume. speechd's range is -100 to 100; not validated here.
    volume: Option<i32>,
}

impl Settings {
    /// Applies one `key=value` pair from a `SET` block.
    ///
    /// A value of `"NULL"` clears the field. Returns `false` if `key`
    /// isn't a recognized setting, or if `value` isn't a valid integer
    /// for an integer-valued key — either case should be reported to
    /// speechd as `303 ERROR INVALID PARAMETER OR VALUE`.
    ///
    /// speechd's server always sends `pitch_range`, `punctuation_mode`,
    /// `spelling_mode`, and `cap_let_recogn` alongside the fields this
    /// module actually uses (see `output_send_settings()` in speechd's
    /// `output.c`); they're accepted and ignored rather than rejected,
    /// since crane-wyoming's neural TTS has no equivalent behavior to
    /// configure for them, but a real `SET` block always includes them.
    #[must_use]
    fn apply(&mut self, key: &str, value: &str) -> bool {
        match key {
            "voice" => self.voice = (value != "NULL").then(|| value.to_string()),
            "synthesis_voice" => {
                self.synthesis_voice = (value != "NULL").then(|| value.to_string());
            },
            "language" => self.language = (value != "NULL").then(|| value.to_string()),
            "rate" | "pitch" | "volume" => {
                let parsed = if value == "NULL" {
                    None
                } else {
                    match value.parse::<i32>() {
                        Ok(n) => Some(n),
                        Err(_) => return false,
                    }
                };
                match key {
                    "rate" => self.rate = parsed,
                    "pitch" => self.pitch = parsed,
                    "volume" => self.volume = parsed,
                    _ => unreachable!(),
                }
            },
            "pitch_range" | "punctuation_mode" | "spelling_mode" | "cap_let_recogn" => {},
            _ => return false,
        }
        true
    }

    /// Builds the Wyoming `synthesize` request for `text` under these
    /// settings.
    ///
    /// `synthesis_voice` and `language` map onto the request's voice
    /// specification; `rate`/`pitch`/`volume` have no equivalent in
    /// Wyoming's `synthesize` event and are not sent.
    fn to_synthesize_data(&self, text: &str) -> SynthesizeData {
        if self.synthesis_voice.is_none() && self.language.is_none() {
            return SynthesizeData::new(text);
        }
        let mut voice = match &self.synthesis_voice {
            Some(name) => SynthesizeVoice::with_name(name.clone()),
            None => SynthesizeVoice::new(),
        };
        voice.language.clone_from(&self.language);
        SynthesizeData::new(text).with_voice(voice)
    }
}

/// Returns the lazily-built tokio runtime used to drive Wyoming client
/// calls, constructing it on first use.
fn runtime_handle(
    runtime: &mut Option<tokio::runtime::Runtime>,
) -> io::Result<&tokio::runtime::Runtime> {
    if runtime.is_none() {
        *runtime = Some(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?,
        );
    }
    Ok(runtime.as_ref().expect("just initialized above"))
}

/// Why a `key=value` block read by [`read_kv_block`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockError {
    /// A line wasn't of the form `key=value`.
    BadSyntax,
    /// `apply` rejected a line's key or value.
    BadValue,
}

/// Reads a dot-terminated `key=value` block from `lines`, calling `apply`
/// with each line's key and value.
///
/// `apply` returns `false` to reject a line's value. A line that isn't
/// `key=value` at all is a [`BlockError::BadSyntax`], which takes
/// priority over a [`BlockError::BadValue`] if both occur in the same
/// block, since a malformed line is a more fundamental protocol
/// violation than a bad value. Every line in the block is processed
/// regardless of earlier errors.
fn read_kv_block(
    lines: &mut impl Iterator<Item = io::Result<String>>,
    mut apply: impl FnMut(&str, &str) -> bool,
) -> io::Result<Option<BlockError>> {
    let mut bad_syntax = false;
    let mut bad_value = false;
    for line in lines {
        let line = line?;
        let line = line.trim();
        if line == "." {
            break;
        }
        match line.split_once('=') {
            Some((key, value)) => {
                if !apply(key, value) {
                    bad_value = true;
                }
            },
            None => bad_syntax = true,
        }
    }
    Ok(if bad_syntax {
        Some(BlockError::BadSyntax)
    } else if bad_value {
        Some(BlockError::BadValue)
    } else {
        None
    })
}

/// Handles the `AUDIO` command: reads a `key=value` block and accepts
/// only `audio_output_method=server`, the only output method this
/// module supports — it streams PCM back over `705 AUDIO` rather than
/// playing audio itself.
fn handle_audio(
    lines: &mut impl Iterator<Item = io::Result<String>>,
    output: &mut impl Write,
) -> io::Result<()> {
    write_reply(output, "207 OK RECEIVING AUDIO SETTINGS\n")?;
    match read_kv_block(lines, |key, value| {
        key != "audio_output_method" || value == "server"
    })? {
        Some(BlockError::BadSyntax) => write_reply(output, "302 ERROR BAD SYNTAX\n"),
        Some(BlockError::BadValue) => write_reply(output, "303 ERROR INVALID PARAMETER OR VALUE\n"),
        None => write_reply(output, "203 OK AUDIO INITIALIZED\n"),
    }
}

/// Handles the `LOGLEVEL` command: reads a `key=value` block setting the
/// module's log level.
///
/// speechd's `module.c` kills and unregisters the whole module if this
/// command doesn't return a `2xx` reply, right after `AUDIO` and before
/// ever requesting `LIST VOICES` — so this module must at least accept
/// the command, even though it has no log-level-dependent behavior to
/// configure. `log_level` is validated as an integer (matching speechd's
/// own reference module) but otherwise ignored.
///
/// Unlike `handle_audio`, unrecognized keys are rejected here, matching
/// `SET`'s strictness, since there is no other accepted key to be lenient
/// about.
fn handle_loglevel(
    lines: &mut impl Iterator<Item = io::Result<String>>,
    output: &mut impl Write,
) -> io::Result<()> {
    write_reply(output, "207 OK RECEIVING LOGLEVEL SETTINGS\n")?;
    match read_kv_block(lines, |key, value| {
        key == "log_level" && value.parse::<i32>().is_ok()
    })? {
        Some(BlockError::BadSyntax) => write_reply(output, "302 ERROR BAD SYNTAX\n"),
        Some(BlockError::BadValue) => write_reply(output, "303 ERROR INVALID PARAMETER OR VALUE\n"),
        None => write_reply(output, "203 OK LOGLEVEL SET\n"),
    }
}

/// Handles the `SET` command: reads a `key=value` block and applies it
/// to `settings` for the next `SPEAK`.
///
/// A block with any invalid line leaves `settings` untouched — either
/// every line in the block commits, or none of them do.
fn handle_set(
    lines: &mut impl Iterator<Item = io::Result<String>>,
    output: &mut impl Write,
    settings: &mut Settings,
) -> io::Result<()> {
    write_reply(output, "203 OK RECEIVING SETTINGS\n")?;
    let mut scratch = settings.clone();
    match read_kv_block(lines, |key, value| scratch.apply(key, value))? {
        Some(BlockError::BadSyntax) => write_reply(output, "302 ERROR BAD SYNTAX\n"),
        Some(BlockError::BadValue) => write_reply(output, "303 ERROR INVALID PARAMETER OR VALUE\n"),
        None => {
            *settings = scratch;
            write_reply(output, "203 OK SETTINGS RECEIVED\n")
        },
    }
}

/// Synthesizes `text` via `client` and streams it back to `output` as
/// `705 AUDIO` events, bracketed by `701 BEGIN`/`702 END`.
///
/// `Client::start_synthesis` (which sends the request and reads
/// `audio-start`) is private, so `200 OK SPEAKING`/`701 BEGIN` can't be
/// emitted between it and the first audio chunk directly; instead the
/// chunk callback emits that preamble the first time it runs, and its
/// absence afterward distinguishes "synthesis never started" (still
/// reply `301 ERROR CANT SPEAK`) from "synthesis started but failed
/// partway through" (audio was already delivered, so just close with
/// `702 END`, per the mid-stream error convention documented on
/// `Client::synthesize_streaming`).
///
/// `poll_interrupt` is polled once per audio piece; when it reports a
/// `STOP`/`PAUSE`, synthesis is aborted (the chunk callback returns
/// `ControlFlow::Break`, so `synthesize_streaming` drains and discards
/// the remaining `audio-chunk`/`audio-stop` events itself, leaving the
/// connection ready for the next `SPEAK`) and `703 STOP`/`704 PAUSE`
/// is sent instead of `702 END`.
fn speak(
    rt: &tokio::runtime::Runtime,
    client: &mut Client,
    settings: &Settings,
    text: &str,
    output: &mut impl Write,
    poll_interrupt: &mut impl FnMut() -> Option<Interrupt>,
) -> io::Result<()> {
    let synth_data = settings.to_synthesize_data(text);
    let mut preamble_sent = false;
    let mut bad_format = false;
    let mut io_error = None;
    let mut interrupt = None;

    let result = rt.block_on(client.synthesize_streaming(synth_data, |pcm, format| {
        let sample_size = usize::from(format.channels) * usize::from(format.width);
        if sample_size == 0 {
            bad_format = true;
            return ControlFlow::Break(());
        }
        if !preamble_sent {
            if let Err(e) = write_reply(output, "200 OK SPEAKING\n701 BEGIN\n") {
                io_error = Some(e);
                return ControlFlow::Break(());
            }
            preamble_sent = true;
        }
        let max_bytes = (MAX_CHUNK / sample_size) * sample_size;
        for piece in pcm.chunks(max_bytes.max(sample_size)) {
            if let Some(intr) = poll_interrupt() {
                interrupt = Some(intr);
                return ControlFlow::Break(());
            }
            if let Err(e) = write_audio_event(output, *format, piece) {
                io_error = Some(e);
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }));

    if let Some(e) = io_error {
        return Err(e);
    }
    if let Some(intr) = interrupt {
        return write_reply(output, intr.reply());
    }
    match result {
        Ok(_) if bad_format => write_reply(output, "301 ERROR CANT SPEAK\n"),
        Ok(_) => {
            if !preamble_sent {
                write_reply(output, "200 OK SPEAKING\n701 BEGIN\n")?;
            }
            write_reply(output, "702 END\n")
        },
        Err(e) if preamble_sent => {
            eprintln!("crane-wyoming: synthesis error after audio started: {e}");
            write_reply(output, "702 END\n")
        },
        Err(_) => write_reply(output, "301 ERROR CANT SPEAK\n"),
    }
}

/// Runs the speechd module command loop, reading commands from `input`
/// and writing replies to `output`.
///
/// Builds its own single-threaded tokio runtime to drive Wyoming client
/// calls from this otherwise-synchronous, blocking command loop.
///
/// `interrupt_fd`, if given, is polled for a pending `STOP`/`PAUSE` line
/// between audio chunks during `SPEAK` (see [`poll_fd_interrupt`]);
/// `None` disables interrupt polling (used by tests driving `run` with
/// an in-memory `input` that has no underlying file descriptor to poll).
///
/// # Errors
///
/// Returns an error if `input`/`output` fail, or if the tokio runtime
/// can't be built.
pub fn run(
    uri: &str,
    input: impl BufRead,
    mut output: impl Write,
    interrupt_fd: Option<BorrowedFd<'_>>,
) -> anyhow::Result<()> {
    let mut runtime: Option<tokio::runtime::Runtime> = None;
    let mut client: Option<Client> = None;
    let mut voices: Vec<Voice> = Vec::new();
    let mut settings = Settings::default();
    let mut lines = input.lines();

    while let Some(line) = lines.next() {
        let line = line?;
        match Command::parse(&line) {
            Command::Init => {
                let rt = runtime_handle(&mut runtime)?;
                match rt.block_on(init(uri)) {
                    Ok((connected, discovered_voices)) => {
                        client = Some(connected);
                        voices = discovered_voices;
                        write_reply(
                            &mut output,
                            "299-LOADED SUCCESSFULLY\n299 OK LOADED SUCCESSFULLY\n",
                        )?;
                    },
                    Err(e) => {
                        let message = e.to_string().replace('\n', " ");
                        write_reply(
                            &mut output,
                            &format!("399-{message}\n399 ERR CANT INIT MODULE\n"),
                        )?;
                    },
                }
            },
            Command::Set => handle_set(&mut lines, &mut output, &mut settings)?,
            Command::Audio => handle_audio(&mut lines, &mut output)?,
            Command::LogLevel => handle_loglevel(&mut lines, &mut output)?,
            Command::Speak => {
                write_reply(&mut output, "202 OK RECEIVING MESSAGE\n")?;
                let Some(text) = read_text_block(&mut lines)? else {
                    break;
                };
                if text.is_empty() {
                    write_reply(&mut output, "301 ERROR CANT SPEAK\n")?;
                    continue;
                }
                let Some(connected) = client.as_mut() else {
                    write_reply(&mut output, "301 ERROR CANT SPEAK\n")?;
                    continue;
                };
                let rt = runtime_handle(&mut runtime)?;
                let mut poll_interrupt = || interrupt_fd.and_then(poll_fd_interrupt);
                speak(
                    rt,
                    connected,
                    &settings,
                    &text,
                    &mut output,
                    &mut poll_interrupt,
                )?;
            },
            Command::ListVoices => {
                if voices.is_empty() {
                    write_reply(&mut output, "304 CANT LIST VOICES\n")?;
                } else {
                    for voice in &voices {
                        // The third column is speechd's voice "variant"
                        // (e.g. "kal16" for Festival); crane-wyoming voices
                        // never have one, so it's always "none".
                        write_reply(
                            &mut output,
                            &format!("200-{}\t{}\tnone\n", voice.name, voice.language),
                        )?;
                    }
                    write_reply(&mut output, "200 OK VOICE LIST SENT\n")?;
                }
            },
            Command::Stop => write_reply(&mut output, Interrupt::Stop.reply())?,
            Command::Pause => write_reply(&mut output, Interrupt::Pause.reply())?,
            Command::Quit => {
                client.take();
                write_reply(&mut output, "210 OK QUIT\n")?;
                break;
            },
            Command::Unknown => write_reply(&mut output, "300 ERR UNKNOWN COMMAND\n")?,
        }
    }
    Ok(())
}

/// Parses `argv[1]` as a speechd module config path and runs the module
/// command loop against stdin/stdout.
///
/// # Errors
///
/// Returns an error if the config path is missing or unreadable, or if
/// it has no `CraneURI` directive.
pub fn cli_main() -> anyhow::Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .context("usage: sd_crane_wyoming <config-path>")?;
    let config = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read config file {config_path}"))?;
    let uri = match parse_crane_uri(&config) {
        Some(result) => {
            result.with_context(|| format!("invalid CraneURI directive in {config_path}"))?
        },
        None => anyhow::bail!("missing CraneURI directive in {config_path}"),
    };

    let stdin = io::stdin();
    // SAFETY: fd 0 (stdin) is open and owned by this process for its
    // entire lifetime.
    let interrupt_fd = unsafe { BorrowedFd::borrow_raw(0) };
    run(&uri, stdin.lock(), io::stdout(), Some(interrupt_fd))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::net::TcpListener as StdTcpListener;
    use std::os::fd::AsFd;
    use std::os::unix::net::UnixStream;

    use serde_json::json;
    use wyoming_protocol::Event;
    use wyoming_protocol::event::{
        AudioChunkData, AudioStartData, AudioStopData, ErrorData, InfoData,
    };
    use wyoming_protocol::wire::{read_event, write_event};

    use super::*;

    /// Spawns a mock Wyoming server on its own thread (with its own tokio
    /// runtime, independent of the one `run` builds) that answers one
    /// `describe` request with the given `info` response.
    fn spawn_describe_server(info: InfoData) -> String {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nonblocking(true).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let stream = tokio::net::TcpStream::from_std(stream).unwrap();
                let (read_half, mut write_half) = tokio::io::split(stream);
                let mut reader = tokio::io::BufReader::new(read_half);
                read_event(&mut reader).await.unwrap();
                write_event(&mut write_half, &Event::Info(info))
                    .await
                    .unwrap();
            });
        });
        format!("tcp://{addr}")
    }

    /// Spawns a mock Wyoming server that answers one `describe` request
    /// with `info`, then one `synthesize` request by streaming `chunks`
    /// as `audio-chunk` events under `format` and closing with
    /// `audio-stop`.
    fn spawn_tts_server(info: InfoData, format: AudioFormat, chunks: Vec<Vec<u8>>) -> String {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nonblocking(true).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let stream = tokio::net::TcpStream::from_std(stream).unwrap();
                let (read_half, mut write_half) = tokio::io::split(stream);
                let mut reader = tokio::io::BufReader::new(read_half);

                read_event(&mut reader).await.unwrap();
                write_event(&mut write_half, &Event::Info(info))
                    .await
                    .unwrap();

                read_event(&mut reader).await.unwrap();
                write_event(
                    &mut write_half,
                    &Event::AudioStart(AudioStartData::new(format)),
                )
                .await
                .unwrap();
                for chunk in chunks {
                    write_event(
                        &mut write_half,
                        &Event::AudioChunk {
                            data: AudioChunkData::new(format),
                            audio: chunk,
                        },
                    )
                    .await
                    .unwrap();
                }
                write_event(&mut write_half, &Event::AudioStop(AudioStopData::new()))
                    .await
                    .unwrap();
            });
        });
        format!("tcp://{addr}")
    }

    /// Spawns a mock Wyoming server that answers `describe` with `info`,
    /// then rejects the following `synthesize` request with an
    /// `Event::Error` before ever sending `audio-start`.
    fn spawn_tts_error_server(info: InfoData) -> String {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nonblocking(true).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let stream = tokio::net::TcpStream::from_std(stream).unwrap();
                let (read_half, mut write_half) = tokio::io::split(stream);
                let mut reader = tokio::io::BufReader::new(read_half);

                read_event(&mut reader).await.unwrap();
                write_event(&mut write_half, &Event::Info(info))
                    .await
                    .unwrap();

                read_event(&mut reader).await.unwrap();
                write_event(
                    &mut write_half,
                    &Event::Error(ErrorData::new("synthesis failed")),
                )
                .await
                .unwrap();
            });
        });
        format!("tcp://{addr}")
    }

    /// An `InfoData` with two TTS programs offering three voices total,
    /// used by tests that need a realistic `LIST VOICES` response.
    fn sample_info() -> InfoData {
        InfoData::new().with_tts(vec![
            json!({
                "name": "voxtral",
                "voices": [
                    {"name": "de_female", "languages": ["de"]},
                    {"name": "en_male", "languages": ["en"]},
                ],
            }),
            json!({
                "name": "other",
                "voices": [
                    {"name": "no_lang", "languages": []},
                ],
            }),
        ])
    }

    #[test]
    fn run_init_success_replies_ok_then_quit() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(&uri, Cursor::new(&b"INIT\nQUIT\n"[..]), &mut output, None).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("299 OK LOADED SUCCESSFULLY"));
        assert!(output.contains("210 OK QUIT"));
    }

    #[test]
    fn run_init_failure_replies_err() {
        let mut output = Vec::new();
        run(
            "unix:///nonexistent/definitely-not-a-socket",
            Cursor::new(&b"INIT\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("399 ERR CANT INIT MODULE"));
    }

    #[test]
    fn run_unknown_command_replies_err() {
        let mut output = Vec::new();
        run(
            "tcp://127.0.0.1:1",
            Cursor::new(&b"FOO\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("300 ERR UNKNOWN COMMAND"));
        assert!(output.contains("210 OK QUIT"));
    }

    #[test]
    fn parse_crane_uri_extracts_quoted_value() {
        let config = "CraneURI \"unix:///run/user/1000/crane-wyoming/tts.sock\"\n";
        assert_eq!(
            parse_crane_uri(config).unwrap().unwrap(),
            "unix:///run/user/1000/crane-wyoming/tts.sock"
        );
    }

    #[test]
    fn parse_crane_uri_ignores_unrelated_lines() {
        let config = "# comment\nOtherKey \"value\"\n";
        assert_eq!(parse_crane_uri(config), None);
    }

    #[test]
    fn parse_crane_uri_missing_returns_none() {
        assert_eq!(parse_crane_uri(""), None);
    }

    #[test]
    fn parse_crane_uri_rejects_unquoted_value() {
        let config = "CraneURI unix:///run/user/1000/crane-wyoming/tts.sock\n";
        assert_eq!(parse_crane_uri(config), Some(Err(CraneUriError)));
    }

    #[test]
    fn parse_crane_uri_requires_word_boundary() {
        let config = "CraneURITimeout \"30\"\n";
        assert_eq!(parse_crane_uri(config), None);
    }

    #[test]
    fn extract_voices_reads_name_and_first_language() {
        let voices = extract_voices(&sample_info());
        assert_eq!(
            voices,
            vec![
                Voice {
                    name: "de_female".to_string(),
                    language: "de".to_string()
                },
                Voice {
                    name: "en_male".to_string(),
                    language: "en".to_string()
                },
                Voice {
                    name: "no_lang".to_string(),
                    language: "none".to_string()
                },
            ]
        );
    }

    #[test]
    fn extract_voices_empty_info_returns_empty() {
        assert_eq!(extract_voices(&InfoData::new()), Vec::new());
    }

    #[test]
    fn run_set_valid_block_replies_ok() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(
                &b"INIT\nSET\nvoice=male1\nsynthesis_voice=Chelsie\nlanguage=en\nrate=50\npitch=-10\nvolume=80\npitch_range=0\npunctuation_mode=none\nspelling_mode=off\ncap_let_recogn=none\n.\nQUIT\n"[..],
            ),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("203 OK RECEIVING SETTINGS"));
        assert!(output.contains("203 OK SETTINGS RECEIVED"));
    }

    #[test]
    fn run_set_bad_syntax_replies_302() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSET\nnotakeyvalue\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("302 ERROR BAD SYNTAX"));
    }

    #[test]
    fn run_set_unknown_key_replies_303() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSET\nbogus=1\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
    }

    #[test]
    fn run_set_non_integer_rate_replies_303() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSET\nrate=fast\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
    }

    #[test]
    fn settings_apply_null_clears_previously_set_field() {
        let mut settings = Settings::default();
        assert!(settings.apply("voice", "male1"));
        assert_eq!(settings.voice, Some("male1".to_string()));
        assert!(settings.apply("voice", "NULL"));
        assert_eq!(settings.voice, None);

        assert!(settings.apply("rate", "50"));
        assert_eq!(settings.rate, Some(50));
        assert!(settings.apply("rate", "NULL"));
        assert_eq!(settings.rate, None);
    }

    #[test]
    fn run_set_invalid_line_does_not_commit_earlier_valid_fields() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSET\nrate=50\nbogus=1\n.\nSET\nnotakeyvalue\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        // Both SET blocks fail, so neither `rate=50` (rejected alongside
        // `bogus=1`) nor a stray earlier value should ever reach a `SPEAK`.
        assert!(output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
        assert!(output.contains("302 ERROR BAD SYNTAX"));
        assert!(!output.contains("203 OK SETTINGS RECEIVED"));
    }

    #[test]
    fn run_set_bad_syntax_takes_priority_over_bad_value() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSET\nbogus=1\nnotakeyvalue\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("302 ERROR BAD SYNTAX"));
        assert!(!output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
    }

    #[test]
    fn run_list_voices_reports_cached_voices() {
        let uri = spawn_describe_server(sample_info());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nLIST VOICES\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("200-de_female\tde\tnone\n"));
        assert!(output.contains("200-en_male\ten\tnone\n"));
        assert!(output.contains("200-no_lang\tnone\tnone\n"));
        assert!(output.contains("200 OK VOICE LIST SENT"));
    }

    #[test]
    fn run_list_voices_with_no_voices_replies_304() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nLIST VOICES\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("304 CANT LIST VOICES"));
    }

    #[test]
    fn hdlc_escape_passes_through_ordinary_bytes() {
        assert_eq!(hdlc_escape(b"hello"), b"hello".to_vec());
    }

    #[test]
    fn hdlc_escape_escapes_newline() {
        assert_eq!(
            hdlc_escape(&[0x01, 0x0a, 0x02]),
            vec![0x01, 0x7d, 0x2a, 0x02]
        );
    }

    #[test]
    fn hdlc_escape_escapes_escape_byte() {
        assert_eq!(hdlc_escape(&[0x7d]), vec![0x7d, 0x5d]);
    }

    #[test]
    fn hdlc_escape_empty_input() {
        assert_eq!(hdlc_escape(&[]), Vec::<u8>::new());
    }

    #[test]
    fn write_audio_event_frames_one_chunk() {
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };
        let mut output = Vec::new();
        write_audio_event(&mut output, format, &[0x01, 0x02, 0x03, 0x04]).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.starts_with(
            "705-bits=16\n705-num_channels=1\n705-sample_rate=24000\n705-num_samples=2\n705-big_endian=0\n705-AUDIO\0"
        ));
        assert!(output.ends_with("\n705 AUDIO\n"));
    }

    #[test]
    fn read_text_block_reads_until_dot() {
        let mut lines = vec![
            Ok("hello".to_string()),
            Ok("world".to_string()),
            Ok(".".to_string()),
        ]
        .into_iter();
        assert_eq!(
            read_text_block(&mut lines).unwrap(),
            Some("hello\nworld".to_string())
        );
    }

    #[test]
    fn read_text_block_unstuffs_leading_dot() {
        let mut lines = vec![Ok("..still text".to_string()), Ok(".".to_string())].into_iter();
        assert_eq!(
            read_text_block(&mut lines).unwrap(),
            Some(".still text".to_string())
        );
    }

    #[test]
    fn read_text_block_empty_body() {
        let mut lines = vec![Ok(".".to_string())].into_iter();
        assert_eq!(read_text_block(&mut lines).unwrap(), Some(String::new()));
    }

    #[test]
    fn read_text_block_eof_before_terminator_returns_none() {
        let mut lines = vec![Ok("hello".to_string())].into_iter();
        assert_eq!(read_text_block(&mut lines).unwrap(), None);
    }

    #[test]
    fn read_text_block_preserves_leading_blank_lines() {
        let mut lines = vec![
            Ok(String::new()),
            Ok("foo".to_string()),
            Ok(".".to_string()),
        ]
        .into_iter();
        assert_eq!(
            read_text_block(&mut lines).unwrap(),
            Some("\nfoo".to_string())
        );
    }

    #[test]
    fn settings_to_synthesize_data_default_has_no_voice() {
        let settings = Settings::default();
        assert_eq!(settings.to_synthesize_data("hi"), SynthesizeData::new("hi"));
    }

    #[test]
    fn settings_to_synthesize_data_with_voice_name() {
        let settings = Settings {
            synthesis_voice: Some("de_female".to_string()),
            ..Settings::default()
        };
        assert_eq!(
            settings.to_synthesize_data("hi"),
            SynthesizeData::new("hi").with_voice(SynthesizeVoice::with_name("de_female"))
        );
    }

    #[test]
    fn settings_to_synthesize_data_with_language() {
        let settings = Settings {
            language: Some("de".to_string()),
            ..Settings::default()
        };
        let mut expected_voice = SynthesizeVoice::new();
        expected_voice.language = Some("de".to_string());
        assert_eq!(
            settings.to_synthesize_data("hi"),
            SynthesizeData::new("hi").with_voice(expected_voice)
        );
    }

    #[test]
    fn run_audio_server_method_replies_ok() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nAUDIO\naudio_output_method=server\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("207 OK RECEIVING AUDIO SETTINGS"));
        assert!(output.contains("203 OK AUDIO INITIALIZED"));
    }

    #[test]
    fn run_audio_other_method_replies_error() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nAUDIO\naudio_output_method=local\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
    }

    #[test]
    fn run_audio_bad_syntax_replies_302() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nAUDIO\nnotakeyvalue\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("302 ERROR BAD SYNTAX"));
    }

    #[test]
    fn run_audio_unknown_key_with_server_method_replies_ok() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(
                &b"INIT\nAUDIO\naudio_output_method=server\nunknown_key=value\n.\nQUIT\n"[..],
            ),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("203 OK AUDIO INITIALIZED"));
    }

    #[test]
    fn run_loglevel_valid_value_replies_ok() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nLOGLEVEL\nlog_level=5\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("207 OK RECEIVING LOGLEVEL SETTINGS"));
        assert!(output.contains("203 OK LOGLEVEL SET"));
    }

    #[test]
    fn run_loglevel_non_integer_value_replies_303() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nLOGLEVEL\nlog_level=nope\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
    }

    #[test]
    fn run_loglevel_unknown_key_replies_303() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nLOGLEVEL\nunknown_key=5\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
    }

    #[test]
    fn run_loglevel_bad_syntax_replies_302() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nLOGLEVEL\nnotakeyvalue\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("302 ERROR BAD SYNTAX"));
    }

    #[test]
    fn run_loglevel_bad_syntax_takes_priority_over_bad_value() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nLOGLEVEL\nunknown_key=5\nnotakeyvalue\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("302 ERROR BAD SYNTAX"));
        assert!(!output.contains("303 ERROR INVALID PARAMETER OR VALUE"));
    }

    #[test]
    fn run_speak_basic_streams_audio() {
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };
        let uri = spawn_tts_server(InfoData::new(), format, vec![vec![0x01, 0x02, 0x03, 0x04]]);
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSPEAK\nhello world\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("202 OK RECEIVING MESSAGE"));
        assert!(output.contains("200 OK SPEAKING"));
        assert!(output.contains("701 BEGIN"));
        assert!(output.contains("705-sample_rate=24000"));
        assert!(output.contains("705 AUDIO"));
        assert!(output.contains("702 END"));
        // BEGIN must precede END.
        assert!(output.find("701 BEGIN").unwrap() < output.find("702 END").unwrap());
    }

    #[test]
    fn run_speak_empty_text_replies_error() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSPEAK\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("301 ERROR CANT SPEAK"));
        assert!(!output.contains("701 BEGIN"));
    }

    #[test]
    fn run_speak_without_init_replies_error() {
        let mut output = Vec::new();
        run(
            "tcp://127.0.0.1:1",
            Cursor::new(&b"SPEAK\nhello\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("301 ERROR CANT SPEAK"));
    }

    #[test]
    fn run_speak_synthesis_error_replies_error() {
        let uri = spawn_tts_error_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSPEAK\nhello\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("301 ERROR CANT SPEAK"));
        assert!(!output.contains("701 BEGIN"));
    }

    #[test]
    fn run_speak_zero_width_format_replies_error() {
        let format = AudioFormat {
            rate: 24000,
            width: 0,
            channels: 1,
        };
        let uri = spawn_tts_server(InfoData::new(), format, vec![vec![0x00, 0x00]]);
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSPEAK\nhello\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("301 ERROR CANT SPEAK"));
        assert!(!output.contains("701 BEGIN"));
    }

    #[test]
    fn run_speak_escapes_newline_and_escape_bytes_in_audio() {
        let format = AudioFormat {
            rate: 16000,
            width: 1,
            channels: 1,
        };
        let uri = spawn_tts_server(InfoData::new(), format, vec![vec![0x0a, 0x7d, 0x03]]);
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSPEAK\nhi\n.\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let marker = b"705-AUDIO\0";
        let start = output
            .windows(marker.len())
            .position(|w| w == marker)
            .unwrap()
            + marker.len();
        let end = start
            + output[start..]
                .windows(2)
                .position(|w| w == b"\n7")
                .unwrap();
        assert_eq!(&output[start..end], &[0x7d, 0x2a, 0x7d, 0x5d, 0x03]);
    }

    #[test]
    fn poll_fd_interrupt_detects_stop() {
        let (mut tx, rx) = UnixStream::pair().unwrap();
        tx.write_all(b"STOP\n").unwrap();
        assert_eq!(poll_fd_interrupt(rx.as_fd()), Some(Interrupt::Stop));
    }

    #[test]
    fn poll_fd_interrupt_detects_pause() {
        let (mut tx, rx) = UnixStream::pair().unwrap();
        tx.write_all(b"PAUSE\n").unwrap();
        assert_eq!(poll_fd_interrupt(rx.as_fd()), Some(Interrupt::Pause));
    }

    #[test]
    fn poll_fd_interrupt_no_data_returns_none() {
        let (_tx, rx) = UnixStream::pair().unwrap();
        assert_eq!(poll_fd_interrupt(rx.as_fd()), None);
    }

    #[test]
    fn poll_fd_interrupt_unrecognized_line_returns_none() {
        let (mut tx, rx) = UnixStream::pair().unwrap();
        tx.write_all(b"FOO\n").unwrap();
        assert_eq!(poll_fd_interrupt(rx.as_fd()), None);
    }

    #[test]
    fn run_stop_when_not_speaking_replies_703() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSTOP\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("703 STOP"));
    }

    #[test]
    fn run_pause_when_not_speaking_replies_704() {
        let uri = spawn_describe_server(InfoData::new());
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nPAUSE\nQUIT\n"[..]),
            &mut output,
            None,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("704 PAUSE"));
    }

    /// `input` (the command script) and `interrupt_fd` (polled for
    /// `STOP`/`PAUSE`) are deliberately separate channels here: a
    /// `Cursor` has no real file descriptor to poll, so the interrupt is
    /// pre-loaded onto its own socketpair instead, simulating the case
    /// where a `STOP`/`PAUSE` line is already sitting unread on stdin by
    /// the time synthesis starts streaming chunks.
    #[test]
    fn run_speak_stop_interrupts_before_first_chunk() {
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };
        let uri = spawn_tts_server(
            InfoData::new(),
            format,
            vec![vec![0x01, 0x02], vec![0x03, 0x04], vec![0x05, 0x06]],
        );
        let (mut tx, rx) = UnixStream::pair().unwrap();
        tx.write_all(b"STOP\n").unwrap();
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSPEAK\nhello\n.\nQUIT\n"[..]),
            &mut output,
            Some(rx.as_fd()),
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("701 BEGIN"));
        assert!(output.contains("703 STOP"));
        assert!(!output.contains("702 END"));
        assert!(!output.contains("705 AUDIO"));
    }

    #[test]
    fn run_speak_pause_interrupts_before_first_chunk() {
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };
        let uri = spawn_tts_server(
            InfoData::new(),
            format,
            vec![vec![0x01, 0x02], vec![0x03, 0x04], vec![0x05, 0x06]],
        );
        let (mut tx, rx) = UnixStream::pair().unwrap();
        tx.write_all(b"PAUSE\n").unwrap();
        let mut output = Vec::new();
        run(
            &uri,
            Cursor::new(&b"INIT\nSPEAK\nhello\n.\nQUIT\n"[..]),
            &mut output,
            Some(rx.as_fd()),
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("701 BEGIN"));
        assert!(output.contains("704 PAUSE"));
        assert!(!output.contains("702 END"));
        assert!(!output.contains("705 AUDIO"));
    }

    /// Calls `speak` directly with a `poll_interrupt` closure that lets
    /// the first audio chunk through before stopping, covering the
    /// mid-stream abort case — as opposed to the `before_first_chunk`
    /// tests above, which interrupt before any audio is written.
    #[test]
    fn speak_stop_mid_stream_emits_audio_then_703() {
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };
        let uri = spawn_tts_server(
            InfoData::new(),
            format,
            vec![vec![0x01, 0x02], vec![0x03, 0x04], vec![0x05, 0x06]],
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (mut client, _voices) = runtime.block_on(init(&uri)).unwrap();
        let settings = Settings::default();
        let mut output = Vec::new();
        let mut calls = 0;
        let mut poll_interrupt = || {
            calls += 1;
            if calls == 1 {
                None
            } else {
                Some(Interrupt::Stop)
            }
        };
        speak(
            &runtime,
            &mut client,
            &settings,
            "hello",
            &mut output,
            &mut poll_interrupt,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("701 BEGIN"));
        assert!(output.contains("705 AUDIO"));
        assert!(output.contains("703 STOP"));
        assert!(!output.contains("702 END"));
    }
}
