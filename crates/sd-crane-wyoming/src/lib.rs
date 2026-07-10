//! speech-dispatcher output module for crane-wyoming.
//!
//! speechd launches this binary as a subprocess and talks to it over
//! stdin/stdout using speechd's own line-oriented, SMTP-style module
//! protocol. Internally it is a Wyoming client to a running
//! `crane-wyoming` server. This is the handshake skeleton: `INIT`
//! connects and calls `describe`, `QUIT` disconnects, and anything else
//! is rejected as unknown. Later steps add `SET`, `LIST VOICES`, `SPEAK`,
//! and `STOP`/`PAUSE`.

use std::io::{self, BufRead, Write};

use anyhow::Context;
use wyoming_protocol::{Client, ClientError};

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

/// Connects to the Wyoming server at `uri` and confirms it responds to
/// `describe`, as an `INIT`-time sanity check.
async fn init(uri: &str) -> Result<Client, ClientError> {
    let mut client = Client::connect(uri).await?;
    client.describe().await?;
    Ok(client)
}

/// A speechd module command line.
///
/// More variants (`Set`, `ListVoices`, `Speak`, `Stop`, `Pause`) are added
/// as later steps implement them.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// `INIT` — connect to the Wyoming server and confirm it responds.
    Init,
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
            "QUIT" => Self::Quit,
            _ => Self::Unknown,
        }
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

/// Runs the speechd module command loop, reading commands from `input`
/// and writing replies to `output`.
///
/// Builds its own single-threaded tokio runtime to drive Wyoming client
/// calls from this otherwise-synchronous, blocking command loop.
///
/// # Errors
///
/// Returns an error if `input`/`output` fail, or if the tokio runtime
/// can't be built.
pub fn run(uri: &str, input: impl BufRead, mut output: impl Write) -> anyhow::Result<()> {
    let mut runtime: Option<tokio::runtime::Runtime> = None;
    let mut client: Option<Client> = None;

    for line in input.lines() {
        let line = line?;
        match Command::parse(&line) {
            Command::Init => {
                let rt = runtime_handle(&mut runtime)?;
                match rt.block_on(init(uri)) {
                    Ok(connected) => {
                        client = Some(connected);
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
    run(&uri, stdin.lock(), io::stdout())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::net::TcpListener as StdTcpListener;

    use wyoming_protocol::Event;
    use wyoming_protocol::event::InfoData;
    use wyoming_protocol::wire::{read_event, write_event};

    use super::*;

    /// Spawns a mock Wyoming server on its own thread (with its own tokio
    /// runtime, independent of the one `run` builds) that answers one
    /// `describe` request with an empty `info` response.
    fn spawn_describe_server() -> String {
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
                write_event(&mut write_half, &Event::Info(InfoData::new()))
                    .await
                    .unwrap();
            });
        });
        format!("tcp://{addr}")
    }

    #[test]
    fn run_init_success_replies_ok_then_quit() {
        let uri = spawn_describe_server();
        let mut output = Vec::new();
        run(&uri, Cursor::new(&b"INIT\nQUIT\n"[..]), &mut output).unwrap();
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
}
