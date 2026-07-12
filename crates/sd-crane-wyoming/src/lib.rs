//! speech-dispatcher output module for crane-wyoming.
//!
//! speechd launches this binary as a subprocess and talks to it over
//! stdin/stdout using speechd's own line-oriented, SMTP-style module
//! protocol. Internally it is a Wyoming client to a running
//! `crane-wyoming` server. `INIT` connects, calls `describe`, and caches
//! the resulting voice list; `SET` stores per-utterance settings for a
//! later `SPEAK`; `LIST VOICES` answers from the cached list; `QUIT`
//! disconnects; anything else is rejected as unknown. Later steps add
//! `SPEAK` and `STOP`/`PAUSE`.

use std::io::{self, BufRead, Write};

use anyhow::Context;
use wyoming_protocol::event::InfoData;
use wyoming_protocol::{Client, ClientError};

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

/// Connects to the Wyoming server at `uri`, confirms it responds to
/// `describe`, and returns its voice list for `LIST VOICES` to serve
/// later.
async fn init(uri: &str) -> Result<(Client, Vec<Voice>), ClientError> {
    let mut client = Client::connect(uri).await?;
    let info = client.describe().await?;
    Ok((client, extract_voices(&info)))
}

/// A speechd module command line.
///
/// More variants (`Speak`, `Stop`, `Pause`) are added as later steps
/// implement them.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// `INIT` — connect to the Wyoming server and confirm it responds.
    Init,
    /// `SET` — receive a multiline block of `key=value` settings.
    Set,
    /// `LIST VOICES` — report the voice list cached at `INIT` time.
    ListVoices,
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
            "LIST VOICES" => Self::ListVoices,
            "QUIT" => Self::Quit,
            _ => Self::Unknown,
        }
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
            Command::Set => {
                write_reply(&mut output, "203 OK RECEIVING SETTINGS\n")?;
                let mut scratch = settings.clone();
                let mut bad_syntax = false;
                let mut bad_value = false;
                for line in lines.by_ref() {
                    let line = line?;
                    let line = line.trim();
                    if line == "." {
                        break;
                    }
                    match line.split_once('=') {
                        Some((key, value)) => {
                            if !scratch.apply(key, value) {
                                bad_value = true;
                            }
                        },
                        None => bad_syntax = true,
                    }
                }
                // A malformed line is a more fundamental protocol violation
                // than a bad value, so it takes priority when both occur in
                // the same block.
                if bad_syntax {
                    write_reply(&mut output, "302 ERROR BAD SYNTAX\n")?;
                } else if bad_value {
                    write_reply(&mut output, "303 ERROR INVALID PARAMETER OR VALUE\n")?;
                } else {
                    settings = scratch;
                    write_reply(&mut output, "203 OK SETTINGS RECEIVED\n")?;
                }
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

    use serde_json::json;
    use wyoming_protocol::Event;
    use wyoming_protocol::event::InfoData;
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
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("304 CANT LIST VOICES"));
    }
}
