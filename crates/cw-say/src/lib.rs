//! Standalone Wyoming protocol TTS CLI client.
//!
//! Connects to a running Wyoming server, synthesizes text
//! to WAV or raw PCM, and can list the server's available
//! voices/languages. Depends only on `wyoming-protocol`.

use std::collections::BTreeSet;
use std::io::{self, Read, Write};
use std::ops::ControlFlow;
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use wyoming_protocol::event::{InfoData, SynthesizeData, SynthesizeVoice};
use wyoming_protocol::{AudioFormat, Client, SynthesizeResponse};

/// Default Unix socket path suffix under `$XDG_RUNTIME_DIR`.
const DEFAULT_SOCKET_SUFFIX: &str = "crane-wyoming/tts.sock";
/// Fallback URI used when `$XDG_RUNTIME_DIR` is not set.
const DEFAULT_TCP_URI: &str = "tcp://127.0.0.1:10200";

/// Output audio format for synthesized speech.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    /// WAV container (default).
    Wav,
    /// Bare PCM samples, no header.
    Raw,
}

/// Wyoming protocol TTS CLI client.
#[derive(Parser)]
#[command(version, about = "Wyoming protocol TTS CLI client")]
struct Args {
    /// Text to synthesize. Reads all of stdin if omitted.
    #[arg(long)]
    text: Option<String>,

    /// Wyoming server URI (`tcp://host:port` or `unix:///path`).
    /// Defaults to `unix://$XDG_RUNTIME_DIR/crane-wyoming/tts.sock`, or
    /// `tcp://127.0.0.1:10200` if `XDG_RUNTIME_DIR` is unset.
    #[arg(long)]
    uri: Option<String>,

    /// Voice name to request.
    #[arg(long)]
    voice: Option<String>,

    /// Voice language to request.
    #[arg(long)]
    language: Option<String>,

    /// Output file path. Omit or pass "-" to write to stdout.
    #[arg(long, short = 'o')]
    output: Option<String>,

    /// Output audio format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Wav)]
    format: OutputFormat,

    /// Print available voices and exit, without synthesizing.
    #[arg(long)]
    list_voices: bool,

    /// Print available voice languages and exit, without synthesizing.
    #[arg(long)]
    list_languages: bool,

    /// Seconds to wait for a server response before giving up.
    #[arg(long, default_value_t = 30.0)]
    timeout: f64,
}

/// Computes the default Wyoming server URI.
///
/// Prefers a Unix socket under `$XDG_RUNTIME_DIR` (passed as
/// `xdg_runtime_dir` so callers can test this without touching process
/// environment), falling back to the server's default TCP address.
fn default_uri(xdg_runtime_dir: Option<&str>) -> String {
    match xdg_runtime_dir {
        Some(dir) => format!("unix://{dir}/{DEFAULT_SOCKET_SUFFIX}"),
        None => DEFAULT_TCP_URI.to_string(),
    }
}

/// Resolves the text to synthesize: `text` if given, otherwise all of stdin.
///
/// Both sources are trimmed and rejected if empty, so `--text ""` and an
/// empty/whitespace-only stdin pipe fail the same way.
fn resolve_text(text: Option<&str>) -> anyhow::Result<String> {
    let buf = if let Some(text) = text {
        text.trim().to_string()
    } else {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .context("failed to read text from stdin")?;
        buf.trim().to_string()
    };
    if buf.is_empty() {
        anyhow::bail!("no text provided; use --text or pipe text to stdin");
    }
    Ok(buf)
}

/// Writes a canonical 44-byte WAV header.
///
/// `data_size` is the number of PCM audio bytes that follow. When `None`,
/// writes `0xFFFF_FFFF` placeholders for the RIFF and data chunk sizes --
/// the conventional "read until EOF" marker for streaming WAV to a pipe
/// that isn't seekable enough to patch in the real size afterward.
fn write_wav_header(
    writer: &mut impl Write,
    format: AudioFormat,
    data_size: Option<u32>,
) -> io::Result<()> {
    let byte_rate = format
        .rate
        .saturating_mul(u32::from(format.channels))
        .saturating_mul(u32::from(format.width));
    let block_align = format.channels.saturating_mul(format.width);
    let bits_per_sample = format.width * 8;
    let (riff_size, data_chunk_size) = match data_size {
        Some(n) => (n + 36, n),
        None => (0xFFFF_FFFF, 0xFFFF_FFFF),
    };

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&format.channels.to_le_bytes())?;
    writer.write_all(&format.rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_chunk_size.to_le_bytes())
}

/// Iterates over `(program_name, voice)` pairs across all TTS programs in `info`.
fn iter_voices(info: &InfoData) -> impl Iterator<Item = (&str, &serde_json::Value)> {
    info.tts.iter().flat_map(|program| {
        let program_name = program
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unnamed>");
        program
            .get("voices")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .map(move |voice| (program_name, voice))
    })
}

/// Prints available voices (name, program, languages) to stdout.
///
/// Pattern mirrors `tests/test_wyoming_tts.py`'s `describe()`.
fn print_voices(info: &InfoData) {
    println!("Available voices:");
    for (program_name, voice) in iter_voices(info) {
        let voice_name = voice
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unnamed>");
        let languages = voice.get("languages").cloned().unwrap_or_default();
        println!("  - {voice_name} ({program_name}, {languages})");
    }
}

/// Prints unique voice languages, sorted, to stdout.
///
/// Pattern mirrors `tests/test_wyoming_tts.py`'s `print_languages()`.
fn print_languages(info: &InfoData) {
    let mut languages = BTreeSet::new();
    for (_, voice) in iter_voices(info) {
        let Some(voice_languages) = voice.get("languages").and_then(serde_json::Value::as_array)
        else {
            continue;
        };
        for language in voice_languages {
            if let Some(language) = language.as_str() {
                languages.insert(language.to_string());
            }
        }
    }
    println!("Available languages:");
    for language in languages {
        println!("  - {language}");
    }
}

/// Computes audio duration in seconds from a byte count and format.
#[allow(clippy::cast_precision_loss)]
fn duration_secs(format: AudioFormat, total_bytes: usize) -> f64 {
    let denom = f64::from(format.width) * f64::from(format.channels) * f64::from(format.rate);
    if denom == 0.0 {
        0.0
    } else {
        total_bytes as f64 / denom
    }
}

/// Streams synthesized audio to stdout as it arrives.
///
/// Writes a WAV header with pipe-friendly placeholder sizes before the
/// first chunk (or, if synthesis produced no chunks, a zero-size header
/// afterward) unless `format` is [`OutputFormat::Raw`].
async fn synthesize_to_stdout(
    client: &mut Client,
    data: SynthesizeData,
    format: OutputFormat,
) -> anyhow::Result<()> {
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut header_written = false;
    let mut write_error: Option<io::Error> = None;
    let mut total_bytes: usize = 0;

    let audio_format = client
        .synthesize_streaming(data, |chunk, chunk_format| {
            if !header_written && format == OutputFormat::Wav {
                if let Err(e) = write_wav_header(&mut out, *chunk_format, None) {
                    write_error = Some(e);
                    return ControlFlow::Break(());
                }
                header_written = true;
            }
            match out.write_all(chunk).and_then(|()| out.flush()) {
                Ok(()) => {
                    total_bytes += chunk.len();
                    ControlFlow::Continue(())
                },
                Err(e) => {
                    write_error = Some(e);
                    ControlFlow::Break(())
                },
            }
        })
        .await?;

    if let Some(e) = write_error {
        return Err(e.into());
    }
    if !header_written && format == OutputFormat::Wav {
        write_wav_header(&mut out, audio_format, Some(0))?;
        out.flush()?;
    }

    eprintln!(
        "wrote stdout: {total_bytes} bytes, {:.2}s",
        duration_secs(audio_format, total_bytes)
    );
    Ok(())
}

/// Synthesizes audio and writes it to a file with correct WAV sizes.
async fn synthesize_to_file(
    client: &mut Client,
    data: SynthesizeData,
    path: &str,
    format: OutputFormat,
) -> anyhow::Result<()> {
    let response = client.synthesize(data).await?;
    if let Err(e) = write_response_to_file(&response, path, format) {
        let _ = std::fs::remove_file(path);
        return Err(e);
    }

    eprintln!(
        "wrote {path}: {} bytes, {:.2}s",
        response.audio.len(),
        duration_secs(response.format, response.audio.len())
    );
    Ok(())
}

/// Writes a synthesis response to `path`, with a WAV header when `format` is
/// [`OutputFormat::Wav`].
fn write_response_to_file(
    response: &SynthesizeResponse,
    path: &str,
    format: OutputFormat,
) -> anyhow::Result<()> {
    let mut file =
        std::fs::File::create(path).with_context(|| format!("failed to create {path}"))?;
    if format == OutputFormat::Wav {
        let data_size =
            u32::try_from(response.audio.len()).context("audio too large to fit in a WAV file")?;
        write_wav_header(&mut file, response.format, Some(data_size))?;
    }
    file.write_all(&response.audio)?;
    Ok(())
}

/// Parses CLI arguments and runs the requested Wyoming client operation.
///
/// # Errors
///
/// Returns an error if the server can't be reached or times out, if the
/// server reports a protocol/synthesis error, or if no text is available
/// to synthesize.
pub async fn cli_main() -> anyhow::Result<()> {
    let Args {
        text,
        uri,
        voice,
        language,
        output,
        format,
        list_voices,
        list_languages,
        timeout,
    } = Args::parse();

    let timeout = Duration::from_secs_f64(timeout.max(0.0));
    let uri = uri.unwrap_or_else(|| default_uri(std::env::var("XDG_RUNTIME_DIR").ok().as_deref()));

    if list_voices || list_languages {
        let mut client = tokio::time::timeout(timeout, Client::connect(&uri))
            .await
            .map_err(|_| anyhow::anyhow!("timed out connecting to {uri}"))?
            .with_context(|| format!("could not connect to {uri}"))?;
        let info = tokio::time::timeout(timeout, client.describe())
            .await
            .map_err(|_| anyhow::anyhow!("server did not respond in time"))??;
        if list_voices {
            print_voices(&info);
        }
        if list_languages {
            print_languages(&info);
        }
        return Ok(());
    }

    // Resolved before connecting: reading stdin can block indefinitely, and
    // we don't want to hold a connection slot open on the server while we
    // wait for it.
    let text = resolve_text(text.as_deref())?;

    let mut client = tokio::time::timeout(timeout, Client::connect(&uri))
        .await
        .map_err(|_| anyhow::anyhow!("timed out connecting to {uri}"))?
        .with_context(|| format!("could not connect to {uri}"))?;

    let mut synth = SynthesizeData::new(text);
    if voice.is_some() || language.is_some() {
        let mut synth_voice = SynthesizeVoice::new();
        synth_voice.name = voice;
        synth_voice.language = language;
        synth = synth.with_voice(synth_voice);
    }

    let to_file = output.as_deref().filter(|path| *path != "-");
    let result = if let Some(path) = to_file {
        tokio::time::timeout(
            timeout,
            synthesize_to_file(&mut client, synth, path, format),
        )
        .await
    } else {
        tokio::time::timeout(timeout, synthesize_to_stdout(&mut client, synth, format)).await
    };
    result.map_err(|_| anyhow::anyhow!("server did not respond in time"))??;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_wav_header_streaming_uses_placeholder_sizes() {
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };
        let mut buf = Vec::new();
        write_wav_header(&mut buf, format, None).unwrap();
        assert_eq!(buf.len(), 44);
        assert_eq!(&buf[0..4], b"RIFF");
        assert_eq!(&buf[4..8], &0xFFFF_FFFFu32.to_le_bytes());
        assert_eq!(&buf[8..12], b"WAVE");
        assert_eq!(&buf[36..40], b"data");
        assert_eq!(&buf[40..44], &0xFFFF_FFFFu32.to_le_bytes());
    }

    #[test]
    fn write_wav_header_fixed_size_computes_riff_and_data_sizes() {
        let format = AudioFormat {
            rate: 24000,
            width: 2,
            channels: 1,
        };
        let mut buf = Vec::new();
        write_wav_header(&mut buf, format, Some(1000)).unwrap();
        assert_eq!(&buf[4..8], &1036u32.to_le_bytes());
        assert_eq!(&buf[40..44], &1000u32.to_le_bytes());
    }

    #[test]
    fn default_uri_prefers_xdg_runtime_dir() {
        assert_eq!(
            default_uri(Some("/run/user/1000")),
            "unix:///run/user/1000/crane-wyoming/tts.sock"
        );
    }

    #[test]
    fn default_uri_falls_back_to_tcp() {
        assert_eq!(default_uri(None), "tcp://127.0.0.1:10200");
    }
}
