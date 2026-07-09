//! Async read/write functions for the Wyoming wire protocol.
//!
//! Wraps [`Event`](crate::event::Event) (de)serialization in the
//! three-part wire framing: a newline-terminated JSON header line,
//! an optional extended-data JSON segment, and an optional binary
//! payload segment.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::error::{IoStage, ProtocolError};
use crate::event::{Event, to_json_vec};

/// Protocol version string included in every outgoing header.
///
/// Purely informational -- the reader does not interpret it. Uses the
/// crate version, following the Python reference implementation's
/// convention of stamping the package version on the wire.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum accepted length of the JSON header line, in bytes.
///
/// Guards against a peer that never sends a newline, which would
/// otherwise grow the header buffer without bound.
pub const MAX_HEADER_LINE: usize = 64 * 1024;

/// Maximum accepted value of a header's `data_length`, in bytes.
///
/// Guards against a peer declaring an oversized `data_length` in the
/// header, which would otherwise trigger an unbounded allocation
/// before any data is actually read.
pub const MAX_DATA_LENGTH: usize = 1024 * 1024;

/// Maximum accepted value of a header's `payload_length`, in bytes.
///
/// Guards against a peer declaring an oversized `payload_length` in
/// the header, which would otherwise trigger an unbounded allocation
/// before any data is actually read.
pub const MAX_PAYLOAD_LENGTH: usize = 10 * 1024 * 1024;

/// The JSON header line sent/received at the start of every message.
///
/// `data` is only ever populated when reading (for backward
/// compatibility with senders that inline small data dicts in the
/// header); [`write_event`] never sets it, always sending `data` as a
/// separate segment instead.
#[derive(Serialize, Deserialize)]
struct Header {
    r#type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    payload_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// The JSON header line as sent by [`write_event`].
///
/// Borrows its string fields instead of owning them -- [`write_event`]
/// always has `'static` or caller-owned string data on hand, so there
/// is no need to allocate a fresh `String` per outgoing event.
#[derive(Serialize)]
struct WriteHeader<'a> {
    r#type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data_length: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload_length: Option<usize>,
}

/// Writes a single Wyoming event to `writer` and flushes it.
///
/// The event's `data` dict (if any) is always sent as the external
/// data segment, never inlined into the header line, matching the
/// behavior of the reference Python writer.
///
/// # Errors
///
/// Returns an error if JSON serialization of the header or data fails,
/// or if the underlying writer returns an I/O error.
pub async fn write_event<W>(writer: &mut W, event: &Event) -> Result<(), ProtocolError>
where
    W: AsyncWrite + Unpin,
{
    let data_bytes = event.serialize_data()?;
    let payload = event.payload();

    let header = WriteHeader {
        r#type: event.event_type(),
        version: Some(VERSION),
        data_length: data_bytes.as_ref().map(Vec::len),
        payload_length: payload.map(<[u8]>::len),
    };
    let mut header_bytes = to_json_vec(&header)?;
    header_bytes.push(b'\n');

    writer
        .write_all(&header_bytes)
        .await
        .map_err(|e| ProtocolError::Io {
            source: e,
            stage: IoStage::WriteHeader,
        })?;
    if let Some(data) = &data_bytes {
        writer
            .write_all(data)
            .await
            .map_err(|e| ProtocolError::Io {
                source: e,
                stage: IoStage::WriteData,
            })?;
    }
    if let Some(payload) = payload {
        writer
            .write_all(payload)
            .await
            .map_err(|e| ProtocolError::Io {
                source: e,
                stage: IoStage::WritePayload,
            })?;
    }
    writer.flush().await.map_err(|e| ProtocolError::Io {
        source: e,
        stage: IoStage::Flush,
    })?;
    Ok(())
}

/// Reads a single newline-terminated line from `reader` into `line`,
/// bailing out once more than `max_len` bytes have been read without
/// finding a newline.
///
/// Unlike [`AsyncBufReadExt::read_line`], this never grows `line`
/// without bound in response to a peer that withholds the newline.
async fn read_line_bounded<R>(
    reader: &mut R,
    line: &mut String,
    max_len: usize,
) -> Result<usize, ProtocolError>
where
    R: AsyncBufRead + Unpin,
{
    let mut buf = Vec::new();
    loop {
        let available = reader.fill_buf().await.map_err(|e| ProtocolError::Io {
            source: e,
            stage: IoStage::ReadHeader,
        })?;
        if available.is_empty() {
            break;
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            buf.extend_from_slice(&available[..=pos]);
            reader.consume(pos + 1);
            break;
        }
        let consumed = available.len();
        buf.extend_from_slice(available);
        reader.consume(consumed);
        if buf.len() > max_len {
            return Err(ProtocolError::HeaderTooLong { max: max_len });
        }
    }
    if buf.len() > max_len {
        return Err(ProtocolError::HeaderTooLong { max: max_len });
    }

    let bytes_read = buf.len();
    if bytes_read > 0 {
        line.push_str(&String::from_utf8(buf).map_err(ProtocolError::HeaderNotUtf8)?);
    }
    Ok(bytes_read)
}

/// Reads a single Wyoming event from `reader`.
///
/// Returns `Ok(None)` when the stream is closed cleanly at a message
/// boundary (EOF on the header line). Returns `Err` for malformed
/// headers, truncated data/payload segments, or I/O errors.
///
/// # Wire format
///
/// 1. Reads one newline-terminated JSON header line.
/// 2. If `data_length` is present and non-zero, reads that many bytes
///    as a UTF-8 JSON object. If the header line also carried an
///    inline `data` object, the freshly read data is merged on top of
///    it (matching keys are overwritten; inline-only keys survive).
/// 3. If `payload_length` is present and non-zero, reads that many
///    raw bytes as the binary payload.
/// 4. Dispatches on the type string to build the corresponding
///    [`Event`].
///
/// # Cancellation safety
///
/// This function is not cancellation-safe: if the returned future is
/// dropped before completion, bytes may have already been consumed
/// from `reader` without producing an `Event`, desynchronizing the
/// stream.
///
/// # Errors
///
/// Returns an error if the header line is not valid JSON, if the
/// header line or a declared `data_length`/`payload_length` exceeds
/// [`MAX_HEADER_LINE`]/[`MAX_DATA_LENGTH`]/[`MAX_PAYLOAD_LENGTH`], if a
/// `data`/`payload` segment is shorter than its declared length, if an
/// inline or external `data` value is present but not a JSON object,
/// or if `data` does not match the schema expected for a recognized
/// event type.
pub async fn read_event<R>(reader: &mut R) -> Result<Option<Event>, ProtocolError>
where
    R: AsyncBufRead + Unpin,
{
    let mut line = String::new();
    let bytes_read = read_line_bounded(reader, &mut line, MAX_HEADER_LINE).await?;
    if bytes_read == 0 {
        return Ok(None);
    }

    let header: Header =
        serde_json::from_str(line.trim_end()).map_err(ProtocolError::HeaderInvalidJson)?;

    let mut merged_data = match header.data {
        Some(Value::Object(map)) => map,
        Some(other) => return Err(ProtocolError::NonObjectData(Box::new(other))),
        None => Map::new(),
    };

    if let Some(len) = header.data_length.filter(|&len| len > 0) {
        if len > MAX_DATA_LENGTH {
            return Err(ProtocolError::DataLengthExceeded {
                length: len,
                max: MAX_DATA_LENGTH,
            });
        }
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .await
            .map_err(|e| ProtocolError::Io {
                source: e,
                stage: IoStage::ReadData,
            })?;
        let external: Value =
            serde_json::from_slice(&buf).map_err(ProtocolError::InvalidEventData)?;
        match external {
            Value::Object(external_map) => {
                for (key, value) in external_map {
                    merged_data.insert(key, value);
                }
            },
            other => return Err(ProtocolError::NonObjectData(Box::new(other))),
        }
    }

    let payload = if let Some(len) = header.payload_length.filter(|&len| len > 0) {
        if len > MAX_PAYLOAD_LENGTH {
            return Err(ProtocolError::PayloadLengthExceeded {
                length: len,
                max: MAX_PAYLOAD_LENGTH,
            });
        }
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .await
            .map_err(|e| ProtocolError::Io {
                source: e,
                stage: IoStage::ReadPayload,
            })?;
        Some(buf)
    } else {
        None
    };

    let data = if merged_data.is_empty() {
        Value::Null
    } else {
        Value::Object(merged_data)
    };
    Event::from_wire(&header.r#type, data, payload).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{
        AudioChunkData, AudioStartData, AudioStopData, ErrorData, PingData, PongData,
        SynthesizeData,
    };
    use std::io::Cursor;

    async fn write_to_vec(event: &Event) -> Vec<u8> {
        let mut buf = Vec::new();
        write_event(&mut buf, event).await.expect("write_event");
        buf
    }

    #[tokio::test]
    async fn test_round_trip_audio_start() {
        let event = Event::AudioStart(AudioStartData {
            rate: 22050,
            width: 2,
            channels: 1,
            timestamp: None,
        });
        let bytes = write_to_vec(&event).await;
        let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));
        let read_back = read_event(&mut reader).await.unwrap().unwrap();
        assert_eq!(read_back, event);
    }

    #[tokio::test]
    async fn test_round_trip_synthesize() {
        let event = Event::Synthesize(SynthesizeData {
            text: "hello".to_string(),
            voice: None,
        });
        let bytes = write_to_vec(&event).await;
        let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));
        let read_back = read_event(&mut reader).await.unwrap().unwrap();
        assert_eq!(read_back, event);
    }

    #[tokio::test]
    async fn test_round_trip_describe_no_data() {
        let event = Event::Describe;
        let bytes = write_to_vec(&event).await;
        let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));
        let read_back = read_event(&mut reader).await.unwrap().unwrap();
        assert_eq!(read_back, event);
    }

    #[tokio::test]
    async fn test_round_trip_audio_chunk_payload() {
        let audio = vec![0u8, 1, 2, 3, 255, 254];
        let event = Event::AudioChunk {
            data: AudioChunkData {
                rate: 24000,
                width: 2,
                channels: 1,
                timestamp: Some(10),
            },
            audio: audio.clone(),
        };
        let bytes = write_to_vec(&event).await;
        let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));
        let read_back = read_event(&mut reader).await.unwrap().unwrap();
        match read_back {
            Event::AudioChunk { audio: got, .. } => assert_eq!(got, audio),
            _ => panic!("expected AudioChunk"),
        }
    }

    #[tokio::test]
    async fn test_header_omits_data_length_when_no_data() {
        let bytes = write_to_vec(&Event::Describe).await;
        let newline = bytes.iter().position(|&b| b == b'\n').unwrap();
        let header: Value = serde_json::from_slice(&bytes[..newline]).unwrap();
        assert!(header.get("data_length").is_none());
        assert!(header.get("payload_length").is_none());
        assert_eq!(&bytes[newline + 1..], b"" as &[u8]);
    }

    #[tokio::test]
    async fn test_round_trip_all_optional_fields_unset() {
        // Events whose data struct has only optional fields (all unset)
        // serialize to no data segment at all; the reader must still be
        // able to reconstruct the typed struct from the absent segment.
        let events = vec![
            Event::AudioStop(AudioStopData { timestamp: None }),
            Event::Ping(PingData { text: None }),
            Event::Pong(PongData { text: None }),
        ];
        for event in events {
            let bytes = write_to_vec(&event).await;
            let newline = bytes.iter().position(|&b| b == b'\n').unwrap();
            let header: Value = serde_json::from_slice(&bytes[..newline]).unwrap();
            assert!(
                header.get("data_length").is_none(),
                "expected no data_length for {event:?}"
            );

            let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));
            let read_back = read_event(&mut reader).await.unwrap().unwrap();
            assert_eq!(read_back, event);
        }
    }

    #[tokio::test]
    async fn test_header_has_expected_fields() {
        // AudioStart always has data (rate/width/channels are required),
        // unlike AudioStop whose only field is an optional timestamp.
        let event = Event::AudioStart(AudioStartData {
            rate: 24000,
            width: 2,
            channels: 1,
            timestamp: None,
        });
        let bytes = write_to_vec(&event).await;
        let newline = bytes.iter().position(|&b| b == b'\n').unwrap();
        let header: Value = serde_json::from_slice(&bytes[..newline]).unwrap();
        assert_eq!(header.get("type").unwrap(), "audio-start");
        assert_eq!(header.get("version").unwrap(), VERSION);
        assert!(header.get("data_length").is_some());
        assert!(header.get("payload_length").is_none());
    }

    #[tokio::test]
    async fn test_audio_chunk_payload_length_matches() {
        let audio = vec![7u8; 4096];
        let event = Event::AudioChunk {
            data: AudioChunkData {
                rate: 24000,
                width: 2,
                channels: 1,
                timestamp: None,
            },
            audio,
        };
        let bytes = write_to_vec(&event).await;
        let newline = bytes.iter().position(|&b| b == b'\n').unwrap();
        let header: Value = serde_json::from_slice(&bytes[..newline]).unwrap();
        assert_eq!(header.get("payload_length").unwrap(), 4096);
    }

    #[tokio::test]
    async fn test_inline_data_merge() {
        // Header carries {"text": "old", "extra": "kept"}, and the
        // external data segment carries {"text": "new"}. The merged
        // result should be {"text": "new", "extra": "kept"}: matching
        // keys are overwritten, inline-only keys survive. Uses an
        // unrecognized event type so the merged map round-trips as
        // `Event::Unknown` without a typed struct dropping "extra".
        let external_data = serde_json::to_vec(&serde_json::json!({"text": "new"})).unwrap();
        let header = serde_json::json!({
            "type": "test-merge",
            "version": "1.0.0",
            "data_length": external_data.len(),
            "data": {"text": "old", "extra": "kept"},
        });
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        raw.extend_from_slice(&external_data);

        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        let event = read_event(&mut reader).await.unwrap().unwrap();
        match event {
            Event::Unknown { data, .. } => {
                assert_eq!(data, serde_json::json!({"text": "new", "extra": "kept"}));
            },
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_header_too_long() {
        let mut raw = vec![b'a'; MAX_HEADER_LINE + 1];
        raw.push(b'\n');
        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_read_data_length_too_large() {
        let header = serde_json::json!({
            "type": "ping",
            "data_length": MAX_DATA_LENGTH + 1,
        });
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_read_payload_length_too_large() {
        let header = serde_json::json!({
            "type": "audio-chunk",
            "payload_length": MAX_PAYLOAD_LENGTH + 1,
        });
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_read_non_object_data_error() {
        let header = serde_json::json!({
            "type": "ping",
            "data": ["not", "an", "object"],
        });
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_read_non_object_external_data_error() {
        let external_data =
            serde_json::to_vec(&serde_json::json!(["not", "an", "object"])).unwrap();
        let header = serde_json::json!({
            "type": "ping",
            "data_length": external_data.len(),
        });
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        raw.extend_from_slice(&external_data);
        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_read_eof_returns_none() {
        let mut reader = tokio::io::BufReader::new(Cursor::new(Vec::<u8>::new()));
        assert!(read_event(&mut reader).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_read_malformed_header() {
        let mut reader = tokio::io::BufReader::new(Cursor::new(b"not json\n".to_vec()));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_read_truncated_data() {
        let header = serde_json::json!({"type": "ping", "data_length": 100});
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        raw.extend_from_slice(b"{}"); // far short of 100 bytes
        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_read_truncated_payload() {
        let header = serde_json::json!({"type": "audio-chunk", "payload_length": 100});
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        raw.extend_from_slice(&[0u8; 10]); // far short of 100 bytes
        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        assert!(read_event(&mut reader).await.is_err());
    }

    #[tokio::test]
    async fn test_multiple_events_sequential() {
        let events = vec![
            Event::Ping(PingData {
                text: Some("a".to_string()),
            }),
            Event::Pong(PongData {
                text: Some("b".to_string()),
            }),
            Event::Describe,
        ];
        let mut buf = Vec::new();
        for event in &events {
            write_event(&mut buf, event).await.unwrap();
        }
        let mut reader = tokio::io::BufReader::new(Cursor::new(buf));
        for expected in &events {
            let got = read_event(&mut reader).await.unwrap().unwrap();
            assert_eq!(&got, expected);
        }
        assert!(read_event(&mut reader).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_large_payload() {
        let audio = vec![42u8; 1024 * 1024];
        let event = Event::AudioChunk {
            data: AudioChunkData {
                rate: 24000,
                width: 2,
                channels: 1,
                timestamp: None,
            },
            audio: audio.clone(),
        };
        let bytes = write_to_vec(&event).await;
        let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));
        let read_back = read_event(&mut reader).await.unwrap().unwrap();
        match read_back {
            Event::AudioChunk { audio: got, .. } => assert_eq!(got, audio),
            _ => panic!("expected AudioChunk"),
        }
    }

    #[tokio::test]
    async fn test_unicode_data() {
        let event = Event::Error(ErrorData {
            text: "エラー 😀".to_string(),
            code: None,
        });
        let bytes = write_to_vec(&event).await;
        let mut reader = tokio::io::BufReader::new(Cursor::new(bytes));
        let read_back = read_event(&mut reader).await.unwrap().unwrap();
        assert_eq!(read_back, event);
    }

    #[tokio::test]
    async fn test_version_string_in_header() {
        let bytes = write_to_vec(&Event::Describe).await;
        let newline = bytes.iter().position(|&b| b == b'\n').unwrap();
        let header: Value = serde_json::from_slice(&bytes[..newline]).unwrap();
        assert_eq!(header.get("version").unwrap().as_str().unwrap(), VERSION);
    }

    #[tokio::test]
    async fn test_interop_python_format() {
        // Mirrors .tmp/wyoming/tests/test_event.py::test_write_event: a
        // header line followed immediately by data bytes and payload
        // bytes, with no delimiters between segments.
        let data = serde_json::json!({"text": "data"});
        let data_bytes = serde_json::to_vec(&data).unwrap();
        let payload = b"test\npayload".to_vec();
        let header = serde_json::json!({
            "type": "error",
            "version": "1.5.4",
            "data_length": data_bytes.len(),
            "payload_length": payload.len(),
        });
        let mut raw = serde_json::to_vec(&header).unwrap();
        raw.push(b'\n');
        raw.extend_from_slice(&data_bytes);
        raw.extend_from_slice(&payload);

        let mut reader = tokio::io::BufReader::new(Cursor::new(raw));
        let event = read_event(&mut reader).await.unwrap().unwrap();
        match event {
            Event::Error(ErrorData { text, .. }) => assert_eq!(text, "data"),
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
