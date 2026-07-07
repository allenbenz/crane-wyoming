//! Typed Wyoming event model.
//!
//! Each [`Event`] variant corresponds to a wire-format event type
//! string (e.g. `"audio-chunk"`). The [`wire`](crate::wire) module
//! handles framing; this module handles the mapping between an event's
//! `data` dict / binary payload and its typed Rust representation.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Data for an `audio-start` event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioStartData {
    /// Sample rate in Hz (e.g. 22050, 24000).
    pub rate: u32,
    /// Sample width in bytes (e.g. 2 for 16-bit PCM).
    pub width: u16,
    /// Number of audio channels (1 = mono).
    pub channels: u16,
    /// Optional timestamp in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

/// Data for an `audio-chunk` event.
///
/// The raw PCM bytes are carried in the event's binary payload, not in
/// this struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioChunkData {
    /// Sample rate in Hz.
    pub rate: u32,
    /// Sample width in bytes.
    pub width: u16,
    /// Number of audio channels.
    pub channels: u16,
    /// Optional timestamp in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

/// Data for an `audio-stop` event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioStopData {
    /// Optional timestamp in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

/// Voice specification carried in a `synthesize` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthesizeVoice {
    /// Voice name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Voice language.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Speaker within the voice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// Data for a `synthesize` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynthesizeData {
    /// Text to synthesize.
    pub text: String,
    /// Optional voice specification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<SynthesizeVoice>,
}

/// Data for a `ping` event.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PingData {
    /// Optional text to echo back in the `pong` response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Data for a `pong` event.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PongData {
    /// Text echoed from the `ping` request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Data for an `error` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorData {
    /// Human-readable error message.
    pub text: String,
    /// Optional machine-readable error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Data for an `info` event (service discovery response).
///
/// Service descriptors are stored as opaque [`serde_json::Value`]s
/// because the full schema (`Artifact` -> `TtsProgram` -> `TtsVoice`
/// -> ...) is deeply nested and only needed by callers building
/// discovery responses. Typed builders belong to a later step; this
/// type just needs to round-trip the wire format faithfully.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InfoData {
    /// TTS service descriptors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tts: Vec<serde_json::Value>,
    /// ASR service descriptors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub asr: Vec<serde_json::Value>,
    /// Wake word service descriptors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wake: Vec<serde_json::Value>,
}

/// A Wyoming protocol event.
///
/// Each variant corresponds to a wire-format event type. The
/// [`Event::Unknown`] variant preserves events with unrecognized type
/// strings for forward compatibility -- a handler can log and ignore
/// them rather than disconnecting.
///
/// `AudioChunk` and `Unknown` may carry a large `Vec<u8>` buffer;
/// avoid cloning an in-flight `Event` on a hot per-chunk path.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Audio stream has started. Carries format metadata.
    AudioStart(AudioStartData),
    /// A chunk of raw PCM audio.
    AudioChunk {
        /// Audio format and optional timestamp.
        data: AudioChunkData,
        /// Raw PCM audio bytes.
        audio: Vec<u8>,
    },
    /// Audio stream has stopped.
    AudioStop(AudioStopData),
    /// Request to synthesize speech from text.
    Synthesize(SynthesizeData),
    /// Request for service information. No data, no payload.
    Describe,
    /// Service information response.
    Info(InfoData),
    /// Keepalive request.
    Ping(PingData),
    /// Keepalive response.
    Pong(PongData),
    /// Error report.
    Error(ErrorData),
    /// An event with an unrecognized type string.
    ///
    /// Preserves the original type string, the merged data dict, and
    /// any binary payload so callers can log, forward, or ignore it
    /// without losing information.
    Unknown {
        /// The event type string from the wire.
        event_type: String,
        /// The merged data dict (inline header data + external data).
        data: serde_json::Value,
        /// Optional binary payload.
        payload: Option<Vec<u8>>,
    },
}

/// Wire-format type string for `audio-start` events.
pub const TYPE_AUDIO_START: &str = "audio-start";
/// Wire-format type string for `audio-chunk` events.
pub const TYPE_AUDIO_CHUNK: &str = "audio-chunk";
/// Wire-format type string for `audio-stop` events.
pub const TYPE_AUDIO_STOP: &str = "audio-stop";
/// Wire-format type string for `synthesize` events.
pub const TYPE_SYNTHESIZE: &str = "synthesize";
/// Wire-format type string for `describe` events.
pub const TYPE_DESCRIBE: &str = "describe";
/// Wire-format type string for `info` events.
pub const TYPE_INFO: &str = "info";
/// Wire-format type string for `ping` events.
pub const TYPE_PING: &str = "ping";
/// Wire-format type string for `pong` events.
pub const TYPE_PONG: &str = "pong";
/// Wire-format type string for `error` events.
pub const TYPE_ERROR: &str = "error";

impl Event {
    /// Returns the wire-format type string for this event.
    #[must_use]
    pub fn event_type(&self) -> &str {
        match self {
            Event::AudioStart(_) => TYPE_AUDIO_START,
            Event::AudioChunk { .. } => TYPE_AUDIO_CHUNK,
            Event::AudioStop(_) => TYPE_AUDIO_STOP,
            Event::Synthesize(_) => TYPE_SYNTHESIZE,
            Event::Describe => TYPE_DESCRIBE,
            Event::Info(_) => TYPE_INFO,
            Event::Ping(_) => TYPE_PING,
            Event::Pong(_) => TYPE_PONG,
            Event::Error(_) => TYPE_ERROR,
            Event::Unknown { event_type, .. } => event_type,
        }
    }

    /// Serializes this event's `data` dict to JSON bytes.
    ///
    /// Returns `Ok(None)` for events with no data (e.g. [`Event::Describe`]
    /// or an [`Event::Unknown`] with a `null`/empty data value).
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails.
    pub(crate) fn serialize_data(&self) -> Result<Option<Vec<u8>>> {
        let value = match self {
            Event::AudioStart(data) => Some(serde_json::to_vec(data)?),
            Event::AudioChunk { data, .. } => Some(serde_json::to_vec(data)?),
            Event::AudioStop(data) => Some(serde_json::to_vec(data)?),
            Event::Synthesize(data) => Some(serde_json::to_vec(data)?),
            Event::Describe => None,
            Event::Info(data) => Some(serde_json::to_vec(data)?),
            Event::Ping(data) => Some(serde_json::to_vec(data)?),
            Event::Pong(data) => Some(serde_json::to_vec(data)?),
            Event::Error(data) => Some(serde_json::to_vec(data)?),
            Event::Unknown { data, .. } => {
                if data.is_null() {
                    None
                } else {
                    Some(serde_json::to_vec(data)?)
                }
            },
        };
        // An empty object serializes to non-empty bytes ("{}") but carries
        // no information; the Wyoming wire format omits data_length/the
        // data segment entirely when there is nothing to send.
        Ok(value.filter(|bytes| bytes.as_slice() != b"{}"))
    }

    /// Returns the binary payload for this event, if any.
    #[must_use]
    pub(crate) fn payload(&self) -> Option<&[u8]> {
        match self {
            Event::AudioChunk { audio, .. } => Some(audio),
            Event::Unknown { payload, .. } => payload.as_deref(),
            _ => None,
        }
    }

    /// Constructs an [`Event`] from parsed wire components.
    ///
    /// Dispatches on `event_type` and deserializes `data` into the
    /// matching typed struct. Unrecognized type strings become
    /// [`Event::Unknown`] rather than an error, so servers can tolerate
    /// events from newer protocol versions.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` does not match the schema expected
    /// for a recognized `event_type`.
    pub(crate) fn from_wire(
        event_type: &str,
        data: serde_json::Value,
        payload: Option<Vec<u8>>,
    ) -> Result<Self> {
        // No data segment on the wire (or an explicit `null`) means "no
        // fields set", equivalent to an empty object -- not the absence
        // of a valid data value. Structs whose fields are all optional
        // (e.g. `PingData`, `AudioStopData`) must deserialize from this.
        let data = if data.is_null() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            data
        };
        Ok(match event_type {
            TYPE_AUDIO_START => Event::AudioStart(serde_json::from_value(data)?),
            TYPE_AUDIO_CHUNK => Event::AudioChunk {
                data: serde_json::from_value(data)?,
                audio: payload.unwrap_or_default(),
            },
            TYPE_AUDIO_STOP => Event::AudioStop(serde_json::from_value(data)?),
            TYPE_SYNTHESIZE => Event::Synthesize(serde_json::from_value(data)?),
            TYPE_DESCRIBE => Event::Describe,
            TYPE_INFO => Event::Info(serde_json::from_value(data)?),
            TYPE_PING => Event::Ping(serde_json::from_value(data)?),
            TYPE_PONG => Event::Pong(serde_json::from_value(data)?),
            TYPE_ERROR => Event::Error(serde_json::from_value(data)?),
            other => Event::Unknown {
                event_type: other.to_string(),
                data,
                payload,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip(event: Event) -> Event {
        let event_type = event.event_type().to_string();
        let data_bytes = event.serialize_data().expect("serialize_data");
        let payload = event.payload().map(<[u8]>::to_vec);
        let data_value = match data_bytes {
            Some(bytes) => serde_json::from_slice(&bytes).expect("data json"),
            None => serde_json::Value::Null,
        };
        Event::from_wire(&event_type, data_value, payload).expect("from_wire")
    }

    #[test]
    fn test_audio_start_round_trip() {
        let event = Event::AudioStart(AudioStartData {
            rate: 22050,
            width: 2,
            channels: 1,
            timestamp: Some(123),
        });
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn test_audio_start_no_timestamp() {
        let event = Event::AudioStart(AudioStartData {
            rate: 24000,
            width: 2,
            channels: 1,
            timestamp: None,
        });
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn test_audio_chunk_round_trip() {
        let event = Event::AudioChunk {
            data: AudioChunkData {
                rate: 24000,
                width: 2,
                channels: 1,
                timestamp: None,
            },
            audio: vec![1, 2, 3, 4],
        };
        let result = round_trip(event.clone());
        assert_eq!(result, event);
        match result {
            Event::AudioChunk { audio, .. } => assert_eq!(audio, vec![1, 2, 3, 4]),
            _ => panic!("expected AudioChunk"),
        }
    }

    #[test]
    fn test_audio_chunk_data_excludes_audio_bytes() {
        let event = Event::AudioChunk {
            data: AudioChunkData {
                rate: 24000,
                width: 2,
                channels: 1,
                timestamp: None,
            },
            audio: vec![9, 9, 9],
        };
        let bytes = event.serialize_data().unwrap().unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(!text.contains('9'));
    }

    #[test]
    fn test_audio_stop_round_trip() {
        let event = Event::AudioStop(AudioStopData {
            timestamp: Some(42),
        });
        assert_eq!(round_trip(event.clone()), event);

        let event_no_ts = Event::AudioStop(AudioStopData { timestamp: None });
        assert_eq!(round_trip(event_no_ts.clone()), event_no_ts);
    }

    #[test]
    fn test_synthesize_round_trip_full_voice() {
        let event = Event::Synthesize(SynthesizeData {
            text: "hello world".to_string(),
            voice: Some(SynthesizeVoice {
                name: Some("Chelsie".to_string()),
                language: Some("en".to_string()),
                speaker: Some("default".to_string()),
            }),
        });
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn test_synthesize_minimal() {
        let event = Event::Synthesize(SynthesizeData {
            text: "hi".to_string(),
            voice: None,
        });
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn test_describe_round_trip() {
        let event = Event::Describe;
        assert_eq!(event.serialize_data().unwrap(), None);
        assert_eq!(event.payload(), None);
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn test_info_round_trip() {
        let event = Event::Info(InfoData {
            tts: vec![json!({"name": "Qwen3-TTS"})],
            asr: vec![],
            wake: vec![],
        });
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn test_ping_pong_round_trip() {
        let ping = Event::Ping(PingData {
            text: Some("hi".to_string()),
        });
        assert_eq!(round_trip(ping.clone()), ping);

        let pong = Event::Pong(PongData { text: None });
        assert_eq!(round_trip(pong.clone()), pong);
    }

    #[test]
    fn test_error_round_trip() {
        let event = Event::Error(ErrorData {
            text: "bad voice".to_string(),
            code: Some("voice-not-found".to_string()),
        });
        assert_eq!(round_trip(event.clone()), event);

        let event_no_code = Event::Error(ErrorData {
            text: "oops".to_string(),
            code: None,
        });
        assert_eq!(round_trip(event_no_code.clone()), event_no_code);
    }

    #[test]
    fn test_unknown_event_preserves_data_and_payload() {
        let event = Event::Unknown {
            event_type: "future-event".to_string(),
            data: json!({"foo": "bar"}),
            payload: Some(vec![5, 6, 7]),
        };
        assert_eq!(round_trip(event.clone()), event);
    }

    #[test]
    fn test_unknown_event_no_data() {
        let event = Event::Unknown {
            event_type: "future-event".to_string(),
            data: json!(null),
            payload: None,
        };
        assert_eq!(event.serialize_data().unwrap(), None);
        // Round-tripping through the wire normalizes an absent/null data
        // segment to an empty object, matching the Python reference's
        // dict-default semantics (see `Event::from_wire`).
        let expected = Event::Unknown {
            event_type: "future-event".to_string(),
            data: json!({}),
            payload: None,
        };
        assert_eq!(round_trip(event), expected);
    }

    #[test]
    fn test_event_type_strings() {
        assert_eq!(Event::Describe.event_type(), "describe");
        assert_eq!(
            Event::AudioStop(AudioStopData { timestamp: None }).event_type(),
            "audio-stop"
        );
    }
}
