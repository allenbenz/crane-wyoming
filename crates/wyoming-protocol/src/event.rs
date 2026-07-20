//! Typed Wyoming event model.
//!
//! Each [`Event`] variant corresponds to a wire-format event type
//! string (e.g. `"audio-chunk"`). The [`wire`](crate::wire) module
//! handles framing; this module handles the mapping between an event's
//! `data` dict / binary payload and its typed Rust representation.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::ProtocolError;

/// Serializes `data` to JSON bytes, mapping the error to
/// [`ProtocolError::Serialization`].
pub(crate) fn to_json_vec(data: &impl Serialize) -> Result<Vec<u8>, ProtocolError> {
    serde_json::to_vec(data).map_err(ProtocolError::Serialization)
}

/// Deserializes a JSON value into `T`, mapping the error to
/// [`ProtocolError::InvalidEventData`].
fn from_json_value<T: DeserializeOwned>(data: serde_json::Value) -> Result<T, ProtocolError> {
    serde_json::from_value(data).map_err(ProtocolError::InvalidEventData)
}

/// PCM audio sample format shared by `audio-start` and `audio-chunk` events.
///
/// Groups the three fields that describe an audio format into a single
/// struct with named fields, so constructing [`AudioStartData`] or
/// [`AudioChunkData`] cannot accidentally transpose `width` and
/// `channels` (both `u16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    /// Sample rate in Hz (e.g. 22050, 24000).
    pub rate: u32,
    /// Sample width in bytes (e.g. 2 for 16-bit PCM).
    pub width: u16,
    /// Number of audio channels (1 = mono).
    pub channels: u16,
}

/// Data for an `audio-start` event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
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

impl AudioStartData {
    /// Creates a new `AudioStartData` with the given audio format.
    ///
    /// Optional fields (`timestamp`) default to `None`.
    #[must_use]
    pub fn new(format: AudioFormat) -> Self {
        Self {
            rate: format.rate,
            width: format.width,
            channels: format.channels,
            timestamp: None,
        }
    }
}

/// Data for an `audio-chunk` event.
///
/// The raw PCM bytes are carried in the event's binary payload, not in
/// this struct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
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

impl AudioChunkData {
    /// Creates a new `AudioChunkData` with the given audio format.
    ///
    /// Optional fields (`timestamp`) default to `None`.
    #[must_use]
    pub fn new(format: AudioFormat) -> Self {
        Self {
            rate: format.rate,
            width: format.width,
            channels: format.channels,
            timestamp: None,
        }
    }
}

/// Data for an `audio-stop` event.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AudioStopData {
    /// Optional timestamp in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
}

impl AudioStopData {
    /// Creates a new `AudioStopData` with no timestamp.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Voice specification carried in a `synthesize` event.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
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

impl SynthesizeVoice {
    /// Creates a new `SynthesizeVoice` with all fields unset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a `SynthesizeVoice` with the given voice name.
    #[must_use]
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Self::default()
        }
    }
}

/// Data for a `synthesize` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SynthesizeData {
    /// Text to synthesize.
    pub text: String,
    /// Optional voice specification.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<SynthesizeVoice>,
}

impl SynthesizeData {
    /// Creates a new `SynthesizeData` with the given text and no voice.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: None,
        }
    }

    /// Sets the voice specification for this synthesize request.
    #[must_use]
    pub fn with_voice(mut self, voice: SynthesizeVoice) -> Self {
        self.voice = Some(voice);
        self
    }
}

/// Data for a `transcribe` event (ASR request).
///
/// `context` and `vad_sensitivity` from the upstream Wyoming schema are
/// not yet modeled; extend this struct when ASR context/VAD support
/// lands.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TranscribeData {
    /// Model name to use for transcription.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Language hint for transcription.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

impl TranscribeData {
    /// Creates a new `TranscribeData` with no fields set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model name to use for transcription.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the language hint for transcription.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

/// Data for a `transcript` event (ASR result).
///
/// `context` from the upstream Wyoming schema is not yet modeled; extend
/// this struct when ASR context support lands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TranscriptData {
    /// Transcribed text.
    pub text: String,
    /// Language of the transcription.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

impl TranscriptData {
    /// Creates a new `TranscriptData` with the given text and no language.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
        }
    }

    /// Sets the language of the transcription.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

/// Data for a `ping` event.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PingData {
    /// Optional text to echo back in the `pong` response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl PingData {
    /// Creates a new `PingData` with no text.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a `PingData` with the given text.
    #[must_use]
    pub fn with_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
        }
    }
}

/// Data for a `pong` event.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PongData {
    /// Text echoed from the `ping` request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl PongData {
    /// Creates a new `PongData` with no text.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a `PongData` with the given text.
    #[must_use]
    pub fn with_text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
        }
    }
}

/// Data for an `error` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ErrorData {
    /// Human-readable error message.
    pub text: String,
    /// Optional machine-readable error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorData {
    /// Creates a new `ErrorData` with the given message and no error code.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            code: None,
        }
    }

    /// Sets the machine-readable error code.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

/// Data for an `info` event (service discovery response).
///
/// Service descriptors are stored as opaque [`serde_json::Value`]s
/// because the full schema (`Artifact` -> `TtsProgram` -> `TtsVoice`
/// -> ...) is deeply nested and only needed by callers building
/// discovery responses. Typed builders belong to a later step; this
/// type just needs to round-trip the wire format faithfully.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
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

impl InfoData {
    /// Creates a new `InfoData` with empty service descriptor lists.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the TTS service descriptors.
    #[must_use]
    pub fn with_tts(mut self, tts: Vec<serde_json::Value>) -> Self {
        self.tts = tts;
        self
    }
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
    /// Request to transcribe speech to text.
    Transcribe(TranscribeData),
    /// Result of transcribing speech to text.
    Transcript(TranscriptData),
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
/// Wire-format type string for `transcribe` events.
pub const TYPE_TRANSCRIBE: &str = "transcribe";
/// Wire-format type string for `transcript` events.
pub const TYPE_TRANSCRIPT: &str = "transcript";
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
            Event::Transcribe(_) => TYPE_TRANSCRIBE,
            Event::Transcript(_) => TYPE_TRANSCRIPT,
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
    pub(crate) fn serialize_data(&self) -> Result<Option<Vec<u8>>, ProtocolError> {
        let value = match self {
            Event::AudioStart(data) => Some(to_json_vec(data)?),
            Event::AudioChunk { data, .. } => Some(to_json_vec(data)?),
            Event::AudioStop(data) => Some(to_json_vec(data)?),
            Event::Synthesize(data) => Some(to_json_vec(data)?),
            Event::Transcribe(data) => Some(to_json_vec(data)?),
            Event::Transcript(data) => Some(to_json_vec(data)?),
            Event::Describe => None,
            Event::Info(data) => Some(to_json_vec(data)?),
            Event::Ping(data) => Some(to_json_vec(data)?),
            Event::Pong(data) => Some(to_json_vec(data)?),
            Event::Error(data) => Some(to_json_vec(data)?),
            Event::Unknown { data, .. } => {
                if data.is_null() {
                    None
                } else {
                    Some(to_json_vec(data)?)
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
    ) -> Result<Self, ProtocolError> {
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
            TYPE_AUDIO_START => Event::AudioStart(from_json_value(data)?),
            TYPE_AUDIO_CHUNK => Event::AudioChunk {
                data: from_json_value(data)?,
                audio: payload.unwrap_or_default(),
            },
            TYPE_AUDIO_STOP => Event::AudioStop(from_json_value(data)?),
            TYPE_SYNTHESIZE => Event::Synthesize(from_json_value(data)?),
            TYPE_TRANSCRIBE => Event::Transcribe(from_json_value(data)?),
            TYPE_TRANSCRIPT => Event::Transcript(from_json_value(data)?),
            TYPE_DESCRIBE => Event::Describe,
            TYPE_INFO => Event::Info(from_json_value(data)?),
            TYPE_PING => Event::Ping(from_json_value(data)?),
            TYPE_PONG => Event::Pong(from_json_value(data)?),
            TYPE_ERROR => Event::Error(from_json_value(data)?),
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

    fn round_trip(event: &Event) -> Event {
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
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_audio_start_no_timestamp() {
        let event = Event::AudioStart(AudioStartData {
            rate: 24000,
            width: 2,
            channels: 1,
            timestamp: None,
        });
        assert_eq!(round_trip(&event), event);
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
        let result = round_trip(&event);
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
        assert_eq!(round_trip(&event), event);

        let event_no_ts = Event::AudioStop(AudioStopData { timestamp: None });
        assert_eq!(round_trip(&event_no_ts), event_no_ts);
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
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_synthesize_minimal() {
        let event = Event::Synthesize(SynthesizeData {
            text: "hi".to_string(),
            voice: None,
        });
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_transcribe_round_trip() {
        let event = Event::Transcribe(TranscribeData {
            name: Some("Qwen3-ASR".to_string()),
            language: Some("en".to_string()),
        });
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_transcribe_minimal() {
        let event = Event::Transcribe(TranscribeData::default());
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_transcribe_name_only() {
        let event = Event::Transcribe(TranscribeData {
            name: Some("Qwen3-ASR".to_string()),
            language: None,
        });
        assert_eq!(round_trip(&event), event);
        let bytes = event.serialize_data().unwrap().unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("name"));
        assert!(!text.contains("language"));
    }

    #[test]
    fn test_transcript_round_trip() {
        let event = Event::Transcript(TranscriptData {
            text: "hello world".to_string(),
            language: Some("en".to_string()),
        });
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_transcript_minimal() {
        let event = Event::Transcript(TranscriptData::new("hi"));
        assert_eq!(
            event,
            Event::Transcript(TranscriptData {
                text: "hi".to_string(),
                language: None,
            })
        );
        assert_eq!(round_trip(&event), event);
        let bytes = event.serialize_data().unwrap().unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains("text"));
        assert!(!text.contains("language"));
    }

    #[test]
    fn test_describe_round_trip() {
        let event = Event::Describe;
        assert_eq!(event.serialize_data().unwrap(), None);
        assert_eq!(event.payload(), None);
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_info_round_trip() {
        let event = Event::Info(InfoData {
            tts: vec![json!({"name": "Qwen3-TTS"})],
            asr: vec![],
            wake: vec![],
        });
        assert_eq!(round_trip(&event), event);
    }

    #[test]
    fn test_ping_pong_round_trip() {
        let keepalive_request = Event::Ping(PingData {
            text: Some("hi".to_string()),
        });
        assert_eq!(round_trip(&keepalive_request), keepalive_request);

        let keepalive_reply = Event::Pong(PongData { text: None });
        assert_eq!(round_trip(&keepalive_reply), keepalive_reply);
    }

    #[test]
    fn test_error_round_trip() {
        let event = Event::Error(ErrorData {
            text: "bad voice".to_string(),
            code: Some("voice-not-found".to_string()),
        });
        assert_eq!(round_trip(&event), event);

        let event_no_code = Event::Error(ErrorData {
            text: "oops".to_string(),
            code: None,
        });
        assert_eq!(round_trip(&event_no_code), event_no_code);
    }

    #[test]
    fn test_unknown_event_preserves_data_and_payload() {
        let event = Event::Unknown {
            event_type: "future-event".to_string(),
            data: json!({"foo": "bar"}),
            payload: Some(vec![5, 6, 7]),
        };
        assert_eq!(round_trip(&event), event);
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
        assert_eq!(round_trip(&event), expected);
    }

    #[test]
    fn test_event_type_strings() {
        assert_eq!(Event::Describe.event_type(), "describe");
        assert_eq!(
            Event::AudioStop(AudioStopData { timestamp: None }).event_type(),
            "audio-stop"
        );
        assert_eq!(
            Event::Transcribe(TranscribeData::default()).event_type(),
            "transcribe"
        );
        assert_eq!(
            Event::Transcript(TranscriptData::new("hi")).event_type(),
            "transcript"
        );
    }
}
