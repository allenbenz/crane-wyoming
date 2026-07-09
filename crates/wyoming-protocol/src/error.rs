//! Error types for the Wyoming wire protocol.

/// Which I/O operation within the wire protocol failed.
///
/// Attached to [`ProtocolError::Io`] so error messages and log lines
/// identify the protocol stage (header read, data write, flush, etc.)
/// rather than just reporting a bare I/O error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoStage {
    /// Reading the newline-terminated header line.
    ReadHeader,
    /// Reading the extended-data JSON segment.
    ReadData,
    /// Reading the binary payload segment.
    ReadPayload,
    /// Writing the header line.
    WriteHeader,
    /// Writing the extended-data JSON segment.
    WriteData,
    /// Writing the binary payload segment.
    WritePayload,
    /// Flushing the writer after a complete event.
    Flush,
}

impl std::fmt::Display for IoStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ReadHeader => "header read",
            Self::ReadData => "data read",
            Self::ReadPayload => "payload read",
            Self::WriteHeader => "header write",
            Self::WriteData => "data write",
            Self::WritePayload => "payload write",
            Self::Flush => "flush",
        })
    }
}

/// Returns a human-readable label for a JSON value's type.
fn value_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Errors that can occur while reading or writing Wyoming protocol events.
///
/// Each variant corresponds to a distinct failure mode in the wire
/// protocol. I/O errors carry the [`IoStage`] they occurred in; other
/// protocol-level violations (malformed headers, length limits, schema
/// mismatches) each get a dedicated variant so callers can react to
/// specific failures without string-matching.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ProtocolError {
    /// An I/O error occurred while reading from or writing to the transport.
    #[error("I/O error during {stage}: {source}")]
    Io {
        /// The underlying I/O error.
        source: std::io::Error,
        /// Which protocol stage the error occurred in.
        stage: IoStage,
    },

    /// The header line exceeded the maximum allowed length.
    #[error("header line exceeds maximum length of {max} bytes")]
    HeaderTooLong {
        /// The maximum allowed header length in bytes.
        max: usize,
    },

    /// The header line contained invalid UTF-8.
    #[error("header is not valid UTF-8: {0}")]
    HeaderNotUtf8(#[source] std::string::FromUtf8Error),

    /// The header line was valid UTF-8 but not valid JSON.
    #[error("header is not valid JSON: {0}")]
    HeaderInvalidJson(#[source] serde_json::Error),

    /// The declared `data_length` exceeds the maximum allowed value.
    #[error("event data_length {length} exceeds maximum of {max}")]
    DataLengthExceeded {
        /// The declared data length.
        length: usize,
        /// The maximum allowed data length.
        max: usize,
    },

    /// The declared `payload_length` exceeds the maximum allowed value.
    #[error("event payload_length {length} exceeds maximum of {max}")]
    PayloadLengthExceeded {
        /// The declared payload length.
        length: usize,
        /// The maximum allowed payload length.
        max: usize,
    },

    /// A `data` value (inline or external) was not a JSON object.
    ///
    /// Carries the offending value (boxed, since [`serde_json::Value`]
    /// can be arbitrarily large) so error messages can report what was
    /// received.
    #[error("expected event data to be a JSON object, got {}", value_kind(.0))]
    NonObjectData(Box<serde_json::Value>),

    /// The data dict did not match the schema expected for the event type.
    #[error("invalid event data: {0}")]
    InvalidEventData(#[source] serde_json::Error),

    /// Serialization of an event's data dict or header failed.
    #[error("event serialization failed: {0}")]
    Serialization(#[source] serde_json::Error),
}
