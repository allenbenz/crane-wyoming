//! Async Wyoming protocol client.
//!
//! Connects to a running Wyoming server (TCP or Unix domain socket) and
//! drives the request/response flows described in
//! [`crate`](crate)'s module documentation: `describe`/`info`,
//! `synthesize`/`audio-start`/`audio-chunk`/`audio-stop`, and
//! `ping`/`pong`. Built on top of [`crate::wire`]'s
//! `read_event`/`write_event`, so callers never touch framing directly.

use std::ops::ControlFlow;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, BufReader};
use tokio::net::TcpStream;
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::time::timeout;

use crate::error::ProtocolError;
use crate::event::{AudioFormat, Event, InfoData, PingData, PongData, SynthesizeData};
use crate::wire::{read_event, write_event};

/// How long to wait for a trailing `error` event after `audio-stop`.
///
/// crane-wyoming's server sends `audio-stop` immediately followed by
/// `error` on the same connection when generation fails partway
/// through (see [`Client::check_trailing_error`]); on a successful
/// synthesis nothing more arrives, so this bounds how long a
/// successful call waits to confirm that. This ordering is not part
/// of the Wyoming protocol spec, which does not define any
/// relationship between `error` and `audio-stop` -- it is specific to
/// this crate's own server. Kept short since `Client` is only used
/// locally or over a LAN, not across a high-latency link.
const TRAILING_ERROR_TIMEOUT: Duration = Duration::from_millis(50);

/// Errors that can occur while using a [`Client`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// Failed to establish the underlying TCP or Unix socket connection.
    #[error("failed to connect: {0}")]
    Connect(#[source] std::io::Error),

    /// The given URI did not have a recognized `tcp://` or `unix://` scheme.
    #[error("invalid Wyoming URI (expected tcp:// or unix://): {0}")]
    InvalidUri(String),

    /// A wire-level framing or serialization error occurred.
    #[error(transparent)]
    Protocol(#[from] ProtocolError),

    /// The server responded with an [`Event::Error`].
    #[error("server error: {text}")]
    ServerError {
        /// Human-readable error message from the server.
        text: String,
        /// Optional machine-readable error code from the server.
        code: Option<String>,
    },

    /// The server sent an event of a different type than expected at
    /// this point in the protocol flow.
    #[error("expected {expected} event, got {actual}")]
    UnexpectedEvent {
        /// A description of the expected event type(s).
        expected: &'static str,
        /// The wire-format type string of the event that was received.
        actual: String,
    },

    /// The connection closed before the expected response arrived.
    #[error("connection closed unexpectedly")]
    UnexpectedEof,

    /// The `unix://` URI scheme was used on a platform that does not
    /// support Unix domain sockets.
    #[error("unix:// sockets are not supported on this platform")]
    UnixNotSupported,
}

/// The complete result of a non-streaming [`Client::synthesize`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesizeResponse {
    /// The audio format the server used, from its `audio-start` event.
    pub format: AudioFormat,
    /// The complete raw PCM audio, concatenated across all `audio-chunk`
    /// events.
    pub audio: Vec<u8>,
}

/// A connected Wyoming protocol client.
///
/// Wraps a byte stream (TCP or Unix domain socket) and offers typed
/// request/response methods for the events a Wyoming TTS server
/// supports: `describe`, `synthesize` (blob or streaming), and `ping`.
pub struct Client {
    reader: BufReader<Box<dyn AsyncRead + Unpin + Send>>,
    writer: Box<dyn AsyncWrite + Unpin + Send>,
}

impl Client {
    /// Connects to a Wyoming server at `uri`.
    ///
    /// Accepts the same URI schemes as `crane-wyoming`'s `--uri` flag:
    /// `tcp://host:port` or `unix:///path/to/socket`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::InvalidUri`] if `uri` has an unrecognized
    /// scheme, [`ClientError::UnixNotSupported`] if `uri` is a
    /// `unix://` URI on a platform without Unix domain sockets, or
    /// [`ClientError::Connect`] if the connection attempt fails.
    pub async fn connect(uri: &str) -> Result<Self, ClientError> {
        if let Some(addr) = uri.strip_prefix("tcp://") {
            let stream = TcpStream::connect(addr)
                .await
                .map_err(ClientError::Connect)?;
            return Ok(Self::from_stream(stream));
        }
        if let Some(path) = uri.strip_prefix("unix://") {
            #[cfg(unix)]
            {
                let stream = UnixStream::connect(path)
                    .await
                    .map_err(ClientError::Connect)?;
                return Ok(Self::from_stream(stream));
            }
            #[cfg(not(unix))]
            {
                let _ = path;
                return Err(ClientError::UnixNotSupported);
            }
        }
        Err(ClientError::InvalidUri(uri.to_string()))
    }

    /// Wraps an already-connected stream as a [`Client`].
    ///
    /// Useful for tests or callers that already hold a connected stream
    /// (e.g. a pre-established socket) and want to skip URI parsing.
    #[must_use]
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let (reader, writer) = tokio::io::split(stream);
        Self {
            reader: BufReader::new(Box::new(reader)),
            writer: Box::new(writer),
        }
    }

    /// Reads the next event, mapping a clean EOF to
    /// [`ClientError::UnexpectedEof`].
    async fn read_expected(&mut self) -> Result<Event, ClientError> {
        read_event(&mut self.reader)
            .await?
            .ok_or(ClientError::UnexpectedEof)
    }

    /// Maps a server [`Event::Error`] to [`ClientError::ServerError`].
    fn server_error(text: String, code: Option<String>) -> ClientError {
        ClientError::ServerError { text, code }
    }

    /// Reads one event and resolves it against an expected variant.
    ///
    /// Maps [`Event::Error`] to [`ClientError::ServerError`] and any
    /// event `extract` doesn't recognize to
    /// [`ClientError::UnexpectedEvent`] (labeled `expected`), sharing
    /// this three-way resolution across [`Client::describe`],
    /// [`Client::ping`], and [`Client::start_synthesis`].
    async fn expect_response<T>(
        &mut self,
        expected: &'static str,
        extract: impl FnOnce(Event) -> Option<T>,
    ) -> Result<T, ClientError> {
        let event = self.read_expected().await?;
        if let Event::Error(data) = event {
            return Err(Self::server_error(data.text, data.code));
        }
        let actual = event.event_type().to_string();
        extract(event).ok_or(ClientError::UnexpectedEvent { expected, actual })
    }

    /// Checks for a server error sent immediately after `audio-stop`.
    ///
    /// See [`TRAILING_ERROR_TIMEOUT`] for why this check exists --
    /// crane-wyoming's server, not the Wyoming protocol spec, is what
    /// guarantees `error` can follow `audio-stop`. Waits up to
    /// [`TRAILING_ERROR_TIMEOUT`] for more buffered data; a successful
    /// synthesis sends nothing further, so the timeout firing means
    /// success. If an event does arrive and it is [`Event::Error`],
    /// returns [`ClientError::ServerError`].
    /// [`AsyncBufReadExt::fill_buf`] is cancel-safe, so a timeout
    /// never discards bytes the peer already sent.
    async fn check_trailing_error(&mut self) -> Result<(), ClientError> {
        if timeout(TRAILING_ERROR_TIMEOUT, self.reader.fill_buf())
            .await
            .is_err()
        {
            return Ok(());
        }
        if let Event::Error(data) = self.read_expected().await? {
            return Err(Self::server_error(data.text, data.code));
        }
        Ok(())
    }

    /// Sends a `describe` request and returns the server's `info` response.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::ServerError`] if the server responds with
    /// an error, [`ClientError::UnexpectedEvent`] if it responds with
    /// anything other than `info`, or [`ClientError::Protocol`]/
    /// [`ClientError::UnexpectedEof`] on a transport failure.
    pub async fn describe(&mut self) -> Result<InfoData, ClientError> {
        write_event(&mut self.writer, &Event::Describe).await?;
        self.expect_response("info", |e| match e {
            Event::Info(data) => Some(data),
            _ => None,
        })
        .await
    }

    /// Sends a `synthesize` request and collects the full audio response.
    ///
    /// Reads `audio-start`, then all `audio-chunk` events (concatenating
    /// their PCM data), until `audio-stop`.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::ServerError`] if the server responds with
    /// an error (before or during the audio stream),
    /// [`ClientError::UnexpectedEvent`] on an out-of-sequence event, or
    /// [`ClientError::Protocol`]/[`ClientError::UnexpectedEof`] on a
    /// transport failure.
    pub async fn synthesize(
        &mut self,
        data: SynthesizeData,
    ) -> Result<SynthesizeResponse, ClientError> {
        let format = self.start_synthesis(data).await?;
        let mut audio = Vec::new();
        loop {
            match self.read_expected().await? {
                Event::AudioChunk { audio: chunk, .. } => audio.extend_from_slice(&chunk),
                Event::AudioStop(_) => break,
                Event::Error(data) => return Err(Self::server_error(data.text, data.code)),
                other => {
                    return Err(ClientError::UnexpectedEvent {
                        expected: "audio-chunk or audio-stop",
                        actual: other.event_type().to_string(),
                    });
                },
            }
        }
        self.check_trailing_error().await?;
        Ok(SynthesizeResponse { format, audio })
    }

    /// Sends a `synthesize` request and streams audio chunks to `on_chunk`
    /// as they arrive.
    ///
    /// `on_chunk` receives each chunk's raw PCM bytes and the audio
    /// format. Returning [`ControlFlow::Break`] stops delivering chunks
    /// to the callback; the client still drains and discards the
    /// remaining `audio-chunk` events up to `audio-stop` so the
    /// connection is left ready for a subsequent request.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::ServerError`] if the server responds with
    /// an error (before or during the audio stream),
    /// [`ClientError::UnexpectedEvent`] on an out-of-sequence event, or
    /// [`ClientError::Protocol`]/[`ClientError::UnexpectedEof`] on a
    /// transport failure.
    pub async fn synthesize_streaming(
        &mut self,
        data: SynthesizeData,
        mut on_chunk: impl FnMut(&[u8], &AudioFormat) -> ControlFlow<()>,
    ) -> Result<AudioFormat, ClientError> {
        let format = self.start_synthesis(data).await?;
        let mut stopped_early = false;
        loop {
            match self.read_expected().await? {
                Event::AudioChunk { audio, .. } => {
                    if !stopped_early && on_chunk(&audio, &format).is_break() {
                        stopped_early = true;
                    }
                },
                Event::AudioStop(_) => break,
                Event::Error(data) => return Err(Self::server_error(data.text, data.code)),
                other => {
                    return Err(ClientError::UnexpectedEvent {
                        expected: "audio-chunk or audio-stop",
                        actual: other.event_type().to_string(),
                    });
                },
            }
        }
        self.check_trailing_error().await?;
        Ok(format)
    }

    /// Sends a `synthesize` request and reads the leading `audio-start`
    /// response, shared by [`Client::synthesize`] and
    /// [`Client::synthesize_streaming`].
    async fn start_synthesis(&mut self, data: SynthesizeData) -> Result<AudioFormat, ClientError> {
        write_event(&mut self.writer, &Event::Synthesize(data)).await?;
        self.expect_response("audio-start", |e| match e {
            Event::AudioStart(data) => Some(AudioFormat {
                rate: data.rate,
                width: data.width,
                channels: data.channels,
            }),
            _ => None,
        })
        .await
    }

    /// Sends a `ping` request and returns the server's `pong` response.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::ServerError`] if the server responds with
    /// an error, [`ClientError::UnexpectedEvent`] if it responds with
    /// anything other than `pong`, or [`ClientError::Protocol`]/
    /// [`ClientError::UnexpectedEof`] on a transport failure.
    pub async fn ping(&mut self, data: PingData) -> Result<PongData, ClientError> {
        write_event(&mut self.writer, &Event::Ping(data)).await?;
        self.expect_response("pong", |e| match e {
            Event::Pong(data) => Some(data),
            _ => None,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use tokio::net::TcpListener;

    use super::*;
    use crate::event::{AudioChunkData, AudioStartData, AudioStopData, ErrorData};

    #[tokio::test]
    async fn connect_rejects_unknown_scheme() {
        let err = Client::connect("http://example.com").await.err().unwrap();
        assert!(matches!(err, ClientError::InvalidUri(_)));
    }

    #[tokio::test]
    async fn connect_rejects_empty_uri() {
        let err = Client::connect("").await.err().unwrap();
        assert!(matches!(err, ClientError::InvalidUri(_)));
    }

    #[tokio::test]
    async fn connect_rejects_missing_scheme() {
        let err = Client::connect("localhost:10200").await.err().unwrap();
        assert!(matches!(err, ClientError::InvalidUri(_)));
    }

    /// Spawns a mock TCP server that answers one `synthesize` request
    /// with the mid-stream error convention (`audio-start`,
    /// `audio-chunk`, `audio-stop`, `error`), then keeps answering
    /// `ping` requests with `pong` on the same connection so tests can
    /// check the connection is still usable afterwards.
    async fn spawn_mid_stream_error_server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, mut write_half) = tokio::io::split(stream);
            let mut reader = BufReader::new(read_half);

            read_event(&mut reader).await.unwrap();
            let format = AudioFormat {
                rate: 24000,
                width: 2,
                channels: 1,
            };
            write_event(
                &mut write_half,
                &Event::AudioStart(AudioStartData::new(format)),
            )
            .await
            .unwrap();
            write_event(
                &mut write_half,
                &Event::AudioChunk {
                    data: AudioChunkData::new(format),
                    audio: vec![0u8; 4],
                },
            )
            .await
            .unwrap();
            write_event(&mut write_half, &Event::AudioStop(AudioStopData::new()))
                .await
                .unwrap();
            write_event(
                &mut write_half,
                &Event::Error(ErrorData::new("mid-stream failure")),
            )
            .await
            .unwrap();

            while let Some(Event::Ping(_)) = read_event(&mut reader).await.unwrap() {
                write_event(&mut write_half, &Event::Pong(PongData::new()))
                    .await
                    .unwrap();
            }
        });
        format!("tcp://{addr}")
    }

    #[tokio::test]
    async fn synthesize_detects_mid_stream_error() {
        let uri = spawn_mid_stream_error_server().await;
        let mut client = Client::connect(&uri).await.unwrap();

        let err = client
            .synthesize(SynthesizeData::new("hi"))
            .await
            .err()
            .unwrap();

        match err {
            ClientError::ServerError { text, .. } => assert!(text.contains("mid-stream failure")),
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn synthesize_streaming_detects_mid_stream_error() {
        let uri = spawn_mid_stream_error_server().await;
        let mut client = Client::connect(&uri).await.unwrap();

        let err = client
            .synthesize_streaming(SynthesizeData::new("hi"), |_, _| ControlFlow::Continue(()))
            .await
            .err()
            .unwrap();

        match err {
            ClientError::ServerError { text, .. } => assert!(text.contains("mid-stream failure")),
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn synthesize_mid_stream_error_leaves_connection_usable() {
        let uri = spawn_mid_stream_error_server().await;
        let mut client = Client::connect(&uri).await.unwrap();

        client
            .synthesize(SynthesizeData::new("hi"))
            .await
            .err()
            .unwrap();

        client.ping(PingData::new()).await.unwrap();
    }
}
