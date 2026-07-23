//! Wyoming protocol wire format and event types.
//!
//! Implements the JSONL + binary framing used by the
//! [Wyoming protocol](https://github.com/rhasspy/wyoming) for
//! Home Assistant voice integration. This crate handles serialization
//! and deserialization of Wyoming events over async byte streams
//! (`tokio::io::AsyncRead` / `AsyncWrite`).
//!
//! # Wire format
//!
//! Each message consists of three parts:
//!
//! 1. **JSON header line** (newline-terminated) -- type, version,
//!    `data_length`, `payload_length`
//! 2. **Extended data** (`data_length` bytes) -- UTF-8 JSON of the
//!    event's data dict
//! 3. **Binary payload** (`payload_length` bytes) -- raw bytes (PCM
//!    audio for `audio-chunk` events)
//!
//! The protocol is symmetric: both client and server use the same
//! framing.
//!
//! # Event scope
//!
//! This crate intentionally models only the subset of Wyoming events
//! needed for TTS, ASR, and VAD service integration: `synthesize`,
//! `audio-start`, `audio-chunk`, `audio-stop`, `voice-started`,
//! `voice-stopped`, `transcribe`, `transcript`, `transcript-start`,
//! `transcript-chunk`, `transcript-stop`, `describe`/`info`, `ping`/
//! `pong`, and `error`. Wake-word and intent events are not included.
//! Unrecognized event types are preserved as [`Event::Unknown`] for
//! forward compatibility.
//!
//! # Module layout
//!
//! | Module   | Responsibility                                    |
//! |----------|----------------------------------------------------|
//! | `error`  | `ProtocolError` type for wire protocol failures    |
//! | `event`  | Typed event enum and per-event data structs        |
//! | `wire`   | Async read/write functions for the wire protocol   |
//! | `client` | Async client for connecting to a Wyoming server    |

pub mod client;
pub mod error;
pub mod event;
pub mod wire;

pub use client::{Client, ClientError, SynthesizeResponse};
pub use error::{IoStage, ProtocolError};
pub use event::{AudioFormat, Event};
pub use wire::{MAX_DATA_LENGTH, MAX_HEADER_LINE, MAX_PAYLOAD_LENGTH, read_event, write_event};
