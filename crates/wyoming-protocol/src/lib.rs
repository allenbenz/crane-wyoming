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
//! # Module layout
//!
//! | Module  | Responsibility                                    |
//! |---------|----------------------------------------------------|
//! | `event` | Typed event enum and per-event data structs        |
//! | `wire`  | Async read/write functions for the wire protocol   |

pub mod event;
pub mod wire;

pub use event::Event;
pub use wire::{MAX_DATA_LENGTH, MAX_HEADER_LINE, MAX_PAYLOAD_LENGTH, read_event, write_event};
