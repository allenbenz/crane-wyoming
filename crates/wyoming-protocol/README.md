# wyoming-protocol

An async Rust implementation of the [Wyoming protocol](https://github.com/rhasspy/wyoming)
wire format: a newline-terminated JSON event header, an optional JSON data
segment, and an optional raw binary payload (e.g. PCM audio), read and
written over `tokio::io::AsyncRead`/`AsyncWrite`.

This crate only depends on `tokio` (`io-util`), `serde`, `serde_json`, and
`anyhow`.

## Scope

Currently implements the TTS-relevant subset of the Wyoming event catalog plus
the shared control events: `synthesize`, `audio-start`/`audio-chunk`/
`audio-stop`, `describe`/`info`, `ping`/`pong`, and `error`. ASR, wake word,
and other Wyoming domains aren't modeled yet; unrecognized event types
deserialize to `Event::Unknown` instead of failing, so unmodeled events pass
through rather than breaking the connection.

## Modules

- `event` — the typed `Event` enum and per-event data structs
- `wire` — `read_event`/`write_event`, the async framing layer

## Used by

[`crane-wyoming`](https://github.com/cryptomilk/crane-wyoming) (the Wyoming TTS
server) depends on this crate for its wire protocol implementation.
