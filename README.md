# crane-wyoming: Self-hosted voice services for Home Assistant

**Alpha quality software.** In its current state, this is for developers and
advanced users: it depends on unreleased, git-pinned Crane dependencies and
requires building from source. You have to expect rough edges and no
compatibility guarantees yet.

[Wyoming](https://github.com/rhasspy/wyoming) is the protocol Home Assistant
uses to talk to local voice services (wake word, speech-to-text,
text-to-speech, intent recognition) over the network: a JSON event header per
message, optionally followed by a raw binary payload (e.g. PCM audio), sent
over a TCP or Unix domain socket connection. It's how Home Assistant's local
voice pipeline stays decoupled from whatever engine actually runs each step.

`crane-wyoming` implements the server side of that protocol for
[Crane](https://github.com/cryptomilk/Crane)'s TTS models, so Home Assistant
can use them as its text-to-speech service.

## Features

- Wyoming service discovery (`describe`/`info`) — advertises every loaded
  TTS model's voices to Home Assistant
- Text-to-speech synthesis (`synthesize` → `audio-start`/`audio-chunk`/
  `audio-stop`), with true incremental audio streaming on GPU (falls back to
  full-utterance synthesis on CPU, where streaming can't keep up with
  real-time playback)
- Multiple TTS models loaded at once; voice names are resolved across all of
  them, so Home Assistant can pick any loaded voice by name
- TCP or Unix domain socket listeners
- Optional on-disk cache for repeated phrases

## Running the server

```bash
cargo build --release

./target/release/crane-wyoming --model-path models/Voxtral-4B-TTS-2603 --uri unix:///tmp/wyoming.sock
```

Pass `--model-path` more than once to load several TTS models; the first one
becomes the default voice when a client doesn't request one by name. Use
`--uri tcp://host:port` (or plain `--host`/`--port`, default
`0.0.0.0:10200`) to listen on TCP instead of a Unix socket. See
`crane-wyoming --help` for the rest of the flags (`--cpu`, `--max-connections`,
`--tts-cache-dir`/`--tts-cache-max-size`).

## Running the test client

[`tests/test_wyoming_tts.py`](tests/test_wyoming_tts.py) is a
dependency-free Wyoming client (implements the JSONL + binary framing
directly) for exercising a running server by hand.

```bash
./tests/test_wyoming_tts.py --uri unix:///tmp/wyoming.sock --voice de_female --text "Wuff" | pw-play -
```

Omit `--output` (or pass `--output -`) to stream the synthesized WAV to
stdout, as above; pass `--output out.wav` to write it to a file instead.
Other useful flags:

- `--list-voices` / `--list-languages` — query the server's `describe`
  response and exit, without synthesizing
- `--voice` — voice name from `--list-voices` output
- `--language` — voice language, e.g. `en_US`
- `--timeout` — seconds to wait for a server response (default 30)

## Testing

```bash
cargo test
```

## License

MIT
