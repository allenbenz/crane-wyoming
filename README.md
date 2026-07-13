# crane-wyoming: Self-hosted voice services over the Wyoming protocol

**Alpha quality software.** In its current state, this is for developers and
advanced users: it depends on unreleased, git-pinned Crane dependencies and
requires building from source. You have to expect rough edges and no
compatibility guarantees yet.

[Wyoming](https://github.com/rhasspy/wyoming) is the protocol Home Assistant
uses to talk to local voice services (wake word, speech-to-text,
text-to-speech, intent recognition) over the network: a JSON event header per
message, optionally followed by a raw binary payload (e.g. PCM audio), sent
over a TCP or Unix domain socket connection. It's how Home Assistant's local
voice pipeline stays decoupled from whatever engine actually runs each step —
but the protocol and this server are equally usable outside Home Assistant,
e.g. as a desktop TTS backend for speech-dispatcher (see below).

This repository is a Cargo workspace with three binaries built on
[Crane](https://github.com/cryptomilk/Crane)'s TTS models:

- **`crane-wyoming`** — the Wyoming protocol server; any Wyoming client can
  use a loaded Crane TTS model as its text-to-speech service, Home Assistant
  included
- **`cw-say`** — a standalone CLI client for scripting and manual use
- **`sd_crane_wyoming`** — a [speech-dispatcher](https://github.com/brailcom/speechd)
  output module, so screen readers and other speechd-based accessibility
  tools, e.g. Firefox's reader mode "Read Aloud" via speech-dispatcher on
  the desktop, can use a running `crane-wyoming` server too

## Features

- Wyoming service discovery (`describe`/`info`) — advertises every loaded
  TTS model's voices to any connecting client
- Text-to-speech synthesis (`synthesize` → `audio-start`/`audio-chunk`/
  `audio-stop`), with true incremental audio streaming on GPU (falls back to
  full-utterance synthesis on CPU, where streaming can't keep up with
  real-time playback)
- Multiple TTS models loaded at once; voice names are resolved across all of
  them, so a client can pick any loaded voice by name
- TCP or Unix domain socket listeners, including systemd socket activation
- Optional on-disk cache for repeated phrases
- speech-dispatcher integration via `sd_crane_wyoming`

## Building

```bash
cargo build --workspace --release
```

Feature flags `cuda`, `cudnn`, `mkl` forward to the identically-named
features on the `crane-engine` dependency, e.g.:

```bash
cargo build --workspace --release --features cuda
```

This produces three binaries under `target/release/`: `crane-wyoming`,
`cw-say`, and `sd_crane_wyoming`.

## Running the server

```bash
./target/release/crane-wyoming --model-path models/Voxtral-4B-TTS-2603 --uri unix:///tmp/wyoming.sock
```

Pass `--model-path` more than once to load several TTS models; the first one
becomes the default voice when a client doesn't request one by name. Use
`--uri tcp://host:port` (or plain `--host`/`--port`, default
`0.0.0.0:10200`) to listen on TCP instead of a Unix socket. See
`crane-wyoming --help` for the rest of the flags (`--cpu`, `--max-connections`,
`--tts-cache-dir`/`--tts-cache-max-size`).

### Running as a systemd service

Unit files live under `dist/systemd/` (system-wide) and `dist/systemd-user/`
(per-user). Both start `crane-wyoming` as a regular long-running service and
restart it on failure; which one to use depends on whether Home Assistant/
speech-dispatcher should have access to the server independent of any
interactive login session (system-wide), or only while a particular user
session is active (user).

**System-wide** (`dist/systemd/crane-wyoming.service`):

```bash
sudo cp dist/systemd/crane-wyoming.service /etc/systemd/system/
sudo cp dist/systemd/crane-wyoming.default /etc/default/crane-wyoming
# edit /etc/default/crane-wyoming, then:
sudo systemctl enable --now crane-wyoming
```

The service file has no `User=` set — pick a static user (add it to the
`video`/`render` group for GPU builds) or use `DynamicUser=yes`, and grant
read/write access to your model and cache paths via
`systemctl edit crane-wyoming`. See the comments in
`dist/systemd/crane-wyoming.service` for details.

**Per-user** (`dist/systemd-user/crane-wyoming.service`):

```bash
mkdir -p ~/.config/systemd/user
cp dist/systemd-user/crane-wyoming.service ~/.config/systemd/user/
cp dist/systemd-user/crane-wyoming.default ~/.config/crane-wyoming
# edit ~/.config/crane-wyoming, then:
systemctl --user enable --now crane-wyoming
```

Both `.default` files document every `CRANE_WYOMING_*` environment variable
(model path, port/host or URI, CPU-only mode, max connections, TTS cache);
any CLI flag on the command line overrides the matching variable.

### Socket activation (user service only)

Model loading is the expensive part of starting `crane-wyoming`. Socket
activation lets systemd own the listening socket and start the service lazily
on first connection, instead of keeping a model loaded in memory before
anything actually wants to speak:

```bash
cp dist/systemd-user/crane-wyoming.socket ~/.config/systemd/user/
systemctl --user enable --now crane-wyoming.socket
```

With the socket unit active, `crane-wyoming.service` doesn't need to be
started manually — connecting to `$XDG_RUNTIME_DIR/crane-wyoming/tts.sock`
(e.g. from `cw-say` or `sd_crane_wyoming`) triggers systemd to start it.
`CRANE_WYOMING_URI`/`--uri` is ignored in this mode since systemd provides
the listening socket directly.

## cw-say: standalone CLI client

A Wyoming client for manual use and scripting, independent of Home Assistant
or speech-dispatcher:

```bash
cw-say --text "Hello" --uri unix://$XDG_RUNTIME_DIR/crane-wyoming/tts.sock
cw-say --text "Hallo" --voice de_female --format wav -o output.wav
cw-say --list-voices
echo "Hello" | cw-say | pw-play -
```

- `--text` (or stdin if omitted) — text to synthesize
- `--uri` — defaults to `unix://$XDG_RUNTIME_DIR/crane-wyoming/tts.sock`, or
  `tcp://127.0.0.1:10200` if `$XDG_RUNTIME_DIR` is unset
- `--voice`, `--language` — optional, forwarded to the synthesize request
- `--output`/`-o` — file path, or stdout if omitted/`-`
- `--format` — `wav` (default) or `raw` (bare PCM)
- `--list-voices` / `--list-languages` — query the server's `describe`
  response and exit, without synthesizing
- `--timeout` — seconds to wait for a server response (default 30)

## speech-dispatcher integration

`sd_crane_wyoming` is a [speech-dispatcher](https://github.com/brailcom/speechd)
output module: speechd launches it as a subprocess and talks to it over
stdin/stdout, and it forwards synthesis requests to a running
`crane-wyoming` server over the Wyoming protocol.

1. Install the binary where speechd's other output modules live (typically
   `/usr/libexec/speech-dispatcher-modules/` or a per-user equivalent — check
   your distro's speechd package).
2. Install the example config and point it at your `crane-wyoming` socket:

   ```bash
   cp dist/speechd/crane_wyoming.conf /etc/speech-dispatcher/modules/
   # or ~/.config/speech-dispatcher/modules/ for a per-user install
   ```

   Edit the `CraneURI` directive in that file to match your server's URI
   (it defaults to the per-user socket path used by
   `dist/systemd-user/crane-wyoming.socket`).
3. Wire the module into speechd's own config (`speechd.conf`):

   ```
   AddModule "crane" "sd_crane_wyoming" "crane_wyoming.conf"
   DefaultModule crane
   ```

4. Restart speech-dispatcher and test:

   ```bash
   spd-say -o crane "Hello from crane-wyoming"
   ```

See `dist/speechd/crane_wyoming.conf` for the full set of comments,
including how to scope the module to specific languages with
`LanguageDefaultModule` instead of making it the default.

## License

MIT
