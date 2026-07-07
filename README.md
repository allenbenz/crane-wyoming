# crane-wyoming

Wyoming protocol server for [Crane](https://github.com/cryptomilk/Crane),
letting Home Assistant use Crane's TTS models through the standard
[Wyoming](https://github.com/rhasspy/wyoming) voice integration.

**Status: work in progress.** Only the wire protocol framing (JSONL header +
binary payload read/write) is implemented so far. TTS event handling, service
discovery, and the TCP server/CLI are not yet built.

## What lives here

- `event` — typed `Event` enum and per-event data structs (`audio-start`,
  `audio-chunk`, `synthesize`, `describe`, `info`, `ping`/`pong`, `error`)
- `wire` — async `read_event`/`write_event` implementing the wire framing

## Testing

```bash
cargo test
```

## License

MIT
