#!/usr/bin/env python3
"""Minimal Wyoming protocol client for exercising crane-wyoming by hand.

Implements the JSONL + binary framing directly (see
src/wire.rs).

Usage:
    python tests/test_wyoming_tts.py --text "Hello from Crane" --output out.wav

    python tests/test_wyoming_tts.py --uri unix:///tmp/crane-wyoming.sock

    # Omit --output to stream WAV audio to stdout, e.g. for live playback:
    python tests/test_wyoming_tts.py --text "Hello" | pw-play -
"""

import argparse
import asyncio
import json
import struct
import sys
import wave
from urllib.parse import urlparse

# Mirrors src/wire.rs's MAX_DATA_LENGTH/MAX_PAYLOAD_LENGTH so a
# misbehaving server can't make this client allocate unbounded buffers.
MAX_DATA_LENGTH = 1024 * 1024
MAX_PAYLOAD_LENGTH = 10 * 1024 * 1024


async def write_event(writer, event_type, data=None, payload=None):
    data_bytes = json.dumps(data).encode("utf-8") if data is not None else None
    header = {"type": event_type}
    if data_bytes is not None:
        header["data_length"] = len(data_bytes)
    if payload is not None:
        header["payload_length"] = len(payload)
    writer.write(json.dumps(header).encode("utf-8") + b"\n")
    if data_bytes is not None:
        writer.write(data_bytes)
    if payload is not None:
        writer.write(payload)
    await writer.drain()


async def read_event(reader):
    """Read one event, or return None on a clean EOF."""
    line = await reader.readline()
    if not line:
        return None
    header = json.loads(line)
    data = {}
    data_length = header.get("data_length") or 0
    if data_length:
        if data_length > MAX_DATA_LENGTH:
            raise RuntimeError(f"event data_length {data_length} exceeds maximum of {MAX_DATA_LENGTH}")
        data = json.loads(await reader.readexactly(data_length))
    payload = None
    payload_length = header.get("payload_length") or 0
    if payload_length:
        if payload_length > MAX_PAYLOAD_LENGTH:
            raise RuntimeError(f"event payload_length {payload_length} exceeds maximum of {MAX_PAYLOAD_LENGTH}")
        payload = await reader.readexactly(payload_length)
    return header["type"], data, payload


async def read_event_required(reader, timeout):
    """Read one event, raising if the server times out or closes the connection."""
    try:
        result = await asyncio.wait_for(read_event(reader), timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError(f"server did not respond within {timeout}s") from None
    if result is None:
        raise ConnectionError("server closed the connection unexpectedly")
    return result


async def connect(uri):
    parsed = urlparse(uri)
    if parsed.scheme == "unix":
        return await asyncio.open_unix_connection(parsed.path)
    if parsed.scheme == "tcp":
        if parsed.port is None:
            raise ValueError("tcp:// URI requires a port (e.g. tcp://localhost:10200)")
        return await asyncio.open_connection(parsed.hostname, parsed.port)
    raise ValueError(f"unsupported URI scheme: {uri!r} (use tcp:// or unix://)")


async def describe(reader, writer, timeout):
    await write_event(writer, "describe")
    event_type, data, _ = await read_event_required(reader, timeout)
    if event_type != "info":
        raise RuntimeError(f"expected info, got {event_type}")
    print("Available voices:", file=sys.stderr)
    for program in data.get("tts", []):
        for voice in program.get("voices", []):
            print(
                f"  - {voice.get('name', '<unnamed>')} "
                f"({program.get('name', '<unnamed>')}, {voice.get('languages')})",
                file=sys.stderr,
            )
    return data


def print_languages(data):
    languages = set()
    for program in data.get("tts", []):
        for voice in program.get("voices", []):
            languages.update(voice.get("languages") or [])
    print("Available languages:", file=sys.stderr)
    for language in sorted(languages):
        print(f"  - {language}", file=sys.stderr)


def write_streaming_wav_header(stream, rate, width, channels):
    """Write a canonical WAV header with unknown-length placeholder sizes.

    `--output -` writes to a pipe, which isn't seekable, so the RIFF/data
    chunk sizes can't be patched in after the fact like `wave.open` does
    for a real file. `0xFFFFFFFF` is the conventional "read until EOF"
    placeholder aplay/pw-play/ffplay accept for streamed WAV.
    """
    byte_rate = rate * channels * width
    block_align = channels * width
    stream.write(b"RIFF")
    stream.write(struct.pack("<I", 0xFFFFFFFF))
    stream.write(b"WAVE")
    stream.write(b"fmt ")
    stream.write(struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, block_align, width * 8))
    stream.write(b"data")
    stream.write(struct.pack("<I", 0xFFFFFFFF))


async def synthesize(reader, writer, text, voice, language, output_path, timeout):
    voice_data = {"name": voice, "language": language} if voice or language else None
    await write_event(writer, "synthesize", {"text": text, "voice": voice_data})

    event_type, data, _ = await read_event_required(reader, timeout)
    if event_type != "audio-start":
        raise RuntimeError(f"expected audio-start, got {event_type}: {data}")
    rate, width, channels = data["rate"], data["width"], data["channels"]
    if rate <= 0 or width <= 0 or channels <= 0:
        raise RuntimeError(f"invalid audio-start fields: rate={rate} width={width} channels={channels}")
    print(f"audio-start: rate={rate} width={width} channels={channels}", file=sys.stderr)

    to_stdout = output_path is None
    if to_stdout:
        stdout = sys.stdout.buffer
        write_streaming_wav_header(stdout, rate, width, channels)

        def write_chunk(payload):
            stdout.write(payload)
            stdout.flush()

        def close_sink():
            pass
    else:
        wav = wave.open(output_path, "wb")
        wav.setframerate(rate)
        wav.setsampwidth(width)
        wav.setnchannels(channels)

        def write_chunk(payload):
            wav.writeframes(payload)

        def close_sink():
            wav.close()

    total_bytes = 0
    try:
        while True:
            event_type, data, payload = await read_event_required(reader, timeout)
            if event_type == "audio-stop":
                break
            if event_type == "error":
                raise RuntimeError(f"server error: {data}")
            if event_type != "audio-chunk":
                raise RuntimeError(f"unexpected event during synthesis: {event_type}")
            write_chunk(payload)
            total_bytes += len(payload)

        # A mid-stream failure sends audio-stop followed by error (see
        # WYOMING.md); peek briefly for a trailing error event.
        try:
            trailing = await asyncio.wait_for(read_event(reader), timeout=0.1)
        except asyncio.TimeoutError:
            trailing = None
        if trailing is not None and trailing[0] == "error":
            raise RuntimeError(f"server error: {trailing[1]}")
    finally:
        close_sink()

    duration = total_bytes / width / channels / rate
    dest = "stdout" if to_stdout else output_path
    print(f"wrote {dest}: {total_bytes} bytes, {duration:.2f}s", file=sys.stderr)


async def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--uri", default="tcp://127.0.0.1:10200")
    parser.add_argument("--text", default="Hello from Crane's Wyoming server.")
    parser.add_argument("--voice", default=None, help="voice name from --describe output")
    parser.add_argument("--language", default=None, help="voice language, e.g. 'en_US'")
    parser.add_argument(
        "--output",
        default=None,
        help="output WAV file; omit (or pass '-') to stream WAV audio to stdout",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="print available voices and exit, without synthesizing",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="print available voice languages and exit, without synthesizing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for a server response before giving up (default: 30)",
    )
    args = parser.parse_args()
    output_path = None if args.output in (None, "-") else args.output

    try:
        reader, writer = await connect(args.uri)
    except OSError as e:
        print(f"error: could not connect to {args.uri}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        data = await describe(reader, writer, args.timeout)
        if args.list_languages:
            print_languages(data)
        if args.list_voices or args.list_languages:
            return
        await synthesize(reader, writer, args.text, args.voice, args.language, output_path, args.timeout)
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
