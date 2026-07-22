#!/usr/bin/env python3
"""Minimal Wyoming protocol client for exercising crane-wyoming's ASR by hand.

Implements the JSONL + binary framing directly (see
src/wire.rs).

Usage:
    python tests/test_wyoming_asr.py --input speech.wav

    python tests/test_wyoming_asr.py --uri unix:///tmp/crane-wyoming.sock --input speech.wav

    # Read WAV audio from stdin:
    arecord -f S16_LE -r 16000 -c 1 -d 5 - | python tests/test_wyoming_asr.py --input -
"""

import argparse
import array
import asyncio
import json
import sys
import wave
from urllib.parse import urlparse

# Mirrors src/wire.rs's MAX_DATA_LENGTH/MAX_PAYLOAD_LENGTH so a
# misbehaving server can't make this client allocate unbounded buffers.
MAX_DATA_LENGTH = 1024 * 1024
MAX_PAYLOAD_LENGTH = 10 * 1024 * 1024

# Bytes of PCM sent per audio-chunk event. Must stay a multiple of
# TARGET_WIDTH so chunk boundaries land on sample boundaries.
CHUNK_SIZE = 8192

# The only format crane-wyoming's ASR handler accepts (see
# EXPECTED_SAMPLE_WIDTH/EXPECTED_CHANNELS in handler.rs, and Qwen3-ASR's
# fixed 16 kHz input rate -- currently the only supported ASR model).
# Input WAV files not already in this format are resampled/downmixed to
# it before sending.
TARGET_RATE = 16000
TARGET_WIDTH = 2
TARGET_CHANNELS = 1


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
    print("Available ASR models:", file=sys.stderr)
    for program in data.get("asr", []):
        for model in program.get("models", []):
            print(
                f"  - {model.get('name', '<unnamed>')} "
                f"({program.get('name', '<unnamed>')}, {model.get('languages')})",
                file=sys.stderr,
            )
    return data


def downmix_to_mono(samples, channels):
    """Average interleaved `channels`-channel int16 `samples` down to mono."""
    if channels == 1:
        return samples
    mono = array.array("h", bytes(TARGET_WIDTH * (len(samples) // channels)))
    for i in range(len(mono)):
        frame = samples[i * channels : (i + 1) * channels]
        mono[i] = round(sum(frame) / channels)
    return mono


def resample_linear(samples, src_rate, dst_rate):
    """Linearly resample mono int16 `samples` from `src_rate` to `dst_rate`.

    Endpoint-fixed: the first and last output samples align exactly with
    the first and last input samples, so this is a time-warped stretch
    rather than a constant-step resample. That guarantees the output has
    exactly `round(len(samples) * dst_rate / src_rate)` samples, and the
    warping is inaudible for any recording of realistic length.
    """
    # A 0- or 1-sample signal carries no rate information worth resampling.
    if src_rate == dst_rate or len(samples) < 2:
        return samples
    src_len = len(samples)
    dst_len = max(1, round(src_len * dst_rate / src_rate))
    out = array.array("h", bytes(TARGET_WIDTH * dst_len))
    for i in range(dst_len):
        src_pos = i * (src_len - 1) / (dst_len - 1) if dst_len > 1 else 0.0
        idx = int(src_pos)
        frac = src_pos - idx
        a = samples[idx]
        b = samples[idx + 1] if idx + 1 < src_len else a
        out[i] = round(a + (b - a) * frac)
    return out


def prepare_pcm16(rate, width, channels, frames):
    """Convert raw WAV `frames` to `TARGET_RATE`/`TARGET_WIDTH`/`TARGET_CHANNELS`.

    Only 16-bit input is supported -- other bit depths need converting
    externally first, e.g.: ffmpeg -i input.wav -sample_fmt s16 out.wav
    """
    if width != TARGET_WIDTH:
        raise RuntimeError(
            f"unsupported sample width: {width * 8}-bit (expected {TARGET_WIDTH * 8}-bit); "
            "convert to 16-bit PCM first, e.g.: "
            "ffmpeg -i input.wav -sample_fmt s16 out.wav"
        )
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()  # WAV data is always little-endian.
    samples = downmix_to_mono(samples, channels)
    samples = resample_linear(samples, rate, TARGET_RATE)
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


async def transcribe(reader, writer, wav, model, language, timeout):
    rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
    already_target = (rate, width, channels) == (TARGET_RATE, TARGET_WIDTH, TARGET_CHANNELS)
    if not already_target:
        print(
            f"resampling: {rate}Hz/{width * 8}-bit/{channels}ch -> "
            f"{TARGET_RATE}Hz/{TARGET_WIDTH * 8}-bit/{TARGET_CHANNELS}ch",
            file=sys.stderr,
        )
    print(
        f"audio-start: rate={TARGET_RATE} width={TARGET_WIDTH} channels={TARGET_CHANNELS}",
        file=sys.stderr,
    )

    model_data = {"name": model, "language": language} if model or language else None
    await write_event(writer, "transcribe", model_data)
    await write_event(
        writer,
        "audio-start",
        {"rate": TARGET_RATE, "width": TARGET_WIDTH, "channels": TARGET_CHANNELS},
    )

    audio_data = {"rate": TARGET_RATE, "width": TARGET_WIDTH, "channels": TARGET_CHANNELS}
    if already_target:
        # Already in the target format: stream chunk-by-chunk instead of
        # buffering the whole file, matching the pre-resampling behavior.
        frames_per_chunk = max(1, CHUNK_SIZE // (width * channels))
        while True:
            frames = wav.readframes(frames_per_chunk)
            if not frames:
                break
            await write_event(writer, "audio-chunk", audio_data, payload=frames)
    else:
        pcm = prepare_pcm16(rate, width, channels, wav.readframes(wav.getnframes()))
        for offset in range(0, len(pcm), CHUNK_SIZE):
            await write_event(
                writer, "audio-chunk", audio_data, payload=pcm[offset : offset + CHUNK_SIZE]
            )
    await write_event(writer, "audio-stop")

    text = None
    while text is None:
        event_type, data, _ = await read_event_required(reader, timeout)
        if event_type == "transcript-start":
            print(f"transcript-start: language={data.get('language')}", file=sys.stderr)
        elif event_type == "transcript-chunk":
            print(f"  partial: {data.get('text')}", file=sys.stderr)
        elif event_type == "transcript":
            text = data.get("text", "")
        elif event_type == "error":
            raise RuntimeError(f"server error: {data}")
        else:
            raise RuntimeError(f"unexpected event during transcription: {event_type}")

    # Streaming mode sends a trailing transcript-stop after the final
    # transcript; batch mode does not send one at all. Peek briefly for it
    # rather than assuming either way.
    try:
        trailing = await asyncio.wait_for(read_event(reader), timeout=0.1)
    except asyncio.TimeoutError:
        trailing = None
    if trailing is not None and trailing[0] != "transcript-stop":
        raise RuntimeError(f"unexpected trailing event: {trailing[0]}")

    return text


async def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--uri", default="tcp://127.0.0.1:10200")
    parser.add_argument(
        "--input",
        default=None,
        help="input WAV file to transcribe; pass '-' to read from stdin",
    )
    parser.add_argument("--model", default=None, help="ASR model name from --describe output")
    parser.add_argument("--language", default=None, help="language hint, e.g. 'en'")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print available ASR models and exit, without transcribing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for a server response before giving up (default: 30)",
    )
    args = parser.parse_args()

    if not args.list_models and not args.input:
        parser.error("--input is required unless --list-models is given")

    try:
        reader, writer = await connect(args.uri)
    except OSError as e:
        print(f"error: could not connect to {args.uri}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        await describe(reader, writer, args.timeout)
        if args.list_models:
            return
        wav_source = sys.stdin.buffer if args.input == "-" else args.input
        with wave.open(wav_source, "rb") as wav:
            text = await transcribe(reader, writer, wav, args.model, args.language, args.timeout)
        print(text)
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
