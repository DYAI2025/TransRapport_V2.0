import argparse
import asyncio
import json
import uuid
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

try:
    import librosa  # for resampling if device doesn't support target SR
except Exception:  # pragma: no cover
    librosa = None


def pcm16le(audio: np.ndarray, scale: float = 32767.0) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * scale).astype("<i2").tobytes()


def pick_input_device(name_or_index: Optional[str]) -> int:
    # If no explicit device: prefer "BlackHole" (system/virtual loopback) if present, else default input
    if name_or_index is None:
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0 and "blackhole" in str(d.get("name", "")).lower():
                    return i
        except Exception:
            pass
        return sd.default.device[0]  # type: ignore[index]
    # numeric index
    try:
        return int(name_or_index)
    except Exception:
        pass
    # fuzzy by name
    name_ = name_or_index.lower()
    devices = sd.query_devices()
    candidates = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and name_ in str(d.get("name", "")).lower():
            candidates.append(i)
    if not candidates:
        raise RuntimeError(f"No input device matching '{name_or_index}' found")
    return candidates[0]


async def main():
    ap = argparse.ArgumentParser(description="Live mic/system capture to Serapi WS")
    ap.add_argument("--url", default="ws://127.0.0.1:8710/ws/stream")
    ap.add_argument("--sid", default=None)
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate for WS stream")
    ap.add_argument("--frame_ms", type=int, default=30)
    ap.add_argument("--device", default=None, help="Input device name substring or index (e.g. 'BlackHole')")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--idle-flush", type=float, default=None)
    ap.add_argument("--auto-start", action="store_true", help="Create session via HTTP if no --sid provided")
    ap.add_argument("--flush-end", action="store_true", help="Send a final flush on exit")
    args = ap.parse_args()

    dev_index = pick_input_device(args.device)
    dev_info = sd.query_devices(dev_index)
    # try to use target SR; fall back to device default if unsupported
    target_sr = args.sr
    device_sr = int(dev_info.get("default_samplerate") or target_sr)
    capture_sr = target_sr
    try:
        sd.check_input_settings(device=dev_index, samplerate=target_sr, channels=args.channels)
    except Exception:
        capture_sr = device_sr

    # build WS URI
    sid = args.sid or str(uuid.uuid4())
    qp = {"session_id": sid, "sr": str(target_sr)}
    if args.window is not None:
        qp["window"] = str(args.window)
    if args.idle_flush is not None:
        qp["idle_flush"] = str(args.idle_flush)
    query = "&".join([f"{k}={v}" for k, v in qp.items()])
    uri = f"{args.url}?{query}"

    # optional auto-start via HTTP derived from WS URL
    if args.auto_start and not args.sid:
        import urllib.parse, urllib.request
        parsed = urllib.parse.urlparse(args.url)
        base_http = ("https" if parsed.scheme == "wss" else "http") + "://" + parsed.netloc
        start_url = base_http + "/session/start"
        req = urllib.request.Request(start_url, data=json.dumps({"lang": None}).encode("utf-8"), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            obj = json.loads(r.read().decode("utf-8"))
            sid = obj.get("session_id") or sid
            qp["session_id"] = sid
            query = "&".join([f"{k}={v}" for k, v in qp.items()])
            uri = f"{args.url}?{query}"
            print(f"[auto-start] session_id={sid}")

    samples_per_frame = int(target_sr * (args.frame_ms / 1000.0))
    q: asyncio.Queue = asyncio.Queue(maxsize=64)

    def on_audio(indata, frames, time_info, status):  # sounddevice callback
        if status:  # pragma: no cover
            pass
        data = indata.copy()
        # mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        # resample if needed
        if capture_sr != target_sr:
            if librosa is None:
                # simple decimate/interp fallback (lower quality)
                ratio = target_sr / float(capture_sr)
                new_n = int(round(data.shape[0] * ratio))
                data = np.interp(np.linspace(0, len(data), new_n, endpoint=False), np.arange(len(data)), data)
            else:
                data = librosa.resample(data, orig_sr=capture_sr, target_sr=target_sr)
        # chunk into frame-sized pieces
        for i in range(0, len(data), samples_per_frame):
            chunk = data[i:i+samples_per_frame]
            if len(chunk) == 0:
                continue
            try:
                q.put_nowait(chunk.astype(np.float32))
            except asyncio.QueueFull:  # drop if slow consumer
                pass

    stream = sd.InputStream(
        device=dev_index,
        channels=args.channels,
        samplerate=capture_sr,
        blocksize=int(capture_sr * (args.frame_ms / 1000.0)),
        dtype="float32",
        callback=on_audio,
    )

    async with websockets.connect(uri, max_size=2**23) as ws:
        async def reader():
            while True:
                try:
                    await ws.recv()
                except Exception:
                    break

        async def writer():
            while True:
                chunk = await q.get()
                await ws.send(pcm16le(chunk))

        reader_task = asyncio.create_task(reader())
        writer_task = asyncio.create_task(writer())
        try:
            with stream:
                try:
                    dev_name = sd.query_devices(dev_index).get("name")
                except Exception:
                    dev_name = str(dev_index)
                print(f"[capture] device={dev_index} ({dev_name}) sr={capture_sr} -> target={target_sr} frame={args.frame_ms}ms")
                await asyncio.Event().wait()  # run until interrupted
        except KeyboardInterrupt:
            if args.flush_end:
                try:
                    await ws.send("flush")
                except Exception:
                    pass
        finally:
            for t in (reader_task, writer_task):
                if not t.done():
                    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
