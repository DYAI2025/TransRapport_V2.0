import argparse, asyncio, websockets, soundfile as sf, numpy as np, json, uuid, urllib.parse, urllib.request

def pcm16le(audio, scale=32767.0):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * scale).astype("<i2").tobytes()

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="Pfad zu WAV/FLAC/OGG/MP3 (wird auf 16k mono resampelt, wenn nötig)")
    ap.add_argument("--url", default="ws://127.0.0.1:8710/ws/stream")
    ap.add_argument("--sid", default=None)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--frame_ms", type=int, default=30)
    ap.add_argument("--window", type=float, default=None, help="Fenstergröße in Sekunden (optional)")
    ap.add_argument("--idle-flush", type=float, default=None, help="Idle-Flush-Sekunden (optional)")
    ap.add_argument("--no-flush-end", action="store_true", help="Kein Flush am Ende senden")
    ap.add_argument("--flush-json", action="store_true", help="Flush als JSON senden statt Text")
    ap.add_argument("--post-wait", type=float, default=2.0, help="Sekunden nach letztem Flush warten")
    ap.add_argument("--auto-start", action="store_true", help="Erzeuge Session automatisch via HTTP und nutze deren ID")
    args = ap.parse_args()

    audio, sr = sf.read(args.wav, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != args.sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sr)
        sr = args.sr

    samples_per_frame = int(sr * (args.frame_ms / 1000.0))

    # Auto-start session if requested
    sid = args.sid
    if args.auto_start and not sid:
        parsed = urllib.parse.urlparse(args.url)
        base_http = ("https" if parsed.scheme == "wss" else "http") + "://" + parsed.netloc
        start_url = base_http + "/session/start"
        req = urllib.request.Request(start_url, data=json.dumps({"lang": None}).encode("utf-8"), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            obj = json.loads(r.read().decode("utf-8"))
            sid = obj.get("session_id")
            print(f"[auto-start] session_id={sid}")
    if not sid:
        sid = str(uuid.uuid4())

    # Build WS URI with optional params
    qp = {"session_id": sid, "sr": str(sr)}
    if args.window is not None:
        qp["window"] = str(args.window)
    if args.idle_flush is not None:
        qp["idle_flush"] = str(args.idle_flush)
    query = "&".join([f"{k}={v}" for k, v in qp.items()])
    uri = f"{args.url}?{query}"
    async with websockets.connect(uri, max_size=2**23) as ws:
        # Server-Events lesen (optional)
        async def reader():
            while True:
                try:
                    msg = await ws.recv()
                    try:
                        evt = json.loads(msg)
                        if evt.get("type") == "partial_transcript":
                            print(f"[partial] +{len(evt.get('added', []))} segs, total={evt.get('total_segments')}")
                    except Exception:
                        pass
                except Exception:
                    break
        asyncio.create_task(reader())

        # Audio in Frames schicken
        i = 0
        n = len(audio)
        while i < n:
            frame = audio[i:i+samples_per_frame]
            i += samples_per_frame
            await ws.send(pcm16le(frame))
            await asyncio.sleep(args.frame_ms / 1000.0)

        # Optional: final flush to force early transcription of remaining buffer
        if not args.no_flush_end:
            try:
                if args.flush_json:
                    await ws.send(json.dumps({"type": "flush"}))
                else:
                    await ws.send("flush")
            except Exception:
                pass

        # Give receiver a moment to deliver final partials
        if args.post_wait > 0:
            try:
                await asyncio.sleep(args.post_wait)
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(main())
