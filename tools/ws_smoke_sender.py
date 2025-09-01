import argparse
import asyncio
import json
import websockets


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8710/ws/stream")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--frame_ms", type=int, default=30)
    args = ap.parse_args()

    bytes_per_sec = 2 * args.sr  # PCM16 mono
    frame_bytes = int(bytes_per_sec * (args.frame_ms / 1000.0))
    total_frames = int(args.seconds * 1000.0 / args.frame_ms)

    uri = f"{args.url}?session_id={args.sid}&sr={args.sr}"

    async with websockets.connect(uri, max_size=2**23) as ws:
        async def reader():
            while True:
                try:
                    msg = await ws.recv()
                    try:
                        evt = json.loads(msg)
                        print("[srv]", evt)
                    except Exception:
                        print("[srv]", type(msg), len(msg))
                except Exception:
                    break

        asyncio.create_task(reader())

        silence = b"\x00" * frame_bytes
        for _ in range(total_frames):
            await ws.send(silence)
            await asyncio.sleep(args.frame_ms / 1000.0)

        # Flush
        await ws.send("flush")
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(main())

