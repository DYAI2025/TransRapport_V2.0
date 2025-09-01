#!/usr/bin/env python3
"""
One-shot detector/exporter for an existing session on the local API.

Pulls transcript, stats, topics, chapters, summaries and generates
report files (md/txt/docx/pdf). Writes a consolidated JSON to
data/sessions/<sid>/detect_report.json.

Usage:
  python tools/detect.py --sid <session_id> [--host 127.0.0.1 --port 8710]
"""
import argparse
import json
import os
from pathlib import Path
from urllib import request, parse


def _url(host: str, port: int, path: str) -> str:
    base = f"http://{host}:{port}"
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _get_json(url: str):
    req = request.Request(url, headers={"accept": "application/json"})
    with request.urlopen(req, timeout=20) as r:
        ct = r.headers.get("content-type", "")
        data = r.read()
        if "application/json" in ct:
            return json.loads(data.decode("utf-8"))
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {"raw": data.decode("utf-8", errors="replace")}


def _get_text(url: str) -> str:
    req = request.Request(url)
    with request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8710)
    args = ap.parse_args()

    sid = args.sid
    host, port = args.host, args.port

    out_dir = Path("data/sessions") / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {"session_id": sid, "ok": True, "errors": []}

    try:
        health = _get_json(_url(host, port, "/healthz"))
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"healthz failed: {e}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    result["health"] = health

    # transcript, stats
    try:
        tr = _get_json(_url(host, port, f"/session/{sid}/transcript"))
        result["transcript"] = {"segments": len(tr.get("segments", []))}
    except Exception as e:
        result["errors"].append(f"transcript failed: {e}")

    try:
        st = _get_json(_url(host, port, f"/session/{sid}/stats"))
        result["stats"] = st
    except Exception as e:
        result["errors"].append(f"stats failed: {e}")

    # topics/chapters/summary
    try:
        result["topics"] = _get_json(_url(host, port, f"/session/{sid}/topics"))
    except Exception as e:
        result["errors"].append(f"topics failed: {e}")
    try:
        result["chapters"] = _get_json(_url(host, port, f"/session/{sid}/chapters"))
    except Exception as e:
        result["errors"].append(f"chapters failed: {e}")
    try:
        result["summary"] = _get_json(_url(host, port, f"/session/{sid}/summary"))
    except Exception as e:
        result["errors"].append(f"summary failed: {e}")

    # reports (generate files server-side)
    try:
        _ = _get_text(_url(host, port, f"/session/{sid}/report.md"))
    except Exception as e:
        result["errors"].append(f"report.md failed: {e}")
    try:
        _ = _get_text(_url(host, port, f"/session/{sid}/report.txt"))
    except Exception as e:
        result["errors"].append(f"report.txt failed: {e}")
    try:
        result["docx"] = _get_json(_url(host, port, f"/session/{sid}/report.docx"))
    except Exception as e:
        result["errors"].append(f"report.docx failed: {e}")
    try:
        result["pdf"] = _get_json(_url(host, port, f"/session/{sid}/report.pdf"))
    except Exception as e:
        result["errors"].append(f"report.pdf failed: {e}")

    # marker debug + telemetry (if exists)
    try:
        dbg = _get_json(_url(host, port, f"/session/{sid}/debug/markers"))
        result["markers_debug"] = {"loaded": dbg.get("loaded"), "bundle": dbg.get("bundle_id"), "count": len((dbg.get("markers") or {}).keys())}
    except Exception as e:
        result["errors"].append(f"debug markers failed: {e}")

    tele_path = out_dir / "telemetry_intuitions.json"
    if tele_path.exists():
        try:
            result["intuition_telemetry"] = json.loads(tele_path.read_text(encoding="utf-8"))
        except Exception as e:
            result["errors"].append(f"read telemetry failed: {e}")

    (out_dir / "detect_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": result.get("ok", True), "errors": len(result.get("errors", [])), "out": str(out_dir / 'detect_report.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()

