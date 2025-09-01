from typing import List, Dict
import datetime as dt


def _fmt_ts(sec: float) -> str:
    t = dt.timedelta(seconds=max(sec, 0.0))
    # SRT: HH:MM:SS,mmm
    total = int(t.total_seconds())
    ms = int((t.total_seconds() - total) * 1000)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"


def segments_to_srt(segments: List[Dict]) -> str:
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        start = _fmt_ts(seg["t0"])
        end = _fmt_ts(seg["t1"])
        text = (seg.get("text") or "").strip()
        spk = seg.get("speaker", "")
        head = f"{start} --> {end}"
        if spk:
            text = f"[{spk}] {text}"
        lines += [str(i), head, text, ""]
    return "\n".join(lines)


def segments_to_txt(segments: List[Dict], with_timestamps: bool = False, ts_mode: str = "none") -> str:
    lines: List[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        spk = seg.get("speaker", "")
        mode = ts_mode if ts_mode in {"none", "start", "span"} else ("span" if with_timestamps else "none")
        if mode == "start":
            start = _fmt_ts(seg.get("t0", 0.0))
            head = f"[{start}]"
        elif mode == "span":
            start = _fmt_ts(seg.get("t0", 0.0))
            end = _fmt_ts(seg.get("t1", 0.0))
            head = f"[{start} - {end}]"
        else:
            head = ""
        if spk:
            text = f"[{spk}] {text}"
        line = (head + (" " if head and text else "") + text).strip()
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")
