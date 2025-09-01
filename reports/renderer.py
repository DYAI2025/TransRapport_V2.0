from typing import List, Dict, Optional


def _h(ms: float) -> str:
    s = int(max(ms, 0.0))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02}:{mm:02}:{ss:02}"


def render_markdown(
    session_id: str,
    segments: List[Dict],
    topics: List[Dict],
    chapters: List[Dict],
    summary_global: Optional[str] = None,
    per_chapter_summaries: Optional[List[str]] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"# Sitzung {session_id}")
    lines.append("")
    if topics:
        lines.append("## Themen")
        for t in topics:
            lines.append(f"- {t.get('label')} — Gewicht {t.get('weight')}")
        lines.append("")
    if summary_global:
        lines.append("## Gesamtsummary")
        lines.append(summary_global.strip())
        lines.append("")
    if chapters:
        lines.append("## Kapitel")
        for i, ch in enumerate(chapters, 1):
            t0 = _h(ch.get("t0", 0.0))
            t1 = _h(ch.get("t1", 0.0))
            title = ch.get("title") or f"Kapitel {i}"
            lines.append(f"### {i}. {title} [{t0}–{t1}]")
            if per_chapter_summaries and i - 1 < len(per_chapter_summaries):
                s = per_chapter_summaries[i - 1]
                if s:
                    lines.append("")
                    lines.append(s.strip())
                    lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")

