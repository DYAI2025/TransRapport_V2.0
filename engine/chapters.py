from typing import Dict, List, Tuple
import re


def _tokens(text: str) -> List[str]:
    s = (text or "").lower()
    return [t for t in re.findall(r"\w+", s, flags=re.UNICODE) if len(t) >= 3]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return float(len(inter)) / float(len(union))


def detect_chapters(
    segments: List[Dict],
    min_segments: int = 6,
    min_seconds: float = 90.0,
    drift_threshold: float = 0.55,
) -> List[Dict]:
    """
    Simple chapter boundaries by lexical drift between adjacent windows.
    Creates a new chapter when similarity drops below (1 - drift_threshold), subject to minimum length.
    """
    if not segments:
        return []
    # precompute tokens per segment
    toks = [set(_tokens(str(s.get("text", "")))) for s in segments]
    n = len(segments)

    def span_seconds(i0: int, i1: int) -> float:
        if i1 <= i0:
            return 0.0
        t0 = float(segments[i0].get("t0", 0.0))
        t1 = float(segments[i1 - 1].get("t1", 0.0))
        return max(0.0, t1 - t0)

    cuts: List[int] = [0]
    i = 0
    while i + min_segments < n:
        # windows: current chapter tail vs following window
        left = set().union(*toks[cuts[-1]: i + 1])
        j = min(n, i + min_segments)
        right = set().union(*toks[i + 1: j])
        sim = _jaccard(left, right)
        drift = 1.0 - sim
        # check min seconds on current chapter
        long_enough = span_seconds(cuts[-1], i + 1) >= min_seconds
        if drift >= drift_threshold and long_enough:
            cuts.append(i + 1)
            i = i + 1
        else:
            i += 1

    # finalize chapters
    if cuts[-1] != n:
        cuts.append(n)
    chapters: List[Dict] = []
    for k in range(len(cuts) - 1):
        i0, i1 = cuts[k], cuts[k + 1]
        if i1 - i0 <= 0:
            continue
        t0 = float(segments[i0].get("t0", 0.0))
        t1 = float(segments[i1 - 1].get("t1", 0.0))
        title = (segments[i0].get("text", "") or "").strip()
        title = (title[:72] + "…") if len(title) > 72 else title
        chapters.append({
            "start_idx": i0,
            "end_idx": i1,
            "t0": t0,
            "t1": t1,
            "title": title,
        })
    return chapters

