import os
import re
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml


@dataclass
class BundleConfig:
    id: str
    version: str
    includes: Dict[str, List[str]]
    windows: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name -> (at_least, in_messages)
    sem_rules: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)


def parse_window_rule(rule: str) -> Tuple[int, int]:
    # format: "AT_LEAST 2 IN 6 messages"
    m = re.match(r"\s*AT_LEAST\s+(\d+)\s+IN\s+(\d+)\s+messages\s*", rule, re.I)
    if not m:
        return (2, 6)
    return (int(m.group(1)), int(m.group(2)))


class MarkerBundle:
    def __init__(self, bundle_path: str, markers_root: str):
        self.bundle_path = Path(bundle_path)
        self.markers_root = Path(markers_root)
        self.cfg = self._load()
        self.examples: Dict[str, Dict[str, List[str]]] = self._load_examples()

    def _load(self) -> BundleConfig:
        data = yaml.safe_load(self.bundle_path.read_text(encoding="utf-8"))
        includes = data.get("includes", {}) or {}
        windows_raw = (data.get("overrides", {}) or {}).get("windows", {}) or {}
        windows = {k: parse_window_rule(v) for k, v in windows_raw.items()}
        return BundleConfig(
            id=data.get("bundle_id") or data.get("name") or "bundle",
            version=str(data.get("version") or "1.0"),
            includes={k: list(v or []) for k, v in includes.items()},
            windows=windows,
            sem_rules=(data.get("overrides", {}) or {}).get("sem_rules", {}),
            scoring=(data.get("overrides", {}) or {}).get("scoring", {}),
            policy=data.get("policy", {}) or {},
        )

    def _load_examples(self) -> Dict[str, Dict[str, List[str]]]:
        """Load examples in a forgiving way (list or dict shapes)."""
        out: Dict[str, Dict[str, List[str]]] = {}
        for fam, names in self.cfg.includes.items():
            for name in names:
                    p = self.markers_root / fam / f"{name}.yaml"
                    if not p.exists():
                        continue
                    try:
                        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    except Exception:
                        doc = {}
                    ex = doc.get("examples")
                    pos: List[str] = []
                    neg: List[str] = []
                    if isinstance(ex, list):
                        pos = [str(x).strip() for x in ex if str(x).strip()]
                    elif isinstance(ex, dict):
                        p_list = ex.get("positive") or ex.get("pos") or []
                        n_list = ex.get("negative") or ex.get("neg") or []
                        if (not p_list) and isinstance(ex.get("examples"), list):
                            p_list = ex.get("examples") or []
                        pos = [str(x).strip() for x in p_list if str(x).strip()]
                        neg = [str(x).strip() for x in n_list if str(x).strip()]
                    if pos or neg:
                        out[name] = {"positive": pos, "negative": neg}
        return out

    def whitelist(self) -> set:
        s = set()
        for fam, names in self.cfg.includes.items():
            s.update(names)
        return s


def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


class ExampleMatcher:
    def __init__(self, examples: Dict[str, Dict[str, List[str]]]):
        # configuration (env with sensible defaults)
        self.fuzzy_enabled = (os.environ.get("SERAPI_MARKER_FUZZY", "1").lower() in {"1", "true", "yes"})
        self.j_pos = float(os.environ.get("SERAPI_FUZZY_POS_JACCARD", "0.5"))
        self.j_neg = float(os.environ.get("SERAPI_FUZZY_NEG_JACCARD", "0.6"))
        self.min_overlap = int(os.environ.get("SERAPI_FUZZY_MIN_OVERLAP", "2"))

        self.pos: Dict[str, List[str]] = {}
        self.neg: Dict[str, List[str]] = {}
        self.pos_tok: Dict[str, List[set]] = {}
        self.neg_tok: Dict[str, List[set]] = {}

        for name, ex in (examples or {}).items():
            p_list = [_norm_text(x) for x in ex.get("positive", [])]
            n_list = [_norm_text(x) for x in ex.get("negative", [])]
            self.pos[name] = p_list
            self.neg[name] = n_list
            self.pos_tok[name] = [self._tokens(x) for x in p_list]
            self.neg_tok[name] = [self._tokens(x) for x in n_list]

    @staticmethod
    def _tokens(s: str) -> set:
        # Unicode-friendly tokenization: use \w and the UNICODE flag, then filter short tokens
        if not s:
            return set()
        s = s.lower()
        # replace non-word characters (respecting Unicode) with spaces
        s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
        parts = re.split(r"\W+", s, flags=re.UNICODE)
        toks = {w for w in parts if len(w) >= 3}
        return toks

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = a & b
        if not inter:
            return 0.0
        union = a | b
        return float(len(inter)) / float(len(union))

    def match(self, name: str, text: str) -> bool:
        t = _norm_text(text)
        if not t:
            return False
        pos = self.pos.get(name, [])
        neg = self.neg.get(name, [])
        if not pos:
            return False

        # 1) strict negative substring block
        if any(p and p in t for p in neg):
            return False
        # 2) strict positive substring pass
        if any(p and p in t for p in pos):
            return True

        if not self.fuzzy_enabled:
            return False

        # 3) fuzzy token overlap
        toks_t = self._tokens(t)
        # block by negative fuzzy
        for n in self.neg_tok.get(name, []):
            if len(toks_t & n) >= self.min_overlap and self._jaccard(toks_t, n) >= self.j_neg:
                return False
        # allow by positive fuzzy
        for p in self.pos_tok.get(name, []):
            if len(toks_t & p) >= self.min_overlap and self._jaccard(toks_t, p) >= self.j_pos:
                return True
        return False


class AtoHeuristics:
    def __init__(self):
        pass

    def detect(self, current_idx: int, segments: List[Dict[str, Any]]) -> List[str]:
        # crude heuristics using timing/text
        seg = segments[current_idx]
        prev = segments[current_idx - 1] if current_idx > 0 else None
        markers: List[str] = []
        # gaps
        if prev is not None:
            gap = max(0.0, float(seg.get("t0", 0.0)) - float(prev.get("t1", 0.0)))
            if gap >= 2.5:
                markers.append("ATO_LONG_RESPONSE_GAP")
            elif gap >= 0.3:
                markers.append("ATO_PAUSE_MICRO")
        # rising intonation (question mark)
        txt = (seg.get("text") or "").strip()
        pros = seg.get("prosody") or {}
        if txt.endswith("?") or float(pros.get("f0_slope_end", 0.0)) >= 30.0:
            markers.append("ATO_RISING_INTONATION")
        # laughter/sigh
        ltxt = _norm_text(txt)
        if any(k in ltxt for k in ["haha", "lachen", "laugh", "[laughter]", "(laugh)"]):
            markers.append("ATO_LAUGHTER_EVENT")
        if any(k in ltxt for k in ["seufz", "*seufz*", "[sigh]", "(sigh)"]):
            markers.append("ATO_SIGH")
        # tempo (words per second)
        dur = max(0.2, float(seg.get("t1", 0.0)) - float(seg.get("t0", 0.0)))
        wps = len(ltxt.split()) / dur
        # variation vs previous turn
        prev_wps = None
        if prev is not None:
            ptxt = _norm_text(prev.get("text", ""))
            pdur = max(0.2, float(prev.get("t1", 0.0)) - float(prev.get("t0", 0.0)))
            prev_wps = (len(ptxt.split()) / pdur) if pdur > 0 else None
        zcr = float(pros.get("zcr", 0.0))
        if wps >= 3.5 or zcr >= 0.12:
            markers.append("ATO_TEMPO_FAST")
        elif (wps <= 1.2 and len(ltxt.split()) >= 2) or (float(pros.get("rms", 0.0)) < 0.01 and dur >= 1.0):
            markers.append("ATO_TEMPO_SLOW")
        if prev_wps is not None and abs(wps - prev_wps) >= 1.5:
            markers.append("ATO_SPEECH_RATE_VARIATION")
        # pitch spread (optional)
        f0_rng = float(pros.get("f0_range", 0.0))
        if f0_rng >= 70.0:
            markers.append("ATO_PITCH_SPREAD_WIDE")
        elif 0.0 < f0_rng <= 30.0:
            markers.append("ATO_PITCH_SPREAD_NARROW")
        # hedging cues
        if any(k in ltxt for k in ["vielleicht", "maybe", "glaub", "ich denke", "i think", "unsicher"]):
            markers.append("ATO_HEDGING_CUE")
        return markers


class MarkerEngine:
    def __init__(self, bundle: MarkerBundle):
        self.bundle = bundle
        self.whitelist = bundle.whitelist()
        self.matcher = ExampleMatcher(bundle.examples)
        self.ato = AtoHeuristics()
        self.msg_idx = 0
        # sliding indices for CLU confirms: name -> deque of message indices
        self.clu_hits: Dict[str, deque] = {name: deque() for name in self.whitelist if name.startswith("CLU_")}
        # recent ATO history for SEM gating (msg_idx, name)
        self.ato_hist: deque = deque(maxlen=256)
        # SEM rules
        self.sem_min_distinct = int(self.bundle.cfg.sem_rules.get("min_distinct_ato", 2) or 2)
        # If seconds are provided but segment times are local, approximate by messages
        self.sem_max_gap_seconds = float(self.bundle.cfg.sem_rules.get("max_gap_seconds", 20) or 20.0)
        # Intuition rate limit (seconds, approximated to messages)
        self.intuition_min_interval_seconds = float((self.bundle.cfg.scoring or {}).get("intuition_min_interval_seconds", 120.0))
        self._last_intuition_idx: Optional[int] = None

    def process(self, new_segments: List[Dict[str, Any]], all_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for seg in new_segments:
            idx = self.msg_idx
            # ATO heuristics
            ato_marks = [m for m in self.ato.detect(len(all_segments), all_segments + [seg]) if m in self.whitelist]
            for m in ato_marks:
                events.append({"family": "ATO", "name": m, "msg_idx": idx, "t0": seg.get("t0", 0.0), "t1": seg.get("t1", 0.0), "text": seg.get("text", "")})
                # record into recent ATO history
                try:
                    self.ato_hist.append((idx, m))
                except Exception:
                    pass

            # SEM/CLU by examples
            for name in list(self.whitelist):
                if name.startswith("SEM_") or name.startswith("CLU_"):
                    if self.matcher.match(name, seg.get("text", "")):
                        # apply SEM gating: require distinct ATOs in recent window
                        if name.startswith("SEM_") and not self._sem_gate(idx, all_segments + [seg]):
                            continue
                        # apply CLU_INTUITION min-interval gate
                        if name.startswith("CLU_INTUITION") and not self._intuition_gate(idx, all_segments + [seg]):
                            continue
                        fam = "SEM" if name.startswith("SEM_") else "CLU"
                        events.append({"family": fam, "name": name, "msg_idx": idx, "t0": seg.get("t0", 0.0), "t1": seg.get("t1", 0.0), "text": seg.get("text", "")})
                        # window confirmations for CLU
                        if fam == "CLU":
                            self._update_clu_window(name, idx, events, seg)
                        if name.startswith("CLU_INTUITION"):
                            self._last_intuition_idx = idx

            self.msg_idx += 1
        return events

    def _sem_gate(self, idx: int, segments: List[Dict[str, Any]]) -> bool:
        """Require min distinct ATOs within a recent message window approximated from max_gap_seconds.

        Since segment timestamps are per-chunk, estimate window size in messages using
        the mean duration of recent segments.
        """
        try:
            # compute average segment duration over last up to 12 segments
            k = min(12, len(segments))
            if k <= 0:
                win_msgs = 6
            else:
                durs = []
                for s in segments[-k:]:
                    try:
                        d = max(0.0, float(s.get("t1", 0.0)) - float(s.get("t0", 0.0)))
                        if d > 0:
                            durs.append(d)
                    except Exception:
                        pass
                avg_dur = (sum(durs) / len(durs)) if durs else 3.0
                approx = int(max(3, min(12, round(self.sem_max_gap_seconds / max(0.2, avg_dur)))))
                win_msgs = approx
            # collect distinct ATOs within idx - win_msgs .. idx
            recent = {name for (i, name) in self.ato_hist if (idx - i) <= win_msgs}
            return len(recent) >= self.sem_min_distinct
        except Exception:
            # be permissive on errors
            return True

    def _update_clu_window(self, name: str, idx: int, events: List[Dict[str, Any]], seg: Dict[str, Any]):
        rule = self.bundle.cfg.windows.get(name)
        if not rule:
            return
        at_least, in_msgs = rule
        dq = self.clu_hits.setdefault(name, deque())
        dq.append(idx)
        # trim outside window
        while dq and (idx - dq[0] + 1) > in_msgs:
            dq.popleft()
        if len(dq) >= at_least:
            events.append({
                "family": "CLU",
                "name": name,
                "confirmed": True,
                "msg_idx": idx,
                "t0": seg.get("t0", 0.0),
                "t1": seg.get("t1", 0.0),
                "text": seg.get("text", ""),
            })

    def _intuition_gate(self, idx: int, segments: List[Dict[str, Any]]) -> bool:
        """Allow at most one CLU_INTUITION event per configured interval (approx by messages)."""
        try:
            if self._last_intuition_idx is None:
                return True
            # average duration of recent segments (up to last 12)
            k = min(12, len(segments))
            durs: List[float] = []
            for s in segments[-k:]:
                try:
                    d = max(0.0, float(s.get("t1", 0.0)) - float(s.get("t0", 0.0)))
                    if d > 0:
                        durs.append(d)
                except Exception:
                    pass
            avg = (sum(durs) / len(durs)) if durs else 3.0
            min_msgs = int(max(1, round(self.intuition_min_interval_seconds / max(0.2, avg))))
            return (idx - self._last_intuition_idx) >= min_msgs
        except Exception:
            return True
