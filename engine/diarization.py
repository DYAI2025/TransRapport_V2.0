import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import re


def _labels_for(n: int) -> List[str]:
    base = [chr(ord('A') + i) for i in range(26)]
    out: List[str] = []
    for i in range(n):
        out.append(base[i] if i < len(base) else f"S{i+1}")
    return out


class OnlineCluster:
    """Simple online clustering helper (euclidean or cosine) with optional momentum update."""

    def __init__(self, tau: float = 0.18, metric: str = "l2", momentum: float = 0.0):
        self.tau = float(tau)
        self.metric = metric
        self.momentum = float(momentum)  # 0..1; if >0, centroid = (1-a)*c + a*x
        self.centroids: List[np.ndarray] = []
        self.counts: List[int] = []

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cos":
            na = np.linalg.norm(a) + 1e-9
            nb = np.linalg.norm(b) + 1e-9
            return 1.0 - float(np.dot(a, b) / (na * nb))
        return float(np.linalg.norm(a - b))

    def assign(self, x: np.ndarray, last_idx: Optional[int] = None, stickiness_delta: float = 0.0) -> int:
        if len(self.centroids) == 0:
            self.centroids.append(x.copy())
            self.counts.append(1)
            return 0
        dists = [self._dist(x, c) for c in self.centroids]
        k = int(np.argmin(dists))
        # apply stickiness to last_idx if within tau + delta
        if last_idx is not None and 0 <= last_idx < len(self.centroids):
            d_last = dists[last_idx]
            if d_last <= (self.tau + float(stickiness_delta)):
                k = last_idx
        if dists[k] <= self.tau:
            if self.momentum > 0.0:
                self.centroids[k] = (1.0 - self.momentum) * self.centroids[k] + self.momentum * x
                self.counts[k] += 1
            else:
                n = self.counts[k]
                self.centroids[k] = (self.centroids[k] * n + x) / (n + 1)
                self.counts[k] = n + 1
            return k
        self.centroids.append(x.copy())
        self.counts.append(1)
        return len(self.centroids) - 1


class HeuristicDiarizer:
    """Lightweight diarizer using [log_energy, zcr] features + L2 clustering."""

    def __init__(self, tau: float = 0.18, stickiness_delta: float = 0.03, min_turn_sec: float = 0.6, momentum: float = 0.0,
                 vad_energy_db: float = -35.0, min_voiced_ms: int = 250):
        self.cluster = OnlineCluster(tau=tau, metric="l2", momentum=momentum)
        self.stickiness_delta = float(stickiness_delta)
        self.min_turn_sec = float(min_turn_sec)
        self.vad_energy_db = float(vad_energy_db)
        self.min_voiced_ms = int(min_voiced_ms)
        self._last_idx: Optional[int] = None
        self._last_end: float = 0.0

    @staticmethod
    def _features(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(2, dtype=np.float32)
        energy = float(np.mean(x.astype(np.float32) ** 2) + 1e-9)
        loge = math.log(energy)
        zc = float(np.mean(np.abs(np.diff(np.signbit(x))) > 0))
        return np.array([loge, zc], dtype=np.float32)

    @staticmethod
    def _trim_active(pcm: np.ndarray, sr: int, thr_db: float, min_voiced_ms: int) -> Tuple[np.ndarray, bool]:
        if pcm.size == 0:
            return pcm, False
        # frame 25ms, hop 10ms
        frame = max(1, int(0.025 * sr))
        hop = max(1, int(0.010 * sr))
        rms = []
        for i in range(0, len(pcm) - frame + 1, hop):
            w = pcm[i:i+frame]
            rms.append(float(np.sqrt(np.mean(w*w) + 1e-9)))
        rms = np.array(rms, dtype=np.float32)
        if rms.size == 0:
            return pcm, False
        db = 20.0 * np.log10(np.maximum(rms, 1e-9))
        active = db > thr_db
        if not np.any(active):
            return np.zeros(0, dtype=np.float32), False
        idx = np.where(active)[0]
        i0 = int(idx[0] * hop)
        i1 = int(min(len(pcm), (idx[-1] * hop + frame)))
        voiced_len_ms = (i1 - i0) * 1000.0 / sr
        ok = voiced_len_ms >= min_voiced_ms
        return pcm[i0:i1].copy(), ok

    def label_segments(self, pcm_f32: np.ndarray, sr: int, segments: List[Dict]) -> List[str]:
        labs: List[str] = []
        for seg in segments:
            t0, t1 = float(seg.get("t0", 0.0)), float(seg.get("t1", 0.0))
            i0, i1 = max(int(t0 * sr), 0), max(int(t1 * sr), 0)
            sl = pcm_f32[i0:i1] if i1 > i0 else np.zeros(0, dtype=np.float32)
            # VAD trim
            sl_t, ok = self._trim_active(sl, sr, self.vad_energy_db, self.min_voiced_ms)
            dur = (i1 - i0) / float(sr)
            if not ok:
                # too little voiced content
                labs.append("UNK")
                continue
            # discourage flips on very short turns
            if dur < self.min_turn_sec and self._last_idx is not None:
                idx = self._last_idx
            else:
                f = self._features(sl_t)
                idx = self.cluster.assign(f, last_idx=self._last_idx, stickiness_delta=self.stickiness_delta)
            labs.append(_labels_for(idx + 1)[idx])
            self._last_idx = idx
            self._last_end = t1
        return labs


class EcapaDiarizer:
    """
    ECAPA-TDNN based diarizer using SpeechBrain EncoderClassifier embeddings and
    online cosine clustering. Operates offline once a local model is provided.

    Provide a local model directory via `model_dir` containing SpeechBrain's
    `hyperparams.yaml` and checkpoint, or pass a `source` name if you allow
    a one-time download externally and cache locally.
    """

    def __init__(self, model_dir: Optional[str] = None, source: Optional[str] = None, tau: float = 0.25,
                 stickiness_delta: float = 0.03, min_turn_sec: float = 0.6, momentum: float = 0.15,
                 vad_energy_db: float = -35.0, min_voiced_ms: int = 250):
        self.cluster = OnlineCluster(tau=tau, metric="cos", momentum=momentum)
        self.stickiness_delta = float(stickiness_delta)
        self.min_turn_sec = float(min_turn_sec)
        self.vad_energy_db = float(vad_energy_db)
        self.min_voiced_ms = int(min_voiced_ms)
        self._last_idx: Optional[int] = None
        self._last_end: float = 0.0
        self.ok = False
        self.err: Optional[str] = None
        try:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
            if model_dir:
                self.clf = EncoderClassifier.from_hparams(source=model_dir, savedir=model_dir, run_opts={"device": "cpu"})
            elif source:
                self.clf = EncoderClassifier.from_hparams(source=source, run_opts={"device": "cpu"})
            else:
                raise RuntimeError("No ECAPA model source provided")
            self.ok = True
        except Exception as e:  # fallback handled by factory
            self.err = str(e)
            self.ok = False
            self.clf = None  # type: ignore

    @staticmethod
    def _trim_active(pcm: np.ndarray, sr: int, thr_db: float, min_voiced_ms: int) -> Tuple[np.ndarray, bool]:
        if pcm.size == 0:
            return pcm, False
        frame = max(1, int(0.025 * sr))
        hop = max(1, int(0.010 * sr))
        rms = []
        for i in range(0, len(pcm) - frame + 1, hop):
            w = pcm[i:i+frame]
            rms.append(float(np.sqrt(np.mean(w*w) + 1e-9)))
        rms = np.array(rms, dtype=np.float32)
        if rms.size == 0:
            return pcm, False
        db = 20.0 * np.log10(np.maximum(rms, 1e-9))
        active = db > thr_db
        if not np.any(active):
            return np.zeros(0, dtype=np.float32), False
        idx = np.where(active)[0]
        i0 = int(idx[0] * hop)
        i1 = int(min(len(pcm), (idx[-1] * hop + frame)))
        voiced_len_ms = (i1 - i0) * 1000.0 / sr
        ok = voiced_len_ms >= min_voiced_ms
        return pcm[i0:i1].copy(), ok

    def label_segments(self, pcm_f32: np.ndarray, sr: int, segments: List[Dict]) -> List[str]:
        if not self.ok:
            return ["UNK" for _ in segments]
        import torch  # type: ignore
        labs: List[str] = []
        wav = torch.from_numpy(pcm_f32).float().unsqueeze(0)  # 1 x T
        for seg in segments:
            t0, t1 = float(seg.get("t0", 0.0)), float(seg.get("t1", 0.0))
            i0, i1 = max(int(t0 * sr), 0), max(int(t1 * sr), 0)
            if i1 <= i0:
                labs.append("UNK")
                continue
            np_chunk = pcm_f32[i0:i1]
            # VAD trim
            np_chunk_t, ok = self._trim_active(np_chunk, sr, self.vad_energy_db, self.min_voiced_ms)
            dur = (i1 - i0) / float(sr)
            if not ok:
                labs.append("UNK")
                continue
            # discourage flips on very short turns
            if dur < self.min_turn_sec and self._last_idx is not None:
                idx = self._last_idx
                labs.append(_labels_for(idx + 1)[idx])
                continue
            chunk = torch.from_numpy(np_chunk_t).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                emb = self.clf.encode_batch(chunk)  # 1 x 1 x D
            vec = emb.squeeze().cpu().numpy().astype(np.float32)
            idx = self.cluster.assign(vec, last_idx=self._last_idx, stickiness_delta=self.stickiness_delta)
            labs.append(_labels_for(idx + 1)[idx])
            self._last_idx = idx
            self._last_end = t1
        return labs


class DiarizerFactory:
    @staticmethod
    def from_env(
        ecapa_dir: Optional[str],
        ecapa_source: Optional[str],
        tau: float = 0.25,
        stickiness_delta: float = 0.03,
        min_turn_sec: float = 0.6,
        momentum: float = 0.15,
        vad_energy_db: float = -35.0,
        min_voiced_ms: int = 250,
    ) -> "object":
        # Prefer ECAPA if a local dir is available or a source is provided and the library is installed
        if ecapa_dir or ecapa_source:
            d = EcapaDiarizer(
                model_dir=ecapa_dir,
                source=ecapa_source,
                tau=tau,
                stickiness_delta=stickiness_delta,
                min_turn_sec=min_turn_sec,
                momentum=momentum,
                vad_energy_db=vad_energy_db,
                min_voiced_ms=min_voiced_ms,
            )
            if d.ok:
                return d
        # Fallback
        return HeuristicDiarizer(
            tau=0.18,
            stickiness_delta=stickiness_delta,
            min_turn_sec=min_turn_sec,
            momentum=momentum,
            vad_energy_db=vad_energy_db,
            min_voiced_ms=min_voiced_ms,
        )
