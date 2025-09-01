import math
from typing import Dict, Tuple

import numpy as np


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(float(np.mean(x.astype(np.float32) ** 2)) + 1e-9))


def _zcr(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(x))) > 0))


def _spectral_flatness(x: np.ndarray, sr: int) -> float:
    if x.size == 0:
        return 0.0
    n = 1 << (int(np.ceil(np.log2(len(x)))) or 1)
    X = np.abs(np.fft.rfft(x, n=n)) + 1e-12
    geo = np.exp(np.mean(np.log(X)))
    arith = np.mean(X)
    return float(geo / arith)


def _f0_py(x: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """Estimate F0 contour using librosa.pyin if available, else return empty contour."""
    try:
        import librosa  # type: ignore
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=256)
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced = f0 > 0
        f0v = f0[voiced]
        voicing = float(np.mean(voiced)) if f0.size else 0.0
        return (f0v.astype(np.float32) if f0v.size else np.zeros(0, dtype=np.float32), voicing)
    except Exception:
        return (np.zeros(0, dtype=np.float32), 0.0)


def extract_prosody(pcm_f32: np.ndarray, sr: int, t0: float, t1: float) -> Dict:
    """
    Compute lightweight prosody features for a segment.
    Returns dict with f0 stats (if available), energy/zcr/flatness, and simple slopes.
    """
    i0, i1 = max(int(t0 * sr), 0), max(int(t1 * sr), 0)
    seg = pcm_f32[i0:i1] if i1 > i0 else np.zeros(0, dtype=np.float32)
    if seg.size == 0:
        return {
            "rms": 0.0,
            "zcr": 0.0,
            "flatness": 0.0,
            "f0_med": 0.0,
            "f0_p10": 0.0,
            "f0_p90": 0.0,
            "f0_range": 0.0,
            "f0_slope_end": 0.0,
            "voicing": 0.0,
        }

    rms = _rms(seg)
    zcr = _zcr(seg)
    flat = _spectral_flatness(seg, sr)

    f0v, voicing = _f0_py(seg, sr)
    if f0v.size:
        f0_med = float(np.median(f0v))
        f0_p10 = float(np.percentile(f0v, 10))
        f0_p90 = float(np.percentile(f0v, 90))
        f0_rng = float(f0_p90 - f0_p10)
        # slope over last 300 ms or last 10 frames
        n = len(f0v)
        k = max(3, min(n, 10))
        y = f0v[-k:]
        x = np.arange(len(y), dtype=np.float32)
        x = (x - x[0]) / max(1.0, len(y) - 1)
        # simple linear fit slope (Hz per normalized unit) -> approx Hz/frame; normalize to Hz/s using hop ~ (len(seg)/n)
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        # approximate frames per second
        fps = n / max(1e-3, (t1 - t0))
        f0_slope = float(m * fps)
    else:
        f0_med = f0_p10 = f0_p90 = f0_rng = f0_slope = 0.0

    return {
        "rms": float(rms),
        "zcr": float(zcr),
        "flatness": float(flat),
        "f0_med": float(f0_med),
        "f0_p10": float(f0_p10),
        "f0_p90": float(f0_p90),
        "f0_range": float(f0_rng),
        "f0_slope_end": float(f0_slope),
        "voicing": float(voicing),
    }

