"""tempo.py -- rough BPM estimation via onset-envelope autocorrelation.

Timing stays correct regardless of BPM accuracy (see the .sm export notes: rows
round-trip back to the same seconds at whatever BPM we export). BPM mainly
affects how *musical* the beat-grid quantization looks. So this is best-effort:
autocorrelate the onset envelope, find the strongest lag in a sane BPM band,
fall back to 120 if nothing stands out.
"""
from __future__ import annotations

import numpy as np

from .onsets import onset_envelope

MIN_BPM = 60.0
MAX_BPM = 180.0
FALLBACK_BPM = 120.0


def estimate_bpm(mono: np.ndarray, sr: int, hop: int = 512) -> float:
    times, env = onset_envelope(mono, sr, hop=hop)
    if len(env) < 4:
        return FALLBACK_BPM

    # Frame period in seconds (median spacing of STFT frame times).
    dt = float(np.median(np.diff(times))) if len(times) > 1 else hop / sr
    if dt <= 0:
        return FALLBACK_BPM

    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]  # non-negative lags
    if ac[0] <= 0:
        return FALLBACK_BPM

    # Candidate lags (in frames) for the BPM band.
    lag_min = int(round((60.0 / MAX_BPM) / dt))
    lag_max = int(round((60.0 / MIN_BPM) / dt))
    lag_min = max(1, lag_min)
    lag_max = min(len(ac) - 1, lag_max)
    if lag_max <= lag_min:
        return FALLBACK_BPM

    band = ac[lag_min:lag_max + 1]
    best_lag = lag_min + int(np.argmax(band))
    bpm = 60.0 / (best_lag * dt)

    # Fold into a comfortable range (avoid half/double-time artifacts).
    while bpm < MIN_BPM:
        bpm *= 2.0
    while bpm > MAX_BPM:
        bpm /= 2.0
    return float(round(bpm, 2))
