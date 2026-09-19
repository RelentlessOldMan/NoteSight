"""onsets.py -- format-neutral onset detection.

Lifts the STFT / spectral-flux / adaptive-peak-pick math out of the old
charter.py. This layer knows NOTHING about difficulty, lanes, or game formats.
It just turns audio into a list of candidate OnsetEvents.

  audio (mono float32) + samplerate  ->  [OnsetEvent(time, strength, brightness)]

  time:       seconds from start of audio
  strength:   normalized onset salience in 0..1 (spectral-flux peak height)
  brightness: spectral centroid (Hz) at that frame, used later for lane choice

Deliberately plain array math so it ports to C++ cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


# Spectral flux (a frame-to-frame difference) reports an onset ~2 hop-frames
# after the true transient: one frame for the diff, one for the peak to form.
# Measured against hand-authored charts as +21ms at hop=512/48kHz (= 2*hop/sr).
# Subtract it so notes land on the transient instead of consistently late.
LATENCY_FRAMES = 2.0


@dataclass
class OnsetEvent:
    time: float        # seconds
    strength: float    # 0..1 salience
    brightness: float  # spectral centroid (Hz)


def _spectral_flux(mono: np.ndarray, sr: int, hop: int = 512, win: int = 1024):
    """Return (times, flux, centroid) frame-wise.

    flux: half-wave-rectified sum of positive magnitude increases (onset
    envelope). centroid: spectral centroid per frame (Hz), for lane assignment.
    """
    f, t, Z = stft(mono, fs=sr, nperseg=win, noverlap=win - hop,
                   window="hann", boundary=None, padded=False)
    mag = np.abs(Z)  # (freq_bins, frames)

    # Spectral flux: positive differences between successive frames, summed.
    diff = np.diff(mag, axis=1)
    diff[diff < 0] = 0.0
    flux = diff.sum(axis=0)
    flux = np.concatenate([[0.0], flux])  # align length with frame count

    # Spectral centroid per frame (weighted mean frequency).
    freqs = f[:, None]
    denom = mag.sum(axis=0) + 1e-9
    centroid = (mag * freqs).sum(axis=0) / denom

    return t, flux, centroid


def _pick_peaks(times, flux, sr, hop):
    """Adaptive peak picking on the onset envelope.

    A frame is an onset if it is a local max and exceeds a moving-average
    threshold plus a small delta. Returns list of (time, strength, frame_idx),
    where strength is on a normalized 0..1 scale and frame_idx lets callers
    look up the per-frame centroid.
    """
    if flux.size == 0:
        return []
    # Normalize.
    flux = flux / (flux.max() + 1e-9)

    # Moving-average threshold over ~0.15s window.
    w = max(3, int(0.15 * sr / hop))
    kernel = np.ones(w) / w
    local_mean = np.convolve(flux, kernel, mode="same")
    thresh = local_mean + 0.06  # delta above local mean

    onsets = []
    for i in range(1, len(flux) - 1):
        if flux[i] < thresh[i]:
            continue
        if flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            onsets.append((float(times[i]), float(flux[i]), i))
    return onsets


def detect_onsets(mono: np.ndarray, sr: int, hop: int = 512) -> list[OnsetEvent]:
    """Detect candidate onsets. NO thinning, NO lanes -- just the raw events.

    How many to keep (the difficulty) is decided downstream in selection.select().
    """
    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)

    times, flux, centroid = _spectral_flux(mono, sr, hop=hop)
    peaks = _pick_peaks(times, flux, sr, hop)

    latency = LATENCY_FRAMES * hop / sr  # seconds
    events: list[OnsetEvent] = []
    n_frames = len(centroid)
    for t, strength, frame_idx in peaks:
        bright = float(centroid[min(frame_idx, n_frames - 1)])
        events.append(OnsetEvent(time=max(0.0, t - latency),
                                 strength=strength, brightness=bright))
    return events


def onset_envelope(mono: np.ndarray, sr: int, hop: int = 512):
    """Convenience: (times, normalized_flux) for tempo estimation / plotting."""
    times, flux, _ = _spectral_flux(mono, sr, hop=hop)
    return times, flux / (flux.max() + 1e-9)


def energy_envelope(mono: np.ndarray, sr: int, hop: int = 512, win: int = 1024,
                    smooth_sec: float = 0.08):
    """Frame-wise RMS energy (times, normalized 0..1).

    This is the "is the sound still going?" signal the hold detector needs: after
    a note, sustained energy (a held vocal note, a pad) means a hold; energy that
    decays to silence means a rest. Smoothed a little so a momentary dip inside a
    sustained note doesn't look like the note ended.
    """
    mono = np.asarray(mono, dtype=np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    if mono.size == 0:
        return np.zeros(0), np.zeros(0)
    n = 1 + (len(mono) - 1) // hop
    starts = np.arange(n) * hop
    ends = np.minimum(starts + win, len(mono))
    # RMS per frame via a cumulative sum of squares (fast, no Python loop).
    csum = np.concatenate([[0.0], np.cumsum(mono * mono)])
    rms = np.sqrt((csum[ends] - csum[starts]) / np.maximum(ends - starts, 1))
    w = max(1, int(smooth_sec * sr / hop))
    if w > 1:
        rms = np.convolve(rms, np.ones(w) / w, mode="same")
    times = starts / sr
    return times, (rms / (rms.max() + 1e-9)).astype(np.float32)
