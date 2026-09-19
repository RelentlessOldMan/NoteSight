"""Shared test fixtures.

We don't ship copyrighted audio, so tests synthesize their own signal: a click
track (short decaying tone bursts on a fixed grid). It has unambiguous onsets on
a steady tempo, which is all the pipeline needs to exercise end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

SR = 44100
BPM = 120.0
DUR = 8.0            # seconds


def _click_track(sr=SR, bpm=BPM, dur=DUR):
    """Mono float32 click track: a 1 kHz, 40 ms decaying burst on every beat."""
    n = int(dur * sr)
    x = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    burst_len = int(0.04 * sr)
    t = np.arange(burst_len) / sr
    burst = (np.sin(2 * np.pi * 1000.0 * t) * np.exp(-t * 60.0)).astype(np.float32)
    beat = 0
    while True:
        start = int(beat * period * sr)
        if start >= n:
            break
        end = min(start + burst_len, n)
        x[start:end] += burst[: end - start]
        beat += 1
    return x


@pytest.fixture(scope="session")
def click_audio():
    """(mono, sr) for a steady 120 BPM click track."""
    return _click_track(), SR


@pytest.fixture(scope="session")
def click_wav(tmp_path_factory):
    """Path to a written .wav of the click track (exercises audio_io + copies)."""
    import soundfile as sf
    p = tmp_path_factory.mktemp("audio") / "click.wav"
    sf.write(str(p), _click_track(), SR)
    return str(p)
