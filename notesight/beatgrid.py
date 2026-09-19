"""beatgrid.py -- estimate a musical beat grid: tempo AND phase.

tempo.py answers "how fast" (BPM). This answers "how fast AND where the beats
land" -- the phase/offset the .sm export needs so notes sit ON the grid instead
of floating 50-ish ms early at raw onset times.

  estimate_grid(mono, sr) -> BeatGrid(bpm, beat0)
    bpm    : refined tempo (parabolic-interpolated autocorr peak, so it is not
             quantized to the coarse frame-lag grid the way tempo.estimate_bpm is)
    beat0  : audio time (seconds, in [0, beat_period)) of the first beat -- the
             grid is beat0 + k*60/bpm. StepMania #OFFSET = -beat0.

Plain array math (autocorrelation + a comb-filter phase search), so it ports to
C++ alongside the rest of the core.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .onsets import onset_envelope

# Rhythm-game tempos run high -- DDR packs reach ~280 BPM. Keep the window wide
# enough that fast songs are detected at their true tempo instead of folded to
# half; a perceptual prior (below) picks the right octave within it.
MIN_BPM = 90.0
MAX_BPM = 285.0
FALLBACK_BPM = 150.0

# Log-Gaussian tempo prior for octave disambiguation: center near the median
# rhythm-game tempo, width in octaves. Weights the autocorrelation so a song's
# true octave beats its half/double relatives.
TEMPO_CENTER = 155.0
TEMPO_OCTAVES = 0.9


@dataclass
class BeatGrid:
    bpm: float
    beat0: float  # seconds; time of the first beat, in [0, 60/bpm)

    @property
    def period(self) -> float:
        return 60.0 / self.bpm

    @property
    def offset(self) -> float:
        """StepMania #OFFSET convention: time(beat) = -offset + beat*period."""
        return -self.beat0

    def beat_of(self, t: float) -> float:
        return (t - self.beat0) / self.period

    def time_of(self, beat: float) -> float:
        return self.beat0 + beat * self.period


def _refined_bpm(env: np.ndarray, dt: float) -> float:
    """Autocorrelation peak in the BPM band, with parabolic sub-lag interp.

    The raw autocorrelation peaks equally at a tempo and its octaves/harmonics
    (a 200-BPM song also correlates at 100 and 400). We weight the band by a
    log-Gaussian perceptual prior centered on typical rhythm-game tempo, so the
    peak lands in the right octave instead of half/double -- the standard way to
    break octave ambiguity (cf. Ellis 2007 beat tracking).
    """
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]  # lags >= 0
    if ac.size < 3 or ac[0] <= 0:
        return FALLBACK_BPM

    lag_min = max(1, int(round((60.0 / MAX_BPM) / dt)))
    lag_max = min(len(ac) - 2, int(round((60.0 / MIN_BPM) / dt)))
    if lag_max <= lag_min:
        return FALLBACK_BPM

    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 / (lags * dt)
    prior = np.exp(-0.5 * (np.log2(bpms / TEMPO_CENTER) / TEMPO_OCTAVES) ** 2)
    band = ac[lag_min:lag_max + 1] * prior
    peak = lag_min + int(np.argmax(band))

    # Parabolic interpolation using the two neighboring lags -> fractional lag.
    y0, y1, y2 = ac[peak - 1], ac[peak], ac[peak + 1]
    denom = y0 - 2 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    bpm = 60.0 / ((peak + delta) * dt)
    return float(np.clip(bpm, MIN_BPM, MAX_BPM))


def _comb(times: np.ndarray, env: np.ndarray, period: float,
          phase_res: float = 0.006):
    """Best beat phase for a given period + the on-grid onset energy there.

    Slides a pulse train of spacing `period` over the onset envelope and returns
    the (phase, score) whose beats collect the most energy.
    """
    end = float(times[-1])
    n_steps = max(16, int(round(period / phase_res)))
    best_phi, best_score = 0.0, -1.0
    for k in range(n_steps):
        phi = period * k / n_steps
        grid = np.arange(phi, end, period)
        score = float(np.interp(grid, times, env, left=0.0, right=0.0).sum())
        if score > best_score:
            best_score, best_phi = score, phi
    return best_phi, best_score


def _joint_refine(times: np.ndarray, env: np.ndarray, bpm0: float):
    """Search BPM near the autocorr seed for the grid that best fits onsets.

    Autocorrelation is quantized to coarse frame-lags; here we test a fine sweep
    of nearby tempos and keep the (bpm, phase) with the most on-grid energy. A
    tight tempo matters over a long song: a 0.2%% error drifts >100ms end-to-end,
    enough to snap late notes onto the wrong subdivision.
    """
    if times.size < 2:
        return bpm0, 0.0
    # Coarse pass (+/-4%, ~0.1% steps) then a fine pass around the winner.
    def sweep(lo, hi, step):
        best = (bpm0, 0.0, -1.0)
        bpm = lo
        while bpm <= hi:
            phi, score = _comb(times, env, 60.0 / bpm)
            if score > best[2]:
                best = (bpm, phi, score)
            bpm += step
        return best

    b, _, _ = sweep(bpm0 * 0.96, bpm0 * 1.04, bpm0 * 0.001)
    b, phi, _ = sweep(b * 0.999, b * 1.001, b * 0.00005)
    return b, phi


def _align_phase(onset_times: np.ndarray, strengths: np.ndarray,
                 period: float) -> float:
    """Phase that best lands the beat grid ON the detected onset clusters.

    The comb (above) locks phase to the strongest ENERGY, which for syncopated
    songs is often an offbeat -- fine for gameplay (still a clean sub-beat) but
    it must land on a real cluster, not float between them. This slides the grid
    over the DISCRETE onsets and picks the phase collecting the most onset
    strength within a tight window, so snapping stays clean and drift-free.
    """
    if onset_times.size < 2:
        return 0.0
    win = 0.025  # seconds; tight so only near-grid onsets count
    n_steps = max(48, int(round(period / 0.005)))
    best_phi, best_score = 0.0, -1.0
    for k in range(n_steps):
        phi = period * k / n_steps
        dist = np.abs(((onset_times - phi + period / 2) % period) - period / 2)
        score = float(strengths[dist < win].sum())
        if score > best_score:
            best_score, best_phi = score, phi
    return best_phi % period


def estimate_grid(mono: np.ndarray, sr: int, hop: int = 512,
                  onsets=None) -> BeatGrid:
    """Estimate tempo AND phase. Pass detected `onsets` (OnsetEvents) for the
    accurate discrete-onset phase alignment; without them, phase falls back to
    the energy comb."""
    times, env = onset_envelope(mono, sr, hop=hop)
    if len(env) < 4:
        return BeatGrid(FALLBACK_BPM, 0.0)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else hop / sr
    if dt <= 0:
        return BeatGrid(FALLBACK_BPM, 0.0)

    bpm = _refined_bpm(env, dt)          # octave chosen by the perceptual prior
    bpm, beat0 = _joint_refine(times, env, bpm)  # nails PERIOD; phase is a seed
    period = 60.0 / bpm

    if onsets:
        ot = np.array([o.time for o in onsets], dtype=float)
        ws = np.array([o.strength for o in onsets], dtype=float)
        beat0 = _align_phase(ot, ws, period)
    return BeatGrid(bpm=round(bpm, 3), beat0=beat0)
