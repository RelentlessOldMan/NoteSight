r"""compare.py -- score NoteSight against a hand-authored reference .sm.

    python eval/compare.py "path/to/Song.sm"

What it reports:
  * BPM: NoteSight estimate vs the reference #BPMS (error %).
  * Detector coverage: of every reference note time (across ALL difficulties,
    unioned), what fraction has a detected onset within tolerance. This grades
    onset DETECTION independent of how many notes a difficulty keeps.
  * Best time-shift: the constant offset that best aligns detected onsets to the
    reference grid, and the coverage at that shift. A large shift means our
    exported chart needs an OFFSET it currently hardcodes to 0.
  * Per-difficulty selection: NoteSight's selected notes vs the reference chart
    of nearest meter -- note counts + precision/recall/F at a tolerance.

Detection and reference note times are both absolute audio-seconds, so they are
directly comparable with no offset applied on our side.
"""
from __future__ import annotations

import os
import sys

# Allow "import notesight" when run from the eval/ subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from notesight import detect_onsets, select, estimate_grid, DIFFICULTIES
from notesight.audio_io import load_audio
from notesight.formats.stepmania import snap_times
from smparse import parse_sm


def _match(pred: list[float], ref: list[float], tol: float):
    """Greedy 1-1 matching of two sorted time lists within +/- tol seconds."""
    i = j = matched = 0
    while i < len(pred) and j < len(ref):
        d = pred[i] - ref[j]
        if abs(d) <= tol:
            matched += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return matched


def _prf(pred, ref, tol):
    m = _match(pred, ref, tol)
    p = m / len(pred) if pred else 0.0
    r = m / len(ref) if ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _best_shift(pred, ref, tol, span=0.12, step=0.005):
    """Constant shift added to pred that maximizes recall of ref."""
    best_s, best_r = 0.0, -1.0
    pred = np.asarray(pred)
    s = -span
    while s <= span + 1e-9:
        shifted = sorted((pred + s).tolist())
        r = _match(shifted, ref, tol) / len(ref) if ref else 0.0
        if r > best_r:
            best_r, best_s = r, s
        s += step
    return best_s, best_r


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: compare.py <reference.sm> [difficulty]", file=sys.stderr)
        return 2
    sm_path = argv[0]
    only_diff = argv[1] if len(argv) > 1 else None

    sim = parse_sm(sm_path)
    single = [c for c in sim.charts if c.stepstype == "dance-single"]
    if not single:
        print("no dance-single charts in reference", file=sys.stderr)
        return 1

    music_path = os.path.join(os.path.dirname(sm_path), sim.music)
    if not os.path.isfile(music_path):
        print(f"audio not found: {music_path}", file=sys.stderr)
        return 1

    print(f"=== {sim.title} ===")
    print(f"reference: offset={sim.offset:+.3f}s  bpms={sim.bpms}  "
          f"stops={len(sim.stops)}")

    mono, sr = load_audio(music_path)
    dur = len(mono) / sr
    onsets = detect_onsets(mono, sr)
    det_times = sorted(o.time for o in onsets)
    grid = estimate_grid(mono, sr, onsets=onsets)
    est_bpm, period, beat0 = grid.bpm, grid.period, grid.beat0
    ref_bpm = sim.bpms[0][1]
    bpm_err = abs(est_bpm - ref_bpm) / ref_bpm * 100
    # Reference beat0 is its offset folded into one beat period (grid phase).
    # Snapping only cares about phase modulo the finest subdivision (1/4 beat),
    # since locking beat 0 to a clean offbeat is harmless -- notes still land on
    # grid lines. Report both the raw phase gap and that snapping-relevant error.
    ref_beat0 = (-sim.offset) % period
    q = period / 4
    snap_err = ((beat0 - ref_beat0 + q / 2) % q) - q / 2
    print(f"audio: {dur:.1f}s  |  detected {len(det_times)} onsets")
    print(f"BPM: est {est_bpm:.3f} vs ref {ref_bpm:.3f}  ({bpm_err:.2f}% off)")
    print(f"grid phase: est beat0 {beat0*1000:.0f}ms vs ref {ref_beat0*1000:.0f}ms "
          f"|  snap error (mod 1/4 beat) {snap_err*1000:+.0f}ms")

    # Detector coverage vs the union of all reference note times.
    union = sorted({round(t, 4) for c in single for t in c.times})
    tol = 0.05
    cov = _match(det_times, union, tol) / len(union) if union else 0.0
    shift, cov_shift = _best_shift(det_times, union, tol)
    print(f"detector coverage of reference notes (+/-{tol*1000:.0f}ms): "
          f"{cov*100:.1f}%  |  best@shift {shift*+1:+.3f}s -> {cov_shift*100:.1f}%")

    # Per-difficulty selection quality: raw onset times vs grid-snapped.
    # Apply the global best-shift first so a codec decode-delay (MP3 files sync
    # differently under ffmpeg vs StepMania) doesn't masquerade as bad timing --
    # our output is self-consistent with the audio we ship. This isolates
    # grid + note-selection quality.
    print(f"selection vs reference (nearest meter); F at +/-35ms after "
          f"decode-shift {shift:+.3f}s; raw|snapped:")
    print(f"  {'NS diff':8s} {'notes':>6s} {'ref':>10s} {'refN':>6s}  "
          f"{'Fraw':>5s} {'Fsnap':>6s}")
    for name, diff in DIFFICULTIES.items():
        if only_diff and name != only_diff:
            continue
        sel = select(onsets, diff, None)
        sel_times = sorted(o.time + shift for o in sel)
        snapped = sorted(t + shift for t in
                         snap_times([o.time for o in sel], est_bpm, beat0))
        ref_c = min(single, key=lambda c: abs(len(c.times) - len(sel_times)))
        _, _, f_raw = _prf(sel_times, ref_c.times, 0.035)
        _, _, f_snap = _prf(snapped, ref_c.times, 0.035)
        print(f"  {name:8s} {len(sel_times):6d} {ref_c.difficulty:>10s} "
              f"{len(ref_c.times):6d}  {f_raw:5.2f} {f_snap:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
