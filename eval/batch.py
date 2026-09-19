r"""batch.py -- run NoteSight across a whole StepMania Songs folder and summarize.

    python eval/batch.py "path/to/StepMania/Songs/SomePack" [N]

For every song folder with a dance-single .sm chart + its audio, measures how
close NoteSight's estimated grid is to the hand-authored ground truth (BPM error,
snap phase error) and the best snapped note-agreement F over difficulties, then
prints per-song lines and aggregate medians. Songs whose reference has tempo
changes are flagged (our grid is single-tempo) but still scored on nearest BPM.
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from notesight import detect_onsets, select, estimate_grid, DIFFICULTIES
from notesight.audio_io import load_audio
from notesight.formats.stepmania import snap_times
from smparse import parse_sm

AUDIO_EXT = (".ogg", ".mp3", ".wav", ".m4a", ".flac")


def _match(pred, ref, tol):
    i = j = m = 0
    while i < len(pred) and j < len(ref):
        d = pred[i] - ref[j]
        if abs(d) <= tol:
            m += 1; i += 1; j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return m


def _bestF(pred, ref, tol=0.035):
    m = _match(pred, ref, tol)
    p = m / len(pred) if pred else 0.0
    r = m / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _best_shift(pred, ref, tol=0.05, span=0.15, step=0.005):
    pred = np.asarray(pred)
    best = (0.0, -1.0)
    s = -span
    while s <= span + 1e-9:
        r = _match(sorted((pred + s).tolist()), ref, tol) / len(ref) if ref else 0
        if r > best[1]:
            best = (s, r)
        s += step
    return best[0]


def _find_audio(song_dir, music_hint):
    if music_hint:
        p = os.path.join(song_dir, music_hint)
        if os.path.isfile(p):
            return p
    for f in os.listdir(song_dir):
        if f.lower().endswith(AUDIO_EXT):
            return os.path.join(song_dir, f)
    return None


def run_song(sm_path):
    sim = parse_sm(sm_path)
    single = [c for c in sim.charts if c.stepstype == "dance-single"]
    if not single:
        return None
    audio = _find_audio(os.path.dirname(sm_path), sim.music)
    if not audio:
        return None

    mono, sr = load_audio(audio)
    onsets = detect_onsets(mono, sr)
    det = sorted(o.time for o in onsets)
    grid = estimate_grid(mono, sr, onsets=onsets)
    period = grid.period

    ref_bpm = sim.bpms[0][1]
    bpm_err = abs(grid.bpm - ref_bpm) / ref_bpm * 100
    q = period / 4
    ref_beat0 = (-sim.offset) % period
    snap_err = abs(((grid.beat0 - ref_beat0 + q / 2) % q) - q / 2) * 1000

    union = sorted({round(t, 4) for c in single for t in c.times})
    shift = _best_shift(det, union)

    best_f = 0.0
    for diff in DIFFICULTIES.values():
        sel = select(onsets, diff, None)
        snapped = sorted(t + shift for t in
                         snap_times([o.time for o in sel], grid.bpm, grid.beat0))
        ref_c = min(single, key=lambda c: abs(len(c.times) - len(snapped)))
        best_f = max(best_f, _bestF(snapped, ref_c.times))

    return {
        "title": sim.title or os.path.basename(os.path.dirname(sm_path)),
        "tempo_changes": len(sim.bpms) > 1,
        "bpm_err": bpm_err, "snap_err": snap_err, "bestF": best_f,
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: batch.py <SongsFolder> [max_songs]", file=sys.stderr)
        return 2
    root = argv[0]
    limit = int(argv[1]) if len(argv) > 1 else 9999

    # Case-insensitive filesystems match *.sm and *.SM to the same file; dedupe.
    seen = {}
    for p in glob.glob(os.path.join(root, "*", "*.sm")) + \
            glob.glob(os.path.join(root, "*", "*.SM")):
        seen[p.lower()] = p
    sms = sorted(seen.values())[:limit]
    print(f"scanning {len(sms)} songs under {root}\n")
    print(f"{'song':38s} {'BPMerr%':>8s} {'snapErr':>8s} {'bestF':>6s}  flags")

    rows = []
    for sm in sms:
        try:
            r = run_song(sm)
        except Exception as e:
            print(f"{os.path.basename(os.path.dirname(sm))[:38]:38s}  ERROR {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        flag = "tempo-changes" if r["tempo_changes"] else ""
        title = r["title"].encode("ascii", "replace").decode()[:38]
        print(f"{title:38s} {r['bpm_err']:8.2f} "
              f"{r['snap_err']:7.0f}m {r['bestF']:6.2f}  {flag}")

    if not rows:
        print("no scorable songs")
        return 1

    def med(key):
        return float(np.median([r[key] for r in rows]))

    steady = [r for r in rows if not r["tempo_changes"]]
    within = sum(r["bpm_err"] < 0.1 for r in steady)
    print(f"\n=== {len(rows)} songs ({len(steady)} steady-BPM) ===")
    print(f"median BPM error:  {med('bpm_err'):.3f}%   "
          f"steady within 0.1%: {within}/{len(steady)}")
    print(f"median snap error: {med('snap_err'):.0f} ms")
    print(f"median best F:     {med('bestF'):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
