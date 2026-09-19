"""stamina.py -- a stamina/cardio chart: a long continuous STREAM for a workout.

Real ITG stamina charts aren't one giant hand-authored pattern; they're a
continuous stream built from short, measure-aligned pattern IDEAS that change
every phrase. We mirror that hierarchy:

  CELL   4-8 notes -- the atomic idea (a staircase, an anchor, an L/R split).
  MOTIF  16-32     -- a cell repeated, usually A A B B, occasionally 8/32-note.
  PHRASE           -- the motif that's active; the mapper swaps motif each phrase
                      so the stream stays interesting over hundreds of notes.
  SONG             -- phrases tile the whole short song, resting only where it's
                      quiet (energy-gated: fade in/out, breakdowns).

The LONG (e.g. 30-min) chart is then just this whole short-song stream repeated
(see build_ddr_stamina.py) -- the variety lives inside the 2:24, the long version
loops it.

Invariants held everywhere (incl. every cell/motif/loop seam):
  * strict foot alternation -- even slot = LEFT foot, odd = RIGHT foot
  * no crossovers           -- left foot only L/D/U, right foot only R/D/U
  * no jacks / footswitches  -- never the same panel twice in a row
  * no jumps                -- exactly one panel per note
Every cell starts on the LEFT panel, so concatenating cells is always jack-free
(a cell ends on a right-foot panel R/D/U, the next starts on L) and even lengths
keep the L/R alternation locked.
"""
from __future__ import annotations

import bisect
import math

import numpy as np

LEFT, DOWN, UP, RIGHT = 0, 1, 2, 3

# 4-note cells. All start on LEFT; left slots (0,2) in {L,D,U}, right slots (1,3)
# in {R,D,U}; no panel repeats back-to-back (incl. wrap), so a cell is safe to
# repeat and to butt against any other cell.
_CELLS = {
    "stair_up":  (LEFT, DOWN, UP, RIGHT),    # walk up the panels
    "stair_dn":  (LEFT, UP, DOWN, RIGHT),    # ...different inner order
    "anchorL_a": (LEFT, DOWN, LEFT, UP),     # LEFT foot anchors on Left
    "anchorL_b": (LEFT, UP, LEFT, DOWN),     # LEFT foot anchors on Left (var)
    "anchorD":   (LEFT, DOWN, UP, DOWN),     # RIGHT foot anchors on Down
    "anchorU":   (LEFT, UP, DOWN, UP),       # RIGHT foot anchors on Up
    "split_a":   (LEFT, RIGHT, UP, DOWN),    # quick L<->R split then centres
    "split_b":   (LEFT, RIGHT, DOWN, UP),
}

# Phrase recipes: (structure over {'A','B'}, cellA, cellB). Mostly 16-note
# (2-measure) A A B B / A B A B, with an 8 and a 32 sprinkled in so it doesn't
# feel mechanical. The mapper walks this list, one recipe per phrase.
_RECIPES = [
    ("AABB", "stair_up", "stair_dn"),   # the classic A A B B stream
    ("AAAA", "anchorD", "anchorD"),     # a Down-anchor push
    ("ABAB", "stair_up", "anchorL_a"),  # staircase weaving a Left anchor
    ("AABB", "split_a", "split_b"),     # L/R split motif
    ("AAAA", "anchorL_b", "anchorL_b"), # a Left-anchor push
    ("AABB", "stair_dn", "anchorU"),
    ("AB", "stair_up", "split_a"),      # 8-note breather (1 measure)
    ("AABBAABB", "stair_up", "anchorD"),  # 32-note (4-measure) build
]


def _motif(structure, a, b):
    cells = {"A": _CELLS[a], "B": _CELLS[b]}
    seq = []
    for s in structure:
        seq.extend(cells[s])
    return seq


def build_stream(n):
    """A length-n lane sequence: phrases (motifs) concatenated, changing every
    phrase. Deterministic (same every build) so the looped long-song stays exact."""
    out = []
    p = 0
    while len(out) < n:
        struct, a, b = _RECIPES[p % len(_RECIPES)]
        out.extend(_motif(struct, a, b))
        p += 1
    return out[:n]


def stream_segments(n):
    """Segment a length-n build_stream into its phrases, mirroring the recipe walk:
    returns [(start, length, recipe_idx), ...]. Two phrases with the SAME recipe_idx
    are the SAME motif (same lanes), so a voicer can voice one and REUSE it wherever
    that motif recurs -- the 'oh, I know this flow' repetition layer."""
    segs = []
    p = pos = 0
    while pos < n:
        struct, a, b = _RECIPES[p % len(_RECIPES)]
        length = min(len(_motif(struct, a, b)), n - pos)
        segs.append((pos, length, p % len(_RECIPES)))
        pos += length
        p += 1
    return segs


_PANEL = {LEFT: "L", DOWN: "D", UP: "U", RIGHT: "R"}
_LF_OK = {LEFT, DOWN, UP}
_RF_OK = {RIGHT, DOWN, UP}


def foot_of(i):
    """Inferred foot for stream index i (even = left, odd = right)."""
    return "LF" if i % 2 == 0 else "RF"


def annotate(n=32):
    """Debug render of the raw stream with inferred feet, e.g. 'L(LF) D(RF) ...'.
    (Recommended: eyeball this to catch bad footflow far faster than by playing.)"""
    return " ".join(f"{_PANEL[l]}({foot_of(i)})"
                    for i, l in enumerate(build_stream(n)))


def validate_stream(n=4096):
    """Assert the stamina invariants across a long stretch: strict foot
    alternation w/ no crossover (left foot never Right, right never Left) and no
    jack/footswitch (no panel twice in a row). Raises AssertionError on violation."""
    lanes = build_stream(n)
    for i, l in enumerate(lanes):
        ok = l in (_LF_OK if i % 2 == 0 else _RF_OK)
        assert ok, f"crossover @ {i}: {_PANEL[l]} on {foot_of(i)}"
        if i and l == lanes[i - 1]:
            raise AssertionError(f"jack @ {i}: {_PANEL[l]} repeated")
    return True


def _energy_at(e_times, energy, t, song_dur):
    if t < 0 or t > song_dur or not len(energy):
        return 0.0
    i = bisect.bisect_left(e_times, t)
    return float(energy[min(i, len(energy) - 1)])


# STRAIN MODEL. A flat stream is a wall; real stamina charts have an endurance
# SHAPE -- build, climax, recover. We drive local density off the song's energy
# (drops get denser, breakdowns lighter) AROUND the tier's average, then overlay a
# fatigue accumulator: sustained above-average density builds strain, and once it
# crosses a cap we force a brief lighter "break" (which lets strain shed) before
# ramping back up. So even a long loud section gets burst/break structure.
STRAIN_AMP = 1.3        # how hard energy swings density around the tier average
STRAIN_LOW = 0.5        # floor density multiplier (deep recovery / breakdown)
STRAIN_HIGH = 1.6       # ceiling density multiplier (climax)
STRAIN_CAP = 20.0       # "extra notes" of effort tolerated before a forced break
STRAIN_RELIEF = 0.75    # density multiplier during a forced break
STRAIN_RECOVER = 1.6    # strain sheds this much faster when below average


def _base_sub(bpm, target_nps):
    """Notes-per-beat of the fine placement grid -- a power-of-two subdivision with
    headroom above the PEAK local rate (target * STRAIN_HIGH) so climaxes fit."""
    period = 60.0 / bpm
    sub = 4                                   # 16ths
    while sub / period < target_nps * STRAIN_HIGH * 1.1 and sub < 16:
        sub *= 2
    return sub


def _energy_series(e_times, energy, positions, song_dur):
    """Energy (0..1) at each grid position; 0 in the padded/quiet tail so those
    become rests. 0.5 flat if no envelope was supplied."""
    if e_times is None or energy is None or len(e_times) == 0:
        return np.full(len(positions), 0.5)
    v = np.interp(positions, e_times, energy)
    v[(positions < 0) | (positions > song_dur)] = 0.0
    return v


def _smooth(v, positions, win_sec=1.5):
    """Smooth over ~win_sec so the density SHAPE tracks sections, not per-note
    jitter."""
    if len(v) < 2:
        return v
    dt = positions[1] - positions[0]
    w = max(1, int(round(win_sec / max(dt, 1e-6))))
    if w <= 1:
        return v
    return np.convolve(v, np.ones(w) / w, mode="same")


def _density_profile(inten, base_nps, step):
    """Per-position target NPS: energy-scaled around base_nps (centred on the
    loop's mean so the average stays ~base_nps), with a strain accumulator that
    forces periodic breaks in long climaxes."""
    if not len(inten):
        return np.zeros(0)
    mean_i = float(np.mean(inten))
    mult = np.clip(1.0 + STRAIN_AMP * (inten - mean_i), STRAIN_LOW, STRAIN_HIGH)
    out = np.empty(len(inten))
    strain = 0.0
    for k in range(len(inten)):
        m = mult[k]
        if strain > STRAIN_CAP:               # fatigued -> force a break
            m = min(m, STRAIN_RELIEF)
        out[k] = base_nps * m
        d = (m - 1.0) * base_nps * step       # effort above baseline, per step
        strain += d if d > 0 else d * STRAIN_RECOVER
        if strain < 0.0:
            strain = 0.0
    return out


def one_loop(bpm, beat0, e_times, energy, loop_dur, song_dur,
             target_nps=5.0, thresh=0.18):
    """One loop unit whose density has an endurance SHAPE (see the strain model).
    A fine grid is placed at a VARIABLE local rate that follows the energy/strain
    profile; near-silence is gated to rests. Lanes come from the cell->motif stream
    in EMISSION order, so consecutive notes always alternate feet cleanly."""
    period = 60.0 / bpm
    sub = _base_sub(bpm, target_nps)
    step = period / sub
    emax = max(energy) if len(energy) else 1.0
    floor = thresh * emax

    positions = []
    j = math.ceil((0.0 - beat0) / step)       # first grid slot at/after t=0
    while True:
        t = beat0 + j * step
        if t >= loop_dur:
            break
        positions.append(t)
        j += 1
    positions = np.array(positions)
    if not len(positions):
        return [], sub

    e_at = _energy_series(e_times, energy, positions, song_dur)
    nps_prof = _density_profile(_smooth(e_at, positions), target_nps, step)

    # Variable-rate placement: accumulate the local expected note-count and emit
    # when it ticks over 1 -> local rate tracks nps_prof, evenly spread.
    times = []
    acc = 0.0
    for t, npl, e in zip(positions, nps_prof, e_at):
        if e < floor:                         # near-silence -> rest
            acc = 0.0
            continue
        acc += npl * step
        if acc >= 1.0:
            times.append(float(t))
            acc -= 1.0

    lanes = build_stream(len(times))
    return list(zip(times, lanes)), sub


def stamina_notes(bpm, beat0, e_times, energy, loop_dur, n_loops, song_dur,
                  target_nps=5.0, thresh=0.18):
    """Full chart at a target NPS: the one-loop stream repeated n_loops times.
    Each loop is a whole number of measures, so copies stay on the beat grid and
    every loop is the SAME short-song stream. Returns (notes, base_subdivision)."""
    unit, sub = one_loop(bpm, beat0, e_times, energy, loop_dur, song_dur,
                         target_nps, thresh)
    out = []
    for k in range(n_loops):
        off = k * loop_dur
        out.extend((t + off, lane) for t, lane in unit)
    return out, sub
