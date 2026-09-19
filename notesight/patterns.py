"""patterns.py -- music-driven lane assignment (the chart's "feel").

Turns selected onsets into 4-panel arrow columns the way real charts read AND
play -- no user knobs, difficulty is the only dial.

The model is FOOT ALTERNATION, the thing that makes a chart comfortable: steps
alternate left foot / right foot, so consecutive notes never land on the same
panel (no jacks) and never force a crossover (the left foot only uses Left plus
the two center panels, the right foot only uses Right plus the centers). Within
those constraints the pitch (spectral brightness) picks the panel -- higher pitch
leans toward the up/right panels, lower toward left/down -- and a short "avoid
what I just played" memory keeps it varied instead of sweeping the same line.

  * FOOT ALTERNATION -> playable: alternating feet, no jacks, no crossovers.
  * PITCH BIAS       -> musical: the melody's contour shapes the steps.
  * RECENT-PANEL PENALTY -> variety: it won't grind the same 2-panel zigzag.
  * MOTIF MEMORY     -> learnable: a repeated phrase replays the same figure.

Jumps on strong onsets stay automatic (driven by the difficulty preset). Returns
plain (time, lane, strength) tuples so this stays free of the format layer.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .onsets import OnsetEvent
from .difficulty import Difficulty

LEFT, DOWN, UP, RIGHT = 0, 1, 2, 3
# Which panels each foot can reach without crossing over: its own side + the two
# center panels (Down/Up), which either foot can hit.
LEFT_FOOT = (LEFT, DOWN, UP)
RIGHT_FOOT = (RIGHT, DOWN, UP)
# Horizontal "height" of each panel for pitch mapping (Left low .. Right high).
HEIGHT = {LEFT: 0.0, DOWN: 1.0, UP: 2.0, RIGHT: 3.0}

# Pad symmetries (StepMania's Mirror / Flip mods). Each maps the arrow cross onto
# itself, so a comfortable alternating-feet motif stays legal -- it just moves to
# another part of the pad. This is free variety at zero added difficulty: e.g.
# LUL under 180 becomes RDR (same R-L-R flow, body shifted to the far corner).
# Indexed by lane -> new lane.
TRANSFORMS = (
    (LEFT, DOWN, UP, RIGHT),     # identity
    (LEFT, UP, DOWN, RIGHT),     # flip U<->D
    (RIGHT, DOWN, UP, LEFT),     # mirror L<->R
    (RIGHT, UP, DOWN, LEFT),     # 180: L<->R and U<->D  (LUL -> RDR)
)

JUMP_STRENGTH = 0.85     # onset salience that earns a jump (~0.27/s, DDR-like)
JUMP_MIN_GAP = 0.30      # min seconds between jumps -> always a single between
RECENT_PENALTY = 2.0     # how strongly to avoid a just-used panel (variety)
RECENT_LEN = 2           # how many recent panels to remember

# Jacks: when the melody REPEATS a pitch, real DDR charts hammer the same arrow
# a few times ("up up up") instead of alternating -- it reads as emphasis and is
# a big share of hand-authored texture (A20 is 12-30% jacks). We do the same when
# consecutive onsets are within JACK_EPS of the same brightness, capped at
# JACK_MAX repeats so a jack is never longer than 3 in a row (the user's limit).
JACK_EPS = 0.018         # brightness closeness (fraction of range) = same pitch
JACK_MAX = 2             # up to 2 repeats -> at most 3 same-panel notes in a row

# Phrase segmentation for motif reuse: break on a rest, and cap the length so
# repeated hooks stay short enough to match.
GAP_FACTOR = 3.0
MIN_GAP = 0.5
MAX_PHRASE = 16


def _sign(x: float, eps: float) -> int:
    return 0 if abs(x) < eps else (1 if x > 0 else -1)


def _segment(ev: list[OnsetEvent]) -> list[list[OnsetEvent]]:
    if len(ev) <= 1:
        return [ev]
    gaps = np.diff([o.time for o in ev])
    thresh = max(MIN_GAP, GAP_FACTOR * float(np.median(gaps)))
    phrases, cur = [], [ev[0]]
    for i in range(1, len(ev)):
        if ev[i].time - ev[i - 1].time > thresh or len(cur) >= MAX_PHRASE:
            phrases.append(cur)
            cur = [ev[i]]
        else:
            cur.append(ev[i])
    phrases.append(cur)
    return phrases


def _signature(phrase: list[OnsetEvent], eps: float) -> tuple:
    return tuple(_sign(phrase[i + 1].brightness - phrase[i].brightness, eps)
                 for i in range(len(phrase) - 1))


def _pick(cands, target_h: float, recent) -> int:
    """Deterministically pick the panel closest to the pitch height, biased away
    from recently used panels so the chart stays varied."""
    best, best_score = cands[0], -1e9
    for p in cands:
        score = -abs(HEIGHT[p] - target_h)
        if p in recent:
            score -= RECENT_PENALTY
        if score > best_score:
            best_score, best = score, p
    return best


def _gen_motif(phrase, lo: float, rng: float, start_foot: int,
               first_avoid: int = -1) -> list[int]:
    """Generate one phrase's canonical foot-motif: alternating feet, no crossover
    (left foot uses L+centers, right uses R+centers), no jacks, pitch choosing the
    panel. `first_avoid` keeps the opening note off the previous phrase's last
    panel so seams don't jack. Symmetry transforms are applied afterwards."""
    foot = start_foot
    last = first_avoid
    recent = deque(maxlen=RECENT_LEN)
    lanes: list[int] = []
    jack_run = 0
    prev_bright = None
    for o in phrase:
        # A repeated pitch -> jack: stay on the same panel/foot (up to JACK_MAX)
        # instead of alternating. Emphasis on a held-still melody note.
        if (prev_bright is not None and last >= 0 and jack_run < JACK_MAX
                and abs(o.brightness - prev_bright) <= JACK_EPS * rng):
            lanes.append(last)
            recent.append(last)
            jack_run += 1
            prev_bright = o.brightness
            continue
        jack_run = 0
        allowed = LEFT_FOOT if foot == 0 else RIGHT_FOOT
        cands = [p for p in allowed if p != last] or list(allowed)
        target_h = (o.brightness - lo) / rng * 3.0
        lane = _pick(cands, target_h, recent)
        lanes.append(lane)
        recent.append(lane)
        last = lane
        foot ^= 1
        prev_bright = o.brightness
    return lanes


def assign(onsets: list[OnsetEvent], diff: Difficulty):
    """Assign 4-panel lanes to selected onsets. Returns (time, lane, strength)."""
    if not onsets:
        return []
    ev = sorted(onsets, key=lambda o: o.time)
    br = np.array([o.brightness for o in ev], dtype=float)
    lo, rng = float(br.min()), float(br.max() - br.min()) or 1.0
    sig_eps = 0.05 * rng

    motifs: dict[tuple, list[int]] = {}
    singles: list[tuple[float, int, float]] = []   # one lane per onset, foot-flowed
    prev_emitted = -1                              # last emitted lane (seam guard)

    for pidx, phrase in enumerate(_segment(ev)):
        # A phrase's canonical foot-motif is generated once (or reused if this
        # exact contour recurred), then a pad symmetry is applied for the section.
        # So a repeated hook comes back RECOGNIZABLE but shifted/mirrored -- same
        # feet, new arrows -- which is how a good chart varies without getting
        # harder. The transform changes every couple phrases (motif, then its
        # mirror), matching "play it, small break, mirrored version next".
        tf = TRANSFORMS[(pidx // 2) % len(TRANSFORMS)]
        sig = _signature(phrase, sig_eps)
        distinctive = len(phrase) >= 4 and len(set(sig)) > 1
        base = motifs.get(sig) if distinctive else None
        if base is None or len(base) != len(phrase):
            # Avoid opening on the previous phrase's emitted panel (in this
            # phrase's transform frame) so the seam doesn't jack.
            inv = [0, 0, 0, 0]
            for i, v in enumerate(tf):
                inv[v] = i
            avoid = inv[prev_emitted] if prev_emitted >= 0 else -1
            base = _gen_motif(phrase, lo, rng, start_foot=pidx & 1,
                              first_avoid=avoid)
            if distinctive:
                motifs[sig] = base
        for o, lane in zip(phrase, base):
            singles.append((o.time, tf[lane], o.strength))
        prev_emitted = tf[base[-1]]

    # Jump-flow pass: a strong onset earns a 2-foot jump (natural opposite,
    # L+R or U+D), but NEVER back-to-back -- keep at least JUMP_MIN_GAP so a jump
    # is always followed by a single note. That "two-foot, one-foot, two-foot"
    # spacing lets the player push off the free foot into the next landing;
    # a pile of consecutive jumps is what makes a chart a slog.
    partner = {LEFT: RIGHT, RIGHT: LEFT, DOWN: UP, UP: DOWN}
    # Per-difficulty jump tuning (fall back to the module defaults for callers
    # that pass an older Difficulty without these fields).
    jump_strength = getattr(diff, "jump_strength", JUMP_STRENGTH)
    jump_gap = getattr(diff, "jump_gap", JUMP_MIN_GAP)
    out: list[tuple[float, int, float]] = []
    last_jump = -1e9
    for t, lane, s in singles:
        out.append((t, lane, s))
        if (diff.allow_jumps and s > jump_strength
                and t - last_jump >= jump_gap):
            out.append((t, partner[lane], s))
            last_jump = t
    return out
