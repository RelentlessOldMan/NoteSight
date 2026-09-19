r"""bs_rocks.py -- the ROCK VOCABULARY for the big-rocks-then-sand charter (v2).

A ROCK is a short, PRE-VALIDATED motif figure: a sequence of note-slots, each a SINGLE
or a two-hand DOUBLE. Every rock is guaranteed, BY CONSTRUCTION:
  * HITTABLE   -- no double where a saber swings through the other block (_double_ok),
  * PARITY-CLEAN -- a hand never swings the same vertical direction twice in a row,
  * SELF-CLEAR -- no consecutive step drops an opposite-hand note in a just-used cell.

The assembler places big rocks at musical anchors (recurring anchors reuse the SAME rock,
mirrored on alternate hits -> "oh, I know this flow") and fills the gaps between them with
sand. Because rocks are clean up front, there is nothing to nudge afterwards.

Step formats (ABSOLUTE Beat Saber coords; hand 0=red/left 1=blue/right, cut 0-8 per the
game, col 0-3 left->right, row 0-2 bottom->top):
    ("S", hand, cut, col, row)                 -- a single
    ("D", rc, rcol, rrow, bc, bcol, brow)      -- a two-hand double (red then blue)

`mirror_rock` flips a rock left<->right (col->3-col, cut->MCUT, hand swap) -- a mirrored
recurrence still reads as "the same flow, other side" and stays valid (mirror-symmetric).
`validate_rock` is the gatekeeper; `ROCKS` only contains figures that pass it (asserted at
import), so the assembler can trust any rock it pulls.
"""
from __future__ import annotations

from .bs_model import _double_ok, _UP, _DOWN, MCUT

RED, BLUE = 0, 1


def _vclass(cut):
    return "U" if cut in _UP else "D" if cut in _DOWN else "S"


def _units(step):
    """Yield (hand, cut, col, row) for each block a step places (1 for S, 2 for D)."""
    if step[0] == "S":
        _, hand, cut, col, row = step
        yield hand, cut, col, row
    else:
        _, rc, rcol, rrow, bc, bcol, brow = step
        yield RED, rc, rcol, rrow
        yield BLUE, bc, bcol, brow


def entry_state(rock):
    """Per-hand (vclass, col, row) the rock OPENS with -- what the incoming sand must set
    up to (parity-wise). None for a hand the first step doesn't use."""
    st = {RED: None, BLUE: None}
    for step in rock:
        for hand, cut, col, row in _units(step):
            if st[hand] is None:
                st[hand] = (_vclass(cut), col, row)
        if all(st[h] is not None for h in (RED, BLUE)):
            break
    return st


def exit_state(rock):
    """Per-hand (vclass, col, row) the rock ENDS on -- the state the following sand
    inherits so the seam flows."""
    st = {RED: None, BLUE: None}
    for step in rock:
        for hand, cut, col, row in _units(step):
            st[hand] = (_vclass(cut), col, row)
    return st


def validate_rock(rock):
    """True iff every step is hittable, parity-clean per hand, and no consecutive step
    stacks an opposite-hand note in a cell the previous step just used."""
    lastv = {RED: None, BLUE: None}
    prev_cells = set()
    for step in rock:
        if step[0] == "D":
            _, rc, rcol, rrow, bc, bcol, brow = step
            if not _double_ok(rc, rcol, rrow, bc, bcol, brow):
                return False
        cells = set()
        for hand, cut, col, row in _units(step):
            if not (0 <= col <= 3 and 0 <= row <= 2 and 0 <= cut <= 8):
                return False
            v = _vclass(cut)
            if v in ("U", "D") and lastv[hand] == v:        # same-hand parity reset
                return False
            lastv[hand] = v
            cells.add((col, row))
            # opposite-hand note landing in a cell the PREVIOUS step used = occlusion
            if (col, row) in prev_cells:
                return False
        prev_cells = cells
    return True


def mirror_rock(rock):
    """Flip the rock left<->right (and red<->blue). Stays valid by mirror symmetry."""
    out = []
    for step in rock:
        if step[0] == "S":
            _, hand, cut, col, row = step
            out.append(("S", 1 - hand if hand in (RED, BLUE) else hand,
                        MCUT[cut], 3 - col, row))
        else:
            _, rc, rcol, rrow, bc, bcol, brow = step
            # mirror swaps which hand is red/blue: new red = old blue mirrored, etc.
            out.append(("D", MCUT[bc], 3 - bcol, brow, MCUT[rc], 3 - rcol, rrow))
    return out


# --- the starting vocabulary (expandable) -------------------------------------------
# DOUBLE rocks = the fun "hand-circle" motions: OUTWARD chops (sabers swing apart) and
# parallel vertical pumps, arranged so consecutive steps never reset a hand's parity.
# SINGLE rocks = clean connective flows (staircases / anchors) usable as smaller rocks.
_LIB = {
    # big: a 4-step hand circle -- OUTWARD chops (only ever at the wide cols 0 & 3) that
    # alternate with parallel vertical pumps (which sit near, cols 1 & 2). Cells shift
    # each step so nothing stacks; per hand the vclass reads S,D,S,U (no reset).
    "circle": [
        ("D", 2, 0, 1, 3, 3, 1),      # outward chop, WIDE  (red c0 L, blue c3 R) row1
        ("D", 1, 1, 1, 1, 2, 1),      # both DOWN, near     (cols 1,2 row1)
        ("D", 2, 0, 0, 3, 3, 0),      # outward chop, WIDE  (cols 0,3 row0)
        ("D", 0, 1, 0, 0, 2, 0),      # both UP, near       (cols 1,2 row0)
    ],
    # big: a 4-step windshield wiper -- OUTWARD DIAGONAL chops, always WIDE (cols 0 & 3),
    # alternating up/down (per hand vclass U,D,U,D); rows alternate so cells never repeat.
    "wiper": [
        ("D", 4, 0, 1, 5, 3, 1),      # outward up-diagonal, wide  (red UL c0, blue UR c3)
        ("D", 6, 0, 0, 7, 3, 0),      # outward down-diagonal, wide low
        ("D", 4, 0, 1, 5, 3, 1),      # outward up-diagonal, wide
        ("D", 6, 0, 0, 7, 3, 0),      # outward down-diagonal, wide low
    ],
    # medium: a 3-step vertical pump (parallel verticals sit near -- that's fine)
    "pump": [
        ("D", 1, 1, 1, 1, 2, 1),      # both down, near
        ("D", 0, 1, 0, 0, 2, 0),      # both up, near
        ("D", 1, 0, 1, 1, 3, 1),      # both down, wide
    ],
    # small: a 4-step alternating staircase (hands + direction both alternate)
    "stair": [
        ("S", RED, 1, 0, 0),          # red down, far-left low
        ("S", BLUE, 0, 3, 0),         # blue up, far-right low
        ("S", RED, 0, 1, 1),          # red up, center-left mid
        ("S", BLUE, 1, 2, 1),         # blue down, center-right mid
    ],
    # small: a 2-step accent -- WIDE outward chop then a near vertical
    "accent": [
        ("D", 2, 0, 1, 3, 3, 1),      # outward chop, WIDE (cols 0,3)
        ("D", 0, 1, 0, 0, 2, 0),      # both up, near
    ],
}

# keep only valid figures (and assert the whole starting set is clean)
ROCKS = {name: r for name, r in _LIB.items() if validate_rock(r)}
assert set(ROCKS) == set(_LIB), (
    "invalid rock(s): " + ", ".join(n for n in _LIB if n not in ROCKS))

# rough size tiers so the assembler can reach for "smaller and smaller" rocks
BIG = ["circle", "wiper"]
MEDIUM = ["pump"]
SMALL = ["stair", "accent"]
