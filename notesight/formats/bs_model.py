"""bs_model.py -- Beat Saber cut constants + the DATA-DRIVEN double HITTABILITY oracle.

Once the transition-model voicer lived here; it's gone (the letter vocabulary replaced it --
see git history). What remains is used by the letter path: the cut-direction constants
(MCUT / _UP / _DOWN / _HDIR / _VFLIP) and `_double_ok`, which says whether a two-hand double is
physically hittable by checking it against the mined real-map vocabulary (bs_doubles.json).
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

RED, BLUE = 0, 1
ANY = 8                                                         # dot note (no direction)
# Cross-hand coordination: real maps place the two sabers at MIRROR-symmetric spots
# ~43% of the time (both sweeping in/out together). A per-hand model under-produces
# that, so we bias a note toward mirroring the other hand's last placement. Because
# the model is mined in a mirrored "red frame", mirrored hands share the SAME
# red-frame placement -- so mirroring = copy the other hand's red-frame note.
MIRROR_BIAS = 0.10
MCUT = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}   # mirror to red frame
_UP = {0, 4, 5}
_DOWN = {1, 6, 7}
# horizontal component of a cut (which way the saber SWEEPS across): -1 left, +1 right, 0
_HDIR = {2: -1, 4: -1, 6: -1, 3: 1, 5: 1, 7: 1, 0: 0, 1: 0, 8: 0}
# vertical flip of a cut (up<->down, horizontal unchanged) -- HITTABILITY is invariant
# under this (columns & horizontal sweep unchanged), so it's a symmetry of "valid double".
_VFLIP = {0: 1, 1: 0, 4: 6, 5: 7, 6: 4, 7: 5, 2: 2, 3: 3, 8: 8}


_DBL_VOCAB = None


def _canon_double(rc, rcol, rrow, bc, bcol, brow):
    """Mirror-canonical key (a double and its L/R mirror are the same shape)."""
    a = (rc, rcol, rrow, bc, bcol, brow)
    b = (MCUT[bc], 3 - bcol, brow, MCUT[rc], 3 - rcol, rrow)
    return min(a, b)


def _load_double_vocab():
    global _DBL_VOCAB
    if _DBL_VOCAB is None:
        try:
            path = os.path.join(os.path.dirname(__file__), "bs_doubles.json")
            _DBL_VOCAB = set(json.load(open(path, encoding="utf-8"))["shapes"].keys())
        except Exception:
            _DBL_VOCAB = set()
    return _DBL_VOCAB


def _double_ok(rc, rcol, rrow, bc, bcol, brow):
    """Is this two-hand double physically HITTABLE? DATA-DRIVEN: a double is valid iff its
    mirror-canonical shape appears in the mined real-map vocabulary (bs_doubles.json,
    1246 shapes = 99.66% of 468k library doubles at thr>=5/>=3 maps). Real maps only
    contain playable doubles, so this is the ground truth -- it keeps the diagonals the
    old hand-coded geometry wrongly rejected (~19%) yet still drops the genuine impossible
    hits (adjacent-outward pure-horizontal appeared 1x in 468k; inward 0x). Falls back to
    the geometric check only if the data file is missing."""
    if rcol == bcol and rrow == brow:
        return False
    vocab = _load_double_vocab()
    if not vocab:
        return _double_ok_geom(rc, rcol, rrow, bc, bcol, brow)
    # accept if the shape OR its vertical-flip (a hittability symmetry) is a real shape
    for RC, BC in ((rc, bc), (_VFLIP[rc], _VFLIP[bc])):
        k = _canon_double(RC, rcol, rrow, BC, bcol, brow)
        if "%d,%d,%d,%d,%d,%d" % k in vocab:
            return True
    return False


def _double_ok_geom(rc, rcol, rrow, bc, bcol, brow):
    """Geometric fallback (used only if bs_doubles.json is absent)."""
    if rcol == bcol and rrow == brow:                      # same exact cell
        return False
    hr, hb = _HDIR.get(rc, 0), _HDIR.get(bc, 0)
    if abs(rrow - brow) <= 1:                              # only matters at similar height
        if hr < 0 and bcol < rcol:                         # red sweeps left THROUGH blue
            return False
        if hr > 0 and bcol > rcol:                         # red sweeps right through blue
            return False
        if hb < 0 and rcol < bcol:                         # blue sweeps left through red
            return False
        if hb > 0 and rcol > bcol:                         # blue sweeps right through red
            return False
    # OUTWARD opposite-horizontal (blocks swing APART, arrows point away) is only
    # hittable when the blocks are WIDE apart -- adjacent/near crams both hands into the
    # centre = impossible. Require the outer columns (a full gap of 3, i.e. col 0 & 3).
    outward = (rcol < bcol and hr < 0 and hb > 0) or (bcol < rcol and hb < 0 and hr > 0)
    if outward and abs(rcol - bcol) < 3:
        return False
    return True
