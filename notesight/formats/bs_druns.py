r"""bs_druns.py -- data-driven DOUBLE-RUN model (mined from the real-map library).

Holds what maps actually DO with consecutive doubles: the run-length distribution
(~48% solo, then 2/3/4 tapering) plus a canonical shape->shape transition table. This
replaces the hand-authored rock double-sequences -- a "rock" is now a REAL run sampled
from the library (dominant motion turned out to be vertical PUMPS, not L/R wipers).

    sample_run(rng, max_steps) -> [("D", rc,rcol,rrow, bc,bcol,brow), ...]

Shapes are mirror-canonical (as stored); mirror_rock/_vflip in the assembler adapt them
for recurrence and parity. Every shape is drawn from the hittable double vocabulary, and
every transition is one real maps use, so runs are hittable + flow-valid by construction.
"""
from __future__ import annotations

import json
import os

_M = None


def _load():
    global _M
    if _M is None:
        p = os.path.join(os.path.dirname(__file__), "bs_druns.json")
        _M = json.load(open(p, encoding="utf-8"))
        _M["_lens"] = [int(k) for k in _M["runlen"]]
        _M["_lw"] = [_M["runlen"][k] for k in _M["runlen"]]
    return _M


def _parse(k):
    return tuple(int(x) for x in k.split(","))


def _wsample(rng, d):
    tot = sum(d.values())
    r = rng.random() * tot
    for k, c in d.items():
        r -= c
        if r <= 0:
            return k
    return next(iter(d))


def sample_run(rng, max_steps):
    """Sample one double-run: a real length (clamped to max_steps) walked through the
    real start + transition tables. Returns a list of ("D", ...) steps."""
    M = _load()
    length = rng.choices(M["_lens"], weights=M["_lw"])[0]
    length = max(1, min(length, max_steps))
    cur = _wsample(rng, M["start"])
    steps = [("D",) + _parse(cur)]
    for _ in range(length - 1):
        tr = M["trans"].get(cur)
        if not tr:
            break
        cur = _wsample(rng, tr)
        steps.append(("D",) + _parse(cur))
    return steps
