r"""bs_flow.py -- the per-hand FLOW model: which cut->cut transitions a human hand does
comfortably, learned from real maps (bs_transitions.json). This is the ONE source of truth for
"does this flow" -- used by the letter miners (reject awkward vocabulary), the composer/orientation
picker (never place/vflip into an awkward motion), and `_validate` (fail-loud if one slips through).

A transition is ALLOWED iff its real-map conditional probability P(next | prev) for that hand is
>= FLOOR. Dots are neutral connectors (always allowed). The awkward tail this cuts out is the
re-cock (up-diagonal/vertical -> same-side flat horizontal), the off-arc windmill reversal
(on-arc diag <-> off-arc diag), and flat->up resets -- motions real mappers essentially never write.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

FLOOR = 0.02                                    # >=2% of real-map follows = comfortable
_TRANS = None


def _load():
    global _TRANS
    if _TRANS is None:
        t = {}
        try:
            p = os.path.join(os.path.dirname(__file__), "bs_transitions.json")
            raw = json.load(open(p, encoding="utf-8"))
            for h, nm in ((0, "red"), (1, "blue")):
                d = defaultdict(dict)
                tot = defaultdict(int)
                for k, n in raw[nm]["cut2cut_tight"].items():
                    a, b = map(int, k.split("->"))
                    d[a][b] = n
                    tot[a] += n
                t[h] = {a: {b: c / tot[a] for b, c in bs.items()} for a, bs in d.items()}
        except Exception:
            t = {}
        _TRANS = t
    return _TRANS


def prob(h, a, b):
    return _load().get(h, {}).get(a, {}).get(b, 0.0)


def flow_ok(h, a, b, floor=FLOOR):
    """Is the transition cut a -> cut b comfortable for hand h? Dots neutral; no data -> allow."""
    if a == 8 or b == 8:
        return True
    if not _load():
        return True
    return prob(h, a, b) >= floor


def best_follow(h, prev):
    """The most comfortable directional cut to play after `prev` for hand h (the pump/windmill
    continuation) -- used to resolve a flagged transition TOWARD flow, not away from it."""
    m = _load().get(h, {}).get(prev, {})
    cand = [(b, p) for b, p in m.items() if b != 8 and p >= FLOOR]
    return max(cand, key=lambda x: x[1])[0] if cand else None
