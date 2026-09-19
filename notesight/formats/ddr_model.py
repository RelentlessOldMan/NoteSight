"""ddr_model.py -- GENERATE 4-panel DDR rows by sampling the learned arrow-pattern
transition model (see ddr_mine.py). Given a stream of note TIMES (our onset/beat
selection decides WHEN), the model decides WHICH panels, so the foot-flow reads
like the hand-authored A20 charts it learned from.

    model_ddr_notes(times, bpm, seed=0, singles_only=False) -> list[(time, lane)]

`singles_only` forbids jumps (for stamina). A light guard caps consecutive jacks.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

_MODEL = None
_SINGLES = ("1000", "0100", "0010", "0001")


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    raw = json.load(open(os.path.join(os.path.dirname(__file__), "ddr_model.json"),
                        encoding="utf-8"))
    edges = raw["gap_edges"]
    trans = {tuple(k.split(",")[:2]) + (int(k.split(",")[2]),): v
             for k, v in raw["trans"].items()}
    bo1 = {(k.rsplit(",", 1)[0], int(k.rsplit(",", 1)[1])): v
           for k, v in raw["bo1"].items()}
    _MODEL = dict(edges=edges, trans=trans, bo1=bo1, bo2=raw["bo2"],
                  start=raw["start"])
    return _MODEL


def _gap_bucket(g, edges):
    for i, e in enumerate(edges):
        if g <= e:
            return i
    return len(edges)


def _sample(weighted, rng, keep=None):
    if keep is not None:
        weighted = [(m, c) for m, c in weighted if m in keep]
    if not weighted:
        return None
    tot = sum(c for _, c in weighted)
    r = rng.random() * tot
    for m, c in weighted:
        r -= c
        if r <= 0:
            return m
    return weighted[-1][0]


def model_ddr_notes(times, bpm, seed=0, singles_only=False):
    m = _load()
    edges = m["edges"]
    rng = random.Random(0xDD ^ (seed * 2654435761) ^ len(times))
    keep = set(_SINGLES) if singles_only else None
    ts = sorted(set(round(t, 5) for t in times))
    out = []
    p2, p1 = "0000", None
    jack_run = 0
    for i, t in enumerate(ts):
        if p1 is None:
            mask = _sample(m["start"], rng, keep) or "1000"
        else:
            gb = _gap_bucket((t - ts[i - 1]) * bpm / 60.0, edges)
            mask = (_sample(m["trans"].get((p2, p1, gb), []), rng, keep)
                    or _sample(m["bo1"].get((p1, gb), []), rng, keep)
                    or _sample(m["bo2"].get(p1, []), rng, keep))
            if mask is None:                       # last resort: a non-jack single
                mask = next(s for s in _SINGLES if s != p1)
            # cap jacks: no more than 2 same-panel repeats in a row
            if mask == p1:
                jack_run += 1
                if jack_run >= 2:
                    alt = _sample(m["bo2"].get(p1, []), rng,
                                  set(s for s in _SINGLES if s != p1))
                    mask = alt or next(s for s in _SINGLES if s != p1)
                    jack_run = 0
            else:
                jack_run = 0
        for lane in range(4):
            if mask[lane] == "1":
                out.append((t, lane))
        p2, p1 = (p1 if p1 else "0000"), mask
    return out
