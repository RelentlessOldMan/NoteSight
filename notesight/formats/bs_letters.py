r"""bs_letters.py -- the ALPHABET: a vocabulary of concrete, flowing single/double PATTERNS
("letters") that the recurring parts of a song (phrases) are composed from.
Lexicon: alphabet -> letter -> phrase -> sentence -> song; a phrase's flipped answer is its
MIRROR; a LINK is the connective stream between/after phrases.

A letter is a short motion with GOOD FLOW baked in: internal parity/momentum is clean, and
each letter exposes an ENTRY and EXIT state (which hand swung last and whether it ended on an
up- or down-swing) so the composer can chain letters without a reset. Percentages (double
rate, run lengths, ...) emerge from which letters a song uses -- they are a check, not the goal.

Grid: columns 0..3 (left..right), rows 0 bottom / 1 mid / 2 top. Cuts: 0 UP 1 DN 2 L 3 R
4 UL 5 UR 6 DL 7 DR 8 dot. Hands: 0 red (left), 1 blue (right).

Each letter builds `length` note-slots and returns (slots, exit) where a slot is a list of
(hand, col, row, cut) blocks (1 = single, 2 = double) and exit = {hand, "U"/"D" per hand}.
"""
from __future__ import annotations

from .bs_model import _double_ok
from .bs_flow import flow_ok

RED, BLUE = 0, 1
UP, DN, L, R, UL, UR, DL, DR, DOT = 0, 1, 2, 3, 4, 5, 6, 7, 8
_UPS = {0, 4, 5}
_DOWNS = {1, 6, 7}


def _flip(v):
    return "U" if v == "D" else "D"


def _vc(cut):
    """Vertical class of a cut: U(p) / D(own) / S(ideways or dot)."""
    return "U" if cut in _UPS else "D" if cut in _DOWNS else "S"


# ---- SINGLES letters -------------------------------------------------------------

def _place(h, cut):
    """Where a link note sits: row from the vertical component, column on the hand's own side
    (red left 0/1, blue right 2/3) with leftward cuts to the outer-left / rightward to outer-
    right -- keeps horizontals off-cross-body and diagonals on the natural hand arc."""
    row = 2 if cut in _UPS else 0 if cut in _DOWNS else 1
    if h == RED:
        col = 0 if cut in (L, UL, DL) else 1
    else:
        col = 3 if cut in (R, UR, DR) else 2
    return col, row


def stream(length, hand, par, rng=None, allow_dots=False):
    """The calm connective LINK. Hands trade every note; each hand's next cut is drawn from the
    REAL note->next-note flow mined from good maps (bs_transitions.json) -- so a link flows with
    the pump + each hand's diagonal arc instead of a monotonous up/down/up/down. Falls back to the
    plain up/down pump if no rng or no transition data. Red lives left, blue right."""
    trans = _load_trans()
    p = dict(par)
    if rng is None or not trans:                      # fallback: plain alternating pump
        slots, h = [], hand
        for _ in range(length):
            v = p[h]
            col = 1 if h == RED else 2
            slots.append([(h, col, 0 if v == "D" else 2, DN if v == "D" else UP)])
            p[h] = _flip(v)
            h = BLUE if h == RED else RED
        return slots, {"hand": BLUE if h == RED else RED, "par": p}
    slots, h = [], hand
    lastcut = {RED: UP if p.get(RED, "D") == "D" else DN,
               BLUE: UP if p.get(BLUE, "D") == "D" else DN}
    for _ in range(length):
        nc = _next_cut(h, lastcut[h], rng, trans, allow_dots)
        if nc is None:                                # dead end -> just flip vertical
            nc = DN if _vc(lastcut[h]) != "D" else UP
        col, row = _place(h, nc)
        slots.append([(h, col, row, nc)])
        lastcut[h] = nc
        h = BLUE if h == RED else RED
    p = {hd: _vc(lastcut[hd]) if _vc(lastcut[hd]) in ("U", "D") else p.get(hd, "D")
         for hd in (RED, BLUE)}
    return slots, {"hand": BLUE if h == RED else RED, "par": p}


def circle(length, hand, par, direction=+1):
    """A one-hand CIRCLE (windmill): the fun motion. The hand traces an arc
    down(bottom) -> side -> up(top) -> side, so the saber sweeps a full loop. Same hand for
    the whole run (a same-hand run of `length`). direction +1 = right-going loop, -1 = left."""
    centre = 1 if hand == RED else 2
    # phase order tracing the circle; each entry (cut, dcol, row)
    loop = [(DN, 0, 0), (L if direction > 0 else R, -direction, 1),
            (UP, 0, 2), (R if direction > 0 else L, direction, 1)]
    # start the phase on the swing that matches parity (down-swing enters at phase 0)
    ph = 0 if par[hand] == "D" else 2
    slots = []
    for _ in range(length):
        cut, dcol, row = loop[ph % 4]
        col = min(3, max(0, centre + dcol))
        slots.append([(hand, col, row, cut)])
        ph += 1
    endv = "U" if loop[(ph - 1) % 4][0] == DN else "D" if loop[(ph - 1) % 4][0] == UP \
        else par[hand]
    p = dict(par); p[hand] = endv
    return slots, {"hand": BLUE if hand == RED else RED, "par": p}


# DOUBLES are no longer hand-manufactured here -- they come from the MINED double-run bank
# (bs_doubles_bank.json via _load_doubles_bank), so every double we place is a real flowing run
# from good maps. The old chop_pump_run/stack manufacturers were the source of the "only 6 shapes"
# problem and are gone (see git history).


import json as _json
import os as _os
import random as _random

_BANK = None
_DBANK = None


def _load_bank():
    global _BANK
    if _BANK is None:
        try:
            p = _os.path.join(_os.path.dirname(__file__), "bs_letters_bank.json")
            _BANK = _json.load(open(p, encoding="utf-8"))["letters"]
        except Exception:
            _BANK = []
    return _BANK


def _load_doubles_bank():
    """The MINED double-run bank (bs_doubles_bank.json), grouped by family. Each run is a real,
    flowing sequence of two-hand doubles from good maps -- so a double we place is good by
    construction, not manufactured by the guard. Families: pump/windmill (the streams) +
    outward/stack/parallel/diagonal (accents)."""
    global _DBANK
    if _DBANK is None:
        by = {}
        try:
            p = _os.path.join(_os.path.dirname(__file__), "bs_doubles_bank.json")
            for r in _json.load(open(p, encoding="utf-8"))["runs"]:
                by.setdefault(r["family"], []).append(r["slots"])
        except Exception:
            pass
        _DBANK = by
    return _DBANK


_TRANS = None


def _load_trans():
    """Per-hand note->next-note CUT flow mined from good maps (bs_transitions.json), as
    {hand: {prev_cut: [(next_cut, weight), ...]}} from the TIGHT (<=0.5s) transitions -- the
    data that lets links/seams chain the way real mappers do."""
    global _TRANS
    if _TRANS is None:
        t = {}
        try:
            p = _os.path.join(_os.path.dirname(__file__), "bs_transitions.json")
            raw = _json.load(open(p, encoding="utf-8"))
            for h, name in ((RED, "red"), (BLUE, "blue")):
                d = {}
                for k, n in raw[name]["cut2cut_tight"].items():
                    a, b = k.split("->")
                    d.setdefault(int(a), []).append((int(b), n))
                t[h] = d
        except Exception:
            t = {}
        _TRANS = t
    return _TRANS


def _next_cut(h, prev, rng, trans, allow_dots=False):
    """Weighted-sample this hand's next cut from the real followers of `prev`, dropping dots
    (unless allowed) and any cut that would repeat the same vertical swing (parity). None if
    the previous cut has no usable followers."""
    opts = trans.get(h, {}).get(prev)
    if not opts:
        return None
    prevv = _vc(prev)
    pool = [(c, w) for (c, w) in opts
            if (allow_dots or c != DOT) and not (_vc(c) in ("U", "D") and _vc(c) == prevv)
            and flow_ok(h, prev, c)]
    if not pool:
        return None
    x = rng.random() * sum(w for _, w in pool)
    acc = 0
    for c, w in pool:
        acc += w
        if x <= acc:
            return c
    return pool[-1][0]


def _seam_score(exitc, entryc, trans):
    """How well one letter's ENTRY cut flows from the running EXIT cut, per hand: the summed
    real transition probability (0..2). Used to prefer letters that CHAIN, not just add variety."""
    if not trans:
        return 0.0
    s = 0.0
    for h in (RED, BLUE):
        pv, nx = exitc.get(h), entryc.get(h)
        if pv is None or nx is None:
            continue
        opts = trans.get(h, {}).get(pv, [])
        tot = sum(w for _, w in opts)
        if tot:
            s += dict(opts).get(nx, 0) / tot
    return s


def _edge_cuts(slots, i):
    """The cut per hand at slot index i (0 = a letter's entry, -1 = its exit)."""
    return {h: cut for (h, c, r, cut) in slots[i]}


def _pick_double_families(rng, dbank):
    """A song commits to a few double families with an IDENTITY: one PRIMARY stream family
    (pump-heavy or windmill-heavy) drawn from most, a SECONDARY for contrast, and often one
    ACCENT family (outward/stack/parallel/diagonal) for spice. Returns [(family, n_runs), ...]."""
    core = [f for f in ("pump", "windmill") if dbank.get(f)]
    rng.shuffle(core)
    chosen = []
    if core:
        chosen.append((core[0], rng.randint(6, 9)))              # primary
        if len(core) > 1:
            chosen.append((core[1], rng.randint(2, 4)))          # secondary
    accents = [f for f in ("outward", "parallel", "stack", "diagonal") if dbank.get(f)]
    if accents and rng.random() < 0.7:
        chosen.append((rng.choice(accents), rng.randint(1, 2)))  # spice
    return chosen


def _curated_singles():
    """Hand-crafted SINGLE flow families. NOTE: the one-hand windmill `circle()` is DELIBERATELY
    NOT used -- a full down/left/up/right rotation (with a left-hand right-cut reaching across)
    is not intuitive and too hard, especially on Normal. One-hand variety comes from the mined
    single-runs (real, smooth phrases from good maps) instead."""
    par = {RED: "D", BLUE: "D"}
    return [[list(s) for s in stream(n, RED, par)[0]] for n in (4, 6, 8)]


def _mined_doubles(rng, dbank):
    """The song's DOUBLE letters, drawn from the mined double-run bank for its committed
    families (see _pick_double_families). Real flowing runs from good maps -- no manufacturing."""
    out = []
    for fam, k in _pick_double_families(rng, dbank):
        runs = dbank.get(fam, [])
        if runs:
            out += rng.sample(runs, min(len(runs), k))
    return out


def pick_alphabet(seed, allow_dots=False):
    """A song's committed alphabet: a seeded subset of the MINED single bank (thousands of real
    single-runs) PLUS a seeded set of MINED double RUNS from the song's committed double families
    (pump/windmill streams + an accent). Everything is mined from good maps, so every letter is
    good by construction; the guard is only a last-resort net. `allow_dots` keeps mined singles
    that carry dots (regular songs use some; stamina uses none -- double runs are dot-free)."""
    rng = _random.Random(seed)
    bank = _load_bank()
    def _has_dot(slots):
        return any(b[3] == DOT for s in slots for b in s)
    mined_singles = [L["slots"] for L in bank
                     if L["cat"] == "single" and (allow_dots or not _has_dot(L["slots"]))]
    alpha = []
    if mined_singles:
        alpha += [mined_singles[i] for i in
                  rng.sample(range(len(mined_singles)), min(len(mined_singles), 16))]
    alpha += _mined_doubles(rng, _load_doubles_bank())
    if not any(any(len(s) == 2 for s in L) for L in alpha):      # bank missing -> safe fallback
        alpha += rng.sample(_curated_singles(), min(3, len(_curated_singles())))
    rng.shuffle(alpha)
    return alpha


def _mirror(slots):
    """Left/right mirror a slot list (hand swap, column flip, cut mirror) -- for the 2nd, 4th..
    (the mirror = a phrase's 'answer') so it reads as 'the same figure, other side'."""
    mc = {UP: UP, DN: DN, L: R, R: L, UL: UR, UR: UL, DL: DR, DR: DL, DOT: DOT}
    out = []
    for s in slots:
        out.append([(1 - h if h in (0, 1) else h, 3 - c, r, mc[cut]) for (h, c, r, cut) in s])
    return out


def _vflip(slots):
    """Vertical flip (up<->down cuts + rows top<->bottom, columns & horizontal sweep unchanged) --
    a hittability-preserving symmetry that FLIPS a figure's swing, so a letter can start on an
    up- or down-swing to keep a hand's parity alternating across a seam. Diagonals flip U<->D."""
    vf = {0: 1, 1: 0, 4: 6, 5: 7, 6: 4, 7: 5, 2: 2, 3: 3, 8: 8}
    return [[(h, c, 2 - r, vf[cut]) for (h, c, r, cut) in s] for s in slots]


def compose(n_slots, alphabet, seed, dbl_frac=0.30):
    """Build `n_slots` note-slots AND a parallel per-slot LABEL list (which phrase each slot
    belongs to), so the report can show the whole mapping with no note left out.

    Lexicon: LETTERS spell several DISTINCT phrases; the phrases ROTATE (A B C A B C ...) to give
    a song internal variety *and* recurrence -- not one phrase looped. Each phrase is emitted
    as-is or MIRRORed (its 'answer') to keep red/blue even; a LINK (clean stream) fills the tail
    so a phrase is never chopped. `dbl_frac` biases how often a double letter is chosen."""
    rng = _random.Random(seed)
    trans = _load_trans()
    doubles = [L for L in alphabet if any(len(s) == 2 for s in L)]
    singles = [L for L in alphabet if not any(len(s) == 2 for s in L)] or alphabet

    def tags(chunk):
        """The variety 'features' a letter/phrase exhibits -- so we can tell letters apart and
        avoid a phrase that's all one flavour (e.g. up/down center-column only)."""
        t = set()
        for slot in chunk:
            if len(slot) >= 2:
                t.add("dbl")
            for (h, c, r, cut) in slot:
                t.add("left" if c <= 1 else "right")
                t.add("ctr" if c in (1, 2) else "edge")
                if cut in (0, 4, 5):
                    t.add("up")
                elif cut in (1, 6, 7):
                    t.add("dn")
                if cut in (2, 3):
                    t.add("horiz")
                if cut in (4, 5, 6, 7):
                    t.add("diag")
                t.add("row%d" % r)
        return t

    def _edge(chunk, first):
        """Per-hand (vclass, cut) at the chunk's ENTRY (first=True) or EXIT (first=False)."""
        d = {}
        for slot in (chunk if first else reversed(chunk)):
            for (h, c, r, cut) in slot:
                d.setdefault(h, (_vc(cut), cut))
        return d

    def _join_bad(exv, excut, entry):
        """HARD join clashes of an entry vs the running exit: a hand repeating its vertical swing
        (parity) or repeating a horizontal slice. Heavily penalized so letters CHAIN clean by
        construction -- the assembler only has to bridge whatever still doesn't fit."""
        bad = 0
        for h, (v, cut) in entry.items():
            if v in ("U", "D") and exv.get(h) == v:
                bad += 1
            if cut in (2, 3) and excut.get(h) == cut:
                bad += 1
        return bad

    def _self_bad(chunk):
        """Does the letter (in this orientation) break a hard rule on its own -- a down-cut on the
        top row, a cross-body horizontal, a backhand, a non-hittable/non-L-R same-column double,
        OR an awkward per-hand cut transition (bs_flow)? vflip in particular can turn a clean
        windmill into an off-arc/reset motion, so we flow-check every orientation. Never placeable."""
        lastcut = {0: None, 1: None}
        for slot in chunk:
            for (h, c, r, cut) in slot:
                if cut in _DOWNS and r == 2:
                    return True
                if cut == 2 and c >= 2:
                    return True
                if cut == 3 and c <= 1:
                    return True
                if (h == 0 and cut == 5) or (h == 1 and cut == 4):
                    return True                            # red up-right / blue up-left backhand
                if lastcut[h] is not None and not flow_ok(h, lastcut[h], cut):
                    return True                            # awkward transition (re-cock/off-arc)
                lastcut[h] = cut
            if len(slot) == 2:
                rs = [b for b in slot if b[0] == 0]
                bs = [b for b in slot if b[0] == 1]
                if len(rs) == 1 and len(bs) == 1:
                    (_h, rcol, rrow, rc) = rs[0]
                    (_h, bcol, brow, bc) = bs[0]
                    if not _double_ok(rc, rcol, rrow, bc, bcol, brow):
                        return True
                    if rcol == bcol and rrow != brow and {rc, bc} != {2, 3}:
                        return True
        return False

    def build_phrase(length, exv, excut):
        # Build a phrase that is CLEAN by construction: each letter (in its as-is OR vertical-flip
        # orientation) must be self-valid AND not clash at the join (parity / horizontal repeat)
        # with the running exit. Among the clean candidates, take the one adding the most variety +
        # best transition flow. Only if NOTHING is clean do we fall back to least-bad.
        out, cov, prev = [], set(), None
        while len(out) < length:
            pool = doubles if (doubles and rng.random() < dbl_frac) else singles
            clean, fallback = [], []
            for _ in range(10):
                base = rng.choice(pool)
                for cand in (base, _vflip(base)):
                    if _self_bad(cand):
                        continue
                    tg = tags(cand)
                    ent = _edge(cand, True)
                    entcut = {h: cut for h, (v, cut) in ent.items()}
                    score = (len(tg - cov) - (3 if tg == prev else 0)
                             + 2.0 * _seam_score(excut, entcut, trans))
                    (clean if _join_bad(exv, excut, ent) == 0 else fallback).append((score, cand, tg))
            pick = max(clean or fallback, key=lambda x: x[0])
            _, best, best_tg = pick
            out += [list(s) for s in best]
            cov |= best_tg
            prev = best_tg
            for h, (v, cut) in _edge(best, False).items():
                exv[h], excut[h] = v, cut
        return out[:length]

    # several DISTINCT phrases (re-roll one too similar to a kept phrase) -> in-song variety;
    # each built from a neutral join state so it reads clean on its own.
    n_ph = rng.choice((3, 4))
    phrases = []
    for _ in range(n_ph):
        cand = build_phrase(rng.choice((8, 12, 16)), {}, {})
        for _try in range(4):
            if all(len(tags(cand) ^ tags(p)) >= 3 for p in phrases):
                break
            cand = build_phrase(rng.choice((8, 12, 16)), {}, {})
        phrases.append(cand)
    names = [chr(65 + k) for k in range(n_ph)]

    # Emit phrases in rotation (A B C A B C ...) as UNITS -- the assembler chooses each unit's
    # ORIENTATION (as-is / mirror / vertical-flip) against the real running state so parity stays
    # alternating and red/blue stays even, and a phrase is internally clean so orienting the whole
    # unit never breaks its flow. A clean LINK stream fills the exact tail.
    units, k, filled = [], 0, 0
    while filled < n_slots:
        ph = phrases[k % n_ph]
        if filled + len(ph) > n_slots:
            ph = ph[:n_slots - filled]
        if not ph:
            break
        units.append(([list(s) for s in ph], "Phrase " + names[k % n_ph]))
        filled += len(ph)
        k += 1
    if filled < n_slots:
        link, _ = stream(n_slots - filled, RED, {RED: "D", BLUE: "D"}, rng=rng)
        units.append(([list(s) for s in link][:n_slots - filled], "Link"))
    return units


# ---- demo: render each letter as a slice sequence (to eyeball the flow) ----------
if __name__ == "__main__":
    _CUT = {0: "UP", 1: "DN", 2: "L", 3: "R", 4: "UL", 5: "UR", 6: "DL", 7: "DR", 8: "dot"}

    def _slot(s):
        def one(b):
            return f"{'R' if b[0] == 0 else 'B'}:{_CUT[b[3]]}@{b[1]},{b[2]}"
        return one(s[0]) if len(s) == 1 else "[" + " + ".join(one(b) for b in s) + "]"

    dbank = _load_doubles_bank()
    print("double-run families in bank:", {k: len(v) for k, v in dbank.items()})
    for fam, runs in dbank.items():
        if runs:
            print(f"{fam:9}:  " + "   ".join(_slot(s) for s in runs[0]))
    par0 = {RED: "D", BLUE: "D"}
    print("stream   :  " + "   ".join(_slot(s) for s in stream(6, RED, par0)[0]))
