r"""bs_assemble.py -- assemble a Beat Saber chart from the LETTER VOCABULARY (the one generator).

`assemble_letters(times, bpm, ...)` builds the chart CORRECT BY CONSTRUCTION: compose the song's
alphabet into phrase units, orient each against the running state, place with BENIGN resolution,
then `_validate` (a fail-loud assertion, not a mutator). `guard_seams` cleans STITCHED/TILED charts
(mashup, stamina tiling) the same way. The old mutating `_reguard` is gone (see git history).

`bs_model` is kept only for the hittability vocabulary (`_double_ok`) and the cut constants.
"""
from __future__ import annotations

from collections import defaultdict

from .bs_model import _UP, _DOWN, _VFLIP, _double_ok
from .bs_letters import compose, pick_alphabet, _mirror, _vflip
from .bs_flow import flow_ok, best_follow


def _vclass(cut):
    return "U" if cut in _UP else "D" if cut in _DOWN else "S"


# --- same-hand travel-time floor (v2026.9.9.x). Real maps NEVER place a same-hand MOVE
# closer in time than this, and it SCALES with density (a fast song tolerates ~70ms same-hand
# moves; a 3-NPS song basically never goes below ~170ms). These are the p1 (tightest-ever) of
# same-hand-moved gaps mined from the good-map corpus, bucketed by the song's average NPS.
# We enforce it one layer BELOW the letters -- on the ONSET RHYTHM -- so no phrase / single-hand
# run / double is ever *placed* too tight anywhere (zero pattern-reuse cost, 100% vocabulary
# intact); the rare (~1-2%) sub-floor onset pair is NUDGED apart, never dropped. This is why the
# too-close spots simply never get created ("get it right up front").
_ONSET_FLOOR = [           # (song avg NPS <= X) -> min seconds between consecutive onsets
    (3.0, 0.188), (4.0, 0.167), (5.0, 0.140),
    (6.0, 0.109), (7.0, 0.094), (9.0, 0.078),
]
_ONSET_FLOOR_FAST = 0.068  # 9+ NPS


def onset_floor(nps):
    """Min comfortable same-hand-move gap (seconds) for a chart of this average NPS."""
    for cap, f in _ONSET_FLOOR:
        if nps <= cap:
            return f
    return _ONSET_FLOOR_FAST


def space_onsets(times, floor):
    """NUDGE-APART: return `times` (seconds) sorted with no two consecutive closer than `floor`, by
    pushing the later onset forward just enough. Keeps EVERY note (never drops) and preserves order;
    a few ms of give is inaudible while the unhittable same-hand jump is gone. Onsets already >=
    floor apart are untouched -- only the rare sub-floor burst moves. Applied BEFORE placement so
    the placer sees the final timing (its leap/occlude rules are timing-dependent)."""
    out, prev = [], None
    for t in sorted(times):
        if prev is not None and t - prev < floor:
            t = prev + floor
        out.append(t)
        prev = t
    return out


def guard_seams(blocks, bpm, peak_cap=None):
    """Clean a STITCHED / TILED chart the correct-by-construction way (this REPLACES the old
    mutating `_reguard`, now deleted): walk its time-slots and place each via BENIGN resolution
    against the running per-hand state -- flip a swing, pull a note onside, or drop a clashing
    double to a clean single -- then `_validate`. The per-song loops are already clean, so this
    only smooths the song-to-song / loop-wrap JOINS; it never re-authors a figure into an awkward
    shape (no double-rebuild, no cross-body). Used by the mashup/stamina stitch."""
    by = defaultdict(list)
    for b in blocks:
        by[b["_time"]].append(b)
    lastv = {0: None, 1: None}
    lastcut = {0: None, 1: None}
    lastc = {0: None, 1: None}
    lastt = {0: -9.0, 1: -9.0}
    recent, bsec = [], []
    out = []
    for tb in sorted(by):
        t = tb * 60.0 / bpm
        while bsec and t - bsec[0] > 1.0:
            bsec.pop(0)
        while recent and t - recent[0][0] > 0.3:
            recent.pop(0)
        slot = [(b["_type"], b["_lineIndex"], b["_lineLayer"], b["_cutDirection"]) for b in by[tb]]
        slot, _bridged = _resolve(slot, t, lastv, lastcut, lastc, lastt, recent)
        if peak_cap and len(bsec) + len(slot) > peak_cap:
            continue
        _apply(slot, t, lastv, lastcut, lastc, lastt, recent)
        bsec.extend([t] * len(slot))
        for (h, c, r, cut) in slot:
            out.append({"_time": tb, "_lineIndex": int(c), "_lineLayer": int(r),
                        "_type": int(h), "_cutDirection": int(cut)})
    _validate(out, bpm)
    return out


_UD = ("U", "D")


def _violation(slot, t, lastv, lastcut, lastc, lastt, recent):
    """THE single source of truth for whether placing `slot` at time t is hard-rule-clean given
    the running per-hand state. Returns a reason string, or None if clean. Used by BOTH the
    state-aware placer (accept/reject) and the validator (assert) -- so a slot is either placed
    because it's already right, or not placed. Nothing is ever silently nudged into shape.
    A slot is a list of (hand, col, row, cut) tuples (1 = single, 2 = a red+blue double)."""
    reds = [b for b in slot if b[0] == 0]
    blues = [b for b in slot if b[0] == 1]
    if len(reds) == 1 and len(blues) == 1:                 # DOUBLE
        (_h, rcol, rrow, rc) = reds[0]
        (_h, bcol, brow, bc) = blues[0]
        if not _double_ok(rc, rcol, rrow, bc, bcol, brow):
            return "dbl-unhittable"
        if rc == 5 or bc == 4:
            return "dbl-backhand"                           # red up-right / blue up-left = across-body
        if rcol == bcol and rrow != brow and {rc, bc} != {2, 3}:
            return "dbl-samecol"                            # a same-column stack must be L/R
        if (rc in _DOWN and rrow == 2) or (bc in _DOWN and brow == 2):
            return "dbl-downtop"
        if (rc in (2, 3) and lastcut[0] == rc) or (bc in (2, 3) and lastcut[1] == bc):
            return "dbl-hrepeat"
        if lastv[0] is not None and _vclass(rc) in _UD and _vclass(rc) == lastv[0]:
            return "dbl-parity-red"
        if lastv[1] is not None and _vclass(bc) in _UD and _vclass(bc) == lastv[1]:
            return "dbl-parity-blue"
        if lastcut[0] is not None and not flow_ok(0, lastcut[0], rc):
            return "dbl-flow-red"
        if lastcut[1] is not None and not flow_ok(1, lastcut[1], bc):
            return "dbl-flow-blue"
        if lastc[0] is not None and abs(rcol - lastc[0]) >= 3 and t - lastt[0] < 0.34:
            return "dbl-leap-red"
        if lastc[1] is not None and abs(bcol - lastc[1]) >= 3 and t - lastt[1] < 0.34:
            return "dbl-leap-blue"
        return None
    for (h, c, r, cut) in slot:                            # single
        if lastv[h] is not None and _vclass(cut) in _UD and _vclass(cut) == lastv[h]:
            return "parity"                                 # same vertical swing twice in a row
        if lastcut[h] == cut and cut in (2, 3):
            return "hrepeat"                                # same horizontal slice twice
        if lastcut[h] is not None and not flow_ok(h, lastcut[h], cut):
            return "flow"                                   # awkward transition (re-cock/off-arc)
        if cut in _DOWN and r == 2:
            return "downtop"                                # down-cut on the top row
        if cut == 2 and c >= 2:
            return "xbody-L"                                # left slice on the right half
        if cut == 3 and c <= 1:
            return "xbody-R"
        if (h == 0 and cut == 5) or (h == 1 and cut == 4):
            return "backhand"                               # red up-right / blue up-left = across-body
        if lastc[h] is not None and abs(c - lastc[h]) >= 3 and t - lastt[h] < 0.34:
            return "leap"                                   # col0<->col3 fling in a short gap
        if any(hh != h and cc == c and rq == r and 0 < t - tt <= 0.25
               for (tt, cc, rq, hh) in recent):
            return "occlude"                                # other hand just sat on this cell
    return None


def _apply(slot, t, lastv, lastcut, lastc, lastt, recent):
    """Advance per-hand state for a committed slot (a DOT consumes a swing -- carries the implied
    opposite direction forward)."""
    for (h, c, r, cut) in slot:
        if cut == 8 and lastv[h] in _UD:
            lastv[h] = "D" if lastv[h] == "U" else "U"
        else:
            lastv[h] = _vclass(cut)
        lastcut[h] = cut
        lastc[h] = c
        lastt[h] = t
        recent.append((t, c, r, h))


def _bridge(t, lastv, lastcut, lastc, lastt, recent):
    """A guaranteed-valid connective single (the dead-end escape when no composed slot fits). It
    aims for the FLOW continuation -- the most comfortable cut after each hand's last (pump/
    windmill via best_follow) -- then plain vertical fallbacks, trying each hand's own columns,
    and returns the first that passes every rule (incl. flow). So even the escape hatch flows."""
    for h in (0, 1):
        prev = lastcut[h]
        cuts = []
        if prev is not None and best_follow(h, prev) is not None:
            cuts.append(best_follow(h, prev))
        cuts += [1 if lastv[h] == "U" else 0, 0 if lastv[h] == "U" else 1]
        cols = (0, 1) if h == 0 else (3, 2)
        for cut in cuts:
            row = 2 if cut in _UP else 0 if cut in _DOWN else 1
            for col in cols:
                slot = [(h, col, row, cut)]
                if _violation(slot, t, lastv, lastcut, lastc, lastt, recent) is None:
                    return slot
    return [(0, 1, 0, 1)]                                   # practically unreachable


def _resolve_single(blk, t, lastv, lastcut, lastc, lastt, recent):
    """BENIGN single-note resolution (what every mapper does, not awkward authoring): flip the
    swing to keep parity, take a repeated horizontal vertical, drop a down-cut off the top row,
    pull a cross-body / full-width note back onside. Returns a clean block, or None if even that
    can't make it fit."""
    h, c, r, cut = blk
    if h == 0 and cut == 5:              # red up-right backhand -> red up-left (on-arc)
        cut = 4
    elif h == 1 and cut == 4:            # blue up-left backhand -> blue up-right (on-arc)
        cut = 5
    if lastv[h] in _UD and _vclass(cut) in _UD and _vclass(cut) == lastv[h]:
        cut = 1 if lastv[h] == "U" else 0
    if lastcut[h] == cut and cut in (2, 3):
        cut = 0 if lastv[h] == "D" else 1
    if lastcut[h] is not None and not flow_ok(h, lastcut[h], cut):
        bf = best_follow(h, lastcut[h])                 # snap TOWARD flow (pump/windmill), not away
        if bf is not None:
            cut = bf
            r = 2 if cut in _UP else 0 if cut in _DOWN else 1
    if cut in _DOWN and r == 2:
        r = 0
    if cut == 2 and c >= 2:
        c = 1
    elif cut == 3 and c <= 1:
        c = 2
    if lastc[h] is not None and abs(c - lastc[h]) >= 3 and t - lastt[h] < 0.34:
        c += 1 if c < lastc[h] else -1
    blk = (h, c, r, cut)
    return blk if _violation([blk], t, lastv, lastcut, lastc, lastt, recent) is None else None


def _resolve(slot, t, lastv, lastcut, lastc, lastt, recent):
    """Make `slot` placeable WITHOUT awkward authoring. Clean already -> keep it. A DOUBLE that
    clashes -> try its flips/mirror; if none fit, DROP to whichever single hand resolves cleanly
    (never rebuild the pair into a cross-body shape). A single -> benign resolve. Last resort: a
    guaranteed-valid bridge note. Returns (slot, changed_to_link?)."""
    if _violation(slot, t, lastv, lastcut, lastc, lastt, recent) is None:
        return slot, False
    reds = [b for b in slot if b[0] == 0]
    blues = [b for b in slot if b[0] == 1]
    if len(reds) == 1 and len(blues) == 1:
        for cand in (_vflip([slot])[0], _mirror([slot])[0], _vflip(_mirror([slot]))[0]):
            cand = [tuple(b) for b in cand]
            if _violation(cand, t, lastv, lastcut, lastc, lastt, recent) is None:
                return cand, False
        for keep in (reds[0], blues[0]):
            s = _resolve_single(keep, t, lastv, lastcut, lastc, lastt, recent)
            if s is not None:
                return [s], False
    else:
        s = _resolve_single(slot[0], t, lastv, lastcut, lastc, lastt, recent)
        if s is not None:
            return [s], False
    return _bridge(t, lastv, lastcut, lastc, lastt, recent), True


def _validate(blocks, bpm):
    """Fail-loud SAFETY NET (not a mutator): replay the per-hand state over the finished chart and
    raise if any time-slot is dirty -- so a composition bug surfaces instead of being patched."""
    by = defaultdict(list)
    for b in blocks:
        by[b["_time"]].append(b)
    lastv = {0: None, 1: None}
    lastcut = {0: None, 1: None}
    lastc = {0: None, 1: None}
    lastt = {0: -9.0, 1: -9.0}
    recent = []
    for tb in sorted(by):
        t = tb * 60.0 / bpm
        while recent and t - recent[0][0] > 0.3:
            recent.pop(0)
        slot = [(b["_type"], b["_lineIndex"], b["_lineLayer"], b["_cutDirection"]) for b in by[tb]]
        why = _violation(slot, t, lastv, lastcut, lastc, lastt, recent)
        if why is not None:
            raise AssertionError("chart violates %s at beat %.4f: %s" % (why, tb, slot))
        _apply(slot, t, lastv, lastcut, lastc, lastt, recent)
    return True


def _sim_violations(slots, ti, tkeys, bpm, lastv, lastcut, lastc, lastt, recent):
    """Count hard-rule violations if `slots` were placed starting at tkeys[ti], against a CLONE of
    the running state -- used to pick a phrase's best ORIENTATION before committing to it."""
    lv, lct, lc, lt, rec = dict(lastv), dict(lastcut), dict(lastc), dict(lastt), list(recent)
    n = 0
    for j, s in enumerate(slots):
        if ti + j >= len(tkeys):
            break
        t = tkeys[ti + j] * 60.0 / bpm
        while rec and t - rec[0][0] > 0.3:
            rec.pop(0)
        slot = [tuple(b) for b in s]
        if _violation(slot, t, lv, lct, lc, lt, rec) is not None:
            n += 1
        else:
            _apply(slot, t, lv, lct, lc, lt, rec)
    return n


def _place(tkeys, units, bpm, peak_cap):
    """Place composed phrase `units` onto the beat grid `tkeys`, choosing each unit's ORIENTATION
    against the running per-hand state and BENIGN-resolving residual seams. Returns (blocks,
    time2label). Pure of onset spacing -- callable twice (before/after the nudge) since the timing
    lives entirely in tkeys."""
    lastv = {0: None, 1: None}
    lastcut = {0: None, 1: None}
    lastc = {0: None, 1: None}
    lastt = {0: -9.0, 1: -9.0}
    recent, bsec = [], []
    out, time2label, bal, ti = [], {}, [0], 0
    for slots, label in units:
        # choose the orientation with the fewest rule clashes, then the one that best evens R/B
        forms = [slots, _mirror(slots), _vflip(slots), _vflip(_mirror(slots))]
        best, best_key = slots, None
        for fs in forms:
            v = _sim_violations(fs, ti, tkeys, bpm, lastv, lastcut, lastc, lastt, recent)
            r = sum(1 for s in fs for b in s if b[0] == 0)
            bl = sum(1 for s in fs for b in s if b[0] == 1)
            key = (v, abs(bal[0] + (r - bl)))
            if best_key is None or key < best_key:
                best, best_key, best_rb = fs, key, (r, bl)
        bal[0] += best_rb[0] - best_rb[1]
        for s in best:
            if ti >= len(tkeys):
                break
            tb = tkeys[ti]
            t = tb * 60.0 / bpm
            while bsec and t - bsec[0] > 1.0:
                bsec.pop(0)
            while recent and t - recent[0][0] > 0.3:
                recent.pop(0)
            slot = [tuple(b) for b in s]
            slot, bridged = _resolve(slot, t, lastv, lastcut, lastc, lastt, recent)
            lbl = "Link" if bridged else label
            if peak_cap and len(bsec) + len(slot) > peak_cap:         # density thinning
                ti += 1
                continue
            _apply(slot, t, lastv, lastcut, lastc, lastt, recent)
            bsec.extend([t] * len(slot))
            time2label[tb] = lbl
            for (h, c, r, cut) in slot:
                out.append({"_time": tb, "_lineIndex": int(c), "_lineLayer": int(r),
                            "_type": int(h), "_cutDirection": int(cut)})
            ti += 1
    return out, time2label


def assemble_letters(times, bpm, seed=0, peak_cap=None, alphabet=None, space=True):
    """LETTER-VOCABULARY assembler -- correct BY CONSTRUCTION. Compose the song's alphabet into
    PHRASE units, then for each unit pick the ORIENTATION (as-is / mirror / vertical-flip / both)
    that is hard-rule-clean against the real running per-hand state (so parity stays alternating
    and red/blue stays even) and place it verbatim. Only a note that STILL doesn't fit is replaced
    by a guaranteed-valid BRIDGE link note -- never mutated into an awkward shape.

    ONSET FLOOR: two-pass. Place once, measure the chart's REALIZED nps, then NUDGE the onsets apart
    to that nps's same-hand travel floor (see onset_floor) and re-place onto the corrected grid --
    so no letter is ever placed too tight (0 same-hand moves below floor), with zero phrase/letter
    cost. The nudge is BEFORE placement because the placer's leap/occlude rules are timing-dependent;
    `compose` is done once since nudging never changes the slot COUNT. `_validate` then asserts the
    whole chart is clean. (Mashup/stamina STITCH seams are cleaned by `guard_seams`.)

    `space=False` disables the onset floor -- used ONLY by the GAUNTLET, whose whole point is an
    ever-denser brutal ramp (peak_cap=None); flooring it would cap the ramp and, across its many
    sub-floor onsets, accumulate forward-push drift that desyncs the audio."""
    if not times:
        return [], {}
    alpha = alphabet if alphabet is not None else pick_alphabet(seed)
    tkeys = sorted({round(t * bpm / 60.0, 5) for t in times})
    units = compose(len(tkeys), alpha, seed)
    out, time2label = _place(tkeys, units, bpm, peak_cap)
    if not space:
        _validate(out, bpm)
        return out, time2label
    # realized density -> same-hand travel floor -> nudge onsets apart -> re-place on the fixed grid.
    # Iterate to a fixed point: nudging grows the span, which can lower the realized nps across a
    # floor bucket boundary; re-nudge from the ORIGINAL onsets at the new floor until the grid stops
    # changing (2-3 rounds). Converges because a larger floor only ever lowers nps -> same/larger floor.
    for _ in range(4):
        tbs = sorted({b["_time"] for b in out})
        if len(tbs) < 2:
            break
        span = (tbs[-1] - tbs[0]) * 60.0 / bpm
        if span <= 0:
            break
        spaced = space_onsets(times, onset_floor(len(out) / span))
        tkeys2 = sorted({round(t * bpm / 60.0, 5) for t in spaced})
        if tkeys2 == tkeys:
            break
        tkeys = tkeys2
        out, time2label = _place(tkeys, units, bpm, peak_cap)
    _validate(out, bpm)
    return out, time2label
