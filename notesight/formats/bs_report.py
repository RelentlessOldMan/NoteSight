r"""bs_report.py -- write a human-readable GENERATION REPORT for a Beat Saber chart.

Post-hoc analysis (does NOT touch generation): given the note skeleton, the final blocks,
and the section maps the assembler used, it labels the structure and writes
`_generation.txt` in the song folder. Lexicon: alphabet -> letter -> phrase -> sentence ->
song; a phrase's flipped answer is its MIRROR; a LINK is the connective between phrases.

  * a DEFINITION TREE  -- each recurring Phrase A/B.. and Letter A/B.. + its slice sequence,
  * a TIMELINE         -- the order of assembly (Phrase A / Letter A / Link / Phrase B ...),
                          one timestamped line each, with the slice details.

The bigger recurring unit (8-beat) is a PHRASE; a non-recurring stretch is split into its
4-beat parts, each a recurring LETTER or a plain LINK (connective).
"""
from __future__ import annotations

import bisect
import os
from collections import Counter, defaultdict

_CUTN = {0: "UP", 1: "DN", 2: "L", 3: "R", 4: "UL", 5: "UR", 6: "DL", 7: "DR", 8: "dot"}


def _alpha(n):
    s, n = "", n + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _slice(g):
    def one(b):
        return f"{'R' if b['_type'] == 0 else 'B'}:{_CUTN.get(b['_cutDirection'], '?')}" \
               f"@{b['_lineIndex']},{b['_lineLayer']}"
    if len(g) == 1:
        return one(g[0])
    return "[" + " + ".join(one(b) for b in sorted(g, key=lambda x: x["_type"])) + "]"


def _seg_slices(times, at, bpm, start, length):
    out = []
    for j in range(start, min(start + length, len(times))):
        g = at.get(round(times[j] * bpm / 60.0, 5))
        if g:
            out.append(_slice(g))
    return out


def _mmss(t):
    return f"{int(t // 60)}:{t % 60:06.3f}"


_UPF = {0, 4, 5}
_DNF = {1, 6, 7}
_HDIRF = {2: -1, 4: -1, 6: -1, 3: 1, 5: 1, 7: 1, 0: 0, 1: 0, 8: 0}


def counts_lines(blocks, bpm=None):
    """A per-difficulty COUNTS block: total, red/blue, singles/doubles, avg/peak NPS,
    a SUMMARY of double shapes (not every permutation), the longest one-hand run, and
    every cut direction + dots broken out total / red / blue."""
    total = len(blocks)
    red = sum(1 for b in blocks if b["_type"] == 0)
    blue = total - red
    at = defaultdict(list)
    for b in blocks:
        at[b["_time"]].append(b)
    single = sum(len(g) for g in at.values() if len(g) == 1)
    dbl = sum(len(g) for g in at.values() if len(g) >= 2)
    pairs = sum(1 for g in at.values() if len(g) == 2)

    def _vc(c):
        return "U" if c in _UPF else "D" if c in _DNF else "S"

    # DOUBLE-SHAPE SUMMARY (coarse buckets, not per-permutation)
    dtypes = Counter()
    for g in at.values():
        rs = [n for n in g if n["_type"] == 0]
        bs = [n for n in g if n["_type"] == 1]
        if len(rs) == 1 and len(bs) == 1:
            r, bl = rs[0], bs[0]
            rc, bc = r["_cutDirection"], bl["_cutDirection"]
            if rc == 8 or bc == 8:
                k = "dot"
            elif r["_lineIndex"] == bl["_lineIndex"]:
                k = "stack(same col)"
            elif r["_lineLayer"] == bl["_lineLayer"] and (_HDIRF[rc] or _HDIRF[bc]):
                k = "chop(same row)"
            elif _vc(rc) == "U" and _vc(bc) == "U":
                k = "pump-up"
            elif _vc(rc) == "D" and _vc(bc) == "D":
                k = "pump-down"
            else:
                k = "spread/other"
            dtypes[k] += 1
        elif len(g) >= 2:
            dtypes["same-colour/3+"] += 1

    # SINGLE-note runs per hand: consecutive same-hand single-notes (doubles break the run)
    seq = [g[0]["_type"] for _, g in sorted(at.items()) if len(g) == 1]
    red_runs, blue_runs, i = Counter(), Counter(), 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        (red_runs if seq[i] == 0 else blue_runs)[j - i + 1] += 1
        i = j + 1
    # DOUBLE runs: consecutive double snaps in a row (length 1 = a solo double)
    marks = [len(at[t]) >= 2 for t in sorted(at)]
    dbl_runs, i = Counter(), 0
    while i < len(marks):
        if marks[i]:
            j = i
            while j + 1 < len(marks) and marks[j + 1]:
                j += 1
            dbl_runs[j - i + 1] += 1
            i = j + 1
        else:
            i += 1

    # cross-tab: per cut direction -> total / red / blue / per-row / per-col
    cross = {d: {"t": 0, "r": 0, "b": 0, "row": Counter(), "col": Counter()} for d in range(9)}
    for b in blocks:
        cd = cross[b["_cutDirection"]]
        cd["t"] += 1
        cd["r" if b["_type"] == 0 else "b"] += 1
        cd["row"][b["_lineLayer"]] += 1
        cd["col"][b["_lineIndex"]] += 1

    out = ["  -- counts --",
           f"    total {total}    red {red}    blue {blue}",
           f"    singles {single}    doubles {dbl} ({pairs} pairs)"]
    if bpm and total > 1:
        secs = sorted(b["_time"] * 60.0 / bpm for b in blocks)
        span = secs[-1] - secs[0]
        avg = total / span if span > 0 else 0.0
        peak = max(bisect.bisect_left(secs, s + 1.0) - i for i, s in enumerate(secs))
        out.append(f"    avg nps {avg:.2f}    peak nps {peak}  (busiest 1s)")
    if dtypes:
        out.append("    double types:  " + "   ".join(f"{k} {v}" for k, v in dtypes.most_common()))
    maxr = max([1] + list(red_runs) + list(blue_runs))
    out.append("    one-hand run lengths (count of runs):")
    out.append("      len " + "".join(f"{k:>6}" for k in range(1, maxr + 1)))
    out.append("      red " + "".join(f"{red_runs[k]:>6}" for k in range(1, maxr + 1)))
    out.append("      blue" + "".join(f"{blue_runs[k]:>6}" for k in range(1, maxr + 1)))
    if dbl_runs:
        maxd = max(dbl_runs)
        out.append("    double-run lengths (count of runs; 1 = a solo double):")
        out.append("      len " + "".join(f"{k:>6}" for k in range(1, maxd + 1)))
        out.append("      cnt " + "".join(f"{dbl_runs[k]:>6}" for k in range(1, maxd + 1)))
    cols = ["total", "red", "blue", "row0", "row1", "row2", "col0", "col1", "col2", "col3"]
    out.append("    " + f"{'dir':<4}" + "".join(f"{h:>7}" for h in cols))
    tot = dict.fromkeys(cols, 0)
    for d in (0, 1, 2, 4, 6, 3, 5, 7, 8):        # UP DN | L UL DL | R UR DR | dot
        cd = cross[d]
        vals = [cd["t"], cd["r"], cd["b"], cd["row"][0], cd["row"][1], cd["row"][2],
                cd["col"][0], cd["col"][1], cd["col"][2], cd["col"][3]]
        for k, v in zip(cols, vals):
            tot[k] += v
        out.append("    " + f"{_CUTN[d]:<4}" + "".join(f"{v:>7}" for v in vals))
    out.append("    " + f"{'TOT':<4}" + "".join(f"{tot[k]:>7}" for k in cols))
    return out


def chart_report(diff, bpm, times, blocks, rock_sections, pebble_sections):
    at = defaultdict(list)
    for b in blocks:
        at[b["_time"]].append(b)
    rcnt = Counter(sid for (_, _, sid) in rock_sections)
    pcnt = Counter(sid for (_, _, sid) in pebble_sections)
    rlet, plet = {}, {}
    for (_, _, sid) in rock_sections:
        if rcnt[sid] >= 2 and sid not in rlet:
            rlet[sid] = "Phrase " + _alpha(len(rlet))
    for (_, _, sid) in pebble_sections:
        if pcnt[sid] >= 2 and sid not in plet:
            plet[sid] = "Letter " + _alpha(len(plet))
    # global instance index per sid (for the mirror flag: 2nd/4th.. hit is mirrored)
    pinst_idx, seen = {}, Counter()
    for k, (start, length, sid) in enumerate(sorted(pebble_sections)):
        seen[sid] += 1
        pinst_idx[start] = seen[sid]

    # definitions (first instance's slices)
    defs, done = [], set()
    for (start, length, sid) in rock_sections:
        if sid in rlet and sid not in done:
            done.add(sid)
            defs.append((rlet[sid], rcnt[sid], length, _seg_slices(times, at, bpm, start, length)))
    for (start, length, sid) in pebble_sections:
        if sid in plet and sid not in done:
            done.add(sid)
            defs.append((plet[sid], pcnt[sid], length, _seg_slices(times, at, bpm, start, length)))

    # timeline
    peb = {start: (length, sid) for (start, length, sid) in pebble_sections}
    peb_starts = sorted(peb)
    rinst = Counter()
    lines = []
    if not rock_sections:
        # LETTER path (no phrase tier): walk the letters directly, in order, so you can see
        # the reuse -- each recurring cell shows as "Letter X", the rest as "Link".
        pinst = Counter()
        for (start, length, sid) in sorted(pebble_sections):
            pt = times[start] if start < len(times) else 0.0
            pinst[sid] += 1
            if sid in plet:
                mir = " (mirrored)" if pinst[sid] % 2 == 0 else ""
                lines.append((pt, plet[sid] + mir, _seg_slices(times, at, bpm, start, length)))
            else:
                lines.append((pt, "Link", _seg_slices(times, at, bpm, start, length)))
    for (start, length, sid) in sorted(rock_sections):
        t0 = times[start] if start < len(times) else 0.0
        rinst[sid] += 1
        if sid in rlet:
            mir = " (mirrored)" if rinst[sid] % 2 == 0 else ""
            lines.append((t0, rlet[sid] + mir, _seg_slices(times, at, bpm, start, length)))
        else:
            i = bisect.bisect_left(peb_starts, start)
            while i < len(peb_starts) and peb_starts[i] < start + length:
                ps = peb_starts[i]
                pl, psid = peb[ps]
                pt = times[ps] if ps < len(times) else 0.0
                if psid in plet:
                    mir = " (mirrored)" if pinst_idx.get(ps, 1) % 2 == 0 else ""
                    lines.append((pt, plet[psid] + mir, _seg_slices(times, at, bpm, ps, pl)))
                else:
                    lines.append((pt, "Link", _seg_slices(times, at, bpm, ps, pl)))
                i += 1

    out = [f"  [{diff}]  bpm {bpm:.1f}  {len(blocks)} blocks"]
    out += counts_lines(blocks)
    out.append("  -- definitions --")
    for (label, c, length, sl) in defs:
        out.append(f"    {label:11} (x{c}, {length} notes):  " + "  ".join(sl))
    if not defs:
        out.append("    (no recurring phrases or letters)")
    out.append("  -- timeline (phrases / letters / links, in order) --")
    for (t, label, sl) in lines:
        out.append(f"    {_mmss(t):>10}  {label:18}  " + "  ".join(sl))
    return "\n".join(out)


def write_reports(song_dir, title, game_version, charts):
    """charts: list of (diff, bpm, times, blocks, rock_sections, pebble_sections)."""
    from ..version import build_timestamp
    lines = ["NoteSight generation report",
             f"generator: Beat Saber  {game_version}",
             f"generated: {build_timestamp()}",
             f"song: {title}", ""]
    for (diff, bpm, times, blocks, rocks, pebbles) in charts:
        lines.append(chart_report(diff, bpm, times, blocks, rocks, pebbles))
        lines.append("")
    with open(os.path.join(song_dir, "_generation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def letter_timeline(blocks, bpm, time2label):
    """The WHOLE-SONG timeline for a letter-composed chart: EVERY note in order, grouped under
    the phrase it belongs to (a `-- Phrase B --` header when the phrase changes, ' = mirrored).
    No note is left out -- every note-time in `blocks` is listed with its slice."""
    at = defaultdict(list)
    for b in blocks:
        at[b["_time"]].append(b)
    out, cur = [], None
    for tb in sorted(at):
        lbl = time2label.get(tb, "?")
        if lbl != cur:
            out.append(f"    -- {lbl} --")
            cur = lbl
        out.append(f"      {_mmss(tb * 60.0 / bpm):>10}  " + _slice(sorted(at[tb], key=lambda x: x["_type"])))
    return out


def write_reports_letters(song_dir, title, game_version, charts):
    """Letter-path report. charts: list of (diff, bpm, blocks, time2label). Per difficulty:
    the counts block + the whole-song every-note timeline grouped by phrase."""
    from ..version import build_timestamp
    lines = ["NoteSight generation report",
             f"generator: Beat Saber  {game_version}",
             f"generated: {build_timestamp()}",
             f"song: {title}", ""]
    for (diff, bpm, blocks, t2l) in charts:
        lines.append(f"  [{diff}]  bpm {bpm:.1f}  {len(blocks)} blocks")
        lines += counts_lines(blocks, bpm)
        lines.append("  -- timeline (whole song, EVERY note, grouped by phrase; ' = mirrored) --")
        lines += letter_timeline(blocks, bpm, t2l)
        lines.append("")
    with open(os.path.join(song_dir, "_generation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
