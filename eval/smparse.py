r"""smparse.py -- minimal StepMania .sm parser for EVALUATION / ground truth.

Pulls the header timing (#OFFSET, #BPMS, #STOPS) and every #NOTES chart out of a
reference .sm, and converts each chart's note rows into absolute audio-seconds so
NoteSight's output can be scored against a hand-authored chart.

Only what the eval harness needs -- not a full simfile loader. dance-single only
by default; holds/rolls count their HEAD as one onset, mines/tails are ignored.

Timing math (constant or piecewise BPM, StepMania convention):
    time(beat) = -OFFSET + sum_of_bpm_segments_up_to(beat) + stops_before(beat)
A row r of R in measure m is beat = 4*m + 4*r/R (4 beats per measure).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# Note chars that count as a fresh "hit" (an onset the player must act on).
# 1=tap, 2=hold head, 4=roll head. 3=tail, M=mine, 0=empty -> not onsets.
ONSET_CHARS = frozenset("124")


@dataclass
class Chart:
    stepstype: str          # e.g. "dance-single"
    difficulty: str         # e.g. "Easy" / "Medium" / "Hard" / "Challenge"
    meter: int              # numeric rating
    times: list[float]      # sorted distinct onset times (seconds)
    rows: int               # total note rows with >=1 onset (jumps count once)


@dataclass
class SimFile:
    title: str
    music: str
    offset: float
    bpms: list[tuple[float, float]]   # (beat, bpm), sorted
    stops: list[tuple[float, float]]  # (beat, seconds), sorted
    charts: list[Chart] = field(default_factory=list)

    def elapsed(self, beat: float) -> float:
        """Absolute audio time (seconds) of a musical beat."""
        return _elapsed(beat, self.offset, self.bpms, self.stops)

    def chart(self, difficulty: str, stepstype: str = "dance-single"):
        want = difficulty.lower()
        for c in self.charts:
            if c.stepstype == stepstype and c.difficulty.lower() == want:
                return c
        return None


def _elapsed(beat, offset, bpms, stops):
    t = -offset
    for i, (sb, bpm) in enumerate(bpms):
        nb = bpms[i + 1][0] if i + 1 < len(bpms) else float("inf")
        if beat <= sb:
            break
        t += (min(beat, nb) - sb) * (60.0 / bpm)
    for sb, sdur in stops:
        if sb < beat:
            t += sdur
    return t


def _tag_value(raw: str) -> str:
    """Strip //comments and the trailing ; from a raw tag value."""
    # Remove line comments first (StepMania // to end of line).
    raw = re.sub(r"//[^\n]*", "", raw)
    return raw.strip().rstrip(";").strip()


def _parse_pairs(val: str) -> list[tuple[float, float]]:
    """'0.000=136.340,32.5=180' -> [(0.0,136.34),(32.5,180.0)]."""
    out = []
    for chunk in val.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        b, v = chunk.split("=", 1)
        try:
            out.append((float(b), float(v)))
        except ValueError:
            continue
    out.sort()
    return out


def _split_tags(text: str) -> list[tuple[str, str]]:
    """Yield (TAGNAME, raw_value) for each #TAG:value; in the file."""
    tags = []
    for m in re.finditer(r"#([A-Za-z0-9_]+):", text):
        name = m.group(1).upper()
        start = m.end()
        end = text.find(";", start)
        if end == -1:
            end = len(text)
        tags.append((name, text[start:end]))
    return tags


def _parse_notes(raw: str, offset, bpms, stops) -> Chart | None:
    """Parse one #NOTES payload (everything between #NOTES: and its ;)."""
    # 5 colon-separated metadata fields, then the note body:
    #   stepstype : author/desc : difficulty : meter : radar : <body>
    parts = raw.split(":", 5)
    if len(parts) < 6:
        return None
    stepstype = parts[0].strip()
    difficulty = parts[2].strip()
    try:
        meter = int(float(parts[3].strip()))
    except ValueError:
        meter = 0
    body = re.sub(r"//[^\n]*", "", parts[5])  # drop measure comments

    times: list[float] = []
    rows = 0
    measures = body.split(",")
    for m_idx, measure in enumerate(measures):
        lines = [ln.strip() for ln in measure.splitlines() if ln.strip()]
        R = len(lines)
        if R == 0:
            continue
        for r, line in enumerate(lines):
            if any(ch in ONSET_CHARS for ch in line):
                beat = 4.0 * m_idx + 4.0 * r / R
                times.append(_elapsed(beat, offset, bpms, stops))
                rows += 1
    times.sort()
    return Chart(stepstype=stepstype, difficulty=difficulty, meter=meter,
                 times=times, rows=rows)


def parse_sm(path: str) -> SimFile:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    title = music = ""
    offset = 0.0
    bpms: list[tuple[float, float]] = []
    stops: list[tuple[float, float]] = []
    notes_raw: list[str] = []

    for name, raw in _split_tags(text):
        if name == "TITLE":
            title = _tag_value(raw)
        elif name == "MUSIC":
            music = _tag_value(raw)
        elif name == "OFFSET":
            try:
                offset = float(_tag_value(raw))
            except ValueError:
                offset = 0.0
        elif name == "BPMS":
            bpms = _parse_pairs(_tag_value(raw))
        elif name == "STOPS":
            stops = _parse_pairs(_tag_value(raw))
        elif name == "NOTES":
            notes_raw.append(raw)

    if not bpms:
        bpms = [(0.0, 120.0)]

    sim = SimFile(title=title, music=music, offset=offset, bpms=bpms, stops=stops)
    for raw in notes_raw:
        c = _parse_notes(raw, offset, bpms, stops)
        if c is not None and c.times:
            sim.charts.append(c)
    return sim


if __name__ == "__main__":
    import sys
    sim = parse_sm(sys.argv[1])
    print(f"{sim.title!r}  music={sim.music!r}  offset={sim.offset}  "
          f"bpms={sim.bpms}  stops={len(sim.stops)}")
    for c in sim.charts:
        span = f"{c.times[0]:.2f}..{c.times[-1]:.2f}s" if c.times else "-"
        print(f"  {c.stepstype:14s} {c.difficulty:10s} meter={c.meter:2d}  "
              f"onset-rows={c.rows:5d}  [{span}]")
