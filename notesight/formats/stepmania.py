"""formats/stepmania.py -- 4-panel dance mapping + .sm song-folder writer.

Lane mapping reuses the adaptive-brightness assigner from the original
charter.py: split each hit into low/high by the *median* brightness of the
selected onsets (adaptive, not a fixed Hz cutoff), bass -> {Left,Down},
treble -> {Up,Right}, alternating within each pair so a groove becomes a
staircase instead of a Left<->Right ping-pong.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from math import gcd

from ..onsets import OnsetEvent
from ..difficulty import Difficulty
from ..radar import compute_radar, predict_meter
from ..patterns import assign as assign_patterns
from .base import ChartFormat, SongMeta

# 4-panel lanes, matching StepMania column order.
LEFT, DOWN, UP, RIGHT = 0, 1, 2, 3
LANE_NAMES = ["Left", "Down", "Up", "Right"]

ROWS_PER_BEAT = 12          # 1/48-note grid: 48 rows/measure, the snap resolution
ROWS_PER_MEASURE = ROWS_PER_BEAT * 4

# Musical subdivisions we snap to, as notes-per-beat whose row step divides the
# 1/48 grid (12/n integer): 1=quarter, 2=8th, 4=16th (straight), then 3=8th- and
# 6=16th-triplet. STRAIGHT IS TRIED FIRST so a note near a 16th is not mistaken
# for a triplet; triplets are only used when a note genuinely sits there. Mixing
# families in one measure forces the ugly 1/48 rendering, so this ordering keeps
# measures clean (all-straight -> <=16 rows, all-triplet -> <=24 rows).
SNAP_PER_BEAT = (1, 2, 4, 3, 6)
# Tight enough to separate a 16th (0.25 beat) from an 8th-triplet (0.333 beat) --
# their grids are only 0.083 beat apart -- yet loose vs the ~0.02-beat onset
# jitter left after grid estimation.
SNAP_TOL_BEATS = 0.04


@dataclass
class Note:
    time: float            # seconds from start of audio
    lane: int              # 0..3
    strength: float = 0.0  # onset salience, for debugging/weighting
    duration: float = 0.0  # >0 = hold note (freeze), seconds held before release


def map_to_lanes(onsets: list[OnsetEvent], diff: Difficulty, seed: int = 0) -> list[Note]:
    """Assign 4-panel lanes to selected onsets via the music-driven pattern
    engine (staircases/crossovers + motif reuse); jumps on strong hits. `seed`
    varies the symmetry/foot choices for a different-but-equivalent chart."""
    return [Note(time=t, lane=lane, strength=s)
            for t, lane, s in assign_patterns(onsets, diff, seed)]


# Line counts a whole measure may use, split by rhythmic family. Each measure
# commits to ONE family so it renders cleanly (StepMania colors a 16-line measure
# as 16ths, a 24-line as 16th-triplets); mixing families in a measure would force
# the noisy 1/48 rendering. Straight is preferred; triplets only when they fit
# clearly better.
STRAIGHT_LINES = (4, 8, 16)     # quarter / 8th / 16th
TRIPLET_LINES = (12, 24)        # 8th-triplet / 16th-triplet


def _choose_measure_lines(beats_in_measure: list[float],
                          max_lines: int = 48,
                          allow_triplets: bool = True) -> int:
    """Pick the single line count that best represents a measure's notes.

    `beats_in_measure` are continuous positions in [0,4). Returns a value from
    STRAIGHT/TRIPLET_LINES: the coarsest straight grid within tolerance, else the
    coarsest triplet grid within tolerance, else whichever family fits with less
    total error (so a measure is never rendered at the ugly 1/48 resolution).

    `max_lines` caps the finest subdivision (e.g. 8 = "8th notes only"), and
    `allow_triplets` gates the triplet family -- both driven by the ChartSpec.
    """
    def max_err(R):
        return max(abs(b * R / 4 - round(b * R / 4)) * 4 / R
                   for b in beats_in_measure)

    def total_err(R):
        return sum(abs(b * R / 4 - round(b * R / 4)) * 4 / R
                   for b in beats_in_measure)

    straights = [R for R in STRAIGHT_LINES if R <= max_lines] or [STRAIGHT_LINES[0]]
    straight = next((R for R in straights if max_err(R) <= SNAP_TOL_BEATS),
                    straights[-1])
    triplets = [R for R in TRIPLET_LINES if R <= max_lines] if allow_triplets else []
    if triplets:
        triplet = next((R for R in triplets if max_err(R) <= SNAP_TOL_BEATS),
                       triplets[-1])
        # Prefer straight on a tie; triplets must clearly fit better to win.
        if total_err(triplet) < total_err(straight) - 1e-9:
            return triplet
    return straight


def _note_marks(notes: list[Note], sec_per_beat: float, beat0: float):
    """Expand notes into (measure, beat_in_measure, lane, symbol) marks.

    A tap becomes one '1'. A hold (duration>0) becomes a '2' head at its start
    and a '3' tail at its end -- which may fall in a LATER measure; StepMania
    holds the panel through the empty rows between them. Rounding to the 1/48
    grid first keeps a note a hair before a downbeat in the right measure.
    """
    marks: list[tuple[int, float, int, str]] = []
    for n in notes:
        beat = max(0.0, (n.time - beat0) / sec_per_beat)
        m = int(round(beat * ROWS_PER_BEAT)) // ROWS_PER_MEASURE
        dur = getattr(n, "duration", 0.0) or 0.0
        if dur > 1e-3:
            marks.append((m, beat - 4.0 * m, n.lane, "2"))
            ebeat = max(beat, (n.time + dur - beat0) / sec_per_beat)
            em = int(round(ebeat * ROWS_PER_BEAT)) // ROWS_PER_MEASURE
            marks.append((em, ebeat - 4.0 * em, n.lane, "3"))
        else:
            marks.append((m, beat - 4.0 * m, n.lane, "1"))
    return marks


def snap_times(times, bpm: float, beat0: float = 0.0,
               max_lines: int = 48, allow_triplets: bool = True) -> list[float]:
    """Snap note times exactly as the exporter quantizes them (for eval/tools)."""
    spb = 60.0 / bpm
    # measure -> list of (original index, beat_in_measure)
    by_measure: dict[int, list[tuple[int, float]]] = {}
    for i, t in enumerate(times):
        beat = max(0.0, (t - beat0) / spb)
        m = int(round(beat * ROWS_PER_BEAT)) // ROWS_PER_MEASURE
        by_measure.setdefault(m, []).append((i, beat - 4.0 * m))
    out = [0.0] * len(times)
    for m, items in by_measure.items():
        R = _choose_measure_lines([b for _, b in items], max_lines, allow_triplets)
        for i, b in items:
            line = min(int(round(b * R / 4)), R - 1)
            out[i] = beat0 + (4.0 * m + line * 4.0 / R) * spb
    return out


def notes_to_measures(notes: list[Note], bpm: float, beat0: float = 0.0,
                      max_lines: int = 48, allow_triplets: bool = True) -> str:
    """Quantize notes to the musical beat grid and render the .sm measure body.

    Notes are placed by BEAT relative to `beat0` (the audio time of beat 0, which
    the exporter also writes as #OFFSET). Each measure commits to one rhythmic
    family (straight or triplet) so it renders on clean, correctly-colored lines.
    `max_lines` caps the finest subdivision and `allow_triplets` gates triplets.

    KEY INSIGHT (from the handoff): StepMania plays row r at the same BPM we
    export here, so seconds -> rows -> seconds round-trips exactly. Snapping to a
    good grid makes the chart read musically; it does NOT change gameplay sync.
    """
    spb = 60.0 / bpm
    # First pass: every head (and each hold's tail) placed by (measure, beat-in-
    # measure), collecting positions so each measure picks ONE clean resolution.
    heads = []   # (m, bim, lane, is_hold)
    tails = []   # (end_m, end_bim, lane, head_m, head_bim)
    positions: dict[int, list[float]] = {}
    for n in notes:
        beat = max(0.0, (n.time - beat0) / spb)
        m = int(round(beat * ROWS_PER_BEAT)) // ROWS_PER_MEASURE
        bim = beat - 4.0 * m
        positions.setdefault(m, []).append(bim)
        is_hold = getattr(n, "duration", 0.0) > 1e-3
        heads.append((m, bim, n.lane, is_hold))
        if is_hold:
            ebeat = max(beat, (n.time + n.duration - beat0) / spb)
            em = int(round(ebeat * ROWS_PER_BEAT)) // ROWS_PER_MEASURE
            positions.setdefault(em, []).append(ebeat - 4.0 * em)
            tails.append((em, ebeat - 4.0 * em, n.lane, m, bim))
    if not heads:
        return "0000\n0000\n0000\n0000\n"

    R_of = {m: _choose_measure_lines(b, max_lines, allow_triplets)
            for m, b in positions.items()}

    def row_of(m, bim):
        R = R_of[m]
        return min(int(round(bim * R / 4)), R - 1)

    grid: dict[int, dict[tuple[int, int], str]] = {}

    def put(m, row, lane, sym):
        grid.setdefault(m, {})[(row, lane)] = sym

    # Place heads, but CAP each row at 2 arrows -- dance-single never has 3+
    # ("hands" / one foot on two arrows). Hold heads claim slots first so a hold
    # is never the one dropped (its tail would orphan); taps fill what's left.
    # Same-lane collisions collapse to a single note.
    MAX_PER_ROW = 2
    row_lanes: dict[tuple[int, int], set] = {}
    kept_holds: set = set()
    for m, bim, lane, is_hold in sorted(heads, key=lambda h: not h[3]):
        r = row_of(m, bim)
        used = row_lanes.setdefault((m, r), set())
        if lane in used or len(used) >= MAX_PER_ROW:
            continue                            # dup lane, or would be a 3rd arrow
        used.add(lane)
        put(m, r, lane, "2" if is_hold else "1")
        if is_hold:
            kept_holds.add((m, bim, lane))

    # Tails: a hold's '3' lands strictly AFTER its '2', never on an occupied cell,
    # and never as a 3rd arrow on a full row -- so holds stay balanced (2/3) and
    # no row exceeds two arrows. If the head was capped away, drop the tail.
    for em, ebim, lane, hm, hbim in tails:
        if (hm, hbim, lane) not in kept_holds:
            continue
        r = row_of(em, ebim)
        if em == hm:
            r = max(r, row_of(hm, hbim) + 1)   # never collapse onto the head row
        while True:
            if r >= R_of.get(em, 4):           # spilled past the measure -> next
                em, r = em + 1, 0
                R_of.setdefault(em, 4)
                continue
            rowset = row_lanes.setdefault((em, r), set())
            if grid.get(em, {}).get((r, lane)) is not None:
                r += 1                          # cell taken -> next row
                continue
            if lane not in rowset and len(rowset) >= MAX_PER_ROW:
                r += 1                          # row full -> don't make a 3rd
                continue
            break
        put(em, r, lane, "3")
        row_lanes.setdefault((em, r), set()).add(lane)

    max_measure = max(max(grid, default=0), max(R_of, default=0))
    body = []
    for m in range(max_measure + 1):
        R = R_of.get(m, 4)
        cell = grid.get(m, {})
        body.append("\n".join(
            "".join(cell.get((r, lane), "0") for lane in range(4))
            for r in range(R)))
    return "\n,\n".join(body) + "\n"


def _sanitize(name: str) -> str:
    keep = "-_.() "
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return cleaned or "Untitled"


class StepManiaFormat(ChartFormat):
    name = "stepmania"

    def map(self, onsets: list[OnsetEvent], meta: SongMeta) -> list[Note]:
        diff = meta.difficulty
        if diff is None:
            raise ValueError("StepManiaFormat.map needs meta.difficulty")
        return map_to_lanes(onsets, diff)

    def write(self, notes: list[Note], meta: SongMeta, out_dir: str) -> str:
        diff = meta.difficulty
        title = _sanitize(meta.title)
        song_dir = os.path.join(out_dir, title)
        os.makedirs(song_dir, exist_ok=True)

        # Copy the audio in and reference it by basename.
        music_name = ""
        if meta.audio_path and os.path.isfile(meta.audio_path):
            music_name = os.path.basename(meta.audio_path)
            dst = os.path.join(song_dir, music_name)
            if os.path.abspath(dst) != os.path.abspath(meta.audio_path):
                shutil.copyfile(meta.audio_path, dst)

        diff_name = diff.name if diff else "Medium"
        # StepMania's difficulty slot names (Beginner/Easy/Medium/Hard/Challenge).
        slot = {
            "Beginner": "Beginner", "Easy": "Easy", "Medium": "Medium",
            "Hard": "Hard", "Expert": "Challenge",
        }.get(diff_name, "Medium")

        # Derive the numeric meter + radar from the chart's shape (ITG's own
        # PredictMeter), so the difficulty number reflects what we actually
        # generated instead of a hardcoded guess.
        radar = compute_radar(notes, meta.bpm, meta.beat0, meta.duration)
        meter = predict_meter(radar, slot)

        body = notes_to_measures(notes, meta.bpm, meta.beat0,
                                 meta.max_subdivision, meta.allow_triplets)
        # StepMania #OFFSET convention: time(beat) = -OFFSET + beat*period, so
        # the audio time of beat 0 (meta.beat0) is exported as -beat0.
        offset = -meta.beat0
        header = (
            f"#TITLE:{meta.title};\n"
            f"#ARTIST:{meta.artist};\n"
            f"#MUSIC:{music_name};\n"
            f"#OFFSET:{offset:.3f};\n"
            f"#SAMPLESTART:0.000;\n"
            f"#SAMPLELENGTH:12.000;\n"
            f"#SELECTABLE:YES;\n"
            f"#BPMS:0.000={meta.bpm:.3f};\n\n"
        )
        notes_block = (
            "#NOTES:\n"
            "     dance-single:\n"
            "     NoteSight:\n"
            f"     {slot}:\n"
            f"     {meter}:\n"
            f"     {radar.sm_field()}:\n"
            f"{body};\n"
        )
        sm_path = os.path.join(song_dir, f"{title}.sm")
        with open(sm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(header + notes_block)
        return sm_path
