"""pipeline.py -- the one place audio becomes a chart, driven by a ChartSpec.

ChartSpec is the single options object every frontend fills in: the CLI sets it
from flags, the web app sets it from the form, and both call build_chart(). Simple
UIs touch a couple fields and leave the rest on auto; advanced UIs / scripts set
everything. That keeps the faces thin and impossible to drift out of sync.

The expensive, spec-independent work (load audio, detect onsets, estimate the beat
grid) is split into analyze_audio() so a live UI can cache it once and re-chart
instantly as the user drags sliders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

from .onsets import OnsetEvent, detect_onsets, energy_envelope
from .difficulty import DIFFICULTIES, Difficulty
from .selection import select
from .beatgrid import BeatGrid, estimate_grid
from .radar import Radar, compute_radar, predict_meter
from .holds import apply_holds, sanitize_holds
from .structure import detect_structure, apply_structure
from .lyrics import parse_lrc, chorus_structure
from .formats.stepmania import Note, map_to_lanes, snap_times

# Difficulty preset name -> ITG PredictMeter difficulty slot.
_SLOT = {"beginner": "Beginner", "easy": "Easy", "medium": "Medium",
         "hard": "Hard", "expert": "Challenge"}


@dataclass
class ChartSpec:
    """Everything that shapes a chart. Auto-defaults = the simple-UI behavior."""
    difficulty: str = "medium"       # preset name (drives density + meter slot)
    jumps: bool | None = None        # None=use difficulty default, else force on/off
    max_subdivision: int = 16        # finest quantize grid: 4/8/16 (whole..16ths)
    allow_triplets: bool = True      # let measures use 8th/16th triplets
    bpm: float | None = None         # override auto-detected tempo
    diff_override: Difficulty | None = None  # use THIS Difficulty (density) instead of
    #   the shared DIFFICULTIES[name] table -- lets Beat Saber run its own density
    #   ladder without disturbing the DDR-calibrated presets. The meter slot still
    #   comes from `difficulty` (name), so the two stay independent.

    def resolved_difficulty(self) -> Difficulty:
        base = self.diff_override or DIFFICULTIES.get(self.difficulty,
                                                      DIFFICULTIES["medium"])
        if self.jumps is None:
            return base
        return replace(base, allow_jumps=self.jumps)


@dataclass
class Analysis:
    """Spec-independent, cacheable: the costly DSP done once per audio file."""
    onsets: list[OnsetEvent]
    grid: BeatGrid
    duration: float
    sr: int
    energy_times: "object" = None   # RMS envelope times (for hold detection)
    energy: "object" = None         # RMS envelope values, normalized 0..1
    structure: "object" = None      # per-bar source map (repeated-section reuse)


@dataclass
class ChartResult:
    bpm: float
    beat0: float
    duration: float
    meter: int
    radar: Radar
    notes: list[Note]                # (time, lane, strength)
    candidate_onsets: int
    difficulty: str


def analyze_audio(mono, sr: int, lrc_path: str = None) -> Analysis:
    """Detect onsets + beat grid + energy envelope + song structure. Cache this.

    Structure (repeated-section map) comes from the audio's self-similarity, but
    if a timestamped `.lrc` is given and its lyric-repeat structure covers more of
    the song, that wins -- the chorus is where the WORDS come back, a far more
    reliable signal than the mix.
    """
    onsets = detect_onsets(mono, sr)
    grid = estimate_grid(mono, sr, onsets=onsets)
    e_times, e = energy_envelope(mono, sr)
    duration = len(mono) / sr
    structure = detect_structure(mono, sr, grid.bpm, grid.beat0, duration)
    if lrc_path and os.path.isfile(lrc_path):
        lyric = chorus_structure(parse_lrc(lrc_path), grid.bpm, grid.beat0, duration)
        cov = lambda s: sum(1 for b in range(len(s)) if s[b] != b) if s else 0
        if lyric is not None and cov(lyric) >= cov(structure):
            structure = lyric
    return Analysis(onsets=onsets, grid=grid, duration=duration, sr=sr,
                    energy_times=e_times, energy=e, structure=structure)


def build_chart(spec: ChartSpec, analysis: Analysis) -> ChartResult:
    """Turn cached analysis into a chart under the given spec. Fast (no DSP)."""
    diff = spec.resolved_difficulty()
    bpm = spec.bpm if spec.bpm else analysis.grid.bpm
    beat0 = analysis.grid.beat0

    selected = select(analysis.onsets, diff,
                      analysis.energy_times, analysis.energy)
    notes = map_to_lanes(selected, diff)
    # Sustained notes sitting on a long gap become holds (freeze arrows).
    if analysis.energy is not None:
        apply_holds(notes, analysis.onsets, analysis.energy_times,
                    analysis.energy, bpm, beat0, analysis.duration)
    # Lock repeated sections together: stamp each repeat bar with an exact,
    # bar-aligned copy of its source bar (recognizable + consistent colors).
    if analysis.structure:
        notes = apply_structure(notes, analysis.structure,
                                analysis.grid.bpm, beat0)
    # Ensure no hold overlaps the next note in its lane (avoids invalid,
    # unclosed holds after the hold + structure passes).
    sanitize_holds(notes)
    # Radar reflects what actually EXPORTS: compute it from the snapped times so
    # a subdivision cap (e.g. 8ths-only) shows up as lower Chaos and meter.
    # Carry each note's duration so holds populate the Freeze radar value.
    snapped = snap_times([n.time for n in notes], bpm, beat0,
                         spec.max_subdivision, spec.allow_triplets)
    radar = compute_radar([Note(t, n.lane, n.strength, n.duration)
                           for t, n in zip(snapped, notes)],
                          bpm, beat0, analysis.duration)
    meter = predict_meter(radar, _SLOT.get(spec.difficulty, "Medium"))

    return ChartResult(
        bpm=bpm, beat0=beat0, duration=analysis.duration, meter=meter,
        radar=radar, notes=notes, candidate_onsets=len(analysis.onsets),
        difficulty=spec.difficulty,
    )
