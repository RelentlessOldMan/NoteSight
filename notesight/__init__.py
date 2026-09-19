"""notesight -- turn an audio file into a rhythm-game chart.

Pipeline:
  audio_io.load_audio  ->  onsets.detect_onsets  ->  selection.select
  ->  formats.get_format(name).map / .write

The core (onsets, selection, difficulty, tempo) is plain array math. Formats are
pluggable behind a small ABC.
"""
from __future__ import annotations

from .onsets import OnsetEvent, detect_onsets
from .difficulty import Difficulty, DIFFICULTIES
from .selection import select
from .tempo import estimate_bpm
from .beatgrid import BeatGrid, estimate_grid
from .radar import Radar, compute_radar, predict_meter
from .pipeline import ChartSpec, Analysis, ChartResult, analyze_audio, build_chart
from .formats import get_format, available_formats, SongMeta
from .formats.stepmania import Note, map_to_lanes, LANE_NAMES

__all__ = [
    "OnsetEvent", "detect_onsets",
    "Difficulty", "DIFFICULTIES",
    "select",
    "estimate_bpm",
    "BeatGrid", "estimate_grid",
    "Radar", "compute_radar", "predict_meter",
    "ChartSpec", "Analysis", "ChartResult", "analyze_audio", "build_chart",
    "get_format", "available_formats", "SongMeta",
    "Note", "map_to_lanes", "LANE_NAMES",
]
