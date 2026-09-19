"""formats/base.py -- the pluggable-exporter seam.

A ChartFormat turns selected onsets into game-specific notes and writes them to
disk. StepMania is implemented now; Beat Saber (feeds the NoteNinja VR game) and
Clone Hero / Guitar Hero are future formats that only need to implement this ABC
-- nothing upstream (onset detection, note selection, difficulty) changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..onsets import OnsetEvent
from ..difficulty import Difficulty


@dataclass
class SongMeta:
    title: str = "Untitled"
    artist: str = "Unknown Artist"
    audio_path: str = ""     # source audio file, copied into the song folder
    bpm: float = 120.0
    beat0: float = 0.0       # audio time (s) of the first beat; grid phase
    duration: float = 0.0    # seconds
    difficulty: Difficulty | None = None
    max_subdivision: int = 48   # cap the finest quantize grid (8 = "8ths only")
    allow_triplets: bool = True
    extra: dict = field(default_factory=dict)


class ChartFormat(ABC):
    #: short registry key, e.g. "stepmania"
    name: str = ""

    @abstractmethod
    def map(self, onsets: list[OnsetEvent], meta: SongMeta):
        """Turn selected onsets into this format's note objects."""

    @abstractmethod
    def write(self, notes, meta: SongMeta, out_dir: str) -> str:
        """Write the chart to `out_dir`. Returns the primary output path."""
