"""formats/__init__.py -- exporter registry.

get_format("stepmania") -> a ChartFormat instance. Beat Saber (NoteNinja VR) and
Clone Hero / Guitar Hero plug in here later: add a module implementing the
ChartFormat ABC (formats/base.py) and register it below. Nothing upstream --
onset detection, note selection, difficulty -- has to change.
"""
from __future__ import annotations

from .base import ChartFormat, SongMeta
from .stepmania import StepManiaFormat
from .beatsaber import BeatSaberFormat

# name -> ChartFormat subclass. Add clonehero.py entries here.
FORMATS: dict[str, type[ChartFormat]] = {
    StepManiaFormat.name: StepManiaFormat,
    BeatSaberFormat.name: BeatSaberFormat,
}


def get_format(name: str) -> ChartFormat:
    key = name.lower()
    if key not in FORMATS:
        avail = ", ".join(sorted(FORMATS))
        raise ValueError(f"unknown format {name!r}; available: {avail}")
    return FORMATS[key]()


def available_formats() -> list[str]:
    return sorted(FORMATS)
