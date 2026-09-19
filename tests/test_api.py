"""Public-API surface: the things frontends and downstream code import."""
from __future__ import annotations

import notesight


def test_exports_present():
    for name in ("ChartSpec", "analyze_audio", "build_chart",
                 "DIFFICULTIES", "get_format", "available_formats", "SongMeta"):
        assert hasattr(notesight, name), f"missing export: {name}"


def test_difficulty_presets():
    d = notesight.DIFFICULTIES
    for preset in ("beginner", "easy", "medium", "hard", "expert"):
        assert preset in d


def test_formats_registered():
    fmts = notesight.available_formats()
    assert "stepmania" in fmts
    assert "beatsaber" in fmts


def test_get_format_unknown_raises():
    import pytest
    with pytest.raises(ValueError):
        notesight.get_format("clonehero")   # not implemented yet
