"""End-to-end pipeline: audio -> analysis -> chart."""
from __future__ import annotations

from notesight import ChartSpec, analyze_audio, build_chart


def test_analyze_finds_onsets_and_tempo(click_audio):
    mono, sr = click_audio
    a = analyze_audio(mono, sr)
    assert a.duration > 0
    assert len(a.onsets) > 0
    assert a.grid.bpm > 0
    # steady 120 BPM click track -> tempo estimate near 120 (allow octave errors)
    assert any(abs(a.grid.bpm - b) < 6 for b in (60, 120, 240))


def test_build_chart_places_notes(click_audio):
    mono, sr = click_audio
    a = analyze_audio(mono, sr)
    r = build_chart(ChartSpec(difficulty="medium"), a)
    assert len(r.notes) > 0
    assert r.bpm > 0
    # notes are time-ordered and within the song
    times = [n.time for n in r.notes]
    assert times == sorted(times)
    assert max(times) <= r.duration + 1e-6


def test_build_chart_is_deterministic(click_audio):
    mono, sr = click_audio
    a = analyze_audio(mono, sr)
    spec = ChartSpec(difficulty="hard")
    r1 = build_chart(spec, a)
    r2 = build_chart(spec, a)
    assert [(n.time, n.lane) for n in r1.notes] == [(n.time, n.lane) for n in r2.notes]
