"""Exporters write valid, loadable chart files."""
from __future__ import annotations

import json
import os

from notesight import (ChartSpec, analyze_audio, build_chart, get_format,
                       SongMeta)


def _chart(click_wav):
    import soundfile as sf
    mono, sr = sf.read(click_wav, dtype="float32")
    a = analyze_audio(mono, sr)
    spec = ChartSpec(difficulty="medium")
    r = build_chart(spec, a)
    meta = SongMeta(title="Click Test", artist="NoteSight",
                    audio_path=click_wav, bpm=r.bpm, beat0=r.beat0,
                    duration=r.duration, difficulty=spec.resolved_difficulty())
    return r, meta


def test_stepmania_writes_sm(click_wav, tmp_path):
    r, meta = _chart(click_wav)
    out = get_format("stepmania").write(r.notes, meta, str(tmp_path))
    assert out.endswith(".sm")
    assert os.path.isfile(out)
    text = open(out, encoding="utf-8").read()
    assert "#NOTES" in text
    assert "#BPMS" in text


def test_beatsaber_writes_info_and_dat(click_wav, tmp_path):
    r, meta = _chart(click_wav)
    # empty audio_path -> skip egg encoding (fast); we only assert the JSON files
    meta.audio_path = ""
    song_dir = get_format("beatsaber").write(r.notes, meta, str(tmp_path))
    info_path = os.path.join(song_dir, "info.dat")
    assert os.path.isfile(info_path)
    info = json.load(open(info_path, encoding="utf-8"))
    sets = info["_difficultyBeatmapSets"]
    assert sets and sets[0]["_difficultyBeatmaps"]
    dat_name = sets[0]["_difficultyBeatmaps"][0]["_beatmapFilename"]
    dat = json.load(open(os.path.join(song_dir, dat_name), encoding="utf-8"))
    assert "_notes" in dat
