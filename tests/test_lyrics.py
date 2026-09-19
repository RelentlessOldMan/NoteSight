"""The lyric-driven structure feature: a repeated chorus (same words coming back)
becomes a per-bar reuse map so the charter can mirror the pattern."""
from __future__ import annotations

from notesight.lyrics import parse_lrc, chorus_structure


def _write_lrc(path):
    # verse ... a 4-line chorus (spans 4 bars) ... verse ... the SAME chorus again,
    # on a steady grid. lyrics.py needs a block >= 4 bars recurring >= 4 bars later.
    lines = [
        "[00:00.00] first verse line one",
        "[00:02.00] first verse line two",
        "[00:04.00] we all sing the chorus now",
        "[00:06.00] this is the hook you remember",
        "[00:08.00] raise your hands up in the air",
        "[00:10.00] shout it back to me again",
        "[00:12.00] second verse different words",
        "[00:14.00] more second verse words here",
        "[00:16.00] we all sing the chorus now",
        "[00:18.00] this is the hook you remember",
        "[00:20.00] raise your hands up in the air",
        "[00:22.00] shout it back to me again",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_parse_lrc_reads_timestamps(tmp_path):
    p = _write_lrc(tmp_path / "song.lrc")
    lines = parse_lrc(p)
    assert len(lines) == 12
    assert lines[0][0] == 0.0
    assert "chorus" in lines[2][1]


def test_chorus_structure_detects_repeat(tmp_path):
    p = _write_lrc(tmp_path / "song.lrc")
    lines = parse_lrc(p)
    # 120 BPM -> 2 s/bar; the 4-bar chorus recurs 12 s (6 bars) later
    src = chorus_structure(lines, bpm=120.0, beat0=0.0, duration=24.0)
    assert src is not None
    # at least one bar points back at an earlier bar (a detected repeat)
    assert any(src[b] != b for b in range(len(src)))
