"""lyrics.py -- find repeated sections (choruses) from a timestamped .lrc.

Acoustic self-similarity misses a lot of a song's structure (a chorus that isn't
bit-identical in the mix slips past it). But the CHORUS IS WHERE THE WORDS COME
BACK -- and a .lrc timestamps exactly when each line is sung. So we find repeated
runs of lyric lines, map their times onto the beat grid, and hand back the same
per-bar `source` map the acoustic detector produces (structure.py) -- which the
reuse/mirror pass then stamps. Far more reliable for songs with lyrics.
"""
from __future__ import annotations

import math
import re

MIN_LINES = 2            # a repeated block must be at least this many lyric lines
MIN_SHIFT_BARS = 4       # ... and recur at least this many bars later
MIN_BLOCK_BARS = 4       # ... and span at least this many bars
BEATS_PER_BAR = 4

_TS = re.compile(r"\[(\d{1,2}):(\d{1,2})(?:[.:](\d{1,3}))?\]")


def parse_lrc(path: str):
    """Return sorted [(time_sec, normalized_text)] for every timestamped line."""
    out = []
    try:
        raw = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return []
    for line in raw:
        stamps = _TS.findall(line)
        if not stamps:
            continue
        text = re.sub(r"\[[^\]]*\]", "", line)          # strip all [..] tags
        norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not norm:
            continue
        for mm, ss, fr in stamps:
            t = int(mm) * 60 + int(ss) + (int(fr) / 10 ** len(fr) if fr else 0.0)
            out.append((t, norm))
    out.sort()
    return out


def chorus_structure(lyric_lines, bpm: float, beat0: float, duration: float):
    """Repeated lyric runs -> per-bar `source` map (bar b repeats source[b], or
    itself). None if nothing repeats. Same format as structure.detect_structure.
    """
    if bpm <= 0 or duration <= 0 or len(lyric_lines) < 2 * MIN_LINES:
        return None
    times = [t for t, _ in lyric_lines]
    texts = [x for _, x in lyric_lines]
    m = len(texts)
    spb = 60.0 / bpm
    barlen = BEATS_PER_BAR * spb
    start = beat0 if beat0 > 0 else 0.0
    n_bars = max(1, int(math.ceil((duration - start) / barlen)))
    source = list(range(n_bars))
    assigned = [False] * n_bars

    # Every maximal repeated line-run (source i.., repeat j.., length L), longest
    # first so a full chorus wins over a stray shared line.
    cands = []
    for i in range(m):
        for j in range(i + MIN_LINES, m):
            if texts[i] != texts[j]:
                continue
            L = 0
            while i + L < j and j + L < m and texts[i + L] == texts[j + L]:
                L += 1
            if L >= MIN_LINES:
                cands.append((L, i, j))
    cands.sort(reverse=True)

    for L, i, j in cands:
        s_end = times[i + L] if i + L < m else duration
        sb0 = max(0, round((times[i] - start) / barlen))
        sb1 = min(n_bars, round((s_end - start) / barlen))
        shift = round((times[j] - times[i]) / barlen)   # bars the section moved
        if sb1 - sb0 < MIN_BLOCK_BARS or shift < MIN_SHIFT_BARS:
            continue
        for sb in range(sb0, sb1):
            rb = sb + shift
            if 0 <= rb < n_bars and rb > sb and not assigned[rb]:
                source[rb] = sb
                assigned[rb] = True

    return source if any(source[b] != b for b in range(n_bars)) else None
