"""structure.py -- detect repeated sections (choruses/verses) from the audio and
reuse the SAME chart figure for them.

Why this exists (user feedback): when the same musical section comes back, the
chart should come back too -- same rhythm, same note colors -- so it reads as
recognizable instead of a fresh wall of arrows. Today the pattern engine reuses a
phrase's *feet* but re-quantizes each occurrence against its own (jittery) onset
times, so an identical chorus can render blue one time and yellow the next. That
looks wrong.

The fix: find which BARS are acoustic repeats of earlier bars (chroma
self-similarity), then stamp the earlier bar's finished notes -- shifted by a
whole number of bars, so the beat grid and therefore the note COLORS line up
exactly -- over the repeat. Identical rhythm, identical colors, recognizable.

Deliberately conservative: it only reuses a run of bars when the match is strong
and long. If nothing matches, the source map is the identity and the chart is
unchanged -- this can never make a chart worse, only lock real repeats together.

Plain array math (STFT chroma + cosine self-similarity), deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

N_CHROMA = 12
BEATS_PER_BAR = 4
# A bar counts as a repeat only when it is (a) very close in chroma to an earlier
# bar, (b) that match is FAR back -- a real section return, not the trivially
# similar neighbour a couple bars ago, (c) it STANDS OUT above how similar this
# bar is to everything else (so a uniformly-samey song matches nothing), and (d)
# the match holds over a long run of bars. All four guard against the
# self-similarity trap where every bar looks like the bar just before it.
SIM_THRESH = 0.65
MIN_LAG_BARS = 8          # the source must be at least this many bars earlier
MIN_REPEAT_BARS = 6       # a repeat must run at least this many bars
STANDOUT = 0.12           # match must beat the bar's typical similarity by this
# Safety net: real verse/chorus structure reuses a minority of the song. If the
# map would reuse more than this, it's the uniform-song collapse -> reuse nothing.
MAX_REUSE_FRAC = 0.55


N_BANDS = 20    # log-spaced spectral bands (timbre + arrangement, not just key)


def _bar_features(mono: np.ndarray, sr: int, bpm: float, beat0: float,
                  duration: float, hop: int = 1024, win: int = 4096):
    """One feature vector per BAR describing its timbre/arrangement.

    Log-spaced spectral bands (40 Hz..Nyquist), log-compressed, averaged over the
    bar, then z-scored PER BAND across the song so the vector encodes how this
    bar's arrangement differs from average (drums in/out, layers) -- which is what
    separates a chorus from a verse even when the chords never change. L2-
    normalized so a dot product is cosine similarity.
    """
    f, t, Z = stft(mono, fs=sr, nperseg=win, noverlap=win - hop,
                   window="hann", boundary=None, padded=False)
    mag = np.abs(Z)
    edges = np.logspace(np.log10(40.0), np.log10(sr / 2.0), N_BANDS + 1)
    bands = np.zeros((N_BANDS, mag.shape[1]))
    for k in range(N_BANDS):
        sel = (f >= edges[k]) & (f < edges[k + 1])
        if sel.any():
            bands[k] = mag[sel].sum(axis=0)
    bands = np.log1p(bands)

    spb = 60.0 / bpm
    barlen = BEATS_PER_BAR * spb
    start = beat0 if beat0 > 0 else 0.0
    n_bars = max(1, int(np.ceil((duration - start) / barlen)))
    F = np.zeros((n_bars, N_BANDS))
    for b in range(n_bars):
        a = start + b * barlen
        lo, hi = int(np.searchsorted(t, a)), int(np.searchsorted(t, a + barlen))
        if hi > lo:
            F[b] = bands[:, lo:hi].mean(axis=1)
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)   # per-band z-score
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def detect_structure(mono: np.ndarray, sr: int, bpm: float, beat0: float,
                     duration: float) -> list[int]:
    """Return a per-bar `source` map: source[b] is the earlier bar that bar b is
    a repeat of, or b itself if it's original. Chains resolve to a root bar.
    """
    if bpm <= 0 or duration <= 0:
        return []
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    F = _bar_features(mono, sr, bpm, beat0, duration)
    n = len(F)
    source = list(range(n))
    if n < 2 * MIN_REPEAT_BARS:
        return source
    S = F @ F.T  # rows are unit vectors -> dot product is cosine similarity
    # Per-bar "background" similarity: how alike this bar is to earlier bars in
    # general. A real repeat must beat this, so uniformly-samey songs (flat
    # background) never trigger.
    bg = np.array([np.median(S[i, :max(1, i - MIN_LAG_BARS)])
                   if i > MIN_LAG_BARS else 0.0 for i in range(n)])

    def matches(i, j):
        return (i - j >= MIN_LAG_BARS and S[i, j] >= SIM_THRESH
                and S[i, j] >= bg[i] + STANDOUT)

    assigned = [False] * n
    for i in range(n):
        if assigned[i]:
            continue
        # Longest standout diagonal run copying from a far-earlier, original bar.
        best_j, best_k = -1, 0
        for j in range(0, i - MIN_LAG_BARS + 1):
            if assigned[j]:
                continue  # sources must be originals, never chained repeats
            k = 0
            while (i + k < n and not assigned[i + k] and not assigned[j + k]
                   and matches(i + k, j + k)):
                k += 1
            if k > best_k:
                best_j, best_k = j, k
        if best_j >= 0 and best_k >= MIN_REPEAT_BARS:
            for u in range(best_k):
                source[i + u] = best_j + u
                assigned[i + u] = True

    reused = sum(1 for b in range(n) if source[b] != b)
    if reused > MAX_REUSE_FRAC * n:
        return list(range(n))  # collapse -> not real structure, reuse nothing
    return source


# Pad-symmetry lane maps: identity, vertical (D<->U), horizontal (L<->R), 180.
_LANE_T = ((0, 1, 2, 3), (0, 2, 1, 3), (3, 1, 2, 0), (3, 2, 1, 0))
# A returning section repeats -- the FIRST return comes back identical (most
# recognizable, so a chorus reads as "the same"), then later returns rotate
# through mirrors (horizontal / vertical / 180) for variety, the way real
# charters bring a chorus back flipped.
_REPEAT_MIRRORS = (0, 2, 1, 3)


def apply_structure(notes, source: list[int], bpm: float, beat0: float):
    """Rebuild the note list so every repeat SECTION carries a bar-shifted copy
    of its source's notes -- rhythm/holds identical (colors stay consistent) but
    lanes MIRRORED so the section reads as 'the chorus, flipped'. Returns a new
    sorted list.
    """
    if not source or not notes or bpm <= 0:
        return notes
    spb = 60.0 / bpm
    barlen = BEATS_PER_BAR * spb
    n = len(source)
    start = beat0 if beat0 > 0 else 0.0

    def root(b: int) -> int:
        seen = 0
        while 0 <= b < n and source[b] != b and seen < n:
            b = source[b]
            seen += 1
        return b

    # Assign a mirror to each contiguous repeat run (all its bars share one, so
    # the section flips coherently); originals stay identity.
    tf_of = [0] * n
    b = run = 0
    while b < n:
        if source[b] != b:
            s = b
            while b < n and source[b] != b:
                b += 1
            for k in range(s, b):
                tf_of[k] = _REPEAT_MIRRORS[run % len(_REPEAT_MIRRORS)]
            run += 1
        else:
            b += 1

    by_bar: dict[int, list] = {}
    for nt in notes:
        b = int((nt.time - start) // barlen) if nt.time >= start else -1
        by_bar.setdefault(b, []).append(nt)

    out = list(by_bar.get(-1, []))              # anything before bar 0
    for b in range(n):
        r = root(b)
        if r == b:
            out.extend(by_bar.get(b, []))
        else:
            shift = (b - r) * barlen
            lane_map = _LANE_T[tf_of[b]]
            for nt in by_bar.get(r, []):
                out.append(type(nt)(nt.time + shift, lane_map[nt.lane],
                                    nt.strength, nt.duration))
    for b, items in by_bar.items():             # anything past the last bar
        if b >= n:
            out.extend(items)
    out.sort(key=lambda x: x.time)
    return out
