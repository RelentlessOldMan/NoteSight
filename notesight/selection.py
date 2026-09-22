"""selection.py -- note selection.

select(onsets, difficulty) decides which detected onsets become notes: it sees
the whole file at once (offline) and keeps ~target_nps of them, ranked by
strength + energy + local density, so notes BURST in loud/exciting passages and
REST in calm ones. Returns a time-sorted sublist of the input OnsetEvents.
"""
from __future__ import annotations

import bisect

import numpy as np

from .onsets import OnsetEvent
from .difficulty import Difficulty


# Burstiness: a flat "strongest-first, spaced by min_interval" selection turns a
# song into an even stream. Real charts BURST where the music is EXCITING (the
# chorus/drop) and REST where it's calm. Local onset density alone gets this
# wrong -- a fast-hat verse looks "busy" while a big chorus (fewer, stronger
# hits) looks sparse -- so we drive density mainly by ENERGY (the loud/exciting
# parts) with density as a secondary pull to keep fast runs intact. High-energy
# onsets both rank higher AND may pack tighter (down to FLOOR_FRAC of the
# min-interval); quiet ones must space out (up to REST_STRETCH), so rests land in
# the calm bits, not the chorus. Total count is unchanged -> same avg difficulty.
BURST_WIN = 0.7        # seconds; window for measuring local onset density
ENERGY_WEIGHT = 0.55   # how strongly loud/exciting passages attract notes
DENS_WEIGHT = 0.2      # secondary: keep dense onset runs from being thinned
FLOOR_FRAC = 0.72      # a loud burst may pack to this fraction of min_interval
REST_STRETCH = 1.5     # a quiet passage must space to this multiple of it
ABS_FLOOR = 0.07       # hard min seconds between two notes (backfill never goes below this)


def _e_at(e_times, energy, times):
    """Energy (0..1) sampled at each onset time; 0.5 everywhere if unavailable."""
    if e_times is None or energy is None or len(e_times) == 0:
        return np.full(len(times), 0.5)
    return np.interp(times, e_times, energy)


def _select_offline(onsets: list[OnsetEvent], diff: Difficulty,
                    e_times=None, energy=None) -> list[OnsetEvent]:
    """Keep ~target_nps of onsets, but as BURSTS (loud/exciting passages) and
    RESTS (calm ones) rather than an even stream -- scoring onsets on
    strength + energy + local density, and letting loud runs pack tighter than
    the min-interval floor while quiet ones space out."""
    if not onsets:
        return []
    ev = sorted(onsets, key=lambda o: o.time)
    times = [o.time for o in ev]
    span = times[-1] - times[0]
    target_count = max(1, round(diff.target_nps * span)) if span > 0 else len(ev)
    # Keep-all shortcut ONLY when there's no peak cap to enforce -- otherwise a sparse song
    # with one dense burst would return that burst uncapped. With a cap, fall through to the
    # greedy so every 1s window is still ceiling-checked (it keeps ~all onsets anyway).
    if target_count >= len(ev) and diff.peak_nps >= 90:
        return ev

    # Local onset density: how many onsets sit within +-BURST_WIN of each one.
    dens = [bisect.bisect_right(times, t + BURST_WIN)
            - bisect.bisect_left(times, t - BURST_WIN) for t in times]
    maxd = max(dens) or 1
    en = _e_at(e_times, energy, times)          # per-onset energy, 0..1
    mi = diff.min_interval
    order = sorted(range(len(ev)),
                   key=lambda i: (ev[i].strength + ENERGY_WEIGHT * en[i]
                                  + DENS_WEIGHT * (dens[i] / maxd)),
                   reverse=True)

    placed: list[float] = []
    keep = [False] * len(ev)
    n = 0
    for i in order:
        if n >= target_count:
            break
        t = times[i]
        # Floor is tight in loud parts (bursts), stretched in quiet parts (rests).
        floor = mi * (FLOOR_FRAC + (1.0 - en[i]) * (REST_STRETCH - FLOOR_FRAC))
        j = bisect.bisect_left(placed, t)
        near = min((abs(t - placed[k]) for k in (j - 1, j) if 0 <= k < len(placed)),
                   default=1e9)
        if near < floor:
            continue
        # PEAK CAP: don't let any 1-second window exceed the tier's peak-nps ceiling. Since
        # we place strongest-first, rejecting a note here thins the burst to the cap while
        # KEEPING the strongest hits -- caps "Voltage" without touching average density.
        if diff.peak_nps < 90 and _would_exceed_peak(placed, t, diff.peak_nps):
            continue
        bisect.insort(placed, t)
        keep[i] = True
        n += 1

    # DDR presets backfill to hit target_nps exactly (consistent density across songs); Beat
    # Saber leaves diff.backfill False (its targets are pre-compensated) so it stops here.
    if not diff.backfill:
        return [ev[i] for i in range(len(ev)) if keep[i]]

    # BACKFILL to target_count. The energy floor can starve the average below target on
    # dense / high-contrast songs (loud parts pack out at the floor, quiet parts get
    # stretched), so a tier lands well under target_nps AND its density drifts song to song.
    # Add the strongest still-unplaced onsets with a progressively tighter UNIFORM floor --
    # down to an absolute playability minimum -- until we reach target_count. peak_nps still
    # caps every 1s burst, so this fills toward the target average without exceeding the
    # tier's ceiling. Net: each tier reliably hits target_nps and stays consistent across
    # songs (only genuinely onset-sparse songs fall short, which is correct).
    for mult in (0.6, 0.45, 0.3):
        if n >= target_count:
            break
        fl = max(ABS_FLOOR, mi * mult)
        for i in order:
            if n >= target_count:
                break
            if keep[i]:
                continue
            t = times[i]
            j = bisect.bisect_left(placed, t)
            near = min((abs(t - placed[k]) for k in (j - 1, j) if 0 <= k < len(placed)),
                       default=1e9)
            if near < fl:
                continue
            if diff.peak_nps < 90 and _would_exceed_peak(placed, t, diff.peak_nps):
                continue
            bisect.insort(placed, t)
            keep[i] = True
            n += 1
    return [ev[i] for i in range(len(ev)) if keep[i]]


def _would_exceed_peak(placed: list[float], t: float, cap: float) -> bool:
    """True if inserting `t` would push any 1-second window over `cap` notes. A window that
    contains t can always be anchored at a note in [t-1, t], so we check those left edges
    (plus t itself); each candidate window's count includes the new t."""
    lo = bisect.bisect_left(placed, t - 1.0)
    hi = bisect.bisect_right(placed, t)               # notes at/left of t, within 1s
    for a in [t] + placed[lo:hi]:                      # candidate window left edges
        cnt = (bisect.bisect_right(placed, a + 1.0)
               - bisect.bisect_left(placed, a)) + 1    # +1 for the not-yet-inserted t
        if cnt > cap:
            return True
    return False


def select(onsets: list[OnsetEvent], difficulty: Difficulty,
           e_times=None, energy=None) -> list[OnsetEvent]:
    """Pick which onsets become notes. The energy envelope (optional) steers
    WHERE density concentrates (bursts in loud passages, rests in calm ones)."""
    return _select_offline(onsets, difficulty, e_times, energy)


def stream_fill(selected: list[OnsetEvent], all_onsets: list[OnsetEvent],
                diff: Difficulty, bpm: float, beat0: float,
                e_times=None, energy=None) -> list[OnsetEvent]:
    """PRO tiers only: add synthetic grid-aligned notes in the HOTTEST passages so a
    tier can climb past the song's real onset ceiling (a human streaming 16ths
    through a drop). Onset selection runs first; this tops it up toward target_nps.

    The gate is the tier's energy PERCENTILE -- fill never fires in calm passages, so
    the loud/quiet contrast survives. Candidates land on the beat subdivision grid,
    skip any cell already covered by a selected note (anchors stay), and are added
    HOTTEST-first until target_count is hit, honoring the spacing floor + peak cap.
    Filled notes get LOW strength (the jump pass skips them -> streams stay
    single-note rolls) and a brightness interpolated from the real onsets (so the
    voicer's pitch bias still shapes the run). Returns selected + synthetic, sorted.
    """
    if not diff.stream_fill or bpm <= 0 or len(selected) < 2:
        return selected
    if e_times is None or energy is None or len(energy) == 0:
        return selected  # no envelope -> can't gate -> never blind-fill
    times = sorted(o.time for o in selected)
    span = times[-1] - times[0]
    if span <= 0:
        return selected
    target = max(1, round(diff.target_nps * span))
    if len(selected) >= target:
        return selected

    energy_arr = np.asarray(energy, dtype=float)
    thresh = float(np.percentile(energy_arr, diff.stream_energy_pct * 100.0))
    on_t = [o.time for o in all_onsets] or times
    on_b = [o.brightness for o in all_onsets] or [0.0] * len(times)

    step = 60.0 / bpm / (diff.stream_grid / 4.0)   # seconds per subdivision cell
    # Occupancy is by GRID CELL, not time distance: an anchor holds the cell it will
    # SNAP to (round to nearest cell), so a fill one cell over lands on a distinct row
    # even when the anchor sits slightly off-grid. (A time-distance floor wrongly kills
    # those adjacent cells and starves the stream below the grid's capacity.)
    occupied = {int(round((t - beat0) / step)) for t in times}
    placed = list(times)                            # for the peak-cap window check
    t0, t1 = times[0], times[-1]
    k0 = int(np.ceil((t0 - beat0) / step))
    k1 = int(np.floor((t1 - beat0) / step))
    cands = []
    for k in range(k0, k1 + 1):
        if k in occupied:
            continue
        t = beat0 + k * step
        e = float(np.interp(t, e_times, energy))
        if e >= thresh:
            cands.append((e, k, t))
    cands.sort(key=lambda c: c[0], reverse=True)    # hottest first

    synth: list[OnsetEvent] = []
    n = len(placed)
    for e, k, t in cands:
        if n >= target:
            break
        if diff.peak_nps < 90 and _would_exceed_peak(placed, t, diff.peak_nps):
            continue
        bisect.insort(placed, t)
        occupied.add(k)
        synth.append(OnsetEvent(time=t, strength=0.3,
                                brightness=float(np.interp(t, on_t, on_b))))
        n += 1
    if not synth:
        return selected
    return sorted(list(selected) + synth, key=lambda o: o.time)
