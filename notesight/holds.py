"""holds.py -- turn sustained notes into hold notes (the freeze arrows).

The rule, in the user's words: a hold goes where there's a long pause between
hits and the sound is *sustaining* to fill it -- a singer holding one note for
seconds, or an instrumental stretch. Instead of leaving dead air, you press and
hold a panel across that stretch; the tail lands when the sustain decays or the
next hit arrives.

So a note becomes a hold when BOTH are true:
  * the gap to the next note is long (a real pause, not part of a run), AND
  * the audio's energy stays up through that gap (a sustain, not silence).

This self-scales with difficulty for free: easy charts are sparse -> many long
gaps -> many holds (just like the reference DDR charts); hard charts are busy ->
few gaps -> holds only at genuine sustains. And because holds only bridge gaps
where nothing else was selected, the held foot is never fighting another note --
it's a clean single (or a double, if the gap opened on a jump).

Applied as a POST-PROCESS on the placed notes, so the foot-flow/pattern engine
stays untouched: we just set each note's `duration` in place.
"""
from __future__ import annotations

import numpy as np

# A hold needs a real pause: the gap to the next note clears both a musical bar
# (beats) and an absolute floor (seconds).
HOLD_MIN_GAP_BEATS = 0.75
HOLD_MIN_GAP_SEC = 0.3
# The sustain must last at least this (else it's a stab, not a hold).
HOLD_MIN_LEN_BEATS = 0.6
# "Sustaining" = energy staying above this fraction of the note's onset energy.
SUSTAIN_FRAC = 0.30
# Release the hold a little before the next note so there's time to step it.
RELEASE_BEATS = 0.5
# Hand-authored DDR charts sit around ~5% holds; cap at this fraction of notes
# (keeping the LONGEST sustains) so a hold-heavy song can't become a freeze slog.
HOLD_MAX_FRAC = 0.06
# The gap must be a genuine musical pause: any real onset inside it (from the
# FULL detection, not our thinned selection) means the music re-attacked, so it
# is NOT a hold -- it's just a sparse chart. This ignores faint onsets below this
# fraction of the loudest onset.
REATTACK_STRENGTH = 0.18
EDGE_MARGIN = 0.12  # ignore onsets this close to the note / the next note


def apply_holds(notes, onsets, energy_times, energy, bpm: float, beat0: float,
                duration: float) -> int:
    """Set `.duration` on notes that sit on a sustained musical pause. Count out.

    A note becomes a hold only when the gap to the next note (a) is long, (b) has
    NO real onset in it (the music actually paused, vs. us just keeping few
    notes), and (c) the energy sustains through it. `onsets` is the FULL onset
    list. Notes sharing a time (a jump) all become holds -> a two-panel hold.
    """
    if not notes or energy_times.size == 0 or bpm <= 0:
        return 0
    spb = 60.0 / bpm
    min_gap = max(HOLD_MIN_GAP_SEC, HOLD_MIN_GAP_BEATS * spb)
    min_len = HOLD_MIN_LEN_BEATS * spb
    release = RELEASE_BEATS * spb
    et, en = energy_times, energy
    # Onset times that count as a "re-attack" (strong enough to break a pause).
    ot = np.array(sorted(o.time for o in onsets
                         if o.strength >= REATTACK_STRENGTH)) if onsets else np.zeros(0)

    def e_at(t: float) -> float:
        i = int(np.searchsorted(et, t))
        return float(en[min(max(i, 0), len(en) - 1)])

    def e_max(a: float, b: float) -> float:
        lo, hi = int(np.searchsorted(et, a)), int(np.searchsorted(et, b))
        return float(en[lo:hi].max()) if hi > lo else e_at(a)

    # Distinct note onset times, in order (a jump shares one time).
    uts = sorted({round(n.time, 4) for n in notes})
    end_for: dict[float, float] = {}
    for j, t0 in enumerate(uts):
        t_next = uts[j + 1] if j + 1 < len(uts) else duration
        if t_next - t0 < min_gap:
            continue
        # Genuine pause? Allow at most ONE stray re-attack in the gap (a sustained
        # note often has a faint pluck under it); more than that = busy, not held.
        a, b = t0 + EDGE_MARGIN, t_next - EDGE_MARGIN
        if b > a and ot.size:
            if int(np.searchsorted(ot, b)) - int(np.searchsorted(ot, a)) > 1:
                continue
        peak = e_max(t0, t0 + 0.15)
        if peak <= 0:
            continue
        thr = SUSTAIN_FRAC * peak
        # The sustain ends where smoothed energy first drops below the threshold.
        lo = int(np.searchsorted(et, t0 + 0.05))
        hi = int(np.searchsorted(et, t_next))
        seg = en[lo:hi]
        below = np.where(seg < thr)[0]
        decay_t = float(et[lo + below[0]]) if below.size else t_next
        end = min(decay_t, t_next - release)
        if end - t0 < min_len:
            continue  # decays too fast -> a stab, keep it a tap
        end_for[t0] = end

    # Cap holds to ~HOLD_MAX_FRAC of notes, keeping the longest sustains, so a
    # song with endless pads doesn't turn into a wall of freezes.
    max_holds = max(1, int(HOLD_MAX_FRAC * len(notes)))
    if len(end_for) > max_holds:
        keep = sorted(end_for.items(), key=lambda kv: kv[1] - kv[0],
                      reverse=True)[:max_holds]
        end_for = dict(keep)

    count = 0
    for n in notes:
        end = end_for.get(round(n.time, 4))
        if end and end > n.time:
            n.duration = end - n.time
            count += 1
    return count


# A hold must end before the next note in its OWN lane (else the two collide on
# one row and the exporter would drop the hold's tail -> an unclosed, invalid
# hold). After holds + structure reuse can create such overlaps, so clamp them.
HOLD_RELEASE = 0.05     # leave this gap before the next same-lane note
HOLD_MIN_KEEP = 0.15    # shorter than this after clamping -> just a tap


def sanitize_holds(notes) -> None:
    """Clamp each hold so it never overlaps the next note in the same lane."""
    by_lane: dict[int, list] = {}
    for n in notes:
        by_lane.setdefault(n.lane, []).append(n)
    for lane_notes in by_lane.values():
        lane_notes.sort(key=lambda n: n.time)
        for a, b in zip(lane_notes, lane_notes[1:]):
            if a.duration > 0 and a.time + a.duration > b.time - HOLD_RELEASE:
                a.duration = b.time - HOLD_RELEASE - a.time
                if a.duration < HOLD_MIN_KEEP:
                    a.duration = 0.0
