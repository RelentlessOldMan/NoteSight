"""radar.py -- ITG-style chart "shape" values + automatic meter prediction.

Ports the math ITGmania uses to characterize a step chart
(NoteDataUtil::CalculateRadarValues) and to predict a numeric meter from it
(Steps::PredictMeter). Computing these on our OWN output lets NoteSight:
  * write real radar values into the .sm (#NOTES radar field), and
  * derive the difficulty METER from the chart's shape instead of hardcoding it,
    which is exactly how the in-game number is meant to arise.

The five floats (matching ITG order Stream, Voltage, Air, Freeze, Chaos):
  Stream  = taps/sec / 7                 overall average density
  Voltage = peak notes-per-beat in any 8-beat window * avg_bps / 10   burst peak
  Air     = jumps/sec                    how often you hit two panels at once
  Freeze  = holds/sec                    (0 until we emit hold notes)
  Chaos   = off-8th-grid rows/sec * 0.5  rhythmic complexity / syncopation

This is the seam for the future radar-DRIVEN generator: pick target radar values,
invert these equations to a note budget, and let selection/patterns hit them.
"""
from __future__ import annotations

from dataclasses import dataclass

# 12 rows per beat is NoteSight's export grid (48/measure). A note lands on an
# 8th-or-coarser boundary when its row is a multiple of 6 (12/2); anything else
# (16ths, triplets, ...) is a "chaos" row, per ITG's off-8th-grid definition.
ROWS_PER_BEAT = 12
EIGHTH_STEP = ROWS_PER_BEAT // 2

# Steps::PredictMeter coefficients (ITGmania Steps.cpp). Only the five floats
# feed the regression; count categories have zero weight.
_RADAR_COEFF = {"Stream": 10.1, "Voltage": 5.27, "Air": -0.905,
                "Freeze": -1.10, "Chaos": 2.86}
# Difficulty-slot bias. NoteSight's "Expert" maps to ITG's "Challenge" slot.
_SLOT_COEFF = {"Beginner": -0.877, "Easy": -0.877, "Medium": 0.0,
               "Hard": 0.722, "Challenge": 0.722, "Expert": 0.722}


@dataclass
class Radar:
    stream: float
    voltage: float
    air: float
    freeze: float
    chaos: float

    def as_tuple(self):
        return (self.stream, self.voltage, self.air, self.freeze, self.chaos)

    def sm_field(self) -> str:
        """The 5-value radar string for the .sm #NOTES block."""
        return ",".join(f"{v:.3f}" for v in self.as_tuple())


def compute_radar(notes, bpm: float, beat0: float, duration: float,
                  holds: int | None = None) -> Radar:
    """Radar values for a list of Note(time, lane, ...) at a known tempo.

    `duration` is the song length in seconds; `holds` is the hold-note count. If
    left None it is counted from the notes' `duration` field (Freeze radar).
    """
    if duration <= 0 or not notes:
        return Radar(0.0, 0.0, 0.0, 0.0, 0.0)
    if holds is None:
        holds = sum(1 for n in notes if getattr(n, "duration", 0.0) > 1e-3)
    period = 60.0 / bpm

    # Group notes onto rows to count jumps and off-grid (chaos) rows.
    per_row: dict[int, int] = {}
    for n in notes:
        row = int(round((n.time - beat0) / period * ROWS_PER_BEAT))
        per_row[row] = per_row.get(row, 0) + 1

    taps = len(notes)                               # each arrow counts
    jumps = sum(1 for c in per_row.values() if c == 2)
    chaos_rows = sum(1 for row in per_row if row % EIGHTH_STEP != 0)

    # Voltage: peak taps in any 8-beat (= 8*period sec) sliding window.
    win = 8.0 * period
    times = sorted(n.time for n in notes)
    peak = lo = 0
    for hi in range(len(times)):
        while times[hi] - times[lo] > win:
            lo += 1
        peak = max(peak, hi - lo + 1)
    avg_bps = bpm / 60.0

    return Radar(
        stream=taps / duration / 7.0,
        voltage=(peak / 8.0) * avg_bps / 10.0,
        air=jumps / duration,
        freeze=holds / duration,
        chaos=chaos_rows / duration * 0.5,
    )


def predict_meter(radar: Radar, slot: str) -> int:
    """ITG's Steps::PredictMeter regression: chart shape -> numeric meter.

    Radar values are clamped to [0,1] first (as GrooveRadar does for display):
    the regression was fit on real charts whose values stay in that range, and
    the negative Stream*Voltage / Chaos^2 interaction terms otherwise make the
    meter non-monotonic for our unusually syncopated (high-Chaos) charts.
    """
    s, v, a, f, c = (min(1.0, max(0.0, x)) for x in radar.as_tuple())
    m = (0.775
         + _RADAR_COEFF["Stream"] * s + _RADAR_COEFF["Voltage"] * v
         + _RADAR_COEFF["Air"] * a + _RADAR_COEFF["Freeze"] * f
         + _RADAR_COEFF["Chaos"] * c
         + _SLOT_COEFF.get(slot, 0.0)
         - 6.35 * s * v
         - 2.58 * c * c)
    return max(1, int(round(m)))
