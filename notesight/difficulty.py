"""difficulty.py -- difficulty presets (moved out of charter.py).

A Difficulty controls *how dense* the chart is. The primary knob is target_nps
(notes per second): selection keeps roughly that many notes/sec of the strongest
onsets, so density is consistent across songs instead of drifting with each
song's onset count. min_interval caps burst density; allow_jumps adds chords on
strong hits.

Targets are calibrated to hand-authored DDR charts (StepMania 5 "DDR A20" +
5.1 "Won't Stop" packs): Beginner ~1.2 nps ... Challenge ~6.5 nps. The numeric
`meter` here is only a fallback hint -- the exporter derives the real meter from
the chart's radar values via ITG's PredictMeter (see notesight/radar.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Difficulty:
    name: str
    min_interval: float   # minimum seconds between consecutive notes (density cap)
    target_nps: float     # target notes per second (primary density control)
    allow_jumps: bool     # allow 2-note chords on strong onsets
    meter: int = 5        # fallback StepMania meter if radar prediction is unused
    # --- rhythm ceiling: how fine the quantize grid may get, per difficulty ---
    # Low tiers cap at 8ths (all notes land on red 1/4 or blue 1/8 -- the "primary"
    # colors), so easy charts read on-beat instead of a mess of yellow 16ths /
    # green triplets. Finer subdivisions + triplets unlock as difficulty climbs.
    max_subdivision: int = 16     # 4/8/16 -> finest straight grid allowed
    allow_triplets: bool = True   # allow 8th/16th-triplet measures
    # --- jumps: allowed even in easier tiers, but SPARSE (a jump marks a key hit,
    # never a wall). jump_strength = onset salience that earns a jump (lower =
    # more jumps); jump_gap = min seconds between jumps (bigger = rarer, and it
    # guarantees a single note always follows a jump -- no back-to-back jumps).
    jump_strength: float = 0.85
    jump_gap: float = 0.30
    # --- peak cap: max notes in any 1-second window (the "Voltage" ceiling). target_nps
    # sets AVERAGE density; this caps the busiest BURST so a tier can't spike into a
    # "fuck, fail" wall. Default 99 = effectively uncapped (Beat Saber / anything that
    # doesn't set it is unaffected). Real DDR (A20/Won't Stop) peak medians are Medium 6 /
    # Hard 9 / Challenge 13; we run BELOW that on purpose (user chose approachability).
    peak_nps: float = 99.0


# Calibrated to hand-authored DDR A20 charts: triplets are ~never
# used (even on Hard), 16ths only appear from Hard up, jumps sit steady ~6-8% at
# ALL difficulties (yes, even Beginner). So: no triplets anywhere; the finest
# grid climbs 4 -> 8 -> 8 -> 16 -> 16 (Beginner all-quarter like A20's 100% 4th);
# jumps on every tier with a roughly constant rate (uniform strength, gap scaled
# to density).
# peak_nps (busiest-1s ceiling) caps the top tiers so the Medium->Hard->Challenge climb
# stays approachable instead of spiking into a wall: Medium 5 / Hard 6 / Challenge 7.
# Beginner/Easy are naturally below their caps. (Real DDR runs 6/9/13 -- we're gentler.)
DIFFICULTIES = {
    #                     name        min_iv  nps  jumps meter  sub  trip  jstr  jgap  peak
    "beginner": Difficulty("Beginner", 0.50, 1.2, True,  2,      4, False, 0.76, 2.0, peak_nps=3),
    "easy":     Difficulty("Easy",     0.34, 2.2, True,  4,      8, False, 0.70, 1.5, peak_nps=4),
    "medium":   Difficulty("Medium",   0.20, 3.2, True,  6,      8, False, 0.65, 1.0, peak_nps=5),
    "hard":     Difficulty("Hard",     0.13, 4.0, True,  9,     16, False, 0.60, 0.7, peak_nps=6),
    "expert":   Difficulty("Expert",   0.09, 5.0, True,  12,    16, False, 0.60, 0.6, peak_nps=7),
}
