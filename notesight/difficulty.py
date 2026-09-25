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
    # BACKFILL: after the burst/rest pass, relax the spacing floor to HIT target_nps exactly
    # (so a tier's density is consistent song-to-song instead of drifting low on dense/contrasty
    # songs). DDR presets set this True. Beat Saber leaves it False -- its target_nps are
    # pre-compensated for the undershoot, so backfilling would double-count and over-densify.
    backfill: bool = False
    # STREAM-FILL (pro pack only): onset selection tops out at the song's real onset rate
    # (~6-7 nps on our music). To reach the 7-9+ nps of tech-map "streams", the top pro tiers
    # ADD synthetic grid-aligned notes in the song's HOTTEST passages -- decoupled from onsets,
    # like a human streaming 16ths through a sustained drop. It never fires in calm passages
    # (energy gate), so loud/quiet contrast survives; the filled notes are voiced through the
    # SAME flow engines (DDR foot-alternation / BS parity), so a stream reads as a roll, not a
    # mash. off (0.0) for every normal/DDR/BS tier -- only the pro presets turn it on.
    stream_fill: bool = False
    # Only fill grid cells whose local energy is at/above this PERCENTILE of the song's energy
    # (0.65 = hottest 35%). Higher pro tiers lower this so more of the song streams.
    stream_energy_pct: float = 0.65
    stream_grid: int = 16         # subdivision the fill lands on (8 = 8ths, 16 = 16ths)
    # DDR jump RUNS: max consecutive back-to-back jumps allowed (a "run"). 1 = isolated jumps
    # only (the old behaviour). Hand charts grow this with difficulty (Beginner 1, Easy ~2,
    # Medium ~3, Hard ~4) and it's FUN. The paired rule (in patterns.assign): a jump/run must
    # EXIT into space -- never a tight single right after -- since that tight exit is the
    # awkward "jump wedged in a stream" pattern hand charts almost never do.
    max_jump_run: int = 1
    # Minimum spacing (in BEATS) a jump needs to its neighbours on EACH side, and between
    # consecutive jumps in a run. An 8th (0.5) is too little time to plant/peel both feet on
    # mid/low tiers, so those use a QUARTER (1.0); Hard/Challenge allow the tighter 8th.
    jump_gap_beats: float = 0.55


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
    "beginner": Difficulty("Beginner", 0.50, 1.2, True,  2,      4, False, 0.76, 2.0, peak_nps=3, backfill=True),
    # DDR regular-song ladder (via DDR_TIERS in build_ddr_pack): the slot each preset maps to
    # and its target NPS are -- easy->Beginner 2.0, medium->Easy 2.5, midhard->Medium 3.0,
    # hard->Hard 3.5, expert->Challenge 4.0. backfill=True makes each HIT its target exactly.
    # NOTE: max_jump_run is per DDR *slot* (DDR_TIERS maps preset->slot): easy->Beginner=1,
    # medium->Easy=2, midhard->Medium=3, hard->Hard=4, expert->Challenge=4 (from hand-chart study).
    # jump_gap_beats: Beginner..Medium need a QUARTER (1.0) around jumps (an 8th is too tight for
    # casual play -- user feedback); Hard/Challenge allow the tighter 8th (0.5) + 8th runs.
    "easy":     Difficulty("Easy",     0.34, 2.0, True,  4,      8, False, 0.70, 1.5, peak_nps=4, backfill=True, max_jump_run=1, jump_gap_beats=1.0),
    "medium":   Difficulty("Medium",   0.20, 2.5, True,  6,      8, False, 0.65, 1.0, peak_nps=5, backfill=True, max_jump_run=2, jump_gap_beats=1.0),
    # a NEW tier sitting between Medium and Hard (fills the biggest density gap on the
    # DDR ladder). Params interpolated Medium<->Hard. Used only by build_ddr_pack's
    # DDR_TIERS (regular .sm songs); not part of the Beat Saber or stamina ladders.
    "midhard":  Difficulty("Medium",   0.16, 3.0, True,  7,      8, False, 0.62, 0.85, peak_nps=6, backfill=True, max_jump_run=3, jump_gap_beats=1.0),
    "hard":     Difficulty("Hard",     0.13, 3.5, True,  9,     16, False, 0.60, 0.7, peak_nps=6, backfill=True, max_jump_run=4, jump_gap_beats=0.5),
    "expert":   Difficulty("Expert",   0.09, 4.0, True,  12,    16, False, 0.60, 0.6, peak_nps=7, backfill=True, max_jump_run=4, jump_gap_beats=0.5),
}


# PRO ladder (build_ddr_pack/build_bs_pack --pro): a SEPARATE, harder pack that
# climbs above the regular one. Five tiers 4.0 / 5.5 / 7.0 / 8.5 / 9.0:
#   * pro floor (4.0) is the regular "expert" preset VERBATIM -- pro Beginner IS
#     the regular pack's Expert+ chart, zero change.
#   * pro2 (5.5) stays ONSET-HONEST -- it just pushes selection to the song's real
#     onset ceiling (backfill). Songs too sparse to reach it drop this tier.
#   * pro3/pro4/pro5 (7.0/8.5/9.0) turn on STREAM-FILL: onset selection first, then
#     synthetic grid notes added in the hottest passages up to target. The gate
#     opens wider (lower percentile) and the grid gets finer as the tier climbs, so
#     Medium streams the drops lightly and Challenge streams most of the loud half.
# Jumps stay rare here (a stream is a ROLL, not a wall) -- low jump rate, big gap.
PRO_DIFFICULTIES = {
    #                   name        min_iv  nps  jumps meter  sub  trip  jstr  jgap  peak            backfill / stream
    "pro2": Difficulty("Expert",   0.085, 5.5, True,  13,    16, False, 0.62, 0.7, peak_nps=9,  backfill=True,
                       stream_fill=True, stream_energy_pct=0.68, stream_grid=16),
    "pro3": Difficulty("Expert",   0.075, 7.0, True,  15,    16, False, 0.65, 0.8, peak_nps=11, backfill=True,
                       stream_fill=True, stream_energy_pct=0.55, stream_grid=16),
    "pro4": Difficulty("Expert",   0.065, 8.5, True,  17,    16, False, 0.68, 0.9, peak_nps=14, backfill=True,
                       stream_fill=True, stream_energy_pct=0.42, stream_grid=16),
    "pro5": Difficulty("Expert",   0.060, 9.0, True,  18,    16, False, 0.70, 1.0, peak_nps=16, backfill=True,
                       stream_fill=True, stream_energy_pct=0.30, stream_grid=16),
}
