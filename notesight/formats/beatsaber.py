"""formats/beatsaber.py -- export a Beat Saber map (also read by NoteNinja).

The Beat Saber .dat map format is a data schema, not a Beat Games asset, so we can
freely write it: a song folder with `info.dat`, one `.dat` per difficulty, an OGG
audio file (named `.egg`), and a cover image. It drops straight into Beat Saber's
CustomLevels, and NoteNinja will read the same format.

The charting translation turns NoteSight's 4-panel notes into saber notes the way
a real map reads AND plays:

  * HAND ALTERNATION  -> left/right sabers alternate (like alternating feet), so
                         you're never asked for two same-hand hits in a row.
  * CUT-DIRECTION PARITY -> each hand alternates down-cut / up-cut. After a down
                         swing your hand is low, so the next is an up swing. This
                         is THE core saber-flow rule; real Expert maps are ~90%
                         up/down for exactly this reason.
  * PITCH -> POSITION -> the note's panel height (a proxy for spectral pitch)
                         places it higher/lower and inner/outer on the 4x3 grid.
  * JUMP -> DOUBLE     -> a NoteSight jump (two panels at once) becomes a two-hand
                         double (one red, one blue), the natural saber equivalent.

Difficulty (density), holds' timing, and repeated-section reuse all ride in from
the shared pipeline -- this layer only decides saber, position and swing.
"""
from __future__ import annotations

import json
import os
import random
import shutil

from ..onsets import OnsetEvent
from ..version import BS_GEN_VERSION
from .base import ChartFormat, SongMeta

# Cut directions (Beat Saber's _cutDirection encoding).
UP, DOWN, LEFT, RIGHT, ANY = 0, 1, 2, 3, 8
UL, UR, DL, DR = 4, 5, 6, 7          # up-left, up-right, down-left, down-right
# A vertical cut's diagonal, angled toward the note's side (left cols lean left).
_DIAG = {UP: (UL, UR), DOWN: (DL, DR)}
# A spatial CIRCLE for same-hand runs: (cut, column-offset-from-hand-centre, row).
# Learned from 104 real Expert maps -- the POSITIONS trace the circle, not just the
# cut angle: down(bottom, centre) -> left(one col LEFT, mid) -> up(top, centre) ->
# right(one col RIGHT, mid) -> back. (Matches measured transitions DN->L dcol-1
# r0->1, L->UP dcol+1 r1->2, R->DN dcol-1 r1->0.) Stepping +1/-1 = left/right circle;
# down/up still alternate so parity stays clean.
_CIRCLE = ((DOWN, 0, 0), (LEFT, -1, 1), (UP, 0, 2), (RIGHT, 1, 1))
# _type: which saber. 0 = red = left hand, 1 = blue = right hand, 3 = bomb.
RED, BLUE = 0, 1

# NoteSight panel -> pitch height 0..1 (Left low .. Right high), the same ordering
# the pattern engine uses to place arrows by spectral brightness.
PITCH = {0: 0.0, 1: 1 / 3, 2: 2 / 3, 3: 1.0}
# A hand's cut angles diagonally only when its pitch LEAPS this much since its
# last note, and even then only 1-in-DIAG_EVERY of them fire -- so diagonals stay
# a sparse accent (~8%, well under real maps' 13%) instead of the wall the flow
# version produced.
# Rolls/circles (down->side, windmills) need TIME: in real maps 91% of down->side
# transitions sit >= 3/4 beat apart and 0% are in a fast (<=1/8 beat) stream. So we
# only place them where the local stream gap is at least this many beats; faster
# streams stay clean up/down + diagonals + doubles.
_ROLL_MIN_GAP = 0.7
DIAG_MOVE = 0.6
DIAG_EVERY = 3

# NoteSight difficulty name -> (Beat Saber difficulty, rank, note-jump speed).
# 1:1 onto Beat Saber's five Standard slots (Easy/Normal/Hard/Expert/ExpertPlus)
# so no two tiers collide onto one .dat.
# NoteSight difficulty -> (Beat Saber difficulty, rank, NJS). NJS = Note Jump Speed: how fast
# blocks fly at you AND how much reaction time you get. It's NOT density -- it's approach speed.
# The game halves the "half-jump duration" while njs*(60/bpm)*hjd > 18, so once NJS gets high the
# reaction window suddenly HALVES (blocks in your face). The old 16/18 crossed that cliff (~577ms
# at 104bpm vs ~1150ms below it). These casual-friendly speeds keep a comfortable ~1s+ reaction
# across our BPM range -- gentle approach, no "whoosh in your face" (a casual-audience default).
BS_DIFFICULTY = {
    "Beginner": ("Easy", 1, 10.0),
    "Easy": ("Normal", 3, 11.0),
    "Medium": ("Hard", 5, 12.0),
    "Hard": ("Expert", 7, 13.0),
    "Challenge": ("ExpertPlus", 9, 14.0),
}

# Target REACTION WINDOW (spawn -> hit, seconds) per tier -- a gradual casual ramp 1.2 -> 0.8s.
# NJS above sets block SPEED; this sets how long you SEE a block coming. The game only gives a
# coarse, BPM-dependent window on its own (it halves the half-jump-duration in discrete steps), so
# we compute _noteJumpStartBeatOffset to hit these exactly on every song -- what ~80% of real maps
# do. Community medians run tighter (Hard 710 / Expert 600 / Expert+ 510 ms) but skew experienced;
# these are deliberately gentler for a casual, no-bar audience. (jump distance stays ~22-24 = readable.)
BS_REACTION = {"Beginner": 1.2, "Easy": 1.1, "Medium": 1.0, "Hard": 0.9, "Challenge": 0.8}


def jump_offset(njs, bpm, target_rt):
    """The _noteJumpStartBeatOffset (in beats) that lands the reaction window on `target_rt` seconds.
    Beat Saber starts the half-jump at 4 beats and halves it while njs*(60/bpm)*hjd > 18; we add an
    offset so hjd*(60/bpm) == target_rt. Clamped so the final half-jump stays above the 0.25-beat floor."""
    if bpm <= 0 or njs <= 0:
        return 0.0
    num = 60.0 / bpm
    hjd = 4.0
    while njs * num * hjd > 18.0:
        hjd /= 2.0
    offset = target_rt / num - hjd
    if hjd + offset < 0.25:
        offset = 0.25 - hjd
    return round(offset, 3)


def _group_by_time(notes):
    """Group notes onto shared onset times (a jump = two notes at one time)."""
    groups: dict[float, list] = {}
    for n in notes:
        groups.setdefault(round(n.time, 4), []).append(n)
    return [groups[t] for t in sorted(groups)]


def _col(hand: int, pitch: float) -> int:
    """Column for a hand: red on the left half, blue on the right; higher pitch
    sits inner. (Row is chosen in to_bs_notes so blocks don't jump around.)"""
    if hand == RED:
        return 1 if pitch >= 0.5 else 0
    return 2 if pitch >= 0.5 else 3


def to_bs_notes(notes, bpm: float, beat0: float = 0.0,
                flow_only: bool = False, double_rate: float = 0.0,
                twohand_rate: float = 0.0, same_hand_run: float = 0.0,
                max_run: int = 3, reset_gap: float = 2.0,
                lr_rate: float = 0.0) -> list[dict]:
    """Translate NoteSight notes into Beat Saber `_notes` -- tuned from a real-map
    study + on-hands play feedback.

      * MOSTLY UP/DOWN slices (parity), so you swing in place, not reach around.
      * BLOCKS STAY LOW and oscillate ~ONE row (up-cut bottom, down-cut middle) --
        real maps are bottom-row heavy; no big top<->bottom jumps.
      * DIAGONALS are RARE accents -- only on a big melodic leap.
      * OCCASIONAL LEFT/RIGHT slice on a repeated pitch, for variety (never two in
        a row; keeps the pendulum since a horizontal leaves the hand mid-height).
      * DOUBLES are symmetric vertical mirrors.
      * repeated-chorus patterns carry over from the shared pipeline notes.

    `flow_only` (stamina/cardio): suppress diagonals entirely -> a smooth up/down
    stream (with the sparse L/R accent kept) for a comfortable non-stop flow.
    `double_rate`: fraction of straight up/down swings promoted to a SAME-COLOUR
    stack/window/tower -- two (or three) same-hand blocks in one column hit with a
    single swing (real Expert maps ~5.5%). Deterministic (seeded).
    `twohand_rate`: fraction of single notes that ALSO spawn the other hand -> a
    two-hand DOUBLE (red+blue at once, mirrored). `same_hand_run`: chance to keep
    the SAME hand instead of alternating (up to max_run in a row) -> same-saber
    strings/windmills. These are the stamina/gauntlet variety knobs.
    """
    rng = random.Random(0x57ABE1 ^ len(notes))     # deterministic per chart
    parity = {RED: DOWN, BLUE: DOWN}
    last_pitch = {RED: 0.0, BLUE: 1.0}
    horiz_run = {RED: 0, BLUE: 0}
    horiz_gate = {RED: 0, BLUE: 0}
    diag_gate = {RED: 0, BLUE: 0}
    last_hand = BLUE
    run_len = 0
    last_beat = -1e9
    spin = 1                                # flips each run -> alternating circles
    spin_dir = {RED: 1, BLUE: -1}
    spin_phase = {RED: 0, BLUE: 0}          # index into _CIRCLE
    last_col = {RED: 1, BLUE: 2}
    out: list[dict] = []

    def add(hand, pitch, beat, cut_dir, row, flip, col=None):
        c = _col(hand, pitch) if col is None else col
        out.append({"_time": round(beat, 5), "_lineIndex": c,
                    "_lineLayer": row, "_type": hand, "_cutDirection": cut_dir})
        last_pitch[hand] = pitch
        last_col[hand] = c
        if flip:
            parity[hand] = UP if parity[hand] == DOWN else DOWN

    def add_stack(hand, pitch, beat, cut):
        """A same-colour multi-block hit with ONE swing (one parity flip): a stack
        (adjacent rows), window (rows 0 & 2), or tower (all three), in one column
        along the up/down swing axis."""
        col = _col(hand, pitch)
        k = rng.random()
        rows = ([0, 1] if rng.random() < 0.5 else [1, 2]) if k < 0.62 else \
               ([0, 2] if k < 0.88 else [0, 1, 2])          # stack / window / tower
        for r in rows:
            out.append({"_time": round(beat, 5), "_lineIndex": col,
                        "_lineLayer": r, "_type": hand, "_cutDirection": cut})
        last_pitch[hand] = pitch
        parity[hand] = UP if parity[hand] == DOWN else DOWN

    for group in _group_by_time(notes):
        beat = group[0].time * bpm / 60.0
        # PLAYER RESET: after a real gap, the NEXT hand (RED, since last=BLUE) leads
        # with a DOWN-cut, but set the hands ANTI-PHASE (other = UP) -- syncing both
        # to DOWN makes them fall into awkward same-direction pairs/doubles out of the
        # rest (which is exactly the "double down" problem).
        gap_beats = beat - last_beat
        if gap_beats >= reset_gap:
            parity[RED], parity[BLUE] = DOWN, UP
            last_hand = BLUE
            run_len = 0
        last_beat = beat
        slow = gap_beats >= _ROLL_MIN_GAP     # rolls/circles only where there's time
        if len(group) >= 2:
            g = sorted(group, key=lambda n: PITCH[n.lane])
            # A NoteSight jump -> two-hand double, but each hand keeps ITS OWN parity
            # so neither is forced into a double-directional (a dance note when the
            # hands are anti-phase). Vision-safe rows.
            for hand, lane in ((RED, g[0].lane), (BLUE, g[-1].lane)):
                c = parity[hand]
                p = PITCH[lane]
                r = 1 if (c == DOWN and _col(hand, p) in (0, 3)) else 0
                add(hand, p, beat, c, r, flip=True)
                horiz_run[hand] = 0
            last_hand = BLUE
            continue

        # Usually alternate hands; occasionally keep the same hand for a short
        # same-saber string (a one-hand up/down windmill), never past max_run.
        if slow and same_hand_run and run_len < max_run and rng.random() < same_hand_run:
            hand = last_hand
            run_len += 1
        else:
            hand = RED if last_hand == BLUE else BLUE
            run_len = 0
        last_hand = hand
        pitch = PITCH[group[0].lane]
        d = pitch - last_pitch[hand]

        # CIRCULAR same-hand run: trace a SPATIAL circle (_CIRCLE) so both the saber
        # angle AND the block positions arc around -- the fun "left circle / right
        # circle" windmill, placed the way real maps do it (positions move along the
        # arc, NOT frozen on one column). Down/up still alternate -> clean parity.
        if run_len >= 1:
            if run_len == 1:                    # entering a run: seed phase, flip dir
                spin = -spin
                spin_dir[hand] = spin
                spin_phase[hand] = 0 if parity[hand] == UP else 2
            spin_phase[hand] = (spin_phase[hand] + spin_dir[hand]) % 4
            cut, dcol, r = _CIRCLE[spin_phase[hand]]
            centre = 1 if hand == RED else 2
            add(hand, pitch, beat, cut, r, flip=False,
                col=min(3, max(0, centre + dcol)))
            parity[hand] = UP if cut == DOWN else DOWN if cut == UP else parity[hand]
            horiz_run[hand] = 0
            continue

        # LEFT/RIGHT ROLL between a down and an up (parity==UP), placed the way real
        # maps do: the side moves ONE column in its OWN direction, to row 1 -- never
        # the same column as the down (that's the goofy one). flip=False keeps the up.
        if slow and parity[hand] == UP and horiz_run[hand] == 0 and rng.random() < lr_rate:
            horiz_run[hand] = 1
            centre = 1 if hand == RED else 2
            if hand == RED:
                add(hand, pitch, beat, LEFT, 1, flip=False, col=max(0, centre - 1))
            else:
                add(hand, pitch, beat, RIGHT, 1, flip=False, col=min(3, centre + 1))
            continue
        horiz_run[hand] = 0

        # Rare diagonal on a big leap; otherwise a straight up/down slice.
        cut = parity[hand]
        side = 1 if d > DIAG_MOVE else -1 if d < -DIAG_MOVE else 0
        if flow_only:                       # stamina: pure up/down, no angles
            side = 0
        elif side != 0:                     # only 1-in-DIAG_EVERY leaps angle
            diag_gate[hand] += 1
            if diag_gate[hand] % DIAG_EVERY != 0:
                side = 0
        # Occasionally promote a straight swing to a same-colour stack/window/tower.
        if double_rate and side == 0 and rng.random() < double_rate:
            add_stack(hand, pitch, beat, cut)
            continue
        cut_dir = cut if side == 0 else _DIAG[cut][0 if side < 0 else 1]
        # VISION-SAFE + BOTTOM-HEAVY: up-cuts + inner down-cuts on the bottom row;
        # only OUTER (col 0/3) down-cuts rise a row (centre-middle = vision block).
        row = 1 if (cut == DOWN and _col(hand, pitch) in (0, 3)) else 0
        add(hand, pitch, beat, cut_dir, row, flip=True)

        # Two-hand doubles: spawn the other hand at the same time on ITS OWN parity
        # (always clean -- dance note when anti-phase, same-direction when in-phase).
        # Geometry from real maps: up sits row 2 / down row 0, and 48% are ADJACENT
        # (blocks touching), the rest spread to the edge.
        if (twohand_rate and side == 0 and run_len == 0
                and rng.random() < twohand_rate):
            other = BLUE if hand == RED else RED
            oc = parity[other]
            orow = 2 if oc == UP else 0
            if rng.random() < 0.48:                  # adjacent / touching
                ocol = last_col[hand] + (1 if other == BLUE else -1)
            else:                                    # spread to the hand's edge
                ocol = 3 if other == BLUE else 0
            add(other, 1.0 - pitch, beat, oc, orow, flip=True,
                col=min(3, max(0, ocol)))
    out.sort(key=lambda d: d["_time"])
    return out


# Beat Saber songs (and its SFX) are mastered LOUD (~-9..-12 dBFS RMS); tracks at
# streaming loudness sit way under them. Normalize to a target RMS with a tanh
# soft-limiter to a ceiling so the song holds its own against the game.
NORM_TARGET_RMS_DB = -14.0    # was -11: back off ~3 dB so vocals keep headroom + clarity
NORM_CEILING = 0.89          # pre-encode ceiling; the true-peak re-encode below catches any
#                              Vorbis overshoot per-song, so this can stay loud
TRUE_PEAK_MAX = 0.97         # max allowed DECODED peak (post-Vorbis); re-encode down if exceeded
NORM_MAX_GAIN_DB = 16.0
NORM_KNEE = 0.72              # only soft-limit samples above this; the body stays untouched


def _normalize(data):
    import numpy as np
    if data.size == 0:
        return data
    mono = data if data.ndim == 1 else data.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    # ROBUST peak for the gain cap: a hot/clipped master decodes with rare inter-sample
    # overshoots (Hobbit hit 1.49 while its 99.9%ile was 1.14) -- capping gain on the absolute
    # max would drag the whole song quiet. Use the 99.9th percentile and let the soft-knee
    # below tame the handful of samples above it (bounded to the ceiling regardless).
    peak = float(np.percentile(np.abs(data), 99.9))
    if rms < 1e-6 or peak < 1e-6:
        return data
    # PEAK-PROTECTED gain: hit the RMS target, but never push peaks past the ceiling --
    # so we don't rely on saturation and add no distortion to the (exposed) vocals. The
    # old code tanh'd the WHOLE waveform, soft-clipping every sample = the "scuff".
    rms_gain = 10 ** ((NORM_TARGET_RMS_DB - 20 * np.log10(rms)) / 20)
    gain = min(rms_gain, NORM_CEILING / peak, 10 ** (NORM_MAX_GAIN_DB / 20))
    data = data * gain
    # gentle soft-knee ONLY on the rare samples still near the ceiling (transients),
    # leaving everything below NORM_KNEE perfectly linear.
    over = np.abs(data) > NORM_KNEE
    if over.any():
        s = np.sign(data[over]); a = np.abs(data[over])
        span = NORM_CEILING - NORM_KNEE
        data[over] = s * (NORM_KNEE + span * np.tanh((a - NORM_KNEE) / span))
    return data.astype(np.float32)


def _to_ogg(src: str, dst_egg: str) -> bool:
    """Encode audio to loudness-normalized OGG Vorbis (Beat Saber wants OGG,
    named .egg).

    Reads the whole file (safe), normalizes, then writes in blocks -- writing a
    whole multi-minute track to Vorbis in one call overflows libsndfile's stack.
    soundfile reads wav/ogg/flac (keeps stereo); mp3/m4a fall back to the
    ffmpeg-decoded mono from audio_io.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(src, dtype="float32")
        _encode_ogg(data, sr, dst_egg)
        return True
    except Exception:
        pass
    try:  # soundfile couldn't read it (e.g. mp3) -> decode via ffmpeg to mono
        from ..audio_io import load_audio
        mono, sr = load_audio(src)
        _encode_ogg(mono, sr, dst_egg)
        return True
    except Exception:
        return False


def _write_vorbis(data, sr, dst):
    import soundfile as sf
    ch = 1 if data.ndim == 1 else data.shape[1]
    with sf.SoundFile(dst, "w", samplerate=sr, channels=ch,
                      format="OGG", subtype="VORBIS") as out:
        for i in range(0, len(data), 65536):
            out.write(data[i:i + 65536])


def _normalize_for_loop(unit, sr):
    """Return loop-UNIT samples normalized with the SAME robust-peak + true-peak method as
    _encode_ogg, so that streaming the unit (tiled N times) yields a clip-free egg. Tiling just
    repeats identical samples, so the whole egg's decoded peak equals the single unit's -- we
    measure that by encoding the short unit to a temp OGG and iterating the correction. This is
    how stamina/mashup eggs (too long to read back whole) get the common normalization."""
    import numpy as np
    import soundfile as sf
    import tempfile
    data = _normalize(unit)
    tmp = os.path.join(tempfile.gettempdir(), "_ns_loopunit.ogg")
    try:
        for _ in range(4):
            _write_vorbis(data, sr, tmp)
            back, _ = sf.read(tmp, dtype="float32")
            tp = float(np.max(np.abs(back))) if back.size else 0.0
            if tp <= TRUE_PEAK_MAX:
                break
            data = (data * (TRUE_PEAK_MAX * 0.99 / tp)).astype(np.float32)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return data.astype(np.float32)


def _encode_ogg(data, sr, dst):
    """The COMMON egg encoder for every pack: _normalize -> write OGG -> TRUE-PEAK check.
    Vorbis can overshoot near-ceiling content into clipping, so we read the encoded file back
    and, if its DECODED peak exceeds TRUE_PEAK_MAX, pull the whole track down by the exact ratio
    and re-encode ONCE. Result: a clip-free egg at each song's own max loudness, whatever the
    source (clean or hot/clipped). Clean low-variance songs never trip the re-encode."""
    import numpy as np
    import soundfile as sf
    data = _normalize(data)
    for _ in range(4):                       # iterate: Vorbis overshoot isn't linear, so one
        _write_vorbis(data, sr, dst)         # correction can still leave a hair of clipping
        try:
            back, _ = sf.read(dst, dtype="float32")
        except Exception:
            return
        tp = float(np.max(np.abs(back))) if back.size else 0.0
        if tp <= TRUE_PEAK_MAX:
            break
        data = (data * (TRUE_PEAK_MAX * 0.99 / tp)).astype(np.float32)   # aim just under, re-check


def _lighting(bs_notes, seed=0):
    """Procedural lighting synced to the notes, matching real-map stats: mostly
    FADES (the dominant effect ~46%) across all 5 light groups evenly, color flips
    by phrase (~8 beats) with a brief OFF on the flip, and a FLASH on some downbeats.
    Values: blue = on 1 / flash 2 / fade 3; red = on 5 / flash 6 / fade 7."""
    rng = random.Random(0x11 ^ seed)
    times = sorted({round(n["_time"], 4) for n in bs_notes})
    ev = []
    color = 1                                    # 1 = blue family, 5 = red family
    last_flip = -99.0
    groups = (2, 3, 0, 1, 4)                      # L-laser, R-laser, back, ring, center
    for i, t in enumerate(times):
        if t - last_flip >= 8.0:                  # new phrase -> flip colour, pulse off
            color = 5 if color == 1 else 1
            last_flip = t
            ev.append({"_time": t, "_type": rng.choice((0, 1, 4)), "_value": 0})
        flash = (i % 8 == 0) or (abs(t - round(t)) < 0.02 and rng.random() < 0.3)
        val = (color + 1) if flash else (color + 2)   # flash (~12%) | fade
        ev.append({"_time": t, "_type": groups[i % 5], "_value": val})
    return ev


def write_pack(notes_by_diff: dict, meta: SongMeta, out_dir: str,
               cover_src: str = "", note_opts: dict | None = None) -> str:
    """Write a full Beat Saber song folder from {difficulty_name: notes}.

    Returns the folder path. `notes_by_diff` keys are NoteSight difficulty names
    (Beginner..Challenge); each value is that difficulty's note list. `note_opts`
    is forwarded to to_bs_notes (e.g. {'flow_only': True} for stamina charts).
    """
    note_opts = dict(note_opts or {})
    # note_opts={"raw": True}: notes_by_diff already holds finished BS _notes dicts (the letter
    # generator's output). That's how everything is voiced now.
    use_raw = note_opts.pop("raw", False)
    title = "".join(c for c in meta.title if c.isalnum() or c in "-_() ").strip()
    song_dir = os.path.join(out_dir, title or "Untitled")
    os.makedirs(song_dir, exist_ok=True)

    # Audio -> OGG (.egg). Reuse an already-encoded egg (fast chart-only regens);
    # else encode, falling back to a copy if the encode fails.
    song_file = "song.egg"
    egg = os.path.join(song_dir, song_file)
    if os.path.isfile(egg) and os.path.getsize(egg) > 100:
        pass                                    # already encoded -> reuse
    elif not (meta.audio_path and _to_ogg(meta.audio_path, egg)):
        if meta.audio_path and os.path.isfile(meta.audio_path):
            song_file = os.path.basename(meta.audio_path)
            shutil.copyfile(meta.audio_path, os.path.join(song_dir, song_file))

    cover_file = ""
    if cover_src and os.path.isfile(cover_src):
        cover_file = "cover" + os.path.splitext(cover_src)[1].lower()
        shutil.copyfile(cover_src, os.path.join(song_dir, cover_file))

    # One .dat per difficulty + the difficultyBeatmaps entry in info.dat.
    beatmaps = []
    for diff_name, notes in notes_by_diff.items():
        bs_diff, rank, njs = BS_DIFFICULTY.get(diff_name, ("Normal", 5, 13.0))
        dat_name = f"{bs_diff}.dat"
        if use_raw:
            bs_notes = notes                    # already letter-voiced BS _notes dicts
        else:
            bs_notes = to_bs_notes(notes, meta.bpm, meta.beat0, **note_opts)
        # Embed the generator version in the map itself. JSON has no comments, so we
        # tag it as top-level metadata AND inside _customData (the canonical spot the
        # game/editors preserve for custom fields) -- both are ignored by play.
        dat = {"_version": "2.0.0", "_BPMChanges": [],
               "_generatorVersion": f"NoteSight BS {BS_GEN_VERSION}",
               "_events": _lighting(bs_notes, seed=rank),
               "_notes": bs_notes, "_obstacles": [], "_bookmarks": [],
               "_customData": {"_generator": "NoteSight", "_generatorVersion": BS_GEN_VERSION}}
        with open(os.path.join(song_dir, dat_name), "w", encoding="utf-8") as f:
            json.dump(dat, f)
        offset = jump_offset(njs, meta.bpm, BS_REACTION.get(diff_name, 1.0))
        beatmaps.append({
            "_difficulty": bs_diff, "_difficultyRank": rank,
            "_beatmapFilename": dat_name, "_noteJumpMovementSpeed": njs,
            "_noteJumpStartBeatOffset": offset,
        })
    # Beat Saber lists difficulties easiest-first by rank.
    beatmaps.sort(key=lambda b: b["_difficultyRank"])

    info = {
        "_version": "2.0.0",
        "_songName": meta.title, "_songSubName": "",
        "_songAuthorName": meta.artist, "_levelAuthorName": "NoteSight",
        "_beatsPerMinute": round(meta.bpm, 3), "_songTimeOffset": 0,
        "_shuffle": 0, "_shufflePeriod": 0.5,
        "_previewStartTime": round(min(meta.duration * 0.33,
                                       max(0.0, meta.duration - 15.0)), 2),
        "_previewDuration": 10,
        "_songFilename": song_file, "_coverImageFilename": cover_file,
        "_environmentName": "DefaultEnvironment",
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": beatmaps,
        }],
    }
    with open(os.path.join(song_dir, "info.dat"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return song_dir


class BeatSaberFormat(ChartFormat):
    name = "beatsaber"

    def map(self, onsets: list[OnsetEvent], meta: SongMeta):
        # NoteSight feeds already-placed 4-panel notes in via write(); the saber
        # translation happens there. (Kept for the ChartFormat ABC.)
        return onsets

    def write(self, notes, meta: SongMeta, out_dir: str) -> str:
        diff = meta.difficulty.name if meta.difficulty else "Medium"
        return write_pack({diff: notes}, meta, out_dir)
