r"""build_bs_stamina.py -- STAMINA / CARDIO packs for BEAT SABER.

The Beat Saber analogue of build_ddr_stamina.py: each source song becomes
"<Song> <N> Min" -- one continuous saber FLOW (alternating hands, up/down parity)
that fills the song, with the whole song+chart REPEATED to the target length.
Same five NPS paces (2/3/4/5/6) and the same energy/strain density shaping; the
DDR stamina stream is just re-voiced onto sabers via to_bs_notes.

    python build_bs_stamina.py <out_dir> <minutes> <audio_or_folder>...
    python build_bs_stamina.py out/Stamina 30 --dir path/to/songs/
"""
from __future__ import annotations

import math
import os
import shutil
import sys

import numpy as np
import soundfile as sf

from build_ddr_pack import (resolve_audio, artist_of, find_artwork, collect_paths,
                        cached_grid_energy)
from notesight import SongMeta
from notesight.formats.stepmania import Note
from notesight.formats.beatsaber import write_pack, BS_DIFFICULTY, _normalize_for_loop
from notesight.formats.bs_assemble import assemble_letters, guard_seams
from notesight.formats.bs_letters import pick_alphabet
from notesight.formats.bs_report import write_reports_letters
from notesight.version import BS_GEN_VERSION
from notesight.stamina import one_loop, validate_stream, stream_segments

# Seconds of music before the FIRST block, so it flies in with reaction time
# instead of spawning in your face at beat 0. (Later loops get their ~2s seam
# breather as a natural lead-in; only the very start needs this.)
LEAD_IN = 2.5

# Shared BS voicing for stamina AND gauntlet -- the user wants them identical:
# some diagonals (flow_only OFF), two-hand doubles, and same-saber strings for
# variety. NO same-colour stacks/windows (those stay a normal-charting thing).
FLOW_OPTS = {"twohand_rate": 0.45, "same_hand_run": 0.22, "max_run": 5,
             "lr_rate": 0.08}

# Five paces -> Beat Saber's five Standard slots (write_pack maps Beginner..Challenge
# onto Easy..ExpertPlus). Calibrated to the real-library MEDIAN block-NPS per label
# (Easy~2.6, Normal~3.2, Hard~4.1, Expert~5.5, Expert+~7.2): we skip the 2-NPS floor
# (too slow to bother playing) and top out at 7 to match the Expert+ median, giving a
# clean 3/4/5/6/7 progression.
STAMINA_TIERS = [
    (3.0, "Beginner"),
    (4.0, "Easy"),
    (5.0, "Medium"),
    (6.0, "Hard"),
    (7.0, "Challenge"),
]

# PEAK CAP (BS only -- DDR steps are frozen & the user doesn't feel the spike there):
# the strain model's density arcs were bursting to 1.7-2.5x the tier target in any 1s
# window (an "Easy" briefly threw 7.6 blocks/s). Cap each tier's peak-1s BLOCK-NPS at
# target + this headroom (~the next tier up), so the busiest second of a Normal feels
# like a Hard, not an Expert+. The floor stays free -- breakdowns still drop low.
PEAK_HEADROOM = 1.5


def _write_looped_egg(unit, sr, n_loops, dst):
    """Normalize the loop unit with the COMMON robust-peak + true-peak method, then stream it
    n_loops times into the OGG (.egg) -- tiling repeats identical samples, so making the short
    UNIT clip-free makes the whole (30-45 min) egg clip-free without holding it in RAM or reading
    it back. Same normalization as the regular per-song eggs; no soft-clip tanh 'scuff'."""
    unit = _normalize_for_loop(unit, sr)          # common robust-peak + true-peak (on the UNIT)
    ch = 1 if unit.ndim == 1 else unit.shape[1]
    with sf.SoundFile(dst, "w", samplerate=sr, channels=ch, format="OGG",
                      subtype="VORBIS") as out:
        for _ in range(n_loops):                  # tile the already-safe unit (no per-block tanh)
            for i in range(0, len(unit), 65536):
                out.write(unit[i:i + 65536].astype(np.float32))


def _song_seed(audio_path):
    """Stable per-song seed (crc32 of the song name) -> the song commits to one alphabet the
    same way on every rebuild. (hash() is process-randomized, so don't use it.)"""
    import zlib
    return zlib.crc32(os.path.splitext(os.path.basename(audio_path))[0].encode("utf-8"))


def _gen_one_loops(bpm, beat0, e_times, energy, loop_dur, song_dur, song_seed=0):
    """Per-tier SINGLE-loop voicing -- the exact chart a stamina song tiles. Now voiced from
    the LETTER VOCABULARY: the song commits to ONE alphabet (seeded per song) and every tier
    composes its phrases from those letters, so a song has a consistent feel and different songs
    differ. Returns {slot: (one_blocks, loop_times, rocks)}."""
    alpha = pick_alphabet(song_seed)                 # committed for the whole song

    def gen_one(target):
        unit, _ = one_loop(bpm, beat0, e_times, energy, loop_dur, song_dur,
                           target_nps=target)
        return [Note(t, lane) for t, lane in unit]
    out = {}
    for nps, slot in STAMINA_TIERS:
        _bd, rank, _njs = BS_DIFFICULTY[slot]
        cap = nps + PEAK_HEADROOM
        base = gen_one(nps)
        # letters turn some slots into doubles (extra blocks); measure that inflation and
        # regen at a reduced base rate so the GAME's block-NPS still lands on the tier target.
        infl = len(assemble_letters([n.time for n in base], bpm, seed=rank,
                                    peak_cap=cap, alphabet=alpha)[0]) / max(1, len(base))
        if infl > 1.02:
            base = gen_one(nps / infl)
        btimes = [n.time for n in base]
        one, t2l = assemble_letters(btimes, bpm, seed=rank, peak_cap=cap, alphabet=alpha)
        out[slot] = (one, btimes, t2l)
    return out


def one_play(audio_path):
    """Everything for ONE play of a song: the padded loop-audio unit + its per-tier
    single-loop voicing. A single stamina song TILES this; the mashup CONCATENATES one
    of these per song. Returns (bpm, loop_dur, sr, unit, {slot: one_blocks})."""
    data, sr = sf.read(audio_path, dtype="float32")
    bpm, beat0, e_times, energy, song_dur = cached_grid_energy(audio_path)
    measure = 4 * 60.0 / bpm
    loop_dur = math.ceil(song_dur / measure) * measure
    loop_samples = int(round(loop_dur * sr))
    if len(data) < loop_samples:
        pad = (loop_samples - len(data),) + data.shape[1:]
        unit = np.concatenate([data, np.zeros(pad, dtype=data.dtype)])
    else:
        unit = data[:loop_samples]
    loops = _gen_one_loops(bpm, beat0, e_times, energy, loop_dur, song_dur,
                           _song_seed(audio_path))
    return bpm, loop_dur, sr, unit, {slot: v[0] for slot, v in loops.items()}


def build_song(audio_path, out_dir, minutes, label=""):
    folder = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    title = f"{stem} - Stamina" + (f" {label}" if label else "")  # label distinguishes
    artist = artist_of(folder, audio_path)               # e.g. "45" -> "Boogie - Stamina 45"
    print(f"\n=== {title}  [{artist}]  (Beat Saber) ===")

    data, sr = sf.read(audio_path, dtype="float32")
    bpm, beat0, e_times, energy, song_dur = cached_grid_energy(audio_path)

    measure = 4 * 60.0 / bpm
    loop_dur = math.ceil(song_dur / measure) * measure     # whole measures + breather
    n_loops = max(1, math.ceil(minutes * 60.0 / loop_dur))
    total = n_loops * loop_dur
    print(f"  {song_dur:.0f}s @ {bpm:.1f} BPM -> loop {loop_dur:.0f}s x {n_loops} "
          f"= {total/60:.1f} min")

    # Loop unit padded to whole measures (keeps the ~2s seam breather, as on DDR).
    loop_samples = int(round(loop_dur * sr))
    if len(data) < loop_samples:
        padshape = (loop_samples - len(data),) + data.shape[1:]
        unit = np.concatenate([data, np.zeros(padshape, dtype=data.dtype)])
    else:
        unit = data[:loop_samples]

    # Pre-write the looped egg so write_pack reuses it. Skip if it already exists
    # (chart-only rebuilds don't need to re-encode the 30-min audio -- delete the
    # egg to force a re-encode after changing a song's source audio).
    title_safe = "".join(c for c in title if c.isalnum() or c in "-_() ").strip()
    song_dir = os.path.join(out_dir, title_safe or "Untitled")
    os.makedirs(song_dir, exist_ok=True)
    egg_path = os.path.join(song_dir, "song.egg")
    if os.path.isfile(egg_path) and os.path.getsize(egg_path) > 100:
        print("    (reusing existing song.egg)")
    else:
        _write_looped_egg(unit, sr, n_loops, egg_path)

    # One chart per NPS tier, voiced by the learned transition model. The model
    # turns some events into two-hand DOUBLES (2 blocks), which the game counts as 2,
    # so we measure that inflation and regenerate at a reduced base event-rate --
    # making the GAME's block-NPS land on the tier target (fixes the count bug).
    #
    # PATTERN REUSE (#3): the audio is one loop repeated n_loops times, so we voice
    # ONE loop with the model and TILE that exact voicing across every loop. Each
    # repeat of the (identical) music then gets the identical block pattern -- true
    # phrasing/repetition, instead of the model re-improvising each loop differently.
    loop_beats = loop_dur * bpm / 60.0
    lead_beats = LEAD_IN * bpm / 60.0

    def _tile(one):
        # tile the ONE-loop voicing across every loop; lead-in only drops blocks in the
        # first LEAD_IN seconds of loop 0 (later loops get their ~2s seam breather).
        tiled = []
        for k in range(n_loops):
            off = k * loop_beats
            for b in one:
                nb = dict(b)
                nb["_time"] = round(b["_time"] + off, 5)
                tiled.append(nb)
        tiled = [b for b in tiled if b["_time"] >= lead_beats]
        tiled.sort(key=lambda d: d["_time"])
        return tiled

    # Pull one loop per tier from the shared "box" and TILE it across the repeated audio.
    # (The mashup pulls the SAME per-song loops and concatenates ~8 of them instead.)
    loops = _gen_one_loops(bpm, beat0, e_times, energy, loop_dur, song_dur,
                           _song_seed(audio_path))
    notes_by_diff = {}
    report_charts = []                          # (diff, bpm, whole-song blocks, whole-song time2label)
    for nps, slot in STAMINA_TIERS:
        one, btimes, t2l = loops[slot]
        # tile the loop, then clean the WHOLE song's tile-wrap seams (loop end -> next loop start)
        # the correct-by-construction way (benign resolution + validate), not the old guard.
        tiled = guard_seams(_tile(one), bpm, None)
        notes_by_diff[slot] = tiled
        # WHOLE-SONG label map: tile the loop's time->phrase labels across every loop, so the
        # report's timeline covers the entire song, every note (not just one loop).
        whole = {}
        for k in range(n_loops):
            off = k * loop_beats
            for tb, lbl in t2l.items():
                whole[round(tb + off, 5)] = lbl
        report_charts.append((slot, bpm, tiled, whole))
        blocks = len(tiled)
        print(f"    {slot:<10} ~{nps:.0f} nps  {len(btimes):>5} loop-events -> "
              f"{blocks} blocks ({blocks/total:.2f} block/s)")

    meta = SongMeta(title=title, artist=artist, audio_path="",
                    bpm=bpm, beat0=beat0, duration=total)
    out = write_pack(notes_by_diff, meta, out_dir,
                     cover_src=find_artwork(folder, stem) or "",
                     note_opts={"raw": True})
    write_reports_letters(out, title, BS_GEN_VERSION, report_charts)
    print(f"  wrote {out}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    out_dir, minutes = argv[0], int(argv[1])
    rest = argv[2:]
    label = ""
    if "--label" in rest:                             # title suffix (flat CustomLevels)
        i = rest.index("--label")
        label = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    os.makedirs(out_dir, exist_ok=True)
    validate_stream()
    audios = []
    for s, _ in collect_paths(rest):
        audio = resolve_audio(s)
        if audio:
            audios.append(audio)
        else:
            print(f"!! skip (no audio): {s}", file=sys.stderr)
    # Songs are independent (own folder/audio), and the slow part is the per-song 30/45-min
    # egg ENCODE -- so build them in PARALLEL across cores. Set BS_STAMINA_JOBS=1 to force
    # serial (e.g. for debugging). One build_song per worker process.
    jobs = int(os.environ.get("BS_STAMINA_JOBS", 0)) \
        or max(1, min((os.cpu_count() or 2) - 1, len(audios)))
    ok = 0
    if jobs <= 1 or len(audios) <= 1:
        for audio in audios:
            try:
                build_song(audio, out_dir, minutes, label=label)
                ok += 1
            except Exception as e:
                print(f"!! FAILED {audio}: {e}", file=sys.stderr)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut = {ex.submit(build_song, a, out_dir, minutes, label): a for a in audios}
            for f in as_completed(fut):
                try:
                    f.result()
                    ok += 1
                except Exception as e:
                    print(f"!! FAILED {fut[f]}: {e}", file=sys.stderr)
    print(f"\nDone: {ok} Beat Saber stamina song(s) -> {out_dir}  (x{jobs} parallel)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
