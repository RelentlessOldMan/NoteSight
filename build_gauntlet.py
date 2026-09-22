r"""build_gauntlet.py -- PROGRESSIVE "gauntlet" charts (a rhythm-game beep test).

Same looped song as stamina, but the note density RAMPS in chunks: ~30s at 2.0
NPS, then 2.5, then 3.0 ... up past 15.5, until you fail (or survive 15.5+ = win).
The SONG IS NEVER SPED UP -- only the number of notes grows. Each step-up lands on
a breather GAP snapped to a musical lull (a low-energy measure boundary) so it
feels good, beep-test style. Builds BOTH games in one shot, under <out_dir>:
    <out_dir>\DDR\<Song> - Gauntlet\         (StepMania .sm)
    <out_dir>\BeatSaber\<Song> - Gauntlet\   (Beat Saber map)

    python build_gauntlet.py <out_dir> <audio_or_folder>...
    python build_gauntlet.py out/Gauntlet --dir path/to/songs/
"""
from __future__ import annotations

import bisect
import math
import os
import sys

import numpy as np
import soundfile as sf

from build_ddr_pack import (resolve_audio, artist_of, find_artwork, save_png,
                        collect_paths, cached_grid_energy)
from build_bs_stamina import _write_looped_egg, LEAD_IN, FLOW_OPTS
from notesight import SongMeta
from notesight.radar import compute_radar, predict_meter
from notesight.formats.stepmania import Note, notes_to_measures, _sanitize
from notesight.formats.beatsaber import write_pack
from notesight.stamina import build_stream

NPS_START, NPS_END, NPS_STEP = 2.0, 15.5, 0.5
SEG_SEC = 30.0        # roughly this long per level (snapped to a lull)
GAP_BEATS = 4.0       # breather (no notes) at each step-up


def _levels():
    out, n = [], NPS_START
    while n < NPS_END + 1e-9:
        out.append(round(n, 3))
        n += NPS_STEP
    return out


def _e_at(e_times, energy, t, loop_dur, song_dur):
    tt = t % loop_dur                       # energy of the LOOPED audio
    if tt > song_dur or not len(energy):
        return 0.0
    return float(energy[min(bisect.bisect_left(e_times, tt), len(energy) - 1)])


def gauntlet_times(bpm, beat0, loop_dur, song_dur, e_times, energy,
                   thresh=0.15):
    """Return (times, total_dur, marks). `marks` = [(start_time, nps), ...] per
    level. Segments are measure-aligned; each step-up gap is snapped to the
    lowest-energy measure boundary near the ~30s mark (a musical break)."""
    period = 60.0 / bpm
    measure = 4 * period
    emax = max(energy) if len(energy) else 1.0
    floor = thresh * emax

    def lull(target):                       # nearest low-energy measure boundary
        base = beat0 + round((target - beat0) / measure) * measure
        best, bestE = base, 1e9
        for k in range(-3, 4):
            tb = base + k * measure
            if tb <= 0:
                continue
            e = _e_at(e_times, energy, tb, loop_dur, song_dur)
            if e < bestE:
                best, bestE = tb, e
        return best

    times, marks = [], []
    t = beat0 + math.ceil((LEAD_IN) / measure) * measure     # lead-in, measure-aligned
    for nps in _levels():
        seg_end = lull(t + SEG_SEC)
        if seg_end <= t + measure:
            seg_end = t + SEG_SEC
        marks.append((t, nps))
        dt = 1.0 / nps
        tt = t
        while tt < seg_end - 1e-9:
            if _e_at(e_times, energy, tt, loop_dur, song_dur) >= floor:
                times.append(tt)
            tt += dt
        t = seg_end + GAP_BEATS * period     # breather gap
    return times, t, marks


def _loop_unit(data, sr, loop_dur):
    n = int(round(loop_dur * sr))
    if len(data) < n:
        pad = (n - len(data),) + data.shape[1:]
        return np.concatenate([data, np.zeros(pad, dtype=data.dtype)])
    return data[:n]


def build_song(audio_path, ddr_dir, bs_dir):
    folder = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    title = f"ROM Gauntlet - {stem}"   # prefix so all gauntlet songs group together in the song list
    artist = artist_of(folder, audio_path)
    print(f"\n=== {title}  [{artist}] ===")

    data, sr = sf.read(audio_path, dtype="float32")
    bpm, beat0, e_times, energy, song_dur = cached_grid_energy(audio_path)
    measure = 4 * 60.0 / bpm
    loop_dur = math.ceil(song_dur / measure) * measure

    times, total, marks = gauntlet_times(bpm, beat0, loop_dur, song_dur,
                                         e_times, energy)
    n_loops = max(1, math.ceil(total / loop_dur))
    lanes = build_stream(len(times))
    notes = [Note(t, lane) for t, lane in zip(times, lanes)]
    print(f"  {song_dur:.0f}s @ {bpm:.1f} BPM | {len(marks)} levels "
          f"{marks[0][1]:.1f}->{marks[-1][1]:.1f} nps | {total/60:.1f} min | "
          f"{len(notes)} notes")

    unit = _loop_unit(data, sr, loop_dur)

    # ---- DDR (.sm, single Challenge chart) ----
    safe = _sanitize(title)
    dsong = os.path.join(ddr_dir, safe)
    os.makedirs(dsong, exist_ok=True)
    music = safe + ".ogg"
    ch = 1 if data.ndim == 1 else data.shape[1]
    ogg = os.path.join(dsong, music)          # reuse if already encoded (chart-only regens)
    if not (os.path.isfile(ogg) and os.path.getsize(ogg) > 100):
        with sf.SoundFile(ogg, "w", samplerate=sr, channels=ch,
                          format="OGG", subtype="VORBIS") as out:
            for _ in range(n_loops):
                for i in range(0, len(unit), 65536):
                    out.write(unit[i:i + 65536])
    body = notes_to_measures(notes, bpm, beat0, max_lines=48, allow_triplets=False)
    radar = compute_radar(notes, bpm, beat0, total)
    meter = predict_meter(radar, "Challenge")
    banner = ""
    art = find_artwork(folder, stem)
    if art:
        save_png(art, os.path.join(dsong, safe + ".png"))
        save_png(art, os.path.join(dsong, safe + "-bg.png"))
        banner = f"#BANNER:{safe}.png;\n#BACKGROUND:{safe}-bg.png;\n"
    header = (f"#TITLE:{title};\n#ARTIST:{artist};\n#CREDIT:NoteSight Gauntlet;\n"
              f"{banner}#MUSIC:{music};\n#OFFSET:{-beat0:.3f};\n#SAMPLESTART:0.000;\n"
              f"#SAMPLELENGTH:12.000;\n#SELECTABLE:YES;\n#DISPLAYBPM:{bpm:.3f};\n"
              f"#BPMS:0.000={bpm:.3f};\n\n")
    block = ("#NOTES:\n     dance-single:\n     NoteSight:\n     Challenge:\n"
             f"     {meter}:\n     {radar.sm_field()}:\n{body};\n")
    with open(os.path.join(dsong, safe + ".sm"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(header + block)

    # ---- Beat Saber (flow voicing, lead-in already baked into times) ----
    bsong = os.path.join(bs_dir, safe)
    os.makedirs(bsong, exist_ok=True)
    egg = os.path.join(bsong, "song.egg")     # reuse if already encoded
    if not (os.path.isfile(egg) and os.path.getsize(egg) > 100):
        _write_looped_egg(unit, sr, n_loops, egg)
    meta = SongMeta(title=title, artist=artist, audio_path="",
                    bpm=bpm, beat0=beat0, duration=total)
    write_pack({"Challenge": notes}, meta, bs_dir, cover_src=art or "",
               note_opts={"model": True, "double_scale": 1.0})   # learned flow
    print(f"  wrote DDR + BS gauntlet")


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    out_dir, rest = argv[0], argv[1:]
    ddr_dir = os.path.join(out_dir, "DDR")
    bs_dir = os.path.join(out_dir, "BeatSaber")
    os.makedirs(ddr_dir, exist_ok=True)
    os.makedirs(bs_dir, exist_ok=True)
    ok = 0
    for s, _ in collect_paths(rest):
        audio = resolve_audio(s)
        if not audio:
            print(f"!! skip (no audio): {s}", file=sys.stderr)
            continue
        try:
            build_song(audio, ddr_dir, bs_dir)
            ok += 1
        except Exception as e:
            print(f"!! FAILED {s}: {e}", file=sys.stderr)
    print(f"\nDone: {ok} gauntlet song(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
