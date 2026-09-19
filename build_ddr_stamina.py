r"""build_ddr_stamina.py -- STAMINA / CARDIO packs.

Takes each source song and makes "<Song> <N> Min": one seamless streaming pattern
(no jumps, no crossovers -- see notesight/stamina.py) that fills the song, with
the whole song+chart REPEATED until it reaches the target length. Loop units are a
whole number of measures so the repeats stay locked to the beat grid.

Each song ships FIVE selectable difficulties, scaled by target notes-per-second so
you can pick your cardio pace: 2 / 3 / 4 / 5 / 6 NPS.

    python build_ddr_stamina.py <out_dir> <minutes> <audio_or_folder>...
    python build_ddr_stamina.py out/Stamina 30 --dir path/to/songs/
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import soundfile as sf

from build_ddr_pack import (resolve_audio, artist_of, find_artwork, save_png,
                        collect_paths, cached_grid_energy)
from notesight.radar import compute_radar, predict_meter
from notesight.formats.stepmania import Note, notes_to_measures, _sanitize
from notesight.stamina import stamina_notes, annotate, validate_stream
from notesight.version import ITG_GEN_VERSION, build_timestamp

# (target NPS, StepMania difficulty slot). Five paces from a light jog to a sprint.
STAMINA_TIERS = [
    (2.0, "Beginner"),
    (3.0, "Easy"),
    (4.0, "Medium"),
    (5.0, "Hard"),
    (6.0, "Challenge"),
]


def one_play_ddr(audio_path):
    """One play of a song for DDR: the padded loop-audio unit + per-tier ONE-loop notes
    (the exact chart the stamina pack tiles). The mashup concatenates one of these per
    song, each kept at its OWN bpm via #BPMS; a single stamina song tiles one of them.
    Returns (bpm, beat0, loop_dur, sr, unit, {slot: (one_notes, sub)})."""
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
    tiers = {}
    for nps, slot in STAMINA_TIERS:
        notes_tl, sub = stamina_notes(bpm, beat0, e_times, energy, loop_dur,
                                      1, song_dur, target_nps=nps)
        one = [Note(t, lane) for t, lane in notes_tl if t < loop_dur]
        tiers[slot] = (one, sub)
    return bpm, beat0, loop_dur, sr, unit, tiers


def build_song(audio_path, out_dir, minutes):
    folder = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    title = f"{stem} {minutes} Min"
    artist = artist_of(folder, audio_path)
    print(f"\n=== {title}  [{artist}] ===")

    data, sr = sf.read(audio_path, dtype="float32")        # stereo for output
    bpm, beat0, e_times, energy, song_dur = cached_grid_energy(audio_path)

    measure = 4 * 60.0 / bpm
    loop_dur = math.ceil(song_dur / measure) * measure     # pad to whole measures
    n_loops = max(1, math.ceil(minutes * 60.0 / loop_dur))
    total = n_loops * loop_dur
    print(f"  {song_dur:.0f}s @ {bpm:.1f} BPM -> loop {loop_dur:.0f}s x {n_loops} "
          f"= {total/60:.1f} min")

    safe = _sanitize(title)
    song_dir = os.path.join(out_dir, safe)
    os.makedirs(song_dir, exist_ok=True)

    # Looped audio -> OGG (shared by every difficulty): stream the whole-measure
    # unit n_loops times.
    loop_samples = int(round(loop_dur * sr))
    if len(data) < loop_samples:                           # pad tail with silence
        padshape = (loop_samples - len(data),) + data.shape[1:]
        unit = np.concatenate([data, np.zeros(padshape, dtype=data.dtype)])
    else:
        unit = data[:loop_samples]
    ch = 1 if data.ndim == 1 else data.shape[1]
    music_name = safe + ".ogg"
    with sf.SoundFile(os.path.join(song_dir, music_name), "w", samplerate=sr,
                      channels=ch, format="OGG", subtype="VORBIS") as out:
        for _ in range(n_loops):
            for i in range(0, len(unit), 65536):
                out.write(unit[i:i + 65536])

    # One chart block per NPS tier.
    blocks = []
    tier_report = []
    for nps, slot in STAMINA_TIERS:
        notes_tl, sub = stamina_notes(bpm, beat0, e_times, energy, loop_dur,
                                      n_loops, song_dur, target_nps=nps)
        notes = [Note(t, lane) for t, lane in notes_tl]
        body = notes_to_measures(notes, bpm, beat0, max_lines=sub * 4,
                                 allow_triplets=False)
        one = [Note(t, lane) for t, lane in notes_tl if t < loop_dur]
        radar = compute_radar(one, bpm, beat0, loop_dur)
        meter = predict_meter(radar, slot)
        print(f"    {slot:<10} ~{nps:.0f} nps  meter {meter:>2}  {len(notes_tl):>5} "
              f"notes ({len(notes_tl)/total:.1f}/s actual)")
        tier_report.append(f"  {slot:<10} ~{nps:.0f} nps  meter {meter:>2}  "
                           f"{len(notes_tl):>6} notes  ({len(one)} in one loop x{n_loops})")
        blocks.append("#NOTES:\n     dance-single:\n     NoteSight:\n"
                      f"     {slot}:\n     {meter}:\n     {radar.sm_field()}:\n"
                      f"{body};\n")

    banner_line = ""
    art = find_artwork(folder, stem)
    if art:
        save_png(art, os.path.join(song_dir, safe + ".png"))
        save_png(art, os.path.join(song_dir, safe + "-bg.png"))
        banner_line = f"#BANNER:{safe}.png;\n#BACKGROUND:{safe}-bg.png;\n"

    # Embed the ITG generator version as a StepMania #TAG (ignored on load, but travels
    # with the chart) so a deployed .sm always says which generator produced it.
    header = (
        f"#TITLE:{title};\n#ARTIST:{artist};\n#CREDIT:NoteSight Stamina;\n"
        f"#GENERATOR:NoteSight ITG {ITG_GEN_VERSION};\n"
        f"{banner_line}#MUSIC:{music_name};\n#OFFSET:{-beat0:.3f};\n"
        f"#SAMPLESTART:0.000;\n#SAMPLELENGTH:12.000;\n#SELECTABLE:YES;\n"
        f"#DISPLAYBPM:{bpm:.3f};\n#BPMS:0.000={bpm:.3f};\n\n"
    )
    with open(os.path.join(song_dir, safe + ".sm"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(header + "\n".join(blocks))

    with open(os.path.join(song_dir, safe + "_feet.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write(f"# {title}: stamina footflow (first 64 notes of the stream)\n")
        f.write(annotate(64) + "\n")

    # Generation report -- version + per-tier density + the cell->motif->phrase footflow.
    rep = ["NoteSight generation report",
           f"generator: ITG / StepMania  {ITG_GEN_VERSION}",
           f"generated: {build_timestamp()}",
           f"song: {title}",
           f"loop {loop_dur:.0f}s x {n_loops} = {total/60:.1f} min @ {bpm:.1f} BPM", "",
           "  -- difficulty tiers --", *tier_report, "",
           "  -- stamina footflow (first 64 notes of the repeating stream) --",
           annotate(64)]
    with open(os.path.join(song_dir, "_generation.txt"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    print(f"  wrote {song_dir}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    out_dir, minutes = argv[0], int(argv[1])
    rest = argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    validate_stream()          # fail fast if a motif/cell breaks the invariants
    audios = []
    for s, _ in collect_paths(rest):
        audio = resolve_audio(s)
        if audio:
            audios.append(audio)
        else:
            print(f"!! skip (no audio): {s}", file=sys.stderr)
    # PARALLEL per-song build (slow part = the 30/45-min OGG encode; songs are independent).
    # STAMINA_JOBS=1 forces serial.
    jobs = int(os.environ.get("STAMINA_JOBS", 0)) \
        or max(1, min((os.cpu_count() or 2) - 1, len(audios)))
    ok = 0
    if jobs <= 1 or len(audios) <= 1:
        for audio in audios:
            try:
                build_song(audio, out_dir, minutes)
                ok += 1
            except Exception as e:
                print(f"!! FAILED {audio}: {e}", file=sys.stderr)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut = {ex.submit(build_song, a, out_dir, minutes): a for a in audios}
            for f in as_completed(fut):
                try:
                    f.result()
                    ok += 1
                except Exception as e:
                    print(f"!! FAILED {fut[f]}: {e}", file=sys.stderr)
    print(f"\nDone: {ok} stamina song(s) -> {out_dir}  (x{jobs} parallel)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
