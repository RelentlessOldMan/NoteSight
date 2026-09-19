r"""build_bs_pack.py -- turn a batch of audio files into Beat Saber song packs.

Mirror of build_ddr_pack.py (which builds StepMania .sm folders), but emits the Beat
Saber map format: one folder per song with info.dat, a .dat per difficulty, the
audio as OGG (.egg), and a cover. Drops into Beat Saber's CustomLevels, and (once
NoteNinja reads the format) into NoteNinja too.

    python build_bs_pack.py <out_dir> <song-or-folder>... | --list file.txt

Reuses build_ddr_pack's song-resolution helpers and the shared analyze->build_chart
pipeline (so density/holds/structure all carry over); only the writer differs.
"""
from __future__ import annotations

import os
import sys

from build_ddr_pack import (resolve_audio, band_of, artist_of, find_artwork,
                        find_lrc, collect_paths, TIERS, cached_analyze)
from notesight import ChartSpec, Difficulty, build_chart, SongMeta
import zlib
from notesight.formats.beatsaber import write_pack
from notesight.formats.bs_assemble import assemble_letters
from notesight.formats.bs_letters import pick_alphabet
from notesight.formats.bs_report import write_reports_letters
from notesight.version import BS_GEN_VERSION

# Beat Saber's OWN density ladder -- deliberately NOT the DDR-calibrated DIFFICULTIES
# table (steps/sec and difficulty read totally differently between a 4-panel dance
# pad and two sabers; you cannot cross them over -- DDR needs its own measurement).
# Target a clean 2/3/4/5/6 NPS across Easy..Expert+ (well within the real BS library
# spread -- medians run 2.6/3.2/4.1/5.5/7.2 -- an accessible-floor normal pack). The
# model decides doubles from note TIMES (it dedupes chords), so only target_nps,
# min_interval and the subdivision matter here; jumps/triplets/meter are irrelevant.
# Keyed by the same TIERS preset names (which map beginner..expert -> BS Easy..E+).
# target_nps here is pre-compensated for each preset's measured selection undershoot
# (medium/expert land ~0.85x/0.92x of target), so achieved block-NPS hits 2/3/4/5/6.
LEAD_IN = 2.0            # seconds of music before the first block (get-ready / anti-insta-fail)

BS_DIFFICULTIES = {
    #                     name         min_iv  nps  jumps meter sub  trip
    "beginner": Difficulty("Easy",       0.30, 2.0, True,  3,     8, False),
    "easy":     Difficulty("Normal",     0.22, 3.1, True,  5,     8, False),
    "medium":   Difficulty("Hard",       0.14, 4.9, True,  7,    16, False),
    "hard":     Difficulty("Expert",     0.12, 5.0, True,  9,    16, False),
    "expert":   Difficulty("ExpertPlus", 0.10, 6.6, True, 11,    16, False),
}


def build_song(audio_path: str, out_dir: str, title_override: str = "") -> str:
    folder = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    title = title_override or stem
    artist = artist_of(folder, audio_path)

    print(f"\n=== {title}  [{artist}] ===")
    analysis = cached_analyze(audio_path, find_lrc(folder, stem))
    bpm, beat0, duration = analysis.grid.bpm, analysis.grid.beat0, analysis.duration
    print(f"  {duration:.1f}s | BPM {bpm:.2f}")

    # THE ONE GENERATOR: letters. The song commits to a single alphabet (seeded by its name),
    # and every difficulty composes phrases from it onto that tier's onset times. Regular songs
    # keep dots (allow_dots=True); stamina uses the same engine with dots off.
    song_alpha = pick_alphabet(zlib.crc32(stem.encode("utf-8")), allow_dots=False)   # no dots (user pref)
    notes_by_diff = {}
    report_charts = []                          # (diff, bpm, blocks, time2label)
    for si, (preset, slot) in enumerate(TIERS):
        d = BS_DIFFICULTIES[preset]                 # BS ladder, not the DDR table
        spec = ChartSpec(difficulty=preset, diff_override=d,
                         max_subdivision=d.max_subdivision,
                         allow_triplets=d.allow_triplets)
        r = build_chart(spec, analysis)
        # LEAD-IN: never start a note in the first LEAD_IN seconds -- otherwise a block
        # spawns in your face at song start = insta-fail. MIN-SPACING: drop any note
        # closer than this tier's min_iv to the previous kept one (no unhittable stacks).
        kept, last_t = [], -1e9
        for n in r.notes:
            if n.time >= LEAD_IN and n.time - last_t >= d.min_interval:
                kept.append(n)
                last_t = n.time
        times = [n.time for n in kept]
        blocks, t2l = assemble_letters(times, bpm, seed=si + 1,
                                       peak_cap=d.target_nps + 1.5, alphabet=song_alpha)
        notes_by_diff[slot] = blocks
        report_charts.append((d.name, bpm, blocks, t2l))
        print(f"    {slot:<10} {len(kept):>4} notes -> {len(blocks)} blocks")

    meta = SongMeta(title=title, artist=artist, audio_path=audio_path,
                    bpm=bpm, beat0=beat0, duration=duration)
    cover = find_artwork(folder, stem) or ""
    out = write_pack(notes_by_diff, meta, out_dir, cover_src=cover,
                     note_opts={"raw": True})
    write_reports_letters(out, title, BS_GEN_VERSION, report_charts)
    print(f"  wrote {out}")
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    out_dir, song_args = argv[0], argv[1:]
    os.makedirs(out_dir, exist_ok=True)

    songs = collect_paths(song_args)
    jobs_list = []                              # (audio, title) for each resolvable song
    for s, title in songs:
        audio = resolve_audio(s)
        if not audio:
            print(f"!! skip (no audio found): {s}", file=sys.stderr)
            continue
        jobs_list.append((audio, title))
    # PARALLEL: songs are independent + deterministic, so build ONE PER CORE (output is
    # identical to serial). The pool runs cpu-many at a time and queues the rest; set
    # BS_PACK_JOBS=1 to force serial (e.g. deploying over packs you're playing).
    jobs = int(os.environ.get("BS_PACK_JOBS", 0)) \
        or max(1, min(os.cpu_count() or 2, len(jobs_list)))
    ok = 0
    if jobs <= 1 or len(jobs_list) <= 1:
        for audio, title in jobs_list:
            try:
                build_song(audio, out_dir, title)
                ok += 1
            except Exception as e:
                print(f"!! FAILED {title}: {e}", file=sys.stderr)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut = {ex.submit(build_song, a, out_dir, t): t for a, t in jobs_list}
            for f in as_completed(fut):
                try:
                    f.result()
                    ok += 1
                except Exception as e:
                    print(f"!! FAILED {fut[f]}: {e}", file=sys.stderr)
    print(f"\nDone: {ok}/{len(jobs_list)} Beat Saber packs -> {out_dir}  (x{jobs} parallel)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
