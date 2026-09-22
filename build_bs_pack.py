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
from dataclasses import replace

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
#
# target_nps here is the DESIRED FINAL blocks/sec (what the manifest reports, what the
# player feels) -- Easy..Expert+ = 2/3/4/4.5/5. backfill=True + the per-tier CLOSED LOOP
# in build_song tune an internal density until the chart ACTUALLY lands on that number
# (the old table set fake pre-compensated targets with backfill off, so the selector
# stopped short of target by a different amount on every song -- see the 4.9/5.0 wart).
# min_interval is now just the burst floor / post-filter, not the density knob.
LEAD_IN = 2.0            # seconds of music before the first block (get-ready / anti-insta-fail)

BS_DIFFICULTIES = {
    #                     name         min_iv  nps  jumps meter sub  trip           peak / backfill
    "beginner": Difficulty("Easy",       0.25, 2.0, True,  3,     8, False, peak_nps=5,  backfill=True),
    "easy":     Difficulty("Normal",     0.17, 3.0, True,  5,     8, False, peak_nps=6,  backfill=True),
    "medium":   Difficulty("Hard",       0.12, 4.0, True,  7,    16, False, peak_nps=8,  backfill=True),
    "hard":     Difficulty("Expert",     0.11, 4.5, True,  9,    16, False, peak_nps=9,  backfill=True),
    "expert":   Difficulty("ExpertPlus", 0.10, 5.0, True, 11,    16, False, peak_nps=10, backfill=True),
}

# PRO Beat Saber ladder (--pro): a SEPARATE, harder pack. Its floor (5.0) is the regular
# pack's Expert+ VERBATIM (same preset, and seeded by density so the chart is identical),
# then 6/7/8/9 climb above it. 6.0 stays onset-honest (backfill to the song's real onset
# rate); 7/8/9 turn on STREAM-FILL to reach tech-map density. Unlike DDR, Beat Saber has
# no measure-line grid cap, so the fill runs on a 32nd grid (stream_grid=32) and low-BPM
# songs can still reach the top tiers. The 5 tiers map onto BS's five slots Easy..E+.
BS_PRO_DIFFICULTIES = {
    "pro_floor": BS_DIFFICULTIES["expert"],   # 5.0, IDENTICAL to regular Expert+
    "bs_pro2": Difficulty("Normal",     0.040, 6.0, True, 12,    16, False, peak_nps=11, backfill=True,
                          stream_fill=True, stream_energy_pct=0.68, stream_grid=32),
    # Stream tiers: min_interval must sit BELOW the 32nd-grid spacing (~0.042s at 180bpm,
    # ~0.054s at 139bpm) or the post-filter decimates the very streams the fill places. 0.04
    # lets a true 32nd stream through; the peak cap + assemble peak_cap still bound the bursts.
    "bs_pro3": Difficulty("Hard",       0.040, 7.0, True, 13,    16, False, peak_nps=13, backfill=True,
                          stream_fill=True, stream_energy_pct=0.55, stream_grid=32),
    "bs_pro4": Difficulty("Expert",     0.040, 8.0, True, 14,    16, False, peak_nps=15, backfill=True,
                          stream_fill=True, stream_energy_pct=0.42, stream_grid=32),
    "bs_pro5": Difficulty("ExpertPlus", 0.040, 9.0, True, 15,    16, False, peak_nps=17, backfill=True,
                          stream_fill=True, stream_energy_pct=0.30, stream_grid=32),
}

# Ladder = list of (preset_key, NoteSight-slot-name). write_pack maps the slot name onto
# the BS Standard slot (Beginner->Easy .. Challenge->ExpertPlus) + its rank/NJS, so these
# are the SAME five slot labels for both packs (the pro pack is a separate folder).
BS_TIERS = [("beginner", "Beginner"), ("easy", "Easy"), ("medium", "Medium"),
            ("hard", "Hard"), ("expert", "Challenge")]
BS_PRO_TIERS = [("pro_floor", "Beginner"), ("bs_pro2", "Easy"), ("bs_pro3", "Medium"),
                ("bs_pro4", "Hard"), ("bs_pro5", "Challenge")]

# Converge the final blocks/sec to within this fraction of target, or give up after N tries.
_BS_TOL = 0.02
_BS_ITERS = 8
# A tier is KEPT only if it reaches >= this fraction of target (else omit -- onset-limited).
_BS_KEEP_FRAC = 0.90


def _postfilter(notes, min_interval):
    """LEAD-IN: never start a note in the first LEAD_IN seconds (a block spawning in your
    face at song start = insta-fail). MIN-SPACING: drop any note closer than min_interval
    to the previous kept one (no unhittable stacks)."""
    kept, last_t = [], -1e9
    for n in notes:
        if n.time >= LEAD_IN and n.time - last_t >= min_interval:
            kept.append(n)
            last_t = n.time
    return kept


def _tier_blocks(analysis, dd, bpm, alpha, seed):
    """Build ONE Beat Saber tier and converge its FINAL blocks/sec onto dd.target_nps.

    The chain note-times -> post-filter -> letter-voicing (which adds doubles) means the
    played block density isn't the selection's note count, so we can't just set target_nps
    and trust it. Instead we iterate: build at an internal density `eff`, voice it, measure
    the real blocks/sec, and rescale `eff` toward the target. backfill makes each build hit
    `eff` exactly, so this converges in a few passes. Returns (blocks, time2label, blk/s)."""
    target = dd.target_nps
    eff = target
    best = None
    best_err = 1e9
    best_bps = 0.0
    for _ in range(_BS_ITERS):
        spec = ChartSpec(difficulty="expert", diff_override=replace(dd, target_nps=eff),
                         max_subdivision=dd.max_subdivision,
                         allow_triplets=dd.allow_triplets)
        r = build_chart(spec, analysis)
        times = [n.time for n in _postfilter(r.notes, dd.min_interval)]
        # peak_cap gives the voicer burst headroom above the average; too tight and it
        # thins bursts so hard the average can't reach target. Scale headroom with the tier.
        blocks, t2l = assemble_letters(times, bpm, seed=seed,
                                       peak_cap=target * 1.35 + 1.0, alphabet=alpha)
        bt = sorted(b["_time"] * 60.0 / bpm for b in blocks)
        span = (bt[-1] - bt[0]) if len(bt) >= 2 else 0.0
        bps = len(blocks) / span if span > 0 else 0.0
        err = abs(bps - target)
        if err < best_err:
            best, best_err, best_bps = (blocks, t2l), err, bps
        if err <= _BS_TOL * target or bps <= 0:
            break
        eff *= target / max(bps, 0.1)           # rescale internal density toward the block target
    return best[0], best[1], best_bps


def build_song(audio_path: str, out_dir: str, title_override: str = "",
               pro: bool = False) -> str:
    folder = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    title = title_override or stem
    if pro:
        title += " (Pro)"          # coexists with the regular chart in the same install
    artist = artist_of(folder, audio_path)

    print(f"\n=== {title}  [{artist}] ===")
    analysis = cached_analyze(audio_path, find_lrc(folder, stem))
    bpm, beat0, duration = analysis.grid.bpm, analysis.grid.beat0, analysis.duration
    print(f"  {duration:.1f}s | BPM {bpm:.2f}")

    # THE ONE GENERATOR: letters. The song commits to a single alphabet (seeded by its name),
    # and every difficulty composes phrases from it onto that tier's note times.
    song_alpha = pick_alphabet(zlib.crc32(stem.encode("utf-8")), allow_dots=False)   # no dots (user pref)
    tiers = BS_PRO_TIERS if pro else BS_TIERS
    diffs = BS_PRO_DIFFICULTIES if pro else BS_DIFFICULTIES

    rendered = []                               # (slot, dd, blocks, t2l, bps)
    for preset, slot in tiers:
        dd = diffs[preset]
        # Seed by DENSITY (not tier position) so a given target is reproducible across packs
        # -- this is what makes pro-Beginner (5.0) byte-identical to regular Expert+ (5.0).
        seed = zlib.crc32(f"{stem}|{dd.target_nps:.1f}".encode("utf-8"))
        blocks, t2l, bps = _tier_blocks(analysis, dd, bpm, song_alpha, seed)
        rendered.append((slot, dd, blocks, t2l, bps))

    # Omit any TOP tier that can't reach ~90% of its target (onset-limited); never drop the
    # lowest. Same rule the DDR builder uses -- an unreachable "Challenge" that's really a
    # second "Hard" is misleading, so leave it out rather than ship it.
    keep = [True] * len(rendered)
    for i in range(len(rendered) - 1, 0, -1):
        if rendered[i][4] < _BS_KEEP_FRAC * rendered[i][1].target_nps:
            keep[i] = False
        else:
            break

    notes_by_diff = {}
    report_charts = []                          # (diff, bpm, blocks, time2label)
    for (slot, dd, blocks, t2l, bps), k in zip(rendered, keep):
        print(f"    {slot:<10} {len(blocks):>4} blocks  {bps:4.2f} blk/s  (target {dd.target_nps:.1f})"
              + ("" if k else "   [omit: onset-limited]"))
        if k:
            notes_by_diff[slot] = blocks
            report_charts.append((dd.name, bpm, blocks, t2l))

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
    pro = "--pro" in argv          # build the harder PRO ladder (5/6/7/8/9), titles get " (Pro)"
    argv = [a for a in argv if a != "--pro"]
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
                build_song(audio, out_dir, title, pro=pro)
                ok += 1
            except Exception as e:
                print(f"!! FAILED {title}: {e}", file=sys.stderr)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            fut = {ex.submit(build_song, a, out_dir, t, pro): t for a, t in jobs_list}
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
