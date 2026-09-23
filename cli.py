r"""cli.py -- chart an audio file into a drop-in song folder.

    python cli.py song.mp3 --difficulty medium -o out\

Every option here maps to a ChartSpec field -- the single options object the whole
pipeline is driven by -- so scripting the CLI and calling build_chart() directly
produce identical charts. Writes out\<Title>\ with the chart + a copy of the audio
(StepMania: drop it into ITGMania\Songs\<Group>\; Beat Saber: into CustomLevels\).
"""
from __future__ import annotations

import argparse
import os
import sys

from notesight import (DIFFICULTIES, ChartSpec, analyze_audio, build_chart,
                       available_formats, get_format, SongMeta)
from notesight.audio_io import load_audio


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Audio file -> rhythm-game chart.")
    p.add_argument("audio", help="input audio file (wav/ogg/flac/mp3/m4a/...)")
    p.add_argument("--difficulty", "-d", default="medium",
                   choices=list(DIFFICULTIES), help="difficulty preset")
    p.add_argument("--format", "-f", default="stepmania",
                   choices=available_formats(), help="output chart format")
    # ChartSpec style options (fine chart-shape control).
    jumps = p.add_mutually_exclusive_group()
    jumps.add_argument("--jumps", dest="jumps", action="store_true", default=None,
                       help="force jumps on strong beats on")
    jumps.add_argument("--no-jumps", dest="jumps", action="store_false",
                       help="force jumps off")
    p.add_argument("--max-subdivision", type=int, default=16, choices=(4, 8, 16),
                   help="finest quantize grid: 4=1/4, 8=1/8, 16=1/16")
    p.add_argument("--no-triplets", dest="allow_triplets", action="store_false",
                   help="disallow triplet measures")
    p.add_argument("--bpm", type=float, default=None,
                   help="override BPM (default: auto-estimate)")
    # Adjustable knobs: re-roll or retarget without editing presets.
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed: change it to re-roll a different-but-equivalent "
                        "chart; the same seed always reproduces the same chart")
    p.add_argument("--nps", type=float, default=None, dest="target_nps",
                   help="override the difficulty's target notes/sec (density)")
    p.add_argument("--peak", type=float, default=None, dest="peak_nps",
                   help="override the busiest-1-second note ceiling")
    p.add_argument("--lrc", default=None,
                   help="timestamped .lrc lyrics file; its chorus-repeat structure "
                        "guides pattern reuse (default: auto-detect a sidecar "
                        "<audio>.lrc, use --no-lrc to disable)")
    p.add_argument("--no-lrc", dest="lrc", action="store_const", const="",
                   help="ignore any sidecar .lrc file")
    p.add_argument("--out", "-o", default="out", help="output directory")
    p.add_argument("--title", default=None, help="song title (default: filename)")
    p.add_argument("--artist", default="Unknown Artist", help="song artist")
    args = p.parse_args(argv)

    if not os.path.isfile(args.audio):
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 2

    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]
    spec = ChartSpec(difficulty=args.difficulty,
                     jumps=args.jumps, max_subdivision=args.max_subdivision,
                     allow_triplets=args.allow_triplets, bpm=args.bpm,
                     seed=args.seed, target_nps=args.target_nps,
                     peak_nps=args.peak_nps)

    # Lyrics: explicit --lrc wins; --no-lrc ("") disables; otherwise auto-detect a
    # sidecar <audio>.lrc. A timestamped .lrc lets the chorus (where the words come
    # back) drive pattern reuse -- more reliable than acoustic self-similarity.
    if args.lrc is None:
        cand = os.path.splitext(args.audio)[0] + ".lrc"
        lrc_path = cand if os.path.isfile(cand) else None
    else:
        lrc_path = args.lrc or None

    print(f"Loading {args.audio} ...")
    mono, sr = load_audio(args.audio)
    analysis = analyze_audio(mono, sr, lrc_path=lrc_path)
    print(f"  {analysis.duration:.1f}s @ {sr} Hz  |  "
          f"{len(analysis.onsets)} candidate onsets"
          + (f"  |  lyrics: {os.path.basename(lrc_path)}" if lrc_path else ""))

    r = build_chart(spec, analysis)
    print(f"BPM {r.bpm:.2f}"
          + ("" if args.bpm is None else " (override)")
          + f"  |  offset {-r.beat0:+.3f}")
    print(f"Selected {len(r.notes)} notes  "
          f"[{spec.difficulty}]  "
          f"({len(r.notes)/max(1,r.duration):.1f} notes/sec)")
    print(f"Chart shape: stream {r.radar.stream:.2f} voltage {r.radar.voltage:.2f} "
          f"air {r.radar.air:.2f} chaos {r.radar.chaos:.2f}  -> meter {r.meter}")

    meta = SongMeta(title=title, artist=args.artist, audio_path=args.audio,
                    bpm=r.bpm, beat0=r.beat0, duration=r.duration,
                    difficulty=spec.resolved_difficulty(),
                    max_subdivision=spec.max_subdivision,
                    allow_triplets=spec.allow_triplets, seed=args.seed)
    fmt = get_format(args.format)
    out_path = fmt.write(r.notes, meta, args.out)
    print(f"Wrote {out_path}")
    if args.format == "beatsaber":
        print(f"  -> drop {out_path!r} into Beat Saber's CustomLevels\\ (via your mod loader)")
    else:
        print(f"  -> drop {os.path.dirname(out_path)!r} into StepMania/ITGMania Songs\\<Group>\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
