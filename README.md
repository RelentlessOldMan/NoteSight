# NoteSight

[![CI](https://github.com/RelentlessOldMan/NoteSight/actions/workflows/ci.yml/badge.svg)](https://github.com/RelentlessOldMan/NoteSight/actions/workflows/ci.yml)

Turn an **audio file** into a rhythm-game chart. Feed it a song and NoteSight
detects the onsets, finds the beat grid, and writes a playable chart:

- **StepMania / ITGmania** `.sm` song folders (4-panel dance)
- **Beat Saber** maps (`info.dat` + per-difficulty `.dat` + encoded audio)

It reads the music, not a MIDI or a chart template: onset detection finds the
hits, a beat grid locks tempo and phase, an energy model concentrates notes in the
loud/exciting passages (bursts) and thins the calm ones (rests), and repeated
sections — choruses, via the lyrics when present — reuse the same pattern.

The core is plain array math (NumPy/SciPy), so onset detection, note selection, and
difficulty are all format-neutral; exporters plug in behind a small ABC.

## Install

Requires **Python 3.9+**. From a clone:

```bash
git clone https://github.com/RelentlessOldMan/NoteSight
cd NoteSight
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                                  # add [view] for the 3D viewer, [dev] for tests
```

Audio decoding needs no system setup — `soundfile` handles wav/ogg/flac and a
bundled `imageio-ffmpeg` binary decodes mp3/m4a/aac, so there's nothing else to
install.

## Quickstart

The install adds a `notesight` command (equivalent to `python cli.py`):

```bash
# StepMania .sm folder, medium difficulty
notesight samples/Neon.ogg -d medium -o out/

# Beat Saber map, harder, from the same audio
notesight samples/Neon.ogg -f beatsaber -d hard -o out/

# A vocal track with a sidecar .lrc -> lyric-aligned pattern reuse (auto-detected)
notesight "samples/Brain Rot (Only In Ohio).ogg" -d hard -o out/
```

`out/<Title>/` is drop-in: copy a StepMania folder into
`ITGMania/Songs/<Group>/`, or a Beat Saber folder into your `CustomLevels/`.

Short royalty-free tracks ship in [`samples/`](samples/) so it runs out of the
box — see that folder's README for what each demonstrates.

### Options

| flag | meaning |
|---|---|
| `-d, --difficulty` | `beginner \| easy \| medium \| hard \| expert` |
| `-f, --format` | `stepmania` (default) or `beatsaber` |
| `--lrc PATH` / `--no-lrc` | lyrics file for chorus-aware pattern reuse (auto-detects a sidecar `<audio>.lrc`) |
| `--bpm` | override the auto-estimated tempo |
| `--jumps` / `--no-jumps`, `--max-subdivision {4,8,16}`, `--no-triplets` | fine chart-shape control |
| `-o, --out`, `--title`, `--artist` | output dir + metadata |

### Lyrics → structure

If a timestamped `.lrc` sits next to the audio (or you pass `--lrc`), NoteSight
finds **repeated lyric runs** — the chorus, where the words come back — and reuses
/ mirrors the chart pattern there. That's a more reliable structure signal than
acoustic self-similarity, which misses choruses that aren't bit-identical in the
mix. Without lyrics it falls back to the acoustic path automatically.

## Play your chart

**StepMania / ITGmania.** Songs live under `Songs/<Group>/<Song>/`, so drop the
whole output folder into a group folder — make one if you like:

```
StepMania/Songs/NoteSight/<Title>/     <- copy out/<Title>/ here
```

Then restart StepMania/ITGmania (it scans songs on launch); your song shows up
under the **NoteSight** group. Difficulties you generated appear on the select
screen.

**Beat Saber.** Custom songs need a mod loader — this is the same setup any custom
map requires:

- **PC:** install a mod manager (ModAssistant / BSManager) once to get custom-level
  support (SongCore etc.), then copy `out/<Title>/` into
  `…/Beat Saber/Beat Saber_Data/CustomLevels/` and restart. It appears under
  **Custom Levels**.
- **Quest (standalone):** use the usual custom-song sideload path (BMBF / MBF or a
  mod tool) and place the folder in its managed CustomLevels location.

(Chart only music you own or that's royalty-free — see [Legal](#legal).)

## Viewer (optional)

**Beat Saber 3D viewer ("SaberSight")** — `pip install -e ".[view]"` (ursina + panda3d):

```bash
python bs_visualizer.py "out/Neon" Expert
```

Study a generated map's flow without a headset: lit 3D blocks fly at you like the
game, synced to the audio. Controls: **Space**/click = play-pause · drag or click
the bar = seek · wheel = scrub · **←/→** = ±5s · **↑/↓** = difficulty · **`[` / `]`**
= scroll distance · **Esc** = quit. The **Note** button freezes the moment, grabs a
screenshot, and saves your comment to a local `feedback/` folder (`feedback.md` +
`feedback.jsonl` + PNGs) — handy for reviewing a chart and jotting fixes. Nothing
leaves your machine.

## Batch pack builders

The `build_*` scripts turn a folder of audio into a full multi-difficulty pack,
reusing the same engine as the CLI. Each takes an **output dir first**, then songs
— individual files, `--dir <folder>` (every song folder inside), or `--list songs.txt`
(one path per line). A "song path" may be an audio file or the folder holding it.

```bash
# Regular — one chart per song, all five difficulties
python build_ddr_pack.py out/DDR       --dir path/to/songs/     # StepMania .sm
python build_bs_pack.py  out/BeatSaber --dir path/to/songs/     # Beat Saber

# Stamina — one long seamless-looped cardio chart per song, N minutes, 5 NPS paces
python build_ddr_stamina.py out/Stamina 30 --dir path/to/songs/
python build_bs_stamina.py  out/Stamina 30 --dir path/to/songs/

# Gauntlet — a progressive "beep test": density ramps in ~30s steps until you fail.
# Builds BOTH games under out/Gauntlet/{DDR,BeatSaber}/
python build_gauntlet.py out/Gauntlet --dir path/to/songs/

# Mashup — stitch a whole folder of songs into ONE long track, charted two ways:
# a ramping "gauntlet" + a steady "stamina", both games, under out/Mashup/{DDR,BeatSaber}/
python build_mashup.py out/Mashup --dir path/to/songs/ --title "My Mashup"
```

Stamina/gauntlet/mashup read `wav/ogg/flac` (not mp3).

Set a song's artist with a one-line `.artist` file in its folder (otherwise
"Unknown Artist").

## Repo layout

```
notesight/            the engine (importable package)
  onsets.py           format-neutral onset detection (STFT / spectral flux / peak-pick)
  selection.py        note selection: select(onsets, difficulty) -- bursts/rests by energy
  difficulty.py       presets keyed on target notes/sec
  beatgrid.py         tempo AND phase (feeds .sm #OFFSET + quantizer)
  radar.py            ITG-style chart-shape values + predict_meter (auto difficulty meter)
  structure.py        acoustic repeated-section map (pattern reuse)
  lyrics.py           chorus detection from a timestamped .lrc (overrides structure when better)
  pipeline.py         ChartSpec + analyze_audio (cacheable DSP) + build_chart
  formats/            pluggable exporters behind ChartFormat (base.py)
    stepmania.py      4-panel .sm song-folder writer
    beatsaber.py      Beat Saber map writer (+ audio -> .egg encode)
cli.py                audio file -> chart (every ChartSpec field is a flag)
build_*.py            batch pack builders (StepMania / Beat Saber / stamina / gauntlet / mashup)
bs_visualizer.py      Beat Saber 3D map viewer ("SaberSight")
samples/              short royalty-free demo tracks
eval/                 accuracy harness: score output vs hand-authored charts
tests/                pytest suite (run: pytest)
```

## How the timing stays correct

StepMania plays row *r* at the BPM we export, so seconds → rows → seconds
round-trips exactly: a musically "wrong" BPM only makes the quantization look
off-grid, it does not desync. `beatgrid.py` detects tempo **and** phase, so the
export writes a real `#OFFSET` and snaps notes to the grid. On steady-BPM songs
~86% land within 0.1% BPM (the misses are 3:2 tempo ambiguity). Run
`python eval/batch.py <Songs/ folder>` to re-check against reference charts.

## Troubleshooting

- **Everything's on the offbeat, or twice as dense/sparse as expected.** The tempo
  estimate hit a 2:1 octave (a common ambiguity). Pass the real tempo:
  `notesight song.mp3 --bpm 174`.
- **Which difficulty should I pick?** Presets target a notes-per-second density:

  | preset | ~notes/sec |
  |---|---|
  | `beginner` | ~1.2 |
  | `easy` | ~2.2 |
  | `medium` | ~3.2 |
  | `hard` | ~4.0 |
  | `expert` | ~5.0 |

- **Barely any notes.** Onset detection keys off transients — very ambient/quiet
  audio gives few of them. Try a higher difficulty or a more percussive track.
- **Beat Saber blocks feel too fast.** Reaction time is auto-set per difficulty
  (gentler on lower ones); drop a difficulty for more reaction window.
- **An `.mp3`/`.m4a` won't load.** Decoding uses a bundled ffmpeg (`imageio-ffmpeg`);
  if it errors, re-run `pip install -e .` so that dependency installs cleanly.

## Contributing

Run the tests with `pip install -e ".[dev]" && pytest`. New output formats only
need to implement the `ChartFormat` ABC in `notesight/formats/base.py` and
register in `notesight/formats/__init__.py` — nothing upstream changes.

## Legal

Chart only **original or royalty-free** music. The tool is fine to use freely;
just don't distribute charts of copyrighted songs or another game's assets.

## License

[MIT](LICENSE) © 2026 RelentlessOldMan
