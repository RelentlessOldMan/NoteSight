r"""build_ddr_pack.py -- turn a batch of audio files into a StepMania song GROUP (pack).

Each song gets ONE .sm holding every difficulty (Beginner..Expert), so it plays
like a real pack (pick your difficulty in-game). The expensive DSP (analyze_audio)
runs once per song; all five difficulties chart off that single analysis.

Layout produced (a StepMania song group -- drop <out_group_dir> into Songs/):
    <out_group_dir>\<Title>\
        <Title>.<ext>     audio (charted from + shipped -- same file, tight sync)
        <Title>.sm        multi-difficulty chart
        <Title>.png       banner   (from the song's artwork, if found)
        <Title>-bg.png    background(same art)
        <Title>.lrc       lyrics   (copied if present)

Usage:
    python build_ddr_pack.py <out_group_dir> <audio_or_song_folder> [more...]
    python build_ddr_pack.py <out_group_dir> --dir path/to/songs/   # every song folder in a dir
    python build_ddr_pack.py <out_group_dir> --list songs.txt       # one path per line

A "song path" may be the audio file itself or the folder holding it; a folder is
resolved to its <folder-name>.wav/.mp3. Title = the audio file stem. Artist = a
`.artist` sidecar file next to the audio if present, else "Unknown Artist" (set
one per song, or drop a one-line `.artist` file in each song folder).
"""
from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import subprocess
import sys

from notesight import ChartSpec, DIFFICULTIES, analyze_audio, build_chart
from notesight.audio_io import load_audio
from notesight.radar import compute_radar, predict_meter
from notesight.formats.stepmania import Note, notes_to_measures, _sanitize

# The DSP analysis (onsets/grid/energy/structure) is deterministic per audio file
# but SLOW; cache it on disk so charter-only rebuilds skip it entirely. Bump
# ANALYSIS_VERSION when the analysis pipeline changes (file mtime/size handles the
# audio content changing).
CACHE_DIR = "_cache"
ANALYSIS_VERSION = 1


def cached_analyze(audio_path: str, lrc_path: str | None):
    """analyze_audio(), memoized on disk by (audio mtime/size, lrc mtime, version)."""
    st = os.stat(audio_path)
    lrc_m = os.path.getmtime(lrc_path) if lrc_path and os.path.isfile(lrc_path) else 0
    key = hashlib.md5(
        f"{os.path.abspath(audio_path)}|{st.st_mtime_ns}|{st.st_size}"
        f"|{lrc_path}|{lrc_m}|v{ANALYSIS_VERSION}".encode()).hexdigest()
    cf = os.path.join(CACHE_DIR, key + ".pkl")
    if os.path.isfile(cf):
        try:
            with open(cf, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    mono, sr = load_audio(audio_path)
    analysis = analyze_audio(mono, sr, lrc_path=lrc_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cf, "wb") as f:
            pickle.dump(analysis, f)
    except Exception:
        pass
    return analysis


def cached_grid_energy(audio_path: str):
    """(bpm, beat0, e_times, energy, song_dur), memoized on disk -- for the stamina/
    gauntlet builders, which only need the beat grid + energy (not full analysis)."""
    from notesight.onsets import detect_onsets, energy_envelope
    from notesight.beatgrid import estimate_grid
    st = os.stat(audio_path)
    key = hashlib.md5(
        f"ge|{os.path.abspath(audio_path)}|{st.st_mtime_ns}|{st.st_size}"
        f"|v{ANALYSIS_VERSION}".encode()).hexdigest()
    cf = os.path.join(CACHE_DIR, key + ".pkl")
    if os.path.isfile(cf):
        try:
            with open(cf, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    mono, sr = load_audio(audio_path)
    grid = estimate_grid(mono, sr, onsets=detect_onsets(mono, sr))
    e_times, energy = energy_envelope(mono, sr)
    result = (grid.bpm, grid.beat0, e_times, energy, len(mono) / sr)
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(cf, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass
    return result

# All five difficulties, in the order StepMania lists them. (preset, slot).
TIERS = [
    ("beginner", "Beginner"),
    ("easy", "Easy"),
    ("medium", "Medium"),
    ("hard", "Hard"),
    ("expert", "Challenge"),
]

AUDIO_EXTS = (".wav", ".ogg", ".flac", ".mp3", ".m4a")
ART_PREFS = (" - source.png", " - ai-01.png", " - thumbnail.jpg",
             " - yt-thumb.png", ".png", ".jpg")


def resolve_audio(path: str) -> str | None:
    """A song path is either the audio file or the folder holding it."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        stem = os.path.basename(path.rstrip("\\/"))
        # Prefer wav (PCM -> no decoder-delay, tightest sync), then others.
        for ext in AUDIO_EXTS:
            cand = os.path.join(path, stem + ext)
            if os.path.isfile(cand):
                return cand
        # Fall back to any audio file in the folder, preferring by ext priority
        # (wav first) -- archive folders name the file by title, not folder.
        auds = [f for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
        if auds:
            auds.sort(key=lambda f: AUDIO_EXTS.index(os.path.splitext(f)[1].lower()))
            return os.path.join(path, auds[0])
    return None


def band_of(audio_path: str) -> str:
    """Fallback artist when no `.artist` sidecar is present. We deliberately don't
    guess from the folder layout (that produced surprising results); set a real
    artist with a one-line `.artist` file in the song's folder."""
    return "Unknown Artist"


def artist_of(folder: str, audio_path: str) -> str:
    """Artist for a song: prefer a `.artist` sidecar file in the song folder (so a
    staged copy is self-contained regardless of where it came from); otherwise
    "Unknown Artist"."""
    af = os.path.join(folder, ".artist")
    if os.path.isfile(af):
        try:
            a = open(af, encoding="utf-8").read().strip()
            if a:
                return a
        except Exception:
            pass
    return band_of(audio_path)


def find_artwork(folder: str, stem: str) -> str | None:
    for suffix in ART_PREFS:
        cand = os.path.join(folder, stem + suffix)
        if os.path.isfile(cand):
            return cand
    return None


def find_lrc(folder: str, stem: str) -> str | None:
    cand = os.path.join(folder, stem + ".lrc")
    return cand if os.path.isfile(cand) else None


def save_png(src: str, dst: str) -> None:
    """Write a REAL PNG (source art is often JPEG-with-a-.png-name, which makes
    StepMania warn). Falls back to a plain copy if Pillow isn't available."""
    try:
        from PIL import Image
        Image.open(src).convert("RGB").save(dst, "PNG")
    except Exception:
        shutil.copyfile(src, dst)


def render_tier(analysis, preset: str, slot: str, bpm, beat0, duration):
    """Chart one difficulty off a shared analysis -> (meter, radar, body_str)."""
    # Let the difficulty preset drive the rhythm ceiling (grid + triplets), so
    # easy tiers stay on-beat (1/4, 1/8) and finer rhythms unlock as it climbs.
    d = DIFFICULTIES[preset]
    spec = ChartSpec(difficulty=preset,
                     max_subdivision=d.max_subdivision,
                     allow_triplets=d.allow_triplets)
    r = build_chart(spec, analysis)
    # Radar/meter from the SNAPPED notes (what actually exports).
    radar = compute_radar(r.notes, bpm, beat0, duration)
    meter = predict_meter(radar, slot)
    body = notes_to_measures(r.notes, bpm, beat0,
                             spec.max_subdivision, spec.allow_triplets)
    return meter, radar, body, len(r.notes)


def _to_mp3(src: str, dst: str, bitrate: str = "192k") -> bool:
    """Transcode any source audio to MP3 via ffmpeg (libmp3lame). We SHIP mp3, never
    the source .wav -- a lossless wav is ~10x an mp3 and bloats the pack for no
    audible gain in-game. Returns True on success."""
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-i", src, "-vn", "-map", "a", "-codec:a", "libmp3lame",
           "-b:a", bitrate, dst]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (p.returncode == 0 and os.path.isfile(dst)
                and os.path.getsize(dst) > 100)
    except Exception:
        return False


def _place_audio(audio_path: str, song_dir: str, safe: str) -> str:
    """Put COMPRESSED audio in the song folder and return its filename. Clears any
    stale audio first (e.g. an old .wav from a previous build) so a format switch
    doesn't leave both behind. mp3 sources pass through; everything else transcodes."""
    for f in os.listdir(song_dir):
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
            try:
                os.remove(os.path.join(song_dir, f))
            except OSError:
                pass
    if os.path.splitext(audio_path)[1].lower() == ".mp3":
        name = safe + ".mp3"
        shutil.copyfile(audio_path, os.path.join(song_dir, name))
        return name
    name = safe + ".mp3"
    if _to_mp3(audio_path, os.path.join(song_dir, name)):
        return name
    # ffmpeg unavailable -> last resort, ship the source as-is (keeps the pack valid).
    ext = os.path.splitext(audio_path)[1].lower()
    name = safe + ext
    shutil.copyfile(audio_path, os.path.join(song_dir, name))
    return name


def _existing_audio(song_dir: str, safe: str) -> str:
    """Name of the audio already sitting in an existing song folder."""
    for ext in AUDIO_EXTS:
        if os.path.isfile(os.path.join(song_dir, safe + ext)):
            return safe + ext
    for f in sorted(os.listdir(song_dir)):
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
            return f
    return ""


def build_song(audio_path: str, out_group: str, title_override: str = "",
               sm_only: bool = False) -> str:
    folder = os.path.dirname(audio_path)
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    title = title_override or stem
    artist = artist_of(folder, audio_path)

    print(f"\n=== {title}  [{artist}] ===")
    analysis = cached_analyze(audio_path, find_lrc(folder, stem))
    bpm, beat0, duration = analysis.grid.bpm, analysis.grid.beat0, analysis.duration
    print(f"  {duration:.1f}s | {len(analysis.onsets)} onsets | "
          f"BPM {bpm:.2f} | offset {-beat0:+.3f}")

    blocks = []
    for preset, slot in TIERS:
        meter, radar, body, n = render_tier(analysis, preset, slot,
                                             bpm, beat0, duration)
        print(f"    {slot:<10} meter {meter:>2}  {n:>4} notes  "
              f"(str {radar.stream:.2f} air {radar.air:.2f} chaos {radar.chaos:.2f})")
        blocks.append(
            "#NOTES:\n"
            "     dance-single:\n"
            "     NoteSight:\n"
            f"     {slot}:\n"
            f"     {meter}:\n"
            f"     {radar.sm_field()}:\n"
            f"{body};\n"
        )

    # --- assemble the song folder ---
    safe = _sanitize(title)
    song_dir = os.path.join(out_group, safe)

    banner_line = bg_line = lrc_line = ""
    if sm_only:
        # Regen the chart in place: reuse whatever audio/art/lrc is already on
        # disk, only rewrite the .sm. Nothing is copied or deleted.
        if not os.path.isdir(song_dir):
            raise FileNotFoundError(f"--sm-only but no existing folder: {song_dir}")
        music_name = _existing_audio(song_dir, safe)
        if os.path.isfile(os.path.join(song_dir, safe + ".png")):
            banner_line = f"#BANNER:{safe}.png;\n"
        if os.path.isfile(os.path.join(song_dir, safe + "-bg.png")):
            bg_line = f"#BACKGROUND:{safe}-bg.png;\n"
        if os.path.isfile(os.path.join(song_dir, safe + ".lrc")):
            lrc_line = f"#LYRICSPATH:{safe}.lrc;\n"
    else:
        os.makedirs(song_dir, exist_ok=True)
        music_name = _place_audio(audio_path, song_dir, safe)   # ships mp3, not wav

        art = find_artwork(folder, stem)
        if art:
            banner_name, bg_name = safe + ".png", safe + "-bg.png"
            # Save REAL PNGs -- the source art is often JPEG-with-a-.png-name,
            # which makes StepMania log "is really jpeg" + cache warnings.
            save_png(art, os.path.join(song_dir, banner_name))
            save_png(art, os.path.join(song_dir, bg_name))
            banner_line = f"#BANNER:{banner_name};\n"
            bg_line = f"#BACKGROUND:{bg_name};\n"
        lrc = find_lrc(folder, stem)
        if lrc:
            lrc_name = safe + ".lrc"
            shutil.copyfile(lrc, os.path.join(song_dir, lrc_name))
            lrc_line = f"#LYRICSPATH:{lrc_name};\n"

    # Music-wheel preview: start ~1/3 into the song (like Won't Stop's 34.25s),
    # kept inside the track so the 12s sample doesn't run past the end.
    sample_start = round(min(duration * 0.33, max(0.0, duration - 15.0)), 3)
    header = (
        f"#TITLE:{title};\n"
        f"#ARTIST:{artist};\n"
        f"#CREDIT:NoteSight;\n"
        f"{banner_line}{bg_line}{lrc_line}"
        f"#MUSIC:{music_name};\n"
        f"#OFFSET:{-beat0:.3f};\n"
        f"#SAMPLESTART:{sample_start:.3f};\n"
        f"#SAMPLELENGTH:12.000;\n"
        f"#SELECTABLE:YES;\n"
        f"#DISPLAYBPM:{bpm:.3f};\n"
        f"#BPMS:0.000={bpm:.3f};\n\n"
    )
    sm_path = os.path.join(song_dir, safe + ".sm")
    with open(sm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n".join(blocks))
    print(f"  wrote {sm_path}")
    return sm_path


def split_entry(line: str):
    """A song entry is 'path' or 'path | Title' (explicit title override)."""
    if "|" in line:
        path, title = line.split("|", 1)
        return path.strip(), title.strip()
    return line.strip(), ""


def collect_paths(argv):
    paths = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--list", "-L"):
            i += 1
            with open(argv[i], encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        paths.append(split_entry(line))
        elif a == "--dir":
            # A pack folder: build every song sub-folder inside it.
            i += 1
            src = argv[i]
            for name in sorted(os.listdir(src)):
                sub = os.path.join(src, name)
                if os.path.isdir(sub):
                    paths.append((sub, ""))
        else:
            paths.append(split_entry(a))
        i += 1
    return paths


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    # --sm-only: regen charts in place (reuse on-disk audio/art, rewrite only .sm)
    sm_only = "--sm-only" in argv
    rest = [a for a in argv if a != "--sm-only"]
    out_group, song_args = rest[0], rest[1:]
    if not sm_only:
        os.makedirs(out_group, exist_ok=True)

    songs = collect_paths(song_args)
    ok = 0
    for s, title in songs:
        audio = resolve_audio(s)
        if not audio:
            print(f"!! skip (no audio found): {s}", file=sys.stderr)
            continue
        try:
            build_song(audio, out_group, title, sm_only=sm_only)
            ok += 1
        except Exception as e:  # keep the batch going
            print(f"!! FAILED {s}: {e}", file=sys.stderr)
    print(f"\nDone: {ok}/{len(songs)} songs{' (.sm only)' if sm_only else ''} "
          f"-> {out_group}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
