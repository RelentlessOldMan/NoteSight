r"""build_mashup.py -- stitch a folder of songs into ONE long mashup, charted two
ways for both games:

  * Mashup Gauntlet -- one continuous chart whose note density RAMPS every ~30s and
    keeps climbing to the end (starts easy, gets brutal, beep-test style).
  * Mashup - Stamina -- each song's steady one-loop stamina chart laid end-to-end
    (five NPS paces), so it plays like a seamless cardio set.

Notes are TIME-spaced (1/nps apart), not beat-locked, so mixing songs of different
tempos is fine: BS notes land at the right seconds; DDR keeps each song at its own
bpm (stamina) or a nominal grid (gauntlet). Rests fall in real lulls (gated by the
mashup's energy envelope). Audio is wav/ogg/flac (read directly -- no mp3).

    python build_mashup.py <out_dir> <audio_or_folder>... [--title NAME] [--artist NAME]
    python build_mashup.py out/Mashup --dir path/to/songs/ --title "My Mashup"
"""
from __future__ import annotations

import bisect
import math
import os
import sys

import numpy as np
import soundfile as sf

from build_ddr_pack import resolve_audio, collect_paths
from build_bs_stamina import one_play, _write_looped_egg, LEAD_IN, STAMINA_TIERS
from notesight.formats.bs_letters import pick_alphabet
from notesight import SongMeta
from notesight.radar import compute_radar, predict_meter
from notesight.formats.stepmania import Note, notes_to_measures, _sanitize
from notesight.formats.beatsaber import write_pack
from notesight.stamina import build_stream
from notesight.version import ITG_GEN_VERSION, BS_GEN_VERSION, build_timestamp

NOMINAL_BPM = 120.0
NPS_START, NPS_STEP = 2.0, 0.5
SEG_SEC, GAP_BEATS, CROSSFADE = 30.0, 4.0, 1.5
# set by main() from CLI args:
TITLE = "Mashup"
ARTIST = "Unknown Artist"
BS_PACK = ""      # <out_dir>/BeatSaber
DDR_PACK = ""     # <out_dir>/DDR
AUDIO: list[str] = []      # resolved input audio paths, in play order


def build_mashup_audio():
    """Concat every input song (crossfaded) into one stereo track."""
    sr, clips, names = 48000, [], []
    for path in AUDIO:
        data, s = sf.read(path, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        clips.append(data)
        names.append(os.path.splitext(os.path.basename(path))[0])
    cf = int(CROSSFADE * sr)
    out = clips[0]
    for nxt in clips[1:]:
        n = min(cf, len(out), len(nxt))
        if n > 0:
            fade = np.linspace(1.0, 0.0, n)[:, None]
            out = np.concatenate([out[:-n], out[-n:] * fade + nxt[:n] * (1 - fade), nxt[n:]])
        else:
            out = np.concatenate([out, nxt])
    return out, sr, names


def _energy(data, sr, hop_sec=0.05):
    """Normalized RMS envelope over ~hop_sec windows (no file needed)."""
    mono = data.mean(axis=1)
    hop = max(1, int(hop_sec * sr))
    n = len(mono) // hop
    if n < 1:
        return np.array([0.0]), np.array([1.0])
    e = np.sqrt((mono[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return (np.arange(n) + 0.5) * hop / sr, e / (e.max() or 1.0)


def mashup_gauntlet_times(bpm, e_times, energy, song_dur, thresh=0.15):
    """NPS ramps 2.0, 2.5, ... every ~30s (snapped to a lull) until the song ends."""
    period = 60.0 / bpm
    measure = 4 * period
    floor = thresh * (max(energy) if len(energy) else 1.0)

    def e_at(t):
        if t < 0 or t > song_dur or not len(energy):
            return 0.0
        return float(energy[min(bisect.bisect_left(e_times, t), len(energy) - 1)])

    def lull(target):
        base = round(target / measure) * measure
        best, bestE = base, 1e9
        for k in range(-3, 4):
            tb = base + k * measure
            if 0 < tb < song_dur and e_at(tb) < bestE:
                best, bestE = tb, e_at(tb)
        return best

    times, marks = [], []
    t = math.ceil(LEAD_IN / measure) * measure
    nps = NPS_START
    while t < song_dur - measure:
        seg_end = min(lull(t + SEG_SEC), song_dur)
        if seg_end <= t + measure:
            seg_end = min(t + SEG_SEC, song_dur)
        marks.append((t, nps))
        dt, tt = 1.0 / nps, t
        while tt < seg_end - 1e-9:
            if e_at(tt) >= floor:
                times.append(tt)
            tt += dt
        t = seg_end + GAP_BEATS * period
        nps += NPS_STEP
    return times, marks


def _source_wavs():
    """Ordered input audio paths feeding the mashup (same order as the audio)."""
    return list(AUDIO)


def _mmss(t):
    return f"{int(t // 60)}:{t % 60:06.3f}"


def stitch_bs_mashup_stamina():
    """Mashup - Stamina = each song's ONE-loop BS chart pulled from the box, laid
    end-to-end (one play per song) -- audio units concatenated, charts offset to their
    segment. Exactly the individual-stamina recipe, but 8 different songs' loops once
    each instead of 1 song's loop tiled. NO re-generation over the mashup audio."""
    slots = [slot for _, slot in STAMINA_TIERS]
    stitched = {slot: [] for slot in slots}
    units, off, marks, sr = [], 0.0, [], 48000
    for wav in _source_wavs():
        bpm, loop_dur, sr, unit, loops = one_play(wav)     # <- one item from the box
        marks.append((off, os.path.splitext(os.path.basename(wav))[0], bpm))
        units.append(unit if unit.ndim > 1 else np.repeat(unit[:, None], 2, axis=1))
        for slot in slots:
            for b in loops[slot]:
                asec = off + b["_time"] * 60.0 / bpm       # song-beat -> absolute second
                if asec < LEAD_IN:                          # no block in your face at t=0
                    continue
                nb = dict(b)
                nb["_time"] = round(asec * NOMINAL_BPM / 60.0, 5)   # -> mashup beat
                stitched[slot].append(nb)
        off += loop_dur                                     # next song starts one loop on
    from notesight.formats.bs_assemble import guard_seams
    for slot in slots:
        stitched[slot].sort(key=lambda x: x["_time"])
        # clean the WHOLE stitched chart's song-to-song seams the correct-by-construction way
        # (benign resolution + validate) -- NOT the old mutating guard.
        stitched[slot] = guard_seams(stitched[slot], NOMINAL_BPM, None)

    ms_dir = os.path.join(BS_PACK, f"ROM Stamina - {TITLE}")
    os.makedirs(ms_dir, exist_ok=True)
    _write_looped_egg(np.concatenate(units), sr, 1, os.path.join(ms_dir, "song.egg"))
    meta = SongMeta(title=f"ROM Stamina - {TITLE}", artist=ARTIST, audio_path="",
                    bpm=NOMINAL_BPM, beat0=0.0, duration=off)
    write_pack(stitched, meta, BS_PACK, cover_src="", note_opts={"raw": True})

    rep = ["NoteSight generation report",
           f"generator: Beat Saber  {BS_GEN_VERSION}",
           f"generated: {build_timestamp()}",
           f"song: ROM Stamina - {TITLE}",
           f"total {off / 60:.1f} min  ({len(marks)} songs stitched end-to-end)",
           "NOTE: the individual per-song Stamina charts concatenated (one play each),",
           "NOT a fresh generation over the mashup audio.", "",
           "  -- song segments (each plays its own Stamina chart) --"]
    for (start, name, bpm) in marks:
        rep.append(f"    {_mmss(start):>10}  {name}  (bpm {bpm:.1f})")
    from notesight.formats.bs_report import counts_lines
    for slot in slots:
        rep += ["", f"  [{slot}]"] + counts_lines(stitched[slot])
    with open(os.path.join(ms_dir, "_generation.txt"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    return off / 60.0, len(marks)


def stitch_ddr_mashup_stamina():
    """DDR Mashup - Stamina = each song's ONE-loop DDR chart from the box, concatenated
    measure-aligned with EACH SONG AT ITS OWN BPM via #BPMS -- so no song's rhythm is
    force-quantized onto another tempo's grid. The per-song measures are byte-identical
    to that song's individual stamina chart. Audio units are placed so every song's
    downbeat lands on its segment's beat 0 (so a single global #OFFSET stays in sync)."""
    from build_ddr_stamina import one_play_ddr, STAMINA_TIERS as TIERS_DDR
    slots = [slot for _, slot in TIERS_DDR]
    songs = []
    for wav in _source_wavs():
        bpm, beat0, loop_dur, sr, unit, tiers = one_play_ddr(wav)   # one box item
        if unit.ndim == 1:
            unit = np.repeat(unit[:, None], 2, axis=1)
        songs.append(dict(name=os.path.splitext(os.path.basename(wav))[0], bpm=bpm,
                          beat0=beat0, loop_dur=loop_dur, sr=sr, unit=unit, tiers=tiers))
    beat0_0 = songs[0]["beat0"]
    sr = songs[0]["sr"]

    # Segment layout: each song is a whole number of measures at its own bpm. Audio is
    # placed at (sum of prior loop_durs) + (beat0_0 - beat0_i) so its downbeat lands on
    # the segment's beat 0 -- absorbing the per-song phase into the (silent) loop tails.
    cum_beat, cum_time = 0.0, 0.0
    for s in songs:
        s["Bbeat"] = cum_beat
        s["measures"] = int(round(s["loop_dur"] * s["bpm"] / 60.0 / 4.0))
        s["place"] = cum_time + (beat0_0 - s["beat0"])
        cum_beat += s["measures"] * 4
        cum_time += s["loop_dur"]
    end = max(s["place"] + s["loop_dur"] for s in songs)

    # Combined audio (additive: overlaps land in silent loop tails).
    buf = np.zeros((int(math.ceil(end * sr)) + sr, 2), dtype=np.float32)
    for s in songs:
        p = int(round(max(0.0, s["place"]) * sr))
        buf[p:p + len(s["unit"])] += s["unit"][:, :2]
    peak = float(np.max(np.abs(buf))) or 1.0
    if peak > 0.99:
        buf *= 0.99 / peak

    bpms = ",".join(f"{s['Bbeat']:.3f}={s['bpm']:.3f}" for s in songs)
    avg_bpm = sum(s["bpm"] for s in songs) / len(songs)

    # Per-tier body: each song's measures built in ITS OWN beats (fake seconds at bpm 60
    # so notes_to_measures works in beats), padded to its measure count, concatenated.
    EMPTY = "0000\n0000\n0000\n0000"
    blocks = []
    for slot in slots:
        parts, allnotes = [], []
        for s in songs:
            spb = 60.0 / s["bpm"]
            one, sub = s["tiers"][slot]
            # (a pre-downbeat note clamps to beat 0, exactly as notes_to_measures does
            # for the individual chart -- so segments stay byte-identical to the source)
            fake = [Note((n.time - s["beat0"]) / spb, n.lane) for n in one
                    if (n.time - s["beat0"]) / spb < s["measures"] * 4]
            b = notes_to_measures(fake, 60.0, 0.0, max_lines=sub * 4, allow_triplets=False)
            cnt = b.count("\n,\n") + 1
            if cnt < s["measures"]:
                b = b.rstrip("\n") + ("\n,\n" + EMPTY) * (s["measures"] - cnt)
            parts.append(b.rstrip("\n"))
            allnotes += [Note(s["place"] + n.time, n.lane) for n in one]
        body = "\n,\n".join(parts) + "\n"
        radar = compute_radar(allnotes, avg_bpm, 0.0, end)
        meter = predict_meter(radar, slot)
        blocks.append("#NOTES:\n     dance-single:\n     NoteSight:\n"
                      f"     {slot}:\n     {meter}:\n     {radar.sm_field()}:\n{body};\n")

    safe = f"ROM Stamina - {TITLE}"
    song_dir = os.path.join(DDR_PACK, safe)
    os.makedirs(song_dir, exist_ok=True)
    music = f"ROM Stamina - {TITLE}.ogg"
    with sf.SoundFile(os.path.join(song_dir, music), "w", samplerate=sr, channels=2,
                      format="OGG", subtype="VORBIS") as out:
        for i in range(0, len(buf), 65536):
            out.write(buf[i:i + 65536])
    header = (f"#TITLE:{safe};\n#ARTIST:{ARTIST};\n#CREDIT:NoteSight Stamina;\n"
              f"#GENERATOR:NoteSight ITG {ITG_GEN_VERSION};\n#MUSIC:{music};\n"
              f"#OFFSET:{-beat0_0:.3f};\n#SAMPLESTART:0.000;\n#SAMPLELENGTH:12.000;\n"
              f"#SELECTABLE:YES;\n#DISPLAYBPM:{songs[0]['bpm']:.3f};\n#BPMS:{bpms};\n\n")
    with open(os.path.join(song_dir, safe + ".sm"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write(header + "\n".join(blocks))

    rep = ["NoteSight generation report",
           f"generator: ITG / StepMania  {ITG_GEN_VERSION}",
           f"generated: {build_timestamp()}",
           f"song: ROM Stamina - {TITLE}",
           f"total {end / 60:.1f} min  ({len(songs)} songs stitched end-to-end)",
           "NOTE: the individual per-song Stamina charts concatenated (one play each),",
           "each kept at its OWN bpm via #BPMS -- NOT a fresh gen or a 120-quantize.", "",
           "  -- song segments (start beat / start time / bpm) --"]
    for s in songs:
        t0 = s["place"] + s["beat0"]
        rep.append(f"    beat {s['Bbeat']:>7.0f}  {_mmss(t0):>10}  {s['name']:12} "
                   f"(bpm {s['bpm']:.3f}, {s['measures']} measures)")
    with open(os.path.join(song_dir, "_generation.txt"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(rep) + "\n")
    return end / 60.0, len(songs)


def main(argv):
    global TITLE, ARTIST, BS_PACK, DDR_PACK, AUDIO
    rest = list(argv)
    if len(rest) < 2:
        print(__doc__)
        return 2
    out_dir = rest.pop(0)

    def _opt(flag, default):
        if flag in rest:
            i = rest.index(flag)
            val = rest[i + 1]
            del rest[i:i + 2]
            return val
        return default
    TITLE = _opt("--title", "Mashup")
    ARTIST = _opt("--artist", "Unknown Artist")
    BS_PACK = os.path.join(out_dir, "BeatSaber")
    DDR_PACK = os.path.join(out_dir, "DDR")
    os.makedirs(BS_PACK, exist_ok=True)
    os.makedirs(DDR_PACK, exist_ok=True)
    for s, _ in collect_paths(rest):
        a = resolve_audio(s)
        if a:
            AUDIO.append(a)
        else:
            print(f"!! skip (no audio): {s}", file=sys.stderr)
    if len(AUDIO) < 2:
        print("need at least 2 resolvable songs to make a mashup", file=sys.stderr)
        return 2

    print("== building mashup audio ==")
    data, sr, names = build_mashup_audio()
    song_dur = len(data) / sr
    print(f"  {len(names)} songs -> {song_dur / 60:.1f} min: {', '.join(names)}")

    bpm, beat0 = NOMINAL_BPM, 0.0
    e_times, energy = _energy(data, sr)
    times, marks = mashup_gauntlet_times(bpm, e_times, energy, song_dur)
    notes = [Note(t, lane) for t, lane in zip(times, build_stream(len(times)))]
    nps_lo, nps_hi = marks[0][1], marks[-1][1]
    print(f"== MASHUP GAUNTLET: {len(marks)} levels, {nps_lo:.1f} -> {nps_hi:.1f} NPS, "
          f"{len(notes)} notes, {song_dur / 60:.1f} min ==")

    gtitle = f"ROM Gauntlet - {TITLE}"
    safe = _sanitize(gtitle)
    # ---- Beat Saber ----
    bsong = os.path.join(BS_PACK, safe)
    os.makedirs(bsong, exist_ok=True)
    egg = os.path.join(bsong, "song.egg")
    if not (os.path.isfile(egg) and os.path.getsize(egg) > 100):
        _write_looped_egg(data, sr, 1, egg)
    meta = SongMeta(title=gtitle, artist=ARTIST, audio_path="",
                    bpm=bpm, beat0=beat0, duration=song_dur)
    # THE ONE GENERATOR: letters, composed onto the gauntlet's (ever-denser) ramp times. The
    # ramp comes from the note TIMING (spacing tightens as NPS climbs); letters fill it, no
    # peak cap (it's meant to ramp brutally).
    from notesight.formats.bs_assemble import assemble_letters
    gb, _ = assemble_letters([n.time for n in notes], bpm, seed=9, peak_cap=None,
                             alphabet=pick_alphabet(9), space=False)   # gauntlet = intentional brutal ramp
    write_pack({"Challenge": gb}, meta, BS_PACK, cover_src="", note_opts={"raw": True})

    # ---- DDR ----
    dsong = os.path.join(DDR_PACK, safe)
    os.makedirs(dsong, exist_ok=True)
    music = safe + ".ogg"
    ogg = os.path.join(dsong, music)
    if not (os.path.isfile(ogg) and os.path.getsize(ogg) > 100):
        with sf.SoundFile(ogg, "w", samplerate=sr, channels=2, format="OGG",
                          subtype="VORBIS") as out:
            for i in range(0, len(data), 65536):
                out.write(data[i:i + 65536])
    body = notes_to_measures(notes, bpm, beat0, max_lines=48, allow_triplets=False)
    radar = compute_radar(notes, bpm, beat0, song_dur)
    meter = predict_meter(radar, "Challenge")
    header = (f"#TITLE:{gtitle};\n#ARTIST:{ARTIST};\n#CREDIT:NoteSight Gauntlet;\n"
              f"#GENERATOR:NoteSight ITG {ITG_GEN_VERSION};\n"
              f"#MUSIC:{music};\n#OFFSET:{-beat0:.3f};\n#SAMPLESTART:0.000;\n"
              f"#SAMPLELENGTH:12.000;\n#SELECTABLE:YES;\n#DISPLAYBPM:{bpm:.3f};\n"
              f"#BPMS:0.000={bpm:.3f};\n\n")
    block = ("#NOTES:\n     dance-single:\n     NoteSight:\n     Challenge:\n"
             f"     {meter}:\n     {radar.sm_field()}:\n{body};\n")
    with open(os.path.join(dsong, safe + ".sm"), "w", encoding="utf-8", newline="\n") as f:
        f.write(header + block)

    # ---- Mashup STAMINA: the SAME per-song Stamina charts from the box, laid end-to-end
    # (one play each) -- NOT a fresh gen over the concatenated audio. BS lays them on one
    # nominal grid (it places by time); DDR keeps each song at its own bpm via #BPMS. ----
    ms_min, ms_n = stitch_bs_mashup_stamina()
    dd_min, dd_n = stitch_ddr_mashup_stamina()

    print(f"  wrote 'Mashup - Stamina' (BS {ms_n} songs / {ms_min:.1f} min, "
          f"DDR {dd_n} songs / {dd_min:.1f} min) + '{TITLE}' (NPS {nps_lo:.1f}-{nps_hi:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
