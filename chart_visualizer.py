r"""chart_visualizer.py -- ChartSight: native 3D rhythm-chart visualizer (Ursina / Panda3D).

Auto-detects the chart type and shows the right PLAYFIELD -- same window, audio,
timeline, and feedback either way:
  * Beat Saber (folder has info.dat): lit 3D blocks fly at you (cut arrows, red/blue).
  * StepMania / DDR (folder has a .sm): 4 directional arrows on 4 lanes, coloured by
    rhythm subdivision (red 1/4, blue 1/8, green 1/12, yellow 1/16, ...) with freeze holds.
Synced to streamed audio so you can study a chart's flow/readability -- and drop
timestamped feedback -- without booting a headset or a pad.

    python chart_visualizer.py "<song folder>" [Difficulty]
    python chart_visualizer.py "out/Neon" Expert                 # a Beat Saber folder (-f beatsaber)
    python chart_visualizer.py "../DDR Packs/Relentless/NPC" Medium   # a StepMania .sm folder

Controls: Space / click empty space = play-pause . drag or click the bar = seek
. wheel = scrub . Diff -/+ buttons = difficulty . Speed -/+ = playback rate
. Look -/+ = scroll distance . Note button = save timestamped feedback . Esc = quit
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import sys
import textwrap
from collections import Counter
from datetime import datetime
from time import perf_counter

# ----------------------------------------------------------------------------- data

CUTV = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
        4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: None}   # dir on face; 8=dot
RED, BLUE, BOMB = 0, 1, 3


def load_song(folder):
    info_p = None
    for nm in ("info.dat", "Info.dat"):
        if os.path.isfile(os.path.join(folder, nm)):
            info_p = os.path.join(folder, nm)
            break
    if not info_p:
        raise SystemExit(f"no info.dat in {folder}")
    info = json.load(open(info_p, encoding="utf-8"))
    bpm = float(info["_beatsPerMinute"])
    audio_p = os.path.join(folder, info.get("_songFilename") or "")
    if not os.path.isfile(audio_p):
        cands = glob.glob(os.path.join(folder, "*.egg")) + glob.glob(os.path.join(folder, "*.ogg"))
        audio_p = cands[0] if cands else ""
    diffs = []
    for bset in info.get("_difficultyBeatmapSets", []):
        if bset.get("_beatmapCharacteristicName", "Standard") != "Standard":
            continue
        for b in bset.get("_difficultyBeatmaps", []):
            p = os.path.join(folder, b["_beatmapFilename"])
            if os.path.isfile(p):
                diffs.append({"name": b["_difficulty"], "rank": b.get("_difficultyRank", 0),
                              "njs": b.get("_noteJumpMovementSpeed", 0), "path": p})
    diffs.sort(key=lambda d: d["rank"])
    return bpm, audio_p, info.get("_songName", os.path.basename(folder)), diffs


def load_notes(path, bpm):
    dat = json.load(open(path, encoding="utf-8"))
    out = []
    for n in dat.get("_notes", []):
        out.append((round(n["_time"] * 60.0 / bpm, 4), n.get("_lineIndex", 0),
                    n.get("_lineLayer", 0), n.get("_type", 0), n.get("_cutDirection", 8)))
    out.sort(key=lambda x: x[0])
    return out


def stats(notes, duration):
    ts = [n[0] for n in notes if n[3] in (RED, BLUE)]
    total = len(ts)
    reds = sum(1 for n in notes if n[3] == RED)
    blues = sum(1 for n in notes if n[3] == BLUE)
    bombs = sum(1 for n in notes if n[3] == BOMB)
    tc = Counter(round(t, 3) for t in ts)
    doubles = sum(1 for c in tc.values() if c >= 2)
    span = (ts[-1] - ts[0]) if total > 1 else (duration or 1.0)
    avg = total / span if span > 0 else 0.0
    peak, lo = 0, 0
    for i in range(total):
        while ts[i] - ts[lo] >= 1.0:
            lo += 1
        peak = max(peak, i - lo + 1)
    return {"total": total, "reds": reds, "blues": blues, "bombs": bombs,
            "doubles": doubles, "avg": avg, "peak": peak, "times": ts}


# --------------------------------------------------------------------- DDR / .sm
# The viewer auto-detects the chart type: a folder with info.dat is a Beat Saber map;
# a folder with a .sm is a StepMania/DDR song. Same window, audio, timeline, feedback --
# only the PLAYFIELD differs (BS = 3D blocks on a 4x3 grid; DDR = 4 directional arrows
# on 4 lanes, coloured by rhythm subdivision so the beat reads at a glance).

DDR_TAP, DDR_MINE = 0, 1
# dance-single column order L,D,U,R -> a CUTV arrow key (2=left,1=down,0=up,3=right).
PANEL_CUT = {0: 2, 1: 1, 2: 0, 3: 3}
# StepMania note-quantization colours (which subdivision of the measure a note lands on).
# EXACT colours of StepMania 5.1's "default" noteskin, read from its "_arrow 1x8" sprite
# pixels (the frames are indexed by quantization: 4th, 8th, 12th, 16th, 24th, 32nd, 48th, 64th).
# So the viewer matches the pad pixel-for-pixel: on-beat 4ths are the ITG red-orange, triplets
# (12ths) are GREEN, and yellow is 16ths (NOT 12ths).
QUANT_HEX = {1: "#ea2501", 2: "#ea2501", 4: "#ea2501",      # 1/4 & coarser = red (on beat)
             8: "#0170ea",                                   # 1/8  = blue
             3: "#58ea01", 6: "#58ea01", 12: "#58ea01",      # 1/12 (triplets/3rds) = green
             16: "#eac701",                                  # 1/16 = yellow
             24: "#6f01ea",                                  # 1/24 = purple
             32: "#01ea85",                                  # 1/32 = teal
             48: "#ea0164", 64: "#6d905a"}                   # 1/48 pink, 1/64 green
QUANT_DEFAULT = "#c8ccd8"                                     # anything odd = grey


def _quant_hex(r, R):
    """Colour for a note at row r of R rows in a measure, by its rhythm subdivision."""
    from math import gcd
    den = R // gcd(r, R) if r else 1        # measure divided into `den` equal parts
    return QUANT_HEX.get(den, QUANT_DEFAULT)


def _sm_helpers():
    """Import the .sm timing helpers from eval/smparse.py (kept out of the public
    top-level imports so a BS-only run never needs the eval harness on the path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.join(here, "eval") not in sys.path:
        sys.path.insert(0, os.path.join(here, "eval"))
    from smparse import _split_tags, _parse_pairs, _tag_value, _elapsed  # noqa
    return _split_tags, _parse_pairs, _tag_value, _elapsed


def load_song_sm(folder):
    """Parse a StepMania song folder. Returns (bpm, audio_path, title, diffs) mirroring
    load_song(); each diff carries its parsed DDR notes so the caller doesn't re-read."""
    sm_files = sorted(glob.glob(os.path.join(folder, "*.sm"))
                      + glob.glob(os.path.join(folder, "*.SM")))
    if not sm_files:
        raise SystemExit(f"no .sm in {folder}")
    split_tags, parse_pairs, tag_value, elapsed = _sm_helpers()
    text = open(sm_files[0], encoding="utf-8", errors="replace").read()
    title, music, offset = os.path.basename(folder), "", 0.0
    bpms, stops = [(0.0, 120.0)], []
    charts = []
    for name, raw in split_tags(text):
        if name == "TITLE":
            title = tag_value(raw) or title
        elif name == "MUSIC":
            music = tag_value(raw)
        elif name == "OFFSET":
            try:
                offset = float(tag_value(raw))
            except ValueError:
                offset = 0.0
        elif name == "BPMS":
            bpms = parse_pairs(tag_value(raw)) or bpms
        elif name == "STOPS":
            stops = parse_pairs(tag_value(raw))
        elif name == "NOTES":
            try:                                   # one malformed chart can't sink the song
                c = _parse_sm_notes(raw, offset, bpms, stops, elapsed)
            except Exception:
                c = None
            if c:
                charts.append(c)
    bpm = bpms[0][1] if bpms else 120.0
    audio_p = os.path.join(folder, music) if music else ""
    if not os.path.isfile(audio_p):
        cands = (glob.glob(os.path.join(folder, "*.ogg")) + glob.glob(os.path.join(folder, "*.mp3"))
                 + glob.glob(os.path.join(folder, "*.wav")))
        audio_p = cands[0] if cands else ""
    RANK = {"beginner": 0, "easy": 1, "medium": 2, "hard": 3, "challenge": 4, "edit": 5}
    diffs = [{"name": c["difficulty"], "rank": RANK.get(c["difficulty"].lower(), 9),
              "njs": c["meter"], "notes": c["notes"]} for c in charts]
    diffs.sort(key=lambda d: d["rank"])
    return bpm, audio_p, title, diffs


def _parse_sm_notes(raw, offset, bpms, stops, elapsed):
    """One #NOTES payload -> {difficulty, meter, notes}. notes are unified 7-tuples
    (time, col, 0, kind, cut_dir, quant_hex, hold_end_time) for the DDR playfield."""
    parts = raw.split(":", 5)
    if len(parts) < 6:
        return None
    if parts[0].strip() != "dance-single":
        return None
    difficulty = parts[2].strip()
    try:
        meter = int(float(parts[3].strip()))
    except ValueError:
        meter = 0
    import re as _re
    body = _re.sub(r"//[^\n]*", "", parts[5])
    notes, holds = [], {}                       # holds: col -> index into notes (open hold head)
    for m_idx, measure in enumerate(body.split(",")):
        lines = [ln.strip() for ln in measure.splitlines() if ln.strip()]
        lines = [ln for ln in lines if set(ln) <= set("0123456789M")]
        R = len(lines)
        if R == 0:
            continue
        for r, line in enumerate(lines):
            beat = 4.0 * m_idx + 4.0 * r / R
            t = elapsed(beat, offset, bpms, stops)
            qh = _quant_hex(r, R)
            for col, ch in enumerate(line[:4]):
                if ch in "124":                 # tap / hold-head / roll-head
                    if ch in "24":
                        holds[col] = len(notes)
                    notes.append([t, col, 0, DDR_TAP, PANEL_CUT[col], qh, None])
                elif ch == "3" and col in holds:  # hold/roll tail -> close its head
                    notes[holds.pop(col)][6] = t
                elif ch == "M":                 # mine
                    notes.append([t, col, 0, DDR_MINE, PANEL_CUT[col], "#2a2c36", None])
    notes.sort(key=lambda n: n[0])
    return {"difficulty": difficulty, "meter": meter, "notes": [tuple(n) for n in notes]}


def stats_ddr(notes, duration):
    """DDR-side stats mirroring stats(): total onsets, jumps (>=2 same time), avg/peak NPS."""
    ts = [n[0] for n in notes if n[3] == DDR_TAP]
    total = len(ts)
    tc = Counter(round(t, 3) for t in ts)
    jumps = sum(1 for c in tc.values() if c >= 2)
    distinct = sorted(tc)                        # NPS is on distinct rows (a jump = 1)
    span = (distinct[-1] - distinct[0]) if len(distinct) > 1 else (duration or 1.0)
    avg = len(distinct) / span if span > 0 else 0.0
    peak, lo = 0, 0
    for i in range(len(distinct)):
        while distinct[i] - distinct[lo] >= 1.0:
            lo += 1
        peak = max(peak, i - lo + 1)
    return {"total": total, "reds": 0, "blues": 0, "bombs": 0, "doubles": jumps,
            "avg": avg, "peak": peak, "times": distinct}


def detect_mode(folder):
    """'bs' if the folder is a Beat Saber map, 'ddr' if a StepMania song, else None."""
    if not folder or not os.path.isdir(folder):
        return None
    if (os.path.isfile(os.path.join(folder, "info.dat"))
            or os.path.isfile(os.path.join(folder, "Info.dat"))):
        return "bs"
    if glob.glob(os.path.join(folder, "*.sm")) or glob.glob(os.path.join(folder, "*.SM")):
        return "ddr"
    return None


# ------------------------------------------------------------------------- 3D app

APP_NAME = "ChartSight"                 # the viewer's name (pairs with NoteSight; views BS + DDR)
VERSION = "0.1"
BUILD = datetime.fromtimestamp(os.path.getmtime(os.path.abspath(__file__))).strftime("%Y-%m-%d %H:%M")
COPYRIGHT = "(c) 2026 RelentlessOldMan"
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback")


def compute_waveform(path, bins):
    """Peak-amplitude envelope over `bins` slices of the whole song, streamed block
    by block (never loads the 30-min track into RAM). Normalized 0..1."""
    env = [0.0] * bins
    try:
        import numpy as np
        import soundfile as sf
        total = sf.info(path).frames
        per = max(1, total // bins)
        with sf.SoundFile(path) as f:
            for i in range(bins):
                b = f.read(per, dtype="float32", always_2d=True)
                if len(b) == 0:
                    break
                env[i] = float(np.abs(b).max())
        mx = max(env) or 1.0
        env = [e / mx for e in env]
    except Exception:
        pass
    return env


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chartsight.json")
_OLD_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sabersight.json")


def load_config():
    for p in (CONFIG_PATH, _OLD_CONFIG):        # fall back to the pre-rename config once
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
    return {}


def save_config(d):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass


def _has_info(folder):
    return bool(folder) and (os.path.isfile(os.path.join(folder, "info.dat"))
                             or os.path.isfile(os.path.join(folder, "Info.dat")))


def pick_folder(initial=None):
    """Native Win32 Open dialog via ctypes (comdlg32.GetOpenFileNameW) -- in-process,
    no extra toolkit/process, no event loop to fight Panda3D. Pick info.dat / any .dat;
    returns its folder."""
    try:
        import ctypes
        from ctypes import wintypes
        vp = ctypes.c_void_p

        class OFN(ctypes.Structure):
            _fields_ = [("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND),
                        ("hInstance", vp), ("lpstrFilter", wintypes.LPCWSTR),
                        ("lpstrCustomFilter", wintypes.LPWSTR), ("nMaxCustFilter", wintypes.DWORD),
                        ("nFilterIndex", wintypes.DWORD), ("lpstrFile", wintypes.LPWSTR),
                        ("nMaxFile", wintypes.DWORD), ("lpstrFileTitle", wintypes.LPWSTR),
                        ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
                        ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD),
                        ("nFileOffset", wintypes.WORD), ("nFileExtension", wintypes.WORD),
                        ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
                        ("lpfnHook", vp), ("lpTemplateName", wintypes.LPCWSTR),
                        ("pvReserved", vp), ("dwReserved", wintypes.DWORD),
                        ("FlagsEx", wintypes.DWORD)]

        buf = ctypes.create_unicode_buffer(2048)
        filt = ctypes.create_unicode_buffer(
            "Chart (info.dat / *.sm)\0*.dat;*.sm\0Beat Saber (*.dat)\0*.dat\0"
            "StepMania (*.sm)\0*.sm\0All files\0*.*\0\0")
        ofn = OFN()
        ofn.lStructSize = ctypes.sizeof(OFN)
        ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
        ofn.nMaxFile = 2048
        ofn.lpstrFilter = ctypes.cast(filt, wintypes.LPCWSTR)
        ofn.lpstrTitle = "Pick a chart (Beat Saber info.dat/.dat or StepMania .sm)"
        ofn.lpstrInitialDir = initial or os.getcwd()
        ofn.Flags = 0x00081800          # EXPLORER | FILEMUSTEXIST | PATHMUSTEXIST
        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            p = buf.value
            return os.path.dirname(p) if p and os.path.isfile(p) else None
        return None
    except Exception:
        return None
COLW, ROWH, Y0 = 0.85, 0.85, 0.55       # column width, row height, bottom-row height
UPS = 20.0                              # units/sec the notes travel toward you
NOTE = 0.5                              # cube size (smaller vs the wider lanes)


def run(folder, want_diff, mode="bs"):
    if mode == "ddr":
        bpm, audio_p, title, diffs = load_song_sm(folder)
        diff_notes = [d["notes"] for d in diffs]
    else:
        bpm, audio_p, title, diffs = load_song(folder)
        diff_notes = [load_notes(d["path"], bpm) for d in diffs]
    if not diffs:
        raise SystemExit("no difficulties found")
    if not audio_p or not os.path.isfile(audio_p):
        raise SystemExit("no audio in the song folder")

    from ursina import (Ursina, Entity, camera, color, Text, Button, InputField, window,
                        scene, mouse, held_keys, application, Vec2, Vec3, destroy, Mesh)
    from ursina.shaders import lit_with_shadows_shader

    app = Ursina(vsync=True)
    wtitle = f"{APP_NAME}  -  {title}"
    window.title = wtitle
    window.color = color.hex("#12141a")
    window.borderless = False
    try:
        from panda3d.core import WindowProperties       # force the OS titlebar text
        _wp = WindowProperties(); _wp.setTitle(wtitle)
        app.win.requestProperties(_wp)
    except Exception:
        pass
    try:
        application.development_mode = False       # hide the entity/collider debug counters
    except Exception:
        pass
    for attr in ("fps_counter", "entity_counter", "collider_counter",
                 "exit_button", "cog_menu"):
        try:
            getattr(window, attr).enabled = False
        except Exception:
            pass

    ACC = color.hex("#4fd1ff"); ACC2 = color.hex("#8adcff"); GOLD = color.hex("#ffd24f")
    TXT = color.hex("#e6e8ee"); DIM = color.hex("#9aa0b0")
    C_RED = color.hex("#e02531"); C_BLUE = color.hex("#2166e6"); C_BOMB = color.hex("#2a2c36")
    E_RED = color.hex("#ff6b73"); E_BLUE = color.hex("#79abff")        # brighter edge tints
    PANEL = color.hex("#232734"); LINE = color.hex("#2c3040"); ARROWC = color.hex("#f2f6ff")

    # bold Beat-Saber arrow (triangle head + stem), pointing +y. A SEPARATE mesh per
    # note -- ursina reparents a Mesh to its entity, so one shared mesh only renders on
    # the last entity (that was the "no arrows" bug).
    def make_arrow():
        return Mesh(
            vertices=[(0, 0.55, 0), (-0.34, 0.06, 0), (0.34, 0.06, 0),
                      (-0.14, 0.06, 0), (0.14, 0.06, 0), (0.14, -0.5, 0), (-0.14, -0.5, 0)],
            triangles=[(0, 1, 2), (3, 4, 5), (3, 5, 6)], mode="triangle")

    # camera + lighting + atmosphere
    camera.fov = 70
    camera.position = Vec3(0, 1.62, -3.8)        # close first-person, like standing in-game
    camera.rotation = Vec3(6, 0, 0)
    from ursina import DirectionalLight, AmbientLight, PointLight
    sun = DirectionalLight(color=color.hex("#ffffff")); sun.look_at(Vec3(0.6, -1.0, 0.7))
    DirectionalLight(color=color.hex("#4a5a86")).look_at(Vec3(-0.7, -0.3, 0.8))   # cool fill
    AmbientLight(color=color.hex("#41495e"))
    PointLight(parent=camera, color=ACC, position=Vec3(0, 1.6, 1))
    scene.fog_color = color.hex("#12141a")
    scene.fog_density = (26, 74)             # linear fog: notes fade in with depth

    # ---- the floor/track: the 4 note lanes with glowing side RAILS beyond them (the
    # game environment edges) + speed-stripes flowing at you. (BS grid is 4 wide x 3.) --
    ROAD_LEN, ROADY = 70.0, Y0 - 0.32
    RAILX = 2 * COLW + 0.45
    Entity(model="plane", scale=(60, 1, ROAD_LEN * 2.4), position=(0, ROADY - 0.03, ROAD_LEN),
           color=color.hex("#0d0f15"))                          # wide dark ground
    Entity(model="plane", scale=(2 * RAILX, 1, ROAD_LEN), position=(0, ROADY, ROAD_LEN / 2),
           color=color.hex("#161a23"))                          # the track surface
    for sx in (-RAILX, RAILX):                                  # bright side rails
        Entity(model="cube", scale=(0.06, 0.05, ROAD_LEN),
               position=(sx, ROADY + 0.03, ROAD_LEN / 2), color=ACC)
    for i in range(1, 4):                                       # 3 subtle inner lane lines
        Entity(model="cube", scale=(0.02, 0.011, ROAD_LEN),
               position=((i - 2) * COLW, ROADY + 0.011, ROAD_LEN / 2), color=color.hex("#243a48"))
    stripes, SN, SGAP = [], 24, 3.2                            # cross-stripes flowing at you
    for i in range(SN):
        stripes.append(Entity(model="cube", scale=(2 * RAILX, 0.012, 0.09),
                              position=(0, ROADY + 0.014, i * SGAP), color=color.hex("#22364a")))

    # Playfield markers at the hit line (z=0). BS = one outer 4x3 frame; DDR = a dim
    # receptor arrow per lane (points the panel direction). Both are built and toggled by
    # mode, so the Load button can swap between a Beat Saber and a StepMania song in place.
    bs_frame = [Entity(model="wireframe_cube", color=color.hex("#2b4658"),
                       scale=(4 * COLW, 3 * ROWH, 0.02), position=(0, Y0 + ROWH, 0))]
    ddr_recep = []
    for _col in range(4):
        _rx = (_col - 1.5) * COLW
        _va = CUTV[PANEL_CUT[_col]]
        ddr_recep.append(Entity(model="wireframe_cube", color=color.hex("#25384a"),
                                scale=(COLW * 0.92, COLW * 0.92, 0.02), position=(_rx, Y0, 0)))
        _ra = Entity(model=make_arrow(), color=color.hex("#43596e"), double_sided=True,
                     scale=COLW * 0.82, position=(_rx, Y0, 0.015))
        _ra.rotation = Vec3(0, 0, math.degrees(math.atan2(-_va[0], _va[1])))
        ddr_recep.append(_ra)

    def set_playfield():
        for e in bs_frame:
            e.enabled = (mode == "bs")
        for e in ddr_recep:
            e.enabled = (mode == "ddr")
    set_playfield()

    # note pool: cube + arrow(child) + dot(child)
    pool = []
    for _ in range(200):
        cube = Entity(model="cube", color=C_BLUE, shader=lit_with_shadows_shader,
                      scale=NOTE, enabled=False)
        edge = Entity(parent=cube, model="wireframe_cube", color=E_BLUE, scale=1.03)
        arr = Entity(parent=cube, model=make_arrow(), color=ARROWC, double_sided=True,
                     scale=0.9, position=Vec3(0, 0, -0.56))
        dot = Entity(parent=cube, model="circle", color=ARROWC, scale=0.34,
                     position=Vec3(0, 0, -0.56), enabled=False)
        shadow = Entity(model="plane", color=color.hex("#05070c"), scale=NOTE * 1.05,
                        enabled=False)                 # cast on the road under the note
        shadow.alpha = 0.55
        hold = Entity(model="cube", color=C_BLUE, enabled=False)   # DDR freeze/roll body
        try:
            hold.alpha = 0.45
        except Exception:
            pass
        pool.append([cube, edge, arr, dot, shadow, hold])

    # ---- audio (Panda3D -> gives play-rate/speed) + smooth interpolated clock ----
    from panda3d.core import Filename

    def _load_music(path):
        # Panda3D wants its own Unix-style Filename; a raw absolute Windows path
        # (e.g. from the file picker) fails to open. fromOsSpecific converts it.
        s = app.loader.loadMusic(Filename.fromOsSpecific(os.path.abspath(path)))
        s.setLoop(False)
        return s

    snd = _load_music(audio_p)
    dur = snd.length()
    SPEEDS = [0.25, 0.5, 0.75, 1.0]
    au = {"playing": False, "base": 0.0, "rate": 1.0, "si": 3, "est": 0.0, "lastpf": None}

    def tick():
        # advance a smooth monotonic clock and gently pull it toward the true audio
        # time -> no jumps even at low play-rate (snd.getTime() updates coarsely).
        if not au["playing"]:
            return
        if snd.status() != snd.PLAYING:            # reached the end
            au["playing"] = False
            au["base"] = min(au["est"], dur)
            return
        pf = perf_counter()
        if au["lastpf"] is None:
            au["lastpf"] = pf
        dt = pf - au["lastpf"]; au["lastpf"] = pf
        au["est"] = min(au["est"] + dt * au["rate"], dur)
        # run the clock purely on rate*dt (perfectly smooth at any speed). Only resync on
        # a BIG gap (a seek) -- no per-frame pull toward getTime, whose audio-buffer lag
        # was yanking the notes back and making slow-mo look jumpy.
        if abs(snd.getTime() - au["est"]) > 0.4:
            au["est"] = min(max(snd.getTime(), 0.0), dur)

    def anow():
        return au["est"] if au["playing"] else au["base"]

    def _start(t):
        t = max(0.0, min(t, dur - 0.05))
        snd.setPlayRate(au["rate"])
        snd.setTime(t)
        snd.play()
        au["playing"] = True; au["est"] = t; au["lastpf"] = None

    def toggle():
        if au["playing"]:
            au["base"] = anow(); snd.stop(); au["playing"] = False
        else:
            if au["base"] >= dur - 0.05:
                au["base"] = 0.0
            _start(au["base"])

    def seek(t):
        t = max(0.0, min(t, dur - 0.05))
        au["base"] = t; au["est"] = t
        if au["playing"]:
            _start(t)

    # ---- UI ----  (ui space: y in [-.5,.5], x in [-aspect/2, aspect/2]) ----
    LOOK = {"v": 2.2}
    LX = -0.86                                       # left text anchor
    RX = 0.86                                        # right text anchor

    def panel(cx, cy, w, h):
        p = Entity(parent=camera.ui, model="quad", color=PANEL,
                   position=(cx, cy, 0.05), scale=(w, h))
        try:
            p.alpha = 0.7
        except Exception:
            pass
        return p

    # left cluster: title / difficulty / stats (labels + values in aligned columns)
    panel(-0.645, 0.345, 0.47, 0.30)
    st_title = Text(title, parent=camera.ui, position=Vec2(LX, 0.472),
                    origin=(-0.5, 0.5), scale=0.95, color=TXT)
    st_diff = Text("", parent=camera.ui, position=Vec2(LX, 0.427),
                   origin=(-0.5, 0.5), scale=0.78, color=ACC)
    STY, VX = 0.375, LX + 0.18
    Text(f"notes\n{'jumps' if mode == 'ddr' else 'doubles'}\navg NPS\npeak NPS\nnow NPS",
         parent=camera.ui, position=Vec2(LX, STY), origin=(-0.5, 0.5), scale=0.72,
         color=DIM, line_height=1.5)
    st_vals = Text("", parent=camera.ui, position=Vec2(VX, STY),
                   origin=(-0.5, 0.5), scale=0.72, color=TXT, line_height=1.5)

    # right cluster: timestamp / beat / status (panel mirrors the left for symmetry)
    panel(0.645, 0.345, 0.47, 0.30)
    st_time = Text("0:00.000", parent=camera.ui, position=Vec2(RX, 0.47),
                   origin=(0.5, 0.5), scale=1.3, color=TXT)
    st_sub = Text("", parent=camera.ui, position=Vec2(RX, 0.408),
                  origin=(0.5, 0.5), scale=0.7, color=DIM)
    st_status = Text("", parent=camera.ui, position=Vec2(RX, 0.34),
                     origin=(0.5, 0.5), scale=0.72, color=DIM, line_height=1.5)

    st_paused = Text("PAUSED", parent=camera.ui, position=Vec2(0, 0.17),
                     origin=(0, 0), scale=1.4, color=GOLD)

    # timeline: a dense filled song WAVEFORM (SubTap-style) with muted density behind it
    BARW, BARY, BARH = 1.5, -0.32, 0.09
    bar = Entity(parent=camera.ui, model="quad", color=color.hex("#0e1117"), collider="box",
                 scale=(BARW, BARH), position=(0, BARY, 0))
    heat_cells = []

    def build_heat():
        for e in heat_cells:
            destroy(e)
        heat_cells.clear()
        ts = cur["st"]["times"]
        if not ts or not dur:
            return
        bins, bottom = 200, BARY - BARH / 2
        arr = [0] * bins
        for x in ts:
            arr[min(bins - 1, int(x / dur * bins))] += 1
        mx = max(arr) or 1
        for i, val in enumerate(arr):
            if not val:
                continue
            h = 0.008 + (val / mx) * (BARH - 0.01)
            x = -BARW / 2 + (i + 0.5) / bins * BARW
            heat_cells.append(Entity(parent=camera.ui, model="quad", color=color.hex("#171f29"),
                                     scale=(BARW / bins + 0.001, h),
                                     position=(x, bottom + h / 2, -0.02)))    # muted, behind

    # dense filled waveform (many edge-to-edge bars -> reads as a solid envelope)
    WBINS = 420
    W_DIM, W_HOT = color.hex("#3a6b82"), color.hex("#6fc2e8")   # SubTap teal -> bright when played
    wav = compute_waveform(audio_p, WBINS)
    wav_bars = []
    bw = BARW / WBINS * 1.08
    for i, v in enumerate(wav):
        h = max(0.004, v * (BARH - 0.006))
        wav_bars.append(Entity(parent=camera.ui, model="quad", color=W_DIM, scale=(bw, h),
                               position=(-BARW / 2 + (i + 0.5) / WBINS * BARW, BARY, -0.035)))
    wav_state = {"pi": 0}
    phead = Entity(parent=camera.ui, model="quad", color=color.hex("#eaf6ff"),
                   scale=(0.006, BARH + 0.04), position=(-BARW / 2, BARY, -0.1))
    Text("0:00", parent=camera.ui, position=Vec2(-BARW / 2, BARY - BARH / 2 - 0.016),
         origin=(-0.5, 0.5), scale=0.58, color=DIM)
    st_durlabel = Text(f"{int(dur // 60)}:{int(dur % 60):02d}", parent=camera.ui,
                       position=Vec2(BARW / 2, BARY - BARH / 2 - 0.016), origin=(0.5, 0.5),
                       scale=0.58, color=DIM)

    cur = {"di": len(diffs) - 1, "notes": [], "times": [], "st": None, "vals4": ""}
    if want_diff:
        for i, d in enumerate(diffs):
            if d["name"].lower() == want_diff.lower():
                cur["di"] = i

    def set_diff(i):
        cur["di"] = i % len(diffs)
        cur["notes"] = diff_notes[cur["di"]]
        cur["times"] = [n[0] for n in cur["notes"]]
        cur["st"] = (stats_ddr if mode == "ddr" else stats)(cur["notes"], dur)
        d = diffs[cur["di"]]; s = cur["st"]
        if mode == "ddr":
            st_diff.text = f"{d['name']}   meter {d['njs']:g}   BPM {bpm:.1f}"
            cur["vals4"] = f"{s['total']}\n{s['doubles']}\n{s['avg']:.2f}\n{s['peak']}"
        else:
            st_diff.text = f"{d['name']}   NJS {d['njs']:g}   BPM {bpm:.1f}"
            cur["vals4"] = (f"{s['total']}  (R {s['reds']}/B {s['blues']})\n"
                            f"{s['doubles']}  ({100 * s['doubles'] / max(1, s['total']):.1f}%)\n"
                            f"{s['avg']:.2f}\n{s['peak']}")
        st_vals.text = cur["vals4"] + "\n-"
        build_heat()
        save_config({"last_song": os.path.abspath(folder), "last_diff": d["name"], "mode": mode})

    def refresh_status():
        st_status.text = f"speed  {au['rate']:g}x\nlook  {LOOK['v']:.2f}s"

    def set_rate(delta):
        au["si"] = max(0, min(len(SPEEDS) - 1, au["si"] + delta))
        au["rate"] = SPEEDS[au["si"]]
        snd.setPlayRate(au["rate"])
        if au["playing"]:
            au["lastpf"] = None                    # avoid a dt spike at the rate change
        refresh_status()

    def setlook(delta):
        LOOK["v"] = max(0.75, min(5.0, LOOK["v"] + delta))
        refresh_status()

    def load_song_into(newfolder, want_diff=None):
        # swap the whole song IN-PLACE (no restart): audio, notes, waveform, title.
        # Parse EVERYTHING into locals first and only COMMIT on success -- a malformed
        # folder from the Load picker then just no-ops instead of killing the session.
        nonlocal folder, bpm, audio_p, title, diffs, diff_notes, snd, dur, mode
        newmode = detect_mode(newfolder) or mode
        try:
            if newmode == "ddr":
                nbpm, naudio, ntitle, ndiffs = load_song_sm(newfolder)
                ndiff_notes = [d["notes"] for d in ndiffs]
            else:
                nbpm, naudio, ntitle, ndiffs = load_song(newfolder)
                ndiff_notes = [load_notes(d["path"], nbpm) for d in ndiffs]
            if not ndiffs or not naudio or not os.path.isfile(naudio):
                raise ValueError("no charts or audio in folder")
            newsnd = _load_music(naudio)
        except Exception as e:
            print(f"[load skipped] {newfolder}: {e}")
            return
        try:
            if snd is not None:
                snd.stop()
        except Exception:
            pass
        mode = newmode
        folder, bpm, audio_p, title, diffs, diff_notes = \
            newfolder, nbpm, naudio, ntitle, ndiffs, ndiff_notes
        snd = newsnd
        dur = snd.length()
        set_playfield()
        wt = f"{APP_NAME}  -  {title}"
        window.title = wt
        try:
            from panda3d.core import WindowProperties
            _wp = WindowProperties(); _wp.setTitle(wt); app.win.requestProperties(_wp)
        except Exception:
            pass
        st_title.text = title
        st_durlabel.text = f"{int(dur // 60)}:{int(dur % 60):02d}"
        env = compute_waveform(audio_p, WBINS)             # reuse the wav bars, re-shape them
        for i in range(WBINS):
            wav_bars[i].scale_y = max(0.004, env[i] * (BARH - 0.006))
            wav_bars[i].color = W_DIM
        wav_state["pi"] = 0
        au.update(playing=False, base=0.0, est=0.0, lastpf=None)
        nowstate["v"] = -1; pstate["v"] = None
        di = len(diffs) - 1
        if want_diff:                                      # keep the same difficulty if it exists
            for j, d in enumerate(diffs):
                if d["name"].lower() == want_diff.lower():
                    di = j
        set_diff(di)

    def do_load():
        newf = pick_folder(os.path.dirname(os.path.abspath(folder)))
        if detect_mode(newf):
            load_song_into(newf, diffs[cur["di"]]["name"])

    # ---- feedback capture: screenshot + comment box, saved for mapper review ----
    modal = {"on": False, "shot": "", "t": 0.0, "seq": 0}
    modal_els = []
    WRAP_COLS = 80                       # chars/line -> symmetric L/R margin in the box

    def _wrap_modal():
        # Ursina's TextField is a code editor: it never auto-wraps, and its cursor tracks
        # ONE logical line -- so a long comment marches the cursor off the box. Fix: each
        # frame insert REAL newlines into the field text (wrap every paragraph, rejoin)
        # and park the cursor at the end. Idempotent -- a wrapped line re-wraps to itself
        # (drop_whitespace=False keeps typed spaces intact) -- so it only changes state
        # when you actually type, and the cursor stays inside the box because it's now on
        # a real short line.
        if not modal["on"]:
            return
        try:
            tf = m_field.text_field
            out = []
            for para in tf.text.split("\n"):
                out.extend(textwrap.wrap(para, WRAP_COLS, drop_whitespace=False,
                                         break_long_words=True, break_on_hyphens=False) or [""])
            wrapped = "\n".join(out)
            if wrapped != tf.text:
                tf.text = wrapped
                tf.cursor.y = len(out) - 1
                tf.cursor.x = len(out[-1])
                tf.render()
        except Exception:
            pass

    def show_modal(v):
        modal["on"] = v
        for e in modal_els:
            e.enabled = v
        m_field.active = v

    def capture():
        if au["playing"]:
            toggle()                              # freeze on this moment
        t = anow()
        d = diffs[cur["di"]]; s = cur["st"]
        localn = (bisect.bisect_right(s["times"], t + 0.5) - bisect.bisect_left(s["times"], t - 0.5))
        modal["seq"] += 1
        safe = "".join(c for c in title if c.isalnum() or c in " -_").strip().replace(" ", "_")
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        shot = os.path.join(FEEDBACK_DIR,
                            f"{safe}_{d['name']}_{int(t * 1000):08d}ms_{modal['seq']:03d}.png")
        try:
            from panda3d.core import Filename
            app.win.saveScreenshot(Filename.fromOsSpecific(shot))
        except Exception:
            shot = "(screenshot failed)"
        modal["shot"] = shot; modal["t"] = t
        m_info.text = (f"{title}    |    {d['name']}    |    {int(t // 60)}:{t % 60:06.3f}"
                       f"    beat {t * bpm / 60:.1f}    |    {localn} nps here")
        m_field.text = ""
        show_modal(True)

    def do_save():
        t = modal["t"]; d = diffs[cur["di"]]; s = cur["st"]
        localn = (bisect.bisect_right(s["times"], t + 0.5) - bisect.bisect_left(s["times"], t - 0.5))
        comment = (m_field.text or "").strip()
        rec = {"when": datetime.now().isoformat(timespec="seconds"), "song": title,
               "difficulty": d["name"], "njs": d["njs"], "bpm": round(bpm, 2),
               "time": f"{int(t // 60)}:{t % 60:06.3f}", "time_s": round(t, 3),
               "beat": round(t * bpm / 60, 2), "now_nps": localn,
               "avg_nps": round(s["avg"], 2), "peak_nps": s["peak"],
               "comment": comment, "screenshot": os.path.basename(modal["shot"])}
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        with open(os.path.join(FEEDBACK_DIR, "feedback.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        with open(os.path.join(FEEDBACK_DIR, "feedback.md"), "a", encoding="utf-8") as f:
            f.write(f"## {rec['song']} - {rec['difficulty']} @ {rec['time']} (beat {rec['beat']})\n"
                    f"- {rec['when']}  .  {rec['now_nps']} nps here (avg {rec['avg_nps']}, "
                    f"peak {rec['peak_nps']})\n- shot: {rec['screenshot']}\n\n"
                    f"{comment or '(no comment)'}\n\n---\n\n")
        show_modal(False)

    m_dim = Entity(parent=camera.ui, model="quad", color=color.hex("#0a0d12"),
                   scale=(4, 2.2), position=Vec3(0, 0, -0.5), collider="box", enabled=False)
    m_dim.alpha = 0.82
    m_pan = Entity(parent=camera.ui, model="quad", color=color.hex("#232734"),
                   scale=(0.94, 0.46), position=Vec3(0, 0.02, -0.55), enabled=False)
    m_lbl = Text("Feedback -- comment on this moment:", parent=camera.ui,
                 origin=(-0.5, 0.5), scale=0.9, color=TXT, enabled=False)
    m_lbl.position = Vec2(-0.44, 0.21); m_lbl.z = -0.6
    m_info = Text("", parent=camera.ui, origin=(-0.5, 0.5), scale=0.72, color=ACC, enabled=False)
    m_info.position = Vec2(-0.44, 0.16); m_info.z = -0.6
    m_field = InputField(parent=camera.ui, default_value="", max_lines=6, character_limit=500,
                         scale=(0.86, 0.17), enabled=False)
    m_field.position = Vec2(0, 0.0); m_field.z = -0.6
    try:
        m_field.text_field.scale *= 0.5                     # smaller text
    except Exception:
        pass
    # NOTE: Ursina's TextField.render() rewrites text_entity.text (unwrapped) on every
    # keystroke, so .wordwrap never sticks. We re-wrap the DISPLAY each frame in _update
    # (see _wrap_modal) instead -- the only thing that reliably keeps text in the box.
    m_save = Button(text="Save", parent=camera.ui, color=color.hex("#2b6e8c"),
                    highlight_color=ACC, scale=(0.16, 0.06), position=Vec3(0.15, -0.16, -0.6),
                    on_click=do_save, enabled=False)
    m_cancel = Button(text="Cancel", parent=camera.ui, color=PANEL, highlight_color=ACC,
                      scale=(0.16, 0.06), position=Vec3(-0.15, -0.16, -0.6),
                      on_click=lambda: show_modal(False), enabled=False)
    modal_els += [m_dim, m_pan, m_lbl, m_info, m_field, m_save, m_cancel]
    show_modal(False)

    # toolbar buttons -- evenly spaced + centered, consistent text size
    bdefs = [("Load", do_load), ("Restart", lambda: seek(0)), ("Play", toggle),
             ("Diff -", lambda: set_diff(cur["di"] - 1)), ("Diff +", lambda: set_diff(cur["di"] + 1)),
             ("Speed -", lambda: set_rate(-1)), ("Speed +", lambda: set_rate(1)),
             ("Look -", lambda: setlook(-0.25)), ("Look +", lambda: setlook(0.25)),
             ("Note", capture)]
    BW, BG, BTNY = 0.10, 0.013, -0.435
    x0 = -(len(bdefs) * BW + (len(bdefs) - 1) * BG) / 2 + BW / 2
    btns = {}
    play_label = None
    for i, (lab, fn) in enumerate(bdefs):
        px = x0 + i * (BW + BG)
        b = Button(parent=camera.ui, color=PANEL, highlight_color=ACC,
                   scale=(BW, 0.05), position=(px, BTNY), on_click=fn)   # no built-in text
        lt = Text(lab, parent=camera.ui, position=Vec2(px, BTNY), origin=(0, 0),
                  scale=0.55, color=TXT)                # own label -> consistent size for all
        lt.z = -0.2
        btns[lab] = b
        if lab == "Play":
            play_label = lt
    try:
        btns["Note"].color = color.hex("#2b6e8c"); btns["Note"].highlight_color = ACC2
    except Exception:
        pass
    refresh_status()
    Text(f"{APP_NAME} v{VERSION}   |   built {BUILD}   |   {COPYRIGHT}", parent=camera.ui,
         position=Vec2(-0.88, -0.487), origin=(-0.5, 0), scale=0.5, color=color.hex("#6a6f80"))

    drag = {"on": False}
    pstate = {"v": None}                       # last paused-state, to update Play/Pause label
    nowstate = {"v": -1}                        # last now-NPS, to avoid rebuilding vals each frame

    def bar_seek():
        frac = (mouse.x + BARW / 2) / BARW
        seek(max(0.0, min(1.0, frac)) * dur)

    # ---- per-frame update + input, hosted on an entity so Ursina calls them ----
    host = Entity()

    def _update():
        tick()
        _wrap_modal()
        t = anow()
        notes, times = cur["notes"], cur["times"]
        lo = bisect.bisect_left(times, t - 0.12)
        hi = bisect.bisect_right(times, t + LOOK["v"])
        k = 0
        for idx in range(lo, hi):
            if k >= len(pool):
                break
            n = notes[idx]; z = (n[0] - t) * UPS
            cube, edge, arr, dot, shadow, hold = pool[k]; k += 1
            nx = (n[1] - 1.5) * COLW
            cube.enabled = True
            cube.position = (nx, Y0 + n[2] * ROWH, z)
            shadow.enabled = True
            shadow.position = (nx, ROADY + 0.014, z)   # directly below, on the road
            if mode == "ddr":
                hold.enabled = False
                if n[3] == DDR_MINE:
                    cube.color = C_BOMB; edge.enabled = False
                    arr.enabled = False; dot.enabled = True; dot.color = E_RED
                else:
                    qc = color.hex(n[5])                 # rhythm-subdivision colour
                    cube.color = qc
                    edge.enabled = True; edge.color = color.hex("#f2f6ff")
                    v = CUTV.get(n[4])
                    arr.enabled = True; dot.enabled = False; arr.color = ARROWC
                    arr.rotation = Vec3(0, 0, math.degrees(math.atan2(-v[0], v[1])))
                    if n[6] is not None:                 # freeze/roll body: head -> tail
                        tz = (n[6] - t) * UPS
                        length = max(0.05, tz - z)
                        hold.enabled = True; hold.color = qc
                        hold.scale = (NOTE * 0.42, NOTE * 0.42, length)
                        hold.position = (nx, Y0 + n[2] * ROWH, z + length / 2)
            else:
                hold.enabled = False
                typ = n[3]
                if typ == BOMB:
                    cube.color = C_BOMB; edge.enabled = False
                    arr.enabled = False; dot.enabled = False
                else:
                    cube.color = C_RED if typ == RED else C_BLUE
                    edge.enabled = True; edge.color = E_RED if typ == RED else E_BLUE
                    v = CUTV.get(n[4])
                    if v:
                        arr.enabled = True; dot.enabled = False; arr.color = ARROWC
                        arr.rotation = Vec3(0, 0, math.degrees(math.atan2(-v[0], v[1])))
                    else:
                        arr.enabled = False; dot.enabled = True; dot.color = ARROWC
        for j in range(k, len(pool)):
            if pool[j][0].enabled:
                pool[j][0].enabled = False
                pool[j][4].enabled = False              # its road shadow too
                pool[j][5].enabled = False              # its hold body too

        span = SN * SGAP                          # scroll the road speed-stripes at you
        for i, s in enumerate(stripes):
            s.z = (i * SGAP - t * UPS) % span

        if drag["on"] and held_keys["left mouse"]:
            bar_seek()
        # HUD -- only the live values update per frame (static stats set in set_diff)
        s = cur["st"]
        localn = (bisect.bisect_right(s["times"], t + 0.5) - bisect.bisect_left(s["times"], t - 0.5))
        if localn != nowstate["v"]:
            nowstate["v"] = localn
            st_vals.text = cur["vals4"] + f"\n{localn}"
        mm, ss = int(t // 60), t % 60
        st_time.text = f"{mm}:{ss:06.3f}"
        st_sub.text = f"/ {int(dur // 60)}:{int(dur % 60):02d}    beat {t * bpm / 60:.1f}"
        frac = (t / dur) if dur else 0
        phead.x = -BARW / 2 + frac * BARW
        pi = int(frac * WBINS)                     # waveform: brighten the played portion
        op = wav_state["pi"]
        if pi > op:
            for j in range(max(0, op), min(WBINS, pi)):
                wav_bars[j].color = W_HOT
        elif pi < op:
            for j in range(max(0, pi), min(WBINS, op)):
                wav_bars[j].color = W_DIM
        wav_state["pi"] = pi
        paused = not au["playing"]
        st_paused.enabled = paused
        if pstate["v"] != paused:
            pstate["v"] = paused
            play_label.text = "Play" if paused else "Pause"

    def _input(key):
        if modal["on"]:
            return                              # modal open: type the comment, use its buttons
        if key == "left mouse down":
            if mouse.hovered_entity == bar:
                drag["on"] = True; bar_seek()
            elif mouse.hovered_entity is None:
                toggle()                        # click empty 3D space -> play/pause
        elif key == "left mouse up":
            drag["on"] = False
        elif key == "scroll up":
            seek(anow() + 1)
        elif key == "scroll down":
            seek(anow() - 1)

    host.update = _update
    host.input = _input
    set_diff(cur["di"])
    app.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=None)
    ap.add_argument("difficulty", nargs="?", default=None)
    args = ap.parse_args()
    cfg = load_config()
    folder = args.folder or cfg.get("last_song")          # auto-load the last song
    diff = args.difficulty or cfg.get("last_diff")        # remember the last difficulty
    if not detect_mode(folder):
        folder = pick_folder(os.path.dirname(folder) if folder else None)
    mode = detect_mode(folder)
    if not mode:
        print("No song selected (need a Beat Saber info.dat or a StepMania .sm).")
        return 1
    run(folder, diff, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
