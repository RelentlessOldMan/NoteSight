r"""pack_manifest.py -- a sortable HTML song list for a built pack.

Scans a pack folder (StepMania `.sm` OR Beat Saber `.dat`) and writes a single,
self-contained HTML page: one row per (song x difficulty) with the density
metrics, and click-to-sort column headers. Drop the file in the pack's `.zip`
and/or serve it on the web.

    python pack_manifest.py "<pack_dir>" [--title "Pack Name"] [-o out.html]

Columns:
    #             generation order (sort by this to restore the original order)
    Song, BPM, Category
    avg NPS       notes / active span (density while playing)
    peak NPS      densest single second (max notes in any 1s window)
    avg notes/min notes / full-song length (throughput, incl. quiet intro/outro)
    peak notes/min densest single minute (max notes in any 60s window)

"notes" = StepMania steps (a jump counts once) / Beat Saber blocks (red+blue).
"""
from __future__ import annotations

import glob
import html
import json
import os
import sys

# eval/smparse is the repo's .sm reader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))
from smparse import parse_sm  # noqa: E402

try:
    import soundfile as sf
except Exception:  # soundfile is a core dep, but degrade gracefully
    sf = None

AUDIO_EXTS = (".ogg", ".egg", ".mp3", ".wav", ".flac")
# Difficulty sort rank so tiers list easiest->hardest regardless of file order.
RANK = {"Beginner": 0, "Easy": 1, "Basic": 1, "Normal": 2, "Medium": 3,
        "Hard": 4, "Expert": 5, "Challenge": 6, "ExpertPlus": 7, "Edit": 8}


# --------------------------------------------------------------------------- metrics
def _peak(times: list[float], window: float) -> int:
    """Max number of notes falling in any `window`-second span (anchored at a note)."""
    best, j, n = 0, 0, len(times)
    for i in range(n):
        if j < i:
            j = i
        while j < n and times[j] < times[i] + window:
            j += 1
        if j - i > best:
            best = j - i
    return best


def _metrics(times: list[float], song_dur: float | None) -> dict:
    times = sorted(times)
    n = len(times)
    span = times[-1] - times[0] if n >= 2 else 0.0
    avg_nps = n / span if span > 0 else 0.0
    dur_min = (song_dur / 60.0) if song_dur and song_dur > 0 else (span / 60.0 if span > 0 else 0.0)
    avg_npm = n / dur_min if dur_min > 0 else 0.0
    return {
        "notes": n,
        "avg_nps": round(avg_nps, 2),
        "peak_nps": _peak(times, 1.0),
        "avg_npm": round(avg_npm, 1),
        "peak_npm": _peak(times, 60.0),
    }


def _duration(folder: str, hint: str = "") -> float | None:
    cands = []
    if hint:
        cands.append(os.path.join(folder, hint))
    for ext in AUDIO_EXTS:
        cands += glob.glob(os.path.join(folder, "*" + ext))
    for c in cands:
        if sf and os.path.isfile(c):
            try:
                info = sf.info(c)
                return info.frames / info.samplerate
            except Exception:
                continue
    return None


# --------------------------------------------------------------------------- parsers
def _rows_stepmania(song_dir: str) -> list[dict]:
    sm = (glob.glob(os.path.join(song_dir, "*.sm")) + glob.glob(os.path.join(song_dir, "*.ssc"))) or None
    if not sm:
        return []
    sim = parse_sm(sm[0])
    title = sim.title or os.path.basename(song_dir)
    bpm = sim.bpms[0][1] if sim.bpms else 0.0
    dur = _duration(song_dir, sim.music)
    out = []
    for c in sim.charts:
        if c.stepstype != "dance-single" or len(c.times) < 2:
            continue
        m = _metrics(c.times, dur)
        out.append({"song": title, "bpm": round(bpm, 1), "cat": c.difficulty, **m})
    out.sort(key=lambda r: RANK.get(r["cat"], 99))
    return out


def _rows_beatsaber(song_dir: str) -> list[dict]:
    info_p = next((os.path.join(song_dir, n) for n in ("info.dat", "Info.dat")
                   if os.path.isfile(os.path.join(song_dir, n))), None)
    if not info_p:
        return []
    info = json.load(open(info_p, encoding="utf-8"))
    title = info.get("_songName") or os.path.basename(song_dir)
    bpm = float(info.get("_beatsPerMinute", 0) or 0)
    dur = _duration(song_dir, info.get("_songFilename", ""))
    out = []
    for bset in info.get("_difficultyBeatmapSets", []):
        for bm in bset.get("_difficultyBeatmaps", []):
            dat_p = os.path.join(song_dir, bm.get("_beatmapFilename", ""))
            if not os.path.isfile(dat_p):
                continue
            try:
                dat = json.load(open(dat_p, encoding="utf-8"))
            except Exception:
                continue
            spb = 60.0 / bpm if bpm > 0 else 0.0
            times = [n["_time"] * spb for n in dat.get("_notes", []) if n.get("_type") in (0, 1)]
            if len(times) < 2:
                continue
            m = _metrics(times, dur)
            out.append({"song": title, "bpm": round(bpm, 1),
                        "cat": bm.get("_difficulty", "?"), **m})
    out.sort(key=lambda r: RANK.get(r["cat"], 99))
    return out


def scan_pack(pack_dir: str) -> tuple[list[dict], str]:
    """Return (rows, format) for every song folder under pack_dir. Each row already
    carries song/bpm/cat + metrics; we add the # after global ordering."""
    subdirs = sorted(d for d in glob.glob(os.path.join(pack_dir, "*"))
                     if os.path.isdir(d))
    rows, fmt = [], ""
    for d in subdirs:
        if glob.glob(os.path.join(d, "*.sm")) or glob.glob(os.path.join(d, "*.ssc")):
            fmt = fmt or "StepMania"
            rows += _rows_stepmania(d)
        elif os.path.isfile(os.path.join(d, "info.dat")) or os.path.isfile(os.path.join(d, "Info.dat")):
            fmt = fmt or "Beat Saber"
            rows += _rows_beatsaber(d)
    for i, r in enumerate(rows, 1):
        r["idx"] = i
    return rows, fmt or "pack"


# --------------------------------------------------------------------------- html
COLS = [
    ("idx", "#", "num"),
    ("song", "Song", "txt"),
    ("bpm", "BPM", "num"),
    ("cat", "Category", "txt"),
    ("avg_nps", "avg NPS", "num"),
    ("peak_nps", "peak NPS", "num"),
    ("avg_npm", "avg notes/min", "num"),
    ("peak_npm", "peak notes/min", "num"),
]


def render_html(title: str, fmt: str, rows: list[dict]) -> str:
    n_songs = len({r["song"] for r in rows})
    head = "".join(
        f'<th data-k="{k}" data-t="{t}">{html.escape(lbl)}<span class="ar"></span></th>'
        for k, lbl, t in COLS)
    body = []
    for r in rows:
        tds = "".join(f'<td class="{t}">{html.escape(str(r[k]))}</td>' for k, _, t in COLS)
        body.append(f"<tr>{tds}</tr>")
    body = "\n".join(body)
    return _TEMPLATE.format(
        title=html.escape(title), fmt=html.escape(fmt),
        n_songs=n_songs, n_rows=len(rows), head=head, body=body)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — song list</title>
<style>
  :root {{ --bg:#0a0d12; --panel:#121822; --line:#1e2836; --fg:#dfe8f0;
           --dim:#7f8ea3; --acc:#39d6d6; --acc2:#2b93c0; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
          font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:20px 24px 10px; }}
  h1 {{ margin:0; font-size:20px; letter-spacing:.3px; }}
  h1 .acc {{ color:var(--acc); }}
  .sub {{ color:var(--dim); font-size:13px; margin-top:3px; }}
  .wrap {{ padding:8px 16px 40px; }}
  table {{ border-collapse:collapse; width:100%; max-width:1000px; margin:0 auto;
           background:var(--panel); border:1px solid var(--line); border-radius:10px;
           overflow:hidden; }}
  th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid var(--line); }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  thead th {{ position:sticky; top:0; background:#0f1620; color:var(--acc);
              cursor:pointer; user-select:none; white-space:nowrap; font-weight:600; }}
  thead th:hover {{ color:#fff; }}
  thead th .ar {{ font-size:10px; color:var(--acc2); margin-left:5px; }}
  tbody tr:hover {{ background:#0f1620; }}
  tbody tr:nth-child(even) {{ background:rgba(255,255,255,.015); }}
  .legend {{ max-width:1000px; margin:14px auto 0; color:var(--dim); font-size:12px; }}
  .legend b {{ color:var(--fg); font-weight:600; }}
  code {{ color:var(--acc); }}
</style></head>
<body>
<header>
  <h1><span class="acc">{title}</span> — song list</h1>
  <div class="sub">{fmt} pack · {n_songs} songs · {n_rows} charts · click a header to sort ↑/↓</div>
</header>
<div class="wrap">
  <table id="t"><thead><tr>{head}</tr></thead>
  <tbody>
{body}
  </tbody></table>
  <div class="legend">
    <b>avg NPS</b> = notes ÷ active span · <b>peak NPS</b> = densest 1&nbsp;second ·
    <b>avg notes/min</b> = notes ÷ full song · <b>peak notes/min</b> = densest 60&nbsp;seconds.
    "notes" = StepMania steps (a jump = 1) / Beat Saber blocks. Sort by <code>#</code> to restore
    the generated order.
  </div>
</div>
<script>
(function () {{
  const tb = document.querySelector("#t tbody");
  const ths = [...document.querySelectorAll("#t thead th")];
  let sortK = null, dir = 1;
  function val(tr, i, t) {{
    const s = tr.children[i].textContent.trim();
    return t === "num" ? parseFloat(s.replace(/[^0-9.\-]/g, "")) || 0 : s.toLowerCase();
  }}
  ths.forEach((th, i) => th.addEventListener("click", () => {{
    const t = th.dataset.t;
    if (sortK === i) dir = -dir; else {{ sortK = i; dir = 1; }}
    ths.forEach(h => h.querySelector(".ar").textContent = "");
    th.querySelector(".ar").textContent = dir > 0 ? "▲" : "▼";
    const rows = [...tb.rows];
    rows.sort((a, b) => {{
      const x = val(a, i, t), y = val(b, i, t);
      return (x < y ? -1 : x > y ? 1 : 0) * dir;
    }});
    rows.forEach(r => tb.appendChild(r));
  }}));
}})();
</script>
</body></html>
"""


def write_manifest(pack_dir: str, out_path: str = "", title: str = "") -> str:
    rows, fmt = scan_pack(pack_dir)
    if not rows:
        raise SystemExit(f"no songs found in {pack_dir}")
    title = title or os.path.basename(os.path.normpath(pack_dir))
    if not out_path:
        out_path = os.path.join(pack_dir, "songs.html")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_html(title, fmt, rows))
    return out_path


def main(argv) -> int:
    argv = list(argv)
    title = out = ""
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--title":
            i += 1; title = argv[i]
        elif a in ("-o", "--out"):
            i += 1; out = argv[i]
        else:
            rest.append(a)
        i += 1
    if not rest:
        print(__doc__)
        return 2
    p = write_manifest(rest[0], out, title)
    rows, _ = scan_pack(rest[0])
    print(f"wrote {p}  ({len(rows)} charts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
