r"""Generator version numbers -- tracked INDEPENDENTLY per game (they evolve separately).

Format: YYYY.M.D.build   (the last number bumps for another build the SAME day).
Bump the relevant one whenever that game's generation logic changes; the version is
written into every song folder's _generation.txt and embedded in the map itself.

DON'T bump these by hand -- run  `python release.py bs "what changed"`  which bumps the
build number, regenerates every pack (so the version is embedded everywhere), and
commits + tags, so a version always maps to exactly one committed generator state you
can roll back to. `bump()` below is the single place the number changes.
"""
from __future__ import annotations

import datetime
import os

BS_GEN_VERSION = "2026.9.15.0"    # Beat Saber -- no same-cut-twice (incl. horizontals + across stitch/tile SEAMS)
ITG_GEN_VERSION = "2026.9.10.0"   # ITG / StepMania -- regular top-two density ramp softened (Easy 2.0->2.2, Hard 4.5->4.0, Challenge 6.5->5.0)


def build_timestamp() -> str:
    """The moment this build was generated (written into every report). A whole release
    shares ONE timestamp when the orchestrator sets NOTESIGHT_BUILD_TS; else it's now()."""
    return os.environ.get("NOTESIGHT_BUILD_TS") \
        or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def next_version(current: str, today: datetime.date | None = None) -> str:
    """The next version after `current`: new calendar day -> YYYY.M.D.0, same day -> bump
    the build number. Keeps a version monotonic and self-dating."""
    today = today or datetime.date.today()
    stamp = f"{today.year}.{today.month}.{today.day}"
    if current.startswith(stamp + "."):
        build = int(current.rsplit(".", 1)[1]) + 1
        return f"{stamp}.{build}"
    return f"{stamp}.0"
