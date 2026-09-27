"""Draw a stars.solve result on the sequence's own frame (star_calibration.overlay).

Solved: matched tracks in thick green with the fitted stars in magenta on top, the published
pose in orange and a yellow arrow from published to fitted. Failed: the tracks used in cyan,
and the bright stars in orange under the published pose with the shared lens. The drawing is
the library's; this finds the frame -- in a FIgLib archive or the HPWREN cache -- and writes
out/sky/star_solve_<seq>.jpg.

    python -m src.figlib.stars.fig_track_solve <seq> ... [--wide]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from star_calibration import overlay

from . import solve as S

SKY = S.SKY


def ref_frame(seq: str, ref: int) -> np.ndarray:
    if seq.startswith("hpwren_"):
        from star_calibration.hpwren import nights
        _, _, blob = nights.frame_at(seq, ref)
    else:
        from .. import corpus as C
        from ..detect_yolo import read_frames
        arch = {p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}[seq.split("#")[0]]
        _, _, blob = min(read_frames(arch), key=lambda f: abs(f[1] - ref))
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def render(seq: str, wide: bool = False) -> Path:
    """Draw one sequence's solve. `wide` reads the result from `solve_wide_summary.json`
    instead of the per-sequence file, so a camera recovered by the pole search can be looked
    at without overwriting the committed solve the pose ledger is built from."""
    if wide:
        rows = json.loads((S.DATA / "solve_wide_summary.json").read_text())
        r = next(x for x in rows if x["seq"] == seq)
    else:
        r = json.loads((S.DATA / f"solve_{seq}.json").read_text())
    night = S.night(seq)
    img = overlay.draw(r, night, ref_frame(seq, overlay.reference_offset(r, night)))
    dest = SKY / f"star_solve_{seq.replace('#', '_')}.jpg"
    overlay.write(img, dest)
    return dest


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--wide"]
    wide = "--wide" in sys.argv
    for seq in args:
        print(render(seq, wide=wide))
