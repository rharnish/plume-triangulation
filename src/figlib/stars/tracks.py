"""Moving star tracks for one sequence, by name: frames from FIgLib or the HPWREN cache.

Detection and linking are the star_calibration library's (`star_calibration.tracks`); this
module only finds a sequence's frames. A FIgLib sequence is read from its archive
(detect_yolo.read_frames); an "hpwren_*" block from the shared HPWREN cache
(star_calibration.hpwren.nights).

    python -m src.figlib.stars.tracks [seq]
"""
from __future__ import annotations

import sys

import numpy as np
from star_calibration import tracks as L
from star_calibration.hpwren import nights as hpwren_nights

from .. import corpus as C
from ..detect_yolo import read_frames
from .solve import CAMS, SEQS


def frames(seq: str) -> list[tuple[int, int, bytes]]:
    """(epoch, offset, jpeg bytes) in time order, wherever the sequence lives."""
    if seq.startswith("hpwren_"):
        return hpwren_nights.read_frames(seq)
    arch = {p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}
    return sorted(read_frames(arch[seq.split("#")[0]]), key=lambda f: f[1])  # by offset


def collect(seq: str, el_max: float = -8.0, max_step_px: float = 20.0,
            min_frames: int = 8, min_span_px: float = 50.0, keep_frames: bool = True,
            max_gap_s: float | None = None):
    """(tracks, decoded frames, sequence record) for one sequence; see
    star_calibration.tracks.collect for the parameters."""
    s = SEQS[seq]
    mono = CAMS[s["camera"]].get("imager") == "monochrome"
    good, decoded = L.collect(frames(seq), s["lat"], s["lon"], mono=mono, el_max=el_max,
                              max_step_px=max_step_px, min_frames=min_frames,
                              min_span_px=min_span_px, keep_frames=keep_frames,
                              max_gap_s=max_gap_s, label=seq)
    return good, decoded, s


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "20241021_PalomarRidge_hp-s-mobo-c"
    good, decoded, s = collect(seq)
    for t in sorted(good, key=lambda t: -len(t))[:10]:
        offsets = sorted(t)
        (x0, y0, a0), (x1, y1, a1) = t[offsets[0]], t[offsets[-1]]
        print(f"  track: {len(t)} frames, offsets {offsets[0]}..{offsets[-1]}, "
              f"({x0:.0f},{y0:.0f}) amp{a0} -> ({x1:.0f},{y1:.0f}) amp{a1}, "
              f"moved {np.hypot(x1-x0, y1-y0):.1f}px")
