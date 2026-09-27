"""File the FIgLib star solves in the shared HPWREN cache, so its one gallery shows them.

The cache ($HPWREN_CACHE, star_calibration.hpwren) solves and draws its own CDN blocks; the
FIgLib archive sequences are only readable here. This takes each committed
out/sky/data/star_tracks/solve_<seq>.json that isn't a CDN block, draws its overlay on the
archive's own frame, and hands both to star_calibration.hpwren.calibrate.add. Then, from
star-calibration: `python -m star_calibration.hpwren.weather` and `... .gallery`.

    python -m src.figlib.stars.publish [seq ...]    # every FIgLib solve if none named
"""
from __future__ import annotations

import json
import sys

from star_calibration import overlay
from star_calibration.hpwren import calibrate

from . import solve as S
from .fig_track_solve import ref_frame


def publish(seq: str) -> None:
    r = json.loads((S.DATA / f"solve_{seq}.json").read_text())
    night = S.night(seq)
    ref = overlay.reference_offset(r, night)
    image = overlay.draw(r, night, ref_frame(seq, ref))
    calibrate.add(r | {"camera": night.camera, "t0": night.t0, "lat": night.cam["lat"],
                       "lon": night.cam["lon"], "ref_offset": ref}, image, "figlib")


if __name__ == "__main__":
    seqs = sys.argv[1:] or sorted(p.stem[len("solve_"):] for p in S.DATA.glob("solve_*.json")
                                  if p.stem not in ("solve_summary", "solve_wide_summary",
                                                    "solve_weather")
                                  and not p.stem.startswith("solve_hpwren_"))
    for seq in seqs:
        publish(seq)
        print(seq, flush=True)
    print(f"filed {len(seqs)} solves in {calibrate.solves_dir()}")
