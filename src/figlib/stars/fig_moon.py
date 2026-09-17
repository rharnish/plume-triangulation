"""The moon ladder: star solves against how much moon was in the sky.

Left: stars matched per solve, against moon_light (illuminated fraction times sin elevation,
0 whenever the moon is down). Right: each night's pose against that camera's own dark-night
pose, as boresight separation in degrees. If moonlight mattered, the left panel would fall
and the right would rise. Neither happens.

    python -m src.figlib.stars.fig_moon
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import cross_night as X
from . import solve as S

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "sky"
COLORS = {"hp-w-mobo-c": "#2b7bba", "vo-e-mobo-c": "#d1495b", "cp-w-mobo-c": "#3a9e6e"}


def main() -> Path:
    rows = json.loads((OUT / "moon_exp" / "moon_ladder.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for cam, col in COLORS.items():
        g = sorted([r for r in rows if r["camera"] == cam], key=lambda r: r["moon_light"])
        if not g:
            continue
        x = [r["moon_light"] for r in g]
        a1.plot(x, [r["n_stars"] for r in g], "o-", color=col, label=cam, ms=6)
        dark = next((r for r in g if r["moon_light"] < 0.01 and r["pose"]), None)
        if dark:
            c = S.CAMS[cam]
            sep = [X.disagreement(c, r["pose"], dark["pose"])[0] if r["pose"] else np.nan
                   for r in g]
            a2.plot(x, sep, "o-", color=col, label=cam, ms=6)
    a1.set_xlabel("moon in the sky  (illuminated fraction $\\times$ sin(elevation))")
    a1.set_ylabel("stars matched")
    a1.set_title("Stars matched does not fall with moonlight")
    a1.set_ylim(0, 50)
    a2.set_xlabel("moon in the sky  (illuminated fraction $\\times$ sin(elevation))")
    a2.set_ylabel("boresight separation from the dark-night pose (deg)")
    a2.set_title("Neither does the pose move")
    a2.set_ylim(0, 0.12)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(loc="lower left" if a is a1 else "upper left", fontsize=9)
        a.axvspan(-0.02, 0.01, color="0.85", zorder=0)
        a.text(0.0, a.get_ylim()[1] * 0.97, " moon down", fontsize=8, va="top", color="0.35")
        a.set_xlim(-0.03, 0.83)
    full = max(r["moon_light"] for r in rows)
    fig.suptitle(f"23 night blocks, 3 cameras, 2026-08-29 to 09-11: full moon {full:.2f} "
                 f"down to new, every one solved", fontsize=12)
    fig.tight_layout()
    dest = OUT / "moon_ladder.jpg"
    fig.savefig(dest, dpi=110)
    return dest


if __name__ == "__main__":
    print(main())
