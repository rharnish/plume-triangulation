"""Plot the likelihood surface, the bearings that produced it, and the truth.

The point of drawing this rather than reporting a distance is that the shape carries
information a single number cannot. Two cameras 90 degrees apart pin a fire tightly;
two nearly in line with it produce a long ridge of almost equal likelihood, and an
estimate drawn from the middle of that ridge is confident about the wrong thing. The
contours show which situation you are in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def render(fire_id: str, bearings, lats, lons, ll, est, truth, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 8.5), dpi=130)
    rel = ll - ll.max()
    extent = [lons[0], lons[-1], lats[0], lats[-1]]

    ax.imshow(rel, origin="lower", extent=extent, aspect="auto",
              cmap="magma", vmin=-12, vmax=0)
    cs = ax.contour(lons, lats, rel, levels=[-6.0, -3.0, -1.0],
                    colors=["#5fa8ff", "#8fd6ff", "#ffffff"], linewidths=1.1)
    ax.clabel(cs, fmt={-6.0: "-6", -3.0: "95%", -1.0: "peak"}, fontsize=7)

    span = max(lats[-1] - lats[0], lons[-1] - lons[0])
    for b in bearings:
        th = math.radians(b.bearing_deg)
        dlat = math.cos(th) * span
        dlon = math.sin(th) * span / math.cos(math.radians(b.lat))
        ax.plot([b.lon, b.lon + dlon], [b.lat, b.lat + dlat],
                color="#4dd2a0", lw=1.0, alpha=0.85, zorder=3)
        ax.plot(b.lon, b.lat, "^", color="#4dd2a0", ms=7, zorder=4)
        ax.annotate(f"{b.camera}\n{b.conf:.2f}", (b.lon, b.lat),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=6, color="#4dd2a0")

    ax.plot(est[1], est[0], "x", color="#ff4d6d", ms=14, mew=3,
            zorder=5, label="estimate")
    ax.plot(truth[1], truth[0], "o", mfc="none", mec="#ffd166", ms=16, mew=2.5,
            zorder=5, label="official ignition point")

    from .geom import haversine_km
    err = haversine_km(est[0], est[1], truth[0], truth[1])
    ax.set_title(f"{fire_id}   error {err:.2f} km   "
                 f"{len({b.camera.split('-')[0] for b in bearings})} sites",
                 fontsize=11)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.35)
    ax.set_xlim(lons[0], lons[-1]); ax.set_ylim(lats[0], lats[-1])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out
