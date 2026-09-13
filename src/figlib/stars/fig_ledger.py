"""What the star solves say about the camera table: per-camera azimuth corrections and the lens.

Reads data/meta/pose_ledger.json. Left: every solve's azimuth correction, one row per
camera, sorted by the largest |correction|; the gray band is +-1 deg. Right: the same solves'
lens scale (k as a fraction of the nameplate), which should cluster if the lens is one design.
Color is the data source: FIgLib night sequences (2019-2025) or HPWREN CDN nights (2026).

    python -m src.figlib.stars.fig_ledger       # -> docs/figures/star_ledger.png
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "docs" / "figures" / "star_ledger.png"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FIGLIB, CDN = "#2a78d6", "#eb6834"          # categorical slots 1 and 2


def main(dest: Path = DEST) -> Path:
    ledger = json.loads((ROOT / "data" / "meta" / "pose_ledger.json").read_text())
    by_cam = defaultdict(list)
    for e in ledger:
        by_cam[e["camera"]].append(e)
    cams = sorted(by_cam, key=lambda c: max(abs(e["d_az"]) for e in by_cam[c]))
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 0.24 * len(cams) + 1.6), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 1.2]})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(AXIS)
        a.tick_params(colors=MUTED, length=0)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
    ax.axvspan(-1, 1, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    for i, cam in enumerate(cams):
        es = by_cam[cam]
        if len(es) > 1:
            ax.plot([min(e["d_az"] for e in es), max(e["d_az"] for e in es)], [i, i], color=AXIS, lw=1, zorder=2)
        for e in es:
            color = CDN if e["source"].startswith("star:hpwren_") else FIGLIB
            ax.scatter(e["d_az"], i, s=42, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            bx.scatter(e["k_ratio"], i, s=42, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_yticks(range(len(cams)), cams, color=INK2, fontsize=8.5)
    ax.set_ylim(-0.7, len(cams) - 0.3)
    ax.set_xlabel("azimuth correction vs published (deg)", color=INK2)
    bx.set_xlabel("lens scale k / nameplate", color=INK2)
    ks = np.array([e["k_ratio"] for e in ledger])
    bx.set_title(f"median {np.median(ks):.3f}, range {ks.min():.3f}–{ks.max():.3f}", loc="left",
                 fontsize=9, color=INK2)
    n_big = sum(1 for c in cams if max(abs(e["d_az"]) for e in by_cam[c]) > 1)
    ax.set_title(f"{len(ledger)} star solves, {len(cams)} cameras: {n_big} need more than 1° "
                 f"(gray band = ±1°)", loc="left", fontsize=10, color=INK)
    handles = [plt.Line2D([], [], lw=0, marker="o", markersize=7, color=FIGLIB, label="FIgLib night sequence, 2019–2025"),
               plt.Line2D([], [], lw=0, marker="o", markersize=7, color=CDN, label="HPWREN public CDN night, 2026")]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, labelcolor=INK2, fontsize=9.5,
               bbox_to_anchor=(0.01, 0.999))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    print(main())
