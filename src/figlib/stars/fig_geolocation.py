"""Geolocation before and after the star calibration, from geolocate.py's own outputs.

Before: `geolocation.json` (published azimuth, rectilinear nameplate lens). After:
`geolocation_fisheye_ledger_full.json` (FIGLIB_PROFILE=calibrated: the star-measured lens and
each camera's whole star solve). Same detections and solver, so every difference is the
calibration.

  docs/figures/calibration_maps.png    bearing rays and estimates for a few fires
  docs/figures/calibration_bearings.png  every confirmed-fire bearing's miss vs frame position

    python -m src.figlib.stars.fig_geolocation [fire_id ...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import corpus as C
from .. import geom

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BEFORE, AFTER = MUTED, "#2a78d6"
TAG = "center"


def _load(variant: str) -> dict[str, dict]:
    p = C.current().out / f"geolocation{C.tier_suffix(C.tier_filter())}{variant}.json"
    return {r["fire_id"]: r for r in json.loads(p.read_text())}


def _enu(lat, lon, lat0, lon0):
    return (lon - lon0) * 111.32 * math.cos(math.radians(lat0)), (lat - lat0) * 111.13


def _chrome(ax):
    ax.set_facecolor(SURFACE)
    for side in ax.spines.values():
        side.set_color(GRID)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def maps(before, after, fires, cams, dest=DOCS / "calibration_maps.png"):
    cols = 3
    rows = math.ceil(len(fires) / cols)
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5.1 * rows + 0.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat[len(fires):]:
        ax.set_axis_off()
    for ax, fid in zip(axes.flat, fires):
        b, a = before[fid], after[fid]
        lat0, lon0 = b["truth_lat"], b["truth_lon"]
        eb, ea = b[TAG]["error_km"], a[TAG]["error_km"]
        half = max(2.5, 1.6 * max(eb, ea))
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
        _chrome(ax)
        for run, color, lw, z in ((b, BEFORE, 1.5, 2), (a, AFTER, 2.0, 3)):
            for br in run[TAG]["bearings"]:
                cam = cams[br["camera"]]
                x0, y0 = _enu(cam["lat"], cam["lon"], lat0, lon0)
                t = np.linspace(0, math.hypot(x0, y0) + 3 * half, 800)
                th = math.radians(br["deg"])
                xs, ys = x0 + t * math.sin(th), y0 + t * math.cos(th)
                ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=z)
                if run is a:
                    inside = np.where((np.abs(xs) < half * 0.96) & (np.abs(ys) < half * 0.96))[0]
                    if len(inside):
                        label = br["camera"] + (f"  {br['pose']['d_az']:+.1f}°" if br.get("pose") else "")
                        ax.annotate(label, (xs[inside[0]], ys[inside[0]]), xytext=(3, 3),
                                    textcoords="offset points", fontsize=7.5, color=INK2, zorder=6)
            ex, ey = _enu(run[TAG]["est_lat"], run[TAG]["est_lon"], lat0, lon0)
            ax.scatter([ex], [ey], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax.scatter([0], [0], s=150, marker="*", color=INK, edgecolors=SURFACE, linewidths=1.5, zorder=6)
        ax.set_title(f"{fid}  ({b['tier']})\n{eb:.2f} km → {ea:.2f} km", fontsize=10, color=INK, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("km north of truth", color=INK2)
    for ax in axes[-1]:
        ax.set_xlabel("km east of truth", color=INK2)
    handles = [plt.Line2D([], [], color=BEFORE, lw=1.5, marker="o", markersize=7, markeredgecolor=SURFACE,
                          label="before: published azimuth, rectilinear 90° lens"),
               plt.Line2D([], [], color=AFTER, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                          label="after: star-measured fisheye lens + pose ledger (labels show the correction)"),
               plt.Line2D([], [], color=INK, lw=0, marker="*", markersize=11, label="official ignition point")]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 0.998),
               fontsize=10, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(dest, dpi=105)
    plt.close(fig)
    return dest


def bearings(before, after, cams, dest=DOCS / "calibration_bearings.png"):
    pts = []
    for fid, b in before.items():
        if b["tier"] != "confirmed" or fid not in after or b[TAG].get("status") != "solved" \
                or after[fid][TAG].get("status") != "solved":
            continue
        for b0, b1 in zip(b[TAG]["bearings"], after[fid][TAG]["bearings"]):
            cam = cams[b0["camera"]]
            tb = geom.bearing_deg(cam["lat"], cam["lon"], b["truth_lat"], b["truth_lon"])
            pts.append((b0["x"], geom.angdiff_deg(b0["deg"], tb), geom.angdiff_deg(b1["deg"], tb), b0["camera"]))
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.patch.set_facecolor(SURFACE)
    _chrome(ax)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for x, m0, m1, _ in pts:
        ax.plot([x, x], [m0, m1], color=AXIS, lw=1, zorder=1)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=56, color=BEFORE, edgecolors=SURFACE, linewidths=2,
               zorder=3, label="before")
    ax.scatter([p[0] for p in pts], [p[2] for p in pts], s=56, color=AFTER, edgecolors=SURFACE, linewidths=2,
               zorder=4, label="after")
    ax.axhline(0, color=AXIS, lw=1, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("horizontal position of the detection in the frame (0 = left edge, 1 = right edge)", color=INK2)
    ax.set_ylabel("bearing miss vs official ignition point (deg)", color=INK2)
    outer = [p for p in pts if abs(p[0] - 0.5) > 0.25]
    inner = [p for p in pts if abs(p[0] - 0.5) <= 0.25]

    def med(ps, i):
        return float(np.median([abs(p[i]) for p in ps])) if ps else float("nan")
    summary = (f"outer half of the frame (n={len(outer)}): median |miss| {med(outer, 1):.1f}° → {med(outer, 2):.1f}°\n"
               f"central half (n={len(inner)}): {med(inner, 1):.1f}° → {med(inner, 2):.1f}°")
    ax.text(0.01, 0.03, summary, transform=ax.transAxes, va="bottom", fontsize=10, color=INK2)
    for p in sorted(pts, key=lambda p: -abs(abs(p[1]) - abs(p[2])))[:4]:
        ax.annotate(p[3], (p[0], p[2]), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK2)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2)
    ax.set_title("Confirmed fires: every camera bearing's miss, before → after calibration", loc="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest, summary


def main(argv):
    before, after = _load(""), _load("_fisheye_ledger_full")
    cams = geom.load_cams()
    fires = argv
    if not fires:
        both = [f for f in before if f in after and before[f][TAG].get("status") == "solved"
                and after[f][TAG].get("status") == "solved"]
        delta = {f: after[f][TAG]["error_km"] - before[f][TAG]["error_km"] for f in both}
        conf = sorted((f for f in both if before[f]["tier"] == "confirmed"), key=lambda f: delta[f])
        worse = sorted(both, key=lambda f: -delta[f])
        fires = conf[:4] + [f for f in worse if f not in conf[:4]][:2]
    print(maps(before, after, fires, cams))
    dest, summary = bearings(before, after, cams)
    print(dest); print(summary)


if __name__ == "__main__":
    main(sys.argv[1:])
