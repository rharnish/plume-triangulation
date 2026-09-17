"""Where the estimates land relative to the official ignition point, all fires on one plot.

Every scoring fire is re-centred on its own WFIGS/IRWIN coordinate, so the origin is "the
official ignition" and each dot is that fire's estimate displaced east/north in km. Left:
the published pose, full extent, which shows the probable-tier tail. Right: the inner 5 km,
under the star-measured fisheye lens, with a thin tail back to where the published pose put it.

    python -m src.figlib.fig_offsets    # writes out/figures/offsets.png
"""

from __future__ import annotations

import json
import math

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import corpus as C

KM_PER_DEG = 111.195
TIER_COLOR = {"confirmed": "#2a78d6", "probable": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


def offsets(path) -> dict[str, dict]:
    out = {}
    for r in json.loads(path.read_text()):
        c = r.get("center", {})
        if c.get("status") != "solved":
            continue
        de = (c["est_lon"] - r["truth_lon"]) * KM_PER_DEG * math.cos(math.radians(r["truth_lat"]))
        dn = (c["est_lat"] - r["truth_lat"]) * KM_PER_DEG
        out[r["fire_id"]] = dict(tier=r["tier"], name=r["truth_name"], e=de, n=dn,
                                 km=c["error_km"])
    return out


def upper_median(xs):
    return sorted(xs)[len(xs) // 2]


def rings(ax, radii, label_angle=-120):
    for rad in radii:
        ax.add_patch(plt.Circle((0, 0), rad, fill=False, color=GRID, lw=0.9, zorder=0))
        a = math.radians(label_angle)
        ax.text(rad * math.cos(a), rad * math.sin(a), f"{rad:g} km", fontsize=7.5,
                color=INK2, ha="right", va="top", zorder=1)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)


def truth_marker(ax):
    ax.plot(0, 0, "o", mfc="white", mec=INK, ms=11, mew=2, zorder=6)
    ax.plot(0, 0, "o", color=INK, ms=2.5, zorder=7)


def style(ax, lim, title):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("east of official ignition (km)", color=INK2)
    ax.set_ylabel("north of official ignition (km)", color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def main() -> None:
    out = C.current().out
    pub = offsets(out / "geolocation.json")
    fish = offsets(out / "geolocation_fisheye.json")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13.5, 6.9))
    fig.patch.set_facecolor("white")

    # left: published pose, everything
    lim0 = 56
    rings(a0, [10, 25, 50], label_angle=-60)
    for tier in ("probable", "confirmed"):
        pts = [p for p in pub.values() if p["tier"] == tier]
        a0.scatter([p["e"] for p in pts], [p["n"] for p in pts], s=46, color=TIER_COLOR[tier],
                   edgecolor="white", lw=1.2, zorder=4)
    for p in pub.values():
        if p["km"] > 20:
            a0.annotate(f"{p['name']}  {p['km']:.1f} km", (p["e"], p["n"]), xytext=(8, -3),
                        textcoords="offset points", fontsize=8, color=INK2)
    truth_marker(a0)
    a0.add_patch(plt.Rectangle((-5, -5), 10, 10, fill=False, ec=INK2, lw=0.9, ls=(0, (3, 2)),
                               zorder=5))
    style(a0, lim0, f"Published pose, all {len(pub)} scoring fires")

    # right: star-measured fisheye lens, inner 5 km, tails back to the published pose
    lim1 = 5.0
    rings(a1, [1, 2, 3, 4])
    for fid, p in fish.items():
        q = pub.get(fid)
        if q and (abs(q["e"]) <= lim1 * 1.05 and abs(q["n"]) <= lim1 * 1.05) \
                and math.hypot(p["e"] - q["e"], p["n"] - q["n"]) > 0.05:
            a1.plot([q["e"], p["e"]], [q["n"], p["n"]], color=TIER_COLOR[p["tier"]], lw=1,
                    alpha=0.45, zorder=2)
            a1.plot(q["e"], q["n"], "o", mfc="white", mec=TIER_COLOR[p["tier"]], ms=4.5,
                    mew=1, alpha=0.8, zorder=3)
    inside = {f: p for f, p in fish.items() if abs(p["e"]) <= lim1 and abs(p["n"]) <= lim1}
    for tier in ("probable", "confirmed"):
        pts = [p for p in inside.values() if p["tier"] == tier]
        a1.scatter([p["e"] for p in pts], [p["n"] for p in pts], s=70, color=TIER_COLOR[tier],
                   edgecolor="white", lw=1.4, zorder=4)
    for p in inside.values():
        if p["tier"] == "confirmed":
            left = p["name"] == "ROUND"
            a1.annotate(p["name"], (p["e"], p["n"]), xytext=(-7, -9) if left else (7, 4),
                        textcoords="offset points", ha="right" if left else "left",
                        fontsize=7.5, color=INK2, zorder=5)
    truth_marker(a1)
    conf = [p["km"] for p in fish.values() if p["tier"] == "confirmed"]
    med = upper_median(conf)
    a1.add_patch(plt.Circle((0, 0), med, fill=False, color=TIER_COLOR["confirmed"], lw=1.3,
                            ls=(0, (5, 3)), zorder=1))
    a1.text(0.6, med + 0.1, f"confirmed upper median {med:.2f} km", fontsize=8,
            color=INK2, ha="center", va="bottom")
    n_out = len(fish) - len(inside)
    a1.text(0.98, 0.02, f"{n_out} probable-tier fires fall outside this window",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    style(a1, lim1, f"Star-measured fisheye lens, inner {lim1:g} km")

    legend = [
        Line2D([], [], marker="o", ls="none", color=TIER_COLOR["confirmed"], mec="white", ms=8,
               label=f"name-confirmed truth (n={sum(p['tier'] == 'confirmed' for p in pub.values())})"),
        Line2D([], [], marker="o", ls="none", color=TIER_COLOR["probable"], mec="white", ms=8,
               label=f"probable truth (n={sum(p['tier'] == 'probable' for p in pub.values())})"),
        Line2D([], [], marker="o", ls="-", lw=1, color=INK2, mfc="white", ms=5,
               label="right: where the published pose put it"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=2, ms=10,
               label="official ignition (WFIGS/IRWIN)"),
        Line2D([], [], ls=(0, (3, 2)), color=INK2, lw=1, label="left: extent of the right panel"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    dst = out / "figures" / "offsets.png"
    fig.savefig(dst, dpi=140)
    print(f"wrote {dst}")

    for label, d in (("published", pub), ("fisheye", fish)):
        e = np.array([p["e"] for p in d.values() if p["tier"] == "confirmed"])
        n = np.array([p["n"] for p in d.values() if p["tier"] == "confirmed"])
        print(f"{label:9s} confirmed mean offset: east {e.mean():+.2f} km, north {n.mean():+.2f} km")


if __name__ == "__main__":
    main()
