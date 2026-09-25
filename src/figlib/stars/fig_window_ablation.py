"""Figure for stars.window_ablation: what a longer calibration window buys, and what the sky
model does, per camera.

Reads out/sky/data/window_ablation/summary_*.json (python -m src.figlib.stars.window_ablation
--summary). Dashed lines are the current sky model (J2000, no refraction), solid lines have
precession and refraction on; colour is the camera.

  top    -- against window length: repeatability of the boresight across disjoint windows,
            and the median residual on stars outside the window; then how a 90-minute
            window's prediction error grows with time from the window;
  bottom -- the pose shift the two terms cause (the bias in every current solve), the
            full-night fit under each model, and residual against altitude.

    python -m src.figlib.stars.fig_window_ablation
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import solve as S
from .window_ablation import BASE, OUT, REF

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # fixed order, one per camera
STYLE = {"cur": dict(ls=(0, (4, 3)), lw=1.6, marker="o", ms=4, mfc=SURFACE),
         "ref": dict(ls="-", lw=2, marker="o", ms=5)}
KEY = lambda m: tuple(sorted(m.items()))


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _nanmed(rows, field):
    a = np.array([[b[3] if (b[3] is not None and b[2] > 20) else np.nan for b in r[field]] for r in rows], float)
    with np.errstate(all="ignore"):
        return np.nanmedian(a, 0)


def main():
    sums = [json.loads(p.read_text()) for p in sorted(OUT.glob("summary_*.json"))]
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.8))
    fig.patch.set_facecolor(SURFACE)
    night_min = np.mean([s["night_hours"] for s in sums]) * 60
    ticks = [15, 30, 60, 90, 180, 360, night_min]
    labels = ["15", "30", "60", "90", "180", "360", "night"]
    dt_mid = None

    for i, s in enumerate(sums):
        col, cam = SERIES[i], s["camera"]
        el = np.radians(np.mean([r["boresight"][1] for r in s["runs"] if r.get("boresight")]))
        for model, st in ((BASE, "cur"), (REF, "ref")):
            sp = [x for x in s["spread"] if x["sky_model"] == model]
            rep = [x for x in sp if x["solved"] >= 2]
            ax[0, 0].plot([x["length_min"] or night_min for x in rep],
                          [np.hypot(x["az_sd"] * np.cos(el), x["el_sd"]) for x in rep], color=col, **STYLE[st])
            ho = [x for x in sp if x["solved"] and x["held_out_median_px"] is not None]
            ax[0, 1].plot([x["length_min"] or night_min for x in ho], [x["held_out_median_px"] for x in ho],
                          color=col, **STYLE[st])
            w90 = [r for r in s["runs"] if r["length_min"] == 90 and r["sky_model"] == model and r.get("by_dt")]
            if w90:
                bins = w90[0]["by_dt"]
                dt_mid = [0.5 * (b[0] + b[1]) / 60 for b in bins]
                ax[0, 2].plot(dt_mid, _nanmed(w90, "by_dt"), color=col, **STYLE[st])
        pq = s["production_q1"]
        if pq:
            ax[0, 1].plot([90], [pq["held_out_median_px"]], marker="D", ms=8, color=col, lw=0,
                          markeredgecolor=SURFACE, markeredgewidth=1.5)

        full = {KEY(r["sky_model"]): r for r in s["runs"] if r["length_min"] is None and r["status"] == "solved"}
        b0, b1 = np.array(full[KEY(BASE)]["boresight"]), np.array(full[KEY(REF)]["boresight"])
        d = b1 - b0
        d[0] = (d[0] + 180) % 360 - 180
        ax[1, 0].bar(np.arange(3) + (i - 1.5) * 0.2, d, width=0.18, color=col)
        order = [dict(proper_motion=False, precession=p, refraction=r) for p in (False, True) for r in (False, True)]
        ax[1, 1].bar(np.arange(4) + (i - 1.5) * 0.2, [full[KEY(m)]["median_px"] for m in order],
                     width=0.18, color=col)
        for refr, st in ((False, "cur"), (True, "ref")):
            r = full[KEY(dict(proper_motion=False, precession=True, refraction=refr))]
            m = _nanmed([r], "by_alt")
            x = [0.5 * (b[0] + b[1]) for b in r["by_alt"]]
            ax[1, 2].plot(x, m, color=col, **STYLE[st])

    for a in ax[0, :2]:
        a.set_xscale("log"); a.set_xticks(ticks); a.set_xticklabels(labels); a.minorticks_off()
    ax[0, 0].set_yscale("log")
    _style(ax[0, 0], "Boresight repeatability across disjoint windows", "window length (min)",
           "sd of boresight direction (deg)")
    _style(ax[0, 1], "Stars outside the window: median residual\n(◆ = today's 90-frame Q1 solve)",
           "window length (min)", "median residual (px)")
    _style(ax[0, 2], "90-minute windows: error vs time from the window", "|time from window centre| (h)",
           "median residual (px)")
    if dt_mid:
        ax[0, 2].set_xticks(dt_mid); ax[0, 2].set_xticklabels(["0-½", "½-1", "1-2", "2-4", "4+"])
    _style(ax[1, 0], "Full-night pose: shift when both terms are switched on\n(the bias in every current solve)",
           "", "degrees")
    ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels(["azimuth", "pitch", "roll"], color=INK2)
    ax[1, 0].axhline(0, color=AXIS, lw=1)
    _style(ax[1, 1], "Full-night fit residual under each sky model", "", "median residual (px)")
    ax[1, 1].set_xticks(range(4))
    ax[1, 1].set_xticklabels(["J2000, no refr.\n(current)", "J2000,\nrefraction", "precessed,\nno refr.",
                              "precessed,\nrefraction"], fontsize=8.5, color=INK2)
    _style(ax[1, 2], "Full-night fit (precessed): residual by altitude\ndashed: no refraction   solid: refraction",
           "altitude (deg)", "median residual (px)")

    cams = [plt.Line2D([], [], color=SERIES[i], lw=6, label=s["camera"]) for i, s in enumerate(sums)]
    styles = [plt.Line2D([], [], color=MUTED, **STYLE["cur"], label="current model (J2000, no refraction)"),
              plt.Line2D([], [], color=MUTED, **STYLE["ref"], label="precession + refraction")]
    fig.legend(handles=cams + styles, loc="upper left", ncol=6, frameon=False, bbox_to_anchor=(0.005, 0.965),
               labelcolor=INK2, fontsize=9.5)
    fig.text(0.005, 0.995, "How much night does a star calibration need?  Four cameras, one new-moon night "
             "(2026-07-13/14), windows from 15 min to 8.5 h", ha="left", va="top", color=INK, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    dest = S.SKY / "window_ablation.png"
    fig.savefig(dest, dpi=105)
    print(dest)


if __name__ == "__main__":
    main()
