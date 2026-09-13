"""Loss profiles of the star-track pose fit, per parameter, for a few sequences.

The loss is the one stars.solve's final refit minimizes: scipy least_squares with
loss="soft_l1", f_scale=4 px over every x and y residual of the matched star<->track pairs,
    L(p) = sum_i f^2 * 2 * (sqrt(1 + (r_i/f)^2) - 1)        (~ sum r_i^2 for small r)
with the star<->track assignment held at the solve's final matches.

Plotted as a chi-square-like curve so widths mean uncertainty, not just pixel sensitivity:
    y = (L(p) - L(p_hat)) / sigma^2 / m
sigma is the robust residual scale at the optimum (1.4826 * median |r|); m is the mean number
of points per matched star. Points along one track are strongly correlated (one star, one
smooth arc), so dividing by m counts each star once, not each frame. y = 1 is then roughly
the 1-sigma boundary; the titles give the profile's 1-sigma half-width.

Two curves per parameter:
  slice   - vary this parameter, hold the other four at the optimum;
  profile - vary this parameter, re-fit the other four at each step.
A profile much wider than its slice means the parameter trades off against another one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from . import catalog as SG
from . import solve as S
from .fisheye import initial_k, project_fisheye

SKY = S.SKY

F = 4.0
NAMES = ["d_az (deg)", "d_pitch (deg)", "d_roll (deg)", "k / nameplate", "k1"]
YMAX = 8.0
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PROFILE, SLICE = "#2a78d6", MUTED


def problem(seq):
    r = json.loads((S.DATA / f"solve_{seq}.json").read_text())
    tracks, (W, H) = S.load_tracks(seq)
    s, c = S.SEQS[seq], S.CAMS[S.SEQS[seq]["camera"]]
    k0 = initial_k(c, W)
    al, az, obs = [], [], []
    for name, j in r["matches"].items():
        t = tracks[j]
        offs = sorted(t)
        a, z = SG.altaz(*SG.STARS[name], s["t0"] + np.array(offs, float), c["lat"], c["lon"])
        al.append(np.atleast_1d(a)); az.append(np.atleast_1d(z))
        obs.append(np.array([t[o][:2] for o in offs]))
    al, az, obs = np.concatenate(al), np.concatenate(az), np.concatenate(obs)

    def resid(p):   # k expressed as a ratio of nameplate
        x, y = project_fisheye(c, az, al, W, H, p[0], p[1], p[2], p[3] * k0, p[4])
        return np.r_[x * W - obs[:, 0], y * H - obs[:, 1]]

    p0 = np.array([r["pose"][k] for k in ("d_az", "d_pitch", "d_roll", "k_ratio", "k1")])
    sol = least_squares(resid, p0, loss="soft_l1", f_scale=F, max_nfev=5000)
    return r, resid, sol, len(obs)


def total_loss(res):
    return float(np.sum(F ** 2 * 2 * (np.sqrt(1 + (res / F) ** 2) - 1)))


def half_width(grid, y, center):
    """Half-width of the y <= 1 interval around the optimum (None if it runs off the grid)."""
    inside = y <= 1.0
    i0 = int(np.argmin(np.abs(grid - center)))
    lo = i0
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(grid) - 1 and inside[hi + 1]:
        hi += 1
    if lo == 0 or hi == len(grid) - 1:
        return None
    return (grid[hi] - grid[lo]) / 2


def render(seqs, dest):
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, axes = plt.subplots(len(seqs), 5, figsize=(17, 3.0 * len(seqs) + 0.9), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, seq in enumerate(seqs):
        r, resid, sol, n_pts = problem(seq)
        p_hat = sol.x
        l_min = total_loss(sol.fun)
        sigma = 1.4826 * float(np.median(np.abs(sol.fun)))
        m = n_pts / len(r["matches"])
        scale = sigma ** 2 * m
        # Gauss-Newton marginal sigma with the same per-star decorrelation; panels span +-4 of
        # them so every curve is resolved, and the x ticks carry the absolute scale.
        J = sol.jac
        gn = np.sqrt(np.diag(np.linalg.pinv(J.T @ J)) * scale)
        widths = []
        for i in range(5):
            ax = axes[row, i]
            ax.set_facecolor(SURFACE)
            grid = p_hat[i] + np.linspace(-4 * gn[i], 4 * gn[i], 81)
            sl = np.array([(total_loss(resid(np.r_[p_hat[:i], g, p_hat[i + 1:]])) - l_min) / scale
                           for g in grid])
            pr = np.empty(len(grid))
            mid = len(grid) // 2
            # sweep outward from the optimum both ways, each re-fit warm-started from its
            # neighbor nearer the center, so neither half starts from a far-off pose
            for order in (range(mid, len(grid)), range(mid, -1, -1)):
                warm = np.delete(p_hat, i)
                for k in order:
                    sub = least_squares(lambda q, g=grid[k]: resid(np.insert(q, i, g)), warm,
                                        loss="soft_l1", f_scale=F, max_nfev=2000)
                    warm = sub.x
                    pr[k] = (total_loss(sub.fun) - l_min) / scale
            hw = half_width(grid, pr, p_hat[i])
            widths.append(hw)
            ax.plot(grid, sl, color=SLICE, lw=1.5)
            ax.plot(grid, pr, color=PROFILE, lw=2)
            ax.axhline(1.0, color=AXIS, lw=1, zorder=0)
            ax.axvline(p_hat[i], color=GRID, lw=1, zorder=0)
            ax.set_ylim(0, YMAX); ax.set_xlim(grid[0], grid[-1])
            ax.ticklabel_format(axis="x", useOffset=False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=MUTED, length=0, labelsize=8)
            fmt = "{:+.2f}" if i < 3 else "{:.3f}"
            wtxt = f"\u00b1{hw:.2g}" if hw is not None else f"> \u00b1{4 * gn[i]:.2g}"
            ax.set_title(f"{NAMES[i]} = {fmt.format(p_hat[i])}   1σ {wtxt}", fontsize=9,
                         color=INK, loc="left")
            if i == 0:
                tag = "solved" if r["status"] == "solved" else "below cutoff"
                ax.set_ylabel(f"{seq.split('_', 1)[1]}  ({tag})\n{len(r['matches'])} stars, "
                              f"{n_pts} points, σ {sigma:.2f} px\nΔ loss / (σ² · points per star)",
                              color=INK2, fontsize=8.5)
        print(f"{seq}: {r['status']} {len(r['matches'])} stars, {n_pts} pts, robust sigma {sigma:.2f}px, "
              f"pts/star {m:.0f}; 1-sigma half-widths "
              + ", ".join(f"{NAMES[i].split(' ')[0]} {'>' + format(4 * gn[i], '.2g') if w is None else format(w, '.2g')}"
                          for i, w in enumerate(widths)))
    h = [plt.Line2D([], [], color=SLICE, lw=1.5, label="slice: others held at optimum"),
         plt.Line2D([], [], color=PROFILE, lw=2, label="profile: others re-fit at each step"),
         plt.Line2D([], [], color=AXIS, lw=1, label="≈ 1σ level")]
    fig.text(0.005, 0.992, "Star-track fit: soft-L1 loss (f = 4 px) around the optimum, per parameter",
             ha="left", va="top", color=INK, fontsize=12)
    fig.legend(handles=h, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.005, 0.972),
               labelcolor=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(dest, dpi=105)
    print(dest)


if __name__ == "__main__":
    seqs = sys.argv[1:] or ["20191030_CopperCanyon_om-s-mobo-c", "20191030_CopperCanyon_om-s-mobo-m",
                            "20241021_PalomarRidge_hp-s-mobo-c", "20200727_Border11Fire_lp-s-mobo-m"]
    render(seqs, SKY / "star_loss_profiles.png")
