"""Every star-solved fisheye lens in one scatter: focal scale against the radial term.

Each solve fits an equidistant lens with one radial term, r = k*theta*(1 + k1*theta^2), with
k given as a ratio to the nameplate focal length (`k_ratio`). One dot per ledger entry. The +
marks the shared lens that `geom` applies to every 3072 px 90 deg unit when it turns a pixel
into a bearing (FISHEYE_K_RATIO, FISHEYE_K1).

What it shows, on the recent ledger of 2026-09-25 (132 solves, 81 cameras):
* two populations. Big Black Mountain's bm-e and bm-w solve at k 0.775; everything else sits
  at 0.877-0.894. That is a different lens, not scatter.
* within the main population k and k1 trade off along one line (r = -0.89). The solver pins
  the mapping somewhere around 40-45 deg off axis, not each term, so the image radius at 45 deg
  varies by only ~0.2% across cameras (0.4% at worst) -- about 0.2 deg of bearing at the frame
  edge. The shared lens is good for all of them; it is ~3 deg wrong for bm-*.

The right panel zooms on the main population. With a second ledger given, entries it also
holds are drawn in one color and the rest in another (by default: the tracked ledger inside
the recent one).

    python -m src.figlib.stars.fig_intrinsics                      # recent ledger vs tracked
    python -m src.figlib.stars.fig_intrinsics <ledger.json> [--base <ledger.json>] [--out <png>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..geom import FISHEYE_K1, FISHEYE_K_RATIO

ROOT = Path(__file__).resolve().parents[3]
TRACKED = ROOT / "data" / "meta" / "pose_ledger.json"
RECENT = ROOT / "out" / "recent" / "pose_ledger.json"
OUT = ROOT / "out" / "sky" / "fisheye_intrinsics.png"
SPLIT_K = 0.82          # below this, a lens is its own population (bm-* at 0.775)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASE_C, REST_C = "#2a78d6", "#eb6834"


def spread_at(entries: list[dict], deg=np.arange(5, 61, 5)) -> dict[int, float]:
    """Across these lenses, the spread (std / mean, %) of image radius at each off-axis angle."""
    k = np.array([e["k_ratio"] for e in entries])[:, None]
    k1 = np.array([e["k1"] for e in entries])[:, None]
    th = np.radians(deg)
    r = k * th * (1 + k1 * th ** 2)
    return {int(d): round(float(s), 3) for d, s in zip(deg, r.std(0) / r.mean(0) * 100)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0, labelsize=9, colors=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figure(ledger: Path = RECENT, base: Path | None = TRACKED, dest: Path = OUT) -> Path:
    rows = json.loads(Path(ledger).read_text())
    in_base = ({e["source"] for e in json.loads(Path(base).read_text())}
               if base and Path(base).resolve() != Path(ledger).resolve() else None)
    groups = ([([e for e in rows if e["source"] in in_base], BASE_C, Path(base).stem),
               ([e for e in rows if e["source"] not in in_base], REST_C, f"not in {Path(base).stem}")]
              if in_base is not None else [(rows, BASE_C, Path(ledger).stem)])
    main = [e for e in rows if e["k_ratio"] >= SPLIT_K]
    odd = [e for e in rows if e["k_ratio"] < SPLIT_K]
    n_cams = len({e["camera"] for e in rows})

    plt.rcParams.update({"font.family": "sans-serif"})
    if odd:
        fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    else:                      # one population: the zoom is the whole picture
        fig, b = plt.subplots(figsize=(8, 6))
        a = b
    fig.patch.set_facecolor(SURFACE)
    for ax in {a, b}:
        _style(ax)
        for sub, col, lab in groups:
            if sub:
                ax.scatter([e["k_ratio"] for e in sub], [e["k1"] for e in sub], s=46, color=col,
                           edgecolors=SURFACE, linewidths=1.5, zorder=3,
                           label=f"{lab} ({len(sub)} solves, {len({e['camera'] for e in sub})} cameras)")
        ax.scatter([FISHEYE_K_RATIO], [FISHEYE_K1], marker="+", s=260, color=INK, lw=2, zorder=4)
        ax.set_xlabel("k_ratio  (focal scale ÷ nameplate)", color=INK2, fontsize=10)
    a.set_ylabel("k1  (radial term: r = k·θ·(1 + k1·θ²))", color=INK2, fontsize=10)
    if odd:
        a.set_title("All solves", loc="left", color=INK, fontsize=11)
        names = sorted({e["camera"].replace("-mobo-c", "").replace("-mobo-m", " (mono)") for e in odd})
        kx, ky = np.mean([e["k_ratio"] for e in odd]), np.mean([e["k1"] for e in odd])
        a.annotate(f"{', '.join(names)}\nk {kx:.3f}", (kx, ky), (kx + 0.015, ky - 0.012),
                   color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        mk, mk1 = np.median([e["k_ratio"] for e in main]), np.median([e["k1"] for e in main])
        a.annotate(f"the other {len({e['camera'] for e in main})} cameras", (mk, mk1 - 0.004),
                   (mk - 0.065, mk1 - 0.010), color=INK2, fontsize=9,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    k = np.array([e["k_ratio"] for e in main])
    k1 = np.array([e["k1"] for e in main])
    pad_k, pad_k1 = 0.003, 0.004
    b.set_xlim(k.min() - pad_k, k.max() + pad_k)
    b.set_ylim(k1.min() - pad_k1, k1.max() + pad_k1)
    sp = spread_at(main)
    best = min(sp, key=sp.get)
    b.set_title("Main population  (+ = the shared lens every bearing uses)", loc="left",
                color=INK, fontsize=11)
    b.text(0.02, 0.03, f"k and k1 trade off along one line (r = {np.corrcoef(k, k1)[0, 1]:.2f}): "
           f"image radius varies\nleast at {best}° off axis ({sp[best]:.2f}%), at most "
           f"{max(sp.values()):.2f}% within 60°", transform=b.transAxes, color=INK2, fontsize=9)
    if len(groups) > 1:
        b.legend(frameon=False, loc="upper right", labelcolor=INK2, fontsize=9)
    fig.suptitle(f"Star-solved fisheye intrinsics: {len(rows)} solves, {n_cams} cameras "
                 f"({Path(ledger).relative_to(ROOT) if Path(ledger).is_relative_to(ROOT) else ledger})",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return Path(dest)


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = lambda f, d: Path(args[args.index(f) + 1]) if f in args else d
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    ledger = Path(pos[0]) if pos else (RECENT if RECENT.exists() else TRACKED)
    print(figure(ledger, opt("--base", TRACKED), opt("--out", OUT)))
