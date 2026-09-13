"""Showcase figures for the README: star-solved pose corrections and the sea-horizon check.

Composes renders from fig_track_solve and fig_sea_horizon (regenerated if missing; that
needs the archives) into captioned figures under docs/figures/. The burned-in banners are
cropped off and replaced with matplotlib titles and a legend, so the text is legible at
README width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import fig_sea_horizon, fig_track_solve

ROOT = Path(__file__).resolve().parents[3]
SKY = ROOT / "out" / "sky"
DOCS = ROOT / "docs" / "figures"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BANNER_PX = 47          # the 90 px render banner at 1600/3072 scale

# Render colors (cv2 draws BGR; these are the RGB they come out as).
TRACK, FITTED, PUBLISHED, SHIFT = "#3cdc3c", "#ff3cff", "#ff8c00", "#ffff00"


def _load(path: Path, render, seq: str, crop_banner: int, rows: tuple[float, float] = (0.0, 1.0)):
    if not path.exists():
        render(seq)
    im = np.asarray(Image.open(path).convert("RGB"))[crop_banner:]
    h = im.shape[0]
    return im[int(rows[0] * h): int(rows[1] * h)]


def _style(fig):
    fig.patch.set_facecolor(SURFACE)
    plt.rcParams.update({"font.family": "sans-serif"})


def pose_corrections(dest: Path = DOCS / "star_pose_correction.jpg") -> Path:
    panels = [
        ("hpwren_20260911_Q1_mlo-s-mobo-c", (0.0, 0.88),
         "mlo-s-mobo-c, 2026-09-11: Grus, Phoenix and Cetus put the camera 23.0\u00b0 east of its published south azimuth"),
        ("hpwren_20260911_Q1_vo-n-mobo-c", (0.0, 0.72),
         "vo-n-mobo-c, 2026-09-11: Draco says 11.5\u00b0 east of north; July 14 agrees to 0.1\u00b0, and it is one of the JunctionFire cameras"),
        ("20240721_EagleFire_stgo-n-mobo-c", (0.0, 0.85),
         "stgo-n-mobo-c, 2024-07-21: the Big Dipper and Draco put it 9.3\u00b0 west of its published azimuth"),
    ]
    imgs = [_load(SKY / f"star_solve_{seq}.jpg", fig_track_solve.render, seq, BANNER_PX, rows) for seq, rows, _ in panels]
    heights = [im.shape[0] / im.shape[1] for im in imgs]
    fig, axes = plt.subplots(len(imgs), 1, figsize=(12, 12 * sum(heights) + 1.6),
                             gridspec_kw={"height_ratios": heights})
    _style(fig)
    for ax, im, (_, _, title) in zip(axes, imgs, panels):
        ax.imshow(im); ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
    handles = [plt.Line2D([], [], color=TRACK, lw=6, label="star track observed across the sequence"),
               plt.Line2D([], [], color=FITTED, lw=2, label="catalog star under the star-solved pose"),
               plt.Line2D([], [], color=PUBLISHED, lw=2, label="same star under the published pose"),
               plt.Line2D([], [], color=SHIFT, lw=2, marker=">", markersize=6, label="published → solved")]
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=10, labelcolor=INK2,
               bbox_to_anchor=(0.01, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(dest, dpi=110, pil_kwargs={"quality": 85})
    plt.close(fig)
    return dest


def sea_horizon(dest: Path = DOCS / "sea_horizon_check.jpg") -> Path:
    seq = "20210107_Miguelfire_om-w-mobo-c"
    im = _load(SKY / f"sea_horizon_{seq}.jpg", fig_sea_horizon.render, seq, 43)
    h, w = im.shape[:2]
    im = im[int(0.35 * h): int(0.80 * h), int(0.35 * w):]            # the right side, where the two lines part
    fig, ax = plt.subplots(figsize=(12, 12 * im.shape[0] / im.shape[1] + 1.2))
    _style(fig)
    ax.imshow(im); ax.set_axis_off()
    ax.set_title("om-w-mobo-c, 2021-01-07: the ocean horizon's dip follows from camera height alone, and it lies along "
                 "the star-solved tilt,\nnot the published level one — an independent check on pitch and roll "
                 "that shares nothing with the star fit", loc="left", fontsize=10.5, color=INK, pad=6)
    handles = [plt.Line2D([], [], color=FITTED, lw=3, label="sea horizon under the star-solved pose (32 days earlier)"),
               plt.Line2D([], [], color=PUBLISHED, lw=3, label="sea horizon under the published pose")]
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=10, labelcolor=INK2,
               bbox_to_anchor=(0.01, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(dest, dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)
    return dest


if __name__ == "__main__":
    which = sys.argv[1:] or ["pose", "sea"]
    if "pose" in which:
        print(pose_corrections())
    if "sea" in which:
        print(sea_horizon())
