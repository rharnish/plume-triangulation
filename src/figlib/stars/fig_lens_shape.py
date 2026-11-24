"""Is a lens difference visible in the *shape* of the trails, before anything is fitted?

Two figures, both aimed at that question and both drawn with square pixels -- every other
figure in this package stretches the frame to fit a caption, which is fine for showing a
match and useless for judging curvature.

`compare()` puts two cameras side by side, full field of view on top and the track band
underneath. The pair worth looking at is bm-s-mobo-c against mg-s-mobo-c: the same night, the
same hour, latitudes 33.160 and 33.189, so the sky is the same to a fraction of a degree and
the only thing left to explain a difference in the drawing is the optics. bm-s measures
0.779x nameplate and mg-s 0.884x.

`hypotheses()` asks the same question the other way round, on one camera: draw the matched
stars under the measured lens, and again under the shared 0.886x lens refitted as well as it
can be on the same star-track pairs. The second is what a wrong lens looks like when it is
trying its hardest -- it cannot be right at the centre and the edge at once, so it splays.

    python -m src.figlib.stars.fig_lens_shape
    python -m src.figlib.stars.fig_lens_shape --a <seq> --b <seq>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from star_calibration import catalog as SG
from star_calibration.hpwren import nights
from . import solve as S
from star_calibration.fisheye import initial_k, project_fisheye

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "sky"
DEFAULT_A = "hpwren_20260714_Q1_bm-s-mobo-c"
DEFAULT_B = "hpwren_20260714_Q1_mg-s-mobo-c"


def _load(seq: str):
    """Frame, tracks and the solved row for one sequence."""
    raw, (W, H) = S.load_tracks(seq)
    trk = [raw[i] for i in S.prune(raw)]
    row = next(r for r in json.loads((S.DATA / "solve_wide_summary.json").read_text())
               if r["seq"] == seq)
    offs = sorted({o for t in trk for o in t})
    ref = max(offs, key=lambda o: sum(o in t for t in trk))
    frames = nights.read_frames(seq)
    blob = min(frames, key=lambda f: abs(f[1] - ref))[2]
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=2.4, beta=10), cv2.COLOR_BGR2RGB)
    return {"seq": seq, "raw": raw, "trk": trk, "W": W, "H": H, "img": img, "row": row,
            "ref": ref, "offs": offs}


def _draw_tracks(ax, d, crop=False, lw=1.5, color="#38d6ff"):
    W, H = d["W"], d["H"]
    ax.imshow(d["img"], extent=(0, W, H, 0))
    for t in d["trk"]:
        p = np.array([t[o][:2] for o in sorted(t)])
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw)
    y_trk = max(v[1] for t in d["trk"] for v in t.values())
    ax.set_xlim(0, W)
    ax.set_ylim(min(H, y_trk * 1.35 + 60) if crop else H, 0)
    ax.set_aspect("equal")            # square pixels: the whole point of this figure
    ax.set_xticks([]); ax.set_yticks([])


def _angle_rings(ax, d, degs=(30, 50), color="#ffd23f"):
    """Circles of constant angle from the boresight, under this camera's measured lens.

    The same sky angle sits at a different pixel radius on the two cameras, which is the lens
    difference stated as a distance one can hold a ruler against.
    """
    p = d["row"]["pose"]
    k0 = initial_k(S.CAMS[d["row"]["camera"]], d["W"])
    cx, cy = d["W"] / 2, d["H"] / 2
    for deg in degs:
        th = np.radians(deg)
        r = p["k_ratio"] * k0 * th * (1 + p["k1"] * th ** 2)
        a = np.linspace(0, 2 * np.pi, 400)
        ax.plot(cx + r * np.cos(a), cy + r * np.sin(a), color=color, lw=1.2, ls="--", alpha=0.8)
        lx, ly = cx - r * np.cos(np.pi / 4), cy + r * np.sin(np.pi / 4)   # lower left of the ring
        ax.annotate(f"{deg}° from boresight = {r:.0f} px", (lx, ly), color=color, fontsize=9,
                    ha="left", va="top", xytext=(6, 6), textcoords="offset points")


def compare(seq_a: str = DEFAULT_A, seq_b: str = DEFAULT_B) -> Path:
    A, B = _load(seq_a), _load(seq_b)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.4), height_ratios=[2.5, 1])
    for col, d in enumerate((A, B)):
        p = d["row"]["pose"]
        cam = S.CAMS[d["row"]["camera"]]
        k0 = initial_k(cam, d["W"])
        th = np.linspace(0, 1.5, 40000)
        rr = p["k_ratio"] * k0 * th * (1 + p["k1"] * th ** 2)
        half = float(np.degrees(np.interp(d["W"] / 2, rr, th)))
        _draw_tracks(axes[0][col], d, crop=False)
        _angle_rings(axes[0][col], d)
        axes[0][col].set_title(
            f"{d['row']['camera']}   lens {p['k_ratio']:.4f}× nameplate  (half-field ±{half:.1f}°)\n"
            f"boresight {(cam['az'] + p['d_az']) % 360:.1f}°  ·  {len(d['trk'])} tracks  ·  "
            "full frame, square pixels", fontsize=11)
        _draw_tracks(axes[1][col], d, crop=True)
        axes[1][col].set_title("the track band", fontsize=10)
    fig.suptitle("Same night (2026-07-14), latitudes 33.16 and 33.19, different lens — "
                 "trail shape only, nothing fitted", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dest = OUT / f"lens_shape_{A['row']['camera']}_vs_{B['row']['camera']}.jpg"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest


def _refit_fixed_lens(d, k_ratio: float):
    """Best pose for this sequence's matched stars with the lens held at `k_ratio`."""
    cam = S.CAMS[d["row"]["camera"]]
    W, H, seq = d["W"], d["H"], S.SEQS[d["seq"]]
    k0 = initial_k(cam, W)
    lens = (k_ratio * k0, S.K1)
    obs, al, az = [], [], []
    for n, j in d["row"]["matches"].items():
        t = d["raw"][j]
        o = sorted(t)
        obs.append(np.array([t[q][:2] for q in o]))
        a, z = SG.altaz(*SG.STARS[n], seq["t0"] + np.asarray(o, float), cam["lat"], cam["lon"])
        al.append(a); az.append(z)
    obs = np.concatenate(obs); al = np.concatenate(al); az = np.concatenate(az)

    def resid(q):
        x, y = project_fisheye(cam, az, al, W, H, *q, *lens)
        return np.r_[x * W - obs[:, 0], y * H - obs[:, 1]]

    p = d["row"]["pose"]
    sol = least_squares(resid, [p["d_az"], p["d_pitch"], p["d_roll"]], loss="soft_l1",
                        f_scale=4.0, max_nfev=3000)
    err = np.hypot(*sol.fun.reshape(2, -1))
    return tuple(sol.x), lens, float(np.median(err))


def hypotheses(seq: str = DEFAULT_A, rival: float = S.K_RATIO) -> Path:
    """One camera, its stars drawn under the measured lens and under the network-standard one."""
    d = _load(seq)
    cam = S.CAMS[d["row"]["camera"]]
    W, H, seq_meta = d["W"], d["H"], S.SEQS[seq]
    k0 = initial_k(cam, W)
    p = d["row"]["pose"]
    fitted = ((p["d_az"], p["d_pitch"], p["d_roll"]), (p["k_ratio"] * k0, p["k1"]),
              d["row"]["median_px"])
    rival_fit = _refit_fixed_lens(d, rival)

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 3.9))
    for ax, (pose, lens, med), label, kr in (
            (axes[0], fitted, "measured lens", p["k_ratio"]),
            (axes[1], rival_fit, "network-standard lens, pose refitted", rival)):
        _draw_tracks(ax, d, crop=True, lw=4.5, color="#3ddc6b")
        for n, j in d["row"]["matches"].items():
            o = sorted(d["raw"][j])
            alt, az = SG.altaz(*SG.STARS[n], seq_meta["t0"] + np.asarray(o, float),
                               cam["lat"], cam["lon"])
            x, y = project_fisheye(cam, az, alt, W, H, *pose, *lens)
            ax.plot(x * W, y * H, color="#ff3ec8", lw=1.8, zorder=4)
        ax.set_title(f"{label}: {kr:.4f}× nameplate — median {med:.2f} px", fontsize=11.5)
    fig.suptitle(f"{d['row']['camera']}, {seq.split('_')[1]} — green: observed tracks, "
                 "magenta: the same stars predicted", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    dest = OUT / f"lens_hypotheses_{d['row']['camera']}.jpg"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    args = sys.argv[1:]
    a = args[args.index("--a") + 1] if "--a" in args else DEFAULT_A
    b = args[args.index("--b") + 1] if "--b" in args else DEFAULT_B
    print(compare(a, b))
    print(hypotheses(a))
