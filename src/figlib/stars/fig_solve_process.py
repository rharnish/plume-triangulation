"""How a night of trails becomes a camera pose, in four panels over one real frame.

Nothing here is drawn schematically: every panel is the actual intermediate the solver
computes, on the sequence named on the command line, and the score in panel 3 comes from
`solve.make_coincidence` -- the same scorer the search optimises, not a redrawing of it.

  1. What the night gives you. Linked moving tracks over the frame. No pose, no lens, no
     star named; just points that moved together.
  2. The pole, in closed form. Each track's velocity, mapped into the camera's own frame
     through the lens alone, must satisfy d_dot = omega*(p x d). That is linear in p, so one
     least-squares solve puts the celestial pole on the image -- the cross -- with no search.
     |p| comes out of the same fit and is printed: it is 1 only when the lens is right.
  3. The one angle the pole cannot see. Turning the sky about the polar axis leaves every
     velocity unchanged, so the pole leaves a 1-D family of poses. This is that family,
     scored by how many catalog stars land on a track. A 1-D scan at 0.1 deg replaces a 3-D
     grid at 1 deg that could only ever reach +-30 deg from the published pose.
  4. The solve. Matched tracks in green, each matched star's arc under the fitted pose in
     magenta running inside it, and the published pose in orange for contrast.

    python -m src.figlib.stars.fig_solve_process [seq] [--k-ratio 0.78]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import catalog as SG
from . import pole as POLE
from . import solve as S
from .fisheye import initial_k, project_fisheye

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "sky"
DEFAULT_SEQ = "hpwren_20260911_Q1_wc-n-mobo-c"


def _frame(seq: str, ref: int) -> np.ndarray:
    if seq.startswith("hpwren_"):
        from . import nights
        frames = nights.read_frames(seq)
    else:
        from .. import corpus as C
        from ..detect_yolo import read_frames
        arch = {p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}[seq.split("#")[0]]
        frames = read_frames(arch)
    _, _, blob = min(frames, key=lambda f: abs(f[1] - ref))
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(cv2.convertScaleAbs(img, alpha=2.2, beta=8), cv2.COLOR_BGR2RGB)


def _star_arcs(ctx, pose, spans, W, H):
    """Each named star's path under one pose, over *its own matched track's* offsets.

    Spanning the whole night instead would draw a magenta arc far longer than the green track
    beside it, which reads as a mismatch when it is only a longer time base.
    """
    c, lens = ctx["cam"], ctx["lens"]
    out = {}
    for n, offs in spans.items():
        ep = ctx["t0"] + np.asarray(offs, float)
        alt, az = SG.altaz(*SG.STARS[n], ep, c["lat"], c["lon"])
        x, y = project_fisheye(c, az, alt, W, H, *pose, *lens)
        out[n] = (x * W, y * H)
    return out


def main(seq: str = DEFAULT_SEQ, k_ratio: float | None = None) -> Path:
    res0 = next(r for r in json.loads((S.DATA / "solve_wide_summary.json").read_text())
                if r["seq"] == seq)
    if k_ratio is None and res0.get("lens_from_pole") and res0.get("pose"):
        # Draw the scan for the lens hypothesis that actually won. For bm-s-mobo-c the
        # measured 0.80x is the whole reason it solves, and scanning under the shared lens
        # would show a flat, featureless trace that is not how the answer was found.
        kf, kp = res0["pose"]["k_ratio"], res0["lens_from_pole"]
        if abs(kf - kp) < abs(kf - S.K_RATIO):
            k_ratio = kp
    ctx = S.scan_context(seq, k_ratio=k_ratio)
    W, H, fit = ctx["W"], ctx["H"], ctx["fit"]
    res = res0
    pose = (res["pose"]["d_az"], res["pose"]["d_pitch"], res["pose"]["d_roll"])
    img = _frame(seq, ctx["ref"])
    # These cameras look at the horizon and the trails live in the top third, so a full frame
    # spends half its area on dark foreground. Crop to the band the tracks occupy plus enough
    # ground to keep the horizon in shot, which is what makes it read as a real camera.
    y_trk = max(v[1] for t in ctx["tracks"] for v in t.values())
    y_cut = int(min(H, max(y_trk * 1.75, y_trk + 260)))

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.2))
    (a1, a2), (a3, a4) = axes
    for a in (a1, a2, a4):
        a.imshow(img, extent=(0, W, H, 0))
        a.set_xlim(0, W); a.set_ylim(y_cut, 0); a.set_xticks([]); a.set_yticks([])
        a.set_aspect("auto")

    # 1 -- the tracks
    for t in ctx["tracks"]:
        p = np.array([t[o][:2] for o in sorted(t)])
        a1.plot(p[:, 0], p[:, 1], color="#38d6ff", lw=1.6)
    a1.set_title(f"1.  {len(ctx['tracks'])} moving tracks\n"
                 "no pose, no lens, no star named", fontsize=11)

    # 2 -- the velocity field and the closed-form pole
    k, k1 = ctx["lens"]
    px, py = project_fisheye(ctx["cam"], [0.0], [ctx["cam"]["lat"]], W, H, *pose, k, k1)
    PX, PY = px[0] * W, py[0] * H
    for t in ctx["tracks"]:
        o = sorted(t)
        p = np.array([t[q][:2] for q in o])
        i = len(p) // 2
        v = p[-1] - p[0]
        n = np.linalg.norm(v)
        if n > 1e-6:
            v = v / n * 105
            a2.arrow(p[i, 0], p[i, 1], v[0], v[1], color="#ffd23f", width=3.2,
                     head_width=26, length_includes_head=True, zorder=3)
    a2.plot([PX], [PY], marker="+", ms=30, mew=4, color="#ff2e63", zorder=5)
    a2.annotate("celestial pole\n(closed form)", (PX, PY), xytext=(PX + 150, PY + 330),
                color="#ff2e63", fontsize=10.5, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color="#ff2e63", lw=1.4))
    a2.set_title("2.  $\\dot{d} = \\omega\\,(p \\times d)$ is linear in $p$\n"
                 f"one least-squares solve, no search   "
                 f"$|p|$ = {fit['norm']:.3f}, {fit['inlier_frac']:.0%} inliers", fontsize=11)

    # 3 -- the psi scan
    psi, sc = ctx["psi"], ctx["scores"]
    a3.plot(psi, sc, color="#2b7bba", lw=1.1)
    best = int(np.argmax(sc))
    a3.plot([psi[best]], [sc[best]], "o", color="#ff2e63", ms=9, zorder=5)
    a3.annotate(f"$\\psi$ = {psi[best]:.1f}$\\degree$\n"
                f"d_az {ctx['poses'][best][0]:+.2f}, d_pitch {ctx['poses'][best][1]:+.2f}, "
                f"d_roll {ctx['poses'][best][2]:+.2f}",
                (psi[best], sc[best]),
                xytext=(0.58 if psi[best] < 180 else 0.04, 0.94),
                textcoords="axes fraction", color="#ff2e63", fontsize=10,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#ff2e63", lw=1.3))
    a3.set_xlabel("$\\psi$ — turn about the polar axis (deg)")
    a3.set_ylabel("stars landing on a track  (coincidence score)")
    a3.set_xlim(0, 360)
    a3.set_ylim(0, float(sc.max()) * 1.32)   # headroom so the callout clears the peak
    a3.grid(alpha=0.3)
    far = sc[np.abs((psi - psi[best] + 180) % 360 - 180) > 3]
    runner = float(far.max()) if len(far) else 0.0
    a3.axhline(runner, color="0.6", ls="--", lw=1.0)
    a3.text(358, runner, f"best rival {runner:.2f} ", ha="right", va="bottom",
            fontsize=8.5, color="0.45")
    a3.set_title("3.  the one angle a flow field cannot see\n"
                 f"1-D scan at 0.1$\\degree$ — peak {sc[best]:.2f} vs {runner:.2f} "
                 "for every rival angle", fontsize=11)

    # 4 -- the solve
    matched = res["matches"]
    raw, _ = S.load_tracks(seq)
    spans = {n: sorted(raw[j]) for n, j in matched.items()}
    arcs = _star_arcs(ctx, pose, spans, W, H)
    pub = _star_arcs(ctx, (0.0, 0.0, 0.0), spans, W, H)
    for n, j in matched.items():
        t = raw[j]
        p = np.array([t[o][:2] for o in sorted(t)])
        a4.plot(p[:, 0], p[:, 1], color="#3ddc6b", lw=5.5, alpha=0.9, zorder=2)
        x, y = pub[n]
        a4.plot(x, y, color="#ff9f1c", lw=1.3, alpha=0.9, zorder=3)
        x, y = arcs[n]
        a4.plot(x, y, color="#ff3ec8", lw=2.1, zorder=4)
    a4.set_title(f"4.  solved: {res['n_stars']} stars, median {res['median_px']:.2f} px\n"
                 "green = track, magenta = fitted pose, orange = published pose", fontsize=11)

    cam = S.SEQS[seq]["camera"]
    fig.suptitle(f"From trails to pose — {seq}  ({cam}, "
                 f"lens {res['pose']['k_ratio']:.3f}$\\times$ nameplate)", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    dest = OUT / f"solve_process_{seq}.jpg"
    fig.savefig(dest, dpi=105)
    plt.close(fig)
    return dest


if __name__ == "__main__":
    args = sys.argv[1:]
    kr = float(args[args.index("--k-ratio") + 1]) if "--k-ratio" in args else None
    pos = [a for a in args if not a.startswith("--") and a != str(kr)]
    print(main(pos[0] if pos else DEFAULT_SEQ, k_ratio=kr))
