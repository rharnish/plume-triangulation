"""Show the edge map itself, with the predicted ridges drawn over it.

The correlation tests said the layered structure does not stand out. This is the picture
behind that number: what the gradient filter actually returns, where the DEM says the
ridges are, and which rows a per-column peak finder would pick if asked. Three panels,
because the interesting question is not whether any of them looks plausible alone but
whether the second and third agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geom import load_cams
from .terrain import Dem, project, ridges
from .viz_terrain import _best_frame, _range_colour

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
OUT = ROOT / "out" / "edges"
PANEL_W = 1100


def vgrad(img: np.ndarray, sigma_x: float = 9.0) -> np.ndarray:
    """Vertical gradient, standardised per column.

    Smoothed along rows because ridges are near-horizontal: that lifts a long faint crest
    above pixel noise while leaving masts and poles unreinforced. Standardising per column
    matters because a hazy distant crest is faint in absolute terms but is still a local
    maximum within its own column, and only the latter is evidence.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma_x, sigmaY=1.2)
    G = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5))
    return (G - G.mean(axis=0, keepdims=True)) / (G.std(axis=0, keepdims=True) + 1e-6)


def candidates(E: np.ndarray, per_col: int = 6, min_sep: int = 18) -> list[np.ndarray]:
    """Per column, the strongest few gradient peaks -- what a detector would propose."""
    H, W = E.shape
    out = []
    for c in range(W):
        col = E[:, c]
        loc = np.flatnonzero((col[1:-1] >= col[:-2]) & (col[1:-1] > col[2:])) + 1
        loc = loc[np.argsort(-col[loc])]
        keep: list[int] = []
        for r in loc:
            if all(abs(r - k) >= min_sep for k in keep):
                keep.append(int(r))
            if len(keep) >= per_col:
                break
        out.append(np.array(sorted(keep)))
    return out


def _heat(E: np.ndarray) -> np.ndarray:
    v = np.clip((E + 1.0) / 6.0, 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def render(camera: str, img: np.ndarray, dem: Dem) -> np.ndarray:
    cam = load_cams()[camera]
    E = vgrad(img)
    H, W = E.shape
    f = ridges(cam, dem)
    x, y = project(cam, f.az_deg, f.elev_deg, W, H)

    heat = _heat(E)
    over = heat.copy()
    for cid in np.unique(f.layer):
        sel = np.flatnonzero(f.layer == cid)
        if sel.size < 8:
            continue
        sel = sel[np.argsort(f.az_deg[sel])]
        col = _range_colour(float(np.median(f.range_km[sel])))
        pts = [(int(x[i] * W), int(y[i] * H)) for i in sel]
        for a, b in zip(pts, pts[1:]):
            if abs(b[1] - a[1]) <= 0.025 * H:
                cv2.line(over, a, b, col, 2, cv2.LINE_AA)

    prop = heat.copy()
    cands = candidates(E)
    for c in range(0, W, 6):
        for r in cands[c]:
            cv2.circle(prop, (c, int(r)), 3, (120, 255, 120), -1)

    def lab(p, text):
        cv2.rectangle(p, (0, 0), (p.shape[1], 46), (18, 18, 18), -1)
        cv2.putText(p, text, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    (245, 245, 245), 2)
        return p

    panels = [lab(img.copy(), f"{camera}  frame"),
              lab(over, "vertical gradient + DEM-predicted ridges (colour = range)"),
              lab(prop, "vertical gradient + strongest 6 peaks per column")]
    panels = [cv2.resize(p, (PANEL_W, int(PANEL_W * p.shape[0] / p.shape[1])))
              for p in panels]
    return np.vstack(panels)


def main(argv: list[str]) -> None:
    seqs = json.loads((META / "sequences.json").read_text())
    by_cam: dict[str, dict] = {}
    for s in seqs:
        if s["has_pose"] and s["camera"] not in by_cam:
            by_cam[s["camera"]] = s
    names = [k for k in by_cam if any(a in k for a in argv)] if argv else list(by_cam)
    OUT.mkdir(parents=True, exist_ok=True)
    dem = Dem()
    for k, name in enumerate(sorted(names), 1):
        img = _best_frame(by_cam[name])
        if img is None:
            print(f"[{k}/{len(names)}] {name}: no frame")
            continue
        cv2.imwrite(str(OUT / f"{name}.png"), render(name, img, dem))
        print(f"[{k}/{len(names)}] {name} -> {OUT.name}/{name}.png", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
