"""Draw the terrain-predicted skyline over the frame a camera actually recorded.

Everything in the overlay comes from published metadata and a public elevation model --
no pixels are consulted. So wherever the drawn line departs from the visible ridge, the
metadata is wrong, or the lens is not the rectilinear ideal the projection assumes.
Named peaks are marked with their distance, which doubles as a check that the azimuth
is right: a peak drawn on the wrong summit is a bearing error of exactly that offset.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .terrain import Dem, horizon, project, vfov_deg

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"


def observed_skyline(img: np.ndarray) -> np.ndarray:
    """Per-column row of the sky/terrain boundary, found from the image alone.

    Taking the first strong vertical gradient down each column finds cloud edges, not
    the ridge, and clouds sit above the skyline -- so that error is large and one-sided.
    Segment sky instead: at these wavelengths sky is blue-dominant and bright while
    terrain is neither, so the *lowest* sky pixel in a column is the silhouette. The
    vignetting on these lenses darkens the corners, hence the brightness floor.
    """
    b, g, r = (c.astype(np.int16) for c in cv2.split(cv2.GaussianBlur(img, (0, 0), 2)))
    v = img.max(axis=2)
    sky = ((b - r) > 8) & (v > 90)

    H, W = sky.shape
    rows = np.full(W, np.nan)
    for x in range(W):
        col = np.where(sky[:, x])[0]
        if col.size < H * 0.02:
            continue
        # Lowest sky pixel, but ignore isolated sky showing through gaps in the terrain
        # by requiring a run of sky immediately above it.
        for yy in col[::-1]:
            if yy >= 8 and sky[yy - 8:yy, x].mean() > 0.8:
                rows[x] = yy
                break
    return rows


def render(camera: str, img: np.ndarray, pitch_deg: float = 0.0,
           roll_deg: float = 0.0, show_observed: bool = True,
           dem: Dem | None = None) -> tuple[np.ndarray, dict]:
    cams = json.loads((META / "cams.json").read_text())
    cam = cams[camera]
    H, W = img.shape[:2]
    dem = dem or Dem()
    prof = horizon(cam, dem)
    x, y = project(cam, prof.az_deg, prof.elev_deg, W, H, pitch_deg, roll_deg)

    vis = img.copy()
    pts = [(int(px * W), int(py * H)) for px, py, ok in
           zip(x, y, (x > -0.2) & (x < 1.2) & (y > -0.5) & (y < 1.5)) if ok]
    for a, b in zip(pts, pts[1:]):
        cv2.line(vis, a, b, (0, 200, 255), 3)

    # Peaks: local maxima of the elevation profile, which is what the eye picks out too.
    e = prof.elev_deg
    stats = {"camera": camera, "n_peaks": 0}
    for i in range(2, len(e) - 2):
        if e[i] == max(e[i - 2:i + 3]) and e[i] > np.median(e) + 0.15:
            px, py = int(x[i] * W), int(y[i] * H)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(vis, (px, py), 9, (0, 0, 255), -1)
                cv2.circle(vis, (px, py), 9, (255, 255, 255), 2)
                cv2.putText(vis, f"{prof.range_km[i]:.0f}km", (px + 12, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                stats["n_peaks"] += 1

    resid = None
    if show_observed:
        obs = observed_skyline(img)
        for px in range(0, W, 6):
            if not np.isnan(obs[px]):
                cv2.circle(vis, (px, int(obs[px])), 2, (255, 120, 0), -1)
        xs = np.clip((x * W).astype(int), 0, W - 1)
        pred = np.full(W, np.nan)
        order = np.argsort(xs)
        pred[xs[order]] = (y * H)[order]
        m = ~np.isnan(obs) & ~np.isnan(pred)
        if m.sum() > 50:
            resid = (pred - obs)[m]
            stats.update(
                n_cols=int(m.sum()),
                resid_median_px=float(np.median(resid)),
                resid_iqr_px=float(np.percentile(resid, 75) - np.percentile(resid, 25)),
                # Distortion signature: a lens that is not rectilinear leaves a residual
                # that is symmetric about frame centre and grows toward the edges, so a
                # quadratic in (x-0.5) captures it where a constant offset cannot.
                quad_fit=np.polyfit(((np.where(m)[0] / W) - 0.5), resid, 2).tolist(),
            )

    band = np.zeros((116, W, 3), np.uint8)
    cv2.putText(band, f"{camera}  az={cam['az']} fov={cam['fov']} "
                      f"elev={cam['elev']}m agl={cam.get('agl')}m  "
                      f"vfov={vfov_deg(cam, W, H):.1f}",
                (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(band, "orange = terrain-predicted skyline   red dots = peaks (range)   "
                      "blue = skyline found in image",
                (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
    if resid is not None:
        cv2.putText(band, f"median residual {stats['resid_median_px']:+.0f} px   "
                          f"IQR {stats['resid_iqr_px']:.0f} px",
                    (int(W * 0.62), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return np.vstack([band, vis]), stats
