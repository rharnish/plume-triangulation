"""Bearings, visibility, and a likelihood field over the ground.

Two cameras that see the same plume give two directions, and where those directions
cross is the fire. That is the whole idea, but "where they cross" is the wrong way to
compute it: bearings carry angular error, and angular error fans out with range, so
two rays crossing at 60 km pin the fire far more loosely than two crossing at 5 km.
Intersecting lines throws that away and reports a point with false confidence.

So instead of intersecting, score. Every ground cell gets a log-likelihood from each
camera -- how far, in units of that camera's angular error, the cell sits from the
measured bearing -- and the sum over cameras is the posterior. One camera yields a
fan, two an ellipse, near-parallel views an honestly elongated smear.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

EARTH_R_KM = 6371.0


def load_cams() -> dict:
    """Camera table, overridable so a refined pose can be scored through this pipeline.

    Set FIGLIB_CAMS to `data/meta/cams_refined.json` to run everything downstream against
    terrain-fitted azimuths instead of the published ones.
    """
    import os
    p = os.environ.get("FIGLIB_CAMS")
    root = Path(__file__).resolve().parents[2]
    return json.loads(Path(p if p else root / "data" / "meta" / "cams.json").read_text())


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from N."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def angdiff_deg(a: float, b: float) -> float:
    """Signed smallest difference a - b, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def in_view(cam: dict, lat: float, lon: float, margin_deg: float = 0.0) -> bool:
    """Could this camera have seen a fire at (lat, lon)?

    A fixed HPWREN camera points at a published azimuth with a published horizontal
    field of view, so anything outside that wedge is invisible to it -- no detector
    required. That makes this a hard geometric filter on candidate ground truth.
    """
    b = bearing_deg(cam["lat"], cam["lon"], lat, lon)
    half = cam["fov"] / 2.0 + margin_deg
    return abs(angdiff_deg(b, cam["az"])) <= half


def undistort_x(cam: dict, x_frac: float, y_frac: float = 0.5) -> float:
    """Remove the fitted radial distortion, mapping observed x back to ideal x.

    `calibrate.py` fits the forward map (ideal -> observed), so recovering a bearing from
    a detection needs its inverse. There is no closed form for the one-parameter radial
    model, but the fixed-point iteration converges in a handful of steps at these
    magnitudes. Cameras with no fitted `k1` pass straight through.
    """
    k1 = cam.get("k1")
    if not k1:
        return x_frac
    aspect = cam.get("aspect", 0.75)          # 1536/2048 for these units
    dxo, dyo = x_frac - 0.5, (y_frac - 0.5) * aspect
    dx, dy = dxo, dyo
    for _ in range(8):
        s_ = 1.0 + k1 * (dx * dx + dy * dy)
        dx, dy = dxo / s_, dyo / s_
    return 0.5 + dx


# Equidistant fisheye lens for the 90 deg Mobotix units, measured from star tracks on 26
# night sequences across 12 cameras (src/figlib/stars/solve.py; NOTES.md, 2026-09-13):
# pixel radius r = k * theta * (1 + k1 * theta^2) off axis, with k 0.886 of the nameplate
# scale that would put fov/2 at the frame edge. The frame really spans about +-55 deg, not
# +-45. Measured on 3072x2048 frames only, so only those frames get it. Opt in with
# FIGLIB_LENS=fisheye; the default stays rectilinear so every recorded result reproduces.
FISHEYE_K_RATIO = 0.886
FISHEYE_K1 = -0.078


def _fisheye(cam: dict) -> bool:
    """Only where the lens was measured: 90 deg units recording 3072 px wide frames.

    The older 2048x1536 units put a different sensor behind the lens and have no star
    solve (FIgLib holds no night sequence from one, and their dates are past the CDN's
    public window), so they stay rectilinear rather than inherit a scale nobody measured.
    Callers that know the frame width pass it as cam["frame_w"] (see frame_sizes.py);
    without it the lens is not applied.
    """
    import os
    return (os.environ.get("FIGLIB_LENS") == "fisheye" and cam.get("fov") == 90
            and cam.get("frame_w") == 3072)


def offset_bearing_deg(cam: dict, x_frac: float, y_frac: float = 0.5) -> float:
    """Bearing to a feature at horizontal position `x_frac` across the image.

    `x_frac` runs 0 (left edge) to 1 (right edge); 0.5 is the optical axis. Uses the
    rectilinear projection rather than assuming degrees scale linearly with pixels --
    at 90 deg FoV the linear approximation is off by several degrees at the edges,
    which at 20 km is a kilometer of error. With FIGLIB_LENS=fisheye, 90 deg cameras use
    the star-measured equidistant lens instead, read along the horizon row.
    """
    if _fisheye(cam):
        r = (x_frac - 0.5) * math.radians(cam["fov"]) / FISHEYE_K_RATIO
        t = r
        for _ in range(20):           # Newton on t * (1 + k1 t^2) = r
            t -= (t * (1 + FISHEYE_K1 * t * t) - r) / (1 + 3 * FISHEYE_K1 * t * t)
        return (cam["az"] + cam.get("yaw", 0.0) + math.degrees(t)) % 360.0
    half = math.radians(cam["fov"] / 2.0)
    x_frac = undistort_x(cam, x_frac, y_frac)
    # Image plane at unit focal length spans [-tan(half), +tan(half)]
    u = (x_frac - 0.5) * 2.0 * math.tan(half)
    return (cam["az"] + cam.get("yaw", 0.0) + math.degrees(math.atan(u))) % 360.0


def bearing_x_frac(cam: dict, lat: float, lon: float) -> float | None:
    """Inverse of `offset_bearing_deg`: where in the frame would (lat, lon) appear?

    Returns None if the point falls outside the field of view.
    """
    b = bearing_deg(cam["lat"], cam["lon"], lat, lon)
    d = math.radians(angdiff_deg(b, cam["az"] + cam.get("yaw", 0.0)))
    if _fisheye(cam):
        x = 0.5 + d * (1 + FISHEYE_K1 * d * d) * FISHEYE_K_RATIO / math.radians(cam["fov"])
        return x if 0.0 < x < 1.0 else None
    half = math.radians(cam["fov"] / 2.0)
    if abs(d) >= half:
        return None
    return 0.5 + math.tan(d) / (2.0 * math.tan(half))


def loglik_field(cams_bearings: list[tuple[dict, float, float]],
                 lat0: float, lon0: float,
                 half_extent_km: float = 70.0, step_km: float = 0.5):
    """Log-likelihood of ignition over a local ground grid.

    `cams_bearings` is (camera, measured bearing deg, sigma deg) per view. Returns
    (lats, lons, loglik) with loglik[i][j] summed over cameras. Sigma should absorb
    pose error, plume width, and detector localisation -- it is what stops the result
    claiming more precision than the geometry supports.
    """
    dlat = step_km / 111.32
    dlon = step_km / (111.32 * math.cos(math.radians(lat0)))
    n = int(half_extent_km / step_km)
    lats = [lat0 + i * dlat for i in range(-n, n + 1)]
    lons = [lon0 + j * dlon for j in range(-n, n + 1)]

    field = []
    for la in lats:
        row = []
        for lo in lons:
            total = 0.0
            for cam, meas, sigma in cams_bearings:
                b = bearing_deg(cam["lat"], cam["lon"], la, lo)
                total += -0.5 * (angdiff_deg(b, meas) / sigma) ** 2
            row.append(total)
        field.append(row)
    return lats, lons, field


def field_argmax(lats, lons, field) -> tuple[float, float, float]:
    best = (-math.inf, 0, 0)
    for i, row in enumerate(field):
        for j, v in enumerate(row):
            if v > best[0]:
                best = (v, i, j)
    return lats[best[1]], lons[best[2]], best[0]
