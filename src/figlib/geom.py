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


# WGS84
A_WGS = 6378137.0
F_WGS = 1 / 298.257223563
E2_WGS = F_WGS * (2 - F_WGS)


def ecef(lat: float, lon: float, h: float = 0.0) -> tuple[float, float, float]:
    """WGS84 geodetic -> Earth-centred, Earth-fixed metres."""
    la, lo = math.radians(lat), math.radians(lon)
    n = A_WGS / math.sqrt(1 - E2_WGS * math.sin(la) ** 2)
    return ((n + h) * math.cos(la) * math.cos(lo), (n + h) * math.cos(la) * math.sin(lo),
            (n * (1 - E2_WGS) + h) * math.sin(la))


def enu(lat1: float, lon1: float, h1: float, lat2: float, lon2: float, h2: float):
    """Point 2 in point 1's local east-north-up frame, metres. Exact on the ellipsoid."""
    x1, y1, z1 = ecef(lat1, lon1, h1)
    x2, y2, z2 = ecef(lat2, lon2, h2)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    la, lo = math.radians(lat1), math.radians(lon1)
    sla, cla, slo, clo = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    e = -slo * dx + clo * dy
    n = -sla * clo * dx - sla * slo * dy + cla * dz
    u = cla * clo * dx + cla * slo * dy + sla * dz
    return e, n, u


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing from point 1 to point 2 as a camera at point 1 sees it: degrees clockwise from
    north in point 1's local horizontal frame, on the WGS84 ellipsoid.

    This used to be the spherical great-circle formula evaluated at geodetic latitudes. The
    sphere treats a degree of latitude and a degree of longitude as the same length, and the
    ellipsoid does not -- the ratio is ~1 - e^2 cos^2(lat) -- so on a diagonal bearing the
    sphere was off by up to 0.135 deg at 40-60 km here (NOTES.md, 2026-09-25). Over those
    ranges the local-frame direction and the geodesic's initial azimuth agree to < 0.001 deg.
    """
    e, n, _ = enu(lat1, lon1, 0.0, lat2, lon2, 0.0)
    return math.degrees(math.atan2(e, n)) % 360.0


def enu_grid(lat1: float, lon1: float, lats, lons):
    """(east, north) metres from one camera to an array of ground points, exact on WGS84."""
    import numpy as np
    la2, lo2 = np.radians(np.asarray(lats, float)), np.radians(np.asarray(lons, float))
    n2 = A_WGS / np.sqrt(1 - E2_WGS * np.sin(la2) ** 2)
    x1, y1, z1 = ecef(lat1, lon1, 0.0)
    dx = n2 * np.cos(la2) * np.cos(lo2) - x1
    dy = n2 * np.cos(la2) * np.sin(lo2) - y1
    dz = n2 * (1 - E2_WGS) * np.sin(la2) - z1
    la, lo = math.radians(lat1), math.radians(lon1)
    e = -math.sin(lo) * dx + math.cos(lo) * dy
    n = -math.sin(la) * math.cos(lo) * dx - math.sin(la) * math.sin(lo) * dy + math.cos(la) * dz
    return e, n


def bearing_grid(lat1: float, lon1: float, lats, lons):
    """`bearing_deg` from one camera to an array of ground points, vectorised."""
    import numpy as np
    e, n = enu_grid(lat1, lon1, lats, lons)
    return np.degrees(np.arctan2(e, n)) % 360.0


def ray_latlon(lat0: float, lon0: float, az_deg, d_m):
    """Ground points under a camera's straight line of sight: `d_m` metres out along local
    azimuth `az_deg`, as (lat, lon). Broadcasts `az_deg` against `d_m`.

    The point is placed in the camera's east-north plane and dropped onto the ellipsoid, so
    its `bearing_deg` from the camera is `az_deg` exactly. Stepping in latitude and longitude
    with fixed metres-per-degree instead drifts off the line of sight by up to 0.33 deg at
    80 km on diagonal azimuths (NOTES.md, 2026-09-25, "Terrain on the line of sight").
    """
    import numpy as np
    az, d = np.radians(np.asarray(az_deg, float)), np.asarray(d_m, float)
    e, n = d * np.sin(az), d * np.cos(az)
    la, lo = math.radians(lat0), math.radians(lon0)
    x0, y0, z0 = ecef(lat0, lon0, 0.0)
    x = x0 - math.sin(lo) * e - math.sin(la) * math.cos(lo) * n
    y = y0 + math.cos(lo) * e - math.sin(la) * math.sin(lo) * n
    z = z0 + math.cos(la) * n
    # ECEF -> geodetic latitude by fixed-point iteration; four rounds reach < 1e-12 rad here.
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2_WGS))
    for _ in range(4):
        nn = A_WGS / np.sqrt(1 - E2_WGS * np.sin(lat) ** 2)
        h = p / np.cos(lat) - nn
        lat = np.arctan2(z, p * (1 - E2_WGS * nn / (nn + h)))
    return np.degrees(lat), np.degrees(np.arctan2(y, x))


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
