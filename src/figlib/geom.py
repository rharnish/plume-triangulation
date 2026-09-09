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

import math

EARTH_R_KM = 6371.0


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


def offset_bearing_deg(cam: dict, x_frac: float) -> float:
    """Bearing to a feature at horizontal position `x_frac` across the image.

    `x_frac` runs 0 (left edge) to 1 (right edge); 0.5 is the optical axis. Uses the
    rectilinear projection rather than assuming degrees scale linearly with pixels --
    at 90 deg FoV the linear approximation is off by several degrees at the edges,
    which at 20 km is a kilometre of error.
    """
    half = math.radians(cam["fov"] / 2.0)
    # Image plane at unit focal length spans [-tan(half), +tan(half)]
    u = (x_frac - 0.5) * 2.0 * math.tan(half)
    return (cam["az"] + cam.get("yaw", 0.0) + math.degrees(math.atan(u))) % 360.0


def bearing_x_frac(cam: dict, lat: float, lon: float) -> float | None:
    """Inverse of `offset_bearing_deg`: where in the frame would (lat, lon) appear?

    Returns None if the point falls outside the field of view.
    """
    b = bearing_deg(cam["lat"], cam["lon"], lat, lon)
    d = math.radians(angdiff_deg(b, cam["az"] + cam.get("yaw", 0.0)))
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
