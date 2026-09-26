"""The shared terrain geometry (`terrain.py`, `geom.ray_latlon`), without the DEM.

CI has no Copernicus tiles or rasterio, so the DEM is replaced by a small synthetic raster
with a north-up transform. These pin that a ray walks the camera's line of sight on the
ellipsoid, that sight angles carry curvature and refraction, that a pixel's value sits at its
centre, and that a camera whose height disagrees with the terrain is flagged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.figlib import terrain as T
from src.figlib.geom import bearing_deg, enu, ray_latlon

LAT, LON = 33.0, -116.8
RES = 1 / 3600                       # GLO-30's 1 arcsecond


class NorthUp:
    """Just enough of an affine transform for `Dem.sample`: `~t * (lon, lat) -> (col, row)`."""

    def __init__(self, west: float, north: float, res: float = RES):
        self.west, self.north, self.res = west, north, res

    def __invert__(self):
        return self

    def __mul__(self, lonlat):
        lon, lat = lonlat
        return (np.asarray(lon) - self.west) / self.res, (self.north - np.asarray(lat)) / self.res


class FakeDem:
    """A constant-height plain, optionally with one wall, standing in for `terrain.Dem`."""

    def __init__(self, height=0.0, n=4000):
        self.band = np.full((n, n), height, np.float32)
        self.tf = NorthUp(LON - n * RES / 2, LAT + n * RES / 2)

    def window(self, lat0, lon0, pad_deg):
        return self.band, self.tf


@pytest.mark.parametrize("az", [0, 45, 90, 135, 225, 300])
def test_ray_walks_the_line_of_sight(az):
    """Every point is at the ray's own azimuth, and at its distance, even at 80 km."""
    for d in (5_000.0, 40_000.0, 80_000.0):
        la, lo = ray_latlon(LAT, LON, az, d)
        miss = (bearing_deg(LAT, LON, float(la), float(lo)) - az + 180) % 360 - 180
        e, n, _ = enu(LAT, LON, 0.0, float(la), float(lo), 0.0)
        assert abs(miss) < 1e-3
        assert math.hypot(e, n) == pytest.approx(d, abs=10.0)


def test_ray_broadcasts_azimuths_against_distances():
    la, lo = ray_latlon(LAT, LON, np.array([10.0, 20.0, 30.0])[:, None], np.arange(1, 5)[None, :] * 1e3)
    assert la.shape == lo.shape == (3, 4)


def test_sight_angles_carry_curvature_and_refraction():
    """Level ground at eye height dips by d/(2 R_eff): 4/3 of the Earth's radius."""
    d = np.array([10e3, 40e3, 80e3])
    ang = T.sight_angles(np.zeros(3), 0.0, d)
    assert ang == pytest.approx(-np.degrees(d / (2 * T.EARTH_R_M * 4 / 3)), rel=1e-4)
    assert T.sight_angles(100.0, 0.0, 100.0) == pytest.approx(45.0, abs=1e-3)


def test_a_pixel_value_sits_at_its_centre():
    band = np.arange(16, dtype=np.float32).reshape(4, 4)
    tf = NorthUp(0.0, 4 * RES)
    # centre of row 1, column 2
    lat, lon = np.array([4 * RES - 1.5 * RES]), np.array([2.5 * RES])
    assert T.Dem.sample(band, tf, lat, lon)[0] == pytest.approx(band[1, 2])


def test_site_problems_are_flagged():
    dem = FakeDem(height=500.0)
    ok = {"lat": LAT, "lon": LON, "elev": 500.0, "agl": 10.0}
    assert T.Dem.site_problem(ok, dem.band, dem.tf) is None
    assert "elev" in T.Dem.site_problem({**ok, "elev": 0}, dem.band, dem.tf)
    assert "below" in T.Dem.site_problem({**ok, "elev": 480.0, "agl": None}, dem.band, dem.tf)


def test_sightline_sees_over_a_plain_and_not_through_a_wall():
    dem = FakeDem(height=0.0)
    cam = {"lat": LAT, "lon": LON, "elev": 0.0, "agl": 20.0}
    la, lo = ray_latlon(LAT, LON, 60.0, 10_000.0)
    g = T.sightline(dem, cam, float(la), float(lo), h_target=0.0)
    assert g["az"] == pytest.approx(60.0, abs=1e-3)
    assert g["km"] == pytest.approx(10.0, abs=0.01)
    assert g["el"] >= g["occ_el"]

    # a 200 m wall across the ray at 5 km
    wl, wo = ray_latlon(LAT, LON, 60.0, np.arange(4_950.0, 5_050.0, 10.0))
    inv = ~dem.tf
    for a, b in zip(wl, wo):
        c, r = inv * (b, a)
        dem.band[int(r) - 3:int(r) + 4, int(c) - 3:int(c) + 4] = 200.0
    g = T.sightline(dem, cam, float(la), float(lo), h_target=0.0)
    assert g["occ_el"] > g["el"]
    assert g["occ_km"] == pytest.approx(5.0, abs=0.2)
