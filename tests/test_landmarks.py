"""Landmark geometry and static-light extraction: the parts that need no frames or DEM."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.figlib import landmarks as LM


def _vincenty_inverse(lat1, lon1, lat2, lon2):
    """Forward azimuth (deg) and distance (m) on WGS84 -- Vincenty 1975, written out here as
    an independent reference for the ECEF/ENU construction."""
    a, f = 6378137.0, 1 / 298.257223563
    b = a * (1 - f)
    L = math.radians(lon2 - lon1)
    U1, U2 = math.atan((1 - f) * math.tan(math.radians(lat1))), math.atan((1 - f) * math.tan(math.radians(lat2)))
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(200):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        cs = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(ss, cs)
        sa = cU1 * cU2 * sl / ss
        c2a = 1 - sa ** 2
        c2sm = cs - 2 * sU1 * sU2 / c2a
        C = f / 16 * c2a * (4 + f * (4 - 3 * c2a))
        lp, lam = lam, L + (1 - C) * f * sa * (sig + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm ** 2)))
        if abs(lam - lp) < 1e-13:
            break
    u2 = c2a * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds = B * ss * (c2sm + B / 4 * (cs * (-1 + 2 * c2sm ** 2) - B / 6 * c2sm * (-3 + 4 * ss ** 2) * (-3 + 4 * c2sm ** 2)))
    s = b * A * (sig - ds)
    az = math.degrees(math.atan2(cU2 * math.sin(lam), cU1 * sU2 - sU1 * cU2 * math.cos(lam))) % 360
    return az, s


def test_due_north_and_curvature():
    az, el, d = LM.direction(33.0, -117.0, 1000.0, 33.1, -117.0, 1000.0, refract=False)
    assert az == pytest.approx(0.0, abs=1e-9) or az == pytest.approx(360.0, abs=1e-9)
    # same height, 11 km away: the chord dips by half the subtended angle
    assert el == pytest.approx(-math.degrees(d / (2 * 6_371_000)), rel=0.01)


def test_due_east_on_the_equator():
    az, el, _ = LM.direction(0.0, 10.0, 0.0, 0.0, 10.1, 0.0, refract=False)
    assert az == pytest.approx(90.0, abs=1e-9)


@pytest.mark.parametrize("lat2,lon2", [(33.48, -116.52), (32.80, -117.25), (33.16, -116.30)])
def test_azimuth_matches_the_geodesic(lat2, lon2):
    # tens of km across San Diego County: the ENU azimuth of a same-height target is the
    # geodesic's forward azimuth to well under 0.001 deg -- and a sphere is not
    lat1, lon1 = 33.15992, -116.80807
    az, _, d = LM.direction(lat1, lon1, 0.0, lat2, lon2, 0.0, refract=False)
    ref_az, ref_d = _vincenty_inverse(lat1, lon1, lat2, lon2)
    assert az == pytest.approx(ref_az, abs=1e-3)
    assert d == pytest.approx(ref_d, rel=1e-4)


def test_refraction_only_lifts_elevation():
    a0, e0, _ = LM.direction(33.0, -117.0, 1500.0, 33.3, -116.8, 800.0, refract=False)
    a1, e1, d = LM.direction(33.0, -117.0, 1500.0, 33.3, -116.8, 800.0, refract=True)
    assert a1 == a0
    assert e1 - e0 == pytest.approx(math.degrees(LM.K_TERRESTRIAL * d / (2 * 6_371_000)))


def test_dms():
    assert LM.dms("33", "9", "35.7", "N") == pytest.approx(33 + 9 / 60 + 35.7 / 3600)
    assert LM.dms("116", "48", "", "W") == pytest.approx(-(116 + 48 / 60))


def test_static_lights_keeps_still_and_blinking_drops_moving():
    rng = np.random.default_rng(1)
    frames = []
    for i in range(40):
        pts = [[500 + rng.normal(0, 0.2), 300 + rng.normal(0, 0.2), 90]]           # steady light
        if i % 4 == 0:
            pts.append([900 + rng.normal(0, 0.2), 700 + rng.normal(0, 0.2), 60])    # 25% beacon
        pts.append([100 + 3 * i, 100 + i, 50])                                      # a star, moving
        frames.append(np.array(pts, float))
    out = LM.static_lights(frames, 40)
    got = sorted((round(l["x"]), round(l["y"]), round(l["duty"], 2)) for l in out)
    assert got == [(500, 300, 1.0), (900, 700, 0.25)]
