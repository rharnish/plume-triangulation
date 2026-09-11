"""Unit tests for the bearing and likelihood-field maths in `geom.py`.

A sign error here is silent -- every downstream kilometer figure would just be wrong -- so
the primitives are pinned against hand-checkable cases.
"""

from __future__ import annotations

import pytest

from src.figlib.geom import (
    angdiff_deg,
    bearing_deg,
    bearing_x_frac,
    field_argmax,
    haversine_km,
    in_view,
    loglik_field,
    offset_bearing_deg,
    undistort_x,
)


def test_bearing_cardinal_directions():
    # Small offsets near the equator so the great-circle bearing is unambiguous.
    assert bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)      # due north
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0)     # due east
    assert bearing_deg(0.0, 0.0, -1.0, 0.0) == pytest.approx(180.0)   # due south
    assert bearing_deg(0.0, 0.0, 0.0, -1.0) == pytest.approx(270.0)   # due west


def test_bearing_is_in_zero_360():
    b = bearing_deg(32.7, -116.5, 32.6, -116.6)
    assert 0.0 <= b < 360.0


def test_haversine_known_distances():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)
    # One degree of latitude is ~111.2 km anywhere.
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)
    assert haversine_km(45.0, 5.0, 46.0, 5.0) == pytest.approx(111.19, abs=0.1)


def test_angdiff_wraps_the_short_way():
    assert angdiff_deg(10.0, 350.0) == pytest.approx(20.0)
    assert angdiff_deg(350.0, 10.0) == pytest.approx(-20.0)
    assert abs(angdiff_deg(0.0, 180.0)) == pytest.approx(180.0)   # boundary, either sign
    assert abs(angdiff_deg(1.0, 359.0)) == pytest.approx(2.0)


def test_in_view_wedge():
    cam = {"lat": 0.0, "lon": 0.0, "az": 90.0, "fov": 90.0}   # looking east, 45..135
    assert in_view(cam, 0.0, 1.0)                              # dead ahead
    assert in_view(cam, 0.5, 1.0)                              # east-north-east, well inside
    assert not in_view(cam, 0.0, -1.0)                         # behind, due west
    assert not in_view(cam, 1.0, 1.0)                          # ~45 deg, just past the edge
    assert in_view(cam, 1.0, 1.0, margin_deg=1.0)              # margin lets the edge case in


def test_undistort_is_identity_without_k1():
    cam = {"fov": 90.0}
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert undistort_x(cam, x) == x


def test_offset_bearing_and_x_frac_round_trip():
    cam = {"lat": 32.73, "lon": -116.58, "az": 90.0, "fov": 90.0}
    # A place east of the camera, inside the wedge.
    lat, lon = 32.70, -116.40
    x = bearing_x_frac(cam, lat, lon)
    assert x is not None and 0.0 < x < 1.0
    b_direct = bearing_deg(cam["lat"], cam["lon"], lat, lon)
    assert offset_bearing_deg(cam, x) == pytest.approx(b_direct, abs=1e-6)


def test_x_frac_none_outside_view():
    cam = {"lat": 0.0, "lon": 0.0, "az": 90.0, "fov": 90.0}
    assert bearing_x_frac(cam, 0.0, -1.0) is None       # behind the camera


def test_offset_bearing_is_nonlinear_at_the_edges():
    """The rectilinear projection, not a linear deg-per-pixel approximation."""
    cam = {"az": 0.0, "fov": 90.0}
    edge = angdiff_deg(offset_bearing_deg(cam, 1.0), offset_bearing_deg(cam, 0.9))
    mid = angdiff_deg(offset_bearing_deg(cam, 0.6), offset_bearing_deg(cam, 0.5))
    assert edge < mid          # degrees compress toward the frame edge


def test_loglik_field_recovers_a_known_crossing():
    # Two cameras 20 km apart, both aimed at a point between and beyond them.
    target = (34.20, -117.90)
    cam_w = {"lat": 34.05, "lon": -118.05, "az": 45.0, "fov": 90.0}
    cam_e = {"lat": 34.05, "lon": -117.75, "az": 315.0, "fov": 90.0}
    bw = bearing_deg(cam_w["lat"], cam_w["lon"], *target)
    be = bearing_deg(cam_e["lat"], cam_e["lon"], *target)

    lats, lons, field = loglik_field(
        [(cam_w, bw, 2.0), (cam_e, be, 2.0)],
        lat0=34.15, lon0=-117.90, half_extent_km=30.0, step_km=0.25)
    la, lo, _ = field_argmax(lats, lons, field)
    assert haversine_km(la, lo, *target) < 0.5


def test_loglik_field_single_camera_is_a_ridge_not_a_point():
    cam = {"lat": 34.05, "lon": -118.05, "az": 45.0, "fov": 90.0}
    b = bearing_deg(cam["lat"], cam["lon"], 34.20, -117.90)
    lats, lons, field = loglik_field([(cam, b, 2.0)], lat0=34.15, lon0=-117.90,
                                     half_extent_km=20.0, step_km=0.5)
    peak = max(max(row) for row in field)
    # Many cells sit within a hair of the peak -- the bearing constrains one axis only.
    near = sum(v > peak - 0.5 for row in field for v in row)
    assert near > 20
