"""Bearings through the whole star-solved camera (`star_calibration.fisheye.unproject_fisheye`).

`geom.offset_bearing_deg` reads the shared lens along the middle row. A solved camera also
has its own lens scale, radial term, pitch and roll; with the camera pitched or rolled, a
pixel's azimuth depends on its row. This pins that the bearing path only takes it when a
solved camera is attached. That the inverse is exact is star-calibration's own test.
"""

from __future__ import annotations

import pytest

from src.figlib.geom import bearing_x_frac, offset_bearing_deg, ray_latlon

CAM = {"lat": 33.0, "lon": -116.8, "az": 135.0, "fov": 90, "elev": 1000.0}
W, H = 3072, 2048


def test_bearing_path_uses_the_solved_camera_only_when_attached(monkeypatch):
    monkeypatch.setenv("FIGLIB_LENS", "fisheye")
    cam = {**CAM, "frame_w": W, "frame_h": H}
    shared = offset_bearing_deg(cam, 0.8, foot_y=0.9)
    assert shared == offset_bearing_deg(cam, 0.8)              # foot_y ignored without it
    solved = {"d_pitch": 0.0, "d_roll": 0.0, "k_ratio": 0.886, "k1": -0.078}
    on_axis_row = offset_bearing_deg({**cam, "solved": solved}, 0.8, foot_y=0.5)
    assert on_axis_row == pytest.approx(shared, abs=1e-6)      # same lens, middle row: same
    # pitched 16.8 deg up, the horizon sits near row 0.8; the shared lens reads it along row 0.5
    tilted = {**solved, "d_pitch": 16.8}
    assert abs(offset_bearing_deg({**cam, "solved": tilted}, 0.8, foot_y=0.8) - shared) > 0.1


def test_bearings_are_read_from_the_optical_centre(monkeypatch):
    """A solve's d_az is the azimuth of its boresight, which lands `cx` px right of the middle:
    that column, not the middle one, reads the corrected azimuth, on both fisheye paths."""
    monkeypatch.setenv("FIGLIB_LENS", "fisheye")
    cam = {**CAM, "frame_w": W, "frame_h": H, "cx": 30.0, "cy": -12.0}
    on_axis = 0.5 + 30.0 / W
    assert offset_bearing_deg(cam, on_axis) == pytest.approx(CAM["az"], abs=1e-6)
    solved = {"d_pitch": 0.0, "d_roll": 0.0, "k_ratio": 0.886, "k1": -0.078}
    assert offset_bearing_deg({**cam, "solved": solved}, on_axis, foot_y=0.5 - 12.0 / H) \
        == pytest.approx(CAM["az"], abs=1e-6)
    assert bearing_x_frac(cam, *ray_latlon(CAM["lat"], CAM["lon"], CAM["az"], 10_000.0)) \
        == pytest.approx(on_axis, abs=1e-6)
    # read from the middle instead, 30 px is most of a degree
    assert abs(offset_bearing_deg(cam, 0.5) - CAM["az"]) > 0.8
