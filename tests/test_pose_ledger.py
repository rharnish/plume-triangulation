"""corrected_cam: the ledger's correction reaches a camera only when FIGLIB_POSE_LEDGER is on.

The lookup rules themselves are star_calibration.ledger's, and tested there.
"""

from __future__ import annotations

import pytest

from src.figlib import pose_ledger as L

DAY = L.DAY_S


def entry(camera, day, d_az, frame_w=3072):
    return {"camera": camera, "epoch": day * DAY, "frame_w": frame_w, "d_az": d_az,
            "source": f"star:{camera}@{day}"}


LEDGER = [
    entry("a", 100, 2.0), entry("a", 101, 2.4),          # two solves one night apart
    entry("b", 100, 0.3), entry("b", 900, 0.5),          # agree across a long gap
    entry("om-s", 100, -10.5), entry("om-s", 1800, -0.4),  # re-aimed in between
    entry("c", 1000, 1.5),
]


def test_corrected_cam_is_a_no_op_unless_enabled(monkeypatch):
    cam = {"az": 180.0, "fov": 90, "frame_w": 3072}
    monkeypatch.delenv("FIGLIB_POSE_LEDGER", raising=False)
    assert L.corrected_cam("c", cam, 1000 * DAY, LEDGER) == (cam, None)
    monkeypatch.setenv("FIGLIB_POSE_LEDGER", "1")
    out, hit = L.corrected_cam("c", cam, 1000 * DAY, LEDGER)
    assert out["az"] == pytest.approx(181.5) and cam["az"] == 180.0 and hit["rule"] == "same-night"
