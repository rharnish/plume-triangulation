"""The terrain range along one bearing (`terrain_range.py`), on synthetic terrain profiles.

The real profile needs the Copernicus DEM, which CI doesn't have; the logic that turns a
profile and an early box bottom into an allowed interval does not. These pin that the cap
cuts the far end and the band the near end, that truth containment has its stated
tolerance, that the solve penalty lands only where a bearing rules a cell out, and that a
pose from outside the ledger's rules is labelled as such.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

ENV = ("FIGLIB_LENS", "FIGLIB_POSE_LEDGER")


@pytest.fixture
def TR(monkeypatch):
    """Import terrain_range without leaking its env defaults into other test modules."""
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    from src.figlib import terrain_range
    yield terrain_range


def ramp(TR, near_row=1600.0, drop_px_per_km=30.0):
    """A profile whose terrain line rises steadily in the image: row falls with distance."""
    d = np.arange(TR.STEP_M, TR.MAX_KM * 1000.0, TR.STEP_M)
    return {"d": d, "rows": near_row - drop_px_per_km * d / 1000.0}


def test_cap_cuts_the_far_end_and_band_the_near_end(TR):
    prof = ramp(TR)
    y1 = 1300.0
    cap = TR.allowed(prof, y1, None)
    band = TR.allowed(prof, y1, 100.0)
    # Cap alone: from the camera out to where the terrain row is SLACK_PX above the box bottom.
    s = TR.summarise_mask(prof["d"], cap)
    assert s["lo_km"] == pytest.approx(TR.STEP_M / 1000)
    assert s["hi_km"] == pytest.approx((1600 - (y1 - TR.SLACK_PX)) / 30.0, abs=0.05)
    # Band: rows within [y1 - SLACK_PX, y1 + 100], i.e. 6.67 .. 10.67 km, one interval.
    s = TR.summarise_mask(prof["d"], band)
    assert s["lo_km"] == pytest.approx((1600 - (y1 + 100)) / 30.0, abs=0.05)
    assert s["hi_km"] == pytest.approx((1600 - (y1 - TR.SLACK_PX)) / 30.0, abs=0.05)
    idx = np.flatnonzero(band)
    assert np.all(np.diff(idx) == 1)
    assert s["len_km"] == pytest.approx(len(idx) * TR.STEP_M / 1000)


def test_a_wider_band_only_reaches_nearer(TR):
    prof = ramp(TR)
    ok30, ok100, cap = (TR.allowed(prof, 1300.0, bp) for bp in (30.0, 100.0, None))
    assert not np.any(ok30 & ~ok100) and not np.any(ok100 & ~cap)
    hi = [TR.summarise_mask(prof["d"], m)["hi_km"] for m in (ok30, ok100, cap)]
    assert hi[0] == hi[1] == hi[2]


def test_a_box_below_the_nearest_terrain_allows_nothing(TR):
    prof = ramp(TR)
    ok = TR.allowed(prof, 1700.0, 100.0)
    assert not ok.any()
    assert TR.summarise_mask(prof["d"], ok) == {"lo_km": None, "hi_km": None, "len_km": 0.0}


def test_containment_tolerance_is_ten_samples(TR):
    prof = ramp(TR)
    ok = TR.allowed(prof, 1300.0, 100.0)
    s = TR.summarise_mask(prof["d"], ok)
    tol_km = TR.NEAR_SAMPLES * TR.STEP_M / 1000
    assert TR.contains(ok, (s["lo_km"] + s["hi_km"]) / 2)
    assert TR.contains(ok, s["hi_km"] + tol_km * 0.9)
    assert not TR.contains(ok, s["hi_km"] + tol_km * 1.5)
    assert not TR.contains(ok, s["lo_km"] - tol_km * 1.5)


def test_solve_penalty_only_where_the_bearing_rules_a_cell_out(TR):
    cam = {"lat": 33.0, "lon": -117.0}
    bt = {"cam": cam, "b": SimpleNamespace(bearing_deg=0.0), "prof": ramp(TR), "y1": 1300.0}
    north_km = np.array([2.0, 8.0, 15.0])          # allowed interval is 6.67 .. 10.67 km
    lats = cam["lat"] + north_km * 1000 / 111_132.0
    lons = np.array([cam["lon"], cam["lon"] + 5000 / (111_320.0 * math.cos(math.radians(cam["lat"])))])
    pen = TR.terrain_loglik(lats, lons, [bt], 100.0)
    floor = math.log(TR.FLOOR)
    assert pen[:, 0] == pytest.approx([floor, 0.0, floor])
    # 5 km east of the ray at 2 km north is 68 deg off the bearing: outside the fan, no penalty.
    assert pen[0, 1] == 0.0
    # Cap alone keeps the near cells.
    assert TR.terrain_loglik(lats, lons, [bt], None)[:, 0] == pytest.approx([0.0, 0.0, floor])


def solve_entry(camera, day, d_az=1.0, frame_w=3072):
    return {"camera": camera, "epoch": day * 86400.0, "frame_w": frame_w, "d_az": d_az,
            "d_pitch": -2.0, "d_roll": 0.5, "k_ratio": 0.886, "k1": -0.078,
            "source": f"star:{camera}@{day}"}


def test_full_pose_labels_ledger_and_nearest_apart(TR, monkeypatch):
    entries = [solve_entry("xx-n-mobo-c", 1000)]
    monkeypatch.setattr(TR.pose_ledger, "load", lambda p=None: entries)
    hit = TR.full_pose("xx-n-mobo-c", 1000.2 * 86400, 3072)
    assert hit["source"] == "ledger" and hit["gap_days"] == 0
    assert hit["d_pitch"] == pytest.approx(-2.0) and hit["k1"] == pytest.approx(-0.078)

    far = TR.full_pose("xx-n-mobo-c", 3000 * 86400, 3072)
    assert far["source"] == "nearest" and far["gap_days"] == 2000
    assert far["rule"] == "nearest solve, 2000 d away"

    assert TR.full_pose("xx-n-mobo-c", 1000 * 86400, 2048) is None     # lens not measured
    assert TR.full_pose("yy-s-mobo-c", 1000 * 86400, 3072) is None     # never star-solved
