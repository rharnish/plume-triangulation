"""The CDN's 3-hour blocks are local time, and a +-40 min window can straddle two."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.figlib.recent import _blocks

LA = ZoneInfo("America/Los_Angeles")


def _epoch(*local) -> int:
    return int(datetime(*local, tzinfo=LA).timestamp())


def test_window_inside_one_block():
    # 16:30 PDT: 15:50-17:10 sits in Q6 (15:00-17:59).
    assert _blocks(_epoch(2026, 6, 29, 16, 30)) == [("20260629", 6)]


def test_window_straddles_blocks():
    # 15:21 PDT, Bernardo's t0: the window opens at 14:41, in Q5.
    assert _blocks(_epoch(2026, 6, 22, 15, 21)) == [("20260622", 5), ("20260622", 6)]


def test_window_straddles_midnight():
    assert _blocks(_epoch(2026, 7, 1, 0, 10)) == [("20260630", 8), ("20260701", 1)]


# ---- candidate selection ----------------------------------------------------------------

from src.figlib.recent import MAX_KM, fisheye_half_deg, select  # noqa: E402

TRUTH = {"lat": 33.0, "lon": -116.8}
FIRE = {"sequences": ["20260101_TestFire_aa-n-mobo-c"]}


def _cam(lat, lon, az, fov=90):
    return {"lat": lat, "lon": lon, "az": az, "fov": fov, "elev": 500.0, "agl": 10.0}


def test_select_keeps_in_range_in_frame_cameras_not_in_the_archive():
    cams = {
        "aa-n-mobo-c": _cam(32.9, -116.8, 0),         # already in the archive
        "bb-n-mobo-c": _cam(32.9, -116.8, 0),         # 11 km south, looking north: kept
        "bb-n-mobo-m": _cam(32.9, -116.8, 0),         # monochrome twin: not a candidate
        "cc-s-mobo-c": _cam(32.9, -116.8, 180),       # looking away
        "dd-n-mobo-c": _cam(32.0, -116.8, 0),         # 111 km: out of range
        "ee-e-mobo-c": _cam(33.0, -117.0, 90 + 50),   # 50 deg off axis: inside the fisheye frame
        "ff-e-mobo-c": _cam(33.0, -117.0, 90 + 50, fov=60),  # same, narrow lens: outside
    }
    rows = select(FIRE, TRUTH, cams, {}, lambda cam: 0.0)
    assert [r["camera"] for r in rows] == ["bb-n-mobo-c", "ee-e-mobo-c"]
    assert all(r["ok"] and r["km"] <= MAX_KM for r in rows)


def test_select_applies_star_d_az_and_the_terrain_limit():
    cams = {"bb-e-mobo-c": _cam(33.0, -117.0, 90 + 55)}          # just outside the frame
    assert select(FIRE, TRUTH, cams, {}, lambda cam: 0.0) == []
    got = select(FIRE, TRUTH, cams, {"bb-e-mobo-c": -10.0}, lambda cam: 500.0)
    assert [(r["camera"], r["ok"]) for r in got] == [("bb-e-mobo-c", False)]   # hidden


def test_fisheye_frame_is_wider_than_the_nameplate():
    assert 50.0 < fisheye_half_deg() < 60.0
