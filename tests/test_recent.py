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
