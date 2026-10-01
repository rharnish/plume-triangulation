"""A FIgLib night's optical centre is its own segment's, never one borrowed from another date."""

from __future__ import annotations

from src.figlib.stars.intrinsics import centre

ROWS = [
    {"camera": "om-w-mobo-c", "W": 3072, "H": 2048, "cx": -30.1, "cy": -10.3,
     "nights": ["20201206_JEEP-ON-FIRE_om-w-mobo-c", "20240806_Border68_om-w-mobo-c"]},
    {"camera": "lp-w-mobo-c", "W": 3072, "H": 2048, "cx": 0.0, "cy": 0.0,
     "nights": ["20201202_WillowFire_lp-w-mobo-c"]},
]


def test_a_night_takes_its_segments_centre():
    assert centre("20240806_Border68_om-w-mobo-c", "om-w-mobo-c", 3072, 2048, ROWS) == (-30.1, -10.3)


def test_nothing_carries_to_a_night_no_segment_holds():
    # same camera and frame size, but this night didn't solve: the frame's middle, not a neighbour's
    assert centre("20220101_Other_om-w-mobo-c", "om-w-mobo-c", 3072, 2048, ROWS) == (0.0, 0.0)
    # another frame size is another unit
    assert centre("20240806_Border68_om-w-mobo-c", "om-w-mobo-c", 2048, 1536, ROWS) == (0.0, 0.0)


def test_a_thin_segment_keeps_the_middle():
    assert centre("20201202_WillowFire_lp-w-mobo-c", "lp-w-mobo-c", 3072, 2048, ROWS) == (0.0, 0.0)
