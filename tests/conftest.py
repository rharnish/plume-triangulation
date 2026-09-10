"""Shared fixtures.

The heavy pipeline needs 13 GB of FIgLib archives, but the geometry does not: it runs on
the committed camera table, the sequence index, and a detection pass. `tests/fixtures/yolo/`
holds the real pyronear detections for one fire -- `20240701_Kitchenfire`, the four-site
0.08 km result from the README -- which is enough to exercise bearings, the likelihood
field, and evidence accumulation end to end in a second or two.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
FIXTURE_YOLO = Path(__file__).resolve().parent / "fixtures" / "yolo"

FIRE_ID = "20240701_Kitchenfire"


@pytest.fixture(autouse=True)
def _point_at_fixture_detections(monkeypatch):
    """Redirect the detection directory both modules read at import time."""
    from src.figlib import accumulate, geolocate

    monkeypatch.setattr(geolocate, "YOLO_DIR", FIXTURE_YOLO)
    monkeypatch.setattr(accumulate, "YOLO_DIR", FIXTURE_YOLO)


@pytest.fixture(scope="session")
def cams() -> dict:
    from src.figlib.geom import load_cams

    return load_cams()


@pytest.fixture(scope="session")
def seqs() -> dict:
    return {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}


@pytest.fixture(scope="session")
def fires() -> dict:
    return {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}


@pytest.fixture(scope="session")
def kitchen_truth() -> dict:
    resolved = json.loads((META / "resolved.json").read_text())
    return next(r["truth"] for r in resolved if r["fire_id"] == FIRE_ID)
