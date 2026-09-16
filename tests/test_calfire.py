"""The CAL FIRE audit: name correspondence, timestamp parsing, and the Portola result.

No network. The pure functions are exercised directly, and the corpus-level assertions
read the committed `data/meta/calfire.json`, which is what a run produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.figlib import calfire as CF

META = Path(__file__).resolve().parents[1] / "data" / "meta"


@pytest.fixture(scope="session")
def audit_rows() -> dict:
    doc = json.loads((META / "calfire.json").read_text())
    return {r["fire_id"]: r for r in doc["audit"]}


def test_started_parses_calfire_utc_stamp():
    # 2017-10-10T21:57:00Z is the Portola Fire's reported start.
    assert CF._parse_started("2017-10-10T21:57:00Z") == 1507672620
    assert CF._parse_started(None) is None
    assert CF._parse_started("not a date") is None


@pytest.mark.parametrize("wfigs, cf", [
    ("PORTOLA", "Portola Fire"),
    ("RANCH2", "Ranch 2 Fire"),                 # digit glued to the word
    ("MUTUAL AID JENNINGS", "Jennings Fire"),   # dispatch prefix on one side only
    ("Border 3", "Border Fire"),
])
def test_corresponding_names_are_recognized(wfigs, cf):
    assert CF.corresponds(wfigs, cf)


@pytest.mark.parametrize("wfigs, cf", [
    ("CLUB", "Vail Fire"),                      # CAL FIRE has no record of CLUB
    ("CREELMAN", "Rainbow 3 Fire"),
    ("VALLEY 3", "Rainbow 3 Fire"),             # share only the sequence number
    ("KITCHEN", None),
])
def test_unrelated_names_are_rejected(wfigs, cf):
    assert not CF.corresponds(wfigs, cf)


def test_portola_truth_is_the_one_wfigs_gets_wrong(audit_rows):
    """The finding: WFIGS puts PORTOLA 23.63 km from the solve, CAL FIRE ~1 km."""
    row = audit_rows["20171010_FIRE"]
    assert row["corresponds"]
    assert row["calfire_county"] == "Riverside"      # WFIGS files it in San Diego
    assert row["sources_apart_km"] > 20             # the two records disagree wildly
    assert row["err_calfire_km"] < 1.5              # and the geometry agrees with CAL FIRE
    assert row["improvement_km"] > 20


def test_sources_corroborate_each_other_on_nearly_every_other_fire(audit_rows):
    """Portola is an outlier, not a pattern: elsewhere the two records agree closely.

    This is the guard against overclaiming. If a future change makes the audit report
    widespread truth disagreement, that is a matcher bug until proven otherwise.
    """
    apart = sorted(r["sources_apart_km"] for r in audit_rows.values()
                   if r["corresponds"] and r["sources_apart_km"] is not None)
    assert len(apart) >= 12
    assert apart[len(apart) // 2] < 1.0             # median agreement under a km
    assert sum(1 for a in apart if a > 5) == 1      # only PORTOLA


def test_a_missing_calfire_record_is_never_scored(audit_rows):
    """CLUB and CREELMAN are absent from CAL FIRE; the nearest-in-time fire is unrelated."""
    for fid in ("20250823_Clubfire", "20260722_CreelmanFire"):
        row = audit_rows[fid]
        assert not row["corresponds"]
        assert row["improvement_km"] is None
