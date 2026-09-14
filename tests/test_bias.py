"""The per-camera bias term (`bias.py`) on one fire's real detections.

`bias.posterior` marginalises an unknown pointing bias per camera. These check that it
reduces to the accumulator it generalises, that the bias widens rather than moves the
answer, and that the pre-hoc flag's residual reads zero for a camera that agrees.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.figlib import accumulate as A
from src.figlib.geom import bearing_deg, haversine_km

FIRE_ID = "20240701_Kitchenfire"
ENV = ("FIGLIB_LENS", "FIGLIB_POSE_LEDGER")


@pytest.fixture
def B(monkeypatch):
    """Import bias.py without leaking its env defaults into other test modules.

    bias.py (through coverage.py) sets FIGLIB_LENS=fisheye and FIGLIB_POSE_LEDGER=1 at import,
    which would silently switch every later geolocation test onto the calibrated camera model.
    """
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    from src.figlib import bias
    yield bias


def _centre(det_by_cam, cams):
    return (float(np.mean([cams[c]["lat"] for c in det_by_cam])),
            float(np.mean([cams[c]["lon"] for c in det_by_cam])))


def test_zero_bias_reproduces_the_accumulator(B, fires, seqs, cams):
    dets = A.gather(fires[FIRE_ID], seqs, cams, t_max=2400)
    c = _centre(dets, cams)
    lats0, lons0, ll0, la0, lo0 = A.posterior(dets, cams, c, half_extent_km=20.0, step_km=0.4,
                                              alpha=0.5)
    lats1, lons1, ll1, la1, lo1 = B.posterior(dets, cams, c, 20.0, 0.4, A.SIGMA_DEG, 0.0, 0.5)
    assert np.allclose(lats0, lats1) and np.allclose(lons0, lons1)
    assert haversine_km(la0, lo0, la1, lo1) < 0.5
    rel0, rel1 = ll0 - ll0.max(), ll1 - ll1.max()
    near = rel0 > -20.0                          # where the surface says anything at all
    assert np.max(np.abs(rel0[near] - rel1[near])) < 0.05


def test_bias_widens_the_region_but_still_locates(B, fires, seqs, cams, kitchen_truth):
    dets = A.gather(fires[FIRE_ID], seqs, cams, t_max=2400)
    c = _centre(dets, cams)
    areas = {}
    for sb in (0.0, 3.0):
        _, _, ll, la, lo = B.posterior(dets, cams, c, 20.0, 0.4, 2.0, sb, 0.5)
        areas[sb] = float((ll >= ll.max() - 3.0).sum())
        assert haversine_km(la, lo, kitchen_truth["lat"], kitchen_truth["lon"]) < 2.5
    assert areas[3.0] > areas[0.0]


def test_camera_agreement_reads_zero_for_an_agreeing_camera(B, cams):
    camera = "hp-s-mobo-c"
    cam = cams[camera]
    la, lo = cam["lat"] - 0.1, cam["lon"] + 0.05
    b = bearing_deg(cam["lat"], cam["lon"], la, lo)
    agree = B.camera_agreement({camera: [(b, 0.9), (b, 0.4)]}, cams, la, lo)
    assert agree[camera]["resid_deg"] < 0.05
    assert agree[camera]["n_det"] == 2

    off = B.camera_agreement({camera: [((b + 12.0) % 360.0, 0.9)]}, cams, la, lo)
    assert off[camera]["resid_deg"] == pytest.approx(12.0, abs=0.05)
    assert B.passes_flag({"resid_max_deg": 4.0, "n_det_min": 3}, 5.0, 3)
    assert not B.passes_flag({"resid_max_deg": 12.0, "n_det_min": 3}, 5.0, 3)
    assert not B.passes_flag({"resid_max_deg": 4.0, "n_det_min": 2}, 5.0, 3)
