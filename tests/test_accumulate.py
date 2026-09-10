"""The robust mixture accumulator on one fire's real detections.

`accumulate.posterior` folds every detection into the surface through a Normal+Uniform
mixture, so a confident outlier falls back on the uniform term instead of dominating.
These check the two properties that matter: it locates the fire, and its influence
function is bounded.
"""

from __future__ import annotations

import numpy as np

from src.figlib import accumulate as A
from src.figlib.geom import haversine_km, load_cams

FIRE_ID = "20240701_Kitchenfire"


def _centre(det_by_cam, cams):
    return (float(np.mean([cams[c]["lat"] for c in det_by_cam])),
            float(np.mean([cams[c]["lon"] for c in det_by_cam])))


def test_pi_of_is_bounded_and_monotone():
    assert A.pi_of(0.0) == 0.0
    assert A.pi_of(1.0) == A.PI_MAX
    assert A.pi_of(2.0) == A.PI_MAX          # clipped, never above PI_MAX
    assert A.pi_of(0.3) < A.pi_of(0.7)


def test_accumulated_posterior_locates_the_fire(fires, seqs, cams, kitchen_truth):
    dets = A.gather(fires[FIRE_ID], seqs, cams, t_max=2400)
    assert len({c.split("-")[0] for c in dets}) >= 2
    _, _, _, la, lo = A.posterior(dets, cams, _centre(dets, cams))
    assert haversine_km(la, lo, kitchen_truth["lat"], kitchen_truth["lon"]) < 1.0


def test_confident_outlier_bearing_has_bounded_influence(fires, seqs, cams,
                                                        kitchen_truth):
    cams = load_cams()
    dets = A.gather(fires[FIRE_ID], seqs, cams, t_max=2400)
    victim = sorted(dets)[0]
    _, _, _, la0, lo0 = A.posterior(dets, cams, _centre(dets, cams))
    base = haversine_km(la0, lo0, kitchen_truth["lat"], kitchen_truth["lon"])

    # Bury one camera under ten high-confidence detections 40 deg off the true bearing.
    bad_bearing = (dets[victim][0][0] + 40.0) % 360.0
    dets[victim] = dets[victim] + [(bad_bearing, 0.9)] * 10
    _, _, _, la1, lo1 = A.posterior(dets, cams, _centre(dets, cams))
    spoiled = haversine_km(la1, lo1, kitchen_truth["lat"], kitchen_truth["lon"])

    assert spoiled < base + 5.0          # the outlier nudges, it does not capture


def test_temporal_discount_alpha_barely_moves_the_estimate(fires, seqs, cams):
    dets = A.gather(fires[FIRE_ID], seqs, cams, t_max=2400)
    c = _centre(dets, cams)
    _, _, _, la_a, lo_a = A.posterior(dets, cams, c, alpha=0.0)
    _, _, _, la_b, lo_b = A.posterior(dets, cams, c, alpha=0.75)
    assert haversine_km(la_a, lo_a, la_b, lo_b) < 2.0
