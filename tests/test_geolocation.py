"""End-to-end geolocation on one fire's real detections.

This is the smoke test: if bearings, the projection, and the likelihood surface all agree,
`20240701_Kitchenfire` lands within a few hundred meters of its official coordinate. The
full corpus run reports 0.08 km here; the assertion leaves generous margin so detector
noise or a grid-resolution change does not make it flaky.
"""

from __future__ import annotations

import numpy as np

from src.figlib import geolocate as G
from src.figlib.geom import haversine_km

FIRE_ID = "20240701_Kitchenfire"


def _solve(fire, seqs, cams, **kw):
    bs = G.bearings_for_fire(fire, seqs, cams, use_wind=False, x_mode="box", **kw)
    center = (float(np.mean([b.lat for b in bs])),
              float(np.mean([b.lon for b in bs])))
    lats, lons, ll, la, lo = G.solve(bs, center)
    return bs, (la, lo), G.credible_area_km2(lats, lons, ll)


def test_all_four_sites_contribute_a_bearing(fires, seqs, cams):
    bs = G.bearings_for_fire(fires[FIRE_ID], seqs, cams, use_wind=False, x_mode="box")
    assert len(bs) == 4
    assert {b.camera.split("-")[0] for b in bs} == {"lp", "mlo", "pi", "ws"}


def test_kitchen_fire_locates_within_half_a_km(fires, seqs, cams, kitchen_truth):
    _, (la, lo), area = _solve(fires[FIRE_ID], seqs, cams)
    err = haversine_km(la, lo, kitchen_truth["lat"], kitchen_truth["lon"])
    assert err < 0.5
    assert area < 10.0          # a tight credible region, four crossing bearings


def test_dropping_to_two_sites_still_locates_but_less_tightly(fires, seqs, cams,
                                                             kitchen_truth):
    fire = dict(fires[FIRE_ID])
    fire["sequences"] = [s for s in fire["sequences"]
                         if s.split("_")[-1].split("-")[0] in ("pi", "ws")]
    _, (la, lo), area = _solve(fire, seqs, cams)
    err = haversine_km(la, lo, kitchen_truth["lat"], kitchen_truth["lon"])
    assert err < 3.0
    assert area > 0.0


def test_a_wrong_bearing_does_not_run_away_with_the_estimate(fires, seqs, cams,
                                                             kitchen_truth):
    """The Gaussian surface is not robust; check the failure is graceful, not wild."""
    bs = G.bearings_for_fire(fires[FIRE_ID], seqs, cams, use_wind=False, x_mode="box")
    bs[0].bearing_deg = (bs[0].bearing_deg + 25.0) % 360.0
    center = (float(np.mean([b.lat for b in bs])),
              float(np.mean([b.lon for b in bs])))
    _, _, _, la, lo = G.solve(bs, center)
    err = haversine_km(la, lo, kitchen_truth["lat"], kitchen_truth["lon"])
    assert err < 15.0          # pulled off, but still in the right basin


def test_refined_peak_does_not_depend_on_the_grid_step(fires, seqs, cams):
    """Unrefined, the peak is a grid node and moves by up to a cell as the step changes."""
    bs = G.bearings_for_fire(fires[FIRE_ID], seqs, cams, use_wind=False, x_mode="box")
    center = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
    peaks = [G.solve(bs, center, step_km=s, refine_km=0.01)[3:] for s in (0.4, 0.25, 0.1)]
    for la, lo in peaks[1:]:
        assert haversine_km(la, lo, *peaks[0]) < 0.03
