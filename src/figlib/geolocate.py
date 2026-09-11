"""Locate an ignition from detections in two or more cameras.

Each detection gives a direction, not a place: the horizontal position of a box maps to
a bearing, and where bearings from different sites agree is the fire. The obvious
implementation intersects the lines and reports the crossing point, which is wrong in a
way that matters -- bearings carry angular error, that error fans out with range, and a
crossing point states a precision the geometry never had.

So this scores instead of intersecting. Every cell of a ground grid accumulates, from
each camera, how far it sits from that camera's measured bearing in units of that
camera's angular uncertainty. The sum is a log-likelihood surface whose peak is the
estimate and whose spread is the honest uncertainty: a fan from one camera, an ellipse
from two, an elongated smear when the views are nearly parallel and the geometry really
is that weak.

It also filters. A confident detection on a cloud -- 0.81 on mg-e-mobo-c during the
Junction fire, higher than most true detections -- cannot be separated from a plume by
confidence. But it lies in a direction the other cameras do not corroborate, so it
contributes no peak. Requiring cross-site agreement rejects what thresholding cannot.
"""

from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geom import (angdiff_deg, bearing_deg, haversine_km, load_cams,
                    offset_bearing_deg)
from .wind import upwind_x, wind_at

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
# Which detection pass to score. Overridable so the Core ML variants run through this
# exact pipeline rather than a parallel one -- the point of the quantization study is a
# paired comparison, and a second implementation would be a second source of difference.
YOLO_DIR = Path(os.environ.get("FIGLIB_DETS", ROOT / "out" / "yolo"))

# Angular budget per bearing. Pose is published to a degree, the plume is a meters-wide
# object seen as a box several degrees across, and its centroid sits downwind of the
# source. Two degrees is deliberately generous: claiming less would shrink the
# uncertainty region without earning it.
SIGMA_DEG = 2.0

# Scored columns, in the order they are reported. The four box variants run by
# default; everything past them takes the bearing from a pixel mask instead of a box
# edge, needs the mask cache `masks.build_all` builds (hours on CPU), and lost -- see
# NOTES.md, "Segmentation and per-plume wind fits". Name them in FIGLIB_VARIANTS to
# re-score that comparison: `FIGLIB_VARIANTS=upwind,foot_diff,wedge_sam,...`.
BOX_VARIANTS = ("center", "upwind", "early", "early_upwind")
MASK_VARIANTS = ("foot_diff", "foot_sam", "axis_diff", "axis_sam",
                 "wedge_diff", "wedge_sam", "seq_diff", "seq_sam",
                 "field_diff", "field_sam")
ALL_VARIANTS = BOX_VARIANTS + MASK_VARIANTS
if os.environ.get("FIGLIB_VARIANTS"):
    _keep = set(os.environ["FIGLIB_VARIANTS"].split(","))
    VARIANTS = tuple(v for v in ALL_VARIANTS if v in _keep)
else:
    VARIANTS = BOX_VARIANTS


@dataclass
class Bearing:
    camera: str
    lat: float
    lon: float
    bearing_deg: float
    conf: float
    epoch: int
    x_frac: float
    wind_from_deg: float | None = None
    x_source: str = "center"
    # (bearings_deg, loglik) for the `field` mode: the whole curve a mask implies over
    # direction, kept instead of being collapsed to a mean and a sigma first.
    ll_curve: tuple | None = None


def bearings_for_fire(fire: dict, seqs: dict, cams: dict,
                      conf_thr: float = 0.25,
                      window_s: tuple[int, int] = (0, 2400),
                      use_wind: bool = True,
                      pick: str = "best",
                      wind_cache: dict | None = None,
                      x_mode: str = "box") -> list[Bearing]:
    """One bearing per camera: its most confident detection inside the window.

    With `use_wind`, the bearing is taken from the box edge nearest the source rather
    than from the box center -- see wind.py for why the center is biased downwind.

    `x_mode` of `"foot_diff"` or `"foot_sam"` replaces both with the horizontal position
    of the *foot* of a pixel mask -- the lowest visible smoke, which is the least drifted
    part of the plume and the closest thing in the image to the source. See masks.py.
    Where no mask is available the bearing falls back to whatever `use_wind` selects, and
    `x_source` records which of the two actually produced the number, so a variant that
    looks like a mask result but is three-quarters box edges cannot pass unnoticed.
    """
    out: list[Bearing] = []
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not path.exists():
            continue
        cam = cams[s["camera"]]
        best = None
        for rec in sorted(json.loads(path.read_text()), key=lambda r: r["offset"]):
            if not (window_s[0] <= rec["offset"] <= window_s[1]):
                continue
            for d in rec["dets"]:
                if d["conf"] < conf_thr:
                    continue
                # "earliest" trades confidence for freshness: a plume detected two
                # minutes after it appeared has drifted a fraction of the distance one
                # detected forty minutes later has, so its bearing is less biased even
                # though the detector is less sure of it.
                if best is None or (d["conf"] > best[0]["conf"] if pick == "best"
                                    else False):
                    best = (d, rec)
            if best is not None and pick == "earliest":
                break
        if best is None:
            continue
        d, rec = best

        wfrom = None
        if use_wind:
            w = wind_at(cam["lat"], cam["lon"], rec["epoch"], wind_cache)
            wfrom = w and w["from_deg_100m"]
        x = (upwind_x(d["x0"], d["x1"], cam["az"], wfrom) if use_wind
             else (d["x0"] + d["x1"]) / 2)
        source = "upwind" if use_wind else "center"

        ll_curve = None
        if x_mode != "box":
            kind, _, method = x_mode.partition("_")
            f, ll_curve = _fit_x(kind, method, seq_name, rec["offset"], d,
                                 cam, wfrom)
            if f is not None:
                x, source = f, x_mode
            if ll_curve is not None:
                # In `field` mode the curve is what reaches the posterior, so the
                # bearing counts as mask-derived even when the point estimate beside it
                # fell back to the box.
                source = x_mode

        out.append(Bearing(camera=s["camera"], lat=cam["lat"], lon=cam["lon"],
                           bearing_deg=offset_bearing_deg(cam, x),
                           conf=d["conf"], epoch=rec["epoch"], x_frac=round(x, 4),
                           wind_from_deg=wfrom, x_source=source,
                           ll_curve=ll_curve))
    return out


def _fit_x(kind: str, method: str, seq_name: str, offset: int, det: dict,
           cam: dict, wind_from: float | None):
    """One mask-based estimate of `x`, or a likelihood curve for the `field` mode.

    Returns `(x_frac | None, ll_curve | None)`. `None` for `x` is a genuine refusal --
    every fit in `plumefit.py` reports the case where the mask does not contain the
    structure the model is about, rather than returning a number anyway -- and the
    caller falls back to the box, recording that it did.
    """
    from . import plumefit
    from .masks import sequence_masks
    from .wind import crosswind_sign

    if kind == "foot":
        from .masks import foot_for
        return foot_for(seq_name, offset, det).get(method), None

    cache = sequence_masks(seq_name, method)
    masks, horizon_y = cache["masks"], cache["horizon_y"]
    cross = crosswind_sign(cam["az"], wind_from) if wind_from is not None else None
    # The fits reject a plume that leans against the wind. That gate fired on 21 of 93
    # sequences, and an unconstrained lean agrees with the archived wind on only about
    # two thirds of the sequences where the crosswind is decisive -- so it is not
    # obvious the constraint earns its rejections. FIGLIB_WIND_GATE=0 turns it off, to
    # measure that rather than argue about it.
    if os.environ.get("FIGLIB_WIND_GATE") == "0":
        cross = None

    if kind == "seq":
        return plumefit.sequence_fit(masks, cross).get("x"), None

    m = masks.get(offset)
    if m is None:
        return None, None
    if kind == "axis":
        return plumefit.axis_fit(m, horizon_y).get("x"), None
    if kind == "wedge":
        return plumefit.wedge_fit(m, cross).get("x"), None
    if kind == "field":
        xs, ll = plumefit.column_loglik(m, cross)
        if ll.min() > -0.05:            # flat: the mask carries no directional evidence
            return None, None
        degs = np.array([offset_bearing_deg(cam, float(x)) for x in xs])
        o = np.argsort(degs)
        # The wedge fit gives the point estimate that goes in the table; the curve is
        # what actually reaches the posterior.
        return plumefit.wedge_fit(m, cross).get("x"), (degs[o], ll[o])
    raise ValueError(kind)


def solve(bearings: list[Bearing], center: tuple[float, float],
          half_extent_km: float = 60.0, step_km: float = 0.4,
          sigma_deg: float = SIGMA_DEG):
    """Log-likelihood surface over the ground, and its peak."""
    lat0, lon0 = center
    dlat = step_km / 111.32
    dlon = step_km / (111.32 * math.cos(math.radians(lat0)))
    n = int(half_extent_km / step_km)
    lats = lat0 + np.arange(-n, n + 1) * dlat
    lons = lon0 + np.arange(-n, n + 1) * dlon
    LA, LO = np.meshgrid(lats, lons, indexing="ij")

    total = np.zeros_like(LA)
    for b in bearings:
        p1 = np.radians(b.lat)
        p2 = np.radians(LA)
        dl = np.radians(LO - b.lon)
        y = np.sin(dl) * np.cos(p2)
        x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
        brg = (np.degrees(np.arctan2(y, x))) % 360.0
        if b.ll_curve is not None:
            degs, ll = b.ll_curve
            # The curve is dense over the field of view and flat outside it; anything
            # off the sensor gets the curve's floor rather than a hard rejection, so a
            # single camera cannot veto a cell the others agree on.
            unwrapped = ((brg - degs[0] + 180.0) % 360.0) + degs[0] - 180.0
            total += np.interp(unwrapped, degs, ll, left=ll.min(), right=ll.min())
        else:
            d = (brg - b.bearing_deg + 180.0) % 360.0 - 180.0
            total += -0.5 * (d / sigma_deg) ** 2

    i, j = np.unravel_index(np.argmax(total), total.shape)
    return lats, lons, total, float(lats[i]), float(lons[j])


def credible_area_km2(lats, lons, ll, drop: float = 3.0, step_km: float = 0.4) -> float:
    """Area within `drop` log-likelihood of the peak -- roughly a 95% region for 2 dof."""
    return float((ll >= ll.max() - drop).sum()) * step_km * step_km


def main() -> None:
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = json.loads((META / "resolved.json").read_text())

    from .wind import _load as _load_wind, _save as _save_wind
    wind_cache = _load_wind()

    rows = []
    for r in resolved:
        if r["tier"] not in ("confirmed", "probable") or not r.get("triangulable"):
            continue
        fire = fires[r["fire_id"]]
        t = r["truth"]
        row = {"fire_id": r["fire_id"], "tier": r["tier"],
               "truth_lat": t["lat"], "truth_lon": t["lon"], "truth_name": t["name"]}

        for tag, use_wind, pick, win, x_mode in [v for v in (
                ("center", False, "best", (0, 2400), "box"),
                ("upwind", True, "best", (0, 2400), "box"),
                ("early", False, "earliest", (0, 900), "box"),
                ("early_upwind", True, "earliest", (0, 900), "box"),
                # Same detections as `upwind`, so the only difference between the two
                # columns is where in the box the bearing is taken from.
                ("foot_diff", True, "best", (0, 2400), "foot_diff"),
                ("foot_sam", True, "best", (0, 2400), "foot_sam"),
                ("axis_diff", True, "best", (0, 2400), "axis_diff"),
                ("axis_sam", True, "best", (0, 2400), "axis_sam"),
                ("wedge_diff", True, "best", (0, 2400), "wedge_diff"),
                ("wedge_sam", True, "best", (0, 2400), "wedge_sam"),
                ("seq_diff", True, "best", (0, 2400), "seq_diff"),
                ("seq_sam", True, "best", (0, 2400), "seq_sam"),
                ("field_diff", True, "best", (0, 2400), "field_diff"),
                ("field_sam", True, "best", (0, 2400), "field_sam"))
                if v[0] in VARIANTS]:
            bs = bearings_for_fire(fire, seqs, cams, use_wind=use_wind, pick=pick,
                                   window_s=win, wind_cache=wind_cache, x_mode=x_mode)
            sites = {b.camera.split("-")[0] for b in bs}
            if len(sites) < 2:
                row[tag] = {"status": f"only {len(sites)} site(s)",
                            "n_bearings": len(bs)}
                continue
            center = (float(np.mean([b.lat for b in bs])),
                      float(np.mean([b.lon for b in bs])))
            lats, lons, ll, la, lo = solve(bs, center)
            row[tag] = {
                "status": "solved", "n_bearings": len(bs), "n_sites": len(sites),
                "est_lat": round(la, 5), "est_lon": round(lo, 5),
                "error_km": round(haversine_km(la, lo, t["lat"], t["lon"]), 2),
                "area95_km2": round(credible_area_km2(lats, lons, ll), 1),
                "n_from_mask": sum(1 for b in bs
                                   if b.x_source not in ("center", "upwind")),
                "bearings": [{"camera": b.camera, "deg": round(b.bearing_deg, 2),
                              "conf": b.conf, "x": b.x_frac,
                              "wind_from": b.wind_from_deg,
                              "x_source": b.x_source} for b in bs]}
        row["status"] = row["center"]["status"]
        rows.append(row)
    _save_wind(wind_cache)

    dest = ROOT / "out" / "geolocation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=1) + "\n")

    solved = [r for r in rows if r["center"].get("status") == "solved"]
    print(f"{len(rows)} scoring fires, {len(solved)} solved")
    if not solved:
        return

    def stats(tag):
        e = sorted(r[tag]["error_km"] for r in solved if r[tag]["status"] == "solved")
        return e

    for tag in VARIANTS:
        e = stats(tag)
        if not e:
            continue
        print(f"\n[{tag} bearing]  n={len(e)}  median {e[len(e)//2]:.2f} km  "
              f"p25 {e[len(e)//4]:.2f}  p75 {e[3*len(e)//4]:.2f}  max {e[-1]:.2f}")
        print(f"    within 1 km {sum(x<=1 for x in e)}   2 km {sum(x<=2 for x in e)}   "
              f"5 km {sum(x<=5 for x in e)}")

    for tier in ("confirmed", "probable"):
        e = sorted(r["center"]["error_km"] for r in solved
                   if r["tier"] == tier and r["center"]["status"] == "solved")
        if e:
            print(f"  [{tier:>9}] n={len(e):2d}  median {e[len(e)//2]:6.2f} km  "
                  f"max {e[-1]:6.2f}")

    both = [(r["center"]["error_km"], r["upwind"]["error_km"], r["fire_id"])
            for r in solved if r["center"]["status"] == "solved"
            and r["upwind"]["status"] == "solved"]
    if both:
        better = sum(1 for c, u, _ in both if u < c - 0.05)
        worse = sum(1 for c, u, _ in both if u > c + 0.05)
        print(f"\nwind correction: better on {better}, worse on {worse}, "
              f"unchanged on {len(both)-better-worse}")
        print(f"  median error {np.median([c for c,_,_ in both]):.2f} km -> "
              f"{np.median([u for _,u,_ in both]):.2f} km")

    # The mask variants change one thing only -- where in the box the bearing is taken
    # from -- so they are scored against `upwind` pairwise, on the fires where both
    # solved, and split by tier. The probable tier carries the whole error tail and
    # would otherwise swamp any few-hundred-meter effect on the confirmed one.
    for tag in [v for v in VARIANTS if v not in
                ("center", "upwind", "early", "early_upwind")]:
        pairs = [(r["upwind"]["error_km"], r[tag]["error_km"], r["tier"],
                  r[tag].get("n_from_mask", 0), r[tag]["n_bearings"], r["fire_id"])
                 for r in solved
                 if r["upwind"]["status"] == "solved" and r[tag]["status"] == "solved"]
        if not pairs:
            continue
        got = sum(m for _, _, _, m, _, _ in pairs)
        tot = sum(n for _, _, _, _, n, _ in pairs)
        print(f"\n[{tag} vs upwind]  masks on {got}/{tot} bearings")
        for tier in ("confirmed", "probable", "all"):
            sub = [(u, f, fid) for u, f, t, _, _, fid in pairs
                   if tier in (t, "all")]
            if not sub:
                continue
            mu = float(np.median([u for u, _, _ in sub]))
            mf = float(np.median([f for _, f, _ in sub]))
            better = sum(1 for u, f, _ in sub if f < u - 0.05)
            worse = sum(1 for u, f, _ in sub if f > u + 0.05)
            print(f"  [{tier:>9}] n={len(sub):2d}  {mu:6.2f} -> {mf:6.2f} km   "
                  f"better {better}, worse {worse}, unchanged {len(sub)-better-worse}")

    print("\nper fire (center | early):")
    for r in sorted(solved, key=lambda r: r["upwind"]["error_km"]):
        c = r["center"]
        ea = r.get("early", {})
        eas = f"{ea['error_km']:6.2f}" if ea.get("status") == "solved" else "   -- "
        print(f"  {c['error_km']:7.2f} | {eas} km  {c['n_sites']} sites  "
              f"area95 {c['area95_km2']:7.1f} km2  {r['fire_id']} ({r['tier']})")
    for r in rows:
        if r["status"] != "solved":
            print(f"  --      {r['fire_id']}: {r['status']}")


if __name__ == "__main__":
    main()
