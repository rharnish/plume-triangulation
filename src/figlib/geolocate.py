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
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geom import angdiff_deg, bearing_deg, haversine_km, offset_bearing_deg
from .wind import upwind_x, wind_at

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
YOLO_DIR = ROOT / "out" / "yolo"

# Angular budget per bearing. Pose is published to a degree, the plume is a metres-wide
# object seen as a box several degrees across, and its centroid sits downwind of the
# source. Two degrees is deliberately generous: claiming less would shrink the
# uncertainty region without earning it.
SIGMA_DEG = 2.0


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


def bearings_for_fire(fire: dict, seqs: dict, cams: dict,
                      conf_thr: float = 0.25,
                      window_s: tuple[int, int] = (0, 2400),
                      use_wind: bool = True,
                      pick: str = "best",
                      wind_cache: dict | None = None) -> list[Bearing]:
    """One bearing per camera: its most confident detection inside the window.

    With `use_wind`, the bearing is taken from the box edge nearest the source rather
    than from the box centre -- see wind.py for why the centre is biased downwind.
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

        out.append(Bearing(camera=s["camera"], lat=cam["lat"], lon=cam["lon"],
                           bearing_deg=offset_bearing_deg(cam, x),
                           conf=d["conf"], epoch=rec["epoch"], x_frac=round(x, 4),
                           wind_from_deg=wfrom))
    return out


def solve(bearings: list[Bearing], centre: tuple[float, float],
          half_extent_km: float = 60.0, step_km: float = 0.4,
          sigma_deg: float = SIGMA_DEG):
    """Log-likelihood surface over the ground, and its peak."""
    lat0, lon0 = centre
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
        d = (brg - b.bearing_deg + 180.0) % 360.0 - 180.0
        total += -0.5 * (d / sigma_deg) ** 2

    i, j = np.unravel_index(np.argmax(total), total.shape)
    return lats, lons, total, float(lats[i]), float(lons[j])


def credible_area_km2(lats, lons, ll, drop: float = 3.0, step_km: float = 0.4) -> float:
    """Area within `drop` log-likelihood of the peak -- roughly a 95% region for 2 dof."""
    return float((ll >= ll.max() - drop).sum()) * step_km * step_km


def main() -> None:
    cams = json.loads((META / "cams.json").read_text())
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

        for tag, use_wind, pick, win in (
                ("centre", False, "best", (0, 2400)),
                ("upwind", True, "best", (0, 2400)),
                ("early", False, "earliest", (0, 900)),
                ("early_upwind", True, "earliest", (0, 900))):
            bs = bearings_for_fire(fire, seqs, cams, use_wind=use_wind, pick=pick,
                                   window_s=win, wind_cache=wind_cache)
            sites = {b.camera.split("-")[0] for b in bs}
            if len(sites) < 2:
                row[tag] = {"status": f"only {len(sites)} site(s)",
                            "n_bearings": len(bs)}
                continue
            centre = (float(np.mean([b.lat for b in bs])),
                      float(np.mean([b.lon for b in bs])))
            lats, lons, ll, la, lo = solve(bs, centre)
            row[tag] = {
                "status": "solved", "n_bearings": len(bs), "n_sites": len(sites),
                "est_lat": round(la, 5), "est_lon": round(lo, 5),
                "error_km": round(haversine_km(la, lo, t["lat"], t["lon"]), 2),
                "area95_km2": round(credible_area_km2(lats, lons, ll), 1),
                "bearings": [{"camera": b.camera, "deg": round(b.bearing_deg, 2),
                              "conf": b.conf, "x": b.x_frac,
                              "wind_from": b.wind_from_deg} for b in bs]}
        row["status"] = row["centre"]["status"]
        rows.append(row)
    _save_wind(wind_cache)

    dest = ROOT / "out" / "geolocation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=1) + "\n")

    solved = [r for r in rows if r["centre"].get("status") == "solved"]
    print(f"{len(rows)} scoring fires, {len(solved)} solved")
    if not solved:
        return

    def stats(tag):
        e = sorted(r[tag]["error_km"] for r in solved if r[tag]["status"] == "solved")
        return e

    for tag in ("centre", "upwind", "early", "early_upwind"):
        e = stats(tag)
        if not e:
            continue
        print(f"\n[{tag} bearing]  n={len(e)}  median {e[len(e)//2]:.2f} km  "
              f"p25 {e[len(e)//4]:.2f}  p75 {e[3*len(e)//4]:.2f}  max {e[-1]:.2f}")
        print(f"    within 1 km {sum(x<=1 for x in e)}   2 km {sum(x<=2 for x in e)}   "
              f"5 km {sum(x<=5 for x in e)}")

    for tier in ("confirmed", "probable"):
        e = sorted(r["centre"]["error_km"] for r in solved
                   if r["tier"] == tier and r["centre"]["status"] == "solved")
        if e:
            print(f"  [{tier:>9}] n={len(e):2d}  median {e[len(e)//2]:6.2f} km  "
                  f"max {e[-1]:6.2f}")

    both = [(r["centre"]["error_km"], r["upwind"]["error_km"], r["fire_id"])
            for r in solved if r["centre"]["status"] == "solved"
            and r["upwind"]["status"] == "solved"]
    if both:
        better = sum(1 for c, u, _ in both if u < c - 0.05)
        worse = sum(1 for c, u, _ in both if u > c + 0.05)
        print(f"\nwind correction: better on {better}, worse on {worse}, "
              f"unchanged on {len(both)-better-worse}")
        print(f"  median error {np.median([c for c,_,_ in both]):.2f} km -> "
              f"{np.median([u for _,u,_ in both]):.2f} km")

    print("\nper fire (centre | early):")
    for r in sorted(solved, key=lambda r: r["upwind"]["error_km"]):
        c = r["centre"]
        ea = r.get("early", {})
        eas = f"{ea['error_km']:6.2f}" if ea.get("status") == "solved" else "   -- "
        print(f"  {c['error_km']:7.2f} | {eas} km  {c['n_sites']} sites  "
              f"area95 {c['area95_km2']:7.1f} km2  {r['fire_id']} ({r['tier']})")
    for r in rows:
        if r["status"] != "solved":
            print(f"  --      {r['fire_id']}: {r['status']}")


if __name__ == "__main__":
    main()
