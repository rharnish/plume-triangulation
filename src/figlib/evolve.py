"""How the estimate and its uncertainty move as a fire develops.

Collapsing a whole sequence to one detection per camera throws away the question a fire
agency actually asks, which is not "where was the fire" but "where did we think it was,
and when did we think it". Two effects run against each other as the plume grows:

* **Uncertainty should tighten.** A bigger plume is detected by more cameras, and more
  bearings mean a smaller credible region.
* **Accuracy should decay, systematically.** Smoke drifts downwind as it rises, so the
  visible plume separates from its source over time. The bearing to it should therefore
  walk in the direction the wind is blowing.

If the second effect is real it is measurable and correctable: the track of estimates
should move downwind at roughly the wind speed, and extrapolating it back toward the
moment of appearance should recover the source. If it is not real, the drift bias is
smaller than the detector's own noise and the earlier wind correction was rightly a wash.

Two views, because they answer different questions:

* `snapshot` -- detections inside a window around t, giving the instantaneous estimate.
* `cumulative` -- every detection up to t, giving what a system would actually believe
  having watched since the beginning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geom import bearing_deg, haversine_km, offset_bearing_deg
from .geolocate import Bearing, credible_area_km2, solve
from .wind import _load as load_wind, wind_at

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
YOLO_DIR = ROOT / "out" / "yolo"


def bearings_in_window(fire, seqs, cams, lo_s, hi_s, conf_thr=0.25):
    """Most confident detection per camera within [lo_s, hi_s] of that camera's t0."""
    out = []
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not path.exists():
            continue
        cam = cams[s["camera"]]
        best = None
        for rec in json.loads(path.read_text()):
            if not (lo_s <= rec["offset"] <= hi_s):
                continue
            for d in rec["dets"]:
                if d["conf"] >= conf_thr and (best is None or d["conf"] > best[0]["conf"]):
                    best = (d, rec)
        if best is None:
            continue
        d, rec = best
        x = (d["x0"] + d["x1"]) / 2
        out.append(Bearing(camera=s["camera"], lat=cam["lat"], lon=cam["lon"],
                           bearing_deg=offset_bearing_deg(cam, x), conf=d["conf"],
                           epoch=rec["epoch"], x_frac=round(x, 4)))
    return out


def track(fire, truth, seqs, cams, mode="cumulative", edges=None):
    edges = edges or [180, 360, 600, 900, 1200, 1800, 2400]
    rows = []
    for t in edges:
        lo = 0 if mode == "cumulative" else max(0, t - 300)
        bs = bearings_in_window(fire, seqs, cams, lo, t)
        sites = {b.camera.split("-")[0] for b in bs}
        if len(sites) < 2:
            rows.append({"t": t, "n_sites": len(sites), "status": "insufficient"})
            continue
        c = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
        lats, lons, ll, la, lo_ = solve(bs, c)
        rows.append({
            "t": t, "n_sites": len(sites), "n_bearings": len(bs), "status": "ok",
            "lat": round(la, 5), "lon": round(lo_, 5),
            "error_km": round(haversine_km(la, lo_, truth["lat"], truth["lon"]), 2),
            "area95_km2": round(credible_area_km2(lats, lons, ll), 1),
        })
    return rows


def main() -> None:
    cams = json.loads((META / "cams.json").read_text())
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = json.loads((META / "resolved.json").read_text())
    wcache = load_wind()

    out = {}
    for r in resolved:
        if r["tier"] not in ("confirmed", "probable") or not r.get("triangulable"):
            continue
        fire, t = fires[r["fire_id"]], r["truth"]
        rows = track(fire, t, seqs, cams, mode="cumulative")
        ok = [x for x in rows if x["status"] == "ok"]
        entry = {"tier": r["tier"], "truth": t, "cumulative": rows}
        if len(ok) >= 2:
            a, b = ok[0], ok[-1]
            entry["drift_km"] = round(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]), 2)
            entry["drift_bearing"] = round(bearing_deg(a["lat"], a["lon"], b["lat"], b["lon"]), 1)
            w = wind_at(t["lat"], t["lon"], fire["t0_median"], wcache)
            if w:
                entry["wind_toward"] = round((w["from_deg_100m"] + 180) % 360, 1)
                entry["wind_kmh"] = w["speed_kmh_100m"]
                d = (entry["drift_bearing"] - entry["wind_toward"] + 180) % 360 - 180
                entry["drift_vs_wind_deg"] = round(d, 1)
        out[r["fire_id"]] = entry

    dest = ROOT / "out" / "evolution.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")

    print("=== cumulative: does uncertainty tighten and error grow? ===")
    print(f"{'t (s)':>6} {'fires':>6} {'med sites':>10} {'med err km':>11} {'med area95':>11}")
    for t in [180, 360, 600, 900, 1200, 1800, 2400]:
        vals = [x for e in out.values() for x in e["cumulative"]
                if x["t"] == t and x["status"] == "ok"]
        if not vals:
            continue
        err = sorted(v["error_km"] for v in vals)
        ar = sorted(v["area95_km2"] for v in vals)
        ns = sorted(v["n_sites"] for v in vals)
        print(f"{t:>6} {len(vals):>6} {ns[len(ns)//2]:>10} "
              f"{err[len(err)//2]:>11.2f} {ar[len(ar)//2]:>11.1f}")

    print("\n=== does the estimate walk downwind? ===")
    ds = [e for e in out.values() if "drift_vs_wind_deg" in e]
    if ds:
        agree = [e for e in ds if abs(e["drift_vs_wind_deg"]) < 60]
        print(f"n={len(ds)} fires with drift and wind; "
              f"{len(agree)} drift within 60 deg of the wind direction "
              f"({100*len(agree)/len(ds):.0f}%)")
        md = sorted(e["drift_km"] for e in ds)
        print(f"median drift over the sequence: {md[len(md)//2]:.2f} km")
        for fid, e in sorted(out.items(), key=lambda kv: -kv[1].get("drift_km", 0))[:10]:
            if "drift_vs_wind_deg" not in e:
                continue
            print(f"  {e['drift_km']:6.2f} km toward {e['drift_bearing']:5.1f} deg   "
                  f"wind toward {e['wind_toward']:5.1f} ({e['wind_kmh']:4.1f} km/h)   "
                  f"delta {e['drift_vs_wind_deg']:+6.1f}   {fid}")


if __name__ == "__main__":
    main()


def fixed_set_drift(fire, truth, seqs, cams, edges=(300, 600, 1200, 2400),
                    conf_thr=0.25):
    """Track the estimate using only cameras that detect in *every* window.

    The cumulative track conflates two motions: the plume moving, and the set of
    contributing cameras changing as the fire becomes visible from more sites. Adding a
    camera can move an estimate kilometres on its own, which swamps drift and is why the
    unrestricted track correlates with wind no better than chance. Holding the camera set
    fixed removes that, leaving only motion of the thing being observed.
    """
    per_cam: dict[str, dict[int, Bearing]] = {}
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not path.exists():
            continue
        cam = cams[s["camera"]]
        recs = sorted(json.loads(path.read_text()), key=lambda r: r["offset"])
        for t in edges:
            lo = t - 300
            best = None
            for rec in recs:
                if not (lo <= rec["offset"] <= t):
                    continue
                for d in rec["dets"]:
                    if d["conf"] >= conf_thr and (best is None
                                                  or d["conf"] > best[0]["conf"]):
                        best = (d, rec)
            if best:
                d, rec = best
                x = (d["x0"] + d["x1"]) / 2
                per_cam.setdefault(s["camera"], {})[t] = Bearing(
                    camera=s["camera"], lat=cam["lat"], lon=cam["lon"],
                    bearing_deg=offset_bearing_deg(cam, x), conf=d["conf"],
                    epoch=rec["epoch"], x_frac=round(x, 4))

    keep = [c for c, m in per_cam.items() if all(t in m for t in edges)]
    if len({c.split("-")[0] for c in keep}) < 2:
        return None

    rows = []
    for t in edges:
        bs = [per_cam[c][t] for c in keep]
        c0 = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
        lats, lons, ll, la, lo_ = solve(bs, c0)
        rows.append({"t": t, "lat": la, "lon": lo_,
                     "error_km": round(haversine_km(la, lo_, truth["lat"],
                                                    truth["lon"]), 2),
                     "area95_km2": round(credible_area_km2(lats, lons, ll), 1)})
    return {"cameras": keep, "rows": rows}
