"""Join FIgLib fires to official incident records (WFIGS / IRWIN).

WFIGS publishes, unauthenticated, the interagency incident feed: an official ignition
point and a `FireDiscoveryDateTime` -- the moment a *human* first reported the fire.
That second field is the point of this module. FIgLib tells us when the plume became
visible; WFIGS tells us when somebody called it in. The gap between them is the
warning a camera network could have bought, and it is the number this project reports.

Matching is spatial + temporal: a bounding box around the cameras that saw the fire,
and a window around their plume-appearance clock. Candidates are ranked by time
proximity; named fires additionally confirm on name.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "meta"
CACHE = META_DIR / "wfigs_cache.json"

WFIGS = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
         "WFIGS_Incident_Locations/FeatureServer/0/query")
FIELDS = ("IncidentName,FireDiscoveryDateTime,IncidentSize,POOCounty,POOState,"
          "IrwinID,InitialLatitude,InitialLongitude")

BBOX_PAD_DEG = 0.55      # ~60 km: HPWREN cameras see a long way
WINDOW_H = 6             # search +/- this many hours around plume appearance


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def query(bbox: tuple[float, float, float, float], t_lo: datetime, t_hi: datetime,
          cache: dict) -> list[dict]:
    key = f"{bbox}|{t_lo:%Y-%m-%dT%H}|{t_hi:%Y-%m-%dT%H}"
    if key in cache:
        return cache[key]
    where = (f"FireDiscoveryDateTime>=TIMESTAMP '{t_lo:%Y-%m-%d %H:%M:%S}' AND "
             f"FireDiscoveryDateTime<=TIMESTAMP '{t_hi:%Y-%m-%d %H:%M:%S}'")
    params = urllib.parse.urlencode({
        "where": where,
        "geometry": ",".join(str(v) for v in bbox),
        "geometryType": "esriGeometryEnvelope", "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": FIELDS, "returnGeometry": "true", "f": "json",
    })
    with urllib.request.urlopen(f"{WFIGS}?{params}", timeout=60) as fh:
        data = json.load(fh)
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "WFIGS error"))
    out = []
    for f in data.get("features", []):
        a, g = f["attributes"], f.get("geometry") or {}
        if g.get("y") is None or a.get("FireDiscoveryDateTime") is None:
            continue
        out.append({
            "name": a.get("IncidentName"),
            "discovery_epoch": int(a["FireDiscoveryDateTime"] / 1000),
            "lat": g["y"], "lon": g["x"],
            "acres": a.get("IncidentSize"),
            "county": a.get("POOCounty"), "state": a.get("POOState"),
            "irwin": a.get("IrwinID"),
        })
    cache[key] = out
    return out


def match_fire(fire: dict, seq_by_name: dict, cache: dict) -> dict:
    posed = [seq_by_name[s] for s in fire["sequences"]
             if seq_by_name[s]["has_pose"]]
    if not posed:
        return {**{k: fire[k] for k in ("fire_id", "event")}, "status": "no-pose"}

    lats = [s["lat"] for s in posed]
    lons = [s["lon"] for s in posed]
    bbox = (min(lons) - BBOX_PAD_DEG, min(lats) - BBOX_PAD_DEG,
            max(lons) + BBOX_PAD_DEG, max(lats) + BBOX_PAD_DEG)

    t0 = fire["t0_median"]
    center = datetime.fromtimestamp(t0, UTC)
    cands = query(bbox, center - timedelta(hours=WINDOW_H),
                  center + timedelta(hours=WINDOW_H), cache)

    # A named event carries a strong prior; keep name agreement as a separate signal
    # rather than a filter, so a mislabeled name cannot silently discard the truth.
    label = fire["event"].split("_", 1)[1].lower().replace("fire", "").strip("-_ ")
    for c in cands:
        c["dt_s"] = c["discovery_epoch"] - t0
        c["km_to_nearest_cam"] = round(min(
            haversine_km(c["lat"], c["lon"], s["lat"], s["lon"]) for s in posed), 2)
        c["name_match"] = bool(label) and label in (c["name"] or "").lower()

    cands.sort(key=lambda c: (not c["name_match"], abs(c["dt_s"])))
    return {
        "fire_id": fire["fire_id"], "event": fire["event"],
        "t0_median": t0, "sites": fire["sites"],
        "triangulable": fire["triangulable"],
        "status": "matched" if cands else "no-candidate",
        "n_candidates": len(cands),
        "best": cands[0] if cands else None,
        "candidates": cands[:5],
    }


def main() -> None:
    from . import corpus as C
    from . import provenance as P
    started = P.utc_now()
    meta = C.current().meta
    fires = json.loads((meta / "fires.json").read_text())
    seqs = json.loads((meta / "sequences.json").read_text())
    seq_by_name = {s["seq"]: s for s in seqs}
    cache = _load_cache()

    out = []
    for fire in fires:
        try:
            out.append(match_fire(fire, seq_by_name, cache))
        except Exception as exc:                       # network, not logic
            out.append({"fire_id": fire["fire_id"], "event": fire["event"],
                        "status": f"error: {exc}"})
    CACHE.write_text(json.dumps(cache, indent=1) + "\n")
    dest = meta / "truth.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    # The WFIGS cache is an input as much as an output: a query answered from it is only
    # as current as the day it was cached, so its hash goes in the record.
    P.record("truth", [dest, CACHE], started=started,
             params={"bbox_pad_deg": BBOX_PAD_DEG, "window_h": WINDOW_H,
                     "source": WFIGS},
             extra_inputs=[meta / "fires.json", meta / "sequences.json"])

    matched = [r for r in out if r["status"] == "matched"]
    named = [r for r in matched if r["best"]["name_match"]]
    unique = [r for r in matched if r["n_candidates"] == 1]
    tri = [r for r in matched if r.get("triangulable")]
    print(f"{len(fires)} fires -> {len(matched)} with >=1 WFIGS candidate "
          f"within +/-{WINDOW_H} h")
    print(f"  name-confirmed:        {len(named)}")
    print(f"  single candidate:      {len(unique)}")
    print(f"  triangulable & matched:{len(tri)}")
    print(f"  no candidate:          {sum(1 for r in out if r['status']=='no-candidate')}")
    print(f"  no pose:               {sum(1 for r in out if r['status']=='no-pose')}")
    print(f"-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
