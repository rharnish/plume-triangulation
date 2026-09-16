"""Audit the WFIGS ground truth against CAL FIRE's own incident record.

`truth.py` joins each fire to WFIGS/IRWIN, the interagency feed. That feed is the one this
project reports against, and for the probable tier it is sometimes wrong: `20171010_FIRE`
carries the coordinate 33.300000, -116.999722 -- 33 deg 18' 00", 116 deg 59' 59", a
placeholder rounded to the arcminute -- and files the fire in San Diego County. CAL FIRE
records the same fire at 33.50488, -117.02132 in *Riverside* County, on De Portola Road
east of Pauba Road, 22.9 km away. The triangulation lands 1.02 km from CAL FIRE's point.

So this is a second opinion on the truth, not a second estimate. To keep it honest the
candidate set is chosen exactly the way `truth.py` chooses its own -- a bounding box around
the cameras that saw the fire, a window around plume appearance -- and never from where we
think the fire was. Scoring a truth source against an estimate that helped select it would
prove nothing. The geometry is only compared after the candidates are fixed.

CAL FIRE publishes a per-year incident list, unauthenticated:

    https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List?inactive=true&year=YYYY

It covers incidents CAL FIRE reports on, so federal-only fires and anything outside
California are expected misses, and a miss is reported as a miss rather than filled in.
The response is cached per year in `data/meta/calfire_cache.json`, because a year's list
keeps changing as records are revised and a result should pin to the copy it used.

    python -m src.figlib.calfire            # match, write <meta>/calfire.json, report
    python -m src.figlib.calfire --refresh  # refetch every year, ignoring the cache
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import corpus as C
from .geom import haversine_km

ROOT = C.ROOT
CACHE = C.SHARED_META / "calfire_cache.json"

SOURCE = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List"

# Deliberately identical to truth.py, so the two sources are asked the same question.
BBOX_PAD_DEG = 0.55      # ~60 km: HPWREN cameras see a long way
WINDOW_H = 6             # search +/- this many hours around plume appearance

FIELDS = ("Name", "Started", "Latitude", "Longitude", "AcresBurned", "County",
          "Location", "AdminUnit", "Type", "UniqueId", "Url")


def _load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _parse_started(s: str | None) -> int | None:
    """CAL FIRE stamps `Started` as ISO-8601 UTC, e.g. 2017-10-10T21:57:00Z."""
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00"))
                   .astimezone(UTC).timestamp())
    except ValueError:
        return None


def fetch_year(year: int) -> list[dict]:
    """Every CAL FIRE incident for one year, trimmed to the fields this module uses."""
    url = f"{SOURCE}?inactive=true&year={year}"
    req = urllib.request.Request(url, headers={"User-Agent": "plume-triangulation"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        raw = json.load(fh)
    out = []
    for r in raw:
        lat, lon = r.get("Latitude"), r.get("Longitude")
        started = _parse_started(r.get("Started"))
        # A record with no coordinate or no start time cannot be matched on either axis.
        if not lat or not lon or started is None:
            continue
        rec = {k: r.get(k) for k in FIELDS}
        rec["Name"] = (rec["Name"] or "").strip()
        rec["started_epoch"] = started
        out.append(rec)
    return out


def year_records(year: int, cache: dict, refresh: bool = False) -> list[dict]:
    key = str(year)
    if refresh or key not in cache:
        cache[key] = fetch_year(year)
    return cache[key]


def match_fire(fire: dict, seq_by_name: dict, cache: dict,
               refresh: bool = False) -> dict:
    posed = [seq_by_name[s] for s in fire["sequences"]
             if seq_by_name.get(s, {}).get("has_pose")]
    base = {k: fire[k] for k in ("fire_id", "event")}
    if not posed:
        return {**base, "status": "no-pose"}

    lats = [s["lat"] for s in posed]
    lons = [s["lon"] for s in posed]
    lo_lat, hi_lat = min(lats) - BBOX_PAD_DEG, max(lats) + BBOX_PAD_DEG
    lo_lon, hi_lon = min(lons) - BBOX_PAD_DEG, max(lons) + BBOX_PAD_DEG

    t0 = fire["t0_median"]
    t_lo = int((datetime.fromtimestamp(t0, UTC) - timedelta(hours=WINDOW_H)).timestamp())
    t_hi = int((datetime.fromtimestamp(t0, UTC) + timedelta(hours=WINDOW_H)).timestamp())

    # A fire near midnight UTC can be filed under either calendar year.
    years = {datetime.fromtimestamp(t, UTC).year for t in (t_lo, t_hi)}
    pool = [r for y in sorted(years) for r in year_records(y, cache, refresh)]

    label = fire["event"].split("_", 1)[1].lower().replace("fire", "").strip("-_ ")
    cands = []
    for r in pool:
        if not (t_lo <= r["started_epoch"] <= t_hi):
            continue
        if not (lo_lat <= r["Latitude"] <= hi_lat and lo_lon <= r["Longitude"] <= hi_lon):
            continue
        name = r["Name"] or ""
        cands.append({
            "name": name,
            "discovery_epoch": r["started_epoch"],
            "lat": r["Latitude"], "lon": r["Longitude"],
            "acres": r["AcresBurned"], "county": r["County"],
            "location": r["Location"], "admin_unit": (r["AdminUnit"] or "").strip(),
            "type": r["Type"], "url": r["Url"], "uid": r["UniqueId"],
            "dt_s": r["started_epoch"] - t0,
            "km_to_nearest_cam": round(min(
                haversine_km(r["Latitude"], r["Longitude"], s["lat"], s["lon"])
                for s in posed), 2),
            "name_match": bool(label) and label in name.lower(),
        })

    cands.sort(key=lambda c: (not c["name_match"], abs(c["dt_s"])))
    return {
        **base, "t0_median": t0, "sites": fire["sites"],
        "triangulable": fire["triangulable"],
        "status": "matched" if cands else "no-candidate",
        "n_candidates": len(cands),
        "best": cands[0] if cands else None,
        "candidates": cands[:5],
    }


STOP_WORDS = frozenset({"fire", "mutual", "aid", "the", "of", "incident"})


def name_tokens(name: str | None) -> set[str]:
    """Comparable words in an incident name: `RANCH2` and `Ranch 2 Fire` share `ranch`."""
    s = re.sub(r"([a-z])(\d)", r"\1 \2", (name or "").lower())
    return {t for t in re.findall(r"[a-z0-9]+", s) if t not in STOP_WORDS}


def corresponds(wfigs_name: str | None, calfire_name: str | None) -> bool:
    """Do two records name the same incident?

    One shared word, and it may not be a bare number: `VALLEY 3` and `Rainbow 3 Fire`
    share only the `3`, which is a sequence number rather than a name.
    """
    shared = name_tokens(wfigs_name) & name_tokens(calfire_name)
    return any(not t.isdigit() for t in shared)


def audit(records: list[dict], wfigs: dict, geo: dict) -> list[dict]:
    """Join each match to the WFIGS truth and to the solved estimate.

    Three numbers per fire: how far our estimate sits from each source's coordinate, and
    how far the two sources sit from each other. The third is the one that matters -- when
    the sources disagree by more than the estimate misses either, the truth is the variable.

    The CAL FIRE record used is the best-ranked *corresponding* one, matched on the WFIGS
    name rather than on our estimate, so the comparison stays independent of the geometry
    it is meant to test. Where CAL FIRE has no record of the fire at all -- `CLUB` and
    `CREELMAN` are not in its lists -- the nearest-in-time incident is some unrelated fire
    tens of km away, which is a missing record, not a disagreement, and is not scored.
    """
    out = []
    for rec in records:
        fid = rec["fire_id"]
        w = (wfigs.get(fid) or {}).get("best")
        cands = rec.get("candidates") or []
        cf = next((c for c in cands if corresponds((w or {}).get("name"), c["name"])), None)
        agrees = cf is not None
        if cf is None:
            cf = rec.get("best")
        g = (geo.get(fid) or {}).get("center") or {}
        est = (g.get("est_lat"), g.get("est_lon")) if g.get("status") == "solved" else None

        row = {
            "fire_id": fid,
            "tier": (geo.get(fid) or {}).get("tier"),
            "wfigs_name": (w or {}).get("name"),
            "calfire_name": (cf or {}).get("name"),
            "calfire_county": (cf or {}).get("county"),
            "calfire_location": (cf or {}).get("location"),
            "calfire_acres": (cf or {}).get("acres"),
            "n_calfire_candidates": rec.get("n_candidates", 0),
            "status": rec["status"],
            "corresponds": agrees,
            "err_wfigs_km": g.get("error_km"),
            "area95_km2": g.get("area95_km2"),
            "err_calfire_km": None,
            "sources_apart_km": None,
            "improvement_km": None,
        }
        if est and cf:
            row["err_calfire_km"] = round(haversine_km(*est, cf["lat"], cf["lon"]), 2)
        if w and cf:
            row["sources_apart_km"] = round(
                haversine_km(w["lat"], w["lon"], cf["lat"], cf["lon"]), 2)
        # Only a corresponding record is evidence about the truth.
        if agrees and row["err_wfigs_km"] is not None and row["err_calfire_km"] is not None:
            row["improvement_km"] = round(row["err_wfigs_km"] - row["err_calfire_km"], 2)
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> None:
    from . import provenance as P
    argv = sys.argv[1:] if argv is None else argv
    refresh = "--refresh" in argv
    started = P.utc_now()

    c = C.current()
    meta = c.meta
    fires = json.loads((meta / "fires.json").read_text())
    seqs = json.loads((meta / "sequences.json").read_text())
    seq_by_name = {s["seq"]: s for s in seqs}
    cache = _load_cache()

    records = []
    for fire in fires:
        try:
            records.append(match_fire(fire, seq_by_name, cache, refresh))
        except Exception as exc:                       # network, not logic
            records.append({"fire_id": fire["fire_id"], "event": fire["event"],
                            "status": f"error: {exc}"})

    CACHE.write_text(json.dumps(cache, indent=1) + "\n")

    wfigs = {r["fire_id"]: r for r in json.loads((meta / "truth.json").read_text())}
    geo_path = c.out / "geolocation.json"
    geo = ({r["fire_id"]: r for r in json.loads(geo_path.read_text())}
           if geo_path.exists() else {})
    rows = audit(records, wfigs, geo)

    dest = meta / "calfire.json"
    dest.write_text(json.dumps({"source": SOURCE,
                                "params": {"bbox_pad_deg": BBOX_PAD_DEG,
                                           "window_h": WINDOW_H},
                                "matches": records, "audit": rows}, indent=1) + "\n")
    P.record("calfire", [dest, CACHE], started=started,
             params={"bbox_pad_deg": BBOX_PAD_DEG, "window_h": WINDOW_H,
                     "source": SOURCE},
             extra_inputs=[meta / "fires.json", meta / "sequences.json",
                           meta / "truth.json"])

    matched = [r for r in records if r["status"] == "matched"]
    corr = [r for r in rows if r["corresponds"]]
    scored = [r for r in rows if r["improvement_km"] is not None]
    apart = sorted((r["sources_apart_km"] for r in corr
                    if r["sources_apart_km"] is not None), reverse=True)
    better = [r for r in scored if r["improvement_km"] > 0.5]
    worse = [r for r in scored if r["improvement_km"] < -0.5]
    print(f"{len(fires)} fires -> {len(matched)} with >=1 CAL FIRE candidate "
          f"within +/-{WINDOW_H} h")
    print(f"  no candidate:            {sum(1 for r in records if r['status']=='no-candidate')}")
    print(f"  no pose:                 {sum(1 for r in records if r['status']=='no-pose')}")
    print(f"  matched, name corresponds:{len(corr)}")
    print(f"  matched, no such incident:{len(matched) - len(corr)}")
    print(f"  scored against a solve:   {len(scored)}")
    if apart:
        print(f"  the two sources agree to within {apart[len(apart)//2]:.2f} km (median), "
              f"{apart[0]:.2f} km (worst)")
        print(f"  disagree by >5 km:        {sum(1 for a in apart if a > 5)}")
    print(f"  CAL FIRE closer by >0.5 km: {len(better)}")
    print(f"  WFIGS closer by >0.5 km:    {len(worse)}")
    if scored:
        med_w = sorted(r["err_wfigs_km"] for r in scored)[len(scored) // 2]
        med_c = sorted(r["err_calfire_km"] for r in scored)[len(scored) // 2]
        print(f"  median error vs WFIGS:    {med_w:.2f} km")
        print(f"  median error vs CAL FIRE: {med_c:.2f} km")
    print(f"-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
