"""Resolve each fire to a single official incident, with a stated confidence.

Three independent signals, none of which is trusted alone:

* **Name.** Only some FIgLib events are named, but when a name matches it is decisive.
* **Time.** On the name-confirmed fires, official discovery lands within about ten
  minutes of the annotated plume clock (median -0.6 min). Candidates hours away are
  other fires that happened to burn the same day in the same county -- common, since
  an unnamed `..._FIRE` label covers every unnamed fire the network saw that date.
* **Geometry.** A fixed camera sees a 90 deg wedge. An incident outside the wedge of
  most of the cameras that recorded the fire cannot be the fire. This needs no
  detector -- it falls out of the published pose alone.

The geometric filter is applied with a margin, because cameras were annotated as
showing smoke when the *plume* entered frame even though the *ignition point* did not:
at 20260629_JunctionFire, `vo-w` holds the fire at +35.9 deg while `vo-n` at the same
site is 54 deg off axis. Drifting smoke is visible from cameras that cannot see its
source, which is the same displacement that makes plume-axis correction necessary
downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

from .geom import in_view

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "meta"

FOV_MARGIN_DEG = 10      # plume extends beyond the ignition point
DT_TIGHT_S = 45 * 60     # calibrated on name-confirmed fires (|dt| <= ~11 min there)


def visible_to(cands, views, margin=FOV_MARGIN_DEG):
    """Keep candidates seen by at least half the posed cameras."""
    need = (len(views) + 1) // 2
    return [c for c in cands
            if sum(in_view(cam, c["lat"], c["lon"], margin) for cam in views) >= need]


def resolve_one(rec: dict, views: list[dict]) -> dict:
    out = {k: rec[k] for k in ("fire_id", "event") if k in rec}
    out["triangulable"] = rec.get("triangulable", False)
    out["sites"] = rec.get("sites", [])
    if rec.get("status") != "matched" or not views:
        return {**out, "tier": "none", "reason": rec.get("status", "unknown"),
                "truth": None}

    cands = visible_to(rec["candidates"], views)
    timed = [c for c in cands if abs(c["dt_s"]) <= DT_TIGHT_S]

    named = [c for c in timed if c["name_match"]] or [c for c in cands if c["name_match"]]
    if named:
        return {**out, "tier": "confirmed", "reason": "name + geometry",
                "truth": named[0], "n_survivors": len(named)}
    if len(timed) == 1:
        return {**out, "tier": "probable", "reason": "unique after geometry + time",
                "truth": timed[0], "n_survivors": 1}
    if timed:
        timed.sort(key=lambda c: abs(c["dt_s"]))
        return {**out, "tier": "ambiguous", "reason": f"{len(timed)} survive",
                "truth": timed[0], "n_survivors": len(timed)}
    return {**out, "tier": "none",
            "reason": "no candidate within geometry + time", "truth": None,
            "n_survivors": 0}


def main() -> None:
    from . import corpus as C
    from . import provenance as P
    started = P.utc_now()
    meta = C.current().meta
    truth = json.loads((meta / "truth.json").read_text())
    fires = {f["fire_id"]: f for f in json.loads((meta / "fires.json").read_text())}
    seqs = {s["seq"]: s for s in json.loads((meta / "sequences.json").read_text())}
    cams = json.loads((META_DIR / "cams.json").read_text())

    out = []
    for rec in truth:
        fire = fires[rec["fire_id"]]
        views = [cams[seqs[s]["camera"]] for s in fire["sequences"]
                 if seqs[s]["has_pose"]]
        out.append(resolve_one({**rec, "triangulable": fire["triangulable"],
                                "sites": fire["sites"]}, views))

    dest = meta / "resolved.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    P.record("resolve", [dest], started=started,
             params={"fov_margin_deg": FOV_MARGIN_DEG, "dt_tight_s": DT_TIGHT_S},
             extra_inputs=[meta / "truth.json", meta / "fires.json",
                           meta / "sequences.json", META_DIR / "cams.json"])

    from collections import Counter
    tiers = Counter(r["tier"] for r in out)
    usable = [r for r in out if r["tier"] in ("confirmed", "probable")]
    scoreable = [r for r in usable if r["triangulable"]]
    print("tier counts:", dict(tiers))
    print(f"usable ground truth (confirmed + probable): {len(usable)}")
    print(f"  of which triangulable from 2+ sites:      {len(scoreable)}")
    print(f"-> {dest.relative_to(ROOT)}")

    dts = sorted(r["truth"]["dt_s"] / 60 for r in usable)
    print(f"\ndiscovery minus plume-appearance, n={len(dts)}: "
          f"median {dts[len(dts)//2]:+.1f} min, "
          f"IQR {dts[len(dts)//4]:+.1f} .. {dts[3*len(dts)//4]:+.1f}")


if __name__ == "__main__":
    main()
