"""Is the terrain-fitted pose real, or did four parameters just bend a curve?

`calibrate.py` fits azimuth, pitch, roll and one distortion coefficient against a single
frame's skyline, and it reduces the residual dramatically. That proves nothing on its own:
enough free parameters will align almost any curve to almost any other, and several fits
land on their bounds, which is exactly what overfitting looks like.

Two held-out tests, neither of which the fit could have seen:

* **Another day.** Fit on one frame, measure the residual on a frame from a *different
  sequence* of the same camera -- a different date, different weather, different clouds. A
  pose that is real transfers; a pose that traced one afternoon's cloud bank does not.
* **The fires.** Re-run geolocation with the refined azimuths and compare kilometer error
  against official WFIGS coordinates. The fit never saw a fire, a detection or a ground
  truth coordinate, so this is as independent as evidence gets here. If pose refinement is
  real geometry, error should fall; if it is curve-bending, it should not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .terrain import Dem, horizon
from .viz_terrain import observed_skyline, _best_frame
from .calibrate import _predicted_rows, MIN_COVERAGE

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
OUT = ROOT / "out"


def holdout(argv: list[str]) -> None:
    """Residual on a frame from a different day than the one fitted."""
    cams = json.loads((META / "cams.json").read_text())
    seqs = json.loads((META / "sequences.json").read_text())
    fits = {r["camera"]: r for r in json.loads((OUT / "pose_fit.json").read_text())
            if r["status"] == "fitted"}

    by_cam: dict[str, list] = {}
    for s in seqs:
        if s["has_pose"]:
            by_cam.setdefault(s["camera"], []).append(s)

    dem = Dem()
    rows = []
    for name, fit in sorted(fits.items()):
        others = [s for s in by_cam.get(name, []) if s["seq"] != fit["seq"]]
        if not others:
            continue
        img = _best_frame(others[0])
        if img is None:
            continue
        obs = observed_skyline(img)
        if float(np.isfinite(obs).mean()) < MIN_COVERAGE:
            continue
        H, W = img.shape[:2]
        prof = horizon(cams[name], dem)
        p = np.array([fit["d_az"], fit["d_pitch"], fit["d_roll"], fit["k1"]])

        def mad(params):
            pred = _predicted_rows(cams[name], prof, W, H, params)
            m = np.isfinite(pred) & np.isfinite(obs)
            return float(np.median(np.abs((pred - obs)[m]))) if m.sum() > W * 0.2 else None

        b, a = mad(np.zeros(4)), mad(p)
        if b is None or a is None:
            continue
        rows.append(dict(camera=name, fit_seq=fit["seq"], test_seq=others[0]["seq"],
                         before_mad_px=round(b, 1), after_mad_px=round(a, 1),
                         improved=bool(a < b)))
        print(f"{name:22s} {b:7.1f} -> {a:6.1f} px  "
              f"{'better' if a < b else 'WORSE '}   (fit {fit['seq'][:8]}, "
              f"test {others[0]['seq'][:8]})", flush=True)

    (OUT / "pose_holdout.json").write_text(json.dumps(rows, indent=1) + "\n")
    if rows:
        b = np.array([r["before_mad_px"] for r in rows])
        a = np.array([r["after_mad_px"] for r in rows])
        n_better = sum(r["improved"] for r in rows)
        print(f"\n{len(rows)} cameras with a second day available")
        print(f"held-out MAD: median {np.median(b):.0f} -> {np.median(a):.0f} px")
        print(f"improved on {n_better}/{len(rows)} cameras")


def write_refined_cams(fit_file: str = "pose_fit.json") -> Path:
    """cams.json with fitted azimuths applied, so geolocation can be re-run against it."""
    cams = json.loads((META / "cams.json").read_text())
    fits = [r for r in json.loads((OUT / fit_file).read_text())
            if r["status"] == "fitted"]
    out = json.loads(json.dumps(cams))
    n = 0
    for r in fits:
        c = out.get(r["camera"])
        if c is None:
            continue
        c["az"] = round((c["az"] + r["d_az"]) % 360.0, 3)
        c["k1"] = r["k1"]
        c["az_refined"] = True
        n += 1
    # Derived, and (as it turns out) not an improvement -- so it belongs in out/ with
    # the rest of the experiment's artefacts, not beside the published metadata.
    dest = OUT / "cams_refined.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest} ({n} cameras refined of {len(out)})")
    return dest


def _score(cams_path: Path | None) -> dict:
    """Accumulated geolocation error per fire, under a given camera table.

    `load_cams` reads FIGLIB_CAMS at call time rather than import time, so switching
    tables is just an environment change -- no second process and no second code path.
    """
    import os

    from .accumulate import gather, posterior
    from .geom import haversine_km, load_cams

    if cams_path:
        os.environ["FIGLIB_CAMS"] = str(cams_path)
    else:
        os.environ.pop("FIGLIB_CAMS", None)

    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    res = [r for r in json.loads((META / "resolved.json").read_text())
           if r["tier"] in ("confirmed", "probable") and r.get("triangulable")]

    out = {}
    for t_max in (180, 2400):
        rows = []
        for r in res:
            fire, truth = fires[r["fire_id"]], r["truth"]
            dets = gather(fire, seqs, cams, t_max)
            if len({c.split("-")[0] for c in dets}) < 2:
                continue
            c = (float(np.mean([cams[k]["lat"] for k in dets])),
                 float(np.mean([cams[k]["lon"] for k in dets])))
            _, _, _, la, lo = posterior(dets, cams, c, alpha=0.25)
            rows.append((r["fire_id"], r["tier"],
                         round(haversine_km(la, lo, truth["lat"], truth["lon"]), 3)))
        out[t_max] = rows
    return out


def geo_compare() -> None:
    """The decisive test: does terrain-refined pose reduce kilometres of error?

    The fit never saw a fire, a detection, or a ground-truth coordinate -- only ridgelines
    in pre-ignition frames. So this is as held-out as evidence gets here. If refinement
    recovered real geometry the error should fall; if four parameters merely bent a curve
    to one afternoon's skyline, it should not.
    """
    base = _score(None)
    ref = _score(OUT / "cams_refined.json")

    summary = {}
    for t_max in (180, 2400):
        b = {r[0]: r[2] for r in base[t_max]}
        a = {r[0]: r[2] for r in ref[t_max]}
        tiers = {r[0]: r[1] for r in base[t_max]}
        common = sorted(set(b) & set(a))
        db = np.array([b[k] for k in common])
        da = np.array([a[k] for k in common])
        summary[str(t_max)] = s = dict(
            n=len(common), n_base=len(b), n_refined=len(a),
            median_before=round(float(np.median(db)), 2),
            median_after=round(float(np.median(da)), 2),
            mean_before=round(float(db.mean()), 2),
            mean_after=round(float(da.mean()), 2),
            within2_before=int((db <= 2).sum()), within2_after=int((da <= 2).sum()),
            within5_before=int((db <= 5).sum()), within5_after=int((da <= 5).sum()),
            improved=int((da < db).sum()), worsened=int((da > db).sum()),
            per_fire=[dict(fire=k, tier=tiers[k], before=b[k], after=a[k],
                           delta=round(a[k] - b[k], 2)) for k in common])
        print(f"\n== accumulated to {t_max} s  (n={s['n']} scored both ways; "
              f"solvable {s['n_base']} -> {s['n_refined']})")
        print(f"   median  {s['median_before']:6.2f} -> {s['median_after']:6.2f} km")
        print(f"   mean    {s['mean_before']:6.2f} -> {s['mean_after']:6.2f} km")
        print(f"   <=2 km  {s['within2_before']:6d} -> {s['within2_after']:6d}")
        print(f"   <=5 km  {s['within5_before']:6d} -> {s['within5_after']:6d}")
        print(f"   better on {s['improved']}, worse on {s['worsened']}")

    (OUT / "pose_geo_compare.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1:2]
    if mode == ["cams"]:
        write_refined_cams(sys.argv[2] if len(sys.argv) > 2 else "pose_fit.json")
    elif mode == ["geo"]:
        geo_compare()
    else:
        holdout(sys.argv[1:])
