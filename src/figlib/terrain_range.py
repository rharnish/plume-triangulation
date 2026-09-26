"""How far along a bearing can the fire be? A two-sided range from one early smoke box and the DEM.

A bearing is a direction. The early box bottom adds an elevation angle: the lowest smoke the
detector saw. Walking out along the ray, the DEM gives at every distance `d` the lowest line
of sight that still clears the terrain in front (`run(d)`: the source itself where visible,
the crest in front where not), and the star-measured lens turns that angle into an image row.
Two conditions follow:

* **cap** -- the box bottom cannot sit *below* that row: smoke hidden behind a crest can't be
  drawn. `y1 <= row(d) + SLACK_PX`. Validated before (NOTES.md 2026-09-13, "Plume feet").
* **band** -- nor *far above* it: a newly detected plume's lowest visible smoke is near the
  terrain line it rises from. `row(d) <= y1 + BAND_PX`.

The row falls monotonically with distance, so the band cuts the near end of the ray and the
cap the far end: one interval. A bound in metres ("the foot is at most h m above the terrain
line") was tried first and does not work as a near bound -- the height a fixed angular gap
implies grows with `d`, so it never rules out the near end (NOTES.md, 2026-09-14).

    FIGLIB_CORPUS=all python -m src.figlib.terrain_range validate            # containment vs band
    FIGLIB_CORPUS=all python -m src.figlib.terrain_range solve [N|cap] [ledger]
    FIGLIB_CORPUS=all python -m src.figlib.terrain_range figure <fire_id> [N]

`solve` is opt-in: `geolocate` does not use the term. Only star-posed 3072 px cameras get a
range: without a measured lens and pitch, a row is not an angle.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from . import corpus as C
from . import settings
from . import pose_ledger
from . import provenance as P
from . import terrain as T
from .fig_triangulate import _det_for
from .geolocate import (FRAME_SIZES, META, YOLO_DIR, bearings_for_fire, credible_area_km2,
                        refine, refine_step, solve)
from .geom import angdiff_deg, bearing_deg, enu_grid, haversine_km, load_cams, ray_latlon
from .stars.fisheye import initial_k, project_fisheye

OUT = C.current().out / "terrain_range"

CONF_THR = 0.25                 # bearings_for_fire's default, so the box matches the bearing
EARLY_S, N_EARLY = 900, 3       # box bottoms from the first fifteen minutes
SLACK_PX = 20.0                 # the slack the cap was validated with
MAX_KM, STEP_M = 80.0, 30.0
# The near-side bound: the terrain line's row may lie at most this far below the early box
# bottom. 100 px lost no truth the cap kept on 73 validated bearings (validate, 2026-09-14);
# narrower bands start dropping truths.
BAND_PX = float(settings.get("FIGLIB_BAND_PX"))
BAND_GRID = (30, 60, 100, 150, 200, 300)
STEP_KM = 0.4                   # geolocate.solve's default grid, which the penalty is added on
NEAR_SAMPLES = 10       # truth is "contained" within this many samples (0.3 km): the official
                        # point and the DEM ray are each good to a few hundred metres
MISS_MAX_DEG = 5.0      # validation: along-ray distance only means something for a close bearing
FLOOR = 0.05            # solve: a ruled-out cell costs log(0.05), a bounded penalty, not a veto
FAN_DEG = 8.0           # solve: cells this far either side of the bearing take its profile

_DEM: T.Dem | None = None


# ------------------------------------------------------------------ geometry

def along_ray(lat: float, lon: float, az_deg: float, km):
    """Ground points `km` along a bearing, on the camera's line of sight (`geom.ray_latlon`)."""
    return ray_latlon(lat, lon, az_deg, np.asarray(km, float) * 1000.0)


def full_pose(camera: str, epoch: float, frame_w: int | None) -> dict | None:
    """Star pose (azimuth, pitch, roll, lens) for the range, or None.

    The ledger's own rules first. Where they decline -- no solve within a year, or two solves
    that disagree -- fall back to the nearest solve in time, as `ridge_feet.pose_for` does for
    eyeballing: most single-site fires predate every solve, and the row of a crest barely
    depends on azimuth. `source` keeps the two apart so a range from a years-away pose is never
    scored as if it were calibrated; `gap_days` says how far away it was.
    """
    if frame_w != 3072:
        return None
    entries = pose_ledger.load()
    hit = pose_ledger.lookup(entries, camera, epoch, frame_w)
    if hit is not None:
        src = [e for e in entries if e["source"] in hit["sources"]]
        source, rule = "ledger", hit["rule"]
    else:
        mine = [e for e in entries if e["camera"] == camera and e.get("frame_w") == frame_w]
        if not mine:
            return None
        src = [min(mine, key=lambda e: abs(e["epoch"] - epoch))]
        source, rule = "nearest", None
    gap = min(abs(e["epoch"] - epoch) for e in src) / 86400.0
    mean = lambda k: float(np.mean([e[k] for e in src]))
    return {"d_az": mean("d_az") if hit is None else hit["d_az"], "d_pitch": mean("d_pitch"),
            "d_roll": mean("d_roll"), "k_ratio": mean("k_ratio"), "k1": mean("k1"),
            "source": source, "rule": rule or f"nearest solve, {gap:.0f} d away",
            "gap_days": round(gap)}


def early_box_bottom(seq_name: str, det: dict, H: int) -> float | None:
    """Median bottom row (px) of the earliest boxes overlapping the chosen box's columns."""
    path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
    if not path.exists():
        return None
    ys = []
    for rec in sorted(json.loads(path.read_text()), key=lambda r: r["offset"]):
        if not (0 <= rec["offset"] <= EARLY_S):
            continue
        for d in rec["dets"]:
            if d["conf"] >= CONF_THR and d["x1"] >= det["x0"] and d["x0"] <= det["x1"]:
                ys.append(d["y1"] * H)
                break
        if len(ys) >= N_EARLY:
            break
    return float(np.median(ys)) if ys else None


def profile(cam: dict, pose: dict, az_deg: float, W: int, H: int, min_km: float = 0.0) -> dict:
    """Terrain along the ray: distance (m), lowest clearing angle (deg), and its image row (px).

    `min_km` stops terrain nearer than that from hiding anything beyond it: a 30 m surface
    model there is mostly the canopy and buildings around the mast. Off by default.
    Raises `ValueError` when the camera's height disagrees with the DEM at its own site.
    """
    global _DEM
    if _DEM is None:
        _DEM = T.Dem()
    d = np.arange(STEP_M, MAX_KM * 1000.0, STEP_M)
    # A window around the ray alone, half the size of one around the camera. Windows snap to
    # the tiles' pixel grid, so the samples don't depend on which window they came from.
    mid_lat, mid_lon = along_ray(cam["lat"], cam["lon"], az_deg, MAX_KM / 2)
    band, tf = _DEM.window(float(mid_lat), float(mid_lon), MAX_KM / 2 / 111.0 + 0.08)
    why = T.Dem.site_problem(cam, band, tf)
    if why:
        raise ValueError(why)
    lats, lons = along_ray(cam["lat"], cam["lon"], az_deg, d / 1000.0)
    ang = T.sight_angles(T.Dem.sample(band, tf, lats, lons), T.eye_height(cam), d)
    far = d >= min_km * 1000.0
    run = ang.copy()                       # highest terrain angle up to each distance
    run[far] = np.maximum.accumulate(ang[far])
    k = pose["k_ratio"] * initial_k(cam, W)
    _x, y = project_fisheye(cam, np.full_like(run, az_deg), run, W, H,
                            pose["d_az"], pose["d_pitch"], pose["d_roll"], k, pose["k1"])
    return {"d": d, "run": run, "rows": y * H, "horizon_km": float(d[np.argmax(ang)] / 1000.0)}


def allowed(prof: dict, y1: float, band_px: float | None = BAND_PX) -> np.ndarray:
    """Boolean mask over `prof["d"]`: distances the early box bottom `y1` is consistent with.

    `band_px=None` applies the cap alone.
    """
    ok = y1 <= prof["rows"] + SLACK_PX
    if band_px is not None:
        ok = ok & (prof["rows"] <= y1 + band_px)
    return ok


def summarise_mask(d_m: np.ndarray, ok: np.ndarray) -> dict:
    """Nearest and farthest allowed distance, and the allowed length, in km."""
    if not ok.any():
        return {"lo_km": None, "hi_km": None, "len_km": 0.0}
    idx = np.flatnonzero(ok)
    return {"lo_km": round(float(d_m[idx[0]]) / 1000, 2), "hi_km": round(float(d_m[idx[-1]]) / 1000, 2),
            "len_km": round(float(ok.sum() * STEP_M / 1000), 2)}


def contains(ok: np.ndarray, along_km: float) -> bool:
    """Does the mask allow a distance within NEAR_SAMPLES of `along_km`?"""
    i = int(np.clip(round(along_km * 1000 / STEP_M) - 1, 0, len(ok) - 1))
    return bool(ok[max(0, i - NEAR_SAMPLES):i + NEAR_SAMPLES + 1].any())


# ------------------------------------------------------------------ inputs

def _load():
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = [r for r in json.loads((META / "resolved.json").read_text())
                if r["tier"] in ("confirmed", "probable")
                and (r.get("truth") or {}).get("lat") is not None]
    return cams, seqs, fires, resolved


def _inputs() -> list:
    return [META / "sequences.json", META / "fires.json", META / "resolved.json", YOLO_DIR,
            pose_ledger.path()]


def bearing_terrain(b, fire: dict, seqs: dict, cams: dict) -> dict | None:
    """Pose, box bottom and profile for one bearing, or None when terrain can't be used."""
    got = _det_for(b, fire, seqs)
    if got is None:
        return None
    seq_name, det = got
    W, H = (FRAME_SIZES.get(seq_name.split("#")[0]) or [None, None])[:2]
    pose = full_pose(b.camera, b.epoch, W)
    if pose is None:
        return None
    y1 = early_box_bottom(seq_name, det, H)
    if y1 is None:
        return None
    cam = cams[b.camera]
    try:
        prof = profile(cam, pose, b.bearing_deg, W, H)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        print(f"  skip {b.camera}: {exc}", flush=True)
        return None
    return {"cam": cam, "pose": pose, "W": W, "H": H, "y1": y1, "prof": prof, "b": b}


# ------------------------------------------------------------------ validate

def validate() -> dict:
    """Does the allowed set contain the truth's distance along the ray, and how long is it?"""
    started = P.utc_now()
    cams, seqs, fires, resolved = _load()
    rows = []
    for r in resolved:
        fire, t = fires[r["fire_id"]], r["truth"]
        for b in bearings_for_fire(fire, seqs, cams, use_wind=False):
            cam = cams[b.camera]
            miss = angdiff_deg(bearing_deg(cam["lat"], cam["lon"], t["lat"], t["lon"]), b.bearing_deg)
            if abs(miss) > MISS_MAX_DEG:
                continue
            bt = bearing_terrain(b, fire, seqs, cams)
            if bt is None:
                continue
            along = haversine_km(cam["lat"], cam["lon"], t["lat"], t["lon"]) * math.cos(math.radians(miss))
            prof = bt["prof"]
            i = int(np.clip(round(along * 1000 / STEP_M) - 1, 0, len(prof["d"]) - 1))
            rec = {"fire_id": r["fire_id"], "tier": r["tier"], "triangulable": bool(r.get("triangulable")),
                   "camera": b.camera, "pose": bt["pose"]["source"], "gap_days": bt["pose"]["gap_days"],
                   "along_km": round(along, 2), "horizon_km": round(prof["horizon_km"], 2),
                   "terrain_row_minus_box_px_at_truth": round(float(prof["rows"][i] - bt["y1"]))}
            ok = allowed(prof, bt["y1"], None)
            rec["cap_only"] = {"contains": contains(ok, along), **summarise_mask(prof["d"], ok)}
            rec["by_band"] = {}
            for bp in BAND_GRID:
                ok = allowed(prof, bt["y1"], bp)
                rec["by_band"][bp] = {"contains": contains(ok, along), **summarise_mask(prof["d"], ok)}
            rows.append(rec)
            print(f"{r['fire_id']:32s} {b.camera:16s} {rec['pose']:7s} along {along:5.1f} km  "
                  f"terrain row - box {rec['terrain_row_minus_box_px_at_truth']:5d} px", flush=True)

    report = {"n_bearings": len(rows), "slack_px": SLACK_PX, "by_pose": {}}
    for pose in ("ledger", "nearest", "all"):
        sub = [x for x in rows if pose == "all" or x["pose"] == pose]
        if not sub:
            continue
        def stats(c):
            return {"contains": sum(v["contains"] for v in c),
                    "median_len_km": round(float(np.median([v["len_km"] for v in c])), 2),
                    "median_len_over_truth": round(float(np.median(
                        [v["len_km"] / x["along_km"] for v, x in zip(c, sub) if x["along_km"] > 0])), 2)}
        report["by_pose"][pose] = {
            "n": len(sub),
            "cap_only": stats([x["cap_only"] for x in sub]),
            "per_band_px": {bp: stats([x["by_band"][bp] for x in sub]) for bp in BAND_GRID},
            "terrain_row_minus_box_px_pctl": [int(np.percentile(
                [x["terrain_row_minus_box_px_at_truth"] for x in sub], p)) for p in (5, 25, 50, 75, 95)]}
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "validate.json"
    dest.write_text(json.dumps({"report": report, "rows": rows}, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    print(f"-> {dest}")
    P.record("terrain_range.validate", [dest], started=started, extra_inputs=_inputs(),
             params={"band_grid": BAND_GRID, "slack_px": SLACK_PX, "miss_max_deg": MISS_MAX_DEG,
                     "step_m": STEP_M, "max_km": MAX_KM, "early_s": EARLY_S, "n_early": N_EARLY})
    return report


# ------------------------------------------------------------------ solve

def terrain_loglik(lats, lons, terr: list[dict], band_px: float | None) -> np.ndarray:
    """Log penalty over the grid: log(FLOOR) where a nearby bearing's terrain rules the cell out."""
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    total = np.zeros_like(LA)
    for bt in terr:
        cam, b = bt["cam"], bt["b"]
        ok = allowed(bt["prof"], bt["y1"], band_px)
        de, dn = enu_grid(cam["lat"], cam["lon"], LA, LO)
        dist = np.hypot(dn, de)
        rel = (np.degrees(np.arctan2(de, dn)) - b.bearing_deg + 180.0) % 360.0 - 180.0
        idx = np.clip(np.rint(dist / STEP_M).astype(int) - 1, 0, len(ok) - 1)
        pen = np.where(ok[idx], 0.0, math.log(FLOOR))
        pen[(np.abs(rel) > FAN_DEG) | (dist > MAX_KM * 1000)] = 0.0
        total += pen
    return total


def solve_compare(band_px: float | None = BAND_PX, ledger_only: bool = False) -> list[dict]:
    """Calibrated triangulation with and without the terrain term, on the triangulable fires.

    `band_px=None` uses the cap alone. `ledger_only` drops terrain from nearest-solve poses,
    which are years away on most fires.
    """
    started = P.utc_now()
    cams, seqs, fires, resolved = _load()
    tag = ("cap" if band_px is None else f"band_px{int(band_px)}") + ("_ledger" if ledger_only else "")
    rows = []
    for r in resolved:
        if not r.get("triangulable"):
            continue
        fire, t = fires[r["fire_id"]], r["truth"]
        bs = bearings_for_fire(fire, seqs, cams, use_wind=False)
        if len({b.camera.split("-")[0] for b in bs}) < 2:
            continue
        center = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
        lats, lons, ll, elat, elon = solve(bs, center)
        before = haversine_km(elat, elon, t["lat"], t["lon"])
        terr = [x for x in (bearing_terrain(b, fire, seqs, cams) for b in bs) if x is not None
                and (not ledger_only or x["pose"]["source"] == "ledger")]
        rec = {"fire_id": r["fire_id"], "tier": r["tier"], "n_sites": len({b.camera.split('-')[0] for b in bs}),
               "n_terrain": len(terr), "terrain_pose": sorted({x["pose"]["source"] for x in terr}),
               "before_km": round(before, 2), "area_before": round(credible_area_km2(lats, lons, ll), 1)}
        if terr:
            ll2 = ll + terrain_loglik(lats, lons, terr, band_px)
            i, j = np.unravel_index(np.argmax(ll2), ll2.shape)
            alat, alon = float(lats[i]), float(lons[j])
            if refine_step():
                alat, alon = refine(bs, alat, alon, STEP_KM,
                                    extra=lambda la, lo: terrain_loglik(la, lo, terr, band_px))
            rec["after_km"] = round(haversine_km(alat, alon, t["lat"], t["lon"]), 2)
            rec["area_after"] = round(credible_area_km2(lats, lons, ll2), 1)
        else:
            rec["after_km"], rec["area_after"] = rec["before_km"], rec["area_before"]
        rows.append(rec)
        print(f"{r['fire_id']:32s} {r['tier']:9s} {rec['n_sites']}s terrain {rec['n_terrain']} "
              f"{','.join(rec['terrain_pose']) or '-':14s} {rec['before_km']:6.2f} -> {rec['after_km']:6.2f} km  "
              f"area {rec['area_before']:6.1f} -> {rec['area_after']:6.1f}", flush=True)

    summary = {}
    for tier in ("confirmed", "probable"):
        sub = [x for x in rows if x["tier"] == tier]
        touched = [x for x in sub if x["n_terrain"]]
        summary[tier] = {
            "n": len(sub), "with_terrain": len(touched),
            "median_km_before": round(float(np.median([x["before_km"] for x in sub])), 2),
            "median_km_after": round(float(np.median([x["after_km"] for x in sub])), 2),
            "within_2km_before": sum(x["before_km"] <= 2 for x in sub),
            "within_2km_after": sum(x["after_km"] <= 2 for x in sub),
            "better": sum(x["after_km"] < x["before_km"] - 0.05 for x in touched),
            "worse": sum(x["after_km"] > x["before_km"] + 0.05 for x in touched),
            "median_area_before": round(float(np.median([x["area_before"] for x in sub])), 1),
            "median_area_after": round(float(np.median([x["area_after"] for x in sub])), 1),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"solve_{tag}.json"
    dest.write_text(json.dumps({"band_px": band_px, "ledger_only": ledger_only,
                                "summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    P.record("terrain_range.solve", [dest], started=started, extra_inputs=_inputs(),
             params={"band_px": band_px, "ledger_only": ledger_only, "slack_px": SLACK_PX,
                     "floor": FLOOR, "fan_deg": FAN_DEG})
    return rows


def before_after(fire_id: str, band_px: float = BAND_PX):
    """One fire, triangulated without and with the terrain range, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .fig_triangulate import PALETTE, _hillshade, _plume_crop

    started = P.utc_now()
    cams, seqs, fires, resolved = _load()
    r = next(x for x in resolved if x["fire_id"] == fire_id)
    fire, t = fires[fire_id], r["truth"]
    bs = sorted(bearings_for_fire(fire, seqs, cams, use_wind=False), key=lambda b: -b.conf)
    center = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
    lats, lons, ll, e0lat, e0lon = solve(bs, center)
    err0 = haversine_km(e0lat, e0lon, t["lat"], t["lon"]); area0 = credible_area_km2(lats, lons, ll)
    terr = [x for x in (bearing_terrain(b, fire, seqs, cams) for b in bs) if x is not None]
    ll1 = ll + terrain_loglik(lats, lons, terr, band_px)
    i, j = np.unravel_index(np.argmax(ll1), ll1.shape)
    e1lat, e1lon = float(lats[i]), float(lons[j])
    if refine_step():
        e1lat, e1lon = refine(bs, e1lat, e1lon, STEP_KM,
                              extra=lambda la, lo: terrain_loglik(la, lo, terr, band_px))
    err1 = haversine_km(e1lat, e1lon, t["lat"], t["lon"]); area1 = credible_area_km2(lats, lons, ll1)
    colors = {b.camera: c for b, c in zip(bs, PALETTE)}

    # Per camera: allowed interval and the truth's distance along the ray.
    info = {}
    for bt in terr:
        b, cam = bt["b"], bt["cam"]
        ok = allowed(bt["prof"], bt["y1"], band_px)
        miss = angdiff_deg(bearing_deg(cam["lat"], cam["lon"], t["lat"], t["lon"]), b.bearing_deg)
        along = haversine_km(cam["lat"], cam["lon"], t["lat"], t["lon"]) * math.cos(math.radians(miss))
        info[b.camera] = {"ok": ok, "along": along, **summarise_mask(bt["prof"]["d"], ok), "bt": bt}

    # A shared view over cameras, truth and both estimates.
    pts = [(b.lat, b.lon) for b in bs] + [(t["lat"], t["lon"]), (e0lat, e0lon), (e1lat, e1lon)]
    la = [p[0] for p in pts]; lo = [p[1] for p in pts]
    clat, clon = (max(la) + min(la)) / 2, (max(lo) + min(lo)) / 2
    cos_lat = math.cos(math.radians(clat))
    half_km = max(9.0, 0.6 * max((max(la) - min(la)) * 111.32, (max(lo) - min(lo)) * 111.32 * cos_lat))
    dlat, dlon = half_km / 111.32, half_km / (111.32 * cos_lat)
    hs, extent = _hillshade(clat, clon, half_km)

    fig = plt.figure(figsize=(17.0, 12.6), dpi=125)
    fig.patch.set_facecolor("#11131a")
    gs = fig.add_gridspec(2, 4, height_ratios=[2.35, 1.0], hspace=0.16, wspace=0.22,
                          left=0.035, right=0.985, top=0.905, bottom=0.085)

    def draw_map(ax, surface, est, err, area, title, with_terrain):
        ax.set_facecolor("#11131a")
        ax.set_xlim(clon - dlon, clon + dlon); ax.set_ylim(clat - dlat, clat + dlat)
        if hs is not None:
            ax.imshow(hs, origin="lower", extent=extent, cmap="gray", vmin=-0.15, vmax=1.25,
                      alpha=0.55, aspect="auto", zorder=0)
        rel = surface - surface.max()
        a = np.clip((rel + 9.0) / 9.0, 0.0, 1.0)
        rgba = matplotlib.colormaps["magma"](a); rgba[..., 3] = 0.85 * a ** 1.6
        ax.imshow(rgba, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
                  aspect="auto", zorder=2, interpolation="bilinear")
        ax.contour(lons, lats, rel, levels=[-3.0], colors=["#bfe0ff"], linewidths=1.3, zorder=3)
        ranges = []
        for b in bs:
            col = colors[b.camera]
            end = along_ray(b.lat, b.lon, b.bearing_deg, half_km * 3)
            if with_terrain and b.camera in info and info[b.camera]["lo_km"] is not None:
                s = info[b.camera]
                p0 = along_ray(b.lat, b.lon, b.bearing_deg, s["lo_km"])
                p1 = along_ray(b.lat, b.lon, b.bearing_deg, s["hi_km"])
                dashed = dict(color=col, lw=1.2, ls=(0, (4, 4)), alpha=0.75, zorder=4)
                ax.plot([b.lon, p0[1]], [b.lat, p0[0]], **dashed)
                ax.plot([p1[1], end[1]], [p1[0], end[0]], **dashed)
                ax.plot([p0[1], p1[1]], [p0[0], p1[0]], color=col, lw=4.0, zorder=5, solid_capstyle="butt")
                ranges.append((col, f"{b.camera}: {s['lo_km']:.1f}–{s['hi_km']:.1f} km "
                                     f"(official point {s['along']:.1f} km)"))
            else:
                ax.plot([b.lon, end[1]], [b.lat, end[0]], color=col, lw=1.8, alpha=0.95, zorder=4)
            ax.plot(b.lon, b.lat, marker="^", color=col, ms=11, mec="#11131a", mew=1.2, zorder=6)
            ax.annotate(b.camera, (b.lon, b.lat), textcoords="offset points", xytext=(10, 7),
                        fontsize=8.5, color=col, weight="bold", zorder=6)
        if ranges:
            ax.text(0.985, 0.03, "distance the terrain allows", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=9, color="#dfe3ea", zorder=10,
                    bbox=dict(facecolor="#181b24", edgecolor="#3a4050", alpha=0.92, pad=5))
            for n, (col, txt) in enumerate(reversed(ranges)):
                ax.text(0.985, 0.075 + 0.04 * n, txt, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, color=col, weight="bold", zorder=10)
        ax.plot(t["lon"], t["lat"], "o", mfc="none", mec="#ffffff", ms=20, mew=2.4, zorder=8)
        ax.plot(est[1], est[0], "x", color="#ff3860", ms=15, mew=3.4, zorder=9)
        ax.plot([est[1], t["lon"]], [est[0], t["lat"]], color="#ff3860", lw=1.0, ls=(0, (2, 2)), zorder=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#3a4050")
        ax.set_title(f"{title}\nerror {err:.2f} km      95% region {area:.1f} km²",
                     color="#f2f4f8", fontsize=12.5, pad=8)
        bar_km = max(2, int(round(half_km / 2.5)))
        bx0, by = clon - dlon * 0.92, clat - dlat * 0.93
        ax.plot([bx0, bx0 + bar_km / (111.32 * cos_lat)], [by, by], color="#e8e8e8", lw=3, zorder=9)
        ax.text(bx0, by + dlat * 0.03, f"{bar_km} km", color="#e8e8e8", fontsize=9)

    ax0 = fig.add_subplot(gs[0, 0:2]); ax1 = fig.add_subplot(gs[0, 2:4])
    draw_map(ax0, ll, (e0lat, e0lon), err0, area0, "Bearings only (star lens + pose ledger)", False)
    draw_map(ax1, ll1, (e1lat, e1lon), err1, area1, f"Bearings + terrain range ({band_px:.0f} px band)", True)
    ax0.legend(handles=[
        Line2D([], [], color="#8d93a3", lw=1.8, label="bearing"),
        Line2D([], [], color="#8d93a3", lw=4, label="distance the terrain allows"),
        Line2D([], [], color="#8d93a3", lw=1.2, ls=(0, (4, 4)), label="ruled out by terrain"),
        Line2D([], [], color="#bfe0ff", lw=1.3, label="95% credible contour"),
        Line2D([], [], color="#ff3860", marker="x", ls="none", ms=10, mew=3, label="estimate"),
        Line2D([], [], color="#ffffff", marker="o", mfc="none", ls="none", ms=12, mew=2, label="official ignition point"),
    ], loc="upper left", fontsize=9, facecolor="#181b24", edgecolor="#3a4050", labelcolor="#dfe3ea", framealpha=0.92)

    # Bottom row: for each camera, what it saw and why terrain bounds its distance.
    for k, b in enumerate(bs[:2]):
        col = colors[b.camera]
        cax = fig.add_subplot(gs[1, 2 * k])
        cax.set_facecolor("#11131a")
        got = _det_for(b, fire, seqs)
        crop = _plume_crop(got[0], b.camera, b.epoch, got[1]) if got else None
        if crop is not None:
            cax.imshow(crop)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_color(col); sp.set_linewidth(2.4)
        cax.set_title(f"{b.camera}   conf {b.conf:.2f}   bearing {b.bearing_deg:.1f}°", color=col, fontsize=9.5, pad=4)

        pax = fig.add_subplot(gs[1, 2 * k + 1])
        pax.set_facecolor("#11131a")
        s = info.get(b.camera)
        if s is None:
            pax.text(0.5, 0.5, "no star pose / early box", color="#9aa3b2", ha="center", transform=pax.transAxes)
            pax.set_xticks([]); pax.set_yticks([])
            continue
        bt = s["bt"]; km = bt["prof"]["d"] / 1000; rows = bt["prof"]["rows"]; y1 = bt["y1"]
        xmax = min(MAX_KM, max(s["along"], s["hi_km"] or 0) * 1.35 + 3)
        keep = km <= xmax
        pax.plot(km[keep], rows[keep], color="#c8cedb", lw=1.4)
        pax.axhspan(y1 - SLACK_PX, y1 + band_px, color=col, alpha=0.22, lw=0)
        pax.axhline(y1, color=col, lw=1.6)
        if s["lo_km"] is not None:
            pax.axvspan(0, s["lo_km"], color="#ff3860", alpha=0.12, lw=0)
            pax.axvspan(s["hi_km"], xmax, color="#ff3860", alpha=0.12, lw=0)
        pax.axvline(s["along"], color="#ffffff", lw=1.3, ls=(0, (3, 3)))
        lo_r, hi_r = float(rows[keep].min()), float(rows[keep].max())
        pad = max(40.0, 0.12 * (hi_r - lo_r))
        pax.set_ylim(max(hi_r, y1 + band_px) + pad, min(lo_r, y1 - SLACK_PX) - pad)
        pax.set_xlim(0, xmax)
        pax.tick_params(colors="#9aa3b2", labelsize=7.5)
        for sp in pax.spines.values():
            sp.set_color("#3a4050")
        pax.set_xlabel("km along the bearing   (white dashed: official point)", color="#9aa3b2", fontsize=8)
        pax.set_ylabel("image row (px)", color="#9aa3b2", fontsize=8)
        pose = bt["pose"]
        pax.set_title(f"terrain line vs early box bottom (band {band_px:.0f} px)\n"
                      f"pose: {pose['rule']}", color=col, fontsize=8.8, pad=4)

    acres = t.get("acres")
    fig.suptitle(f"{fire_id}  →  {t['name']}" + (f"  ({acres} acres)" if acres else "")
                 + f"      error {err0:.2f} km  →  {err1:.2f} km      [{r['tier']}]",
                 color="#f2f4f8", fontsize=15.5, y=0.975)
    # Caption from this fire's own numbers, so the figure can't carry another fire's story.
    two = [bs[0]] + [b for b in bs[1:] if b.camera.split("-")[0] != bs[0].camera.split("-")[0]][:1]
    cross = abs(angdiff_deg(two[0].bearing_deg, two[1].bearing_deg)) if len(two) == 2 else None
    if cross is None:
        geometry = "One site."
    elif cross < 25 or cross > 155:
        geometry = (f"The bearings are within {max(1.0, min(cross, 180 - cross)):.0f}° of collinear, so bearings alone "
                    "slide the estimate along the shared line.")
    else:
        geometry = (f"The bearings cross at {cross:.0f}°; the likelihood still peaks where one camera's error "
                    "is cheapest, not at the fire.")
    val = OUT / "validate.json"
    keeps = ""
    if val.exists():
        rep = json.loads(val.read_text())["report"]["by_pose"]["all"]
        bk = rep["per_band_px"].get(str(int(band_px)))
        if bk:
            keeps = (f"; a {band_px:.0f} px band keeps {bk['contains']} of {rep['n']} validated truths "
                     f"(the cap alone {rep['cap_only']['contains']})")
    poses = "; ".join(f"{b}: {s['bt']['pose']['rule']}" for b, s in info.items())
    fig.text(0.5, 0.012,
             f"{geometry} Each early box bottom must sit on the terrain line at the fire's distance, "
             "which bounds each camera's range along its bearing.\n"
             f"Poses behind the terrain ranges: {poses or 'none'}{keeps}.",
             color="#9aa3b2", fontsize=9, ha="center", va="bottom", linespacing=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{fire_id}_terrain_band{int(band_px)}.png"
    fig.savefig(dest, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"{err0:.2f} -> {err1:.2f} km, area {area0:.1f} -> {area1:.1f} km2  ->  {dest}")
    P.record("terrain_range.figure", [dest], started=started, extra_inputs=_inputs() + [val],
             params={"fire_id": fire_id, "band_px": band_px, "error_km": [round(err0, 2), round(err1, 2)]})
    return dest


if __name__ == "__main__":
    settings.default_profile("calibrated")   # configs/calibrated.toml unless FIGLIB_PROFILE is set
    a = sys.argv[1:]
    if a and a[0] == "validate":
        validate()
    elif a and a[0] == "solve":
        # solve [N|cap] [ledger]: band N px (default FIGLIB_BAND_PX), or the cap alone;
        # "ledger" uses terrain only from ledger poses
        rest = [x for x in a[1:] if x != "ledger"]
        band = BAND_PX if not rest else None if rest[0] == "cap" else float(rest[0])
        solve_compare(band, ledger_only="ledger" in a)
    elif a and a[0] == "figure" and len(a) > 1:
        before_after(a[1], float(a[2]) if len(a) > 2 else BAND_PX)
    else:
        print(__doc__)
