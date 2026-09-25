"""How much night does a star calibration need? Solve one camera on windows of a whole night.

Every production solve uses one block of ~90 minutes (nights.fetch takes Q1's first 90
frames; FIgLib sequences run ~80). Nothing in the solver caps a track's length -- the block
does. This takes a whole night (nights.fetch_night: ~7.5 dark hours), tiles it into
non-overlapping windows of 15 minutes to the full night, and runs the production solver
(`solve.solve_wide`) on each window's tracks alone, answering three things:

  1. repeatability -- how far apart the poses and lenses from disjoint windows of the same
     length land;
  2. prediction -- how well a window's pose predicts where the stars are at *other* times of
     the night (held-out residuals, binned by time from the window's centre and by altitude);
  3. the sky model -- the solver has done without precession and refraction (catalog.MODEL),
     which a 90-minute block absorbs into the pose; a night may not. Every window is solved
     under the current model and with both terms on; windows of 90 min, 180 min and the full
     night also with each term alone.

Held-out residuals need the stars identified on every track, not just the window's: that is
the reference solve's job -- the full night with both terms on -- and its star <-> track
matches are what every window is scored against. The production pose for the same camera
(the committed Q1 90-frame solve, current model) is scored the same way, for "how wrong is
today's pose at 4 am".

Results, one JSON per solve so an interrupted run resumes: out/sky/data/window_ablation/<seq>/.

    python -m src.figlib.stars.window_ablation hpwren_20260713_N_vo-n-mobo-c [...]
    python -m src.figlib.stars.window_ablation --summary
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import catalog as SG
from . import solve as S
from .fisheye import initial_k, project_fisheye

OUT = S.SKY / "data/window_ablation"
LENGTHS_MIN = (15, 30, 60, 90, 180, 360)          # plus the full night
MODEL_LENGTHS = (90, 180, None)                   # None = the full night
# Proper motion was not in the catalog when this study ran (2026-09-24), so every model here
# holds it off; that keeps BASE the pre-2026-09 solver exactly.
MODELS = [dict(proper_motion=False, precession=p, refraction=r) for p in (False, True) for r in (False, True)]
BASE = dict(proper_motion=False, precession=False, refraction=False)
REF = dict(proper_motion=False, precession=True, refraction=True)
DT_BINS_MIN = (0, 30, 60, 120, 240, 600)
ALT_BINS = (0, 5, 10, 20, 40, 90)


def _key(window, model) -> str:
    m = "".join(k[0] for k, v in sorted(model.items()) if v and k != "proper_motion") or "none"
    return f"w{int(window[0])}_{int(window[1])}_m{m}"


def night_span(seq: str) -> tuple[int, int]:
    raw, _ = S.load_tracks(seq)
    offs = [o for t in raw for o in t]
    return min(offs), max(offs) + 1


def windows(seq: str) -> list[tuple[int, int, int | None]]:
    """(start, end, length_min) for every window: each length tiled from dusk, then the night."""
    lo, hi = night_span(seq)
    out = []
    for L in LENGTHS_MIN:
        for a in range(lo, hi - L * 60 + 1, L * 60):
            out.append((a, a + L * 60, L))
    out.append((lo, hi, None))
    return out


def jobs(seq: str) -> list[tuple]:
    js = [(seq, a, b, L, m) for a, b, L in windows(seq) for m in (BASE, REF)]
    js += [(seq, a, b, L, m) for a, b, L in windows(seq) if L in MODEL_LENGTHS
           for m in MODELS if m not in (BASE, REF)]
    return js


def _solve_one(job) -> dict:
    seq, a, b, L, model = job
    path = OUT / seq / f"{_key((a, b), model)}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = time.time()
    try:
        with SG.using(**model):
            r = S.solve_wide(seq, window=(a, b))
    except Exception as exc:   # one window shouldn't sink the sweep
        r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    r.update(window=[a, b], length_min=L, sky_model=model, seconds=round(time.time() - t, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=1, default=float) + "\n")
    return r


# ---- scoring on held-out stars ---------------------------------------------------------------

def _predict(cam, W, H, pose: dict, names, epochs, model: dict):
    """Pixel positions of catalog stars at epochs through a solved pose, under a sky model."""
    kpx = pose["k_ratio"] * initial_k(cam, W)
    alt = np.empty(len(names)); az = np.empty(len(names))
    with SG.using(**model):
        for i, (n, e) in enumerate(zip(names, epochs)):
            alt[i], az[i] = SG.altaz(*SG.STARS[n], e, cam["lat"], cam["lon"], cam.get("elev") or 1600.0)
    x, y = project_fisheye(cam, az, alt, W, H, pose["d_az"], pose["d_pitch"], pose["d_roll"], kpx, pose["k1"])
    return x * W, y * H, alt


def reference_points(seq: str, ref: dict):
    """Every point of every track the reference solve matched: (names, epochs, xy)."""
    raw, _ = S.load_tracks(seq)
    t0 = S.SEQS[seq]["t0"]
    names, ep, xy = [], [], []
    for n, j in ref["matches"].items():
        for o, v in raw[j].items():
            names.append(n); ep.append(t0 + o); xy.append(v[:2])
    return np.array(names), np.array(ep, float), np.array(xy, float)


def score(cam, W, H, pose, model, pts, in_window: np.ndarray, centre_epoch: float) -> dict:
    names, ep, xy = pts
    x, y, alt = _predict(cam, W, H, pose, names, ep, model)
    res = np.hypot(x - xy[:, 0], y - xy[:, 1])
    dt = np.abs(ep - centre_epoch) / 60
    out = {"in_window_median_px": float(np.median(res[in_window])) if in_window.any() else None,
           "held_out_median_px": float(np.median(res[~in_window])) if (~in_window).any() else None,
           "held_out_p90_px": float(np.percentile(res[~in_window], 90)) if (~in_window).any() else None,
           "by_dt": [], "by_alt": []}
    for lo, hi in zip(DT_BINS_MIN[:-1], DT_BINS_MIN[1:]):
        m = (dt >= lo) & (dt < hi)
        out["by_dt"].append([lo, hi, int(m.sum()), float(np.median(res[m])) if m.any() else None])
    for lo, hi in zip(ALT_BINS[:-1], ALT_BINS[1:]):
        m = (alt >= lo) & (alt < hi)
        out["by_alt"].append([lo, hi, int(m.sum()), float(np.median(res[m])) if m.any() else None])
    return out


def boresight(cam, pose) -> tuple[float, float, float]:
    return ((cam["az"] + cam.get("yaw", 0.0) + pose["d_az"]) % 360,
            (cam.get("pitch") or 0.0) + pose["d_pitch"], (cam.get("roll") or 0.0) + pose["d_roll"])


def summarize(seq: str) -> dict:
    s = S.SEQS[seq]
    cam = S.CAMS[s["camera"]]
    _, (W, H) = S.load_tracks(seq)
    lo, hi = night_span(seq)
    runs = [json.loads(p.read_text()) for p in sorted((OUT / seq).glob("*.json"))]
    for r in runs:   # results from before proper motion existed record no flag for it
        r["sky_model"] = {"proper_motion": False, **r["sky_model"]}
    ref = next(r for r in runs if r["length_min"] is None and r["sky_model"] == REF)
    assert ref["status"] == "solved", ref.get("reason")
    pts = reference_points(seq, ref)
    rows = []
    for r in runs:
        row = {k: r.get(k) for k in ("window", "length_min", "sky_model", "status", "n_stars",
                                       "median_px", "rmse_px", "seconds", "found_by", "n_tracks")}
        if r["status"] == "solved":
            a, b = r["window"]
            inw = (pts[1] >= s["t0"] + a) & (pts[1] < s["t0"] + b)
            row.update(pose=r["pose"], boresight=boresight(cam, r["pose"]),
                       **score(cam, W, H, r["pose"], r["sky_model"], pts, inw, s["t0"] + (a + b) / 2))
        rows.append(row)

    # today's production pose: the Q1 90-frame solve, current model, scored on the whole night
    nxt = (datetime.strptime(s["day"], "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    q1 = f"hpwren_{nxt}_Q1_{s['camera']}"
    prod = None
    qp = S.DATA / f"solve_{q1}.json"
    if qp.exists() and q1 in S.SEQS:
        qr = json.loads(qp.read_text())
        qs = S.SEQS[q1]
        qd = S.ROOT / qs["dir"]
        eps = sorted(int(p.stem) for p in qd.glob("*.jpg"))
        inw = (pts[1] >= eps[0]) & (pts[1] <= eps[-1])
        prod = {"seq": q1, "pose": qr["pose"], "boresight": boresight(cam, qr["pose"]),
                "window_epochs": [eps[0], eps[-1]],
                **score(cam, W, H, qr["pose"], BASE, pts, inw, (eps[0] + eps[-1]) / 2)}

    # repeatability per length and sky model: spread of disjoint windows' solutions
    spread = []
    for model in (BASE, REF):
        for L in (*LENGTHS_MIN, None):
            ok = [r for r in rows if r["length_min"] == L and r["sky_model"] == model and r["status"] == "solved"]
            n_all = sum(r["length_min"] == L and r["sky_model"] == model for r in rows)
            if not ok:
                spread.append({"length_min": L, "sky_model": model, "solved": 0, "windows": n_all})
                continue
            b = np.array([r["boresight"] for r in ok]); k = np.array([r["pose"]["k_ratio"] for r in ok])
            k1 = np.array([r["pose"]["k1"] for r in ok])
            az = (b[:, 0] - b[0, 0] + 180) % 360 - 180 + b[0, 0]
            ho = [r["held_out_median_px"] for r in ok if r["held_out_median_px"] is not None]
            spread.append({"length_min": L, "sky_model": model, "solved": len(ok), "windows": n_all,
                           "az_sd": float(az.std()), "el_sd": float(b[:, 1].std()), "roll_sd": float(b[:, 2].std()),
                           "k_sd": float(k.std()), "k1_sd": float(k1.std()),
                           "k_mean": float(k.mean()), "k1_mean": float(k1.mean()),
                           "median_n_stars": float(np.median([r["n_stars"] for r in ok])),
                           "held_out_median_px": float(np.median(ho)) if ho else None})
    out = {"seq": seq, "camera": s["camera"], "night_offsets": [lo, hi],
           "night_hours": (hi - lo) / 3600, "reference": {k: ref[k] for k in ("pose", "n_stars", "median_px")},
           "production_q1": prod, "spread": spread, "runs": rows}
    (OUT / f"summary_{seq}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


if __name__ == "__main__":
    if "--summary" in sys.argv:
        for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
            sm = summarize(d.name)
            print(f"\n{d.name}  ({sm['night_hours']:.1f} h)")
            for sp in sm["spread"]:
                L = sp["length_min"] or "night"
                tag = "P+R" if sp["sky_model"] == REF else "cur"
                if not sp["solved"]:
                    print(f"  {tag} {L!s:>5}   0/{sp['windows']:2d} solved")
                    continue
                print(f"  {tag} {L!s:>5}  {sp['solved']:2d}/{sp['windows']:2d} solved  "
                      f"sd az {sp['az_sd']:.3f} el {sp['el_sd']:.3f} roll {sp['roll_sd']:.3f} deg  "
                      f"k {sp['k_mean']:.4f}+-{sp['k_sd']:.4f}  held-out median {sp['held_out_median_px'] or float('nan'):.2f} px")
            if sm["production_q1"]:
                pq = sm["production_q1"]
                print(f"  production Q1 pose: held-out median {pq['held_out_median_px']:.2f} px, "
                      f"p90 {pq['held_out_p90_px']:.2f} px; by |dt| {[b[3] and round(b[3], 2) for b in pq['by_dt']]}")
        sys.exit()
    seqs = sys.argv[1:]
    all_jobs = [j for s in seqs for j in jobs(s)]
    # longest windows first: they are the slow ones, and the reference is among them
    all_jobs.sort(key=lambda j: -(j[2] - j[1]))
    print(f"{len(all_jobs)} window solves over {len(seqs)} nights", flush=True)
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(_solve_one, all_jobs), 1):
            w = r.get("window") or [0, 0]
            print(f"  [{i}/{len(all_jobs)}] {r['seq']} {r.get('length_min') or 'night'} "
                  f"{w} {r['sky_model']} -> {r['status']} {r.get('n_stars', '-')} stars "
                  f"{r.get('median_px') or float('nan'):.2f} px  {r.get('seconds')} s", flush=True)
