"""Recover camera pose from terrain, rather than from a calibration routine.

HPWREN publishes each camera's azimuth to the degree and says nothing about pitch, roll,
or lens distortion. The skyline audit (`viz_terrain.main`) shows the published numbers are
not good enough to trust: across 61 cameras the terrain-predicted horizon misses the
observed one by a median of 106 px, and by 674 px at worst.

There is no calibration target on a mountaintop and no way to send anyone to hold one. But
the scene itself is a target -- the ridgeline is a rigid, surveyed object whose geometry is
published as a 30 m elevation model. So pose is recoverable from the imagery and the world,
which is the same trick as recovering an inertial sensor's frame from gravity and motion
priors instead of a calibration jig.

Four parameters are fitted per camera against the observed skyline:

* **d_az** -- the one that matters. Azimuth error moves the horizon *sideways*, and
  bearing-to-fire is read off horizontal position, so this is the term that propagates
  into kilometres of geolocation error. The audit's vertical residual does not measure it.
* **d_pitch** -- moves the horizon up or down. Large here, and largely harmless: it costs
  nothing in bearing.
* **d_roll** -- tilts it.
* **k1** -- one radial distortion coefficient. Frames from some units show visibly bowed
  horizons; if that is real, a rectilinear pixel-to-bearing map carries error that grows
  toward frame edges, which is where detections often sit.

The loss is Huber rather than least squares because the observed skyline is extracted from
pixels and will contain cloud banks and haze that no pose explains. Squared error lets a
hundred bad columns drag the fit; Huber lets them be outliers, which is the same argument
that shaped the detection likelihood in `accumulate.py`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .terrain import Dem, horizon, project
from .viz_terrain import observed_skyline, _best_frame

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"

# Below this fraction of columns yielding a skyline, the extraction failed and the fit
# would be to noise. Monochrome units fail here by construction: the sky test is blue
# dominance, which grayscale imagery cannot express.
MIN_COVERAGE = 0.35

# Generous, but bounded by what the published metadata could plausibly be wrong by.
BOUNDS = dict(d_az=6.0, d_pitch=12.0, d_roll=6.0, k1=0.35)
HUBER_PX = 25.0


def _distort(x: np.ndarray, y: np.ndarray, k1: float, aspect: float):
    """One-parameter radial model about the principal point, in fractional coords."""
    if k1 == 0.0:
        return x, y
    dx, dy = x - 0.5, (y - 0.5) * aspect
    r2 = dx * dx + dy * dy
    s = 1.0 + k1 * r2
    return 0.5 + dx * s, 0.5 + (dy * s) / aspect


def _predicted_rows(cam: dict, prof, W: int, H: int, p) -> np.ndarray:
    d_az, d_pitch, d_roll, k1 = p
    cam2 = {**cam, "az": cam["az"] + d_az}
    x, y = project(cam2, prof.az_deg, prof.elev_deg, W, H, d_pitch, d_roll)
    x, y = _distort(x, y, k1, H / W)
    xs = x * W
    keep = np.isfinite(xs) & np.isfinite(y)
    if keep.sum() < 8:
        return np.full(W, np.nan)
    order = np.argsort(xs[keep])
    return np.interp(np.arange(W), xs[keep][order], (y * H)[keep][order],
                     left=np.nan, right=np.nan)


def _huber(r: np.ndarray, delta: float = HUBER_PX) -> float:
    a = np.abs(r)
    return float(np.mean(np.where(a <= delta, 0.5 * r * r,
                                  delta * (a - 0.5 * delta))))


def fit_camera(cam_name: str, img: np.ndarray, cams: dict, dem: Dem,
               fit_k1: bool = True) -> dict | None:
    from scipy.optimize import minimize

    cam = cams[cam_name]
    H, W = img.shape[:2]
    obs = observed_skyline(img)
    cov = float(np.isfinite(obs).mean())
    if cov < MIN_COVERAGE:
        return dict(camera=cam_name, status="low coverage", coverage=round(cov, 3))

    prof = horizon(cam, dem)

    def loss(p):
        pred = _predicted_rows(cam, prof, W, H, p)
        m = np.isfinite(pred) & np.isfinite(obs)
        if m.sum() < W * 0.2:
            return 1e9
        # Columns the model cannot cover are not free: without this, shifting the
        # horizon out of frame would look like a perfect fit.
        return _huber((pred - obs)[m]) + 4.0 * HUBER_PX * (1.0 - m.mean())

    p0 = np.zeros(4)
    bounds = [(-BOUNDS["d_az"], BOUNDS["d_az"]),
              (-BOUNDS["d_pitch"], BOUNDS["d_pitch"]),
              (-BOUNDS["d_roll"], BOUNDS["d_roll"]),
              (-BOUNDS["k1"], BOUNDS["k1"]) if fit_k1 else (0.0, 0.0)]
    best = None
    # Multi-start: the loss is not convex in azimuth -- a ridgeline can align on the
    # wrong summit -- so seed several azimuth offsets rather than trusting one descent.
    for az0 in (-4.0, -2.0, 0.0, 2.0, 4.0):
        r = minimize(loss, np.array([az0, 0.0, 0.0, 0.0]), method="Powell",
                     bounds=bounds, options=dict(maxiter=4000, xtol=1e-3, ftol=1e-3))
        if best is None or r.fun < best.fun:
            best = r

    p = best.x
    pred0 = _predicted_rows(cam, prof, W, H, np.zeros(4))
    pred1 = _predicted_rows(cam, prof, W, H, p)
    m0 = np.isfinite(pred0) & np.isfinite(obs)
    m1 = np.isfinite(pred1) & np.isfinite(obs)
    return dict(
        camera=cam_name, status="fitted", coverage=round(cov, 3),
        d_az=round(float(p[0]), 3), d_pitch=round(float(p[1]), 3),
        d_roll=round(float(p[2]), 3), k1=round(float(p[3]), 4),
        before_median_px=round(float(np.median((pred0 - obs)[m0])), 1) if m0.any() else None,
        before_mad_px=round(float(np.median(np.abs((pred0 - obs)[m0]))), 1) if m0.any() else None,
        after_median_px=round(float(np.median((pred1 - obs)[m1])), 1) if m1.any() else None,
        after_mad_px=round(float(np.median(np.abs((pred1 - obs)[m1]))), 1) if m1.any() else None,
        loss=round(float(best.fun), 2))


def main(argv: list[str]) -> None:
    cams = json.loads((META / "cams.json").read_text())
    seqs = json.loads((META / "sequences.json").read_text())
    by_cam: dict[str, dict] = {}
    for s in seqs:
        if s["has_pose"] and s["camera"] not in by_cam:
            by_cam[s["camera"]] = s
    if argv:
        by_cam = {k: v for k, v in by_cam.items() if any(a in k for a in argv)}

    dem = Dem()
    rows = []
    for k, (name, seq) in enumerate(sorted(by_cam.items()), 1):
        img = _best_frame(seq)
        if img is None:
            continue
        try:
            r = fit_camera(name, img, cams, dem)
        except Exception as exc:
            print(f"[{k}/{len(by_cam)}] {name}: FAIL {exc}")
            continue
        if r is None:
            continue
        r["seq"] = seq["seq"]
        rows.append(r)
        if r["status"] == "fitted":
            print(f"[{k}/{len(by_cam)}] {name:22s} d_az {r['d_az']:+6.2f} "
                  f"d_pitch {r['d_pitch']:+6.2f} d_roll {r['d_roll']:+5.2f} "
                  f"k1 {r['k1']:+.3f}   MAD {r['before_mad_px']:6.1f} -> "
                  f"{r['after_mad_px']:5.1f} px", flush=True)
        else:
            print(f"[{k}/{len(by_cam)}] {name:22s} {r['status']} "
                  f"(coverage {r['coverage']})", flush=True)

    dest = ROOT / "out" / "pose_fit.json"
    dest.write_text(json.dumps(rows, indent=1) + "\n")
    ok = [r for r in rows if r["status"] == "fitted"]
    if ok:
        b = np.array([r["before_mad_px"] for r in ok])
        a = np.array([r["after_mad_px"] for r in ok])
        az = np.array([abs(r["d_az"]) for r in ok])
        print(f"\n{len(ok)} cameras fitted, {len(rows) - len(ok)} skipped")
        print(f"skyline MAD: median {np.median(b):.0f} -> {np.median(a):.0f} px")
        print(f"|d_az|: median {np.median(az):.2f} deg  p90 {np.percentile(az, 90):.2f} "
              f"max {az.max():.2f}")
        print(f"cameras needing >1 deg of azimuth correction: {(az > 1).sum()}/{len(ok)}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])


# ---------------------------------------------------------------------------
# Second attempt. The dense fit above improves the skyline residual four-fold and
# makes geolocation *worse* -- median error 2.16 -> 3.38 km on held-out fires. The
# reason is measurable: the loss moves ~0.5 per degree of azimuth against ~50 per
# degree of pitch, an 80:1 ratio. A ridgeline is nearly horizontal, so sliding it
# sideways barely changes it, and azimuth is close to unidentifiable. Given free
# rein, the optimizer spends azimuth on noise and runs to its bounds.
#
# The information about azimuth lives in the *distinctive* parts of the skyline --
# peaks, notches, saddles -- not in its overall height. Differentiating along the
# horizontal axis keeps exactly those and suppresses the flat stretches that carry
# nothing. So azimuth is estimated separately, by correlating skyline gradients, and
# accepted only where the correlation actually picks a shift out.

AZ_SEARCH_DEG = 3.0
AZ_STEP_DEG = 0.1
MIN_PEAK_CORR = 0.35        # below this the ridgeline is too bland to localize
MIN_PROMINENCE = 0.12       # peak must stand above the rest of the search, not just be its max


def _grad(v: np.ndarray, sigma: float = 101.0) -> np.ndarray:
    """Smoothed horizontal derivative, with gaps left as NaN."""
    out = np.full_like(v, np.nan, dtype=float)
    m = np.isfinite(v)
    if m.sum() < 32:
        return out
    x = np.arange(len(v))
    filled = np.interp(x, x[m], v[m])
    k = max(3, int(sigma) | 1)   # ridge features span tens to hundreds of px; at
                                 # sigma=9 the derivative was extraction jitter, which
                                 # correlates with nothing (measured: NCC 0.011 where the
                                 # underlying rows correlated at 0.910)
    sm = cv2.GaussianBlur(filled.reshape(-1, 1).astype(np.float32), (1, k), 0).ravel()
    g = np.gradient(sm)
    g[~m] = np.nan
    return g


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 64:
        return -1.0
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-9 else -1.0


def fit_azimuth(cam: dict, prof, obs: np.ndarray, W: int, H: int,
                p_shape: np.ndarray) -> dict:
    """Azimuth from ridgeline features, with an honest identifiability test."""
    g_obs = _grad(obs)
    azs = np.arange(-AZ_SEARCH_DEG, AZ_SEARCH_DEG + 1e-9, AZ_STEP_DEG)
    corr = np.array([_ncc(_grad(_predicted_rows(
        cam, prof, W, H, np.array([a, p_shape[1], p_shape[2], p_shape[3]]))), g_obs)
        for a in azs])

    i = int(np.argmax(corr))
    best, peak = float(azs[i]), float(corr[i])
    # Prominence against the rest of the search, excluding a window around the peak:
    # a broad plateau means many shifts explain the ridge equally well.
    mask = np.abs(azs - best) > 1.0
    background = float(np.median(corr[mask])) if mask.any() else 0.0
    prominence = peak - background
    ok = peak >= MIN_PEAK_CORR and prominence >= MIN_PROMINENCE
    return dict(d_az=round(best, 2), peak_corr=round(peak, 3),
                prominence=round(prominence, 3), identifiable=bool(ok),
                curve=[round(float(c), 3) for c in corr])


def fit_camera_staged(cam_name: str, img: np.ndarray, cams: dict, dem: Dem) -> dict:
    """Shape first with azimuth held fixed, then azimuth from features alone.

    Splitting the two is the whole point: pitch, roll and distortion are well determined
    by the skyline's height and curvature, and letting azimuth float alongside them lets
    it absorb their residual noise.
    """
    from scipy.optimize import minimize

    cam = cams[cam_name]
    H, W = img.shape[:2]
    obs = observed_skyline(img)
    cov = float(np.isfinite(obs).mean())
    if cov < MIN_COVERAGE:
        return dict(camera=cam_name, status="low coverage", coverage=round(cov, 3))

    prof = horizon(cam, dem)

    def loss(q):                       # q = (d_pitch, d_roll, k1); azimuth pinned to 0
        p = np.array([0.0, q[0], q[1], q[2]])
        pred = _predicted_rows(cam, prof, W, H, p)
        m = np.isfinite(pred) & np.isfinite(obs)
        if m.sum() < W * 0.2:
            return 1e9
        return _huber((pred - obs)[m]) + 4.0 * HUBER_PX * (1.0 - m.mean())

    r = minimize(loss, np.zeros(3), method="Powell",
                 bounds=[(-BOUNDS["d_pitch"], BOUNDS["d_pitch"]),
                         (-BOUNDS["d_roll"], BOUNDS["d_roll"]),
                         (-BOUNDS["k1"], BOUNDS["k1"])],
                 options=dict(maxiter=4000, xtol=1e-3, ftol=1e-3))
    p_shape = np.array([0.0, r.x[0], r.x[1], r.x[2]])
    az = fit_azimuth(cam, prof, obs, W, H, p_shape)

    p_final = np.array([az["d_az"] if az["identifiable"] else 0.0,
                        r.x[0], r.x[1], r.x[2]])
    def mad(p):
        pred = _predicted_rows(cam, prof, W, H, p)
        m = np.isfinite(pred) & np.isfinite(obs)
        return round(float(np.median(np.abs((pred - obs)[m]))), 1) if m.any() else None

    return dict(camera=cam_name, status="fitted", coverage=round(cov, 3),
                d_az=float(p_final[0]), d_pitch=round(float(r.x[0]), 3),
                d_roll=round(float(r.x[1]), 3), k1=round(float(r.x[2]), 4),
                az_raw=az["d_az"], peak_corr=az["peak_corr"],
                prominence=az["prominence"], identifiable=az["identifiable"],
                before_mad_px=mad(np.zeros(4)), after_mad_px=mad(p_final))


def main_staged(argv: list[str]) -> None:
    cams = json.loads((META / "cams.json").read_text())
    seqs = json.loads((META / "sequences.json").read_text())
    by_cam: dict[str, dict] = {}
    for s in seqs:
        if s["has_pose"] and s["camera"] not in by_cam:
            by_cam[s["camera"]] = s
    if argv:
        by_cam = {k: v for k, v in by_cam.items() if any(a in k for a in argv)}

    dem = Dem()
    rows = []
    for k, (name, seq) in enumerate(sorted(by_cam.items()), 1):
        img = _best_frame(seq)
        if img is None:
            continue
        try:
            r = fit_camera_staged(name, img, cams, dem)
        except Exception as exc:
            print(f"[{k}/{len(by_cam)}] {name}: FAIL {exc}", flush=True)
            continue
        r["seq"] = seq["seq"]
        rows.append(r)
        if r["status"] == "fitted":
            print(f"[{k}/{len(by_cam)}] {name:22s} d_az {r['d_az']:+5.2f} "
                  f"(raw {r['az_raw']:+5.2f} corr {r['peak_corr']:+.2f} "
                  f"prom {r['prominence']:+.2f} "
                  f"{'USE ' if r['identifiable'] else 'drop'})  "
                  f"pitch {r['d_pitch']:+6.2f} k1 {r['k1']:+.3f}  "
                  f"MAD {r['before_mad_px']:6.1f} -> {r['after_mad_px']:5.1f}", flush=True)

    (ROOT / "out" / "pose_fit_staged.json").write_text(json.dumps(rows, indent=1) + "\n")
    ok = [r for r in rows if r["status"] == "fitted"]
    ident = [r for r in ok if r["identifiable"]]
    if ok:
        print(f"\n{len(ok)} fitted; azimuth identifiable on {len(ident)}")
        if ident:
            a = np.array([abs(r["d_az"]) for r in ident])
            print(f"|d_az| where identifiable: median {np.median(a):.2f} deg  "
                  f"max {a.max():.2f}")
        print(f"skyline MAD: median "
              f"{np.median([r['before_mad_px'] for r in ok]):.0f} -> "
              f"{np.median([r['after_mad_px'] for r in ok]):.0f} px")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["staged"]:
        main_staged(sys.argv[2:])
    else:
        main(sys.argv[1:])
