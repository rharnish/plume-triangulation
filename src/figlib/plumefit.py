"""Four ways to get a source position out of a plume mask, instead of one order statistic.

`masks.foot_x` takes the column centroid of the mask's lowest rows, and it loses to the
box's upwind edge -- see the `foot_diff` / `foot_sam` columns in `geolocate.py`. The
reason is visible in `out/masks/*.jpg`: by the time the detector is most confident, the
plume has flattened into a horizontal sheet, and the lowest fifth of a horizontal sheet
is most of the sheet. The foot silently degenerates into the mask centroid, with extra
variance, exactly when it is asked to work hardest.

Every method here replaces that order statistic with a *fit*: a model with a source
position in it, estimated from the whole mask rather than from one band of rows.

**axis** -- weighted PCA on the mask, extrapolated down the principal axis to the terrain
row. Uses the plume's lean, and gets stronger as the plume leans further, which is where
the foot got weaker. Degenerates when the axis goes horizontal, and says so rather than
returning a number.

**wedge** -- a plume is a cone opening upward and downwind, so its width grows linearly
with height above the source and its centreline drifts linearly too. Two regressions --
width on row, centre on row -- give an apex without needing a horizon estimate at all.

**sequence** -- the fire does not move. Fit one shared apex to every frame's mask in the
sequence at once, with each frame's lean and spread free. Trades one noisy frame for
twenty or forty constrained ones, and makes drift a nuisance parameter rather than the
thing being corrected for.

**field** -- stop producing an `x` at all. For each candidate source column, ask how well
*some* physically-allowed plume rooted there could reproduce the observed mask, and hand
`geolocate.py` that curve instead of a bearing and a sigma. This is the same move the
repo already made when it stopped intersecting lines and started scoring the ground; it
just pushes it one level further down, into the pixels.

The lean is constrained to the downwind side throughout. A plume that leans upwind is not
a plume, and letting the fit choose freely would let it explain any mask from any source.
"""

from __future__ import annotations

import numpy as np

# A fit whose apex sits this far outside the mask's own column span is extrapolating
# beyond what the pixels support, and is reported as a failure rather than a guess.
MAX_EXTRAP = 1.5

# Below this, the principal axis is horizontal enough that projecting it to the terrain
# row is arithmetic on noise.
MIN_AXIS_SLOPE = 0.36  # ~20 degrees from horizontal


def _rows_extent(mask: np.ndarray):
    """Per-row left edge, right edge, centre, width and mass, for occupied rows."""
    occ = mask.any(axis=1)
    ys = np.where(occ)[0]
    if ys.size < 4:
        return None
    sub = mask[ys]
    idx = np.arange(mask.shape[1])
    left = np.array([idx[r][0] for r in sub])
    right = np.array([idx[r][-1] for r in sub])
    return ys, left, right, (left + right) / 2.0, (right - left + 1.0), sub.sum(1)


# ----------------------------------------------------------------------- axis


def axis_fit(mask: np.ndarray, horizon_y: float) -> dict:
    """Principal axis of the mask, extrapolated down to the terrain row.

    The axis is the direction the plume is travelling: up and downwind. Running it
    backwards to the ground is the geometric statement of "where did this come from".
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 50:
        return {"x": None, "why": "mask too small"}
    pts = np.stack([xs, ys]).astype(np.float64)
    mu = pts.mean(1, keepdims=True)
    cov = np.cov(pts - mu)
    w, v = np.linalg.eigh(cov)
    vx, vy = v[:, -1]                      # principal direction
    if abs(vy) < MIN_AXIS_SLOPE * abs(vx) + 1e-9:
        return {"x": None, "why": "axis near-horizontal"}
    if vy < 0:                             # point it downward, towards the ground
        vx, vy = -vx, -vy

    cx, cy = float(mu[0, 0]), float(mu[1, 0])
    target = max(horizon_y, ys.max())      # never extrapolate above the plume's own base
    x = cx + vx * (target - cy) / vy

    span = xs.max() - xs.min() + 1.0
    if not (xs.min() - MAX_EXTRAP * span <= x <= xs.max() + MAX_EXTRAP * span):
        return {"x": None, "why": "extrapolated off the plume"}
    # Elongation: a round mask has no meaningful axis even if it passes the slope test.
    elong = float(np.sqrt(max(w[-1], 1e-9) / max(w[0], 1e-9)))
    # Columns per row *upward*, matching `wedge_fit` and `sequence_fit`. (vx, vy) points
    # down the axis towards the ground, so the upward lean is its negation.
    return {"x": float(x) / mask.shape[1], "elong": elong,
            "lean": float(-vx / vy), "why": None}


# ---------------------------------------------------------------------- wedge


def wedge_fit(mask: np.ndarray, cross_sign: float | None = None) -> dict:
    """Apex of the cone that best explains the mask.

    Width grows linearly with distance above the apex and the centreline drifts
    linearly, so two weighted least-squares lines in row `y` locate the apex without any
    terrain reference: the width line's root is the apex row, and the centre line
    evaluated there is the apex column.
    """
    e = _rows_extent(mask)
    if e is None:
        return {"x": None, "why": "too few rows"}
    ys, left, right, centre, width, mass = e
    if ys.max() - ys.min() < 6:
        return {"x": None, "why": "too few rows"}

    y = ys.astype(np.float64)
    w = mass.astype(np.float64)            # thin, ragged rows carry less weight

    def wls(t):
        A = np.stack([np.ones_like(y), y]).T
        W = w[:, None]
        b, *_ = np.linalg.lstsq(A * W, t * w, rcond=None)
        return b                            # t ~= b0 + b1*y

    b0, b1 = wls(width)
    if b1 >= -1e-6:
        # Width does not shrink downward: the mask has no cone in it. This is the
        # flattened-sheet case, and it is a real answer -- there is no apex to find.
        return {"x": None, "why": "width does not converge"}
    ay = -b0 / b1                           # row where the fitted width reaches zero
    ay = float(np.clip(ay, ys.min(), ys.max() + 0.5 * (ys.max() - ys.min())))

    c0, c1 = wls(centre)
    ax = c0 + c1 * ay
    lean = -c1                              # columns per row *upward*

    span = float(right.max() - left.min() + 1.0)
    if not (left.min() - MAX_EXTRAP * span <= ax <= right.max() + MAX_EXTRAP * span):
        return {"x": None, "why": "apex off the plume"}
    if cross_sign is not None and abs(cross_sign) > 0.25 and lean * cross_sign < 0:
        return {"x": None, "why": "leans upwind"}
    return {"x": float(ax) / mask.shape[1], "apex_y": float(ay), "lean": float(lean),
            "spread": float(-b1), "why": None}


# ------------------------------------------------------------------- sequence


def sequence_fit(masks: dict[int, np.ndarray], cross_sign: float | None = None,
                 min_frames: int = 3) -> dict:
    """One apex shared by every frame; lean and spread free per frame.

    The fire is stationary and the plume is not, so the apex is the only parameter the
    frames genuinely have in common. Solving for it jointly is what turns forty noisy
    single-frame fits into one estimate: each frame contributes its centreline, and a
    single column has to sit at the foot of all of them.

    For a candidate apex row `ay`, every frame's centreline is linear in the shared
    apex column and that frame's own lean, so the whole system is one least-squares
    problem with `1 + T` unknowns. `ay` is then searched over a coarse grid.
    """
    per = []
    for _, m in sorted(masks.items()):
        e = _rows_extent(m)
        if e is None:
            continue
        ys, _, _, centre, _, mass = e
        if ys.max() - ys.min() < 6:
            continue
        per.append((ys.astype(np.float64), centre, mass.astype(np.float64), m.shape[1]))
    if len(per) < min_frames:
        return {"x": None, "why": f"only {len(per)} usable frames"}

    W = per[0][3]
    y_lo = min(p[0].min() for p in per)
    y_hi = max(p[0].max() for p in per)
    # The apex sits at or below the lowest smoke seen in any frame; searching a little
    # past that absorbs a plume whose base is hidden behind a ridge.
    grid = np.linspace(y_lo, y_hi + 0.35 * (y_hi - y_lo), 40)

    n = len(per)
    best = None
    for ay in grid:
        rows = sum(len(p[0]) for p in per)
        A = np.zeros((rows, 1 + n))
        b = np.zeros(rows)
        sw = np.zeros(rows)
        r = 0
        for j, (ys, centre, mass, _) in enumerate(per):
            k = len(ys)
            A[r:r + k, 0] = 1.0
            A[r:r + k, 1 + j] = ay - ys      # per-frame lean, in columns per row
            b[r:r + k] = centre
            sw[r:r + k] = np.sqrt(mass)
            r += k
        sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        resid = float(np.sum((sw * (A @ sol - b)) ** 2) / max(sw.sum(), 1e-9))
        if best is None or resid < best[0]:
            best = (resid, float(sol[0]), sol[1:], float(ay))

    resid, ax, leans, ay = best
    if cross_sign is not None and abs(cross_sign) > 0.25:
        # Frames must agree with the wind, not merely with each other.
        agree = float(np.mean(np.sign(leans) == np.sign(cross_sign)))
        if agree < 0.5:
            return {"x": None, "why": "sequence leans upwind"}
    if not (0 <= ax < W):
        return {"x": None, "why": "apex outside the frame"}
    return {"x": ax / W, "apex_y": ay, "n_frames": n, "resid": resid,
            "lean_med": float(np.median(leans)), "why": None}


# ---------------------------------------------------------------------- field


def column_loglik(mask: np.ndarray, cross_sign: float | None = None,
                  n_cols: int = 128, small_h: int = 96,
                  max_lean: float = 2.5, n_lean: int = 21,
                  spreads=(0.12, 0.25, 0.45, 0.75, 1.2)) -> tuple[np.ndarray, np.ndarray]:
    """How well a plume rooted at each image column could reproduce this mask.

    For every candidate source column, the best-agreeing wedge over an allowed range of
    lean and spread is found, and its agreement with the observed mask becomes that
    column's score. Lean is restricted to the downwind half when the crosswind is
    decisive, which is what stops a wedge from explaining any mask from any column.

    Returns `(x_fracs, loglik)` -- a curve over the frame's width, not a bearing and a
    sigma. `geolocate.py` reads it at whatever column each ground cell projects to, so
    the shape of the curve, including its skew and its multiple modes, survives into the
    posterior instead of being collapsed to a mean and a variance first.
    """
    import cv2
    sm = cv2.resize(mask.astype(np.uint8), (n_cols, small_h),
                    interpolation=cv2.INTER_AREA) > 0
    if not sm.any():
        return np.linspace(0, 1, n_cols), np.zeros(n_cols)

    # The source sits at or below the lowest visible smoke -- often a little below,
    # since the base of a distant plume is usually hidden behind a ridge. Putting the
    # apex one row under the mask keeps the wedge non-empty for a plume that is only a
    # few rows tall, which is the common case for a fire forty kilometres out.
    ay = float(np.nonzero(sm.any(axis=1))[0].max()) + 1.0
    yy = np.arange(small_h, dtype=np.float64)
    dy = ay - yy                                        # positive above the apex
    xx = np.arange(n_cols, dtype=np.float64)
    cols = xx.copy()

    leans = np.linspace(-max_lean, max_lean, n_lean)
    if cross_sign is not None and abs(cross_sign) > 0.25:
        leans = leans[np.sign(leans) * np.sign(cross_sign) >= 0]

    area_m = float(sm.sum())
    best = np.zeros(n_cols)
    above = dy >= 0
    for lean in leans:
        for k in spreads:
            # wedge(c): |x - (c + lean*dy)| <= k*dy, for rows above the apex
            centre = cols[:, None] + lean * dy[None, :]           # (C, H)
            half = k * dy[None, :] + 0.5      # a wedge is at least one column wide
            d = np.abs(xx[None, None, :] - centre[:, :, None])
            inside = (d <= half[:, :, None]) & above[None, :, None]
            inter = (inside & sm[None, :, :]).sum(axis=(1, 2)).astype(np.float64)
            union = (inside | sm[None, :, :]).sum(axis=(1, 2)).astype(np.float64)
            iou = inter / np.maximum(union, 1.0)
            # Recall matters more than precision: a wedge is a crude stand-in for a
            # plume and will always over-cover, but a wedge that misses smoke entirely
            # is rooted in the wrong place.
            rec = inter / max(area_m, 1.0)
            best = np.maximum(best, 0.5 * iou + 0.5 * rec)

    ll = np.log(np.maximum(best, 1e-6))
    ll -= ll.max()
    if ll.min() > -0.05:
        # Every column explains the mask equally well -- a speck a few pixels across
        # contains no wedge. A flat curve is not evidence, and saying so lets the caller
        # fall back rather than adding a uniform term to the posterior.
        return (cols + 0.5) / n_cols, np.zeros_like(ll)
    return (cols + 0.5) / n_cols, ll


# ---------------------------------------------------------------------- report


def fit_report(method: str = "diff") -> None:
    """Why each fit refuses, and how big the masks it is refusing actually are.

    The kilometre columns in `geolocate.py` say these methods lose. This says whether
    they lost or were never really tried: a variant that falls back to the box on three
    quarters of its bearings is not evidence about the fit, it is evidence about the
    masks.
    """
    import collections
    import json
    from pathlib import Path

    from .geom import load_cams
    from .masks import selected_detection, sequence_masks
    from .wind import crosswind_sign, wind_at, _load as _load_wind

    root = Path(__file__).resolve().parents[2]
    meta = root / "data" / "meta"
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((meta / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((meta / "fires.json").read_text())}
    wind_cache = _load_wind()

    tally = {k: collections.Counter() for k in ("axis", "wedge", "seq", "field")}
    sizes = []
    for r in json.loads((meta / "resolved.json").read_text()):
        if r["tier"] not in ("confirmed", "probable") or not r.get("triangulable"):
            continue
        for sn in fires[r["fire_id"]]["sequences"]:
            s = seqs.get(sn)
            if not s or not s["has_pose"]:
                continue
            sel = selected_detection(sn)
            if not sel:
                continue
            cam = cams[s["camera"]]
            w = wind_at(cam["lat"], cam["lon"], sel["epoch"], wind_cache)
            cross = crosswind_sign(cam["az"], w["from_deg_100m"]) if w else None

            c = sequence_masks(sn, method)
            ms = c["masks"]
            tally["seq"][sequence_fit(ms, cross).get("why") or "OK"] += 1
            m = ms.get(sel["offset"])
            if m is None:
                for k in ("axis", "wedge", "field"):
                    tally[k]["no mask at that frame"] += 1
                continue
            sizes.append(int(m.sum()))
            tally["axis"][axis_fit(m, c["horizon_y"]).get("why") or "OK"] += 1
            tally["wedge"][wedge_fit(m, cross).get("why") or "OK"] += 1
            _, ll = column_loglik(m, cross)
            tally["field"]["flat: no wedge in the mask" if ll.min() > -0.05
                           else "OK"] += 1

    for k, c in tally.items():
        print(f"\n{k}  ({c['OK']}/{sum(c.values())} produced a number)")
        for why, n in c.most_common():
            print(f"   {n:3d}  {why}")
    if sizes:
        a = np.array(sizes)
        print(f"\nmask size, px: median {int(np.median(a))}  "
              f"p10 {int(np.percentile(a, 10))}  p90 {int(np.percentile(a, 90))}   "
              f"(the frame is 1024x768)")


if __name__ == "__main__":
    import sys
    fit_report(sys.argv[1] if len(sys.argv) > 1 else "diff")
