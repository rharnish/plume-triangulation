"""The DEM skyline against the image's own ridge edge, under each sky model the solver can use.

A check that shares nothing with the star catalog. Each camera's night is solved again under
every combination of the sky-model corrections (catalog.MODEL). The DEM skyline is then drawn
through each resulting pose onto a daytime frame from the same day, so the camera cannot have
moved in between. The residual in each column is the row of the strongest sky-to-terrain edge
minus the predicted row, taken only where the skyline is a ridge 10-78 km out. Nearer
ridges are tree-lined, and the DEM puts their top somewhere inside the canopy.

The edge is searched within +-SEARCH_PX of the mean of every model's line, so no one model
is favoured. The same window also pulls every residual toward zero, so the comparison
between models is the result, not the absolute size.

`ridge_feet edges` is the older, ledger-only version: one best vertical shift of the whole
skyline per solve. This one measures per column, on far ridges only, under every sky model.

What it can see: pitch and roll (a row offset), and the lens. Not much azimuth: a 0.2 deg
azimuth change moves a ridge ~6 px sideways, which only shows where the ridge slopes.

    python -m src.figlib.stars.skyline_check solve                # cached in out/sky/data/skyline_check/
    python -m src.figlib.stars.skyline_check measure              # -> out/sky/data/skyline_check/summary.json
    python -m src.figlib.stars.skyline_check figure [camera ...]  # -> docs/figures/skyline_sky_model*.{png,jpg}

The daytime frames come from the HPWREN CDN (data/hpwren_nights/<cam>/<day>_Q4, gitignored).
They leave its public window about 90 days after capture.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .. import pose_ledger
from ..terrain import Dem, horizon
from . import catalog as SG
from . import nights
from . import solve as S

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "figures"
OUT = S.SKY / "data" / "skyline_check"
CAMS = json.loads((ROOT / "data/meta/cams.json").read_text())

# Legacy first: the solver as it was before 2026-09-25, then each correction alone, then all.
MODELS = {
    "none": dict(proper_motion=False, precession=False, refraction=False),
    "refraction": dict(proper_motion=False, precession=False, refraction=True),
    "precession": dict(proper_motion=False, precession=True, refraction=False),
    "prec+refr": dict(proper_motion=False, precession=True, refraction=True),
    "all": dict(proper_motion=True, precession=True, refraction=True),
}
MIN_RANGE_KM = 10.0
# terrain.horizon marches rays to 80 km; a skyline "at" 80 km is the march running out (sea,
# or ground beyond the DEM's reach), not a ridge, and there is nothing in the image to meet it.
MAX_RANGE_KM = 78.0
SEARCH_PX = 25
FISHEYE_PAD_DEG = 14.0          # as fig_peaks: the fisheye reaches ~53 deg off-axis


def targets() -> list[dict]:
    """Per camera, the newest 3072-wide CDN star solve that has daytime frames from its day."""
    best = {}
    for e in pose_ledger.load(pose_ledger.LEDGER):
        if e.get("frame_w") != 3072 or not e["source"].startswith("star:hpwren_"):
            continue
        seq = e["source"].split(":", 1)[1]
        day = seq.split("_")[1]
        frames = sorted((nights.FRAMES / e["camera"] / f"{day}_Q4").glob("*.jpg"))
        if frames and (e["camera"] not in best or e["epoch"] > best[e["camera"]]["epoch"]):
            best[e["camera"]] = {"camera": e["camera"], "seq": seq, "epoch": e["epoch"],
                                 "frames": [str(p) for p in frames]}
    return sorted(best.values(), key=lambda t: t["camera"])


def _solve(job) -> dict:
    seq, name, wide = job
    dest = OUT / f"{seq}__{name}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    with SG.using(**MODELS[name]):
        try:
            r = S.solve_wide(seq) if wide else S.solve(seq)
            if r["status"] != "solved" and not wide:   # as resolve_ledger
                r = S.solve_wide(seq)
        except Exception as exc:
            r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    r["model_name"] = name
    dest.write_text(json.dumps(r, indent=1, default=float) + "\n")
    return r


def solve_all(cameras=None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {r["seq"]: r for r in json.loads((S.DATA / "solve_summary.json").read_text())}
    ts = [t for t in targets() if not cameras or t["camera"] in cameras]
    jobs = [(t["seq"], m, "found_by" in summary.get(t["seq"], {})) for t in ts for m in MODELS]
    print(f"{len(ts)} cameras x {len(MODELS)} sky models = {len(jobs)} solves", flush=True)
    with Pool(4) as pool:
        for r in pool.imap_unordered(_solve, jobs):
            p = r.get("pose", {})
            print(f"  {r['seq']:40s} {r.get('model_name', '?'):11s} {r['status']:8s} "
                  f"d_az {p.get('d_az', float('nan')):+.3f}  pitch {p.get('d_pitch', float('nan')):+.3f}",
                  flush=True)


def poses(seq: str) -> dict[str, dict]:
    out = {}
    for m in MODELS:
        p = OUT / f"{seq}__{m}.json"
        r = json.loads(p.read_text()) if p.exists() else {}
        if r.get("status") == "solved":
            out[m] = r["pose"]
    return out


def pick_frame(t: dict):
    """The same-day daytime frame showing the most sky-to-terrain skyline."""
    from ..viz_terrain import observed_skyline
    best = None
    for p in t["frames"]:
        img = cv2.imread(p)
        if img is None or img.shape[1] != 3072:
            continue
        cov = float(np.isfinite(observed_skyline(img)).mean())
        if best is None or cov > best[0]:
            best = (cov, p, img)
    return (best[1], best[2]) if best else (None, None)


def skyline(camera: str, pose: dict, W: int, H: int, dem: Dem):
    from ..fig_peaks import solved_lens
    cam, fn = solved_lens(CAMS[camera], pose)
    prof = horizon(cam, dem, half_fov_pad=FISHEYE_PAD_DEG, step_deg=0.05)
    x, y = fn(prof.az_deg, prof.elev_deg, W, H)
    k = np.isfinite(y) & (x >= 0) & (x < 1)
    return x[k] * W, y[k] * H, prof.range_km[k]


def edges(img: np.ndarray, cols: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Row of the strongest bright-above, dark-below edge within +-SEARCH_PX of `centre`, or NaN
    where nothing stands out from the sky above it (haze, cloud, a featureless column)."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    drop = g[:-1] - g[1:]                      # > 0 where it gets darker going down
    H = g.shape[0]
    out = np.full(len(cols), np.nan)
    for i, (c, m) in enumerate(zip(cols, centre)):
        a, b = int(m) - SEARCH_PX, int(m) + SEARCH_PX + 1
        if a < 60 or b >= H - 1:
            continue
        col = drop[a:b, c]
        floor = np.median(np.abs(drop[a - 60:a, c]))
        j = int(np.argmax(col))
        if col[j] > 4 * floor + 1:
            out[i] = a + j + 0.5
    return out


def measure_one(t: dict, dem: Dem) -> dict | None:
    ps = poses(t["seq"])
    if "none" not in ps or "all" not in ps:
        return None
    path, img = pick_frame(t)
    if img is None:
        return None
    H, W = img.shape[:2]
    lines = {m: skyline(t["camera"], p, W, H, dem) for m, p in ps.items()}
    lo = int(max(l[0].min() for l in lines.values())) + 5
    hi = int(min(l[0].max() for l in lines.values())) - 5
    cols = np.arange(lo, hi, 2)
    rows = {m: np.interp(cols, l[0], l[1]) for m, l in lines.items()}
    rng = np.interp(cols, lines["all"][0], lines["all"][2])
    e = edges(img, cols, np.mean(list(rows.values()), axis=0))
    k = np.isfinite(e) & (rng > MIN_RANGE_KM) & (rng < MAX_RANGE_KM)
    when = datetime.fromtimestamp(int(Path(path).stem), ZoneInfo("America/Los_Angeles"))
    res = {m: e[k] - r[k] for m, r in rows.items()}
    return {"camera": t["camera"], "seq": t["seq"], "frame": str(Path(path).relative_to(ROOT)),
            "frame_local": f"{when:%Y-%m-%d %H:%M}", "n_cols": int(k.sum()),
            "median_px": {m: float(np.median(v)) for m, v in res.items()} if k.sum() else {},
            "mean_abs_px": {m: float(np.mean(np.abs(v))) for m, v in res.items()} if k.sum() else {},
            "pose": ps}


def measure(cameras=None, min_cols: int = 100) -> list[dict]:
    from .. import provenance as P
    started = P.utc_now()
    dem = Dem()
    rows = []
    for t in targets():
        if cameras and t["camera"] not in cameras:
            continue
        r = measure_one(t, dem)
        if r is None:
            print(f"{t['camera']:16s} no solve or frame")
            continue
        rows.append(r)
        med = r["median_px"]
        print(f"{r['camera']:16s} {r['n_cols']:5d} cols  " + "  ".join(
            f"{m} {med[m]:+5.1f}" for m in MODELS if m in med), flush=True)
    used = [r for r in rows if r["n_cols"] >= min_cols]
    agg = {}
    for m in MODELS:
        v = [r["median_px"][m] for r in used if m in r["median_px"]]
        if v:
            agg[m] = {"n_cameras": len(v), "median_of_medians_px": float(np.median(v)),
                      "mean_abs_of_medians_px": float(np.mean(np.abs(v)))}
    print(f"\n{len(used)} cameras with >= {min_cols} usable columns (median over cameras of each "
          f"camera's median edge - line, px; + = the line sits above the ridge)")
    for m, a in agg.items():
        print(f"  {m:11s} median {a['median_of_medians_px']:+5.2f}   mean |.| {a['mean_abs_of_medians_px']:4.2f}")
    if "none" in agg and "all" in agg:
        closer = sum(abs(r["median_px"]["all"]) < abs(r["median_px"]["none"]) for r in used)
        print(f"  'all' closer to the ridge than 'none' on {closer} of {len(used)} cameras")
    dest = OUT / "summary.json"
    dest.write_text(json.dumps({"min_cols": min_cols, "min_range_km": MIN_RANGE_KM,
                                "max_range_km": MAX_RANGE_KM,
                                "search_px": SEARCH_PX, "aggregate": agg, "cameras": rows},
                               indent=1, default=float) + "\n")
    P.record("skyline_check", [dest], started=started,
             params={"min_cols": min_cols, "min_range_km": MIN_RANGE_KM,
                     "max_range_km": MAX_RANGE_KM, "search_px": SEARCH_PX,
                     "models": MODELS},
             extra_inputs=[pose_ledger.LEDGER, SG.CATALOG_PATH])
    return rows


# ----------------------------------------------------------------------------- figures

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BEFORE, AFTER = MUTED, "#2a78d6"
MODEL_LABEL = {"none": "no corrections (before)", "refraction": "refraction only",
               "precession": "precession only", "prec+refr": "precession + refraction",
               "all": "all three (now)"}


def fig_residuals(summary: dict, dest=DOCS / "skyline_sky_model.png") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    used = [r for r in summary["cameras"] if r["n_cols"] >= summary["min_cols"]]
    used.sort(key=lambda r: r["median_px"]["none"])
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 0.24 * len(used) + 1.9),
                                 gridspec_kw={"width_ratios": [3, 1.35]})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)
        a.tick_params(length=0, labelsize=8, colors=MUTED)
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
    y = np.arange(len(used))
    b0 = [r["median_px"]["none"] for r in used]
    b1 = [r["median_px"]["all"] for r in used]
    for yy, u, v in zip(y, b0, b1):
        ax.plot([u, v], [yy, yy], color=AXIS, lw=1, zorder=1)
    ax.scatter(b0, y, s=44, color=BEFORE, edgecolors=SURFACE, linewidths=2, zorder=3,
               label=MODEL_LABEL["none"])
    ax.scatter(b1, y, s=44, color=AFTER, edgecolors=SURFACE, linewidths=2, zorder=4,
               label=MODEL_LABEL["all"])
    ax.axvline(0, color=AXIS, lw=1, zorder=0)
    ax.set_yticks(y, [r["camera"] for r in used], fontsize=8, color=INK2)
    ax.set_ylim(-0.7, len(used) - 0.3)
    ax.set_xlabel("ridge edge in the image minus DEM skyline, median px (− = line below the ridge)",
                  color=INK2, fontsize=9)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK2, fontsize=9)
    ax.set_title(f"Per camera: star pose before → after (ridges {summary['min_range_km']:.0f}–"
                 f"{summary['max_range_km']:.0f} km)", loc="left", color=INK, fontsize=11)

    agg = summary["aggregate"]
    ms = [m for m in MODELS if m in agg]
    vals = [agg[m]["mean_abs_of_medians_px"] for m in ms]
    bx.barh(np.arange(len(ms)), vals, height=0.5,
            color=[AFTER if m == "all" else BEFORE if m == "none" else AXIS for m in ms])
    for i, (m, v) in enumerate(zip(ms, vals)):
        bx.text(v + 0.15, i, f"{v:.1f}  (median {agg[m]['median_of_medians_px']:+.1f})", va="center",
                ha="left", fontsize=8, color=INK2)
    bx.set_yticks(np.arange(len(ms)), [MODEL_LABEL[m] for m in ms], fontsize=8, color=INK2)
    bx.invert_yaxis()
    bx.set_xlim(0, max(vals) * 1.9)
    bx.set_xlabel("mean |camera median|, px", color=INK2, fontsize=9)
    bx.set_title(f"All {len(used)} cameras, by sky model", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return dest


def fig_zooms(summary: dict, cameras: list[str], dest=DOCS / "skyline_sky_model_zooms.jpg",
              cw: int = 320, ch: int = 150, up: int = 3) -> Path:
    """Native-pixel crops at the three highest far ridges per camera, both lines drawn."""
    dem = Dem()
    by = {r["camera"]: r for r in summary["cameras"]}
    cols_bgr = {"none": (255, 0, 255), "all": (0, 165, 255)}
    rows = []
    for camera in cameras:
        r = by[camera]
        img = cv2.imread(str(ROOT / r["frame"]))
        H, W = img.shape[:2]
        lines = {m: skyline(camera, r["pose"][m], W, H, dem) for m in ("none", "all")}
        # Crop where the image shows a ridge edge (the columns the residual is measured on),
        # at the highest such points, spread across the frame.
        cols = np.arange(cw, W - cw, 2)
        rows_all = {m: np.interp(cols, l[0], l[1]) for m, l in lines.items()}
        rng = np.interp(cols, lines["all"][0], lines["all"][2])
        found = np.isfinite(edges(img, cols, (rows_all["none"] + rows_all["all"]) / 2)) & (rng > MIN_RANGE_KM) & (rng < MAX_RANGE_KM)
        # a crop needs mostly measured columns, not one lucky one
        dense = np.convolve(found, np.ones(61) / 61, mode="same") > 0.7
        picks = []
        for i in np.argsort(rows_all["all"]):
            if dense[i] and all(abs(cols[i] - cols[j]) > W / 5 for j in picks):
                picks.append(i)
            if len(picks) == 3:
                break
        tiles = []
        for i in sorted(picks, key=lambda i: cols[i]):
            km = float(rng[i])
            x0, y0 = int(cols[i]) - cw // 2, max(0, int(rows_all["all"][i]) - ch // 2)
            crop = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (cw * up, ch * up), interpolation=cv2.INTER_CUBIC)
            for m in ("none", "all"):
                xs, ys, _ = lines[m]
                k = (xs >= x0 - 2) & (xs < x0 + cw + 2)
                pts = np.stack([(xs[k] - x0) * up, (ys[k] - y0) * up], 1).astype(np.int32)
                cv2.polylines(crop, [pts], False, cols_bgr[m], 2, cv2.LINE_AA)
            cv2.rectangle(crop, (0, 0), (96, 34), (0, 0, 0), -1)
            cv2.putText(crop, f"{km:.0f} km", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
            tiles.append(crop)
        while len(tiles) < 3:
            tiles.append(np.zeros((ch * up, cw * up, 3), np.uint8))
        sep = np.zeros((ch * up, 10, 3), np.uint8)
        body = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]])
        bar = np.zeros((40, body.shape[1], 3), np.uint8)
        med = r["median_px"]
        cv2.putText(bar, f"{camera}   {r['frame_local']}, same day as its star solve   "
                    f"median edge - line: {med['none']:+.1f} px before, {med['all']:+.1f} px now",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, body]))
    legend = np.zeros((44, rows[0].shape[1], 3), np.uint8)
    cv2.putText(legend, "DEM skyline, native pixels x3.   magenta = star pose, no sky-model corrections"
                "   orange = with proper motion, precession and refraction", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, (230, 230, 230), 2, cv2.LINE_AA)
    sheet = np.vstack([legend] + rows)
    cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dest


def figures(cameras=None) -> None:
    from .. import provenance as P
    started = P.utc_now()
    summary = json.loads((OUT / "summary.json").read_text())
    a = fig_residuals(summary)
    used = [r for r in summary["cameras"] if r["n_cols"] >= summary["min_cols"]]
    # By default the four cameras with the most usable ridge, not the four that look best.
    cameras = cameras or [r["camera"] for r in sorted(used, key=lambda r: -r["n_cols"])[:4]]
    b = fig_zooms(summary, cameras)
    print(f"wrote {a}\nwrote {b}  ({', '.join(cameras)})")
    P.record("skyline_check.figure", [a, b], started=started, params={"cameras": cameras},
             extra_inputs=[OUT / "summary.json"])


if __name__ == "__main__":
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "solve":
        solve_all(rest or None)
    elif cmd == "measure":
        measure(rest or None)
    elif cmd == "figure":
        figures(rest or None)
    else:
        print(__doc__)
