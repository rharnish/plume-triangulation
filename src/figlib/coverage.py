"""Is the posterior's confidence honest, and does it stay honest as a fire develops?

`area95_km2` has been reported beside every estimate, but never checked: nothing has asked
whether the official ignition actually falls inside the 95% region 95% of the time. The
evolution study (evolve.py) showed the area stays near 3.7 km2 from three minutes to forty
while the estimate wanders a median 1.8 km -- which is what an overconfident posterior
looks like. This measures it directly, before anything changes how confidence grows.

For each scoring fire and each time t on the fire's shared clock (epoch - t0_median, see
`clock_ref`), and for both estimators --

* `best`: one most-confident box per camera in [0, t], Gaussian bearing term (geolocate.py)
* `accum`: every box in [0, t], robust mixture with the n**alpha discount (accumulate.py)

-- it records the error, the Delta-loglik <= 3 area, and the **HPD level of the truth**: the
posterior mass on cells at least as probable as the truth's cell. A calibrated posterior
puts the truth inside its q-level region a fraction q of the time, so the levels should be
uniform on [0, 1]. Finally it finds the single log-likelihood temperature that would make
the 95% region cover 95% -- how much wider the posterior needs to be.

Uses the calibrated camera model (configs/calibrated.toml: fisheye lens + star pose ledger),
the current best, unless FIGLIB_PROFILE names another.

    python -m src.figlib.coverage          # writes out/confidence/
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import corpus as C
from . import settings
from . import pose_ledger
from .accumulate import posterior as accum_posterior
from .geolocate import FRAME_SIZES, YOLO_DIR, solve
from .geom import haversine_km, load_cams, offset_bearing_deg

ROOT = Path(__file__).resolve().parents[2]
META = C.current().meta
OUT = ROOT / "out" / "confidence"
TIMES = (180, 360, 600, 900, 1800, 2400)
ALPHA = 0.5
FINE_HALF_KM, FINE_STEP_KM = 35.0, 0.3


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M')} {msg}"
    print(line, flush=True)
    with open(OUT / "progress.log", "a") as f:
        f.write(line + "\n")


def clock_ref(fire: dict) -> int:
    """The fire's shared clock origin: fires.json `t0_median`.

    FIgLib offsets count from each camera's own annotated plume appearance, and those
    annotations disagree by up to 36 minutes on one fire (Steele's sm-n is ~16 min early
    against the other two cameras and its own frames). Windows keyed on per-camera offsets
    put cameras on different clocks. Every frame also carries its Unix epoch, so windows
    are taken on epoch - t0_median: one clock for all cameras, and one wrong annotation
    cannot move it. t0_median is the (upper) median over the event's sequences, the same
    reference truth.py matches discovery times against.
    """
    return int(fire["t0_median"])


def gather_calibrated(fire, seqs, cams, t_max, conf_thr=0.10):
    """accumulate.gather with the calibrated bearing (frame width for the lens, ledger
    azimuth), windowed on the shared clock: 0 <= epoch - t0_median <= t_max."""
    t_ref = clock_ref(fire)
    out: dict[str, list[tuple[float, float]]] = {}
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not path.exists():
            continue
        size = FRAME_SIZES.get(seq_name.split("#")[0]) or [None, None]
        cam = {**cams[s["camera"]], "frame_w": size[0], "frame_h": size[1]}
        acc = out.setdefault(s["camera"], [])
        for rec in json.loads(path.read_text()):
            if not (0 <= rec["epoch"] - t_ref <= t_max):
                continue
            cam_b, _ = pose_ledger.corrected_cam(s["camera"], cam, rec["epoch"])
            for d in rec["dets"]:
                if d["conf"] >= conf_thr:
                    acc.append((offset_bearing_deg(cam_b, (d["x0"] + d["x1"]) / 2, foot_y=d["y1"]), d["conf"]))
    return {k: v for k, v in out.items() if v}


def best_calibrated(fire, seqs, cams, t_max, conf_thr=0.25):
    """geolocate's best box per camera (box centre, calibrated), on the shared clock."""
    from .geolocate import Bearing
    t_ref = clock_ref(fire)
    best = {}
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not path.exists():
            continue
        size = FRAME_SIZES.get(seq_name.split("#")[0]) or [None, None]
        cam = {**cams[s["camera"]], "frame_w": size[0], "frame_h": size[1]}
        for rec in json.loads(path.read_text()):
            if not (0 <= rec["epoch"] - t_ref <= t_max):
                continue
            for d in rec["dets"]:
                if d["conf"] >= conf_thr and (s["camera"] not in best
                                              or d["conf"] > best[s["camera"]][0]["conf"]):
                    best[s["camera"]] = (d, rec, cam)
    out = []
    for camera, (d, rec, cam) in best.items():
        cam_b, _ = pose_ledger.corrected_cam(camera, cam, rec["epoch"])
        x = (d["x0"] + d["x1"]) / 2
        out.append(Bearing(camera=camera, lat=cam["lat"], lon=cam["lon"],
                           bearing_deg=offset_bearing_deg(cam_b, x, foot_y=d["y1"]), conf=d["conf"],
                           epoch=rec["epoch"], x_frac=round(x, 4)))
    return out


def hpd_level(ll: np.ndarray, i: int, j: int, temp: float = 1.0) -> float:
    """Posterior mass on cells at least as probable as cell (i, j), under ll / temp."""
    z = (ll - ll.max()) / temp
    p = np.exp(z)
    p /= p.sum()
    return float(p[p >= p[i, j]].sum())


def truth_cell(lats, lons, lat, lon):
    i = int(np.argmin(np.abs(lats - lat))); j = int(np.argmin(np.abs(lons - lon)))
    step = abs(lats[1] - lats[0])
    inside = bool(abs(lats[i] - lat) <= step and abs(lons[j] - lon) <= abs(lons[1] - lons[0]))
    return i, j, inside


def run_case(method, fire, truth, seqs, cams, t):
    if method == "best":
        bs = best_calibrated(fire, seqs, cams, t)
        if len({b.camera.split("-")[0] for b in bs}) < 2:
            return None
        c0 = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
        _, _, _, cla, clo = solve(bs, c0, half_extent_km=90.0, step_km=1.5)
        lats, lons, ll, la, lo = solve(bs, (cla, clo), half_extent_km=FINE_HALF_KM,
                                       step_km=FINE_STEP_KM)
        n_obs = len(bs)
    else:
        det = gather_calibrated(fire, seqs, cams, t)
        if len({k.split("-")[0] for k in det}) < 2:
            return None
        c0 = (float(np.mean([cams[k]["lat"] for k in det])),
              float(np.mean([cams[k]["lon"] for k in det])))
        _, _, _, cla, clo = accum_posterior(det, cams, c0, half_extent_km=90.0, step_km=1.5,
                                            alpha=ALPHA)
        lats, lons, ll, la, lo = accum_posterior(det, cams, (cla, clo),
                                                 half_extent_km=FINE_HALF_KM,
                                                 step_km=FINE_STEP_KM, alpha=ALPHA)
        n_obs = sum(len(v) for v in det.values())
    i, j, inside_grid = truth_cell(lats, lons, truth["lat"], truth["lon"])
    area = float((ll >= ll.max() - 3.0).sum()) * FINE_STEP_KM ** 2
    row = {"error_km": round(haversine_km(la, lo, truth["lat"], truth["lon"]), 2),
           "area95_km2": round(area, 1), "n_obs": n_obs,
           "truth_on_grid": inside_grid,
           "in_dll3": bool(inside_grid and ll[i, j] >= ll.max() - 3.0),
           # log-likelihood the truth sits below the peak; None when it is off the grid
           "dll_truth": round(float(ll.max() - ll[i, j]), 2) if inside_grid else None,
           "hpd_level": round(hpd_level(ll, i, j), 4) if inside_grid else 1.0}
    return row, (ll, i, j, inside_grid)


def main() -> None:
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = [r for r in json.loads((META / "resolved.json").read_text())
                if r["tier"] in ("confirmed", "probable") and r.get("triangulable")]
    log(f"coverage: {len(resolved)} scoring fires x {len(TIMES)} times x 2 estimators, "
        f"lens={settings.get('FIGLIB_LENS')} ledger={settings.get('FIGLIB_POSE_LEDGER')}")

    rows, grids = [], []
    for k, r in enumerate(resolved, 1):
        fire, truth = fires[r["fire_id"]], r["truth"]
        for method in ("best", "accum"):
            for t in TIMES:
                res = run_case(method, fire, truth, seqs, cams, t)
                if res is None:
                    continue
                row, g = res
                rows.append({"fire_id": r["fire_id"], "tier": r["tier"], "method": method,
                             "t": t, **row})
                grids.append(g)
        done = [x for x in rows if x["fire_id"] == r["fire_id"]]
        log(f"  [{k}/{len(resolved)}] {r['fire_id']} ({r['tier']}): {len(done)} cases; "
            + ", ".join(f"{x['method']}@{x['t']} lvl {x['hpd_level']:.2f}"
                        for x in done if x["t"] in (360, 2400)))

    # One temperature for all cases. Scaling the log-likelihood by 1/T widens the Delta-ll <= 3
    # region to Delta-ll <= 3T, so the T that covers 95% is the 95th percentile of the truth's
    # Delta-ll over 3. (An HPD-mass scan is distorted by the finite grid: at large T the
    # posterior goes flat over the window and every level tends to 1.) T = 4 means sigma
    # about twice as wide. Cases with the truth off the 70 km grid count as uncovered.
    temps = {}
    for method in ("best", "accum"):
        dl = [x["dll_truth"] for x in rows if x["method"] == method]
        finite = np.array([v for v in dl if v is not None])
        off = sum(v is None for v in dl)
        need = float(np.percentile(np.r_[finite, np.full(off, np.inf)], 95)) / 3.0
        temps[method] = {"n": len(dl), "off_grid": off,
                         "T_for_95": None if not np.isfinite(need) else round(need, 1),
                         "coverage_at": {str(T): round(float(np.mean([v is not None and v <= 3 * T for v in dl])), 2)
                                         for T in (1, 2, 4, 8, 16, 32)}}
        log(f"temperature {method}: T for 95% = {temps[method]['T_for_95']}, off grid {off}/{len(dl)}, "
            f"coverage at T: {temps[method]['coverage_at']}")

    summary = []
    for method in ("best", "accum"):
        for tier in ("confirmed", "probable", "all"):
            for t in TIMES:
                sel = [x for x in rows if x["method"] == method and x["t"] == t
                       and (tier == "all" or x["tier"] == tier)]
                if not sel:
                    continue
                lv = np.array([x["hpd_level"] for x in sel])
                summary.append({"method": method, "tier": tier, "t": t, "n": len(sel),
                                "cov50": round(float(np.mean(lv <= 0.5)), 2),
                                "cov90": round(float(np.mean(lv <= 0.9)), 2),
                                "cov95": round(float(np.mean(lv <= 0.95)), 2),
                                "in_dll3": round(float(np.mean([x["in_dll3"] for x in sel])), 2),
                                "median_err_km": round(float(np.median([x["error_km"] for x in sel])), 2),
                                "median_area95": round(float(np.median([x["area95_km2"] for x in sel])), 1)})
    (OUT / "coverage.json").write_text(json.dumps({"rows": rows, "summary": summary,
                                                   "temperature": temps}, indent=1) + "\n")
    log("summary (tier=all): method t n  cov50 cov90 cov95  in_dll3  med_err  med_area95")
    for s in summary:
        if s["tier"] == "all":
            log(f"  {s['method']:5s} {s['t']:5d} {s['n']:3d}  {s['cov50']:.2f}  {s['cov90']:.2f}  "
                f"{s['cov95']:.2f}   {s['in_dll3']:.2f}   {s['median_err_km']:5.2f}  {s['median_area95']:6.1f}")
    log("wrote out/confidence/coverage.json")


def figure() -> Path:
    """Coverage of the 95% region over time, and where the truth falls in the posterior."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads((OUT / "coverage.json").read_text())
    ink, muted, grid, surface = "#2b2a27", "#6f6d68", "#e6e4df", "#fcfcfb"
    color = {"accum": "#2a78d6", "best": "#898781"}
    label = {"accum": "all boxes, robust mixture", "best": "best box per camera"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=surface,
                                 gridspec_kw={"width_ratios": [1.2, 1]})
    for ax in (a1, a2):
        ax.set_facecolor(surface)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=muted)
        ax.grid(axis="y", color=grid, lw=0.8)
    for m in ("best", "accum"):
        rows = [x for x in d["summary"] if x["method"] == m and x["tier"] == "all"]
        # The Delta-ll <= 3 region, the one reported as area95 and scored in bias.py. The HPD
        # mass version (cov95) is inflated early, when a wide posterior hits the grid edge.
        t = [x["t"] / 60 for x in rows]; c = [x["in_dll3"] for x in rows]
        a1.plot(t, c, color=color[m], lw=2, marker="o", ms=8, mec=surface, mew=2)
        a1.annotate(label[m], (t[-1], c[-1]), xytext=(8, 0), textcoords="offset points",
                    color=ink, va="center", fontsize=9)
    a1.axhline(0.95, color=muted, lw=1, ls=(0, (4, 3)))
    a1.text(0.2, 0.965, "nominal 95%", color=muted, fontsize=9)
    a1.set_ylim(0, 1.02); a1.set_xlim(0, 52)
    a1.set_xlabel("minutes since the plume appeared", color=ink)
    a1.set_ylabel("share of fires with the truth inside the 95% region", color=ink)
    a1.set_title("How often the 95% region contains the official ignition", color=ink,
                 loc="left", fontsize=11)
    bins = np.linspace(0, 1, 11)
    for k, m in enumerate(("best", "accum")):
        lv = [x["hpd_level"] for x in d["rows"] if x["method"] == m]
        h, _ = np.histogram(lv, bins=bins)
        a2.bar(bins[:-1] + 0.05 + (k - 0.5) * 0.042, h / len(lv), width=0.04,
               color=color[m], label=label[m])
    a2.axhline(0.1, color=muted, lw=1, ls=(0, (4, 3)))
    a2.text(0.01, 0.115, "calibrated: flat at 10%", color=muted, fontsize=9)
    a2.set_xlabel("posterior mass more probable than the truth", color=ink)
    a2.set_ylabel("share of fire-time cases", color=ink)
    a2.set_title("Where the truth falls in the posterior, all times", color=ink,
                 loc="left", fontsize=11)
    a2.legend(frameon=False, fontsize=9, labelcolor=ink, loc="upper left")
    fig.tight_layout()
    dest = OUT / "coverage.png"
    fig.savefig(dest, dpi=150, facecolor=surface)
    return dest


if __name__ == "__main__":
    settings.default_profile("calibrated")   # configs/calibrated.toml unless FIGLIB_PROFILE is set
    import sys
    if sys.argv[1:] == ["figure"]:
        print(figure())
    else:
        main()
        print(figure())
