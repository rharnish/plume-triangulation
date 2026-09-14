"""One figure per single-site fire: one bearing, and how far along it terrain lets the fire be.

`fig_triangulate` needs two sites, because a crossing is what locates a fire. A fire seen from
one site has no crossing, and drawing it with an estimate and an error in km would state a
range nothing measured. So this draws what one camera actually gives:

* **the bearing**, from the box centre through the star-measured lens and the pose ledger --
  the same calibrated camera model `coverage.py` scores;
* **the miss across the ray**: how far the official ignition sits to the side of the bearing,
  at its true distance. This tests pose and detection, and it is the number a one-camera
  alert has to get right;
* **a terrain range** for star-posed cameras (`terrain_range`): the distances along the ray
  where the early box bottom sits on the terrain line -- not below it (smoke hidden behind a
  crest can't be drawn) and not more than `BAND_PX` above it. Where the ledger declines a
  pose, the nearest star solve in time is used and labelled, and the two are scored apart.

The range is checked against the truth's distance *along* the ray, so a bearing miss does not
count against it.

    FIGLIB_CORPUS=all python -m src.figlib.fig_bearing            # every single-site fire
    FIGLIB_CORPUS=all python -m src.figlib.fig_bearing Posta Otay # names containing these
    FIGLIB_CORPUS=all python -m src.figlib.fig_bearing sheet      # contact sheet of the run

Writes `out/<corpus>/bearing/`: one PNG per fire, `index.json` (per fire and per bearing),
`summary.json`, `_contact_sheet.png`.
"""

from __future__ import annotations

import os

# The calibrated camera model, as in coverage.py. Set before the geometry reads them.
os.environ.setdefault("FIGLIB_LENS", "fisheye")
os.environ.setdefault("FIGLIB_POSE_LEDGER", "1")

import json
import math
from pathlib import Path

import numpy as np

from . import corpus as C
from . import provenance as P
from . import terrain_range as TR
from .fig_triangulate import PALETTE, _det_for, _hillshade, _plume_crop, _title
from .geolocate import FRAME_SIZES, META, bearings_for_fire
from .geom import angdiff_deg, bearing_deg, bearing_x_frac, haversine_km, load_cams
from .terrain_range import BAND_PX, MAX_KM, SLACK_PX, along_ray

OUT = C.current().out / "bearing"

FISHEYE_HALF_DEG = 54.0         # star-measured in-frame half-span of the 90 deg Mobotix lens


# ---------------------------------------------------------------- geometry

def analyse(b, fire: dict, seqs: dict, cams: dict, truth: dict) -> dict:
    """Everything the figure and the index report for one bearing."""
    cam = cams[b.camera]
    true_brg = bearing_deg(cam["lat"], cam["lon"], truth["lat"], truth["lon"])
    true_km = haversine_km(cam["lat"], cam["lon"], truth["lat"], truth["lon"])
    miss = angdiff_deg(true_brg, b.bearing_deg)      # + : truth clockwise of the ray
    got = _det_for(b, fire, seqs)
    seq_name, det = got if got else (None, None)
    base = seq_name.split("#")[0] if seq_name else None
    W, H = (FRAME_SIZES.get(base) or [None, None])[:2] if base else (None, None)
    cam_w = {**cam, "frame_w": W}
    view_cam = {**cam_w, "az": cam["az"] + (b.pose["d_az"] if b.pose else 0.0)}
    row = {"camera": b.camera, "seq": seq_name, "conf": round(b.conf, 3), "epoch": b.epoch,
           "x_frac": b.x_frac, "bearing_deg": round(b.bearing_deg, 2),
           "true_bearing_deg": round(true_brg, 2), "miss_deg": round(miss, 2),
           "true_km": round(true_km, 2),
           "along_km": round(true_km * math.cos(math.radians(miss)), 2),
           "lateral_km": round(true_km * math.sin(math.radians(miss)), 2),
           "truth_in_view": bearing_x_frac(view_cam, truth["lat"], truth["lon"]) is not None,
           "pose": b.pose["rule"] if b.pose else "published",
           "d_az": round(b.pose["d_az"], 3) if b.pose else 0.0,
           "frame_w": W, "box_bottom_px": None, "range_status": None,
           "range_pose": None, "range_pose_rule": None, "range_pose_gap_days": None,
           "seg_lo_km": None, "seg_hi_km": None, "seg_len_km": None, "seg_contains": None}
    pose = TR.full_pose(b.camera, b.epoch, W)
    if pose is None or det is None:
        row["range_status"] = ("detection not recovered" if det is None
                               else "lens not measured for this frame format" if W != 3072
                               else "camera never star-solved")
        return row
    row.update(range_pose=pose["source"], range_pose_rule=pose["rule"],
               range_pose_gap_days=pose["gap_days"])
    y1 = TR.early_box_bottom(seq_name, det, H)
    if y1 is None:
        row["range_status"] = "no early box"
        return row
    try:
        prof = TR.profile(cam, pose, b.bearing_deg, W, H)
    except FileNotFoundError:
        row["range_status"] = "no DEM along the ray"
        return row
    ok = TR.allowed(prof, y1)
    row.update(box_bottom_px=round(y1), _profile=(prof["d"], prof["rows"], y1))
    if not ok.any():
        row["range_status"] = "no distance fits the box bottom: pose or box is off"
        return row
    s = TR.summarise_mask(prof["d"], ok)
    row.update(range_status="ranged", seg_lo_km=s["lo_km"], seg_hi_km=s["hi_km"],
               seg_len_km=s["len_km"], seg_contains=TR.contains(ok, row["along_km"]))
    return row


# ---------------------------------------------------------------- drawing

def _view(rows: list[dict], cams: dict, truth: dict) -> dict:
    pts = [(truth["lat"], truth["lon"])]
    for r in rows:
        cam = cams[r["camera"]]
        pts.append((cam["lat"], cam["lon"]))
        reach = max(r["along_km"], r["seg_hi_km"] or 0.0, 4.0) * 1.12
        pts.append(tuple(float(v) for v in along_ray(cam["lat"], cam["lon"], r["bearing_deg"], reach)))
    la = [p[0] for p in pts]; lo = [p[1] for p in pts]
    clat, clon = (max(la) + min(la)) / 2, (max(lo) + min(lo)) / 2
    cos_lat = math.cos(math.radians(clat))
    half_km = max(9.0, 0.62 * max((max(la) - min(la)) * 111.32, (max(lo) - min(lo)) * 111.32 * cos_lat))
    return {"clat": clat, "clon": clon, "cos_lat": cos_lat, "half_km": half_km,
            "dlat": half_km / 111.32, "dlon": half_km / (111.32 * cos_lat)}


def _draw(title: str, view: dict, hillshade, cams: dict, rows: list[dict], truth: dict,
          crops: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    clat, clon, cos_lat = view["clat"], view["clon"], view["cos_lat"]
    half_km, dlat, dlon = view["half_km"], view["dlat"], view["dlon"]
    profiles = [r for r in rows if r.get("_profile") is not None]
    panels = [("crop", c) for c in crops] + [("profile", r) for r in profiles]
    grid_rows = max(len(panels), 3)
    fig = plt.figure(figsize=(16.0, max(9.0, 1.9 * grid_rows + 1.2)), dpi=125)
    fig.patch.set_facecolor("#11131a")
    gs = fig.add_gridspec(grid_rows, 2, width_ratios=[2.35, 1.0], wspace=0.07, hspace=0.42,
                          left=0.035, right=0.975, top=0.90, bottom=0.05)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("#11131a")
    ax.set_xlim(clon - dlon, clon + dlon); ax.set_ylim(clat - dlat, clat + dlat)
    hs, extent = hillshade
    if hs is not None:
        ax.imshow(hs, origin="lower", extent=extent, cmap="gray", vmin=-0.15, vmax=1.25,
                  alpha=0.55, aspect="auto", zorder=0)

    edge_km = half_km * 3.0
    for r, col in zip(rows, PALETTE):
        cam = cams[r["camera"]]
        # Field of view: the star-measured span where that lens applies, else the nameplate.
        az_axis = cam["az"] + r["d_az"]
        fisheye = cam.get("fov") == 90 and r["frame_w"] == 3072
        half = FISHEYE_HALF_DEG if fisheye else cam["fov"] / 2.0
        wedge_km = min(half_km * 0.55, 25.0)
        edges = np.linspace(az_axis - half, az_axis + half, 30)
        wl = [cam["lat"]] + [float(along_ray(cam["lat"], cam["lon"], a, wedge_km)[0]) for a in edges] + [cam["lat"]]
        wo = [cam["lon"]] + [float(along_ray(cam["lat"], cam["lon"], a, wedge_km)[1]) for a in edges] + [cam["lon"]]
        ax.fill(wo, wl, color=col, alpha=0.12, lw=0, zorder=1)
        ax.plot(wo, wl, color=col, lw=0.8, alpha=0.4, zorder=1)

        brg = r["bearing_deg"]
        lo, hi = r["seg_lo_km"], r["seg_hi_km"]
        if hi is not None:
            def pt(km, _cam=cam, _brg=brg):
                la_, lo_ = along_ray(_cam["lat"], _cam["lon"], _brg, km)
                return float(la_), float(lo_)
            a_lat, a_lon = pt(lo); e_lat, e_lon = pt(hi); f_lat, f_lon = pt(edge_km)
            dashed = dict(color=col, lw=1.3, ls=(0, (4, 4)), alpha=0.7, zorder=4)
            if lo > 0:
                ax.plot([cam["lon"], a_lon], [cam["lat"], a_lat], **dashed)
            ax.plot([a_lon, e_lon], [a_lat, e_lat], color=col, lw=3.4, zorder=4, solid_capstyle="butt")
            ax.plot([e_lon, f_lon], [e_lat, f_lat], **dashed)
            # Short ticks across the ray at each end of the allowed range.
            for km_, (p_lat, p_lon) in ((lo, (a_lat, a_lon)), (hi, (e_lat, e_lon))):
                if km_ <= 0:
                    continue
                for s in (-1, 1):
                    t_lat, t_lon = along_ray(p_lat, p_lon, brg + 90 * s, half_km * 0.025)
                    ax.plot([p_lon, t_lon], [p_lat, t_lat], color=col, lw=2.4, zorder=5)
            ax.annotate(f"terrain {lo:.1f}–{hi:.1f} km", (e_lon, e_lat), textcoords="offset points",
                        xytext=(8, -12), fontsize=8.5, color=col, weight="bold", zorder=6)
        else:
            f_lat, f_lon = along_ray(cam["lat"], cam["lon"], brg, edge_km)
            ax.plot([cam["lon"], f_lon], [cam["lat"], f_lat], color=col, lw=2.0, alpha=0.95,
                    zorder=4, solid_capstyle="round")

        # Where the truth falls across the ray.
        p_lat, p_lon = along_ray(cam["lat"], cam["lon"], brg, max(r["along_km"], 0.0))
        ax.plot([p_lon, truth["lon"]], [p_lat, truth["lat"]], color="#ffffff", lw=1.1,
                ls=(0, (1.5, 2.5)), zorder=5)
        ax.plot(cam["lon"], cam["lat"], marker="^", color=col, ms=11, mec="#11131a", mew=1.2, zorder=6)
        ax.annotate(r["camera"], (cam["lon"], cam["lat"]), textcoords="offset points",
                    xytext=(11, 7), fontsize=8.5, color=col, weight="bold", zorder=6)

    ax.plot(truth["lon"], truth["lat"], "o", mfc="none", mec="#ffffff", ms=21, mew=2.4, zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3a4050")

    bar_km = max(2, int(round(half_km / 2.5)))
    bar_dlon = bar_km / (111.32 * cos_lat)
    bx0, by = clon - dlon * 0.92, clat - dlat * 0.93
    ax.plot([bx0, bx0 + bar_dlon], [by, by], color="#e8e8e8", lw=3, solid_capstyle="butt", zorder=9)
    ax.text(bx0, by + dlat * 0.028, f"{bar_km} km", color="#e8e8e8", fontsize=9)

    ax.legend(handles=[
        Line2D([], [], color="#8d93a3", lw=2, label="bearing (no range bound)"),
        Line2D([], [], color="#8d93a3", lw=3.4, label="range the terrain allows"),
        Line2D([], [], color="#8d93a3", lw=1.3, ls=(0, (4, 4)),
               label="ruled out: box bottom off the terrain line"),
        Line2D([], [], color="#8d93a3", lw=6, alpha=0.3, label="camera field of view"),
        Line2D([], [], color="#ffffff", lw=1.1, ls=(0, (1.5, 2.5)), label="miss across the ray"),
        Line2D([], [], color="#ffffff", marker="o", mfc="none", ls="none", ms=12, mew=2,
               label="official ignition point"),
    ], loc="upper left", fontsize=9, facecolor="#181b24", edgecolor="#3a4050",
        labelcolor="#dfe3ea", framealpha=0.92)

    colors = {r["camera"]: c for r, c in zip(rows, PALETTE)}
    for i, (kind, item) in enumerate(panels):
        pax = fig.add_subplot(gs[i, 1])
        pax.set_facecolor("#11131a")
        if kind == "crop":
            ctitle, col, crop = item
            if crop is not None:
                pax.imshow(crop)
            pax.set_xticks([]); pax.set_yticks([])
            for sp in pax.spines.values():
                sp.set_color(col); sp.set_linewidth(2.6)
            pax.set_title(ctitle, fontsize=9.2, color=col, pad=3.5)
            continue
        r = item; col = colors[r["camera"]]
        d_m, rows_px, y1 = r["_profile"]
        km = d_m / 1000.0
        xmax = min(MAX_KM, max(r["true_km"], r["seg_hi_km"] or 0.0) * 1.6 + 4.0)
        keep = km <= xmax
        pax.plot(km[keep], rows_px[keep], color="#c8cedb", lw=1.4, label="lowest visible smoke row")
        # The band the terrain line must fall in: from SLACK_PX above the box bottom to
        # BAND_PX below it.
        pax.axhspan(y1 - SLACK_PX, y1 + BAND_PX, color=col, alpha=0.18, lw=0)
        pax.axhline(y1, color=col, lw=1.6, label="early box bottom")
        s_lo, s_hi = r["seg_lo_km"], r["seg_hi_km"]
        if s_lo is not None:
            if s_lo > 0.05:
                pax.axvspan(0, s_lo, color="#ff3860", alpha=0.10, lw=0)
                pax.axvline(s_lo, color=col, lw=1.8)
            pax.axvspan(s_hi, xmax, color="#ff3860", alpha=0.10, lw=0)
            pax.axvline(s_hi, color=col, lw=1.8)
        pax.axvline(r["along_km"], color="#ffffff", lw=1.3, ls=(0, (3, 3)))
        pax.set_xlim(0, xmax)
        lo, hi = float(np.nanmin(rows_px[keep])), float(np.nanmax(rows_px[keep]))
        pad = max(40.0, 0.15 * (hi - lo))
        pax.set_ylim(max(hi, y1) + pad, min(lo, y1) - pad)
        pax.tick_params(colors="#9aa3b2", labelsize=7.5)
        for sp in pax.spines.values():
            sp.set_color("#3a4050")
        pax.set_xlabel("km along the bearing (white: truth)", color="#9aa3b2", fontsize=8, labelpad=1)
        pax.set_ylabel("row (px)", color="#9aa3b2", fontsize=8, labelpad=1)
        verdict = ("" if r["seg_contains"] is None
                   else "  contains truth" if r["seg_contains"] else "  MISSES truth")
        pax.set_title(f"{r['camera']}  terrain range, {BAND_PX:.0f} px band{verdict}\n"
                      f"pose: {r['range_pose_rule']}", fontsize=8.5, color=col, pad=3.5)

    fig.suptitle(title, color="#f2f4f8", fontsize=14.5, y=0.965)
    return fig


# ---------------------------------------------------------------- driver

def render(fire_id: str, fire: dict, truth: dict, tier: str, seqs: dict, cams: dict,
           out_dir: Path = OUT) -> tuple[Path | None, dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = bearings_for_fire(fire, seqs, cams, use_wind=False)
    bs = sorted(bs, key=lambda b: -b.conf)[:len(PALETTE)]
    rec = {"fire_id": fire_id, "tier": tier, "name": truth.get("name"), "acres": truth.get("acres"),
           "n_cameras": len(bs), "sites": sorted({b.camera.split("-")[0] for b in bs})}
    if not bs:
        rec["status"] = f"no detection at conf >= {TR.CONF_THR} from a posed camera"
        return None, rec

    rows = [analyse(b, fire, seqs, cams, truth) for b in bs]
    view = _view(rows, cams, truth)
    crops = []
    for b, col in zip(bs, PALETTE):
        got = _det_for(b, fire, seqs)
        crops.append((f"{b.camera}   conf {b.conf:.2f}   bearing {b.bearing_deg:.1f}°", col,
                      _plume_crop(got[0], b.camera, b.epoch, got[1]) if got else None))

    p = rows[0]
    range_txt = (f"terrain {p['seg_lo_km']:.1f}–{p['seg_hi_km']:.1f} km vs truth {p['along_km']:.1f} km"
                 if p["seg_lo_km"] is not None else "no terrain range")
    title = _title(fire_id, truth, tier,
                   f"      1 site      miss {p['lateral_km']:+.2f} km ({p['miss_deg']:+.1f}°)"
                   f"      {range_txt}")
    fig = _draw(title, view, _hillshade(view["clat"], view["clon"], view["half_km"]),
                cams, rows, truth, crops)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{fire_id}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)

    rec["status"] = "ok"
    rec["bearings"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    return out, rec


def summarise(index: list[dict]) -> dict:
    ok = [r for r in index if r["status"] == "ok"]
    prim = [r["bearings"][0] for r in ok]
    def med(xs):
        return round(float(np.median(xs)), 2) if xs else None
    out = {"band_px": BAND_PX, "slack_px": SLACK_PX, "fires_selected": len(index),
           "figures": len(ok), "no_detection": sum(r["status"] != "ok" for r in index)}
    for tier in ("confirmed", "probable"):
        t = [r["bearings"][0] for r in ok if r["tier"] == tier]
        by_pose = {}
        for source in ("ledger", "nearest"):
            ranged = [b for b in t if b["seg_lo_km"] is not None and b["range_pose"] == source]
            seen = [b for b in ranged if b["truth_in_view"]]
            by_pose[source] = {
                "ranged": len(ranged),
                "truth_in_view": len(seen),
                "contains_truth": sum(bool(b["seg_contains"]) for b in seen),
                "median_len_over_truth": med([b["seg_len_km"] / b["along_km"] for b in seen
                                              if b["along_km"] > 0]),
            }
        out[tier] = {
            "figures": len(t),
            "median_abs_lateral_km": med([abs(b["lateral_km"]) for b in t]),
            "median_abs_miss_deg": med([abs(b["miss_deg"]) for b in t]),
            "truth_in_view": sum(b["truth_in_view"] for b in t),
            "bearing_star_corrected": sum(b["pose"] != "published" for b in t),
            "range_by_pose": by_pose,
            "range_status": {s: sum(b["range_status"] == s for b in t)
                             for s in sorted({b["range_status"] for b in t})},
        }
    out["all_primary_median_abs_lateral_km"] = med([abs(b["lateral_km"]) for b in prim])
    return out


def main(argv: list[str]) -> None:
    started = P.utc_now()
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = json.loads((META / "resolved.json").read_text())

    todo = [r for r in resolved
            if r["tier"] in ("confirmed", "probable") and not r.get("triangulable")
            and (r.get("truth") or {}).get("lat") is not None]
    if argv:
        todo = [r for r in todo if any(a in r["fire_id"] for a in argv)]

    index = []
    for k, r in enumerate(todo, 1):
        fid = r["fire_id"]
        try:
            path, rec = render(fid, fires[fid], r["truth"], r["tier"], seqs, cams)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] {fid}: FAIL {type(exc).__name__}: {exc}", flush=True)
            index.append({"fire_id": fid, "tier": r["tier"], "status": f"error: {exc}"})
            continue
        index.append(rec)
        if path is None:
            print(f"[{k}/{len(todo)}] {fid:34s} {r['tier']:9s} {rec['status']}", flush=True)
            continue
        b = rec["bearings"][0]
        rng = ("no terrain range" if b["seg_lo_km"] is None
               else f"terrain {b['seg_lo_km']:5.1f}-{b['seg_hi_km']:5.1f} vs {b['along_km']:5.1f} km "
                    f"{'ok' if b['seg_contains'] else 'MISS'}")
        print(f"[{k}/{len(todo)}] {fid:34s} {r['tier']:9s} {b['camera']:18s} "
              f"miss {b['lateral_km']:+6.2f} km  {rng}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    if not argv:
        (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
        summary = summarise(index)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
        P.record("fig_bearing",
                 [OUT / "index.json", OUT / "summary.json"]
                 + [OUT / f"{r['fire_id']}.png" for r in index if r.get("status") == "ok"],
                 started=started, extra_inputs=TR._inputs(),
                 params={"band_px": BAND_PX, "slack_px": SLACK_PX, "conf_thr": TR.CONF_THR,
                         "early_s": TR.EARLY_S, "n_early": TR.N_EARLY})
    print(f"\n{sum(r.get('status') == 'ok' for r in index)} figures -> {OUT}")


def sheet(tile_w: int = 560, cols: int = 4) -> Path:
    """Every figure on one page: confirmed first, smallest miss first."""
    import cv2
    idx = [r for r in json.loads((OUT / "index.json").read_text()) if r.get("status") == "ok"]
    idx.sort(key=lambda r: (r["tier"] != "confirmed", abs(r["bearings"][0]["lateral_km"])))
    tiles = []
    for r in idx:
        img = cv2.imread(str(OUT / f"{r['fire_id']}.png"))
        if img is None:
            continue
        h, w = img.shape[:2]
        t = cv2.resize(img, (tile_w, int(tile_w * h / w)))
        b = r["bearings"][0]
        rng = ("" if b.get("seg_contains") is None
               else "  terrain ok" if b["seg_contains"] else "  terrain X")
        lab = np.full((34, tile_w, 3), (26, 22, 18), np.uint8)
        col = (140, 230, 120) if r["tier"] == "confirmed" else (110, 190, 255)
        cv2.putText(lab, f"{r['fire_id'][:28]}  {b['lateral_km']:+.2f}km{rng}", (6, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        tiles.append(np.vstack([lab, t]))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(26, 22, 18))
             for t in tiles]
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(26, 22, 18))
            for r in rows]
    out = OUT / "_contact_sheet.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"{len(tiles)} tiles -> {out}")
    return out


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a and a[0] == "sheet":
        sheet()
    else:
        main(a)
