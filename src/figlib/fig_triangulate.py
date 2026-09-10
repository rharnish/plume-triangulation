"""One figure per fire: the smoke each camera saw, and where their bearings agree.

A likelihood contour plot on its own is unreadable to anyone who has not been told what
it is. The thing that makes triangulation legible is showing both halves at once and
colour-matching them -- this camera saw *that* plume, and its bearing is *this* ray --
so the crossing point is something the eye arrives at rather than something the caption
asserts.

The basemap is a hillshade of the same Copernicus DEM the terrain work uses, which puts
the rays over the ridges they actually cross and makes the scale legible without a
tile server or a network call.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geolocate import (META, YOLO_DIR, Bearing, bearings_for_fire,
                        credible_area_km2, solve)
from .geom import haversine_km, load_cams

ROOT = Path(__file__).resolve().parents[2]
TGZ = ROOT / "data" / "tgz"
OUT = ROOT / "out" / "triangulate"

# Ray colours, chosen to stay distinguishable against hillshade and against each other.
PALETTE = ["#ff5d5d", "#ffd166", "#4dd2a0", "#5fa8ff", "#c792ea", "#ff9f45"]
CROP_ASPECT = 2.4          # wide crop around the plume: sky above, terrain below


def _hillshade(lat0, lon0, half_km, n=700):
    """Relief basemap from the DEM, or None where tiles are missing."""
    from .terrain import Dem
    try:
        dem = Dem()
        pad = half_km / 111.0 + 0.05
        band, transform = dem.window(lat0, lon0, pad)
    except FileNotFoundError:
        return None, None
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * math.cos(math.radians(lat0)))
    lats = np.linspace(lat0 - dlat, lat0 + dlat, n)
    lons = np.linspace(lon0 - dlon, lon0 + dlon, n)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    z = Dem.sample(band, transform, LA.ravel(), LO.ravel()).reshape(LA.shape)

    # Standard hillshade, sun from the northwest at 45 deg.
    m_per_px_y = (2 * dlat * 111_320.0) / n
    m_per_px_x = (2 * dlon * 111_320.0 * math.cos(math.radians(lat0))) / n
    gy, gx = np.gradient(z, m_per_px_y, m_per_px_x)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(315.0), math.radians(45.0)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1), (lons[0], lons[-1], lats[0], lats[-1])


def _plume_crop(seq_name: str, camera: str, epoch: int, det: dict):
    """The frame that produced this bearing, cropped around the box it was drawn from."""
    import cv2
    from .detect_yolo import read_frames
    tgz = TGZ / f"{seq_name.split('#')[0]}.tgz"
    if not tgz.exists():
        return None
    frames = [(e, o, b) for e, o, b in read_frames(tgz) if e == epoch]
    if not frames:
        return None
    img = cv2.imdecode(np.frombuffer(frames[0][2], np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]

    # Crop wide enough that the plume sits in a landscape rather than filling the frame:
    # a box alone shows the detector worked, not what it was looking at.
    cx, cy = (det["x0"] + det["x1"]) / 2 * W, (det["y0"] + det["y1"]) / 2 * H
    bw = max((det["x1"] - det["x0"]) * W, W * 0.10)
    cw = float(np.clip(bw * 6.0, W * 0.22, W))
    ch = cw / CROP_ASPECT
    x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    x1, y1 = int(x0 + cw), int(y0 + ch)

    crop = img[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (int(det["x0"] * W) - x0, int(det["y0"] * H) - y0),
                  (int(det["x1"] * W) - x0, int(det["y1"] * H) - y0),
                  (255, 255, 255), max(2, int(cw / 240)))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _det_for(bearing: Bearing, fire: dict, seqs: dict) -> tuple[str, dict] | None:
    """Recover the exact detection a bearing came from, for the crop."""
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or s["camera"] != bearing.camera:
            continue
        path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not path.exists():
            continue
        for rec in json.loads(path.read_text()):
            if rec["epoch"] != bearing.epoch:
                continue
            best = max(rec["dets"], key=lambda d: d["conf"], default=None)
            if best:
                return seq_name, best
    return None


def render(fire_id: str, fire: dict, truth: dict, tier: str, seqs: dict, cams: dict,
           wind_cache: dict | None = None, out_dir: Path = OUT) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # use_wind=False deliberately: the reported table is the box-centre variant, and a
    # figure that disagreed with the number beside it would be worse than no figure.
    bs = bearings_for_fire(fire, seqs, cams, use_wind=False, wind_cache=wind_cache)
    if len({b.camera.split("-")[0] for b in bs}) < 2:
        return None
    bs = sorted(bs, key=lambda b: -b.conf)[:len(PALETTE)]

    # Solve on the same grid geolocate.main uses -- centred on the mean camera position,
    # not on the answer. Centring the grid on truth would snap the peak to the truth cell
    # and report an error better than the pipeline's, which is the one way a figure like
    # this can quietly lie.
    centre = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
    lats, lons, ll, elat, elon = solve(bs, centre)
    err = haversine_km(elat, elon, truth["lat"], truth["lon"])
    area = credible_area_km2(lats, lons, ll)

    # Frame on the bounding box of everything that must be visible -- every camera, the
    # estimate and the truth -- centred on that box rather than on the fire. Centring on
    # truth pushes the estimate off the edge exactly in the cases worth looking at, where
    # the two disagree by tens of kilometres.
    pts_lat = [b.lat for b in bs] + [truth["lat"], elat]
    pts_lon = [b.lon for b in bs] + [truth["lon"], elon]
    clat = (max(pts_lat) + min(pts_lat)) / 2
    clon = (max(pts_lon) + min(pts_lon)) / 2
    cos_lat = math.cos(math.radians(clat))
    half_km = max(9.0, 0.60 * max((max(pts_lat) - min(pts_lat)) * 111.32,
                                  (max(pts_lon) - min(pts_lon)) * 111.32 * cos_lat))

    crops = []
    for b, col in zip(bs, PALETTE):
        got = _det_for(b, fire, seqs)
        crops.append((b, col, _plume_crop(*got[:1], b.camera, b.epoch, got[1])
                      if got else None))

    n = len(crops)
    ncol = 1 if n <= 2 else 2
    nrow = -(-n // ncol)
    fig = plt.figure(figsize=(17.0, 9.6), dpi=125)
    fig.patch.set_facecolor("#11131a")
    gs = fig.add_gridspec(nrow, 1 + ncol,
                          width_ratios=[2.05] + [1.15] * ncol,
                          wspace=0.045, hspace=0.14,
                          left=0.035, right=0.985, top=0.90, bottom=0.045)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("#11131a")

    hs, extent = _hillshade(clat, clon, half_km)
    if hs is not None:
        ax.imshow(hs, origin="lower", extent=extent, cmap="gray",
                  vmin=-0.15, vmax=1.25, alpha=0.55, aspect="auto", zorder=0)

    rel = ll - ll.max()
    ax.contourf(lons, lats, rel, levels=[-3.0, 0.0], colors=["#ffd166"],
                alpha=0.22, zorder=2)
    ax.contour(lons, lats, rel, levels=[-3.0], colors=["#ffd166"],
               linewidths=1.2, linestyles="--", zorder=3)

    span = half_km * 2.4 / 111.32
    for b, col, _crop in crops:
        th = math.radians(b.bearing_deg)
        ax.plot([b.lon, b.lon + math.sin(th) * span / math.cos(math.radians(b.lat))],
                [b.lat, b.lat + math.cos(th) * span],
                color=col, lw=2.0, alpha=0.95, zorder=4,
                solid_capstyle="round")
        ax.plot(b.lon, b.lat, marker="^", color=col, ms=11, mec="#11131a", mew=1.2,
                zorder=6)
        ax.annotate(b.camera, (b.lon, b.lat), textcoords="offset points",
                    xytext=(11, 7), fontsize=8.5, color=col, weight="bold",
                    zorder=6)

    ax.plot(truth["lon"], truth["lat"], "o", mfc="none", mec="#ffffff", ms=21, mew=2.4,
            zorder=7)
    ax.plot(elon, elat, "x", color="#ff3860", ms=15, mew=3.4, zorder=8)

    dlat = half_km / 111.32
    dlon = half_km / (111.32 * cos_lat)
    ax.set_xlim(clon - dlon, clon + dlon)
    ax.set_ylim(clat - dlat, clat + dlat)

    # Scale bar, because degrees of longitude mean nothing at a glance.
    bar_km = max(2, int(round(half_km / 2.5)))
    bx0 = clon - dlon * 0.92
    by = clat - dlat * 0.92
    ax.plot([bx0, bx0 + bar_km / (111.32 * cos_lat)],
            [by, by], color="#e8e8e8", lw=3, solid_capstyle="butt", zorder=9)
    ax.text(bx0, by + dlat * 0.028, f"{bar_km} km", color="#e8e8e8", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3a4050")

    ax.legend(handles=[
        Line2D([], [], color="#8d93a3", lw=2, label="bearing from one camera"),
        Line2D([], [], color="#ffd166", lw=1.2, ls="--", label="95% credible region"),
        Line2D([], [], color="#ff3860", marker="x", ls="none", ms=10, mew=3,
               label="estimate"),
        Line2D([], [], color="#ffffff", marker="o", mfc="none", ls="none", ms=12,
               mew=2, label="official ignition point"),
    ], loc="best", fontsize=9, facecolor="#181b24", edgecolor="#3a4050",
        labelcolor="#dfe3ea", framealpha=0.92)

    for i, (b, col, crop) in enumerate(crops):
        cax = fig.add_subplot(gs[i // ncol, 1 + i % ncol])
        cax.set_facecolor("#11131a")
        if crop is not None:
            cax.imshow(crop)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_color(col); sp.set_linewidth(2.6)
        cax.set_title(f"{b.camera}   conf {b.conf:.2f}   bearing {b.bearing_deg:.1f}°",
                      fontsize=9.2, color=col, pad=3.5)

    acres = truth.get("acres")
    fig.suptitle(
        f"{fire_id}  →  {truth['name']}"
        + (f"  ({acres} acres)" if acres else "")
        + f"      {len({b.camera.split('-')[0] for b in bs})} sites"
        f"      error {err:.2f} km      95% region {area:.1f} km²"
        f"      [{tier}]",
        color="#f2f4f8", fontsize=14.5, y=0.965)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{fire_id}.png"
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main(argv: list[str]) -> None:
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = json.loads((META / "resolved.json").read_text())
    from .wind import _load as _load_wind
    wind_cache = _load_wind()

    todo = [r for r in resolved
            if r["tier"] in ("confirmed", "probable") and r.get("triangulable")]
    if argv:
        todo = [r for r in todo if any(a in r["fire_id"] for a in argv)]

    index = []
    for k, r in enumerate(todo, 1):
        fid = r["fire_id"]
        try:
            p = render(fid, fires[fid], r["truth"], r["tier"], seqs, cams, wind_cache)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] {fid}: FAIL {type(exc).__name__}: {exc}",
                  flush=True)
            continue
        if p is None:
            print(f"[{k}/{len(todo)}] {fid}: fewer than two sites", flush=True)
            continue
        bs = bearings_for_fire(fires[fid], seqs, cams, use_wind=False,
                               wind_cache=wind_cache)
        c = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
        lats, lons, ll, elat, elon = solve(bs, c)
        err = haversine_km(elat, elon, r["truth"]["lat"], r["truth"]["lon"])
        index.append({"fire_id": fid, "tier": r["tier"], "name": r["truth"]["name"],
                      "acres": r["truth"].get("acres"),
                      "n_sites": len({b.camera.split("-")[0] for b in bs}),
                      "n_bearings": len(bs), "error_km": round(err, 2),
                      "area95_km2": round(credible_area_km2(lats, lons, ll), 1)})
        print(f"[{k}/{len(todo)}] {fid:26s} {r['tier']:9s} "
              f"{index[-1]['n_sites']} sites  {err:6.2f} km", flush=True)

    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"\n{len(index)} figures -> {OUT}")


def sheet(tile_w: int = 560, cols: int = 4) -> Path:
    """All 26 on one page, best-first within tier, for choosing between them."""
    import cv2
    idx = json.loads((OUT / "index.json").read_text())
    idx.sort(key=lambda r: (r["tier"] != "confirmed", r["error_km"]))
    tiles = []
    for r in idx:
        img = cv2.imread(str(OUT / f"{r['fire_id']}.png"))
        if img is None:
            continue
        h, w = img.shape[:2]
        t = cv2.resize(img, (tile_w, int(tile_w * h / w)))
        lab = np.full((34, tile_w, 3), (26, 22, 18), np.uint8)
        col = (140, 230, 120) if r["tier"] == "confirmed" else (110, 190, 255)
        cv2.putText(lab, f"{r['fire_id'][:30]}  {r['n_sites']}s  {r['error_km']:.2f}km",
                    (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
        tiles.append(np.vstack([lab, t]))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                value=(26, 22, 18)) for t in tiles]
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(26, 22, 18)) for r in rows]
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
