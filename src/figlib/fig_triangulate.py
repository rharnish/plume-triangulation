"""One figure per fire: the smoke each camera saw, and where their bearings agree.

A likelihood contour plot on its own is unreadable to anyone who has not been told what
it is. The thing that makes triangulation legible is showing both halves at once and
color-matching them -- this camera saw *that* plume, and its bearing is *this* ray --
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

# Ray colors, chosen to stay distinguishable against hillshade and against each other.
PALETTE = ["#ff5d5d", "#ffd166", "#4dd2a0", "#5fa8ff", "#c792ea", "#ff9f45",
           "#3fd0d8", "#ff8fc7"]
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
    return _crop_image(img, _crop_window(det, W, H), det)


def _crop_window(det: dict, W: int, H: int) -> tuple:
    """Pixel window around a detection, as (x0, y0, x1, y1, cw, W, H)."""
    # Crop wide enough that the plume sits in a landscape rather than filling the frame:
    # a box alone shows the detector worked, not what it was looking at.
    cx, cy = (det["x0"] + det["x1"]) / 2 * W, (det["y0"] + det["y1"]) / 2 * H
    bw = max((det["x1"] - det["x0"]) * W, W * 0.10)
    cw = float(np.clip(bw * 6.0, W * 0.22, W))
    ch = cw / CROP_ASPECT
    x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    x1, y1 = int(x0 + cw), int(y0 + ch)
    return x0, y0, x1, y1, cw, W, H


def _crop_image(img, window: tuple, det: dict):
    """Cut `window` out of a BGR frame with the detection's box drawn in; returns RGB."""
    import cv2
    x0, y0, x1, y1, cw, W, H = window
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


def _view(bs, truth: dict, elat: float, elon: float) -> dict:
    """Map extent for a set of bearings, an estimate and the truth."""
    # Frame on the bounding box of everything that must be visible -- every camera, the
    # estimate and the truth -- centered on that box rather than on the fire. Centering on
    # truth pushes the estimate off the edge exactly in the cases worth looking at, where
    # the two disagree by tens of kilometers.
    pts_lat = [b.lat for b in bs] + [truth["lat"], elat]
    pts_lon = [b.lon for b in bs] + [truth["lon"], elon]
    clat = (max(pts_lat) + min(pts_lat)) / 2
    clon = (max(pts_lon) + min(pts_lon)) / 2
    cos_lat = math.cos(math.radians(clat))
    half_km = max(9.0, 0.66 * max((max(pts_lat) - min(pts_lat)) * 111.32,
                                  (max(pts_lon) - min(pts_lon)) * 111.32 * cos_lat))
    return {"clat": clat, "clon": clon, "cos_lat": cos_lat, "half_km": half_km,
            "dlat": half_km / 111.32, "dlon": half_km / (111.32 * cos_lat)}


def _inset_spec(bs, truth: dict, elat: float, elon: float, err: float, area: float,
                view: dict) -> dict | None:
    """Where the likelihood zoom goes and how much it shows, or None to leave it out."""
    # The inset answers a second question -- how tightly the surface constrains the answer
    # -- but only earns its space when the credible region is genuinely too small to read
    # on the main map. Once the rays cross wide, or estimate and truth disagree by enough
    # that the falloff is already legible at the main span, it is clutter, and is dropped.
    z_km = max(1.6, 2.6 * math.sqrt(max(area, 0.4)), err * 1.5)
    if z_km >= 0.24 * view["half_km"]:
        return None
    clat, clon, dlat, dlon = view["clat"], view["clon"], view["dlat"], view["dlon"]

    # Drop it into whichever corner holds the fewest markers; the legend takes the
    # upper left, so it is left out of the running.
    def _axfrac(lon, lat):
        return ((lon - (clon - dlon)) / (2 * dlon),
                (lat - (clat - dlat)) / (2 * dlat))
    pts_ax = ([_axfrac(b.lon, b.lat) for b in bs]
              + [_axfrac(elon, elat), _axfrac(truth["lon"], truth["lat"])])
    iw = ih = 0.32
    corners = {"lower left": (0.02, 0.03),
               "lower right": (0.97 - iw, 0.03),
               "upper right": (0.97 - iw, 0.96 - ih)}
    corner, (x0, y0) = min(
        corners.items(),
        key=lambda kv: sum(kv[1][0] - 0.05 <= px <= kv[1][0] + iw + 0.05
                           and kv[1][1] - 0.05 <= py <= kv[1][1] + ih + 0.05
                           for px, py in pts_ax))
    return {"corner": corner, "rect": [x0, y0, iw, ih], "z_km": z_km,
            "mlat": (elat + truth["lat"]) / 2, "mlon": (elon + truth["lon"]) / 2}


def _draw(title: str, view: dict, hillshade, cams: dict, cameras: list, rays: list,
          surface, est, truth: dict, inset: dict | None, crops: list):
    """Draw one figure and return it.

    `cameras` is [(camera, color)] -- every camera gets its wedge and marker whether or not
    it has a bearing yet. `rays` is {camera: Bearing}. `surface` is (lats, lons, ll) and
    `est` is (lat, lon), both None until two sites have bearings. `crops` is
    [(title, color, image-or-None)] in panel order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    clat, clon, cos_lat = view["clat"], view["clon"], view["cos_lat"]
    half_km, dlat, dlon = view["half_km"], view["dlat"], view["dlon"]

    n = len(crops)
    # Every camera view in one vertical stack down the right edge, so the map keeps the
    # whole left half and every ray stays in one frame. Never fewer than three rows, so a
    # two-camera fire does not stretch its crops to fill half a page.
    grid_rows = max(n, 3)
    fig_h = max(9.0, 1.7 * grid_rows + 1.2)
    fig = plt.figure(figsize=(16.0, fig_h), dpi=125)
    fig.patch.set_facecolor("#11131a")
    gs = fig.add_gridspec(grid_rows, 2, width_ratios=[2.35, 1.0],
                          wspace=0.05, hspace=0.22,
                          left=0.035, right=0.985, top=0.90, bottom=0.045)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("#11131a")

    ax.set_xlim(clon - dlon, clon + dlon)
    ax.set_ylim(clat - dlat, clat + dlat)

    hs, extent = hillshade
    if hs is not None:
        ax.imshow(hs, origin="lower", extent=extent, cmap="gray",
                  vmin=-0.15, vmax=1.25, alpha=0.55, aspect="auto", zorder=0)

    if surface is not None:
        lats, lons, ll = surface
        # The likelihood as a heatmap rather than a flat patch: the shape of the falloff
        # is the honest uncertainty, and a single outline throws it away. Alpha ramps to
        # zero at the -9 contour so the relief stays visible everywhere the surface says
        # nothing.
        rel = ll - ll.max()
        a = np.clip((rel + 9.0) / 9.0, 0.0, 1.0)
        rgba = matplotlib.colormaps["magma"](a)
        rgba[..., 3] = 0.90 * a ** 1.6
        ax.imshow(rgba, origin="lower",
                  extent=[lons[0], lons[-1], lats[0], lats[-1]],
                  aspect="auto", zorder=2, interpolation="bilinear")
        cs = ax.contour(lons, lats, rel, levels=[-6.0, -3.0, -1.0],
                        colors=["#7fb2ff", "#bfe0ff", "#ffffff"],
                        linewidths=[0.9, 1.2, 1.4], zorder=3)
        ax.clabel(cs, fmt={-6.0: "", -3.0: "95%", -1.0: "peak"}, fontsize=7.5,
                  colors="#e8eefc")

    span = half_km * 2.4 / 111.32

    # Pixel size of the axes box, to size the FOV wedge in screen pixels rather than
    # ground distance -- a wedge drawn to scale would span kilometers and bury the map,
    # when all it needs to do is show which way the camera looks.
    ax_bbox = ax.get_position()
    fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi
    ax_w_px, ax_h_px = ax_bbox.width * fig_w_px, ax_bbox.height * fig_h_px
    km_per_px = math.sqrt((2 * dlon * 111.32 * cos_lat / ax_w_px)
                          * (2 * dlat * 111.32 / ax_h_px))
    wedge_span = 50 * km_per_px / 111.32

    for camera, col in cameras:
        # The camera's own fixed field of view, faint and short behind the sighting: the
        # bearing ray is one detection, but the wedge is what the camera can ever see, so
        # it's the thing that actually reads as "which way is this camera facing."
        cam = cams[camera]
        clat_c = math.radians(cam["lat"])
        half_fov = math.radians(cam["fov"] / 2.0)
        az0 = math.radians(cam["az"])
        edges = np.linspace(az0 - half_fov, az0 + half_fov, 24)
        wx = [cam["lon"]] + [cam["lon"] + math.sin(a) * wedge_span / math.cos(clat_c)
                              for a in edges] + [cam["lon"]]
        wy = [cam["lat"]] + [cam["lat"] + math.cos(a) * wedge_span for a in edges] \
            + [cam["lat"]]
        ax.fill(wx, wy, color=col, alpha=0.16, lw=0, zorder=1)
        ax.plot(wx, wy, color=col, lw=0.8, alpha=0.45, zorder=1)
        b = rays.get(camera)
        if b is not None:
            th = math.radians(b.bearing_deg)
            ax.plot([b.lon, b.lon + math.sin(th) * span / math.cos(math.radians(b.lat))],
                    [b.lat, b.lat + math.cos(th) * span],
                    color=col, lw=2.0, alpha=0.95, zorder=4,
                    solid_capstyle="round")
        ax.plot(cam["lon"], cam["lat"], marker="^", color=col, ms=11, mec="#11131a",
                mew=1.2, zorder=6)
        ax.annotate(camera, (cam["lon"], cam["lat"]), textcoords="offset points",
                    xytext=(11, 7), fontsize=8.5, color=col, weight="bold",
                    zorder=6)

    ax.plot(truth["lon"], truth["lat"], "o", mfc="none", mec="#ffffff", ms=21, mew=2.4,
            zorder=7)
    if est is not None:
        ax.plot(est[1], est[0], "x", color="#ff3860", ms=15, mew=3.4, zorder=8)

    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3a4050")

    inset_corner = None
    if inset is not None:
        inset_corner = inset["corner"]
        iax = ax.inset_axes(inset["rect"])
        iax.set_facecolor("#0d0f15")
        z_km = inset["z_km"]
        zlat, zlon = z_km / 111.32, z_km / (111.32 * cos_lat)
        mlat, mlon = inset["mlat"], inset["mlon"]
        if surface is not None:
            iax.imshow(rgba, origin="lower",
                       extent=[lons[0], lons[-1], lats[0], lats[-1]],
                       aspect="auto", zorder=1, interpolation="bilinear")
            ics = iax.contour(lons, lats, rel, levels=[-6.0, -3.0, -1.0],
                              colors=["#7fb2ff", "#bfe0ff", "#ffffff"],
                              linewidths=[0.9, 1.2, 1.4], zorder=2)
            iax.clabel(ics, fmt={-6.0: "", -3.0: "95%", -1.0: "peak"}, fontsize=7,
                       colors="#e8eefc")
        for camera, col in cameras:
            b = rays.get(camera)
            if b is None:
                continue
            th = math.radians(b.bearing_deg)
            iax.plot([b.lon,
                      b.lon + math.sin(th) * span / math.cos(math.radians(b.lat))],
                     [b.lat, b.lat + math.cos(th) * span], color=col, lw=1.4,
                     alpha=0.9, zorder=3)
        iax.plot(truth["lon"], truth["lat"], "o", mfc="none", mec="#ffffff", ms=14,
                 mew=2, zorder=5)
        if est is not None:
            iax.plot(est[1], est[0], "x", color="#ff3860", ms=11, mew=2.6, zorder=5)
        iax.set_xlim(mlon - zlon, mlon + zlon); iax.set_ylim(mlat - zlat, mlat + zlat)
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_color("#8d93a3"); sp.set_linewidth(1.1)
        iax.set_title(f"likelihood surface, {2 * z_km:.0f} km across", fontsize=8,
                      color="#c8cedb", pad=2.5)
        ax.indicate_inset_zoom(iax, edgecolor="#8d93a3", alpha=0.75, lw=0.9)

    # Scale bar, because degrees of longitude mean nothing at a glance. Sits bottom-left
    # unless the inset took that corner, then bottom-right.
    bar_km = max(2, int(round(half_km / 2.5)))
    bar_dlon = bar_km / (111.32 * cos_lat)
    bx0 = (clon - dlon * 0.92 if inset_corner != "lower left"
           else clon + dlon * 0.92 - bar_dlon)
    by = clat - dlat * 0.93
    ax.plot([bx0, bx0 + bar_dlon], [by, by],
            color="#e8e8e8", lw=3, solid_capstyle="butt", zorder=9)
    ax.text(bx0, by + dlat * 0.028, f"{bar_km} km", color="#e8e8e8", fontsize=9)

    ax.legend(handles=[
        Line2D([], [], color="#8d93a3", lw=2, label="bearing from one camera"),
        Line2D([], [], color="#8d93a3", lw=6, alpha=0.35, label="camera field of view"),
        Line2D([], [], color="#bfe0ff", lw=1.2, label="95% credible contour"),
        Line2D([], [], color="#ff3860", marker="x", ls="none", ms=10, mew=3,
               label="estimate"),
        Line2D([], [], color="#ffffff", marker="o", mfc="none", ls="none", ms=12,
               mew=2, label="official ignition point"),
    ], loc="upper left", fontsize=9, facecolor="#181b24", edgecolor="#3a4050",
        labelcolor="#dfe3ea", framealpha=0.92)

    for i, (ctitle, col, crop) in enumerate(crops):
        cax = fig.add_subplot(gs[i, 1])
        cax.set_facecolor("#11131a")
        if crop is not None:
            cax.imshow(crop)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_color(col); sp.set_linewidth(2.6)
        cax.set_title(ctitle, fontsize=9.2, color=col, pad=3.5)

    fig.suptitle(title, color="#f2f4f8", fontsize=14.5, y=0.965)
    return fig


def _title(fire_id: str, truth: dict, tier: str, middle: str) -> str:
    acres = truth.get("acres")
    return (f"{fire_id}  →  {truth['name']}"
            + (f"  ({acres} acres)" if acres else "")
            + middle + f"      [{tier}]")


def render(fire_id: str, fire: dict, truth: dict, tier: str, seqs: dict, cams: dict,
           wind_cache: dict | None = None, out_dir: Path = OUT) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # use_wind=False deliberately: the reported table is the box-center variant, and a
    # figure that disagreed with the number beside it would be worse than no figure.
    bs = bearings_for_fire(fire, seqs, cams, use_wind=False, wind_cache=wind_cache)
    if len({b.camera.split("-")[0] for b in bs}) < 2:
        return None
    bs = sorted(bs, key=lambda b: -b.conf)[:len(PALETTE)]

    # Solve on the same grid geolocate.main uses -- centered on the mean camera position,
    # not on the answer. Centering the grid on truth would snap the peak to the truth cell
    # and report an error better than the pipeline's, which is the one way a figure like
    # this can quietly lie.
    center = (float(np.mean([b.lat for b in bs])), float(np.mean([b.lon for b in bs])))
    lats, lons, ll, elat, elon = solve(bs, center)
    err = haversine_km(elat, elon, truth["lat"], truth["lon"])
    area = credible_area_km2(lats, lons, ll)
    view = _view(bs, truth, elat, elon)

    crops = []
    for b, col in zip(bs, PALETTE):
        got = _det_for(b, fire, seqs)
        crops.append((f"{b.camera}   conf {b.conf:.2f}   bearing {b.bearing_deg:.1f}°", col,
                      _plume_crop(*got[:1], b.camera, b.epoch, got[1]) if got else None))

    title = _title(fire_id, truth, tier,
                   f"      {len({b.camera.split('-')[0] for b in bs})} sites"
                   f"      error {err:.2f} km      95% region {area:.1f} km²")
    fig = _draw(title, view, _hillshade(view["clat"], view["clon"], view["half_km"]),
                cams, [(b.camera, col) for b, col in zip(bs, PALETTE)],
                {b.camera: b for b in bs}, (lats, lons, ll), (elat, elon), truth,
                _inset_spec(bs, truth, elat, elon, err, area, view), crops)

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
