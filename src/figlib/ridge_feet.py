"""Is the plume's foot visible, or behind a ridge? Terrain under the star-solved pose.

Two outputs, both drawn from the Copernicus DEM and the pose ledger alone, never from pixels:

* `align <seq>...` -- the nested ridge stack predicted for a pre-ignition frame, under the
  star-solved pose and fisheye lens and, for contrast, under the published pose with the
  rectilinear lens the pipeline used to assume. A by-eye check that the calibration lands
  terrain where it is, which the star fit never looked at.
* `edges` -- same-day daytime frames from the CDN for every star solve, and the vertical shift
  between the real skyline and the one the pose predicts. (`skyline` is the sky-mask version
  it replaced, kept because its failure is documented in NOTES.md.)
* `figure` -- docs/figures/terrain_hidden_ignition.jpg from the two outputs above and below.
* `feet` -- for every confirmed-tier fire and camera, whether the official ignition point
  is in line of sight from the camera, which ridge hides it if not, and where the
  detector's box bottoms sit against the row the foot should appear at: the ignition's own
  row when visible, the occluding crest's row when hidden.

The star pose is taken from the ledger entries `pose_ledger.lookup` would use for that date
(it averages d_az; here pitch, roll and the lens are averaged over the same sources). A
3072-wide frame with no applicable solve keeps its published azimuth with the shared lens;
other frame formats are skipped, since their lens was never measured.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from . import corpus as C
from . import pose_ledger
from . import terrain as T
from .detect_yolo import read_frames
from .stars.fisheye import initial_k, project_fisheye

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "ridges" / "feet"
CAMS = json.loads((ROOT / "data/meta/cams.json").read_text())
SEQS = {s["seq"]: s for s in json.loads((ROOT / "data/meta/all/sequences.json").read_text())}
FIRES = {f["fire_id"]: f for f in json.loads((ROOT / "data/meta/fires.json").read_text())}
SIZES = json.loads((ROOT / "data/meta/frame_sizes.json").read_text())
LEDGER = pose_ledger.load()
DETS = C.CORPORA["all"].dets
K_RATIO, K1 = 0.886, -0.078


def pose_for(camera: str, epoch: float, frame_w: int, nearest: bool = False) -> dict | None:
    """Full star pose (az, pitch, roll, lens) for a camera on a date, or None.

    `nearest` falls back to the closest solve in time when the ledger's rules decline --
    right for eyeballing whether a solve lands the terrain, wrong for scoring, since the
    ledger declines exactly where a re-aim may sit between the two dates.
    """
    if frame_w != 3072:
        return None
    hit = pose_ledger.lookup(LEDGER, camera, epoch, frame_w)
    if hit is None and nearest:
        mine = [e for e in LEDGER if e["camera"] == camera and e.get("frame_w") == frame_w]
        if mine:
            e = min(mine, key=lambda e: abs(e["epoch"] - epoch))
            hit = {"d_az": e["d_az"], "sources": [e["source"]],
                   "rule": f"nearest, {abs(e['epoch'] - epoch) / 86400:.0f} d away"}
    if hit is None:
        return {"d_az": 0.0, "d_pitch": 0.0, "d_roll": 0.0, "k_ratio": K_RATIO, "k1": K1,
                "rule": "published+lens"}
    src = [e for e in LEDGER if e["source"] in hit["sources"]]
    mean = lambda k: float(np.mean([e[k] for e in src]))
    return {"d_az": hit["d_az"], "d_pitch": mean("d_pitch"), "d_roll": mean("d_roll"),
            "k_ratio": mean("k_ratio"), "k1": mean("k1"), "rule": hit["rule"]}


def proj(cam, pose, az, el, W, H):
    k = pose["k_ratio"] * initial_k(cam, W)
    x, y = project_fisheye(cam, az, el, W, H, pose["d_az"], pose["d_pitch"], pose["d_roll"],
                           k, pose["k1"])
    return x * W, y * H


def frames(seq: str):
    tgz = {p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}[seq.split("#")[0]]
    return sorted(read_frames(tgz), key=lambda f: f[1])


def decode(blob):
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def range_color(km: float) -> tuple[int, int, int]:
    """Plasma, near (dark purple) to far (yellow), 2-40 km on a log scale."""
    t = np.clip(math.log(max(km, 2) / 2) / math.log(20), 0, 1)
    c = cv2.applyColorMap(np.uint8([[int(40 + 215 * t)]]), cv2.COLORMAP_PLASMA)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def draw_ridges(img, field, xs, ys, thick=2, color=None):
    W, H = img.shape[1], img.shape[0]
    for cid in np.unique(field.layer):
        sel = np.flatnonzero(field.layer == cid)
        if sel.size < 6:
            continue
        sel = sel[np.argsort(field.az_deg[sel])]
        p = np.c_[xs[sel], ys[sel]]
        ok = (p[:, 0] > -50) & (p[:, 0] < W + 50) & (p[:, 1] > -50) & (p[:, 1] < H + 50)
        if ok.sum() < 2:
            continue
        col = color or range_color(float(np.median(field.range_km[sel])))
        cv2.polylines(img, [p[ok].astype(np.int32)], False, col, thick, cv2.LINE_AA)


def field_for(cam, pose):
    shifted = {**cam, "az": cam["az"] + pose["d_az"]}
    return T.ridges(shifted, DEM, half_fov_pad=14.0)


def banner(W, lines, h=44):
    b = np.zeros((h * len(lines) + 12, W, 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(b, t, (16, 34 + h * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return b


def align(seq: str) -> Path:
    s = SEQS[seq]; camera = s["camera"]; cam = CAMS[camera]
    epoch, off, blob = frames(seq)[0]
    img = decode(blob); H, W = img.shape[:2]
    pose = pose_for(camera, epoch, W, nearest=True)
    if pose is None:
        raise SystemExit(f"{seq}: {W}px frames, lens not measured")
    field = field_for(cam, pose)
    xs, ys = proj(cam, pose, field.az_deg, field.elev_deg, W, H)
    star = img.copy(); draw_ridges(star, field, xs, ys)
    xr, yr = T.project(cam, field.az_deg, field.elev_deg, W, H)
    pub = img.copy(); draw_ridges(pub, field, xr * W, yr * H, color=(0, 140, 255))

    inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    lo = int(max(0, np.percentile(ys[inb], 1) - 140)); hi = int(min(H, np.percentile(ys[inb], 99) + 140))
    strips = [banner(W, [f"{seq}  offset {off}s  raw"]), img[lo:hi],
              banner(W, [f"star pose ({pose['rule']}): d_az {pose['d_az']:+.2f}  pitch {pose['d_pitch']:+.2f}"
                         f"  roll {pose['d_roll']:+.2f}  k/nameplate {pose['k_ratio']:.3f}   colour = ridge range, near purple to far yellow"]),
              star[lo:hi],
              banner(W, ["published pose, rectilinear lens (what the pipeline assumed before)"]),
              pub[lo:hi]]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"align_{seq.replace('#', '_')}.jpg"
    cv2.imwrite(str(dest), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return dest


def sightline(cam, lat, lon):
    """Ignition geometry from the camera: az, el, km, and the highest nearer terrain."""
    m_lat, m_lon = 111_132.0, 111_320.0 * math.cos(math.radians(cam["lat"]))
    dn, de = (lat - cam["lat"]) * m_lat, (lon - cam["lon"]) * m_lon
    D = math.hypot(dn, de); az = math.degrees(math.atan2(de, dn)) % 360
    band, tf = DEM.window(cam["lat"], cam["lon"], D / 111_000 + 0.08)
    h_cam = cam["elev"] + (cam.get("agl") or 0.0)
    d = np.arange(60.0, D - 150.0, 30.0)
    la = cam["lat"] + d * math.cos(math.radians(az)) / m_lat
    lo = cam["lon"] + d * math.sin(math.radians(az)) / m_lon
    h = T.Dem.sample(band, tf, la, lo)
    ang = np.degrees(np.arctan2(h - h_cam - d ** 2 / (2 * T.R_EFF), d))
    h_t = float(T.Dem.sample(band, tf, np.array([lat]), np.array([lon]))[0])
    el = math.degrees(math.atan2(h_t - h_cam - D ** 2 / (2 * T.R_EFF), D))
    i = int(np.argmax(ang)) if ang.size else 0
    occ_el = float(ang[i]) if ang.size else -90.0
    return {"az": az, "el": el, "km": D / 1000, "occ_el": occ_el,
            "occ_km": float(d[i] / 1000) if ang.size else 0.0}


def neighbourhood_visible(cam, lat, lon, radius_m=300.0, n=16):
    """Fraction of points on a ring around the ignition in line of sight -- WFIGS points
    are good to a few hundred metres, so a single point's visibility can flip on noise."""
    vis = []
    for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
        la = lat + radius_m * math.cos(a) / 111_132.0
        lo = lon + radius_m * math.sin(a) / (111_320.0 * math.cos(math.radians(lat)))
        g = sightline(cam, la, lo)
        vis.append(g["el"] >= g["occ_el"])
    return float(np.mean(vis))


def feet(conf_thr: float = 0.1, n_early: int = 3) -> list[dict]:
    geo = json.loads((ROOT / "out/geolocation_fisheye_ledger.json").read_text())
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for g in geo:
        if g["tier"] != "confirmed":
            continue
        lat, lon = g["truth_lat"], g["truth_lon"]
        for seq in FIRES[g["fire_id"]]["sequences"]:
            s = SEQS.get(seq)
            if not s or not s["has_pose"]:
                continue
            base = seq.split("#")[0]
            W, H = (SIZES.get(base) or [None, None])[:2]
            camera = s["camera"]; cam = CAMS[camera]
            pose = pose_for(camera, s["t0"], W)
            if pose is None:
                rows.append({"fire": g["fire_id"], "camera": camera, "skip": f"{W}px lens"})
                continue
            geom = sightline(cam, lat, lon)
            tx, ty = proj(cam, pose, np.array([geom["az"]]), np.array([geom["el"]]), W, H)
            _, oy = proj(cam, pose, np.array([geom["az"]]), np.array([geom["occ_el"]]), W, H)
            tx, ty, oy = float(tx[0]), float(ty[0]), float(oy[0])
            hidden = geom["occ_el"] > geom["el"]
            foot_row = oy if hidden else ty
            row = {"fire": g["fire_id"], "camera": camera, "seq": seq, "rule": pose["rule"],
                   "km": round(geom["km"], 1), "el": round(geom["el"], 2),
                   "occ_el": round(geom["occ_el"], 2), "occ_km": round(geom["occ_km"], 1),
                   "hidden_deg": round(geom["occ_el"] - geom["el"], 2),
                   "ring_visible": neighbourhood_visible(cam, lat, lon),
                   "truth_x": round(tx), "truth_y": round(ty), "occ_y": round(oy),
                   "in_frame": bool(0 <= tx < W)}
            # Earliest post-ignition boxes spanning the ignition's column (+-3% of width).
            path = DETS / f"{base}.json"
            hits = []
            if path.exists() and row["in_frame"]:
                for rec in sorted(json.loads(path.read_text()), key=lambda r: r["offset"]):
                    if rec["offset"] < 0:
                        continue
                    for d in rec["dets"]:
                        if d["conf"] >= conf_thr and d["x0"] * W - 0.03 * W <= tx <= d["x1"] * W + 0.03 * W:
                            hits.append((rec["offset"], d))
                            break
                    if len(hits) >= n_early:
                        break
            row["box_bottoms"] = [round(d["y1"] * H) for _o, d in hits]
            row["box_offsets"] = [o for o, _d in hits]
            row["foot_minus_box"] = (round(float(np.median([d["y1"] * H for _o, d in hits]) - foot_row))
                                     if hits else None)
            rows.append(row)
            if hits:
                panel(row, pose, cam, hits[0], W, H)
            print(json.dumps(row), flush=True)
    (OUT / "feet.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def panel(row, pose, cam, hit, W, H):
    off, d = hit
    blob = next(b for _e, o, b in frames(row["seq"]) if o == off)
    img = decode(blob)
    field = field_for(cam, pose)
    xs, ys = proj(cam, pose, field.az_deg, field.elev_deg, W, H)
    draw_ridges(img, field, xs, ys)
    cv2.rectangle(img, (int(d["x0"] * W), int(d["y0"] * H)), (int(d["x1"] * W), int(d["y1"] * H)),
                  (80, 255, 80), 3)
    tx, ty, oy = row["truth_x"], row["truth_y"], row["occ_y"]
    cv2.drawMarker(img, (tx, ty), (255, 255, 255), cv2.MARKER_CROSS, 40, 3)
    if row["hidden_deg"] > 0:
        cv2.line(img, (tx - 60, oy), (tx + 60, oy), (255, 255, 255), 2, cv2.LINE_AA)
    x0 = int(np.clip(tx - 500, 0, W - 1000)); y0 = int(np.clip(min(ty, oy) - 380, 0, H - 600))
    crop = img[y0:y0 + 600, x0:x0 + 1000]
    state = (f"HIDDEN {row['hidden_deg']:.2f} deg behind crest at {row['occ_km']} km"
             if row["hidden_deg"] > 0 else f"visible, clear by {-row['hidden_deg']:.2f} deg")
    out = np.vstack([banner(1000, [f"{row['fire']}  {row['camera']}  +{off}s",
                                   f"ignition {row['km']} km  {state}",
                                   f"ring visible {row['ring_visible']:.0%}  foot-box {row['foot_minus_box']} px"], h=38),
                     crop])
    cv2.imwrite(str(OUT / f"foot_{row['fire']}_{row['camera']}.jpg"), out,
                [cv2.IMWRITE_JPEG_QUALITY, 88])


def skyline_rows(cam, pose, W, H, step=0.1):
    """Predicted skyline row per image column under a pose, and the skyline's range."""
    shifted = {**cam, "az": cam["az"] + pose["d_az"]}
    prof = T.horizon(shifted, DEM, half_fov_pad=14.0, step_deg=step)
    x, y = proj(cam, pose, prof.az_deg, prof.elev_deg, W, H)
    o = np.argsort(x)
    x, y, rng = x[o], y[o], prof.range_km[o]
    cols = np.arange(W)
    ok = (cols >= x[0]) & (cols <= x[-1])
    return (np.where(ok, np.interp(cols, x, y), np.nan),
            np.where(ok, np.interp(cols, x, rng), np.nan))


def skyline_check(q: int = 4, n_frames: int = 3) -> list[dict]:
    """Observed minus predicted skyline row, per star solve, on a daytime frame of the same day.

    Same-day frames from the CDN, so the pose cannot have changed between the star solve
    and the check. The sky mask's lowest pixel per column is the silhouette
    (viz_terrain.observed_skyline); columns where it disagrees with prediction by more
    than 60 px are masts, cloud or haze rather than pose, and are dropped before the
    median. Split by skyline range, a pitch error shifts near and far alike, while a
    camera-height error moves near crests much more than far ones.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .stars import nights
    from .viz_terrain import observed_skyline
    solves = [e for e in LEDGER if e["source"].startswith("star:hpwren_") and e["frame_w"] == 3072]
    jobs = [(e["camera"], e["source"].split("_")[1]) for e in solves]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: nights.fetch(j[0], j[1], q=q, n_frames=n_frames), jobs))
    out = []
    (OUT / "skyline").mkdir(parents=True, exist_ok=True)
    for e in solves:
        cam = CAMS[e["camera"]]; day = e["source"].split("_")[1]
        paths = sorted((nights.FRAMES / e["camera"] / f"{day}_Q{q}").glob("*.jpg"))
        best = None
        for pth in paths:
            img = cv2.imread(str(pth))
            if img is None:
                continue
            obs = observed_skyline(img)
            if best is None or np.isfinite(obs).mean() > np.isfinite(best[1]).mean():
                best = (img, obs, pth)
        if best is None:
            print(e["camera"], "no daytime frame"); continue
        img, obs, pth = best
        H, W = img.shape[:2]
        if W != 3072:
            print(e["camera"], "frame", W); continue
        pose = {k: e[k] for k in ("d_az", "d_pitch", "d_roll", "k_ratio", "k1")}
        pred, rng = skyline_rows(cam, pose, W, H)
        dy = obs - pred
        m = np.isfinite(dy) & (np.abs(dy) < 60)
        cols = np.arange(W)
        row = {"camera": e["camera"], "source": e["source"], "frame": pth.name,
               "d_pitch": e["d_pitch"], "n_cols": int(m.sum())}
        if m.sum() > 200:
            a, b = np.polyfit((cols[m] - W / 2) / 1000, dy[m], 1)
            near, far = m & (rng < 10), m & (rng > 25)
            row.update({"dy_med": round(float(np.median(dy[m])), 1),
                        "dy_iqr": round(float(np.subtract(*np.percentile(dy[m], [75, 25]))), 1),
                        "slope_px_per_1000": round(float(a), 1),
                        "dy_near": round(float(np.median(dy[near])), 1) if near.sum() > 100 else None,
                        "dy_far": round(float(np.median(dy[far])), 1) if far.sum() > 100 else None,
                        "frac_within60": round(float(m.sum() / max(np.isfinite(dy).sum(), 1)), 2)})
        out.append(row)
        print(json.dumps(row), flush=True)
        vis = img.copy()
        for arr, col in ((pred, (255, 60, 255)), (obs, (80, 255, 80))):
            ok = np.isfinite(arr)
            for x0 in np.flatnonzero(ok)[::3]:
                cv2.circle(vis, (int(x0), int(arr[x0])), 2, col, -1)
        yc = int(np.nanmedian(pred)) if np.isfinite(pred).any() else H // 2
        crop = vis[max(0, yc - 350):yc + 350]
        crop = cv2.resize(crop, (1536, crop.shape[0] // 2), interpolation=cv2.INTER_AREA)
        crop = np.vstack([banner(1536, [f"{e['camera']} {day} {pth.stem}  magenta predicted skyline (star pose), green observed  median dy {row.get('dy_med')} px"], h=40), crop])
        cv2.imwrite(str(OUT / "skyline" / f"{e['camera']}_{day}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT / "skyline" / "skyline_check.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def edge_score(img: np.ndarray) -> np.ndarray:
    """Per-pixel strength of a sky-above / terrain-below step.

    The sky mask in viz_terrain.observed_skyline calls hazy far ridges sky, which puts its
    silhouette on the nearest crisp ridge instead (seen on bm-n, om-e, mlo-s). Distant
    ridges are faint but *horizontal*, so smooth along rows first -- that lifts a long faint
    crest above texture -- then take the downward drop in brightness and in blueness.
    """
    f = img.astype(np.float32)
    gray = f.mean(axis=2)
    blue = f[:, :, 0] - f[:, :, 2]
    k = (31, 3)
    gray = cv2.blur(gray, k); blue = cv2.blur(blue, k)
    d = lambda a: np.pad(a[:-4] - a[4:], ((2, 2), (0, 0)))      # above minus below
    return d(gray) + 0.5 * d(blue)


def skyline_shift(img: np.ndarray, pred: np.ndarray, max_shift: int = 45):
    """Vertical shift of the observed skyline against the predicted one, by total edge score.

    Returns (global shift px, left-half shift, right-half shift, peak sharpness), shift > 0
    meaning the real skyline sits lower in the image than predicted.
    """
    H, W = img.shape[:2]
    e = edge_score(img)
    cols = np.flatnonzero(np.isfinite(pred) & (pred > max_shift + 3) & (pred < H - max_shift - 3))
    if cols.size < 300:
        return None
    base = np.rint(pred[cols]).astype(int)
    shifts = np.arange(-max_shift, max_shift + 1)
    S = np.stack([e[base + sft, cols] for sft in shifts])        # (n_shift, n_cols)
    S = np.clip(S, 0, np.percentile(S, 99))                      # one bright mast cannot win
    half = cols < W / 2
    best = lambda m: int(shifts[np.argmax(S[:, m].sum(axis=1))]) if m.sum() > 150 else None
    tot = S.sum(axis=1)
    sharp = float((tot.max() - np.median(tot)) / (np.std(tot) + 1e-9))
    return best(np.ones_like(half)), best(half), best(~half), sharp


def skyline_edges(q: int = 4) -> list[dict]:
    """Same-day skyline offsets for every CDN star solve, on each daytime frame fetched."""
    from .stars import nights
    solves = [e for e in LEDGER if e["source"].startswith("star:hpwren_") and e["frame_w"] == 3072]
    rows = []
    dest = OUT / "skyline_edges"; dest.mkdir(parents=True, exist_ok=True)
    for e in solves:
        cam = CAMS[e["camera"]]; day = e["source"].split("_")[1]
        pose = {k: e[k] for k in ("d_az", "d_pitch", "d_roll", "k_ratio", "k1")}
        paths = sorted((nights.FRAMES / e["camera"] / f"{day}_Q{q}").glob("*.jpg"))
        pred = None; per = []
        for pth in paths:
            img = cv2.imread(str(pth))
            if img is None or img.shape[1] != 3072:
                continue
            if pred is None:
                pred, rng = skyline_rows(cam, pose, img.shape[1], img.shape[0])
            r = skyline_shift(img, pred)
            if r:
                per.append(r)
        if not per:
            print(e["camera"], day, "no usable frame", flush=True); continue
        g = [p[0] for p in per]
        row = {"camera": e["camera"], "day": day, "d_pitch": e["d_pitch"], "n_frames": len(per),
               "shift_px": float(np.median(g)), "frames": g,
               "left_px": per[0][1], "right_px": per[0][2],
               "sharpness": round(float(np.median([p[3] for p in per])), 1),
               "skyline_km_median": round(float(np.nanmedian(rng)), 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        img = cv2.imread(str(paths[0]))
        vis = img.copy()
        cols = np.flatnonzero(np.isfinite(pred))
        for x0 in cols[::2]:
            cv2.circle(vis, (int(x0), int(pred[x0])), 1, (255, 60, 255), -1)
            cv2.circle(vis, (int(x0), int(pred[x0] + row["shift_px"])), 1, (80, 255, 80), -1)
        yc = int(np.nanmedian(pred))
        crop = vis[max(0, yc - 200):yc + 200]
        cv2.imwrite(str(dest / f"{e['camera']}_{day}.jpg"),
                    np.vstack([banner(3072, [f"{e['camera']} {day}: magenta = predicted skyline under the star pose, green = best-fit shift {row['shift_px']:+.0f} px  (left {row['left_px']}, right {row['right_px']})"]), crop]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (dest / "skyline_edges.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def figure() -> Path:
    """docs/figures/terrain_hidden_ignition.jpg, from outputs of `feet` and `edges`."""
    W = 2000
    def cap(lines, width):
        lines = [lines] if isinstance(lines, str) else lines
        b = np.full((16 + 34 * len(lines), width, 3), 252, np.uint8)
        for i, t in enumerate(lines):
            cv2.putText(b, t, (14, 36 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        return b
    a = cv2.imread(str(OUT / "foot_20260629_JunctionFire_mg-e-mobo-c.jpg"))[126:]   # drop debug banner
    b = cv2.imread(str(OUT / "foot_20251102_ScissorsFire_vo-e-mobo-c.jpg"))[126:]
    top = np.hstack([np.vstack([cap(["JunctionFire from mg-e: ignition 5.3 km away,", "hidden 5.9 deg behind a crest 1.8 km out"], 1000), a]),
                     np.vstack([cap(["ScissorsFire from vo-e: ignition 14.3 km away,", "hidden 2.4 deg behind a crest 3.3 km out"], 1000), b])])
    sky = cv2.imread(str(OUT / "skyline_edges" / "hp-e-mobo-c_20260911.jpg"))[56:]
    sky = cv2.resize(sky, (W, int(sky.shape[0] * W / sky.shape[1])), interpolation=cv2.INTER_AREA)
    legend = cap(["Green box: first detections.  White cross: official ignition.  White bar: the crest that hides it.",
                  "Lines: DEM ridges under the star-solved pose, purple near to yellow far. No pixels were fitted."], W)
    shift = next(r["shift_px"] for r in json.loads((OUT / "skyline_edges" / "skyline_edges.json").read_text())
                 if r["camera"] == "hp-e-mobo-c" and r["day"] == "20260911")
    where = ("on the prediction" if shift == 0 else
             f"{abs(shift):.0f} px {'higher' if shift < 0 else 'lower'}")
    legend2 = cap(["hp-e-mobo-c, 2026-09-11 09:00, the same day as its star solve.",
                   f"Magenta: skyline predicted from the DEM and star pose alone.  Green: best-fit image edge, {where}."], W)
    sheet = np.vstack([top, legend, sky, legend2])
    dest = ROOT / "docs" / "figures" / "terrain_hidden_ignition.jpg"
    cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return dest


DEM = T.Dem()

if __name__ == "__main__":
    if sys.argv[1] == "figure":
        print(figure())
    elif sys.argv[1] == "edges":
        skyline_edges()
    elif sys.argv[1] == "skyline":
        skyline_check()
    elif sys.argv[1] == "align":
        for q in sys.argv[2:]:
            print(align(q))
    elif sys.argv[1] == "feet":
        feet()
