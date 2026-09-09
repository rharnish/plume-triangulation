"""Animate a fire: what the cameras saw, and what the posterior believed.

Two panels moving together. On the left the camera frames with the detector's boxes, so
the evidence arriving is visible. On the right the posterior over the ground, rebuilt
from every detection seen so far, with the bearings that produced it.

The pairing is the point. A still image of the final answer says nothing about when it
became knowable, and the interesting behaviour is all in the transient: a single camera
gives a fan with no depth at all, the second collapses it to a blob, and a confident
false positive on a cloud swings the surface until the other cameras outvote it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .accumulate import gather, posterior
from .detect_yolo import read_frames
from .geom import haversine_km, offset_bearing_deg

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
YOLO_DIR = ROOT / "out" / "yolo"

MAP_PX = 720
STRIP_H = 92          # error-vs-time trace under the map
CAM_W = 480


def _map_panel(lats, lons, ll, est, truth, cam_pts, bearings, px=MAP_PX, trail=None):
    """Posterior as an image: magma heat, camera markers, bearing rays, estimate, truth."""
    rel = ll - ll.max()
    if not np.any(ll):                       # nothing observed yet: draw it as empty
        rel = np.full_like(ll, -12.0)
    img = np.clip((rel + 12.0) / 12.0, 0, 1)
    img = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    img = cv2.resize(img, (px, px), interpolation=cv2.INTER_LINEAR)
    img = cv2.flip(img, 0)                       # lat increases upward

    # Once the posterior sharpens, the heat map is a single bright pixel on black and
    # the shape of the uncertainty disappears. Contours keep it legible throughout.
    for lvl, col in ((-6.0, (120, 70, 40)), (-3.0, (200, 140, 70)), (-1.0, (255, 235, 200))):
        m = cv2.flip((rel >= lvl).astype(np.uint8), 0)
        m = cv2.resize(m, (px, px), interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, col, 1, cv2.LINE_AA)

    def to_px(lat, lon):
        x = (lon - lons[0]) / (lons[-1] - lons[0]) * (px - 1)
        y = (1 - (lat - lats[0]) / (lats[-1] - lats[0])) * (px - 1)
        return int(round(x)), int(round(y))

    span = max(lats[-1] - lats[0], lons[-1] - lons[0])
    for (clat, clon), brg in bearings:
        th = math.radians(brg)
        p = to_px(clat, clon)
        q = to_px(clat + math.cos(th) * span,
                  clon + math.sin(th) * span / math.cos(math.radians(clat)))
        cv2.line(img, p, q, (160, 210, 77), 1, cv2.LINE_AA)
    for clat, clon in cam_pts:
        cv2.drawMarker(img, to_px(clat, clon), (160, 210, 77),
                       cv2.MARKER_TRIANGLE_UP, 13, 2)
    if trail:
        pts = [to_px(a, b) for a, b in trail]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (109, 77, 255), 1, cv2.LINE_AA)

    tp = to_px(*truth)
    if 0 <= tp[0] < px and 0 <= tp[1] < px:
        cv2.circle(img, tp, 13, (102, 209, 255), 3, cv2.LINE_AA)
    else:
        # Say so rather than silently omitting it, so a viewer is never left assuming
        # the truth is somewhere inside the frame.
        q = (int(np.clip(tp[0], 14, px - 14)), int(np.clip(tp[1], 14, px - 14)))
        cv2.drawMarker(img, q, (102, 209, 255), cv2.MARKER_DIAMOND, 20, 3)
        cv2.putText(img, "truth off-frame", (q[0] - 70, min(px - 6, q[1] + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 209, 255), 1, cv2.LINE_AA)
    if est is not None:
        ep = to_px(*est)
        cv2.drawMarker(img, ep, (109, 77, 255), cv2.MARKER_TILTED_CROSS, 20, 3)
    return img


def _label(img, lines, org=(12, 26), scale=0.6, color=(255, 255, 255)):
    for i, t in enumerate(lines):
        y = org[1] + i * int(26 * scale / 0.6)
        cv2.putText(img, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)


def animate(fire_id: str, fps: int = 6, max_cams: int = 4,
            half_extent_km: float = 35.0, step_km: float = 0.3,
            out: Path | None = None) -> Path | None:
    cams = json.loads((META / "cams.json").read_text())
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    res = {r["fire_id"]: r for r in json.loads((META / "resolved.json").read_text())}

    rec = res.get(fire_id)
    if not rec or not rec.get("truth"):
        print(f"  {fire_id}: no truth")
        return None
    truth = (rec["truth"]["lat"], rec["truth"]["lon"])
    fire = fires[fire_id]

    # Cameras with pose and detections, most productive first
    chosen = []
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        dj = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not dj.exists():
            continue
        recs = json.loads(dj.read_text())
        n = sum(len(r["dets"]) for r in recs)
        chosen.append((n, seq_name, s["camera"], recs))
    # Take the most productive camera at each *site* before taking a second from any
    # site. Ranking by detection count alone picked four cameras from one site on the
    # Club fire -- two of them the same camera twice, since that archive holds two
    # annotation passes -- leaving the solve with one site and nothing to triangulate.
    chosen.sort(reverse=True, key=lambda z: z[0])
    by_site, extras, seen = {}, [], set()
    for item in chosen:
        camera = item[2]
        if camera in seen:
            continue
        seen.add(camera)
        site = camera.split("-")[0]
        (by_site.setdefault(site, item) and None) if site in by_site else None
        if site not in by_site:
            by_site[site] = item
        else:
            extras.append(item)
    chosen = (list(by_site.values()) + extras)[:max_cams]
    if len(chosen) < 2:
        print(f"  {fire_id}: fewer than 2 usable cameras")
        return None

    frames_by_cam, dets_by_cam = {}, {}
    for _, seq_name, camera, recs in chosen:
        tgz = ROOT / "data" / "tgz" / f"{seq_name.split('#')[0]}.tgz"
        frames_by_cam[camera] = {o: b for _, o, b in read_frames(tgz)}
        dets_by_cam[camera] = {r["offset"]: r["dets"] for r in recs}

    centre = (float(np.mean([cams[c[2]]["lat"] for c in chosen])),
              float(np.mean([cams[c[2]]["lon"] for c in chosen])))
    offsets = sorted({o for m in dets_by_cam.values() for o in m})
    offsets = [o for o in offsets if -600 <= o <= 2400]

    rows = math.ceil(len(chosen) / 2)
    cam_h = int(CAM_W * 0.75)
    left_w, left_h = CAM_W * 2, cam_h * rows
    height = max(left_h, MAP_PX + STRIP_H)
    out = out or ROOT / "out" / "videos" / f"{fire_id}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (left_w + MAP_PX, height))

    trail: list[tuple[float, float]] = []
    hist: list[tuple[int, float]] = []
    for off in offsets:
        panels = []
        for _, seq_name, camera, _ in chosen:
            fr = frames_by_cam[camera]
            key = min(fr, key=lambda x: abs(x - off)) if fr else None
            img = (cv2.imdecode(np.frombuffer(fr[key], np.uint8), cv2.IMREAD_COLOR)
                   if key is not None and abs(key - off) <= 90 else None)
            if img is None:
                img = np.full((cam_h, CAM_W, 3), 25, np.uint8)
            else:
                H, W = img.shape[:2]
                for d in dets_by_cam[camera].get(key, []):
                    if d["conf"] < 0.10:
                        continue
                    c = (0, 0, 255) if d["conf"] >= 0.3 else (0, 165, 255)
                    cv2.rectangle(img, (int(d["x0"] * W), int(d["y0"] * H)),
                                  (int(d["x1"] * W), int(d["y1"] * H)), c, 3)
                img = cv2.resize(img, (CAM_W, cam_h))
            _label(img, [camera], scale=0.55)
            panels.append(img)
        while len(panels) < rows * 2:
            panels.append(np.full((cam_h, CAM_W, 3), 25, np.uint8))
        left = np.vstack([np.hstack(panels[i * 2:i * 2 + 2]) for i in range(rows)])

        det = gather(fire, seqs, cams, max(off, 0), conf_thr=0.10)
        det = {k: v for k, v in det.items() if k in dets_by_cam}
        est, err, lats, lons, ll = None, None, None, None, None
        bearing_rays, cam_pts = [], [(cams[c[2]]["lat"], cams[c[2]]["lon"])
                                     for c in chosen]
        if len({k.split("-")[0] for k in det}) >= 2:
            # Coarse pass first, then a fine grid about its peak. Centring the fine grid
            # on the cameras instead is a trap: with sites 80 km from the fire the true
            # peak can fall outside a 35 km window entirely, and argmax then returns an
            # edge cell -- which read as a 19 km error on Ranch2 that was pure artefact.
            _, _, _, cla, clo = posterior(det, cams, centre, half_extent_km=90.0,
                                          step_km=1.5, alpha=0.25)
            lats, lons, ll, la, lo = posterior(det, cams, (cla, clo),
                                               half_extent_km=half_extent_km,
                                               step_km=step_km, alpha=0.25)
            est = (la, lo)
            err = haversine_km(la, lo, *truth)
            if not trail or trail[-1] != est:
                trail.append(est)
            hist.append((off, err))
            for camera, ds in det.items():
                b = max(ds, key=lambda z: z[1])[0]
                bearing_rays.append(((cams[camera]["lat"], cams[camera]["lon"]), b))
        if ll is None:
            n = int(half_extent_km / step_km)
            lats = centre[0] + np.arange(-n, n + 1) * (step_km / 111.32)
            lons = centre[1] + np.arange(-n, n + 1) * (
                step_km / (111.32 * math.cos(math.radians(centre[0]))))
            ll = np.zeros((len(lats), len(lons)))

        right = _map_panel(lats, lons, ll, est, truth, cam_pts, bearing_rays,
                           trail=trail)
        # error-vs-time strip: the whole point is that this is a trajectory, not a number
        strip = np.full((STRIP_H, MAP_PX, 3), 18, np.uint8)
        if hist:
            emax = max(4.0, max(e for _, e in hist) * 1.1)
            for gx in range(0, MAP_PX, MAP_PX // 6):
                cv2.line(strip, (gx, 0), (gx, STRIP_H), (38, 38, 38), 1)
            pts = [(int((o - offsets[0]) / (offsets[-1] - offsets[0]) * (MAP_PX - 1)),
                    int(88 - (e / emax) * 80)) for o, e in hist]
            for a, b in zip(pts, pts[1:]):
                cv2.line(strip, a, b, (109, 77, 255), 2, cv2.LINE_AA)
            cv2.circle(strip, pts[-1], 4, (255, 255, 255), -1)
            _label(strip, [f"error vs time   0 - {emax:.0f} km"], org=(8, 18), scale=0.5)
        right = np.vstack([right, strip])
        _label(right, [
            f"t {off:+d} s from plume appearance",
            f"cameras detecting: {len(det)}   detections so far: "
            f"{sum(len(v) for v in det.values())}",
            (f"error {err:.2f} km" if err is not None else "not yet locatable"),
        ], scale=0.62)

        if left.shape[0] != height:
            left = cv2.copyMakeBorder(left, 0, height - left.shape[0], 0, 0,
                                      cv2.BORDER_CONSTANT, value=(25, 25, 25))
        if right.shape[0] != height:
            right = cv2.copyMakeBorder(right, 0, height - right.shape[0], 0, 0,
                                       cv2.BORDER_CONSTANT, value=(25, 25, 25))
        vw.write(np.hstack([left, right]))

    vw.release()
    print(f"  {fire_id}: {len(offsets)} frames -> {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    import sys
    for fid in (sys.argv[1:] or ["20240701_Kitchenfire"]):
        animate(fid)
