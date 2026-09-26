"""Sea horizon as an independent check on star-solved pitch and roll.

Over open ocean the horizon's apparent elevation depends only on camera height: the dip
below level is -acos(R/(R+h)) with R the 4/3 refraction-adjusted Earth radius (terrain.py's
R_EFF). No ephemeris, no catalog, no terrain model beyond "this ray ends at sea". So drawing
that line through the star-solved pose tests pitch and roll (and the lens) with nothing in
common with the star fit. Azimuths count as sea when the DEM along the last 30 km of an
80 km ray is exactly 0 m, west of the coast, and nothing nearer rises above the dip.

Draws, on one daytime frame per camera: magenta = sea horizon under the star-solved pose
(from the pose ledger entry nearest in time) + fitted lens; orange = the published pose with
the same lens.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .. import corpus as C
from .. import terrain as T
from ..geom import ray_latlon
from ..detect_yolo import read_frames
from .fisheye import initial_k, project_fisheye

ROOT = Path(__file__).resolve().parents[3]
SKY = ROOT / "out" / "sky"

CAMS = json.loads((ROOT / "data/meta/cams.json").read_text())
SEQS = {s["seq"]: s for s in json.loads((ROOT / "data/meta/all/sequences.json").read_text())}
LEDGER = json.loads((ROOT / "data/meta/pose_ledger.json").read_text())
K_RATIO, K1 = 0.886, -0.078


def sea_azimuths(cam: dict, dem: T.Dem, max_km: float = 80.0) -> tuple[np.ndarray, float]:
    half = cam["fov"] / 2.0 + 12
    az = np.arange(cam["az"] - half, cam["az"] + half + 1e-9, 0.25)
    d = np.arange(500.0, max_km * 1000.0, 250.0)
    band, tf = dem.window(cam["lat"], cam["lon"], max_km / 111.0 + 0.05)
    lats, lons = ray_latlon(cam["lat"], cam["lon"], az[:, None], d[None, :])
    h = T.Dem.sample(band, tf, lats.ravel(), lons.ravel()).reshape(lats.shape)
    covered = (lats >= 31) & (lats < 35) & (lons >= -119) & (lons < -116)
    h_cam = T.eye_height(cam)
    dip = -math.degrees(math.acos(T.R_EFF / (T.R_EFF + h_cam)))
    ang = T.sight_angles(h, h_cam, d[None, :])
    tail = d >= (max_km - 30) * 1000.0
    sea = (np.all((np.abs(h[:, tail]) < 0.5) & covered[:, tail], axis=1) & (lons[:, -1] < -117.1)
           & (ang.max(axis=1) <= dip + 0.05))
    return az[sea], dip


def render(seq: str, offset_pick: str = "first") -> Path:
    s = SEQS[seq]; cam_name = s["camera"]; cam = CAMS[cam_name]
    frames = sorted(read_frames({p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}[seq.split("#")[0]]),
                    key=lambda f: f[1])
    epoch, off, blob = frames[0] if offset_pick == "first" else frames[len(frames) // 2]
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    entry = min((e for e in LEDGER if e["camera"] == cam_name), key=lambda e: abs(e["epoch"] - epoch))
    az, dip = sea_azimuths(cam, T.Dem())
    k0 = initial_k(cam, W)
    star = (entry["d_az"], entry["d_pitch"], entry["d_roll"], entry["k_ratio"] * k0, entry["k1"])
    pub = (0.0, 0.0, 0.0, K_RATIO * k0, K1)
    canvas = img.copy()
    for pose, color in ((pub, (0, 140, 255)), (star, (255, 60, 255))):
        x, y = project_fisheye(cam, az, np.full_like(az, dip), W, H, *pose)
        pts = np.c_[x * W, y * H]
        # break the polyline wherever the sea mask has an azimuth gap
        breaks = np.where(np.diff(az) > 0.3)[0] + 1
        for seg in np.split(np.arange(len(az)), breaks):
            p = pts[seg]
            ok = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
            if ok.sum() >= 2:
                cv2.polylines(canvas, [p[ok].astype(np.int32)], False, color, 3, cv2.LINE_AA)
    days = abs(entry["epoch"] - epoch) / 86400
    y_mid = float(np.median(project_fisheye(cam, az, np.full_like(az, dip), W, H, *star)[1] * H)) if len(az) else H / 2
    y0 = int(max(0, y_mid - 0.18 * H)); y1 = int(min(H, y_mid + 0.18 * H))
    crop = canvas[y0:y1]
    banner = np.zeros((80, W, 3), np.uint8)
    lines = [f"{seq}  frame offset {off}   sea over {len(az) * 0.25:.0f} deg of azimuth, dip {dip:+.2f} deg",
             f"magenta = star pose ({entry['source'].split(':')[1]}, {days:.0f} d away: pitch {entry['d_pitch']:+.2f} roll {entry['d_roll']:+.2f})   orange = published pose, same lens"]
    for i, t in enumerate(lines):
        cv2.putText(banner, t, (12, 32 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.vstack([banner, crop])
    out = cv2.resize(out, (1600, int(out.shape[0] * 1600 / W)), interpolation=cv2.INTER_AREA)
    dest = SKY / f"sea_horizon_{seq}.jpg"
    cv2.imwrite(str(dest), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return dest


if __name__ == "__main__":
    for seq in sys.argv[1:]:
        print(render(seq))
