"""The triangulation figure, animated: the same map and camera panels, stepped through time.

At each step every camera contributes the most confident detection it has made so far,
exactly as `fig_triangulate` picks one over the whole window, and the likelihood surface is
re-solved from those bearings. The last frame is therefore the still figure, and the
frames before it show how the answer got there: nothing, one ray with no depth, a crossing,
and the crossing tightening as better detections replace weaker ones.

Map extent, inset placement and each camera's crop window are fixed from the final frame,
so nothing on screen moves except the evidence.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .detect_yolo import read_frames
from .fig_triangulate import (META, PALETTE, _crop_image, _crop_window, _det_for, _draw,
                              _hillshade, _inset_spec, _tgz_for, _title, _view)
from .geolocate import bearings_for_fire, credible_area_km2, solve
from .geom import haversine_km, load_cams

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "videos"


def animate(fire_id: str, t_start: int = -120, t_end: int = 2400, step: int = 60,
            fps: int = 3, hold_s: float = 3.0, width: int = 1280,
            out: Path | None = None) -> Path | None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    rec = {r["fire_id"]: r for r in json.loads((META / "resolved.json").read_text())}[fire_id]
    fire, truth, tier = fires[fire_id], rec["truth"], rec["tier"]

    def bearings(t: int):
        return bearings_for_fire(fire, seqs, cams, use_wind=False, window_s=(0, t))

    final = sorted(bearings(t_end), key=lambda b: -b.conf)[:len(PALETTE)]
    if len({b.camera.split("-")[0] for b in final}) < 2:
        print(f"  {fire_id}: fewer than two sites")
        return None
    colors = {b.camera: col for b, col in zip(final, PALETTE)}
    cameras = [(b.camera, colors[b.camera]) for b in final]

    # Everything that frames the picture comes from the final solve and stays put.
    center = (float(np.mean([b.lat for b in final])), float(np.mean([b.lon for b in final])))
    lats, lons, ll, elat, elon = solve(final, center)
    err = haversine_km(elat, elon, truth["lat"], truth["lon"])
    view = _view(final, truth, elat, elon)
    inset = _inset_spec(final, truth, elat, elon, err, credible_area_km2(lats, lons, ll),
                        view)
    hillshade = _hillshade(view["clat"], view["clon"], view["half_km"])

    blobs, windows = {}, {}
    for b in final:
        seq_name, det = _det_for(b, fire, seqs)
        blobs[b.camera] = {e: blob for e, _, blob in read_frames(_tgz_for(seq_name))}
        img = cv2.imdecode(np.frombuffer(blobs[b.camera][b.epoch], np.uint8),
                           cv2.IMREAD_COLOR)
        windows[b.camera] = _crop_window(det, *img.shape[1::-1])

    crop_cache: dict[tuple[str, int], np.ndarray] = {}

    def crop_for(b):
        key = (b.camera, b.epoch)
        if key not in crop_cache:
            _, det = _det_for(b, fire, seqs)
            img = cv2.imdecode(np.frombuffer(blobs[b.camera][b.epoch], np.uint8),
                               cv2.IMREAD_COLOR)
            # Hold the final crop window unless this detection falls outside it -- an early
            # false positive elsewhere in the frame must be seen, not cropped away while
            # its ray points off across the map.
            x0, y0, x1, y1, _, W, H = windows[b.camera]
            cx, cy = (det["x0"] + det["x1"]) / 2 * W, (det["y0"] + det["y1"]) / 2 * H
            win = (windows[b.camera] if x0 <= cx <= x1 and y0 <= cy <= y1
                   else _crop_window(det, W, H))
            crop_cache[key] = _crop_image(img, win, det)
        return crop_cache[key]

    times = list(range(t_start, t_end, step)) + [t_end]
    out = out or OUT / f"triangulate_{fire_id}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for k, t in enumerate(times):
            now = {b.camera: b for b in bearings(max(t, -1)) if b.camera in colors}
            n_sites = len({c.split("-")[0] for c in now})
            if n_sites >= 2:
                bs = list(now.values())
                g_lats, g_lons, g_ll, g_elat, g_elon = solve(bs, center)
                g_err = haversine_km(g_elat, g_elon, truth["lat"], truth["lon"])
                g_area = credible_area_km2(g_lats, g_lons, g_ll)
                surface, est = (g_lats, g_lons, g_ll), (g_elat, g_elon)
                status = f"error {g_err:.2f} km      95% region {g_area:.1f} km²"
            else:
                surface = est = None
                status = "needs two sites"
            title = _title(fire_id, truth, tier,
                           f"      t {t:+d} s      {n_sites} "
                           f"site{'' if n_sites == 1 else 's'}      {status}")

            crops = []
            for camera, col in cameras:
                b = now.get(camera)
                if b is None:
                    crops.append((f"{camera}   no detection yet", col, None))
                else:
                    crops.append((f"{camera}   conf {b.conf:.2f}   bearing "
                                  f"{b.bearing_deg:.1f}°", col, crop_for(b)))

            fig = _draw(title, view, hillshade, cams, cameras, now, surface, est, truth,
                        inset, crops)
            fig.savefig(f"{tmp}/{k:04d}.png", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  t {t:+5d}  {n_sites} sites  {status}", flush=True)

        last = len(times) - 1
        for j in range(1, int(round(hold_s * fps)) + 1):
            Path(f"{tmp}/{last + j:04d}.png").symlink_to(f"{tmp}/{last:04d}.png")

        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", f"{tmp}/%04d.png",
             "-vf", f"scale={width}:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
             str(out)], check=True)
    print(f"  {fire_id}: {len(times)} frames -> {out}")
    return out


if __name__ == "__main__":
    import sys
    for fid in (sys.argv[1:] or ["20240701_Kitchenfire"]):
        animate(fid)
