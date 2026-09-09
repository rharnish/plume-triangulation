"""Render detections against the projected ground truth.

One figure per fire: every camera that saw it, each frame overlaid with the detector's
boxes and a vertical line marking where the official ignition coordinate projects given
the published camera pose. The two are derived independently -- one from pixels, one
from geometry and an incident record -- so their agreement, or lack of it, is visible
at a glance and needs no metric to interpret.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .detect_yolo import detect, make_session, read_frames
from .geom import bearing_x_frac

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
TGZ = ROOT / "data" / "tgz"
PANEL_W = 900


def render_fire(fire_id: str, at_offset: int = 900, conf: float = 0.10,
                out: Path | None = None, cols: int = 2) -> Path | None:
    cams = json.loads((META / "cams.json").read_text())
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    res = {r["fire_id"]: r for r in json.loads((META / "resolved.json").read_text())}

    rec, fire = res.get(fire_id), fires.get(fire_id)
    if not rec or not rec.get("truth"):
        print(f"  {fire_id}: no resolved truth")
        return None
    t = rec["truth"]
    sess = make_session()

    panels = []
    for seq_name in fire["sequences"]:
        s = seqs[seq_name]
        if not s["has_pose"]:
            continue
        cam = cams[s["camera"]]
        xt = bearing_x_frac(cam, t["lat"], t["lon"])

        tgz = TGZ / f"{seq_name.split('#')[0]}.tgz"
        if not tgz.exists():
            continue
        frames = {o: b for _, o, b in read_frames(tgz)}
        if not frames:
            continue
        o = min(frames, key=lambda x: abs(x - at_offset))
        img = cv2.imdecode(np.frombuffer(frames[o], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]

        dets = detect(sess, img, conf_thr=conf)
        for d in dets:
            cv2.rectangle(img, (int(d.x0 * W), int(d.y0 * H)),
                          (int(d.x1 * W), int(d.y1 * H)), (0, 0, 255), 4)
            cv2.putText(img, f"{d.conf:.2f}", (int(d.x0 * W), max(34, int(d.y0 * H) - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        if xt is not None:
            cv2.line(img, (int(xt * W), 0), (int(xt * W), H), (0, 255, 0), 3)
            label = f"{s['camera']}  t{o:+d}s  truth x={xt:.3f}"
        else:
            label = f"{s['camera']}  t{o:+d}s  truth OUTSIDE fov"

        cv2.rectangle(img, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.putText(img, label, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                    (0, 255, 255), 3)
        panels.append(cv2.resize(img, (PANEL_W, int(PANEL_W * H / W))))

    if not panels:
        return None
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
              for p in panels]
    rows = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(20, 20, 20)) for r in rows]
    grid = np.vstack(rows)

    hdr = np.zeros((70, grid.shape[1], 3), np.uint8)
    cv2.putText(hdr, f"{fire_id}  ->  {t['name']}  ({t['acres'] or '?'} ac)   "
                     f"green = official ignition bearing, red = detector",
                (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3)
    grid = np.vstack([hdr, grid])

    out = out or ROOT / "out" / "figures" / f"{fire_id}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 84])
    print(f"  {fire_id}: {len(panels)} cameras -> {out.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    off = 900
    if args and args[0].lstrip("+-").isdigit():
        off = int(args.pop(0))
    for fid in (args or ["20260629_JunctionFire"]):
        render_fire(fid, at_offset=off)
