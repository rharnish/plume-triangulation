"""Contact sheet: terrain-predicted peaks laid over the frames the cameras recorded.

Nothing in the overlay comes from the image. The orange line and the red summit ticks are
synthesised from published camera metadata and Copernicus DEM GLO-30 alone, so wherever
they sit off the visible ridge, the metadata is wrong -- and the blue trace is the skyline
found in the pixels, which is what the residual is measured against.

Full frames are 2048x1536 and the interesting part is a band a few hundred pixels tall, so
each example is cropped to the band spanning both skylines. Cameras are chosen to span the
range rather than to flatter it: the best fits, the median, and the worst.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .terrain import Dem, horizon, project
from .viz_terrain import observed_skyline, render, _best_frame

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
OUT = ROOT / "out"

PANEL_W = 1500
PAD_PX = 90


def band_crop(camera: str, img: np.ndarray, dem: Dem) -> tuple[np.ndarray, dict]:
    cams = json.loads((META / "cams.json").read_text())
    cam = cams[camera]
    H, W = img.shape[:2]
    prof = horizon(cam, dem)
    x, y = project(cam, prof.az_deg, prof.elev_deg, W, H)
    vis, st = render(camera, img, dem=dem)
    vis = vis[116:]                                   # drop render's own caption band

    obs = observed_skyline(img)
    rows = [v for v in (y * H) if np.isfinite(v)] + \
           [v for v in obs if np.isfinite(v)]
    if not rows:
        return vis, st
    lo = int(max(0, min(rows) - PAD_PX))
    hi = int(min(H, max(rows) + PAD_PX))
    if hi - lo < 200:                                  # keep thin bands readable
        mid = (lo + hi) // 2
        lo, hi = max(0, mid - 100), min(H, mid + 100)
    return vis[lo:hi], st


def main(argv: list[str]) -> None:
    audit = [r for r in json.loads((OUT / "terrain_audit.json").read_text())
             if r.get("coverage", 0) >= 0.5 and r.get("resid_median_px") is not None]
    audit.sort(key=lambda r: abs(r["resid_median_px"]))

    if argv:
        chosen = [r for r in audit if any(a in r["camera"] for a in argv)]
        labels = ["" for _ in chosen]
    else:
        mid = len(audit) // 2
        chosen = [audit[0], audit[1], audit[mid], audit[mid + 1], audit[-2], audit[-1]]
        labels = ["best fit", "best fit", "median", "median", "worst", "worst"]

    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    dem = Dem()
    panels = []
    for r, tag in zip(chosen, labels):
        img = _best_frame(seqs[r["seq"]])
        if img is None:
            continue
        crop, st = band_crop(r["camera"], img, dem)
        h = int(crop.shape[0] * PANEL_W / crop.shape[1])
        crop = cv2.resize(crop, (PANEL_W, h), interpolation=cv2.INTER_AREA)
        bar = np.zeros((46, PANEL_W, 3), np.uint8)
        cv2.putText(bar, f"{r['camera']}   az={json.loads((META / 'cams.json').read_text())[r['camera']]['az']}"
                         f"   residual {r['resid_median_px']:+.0f} px   "
                         f"{st['n_peaks']} predicted peaks"
                         + (f"   [{tag}]" if tag else ""),
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        panels.append(np.vstack([bar, crop]))
        print(f"{r['camera']:22s} band {crop.shape[0]:4d}px  "
              f"residual {r['resid_median_px']:+.0f}", flush=True)

    if not panels:
        return
    legend = np.zeros((52, PANEL_W, 3), np.uint8)
    cv2.putText(legend, "orange = skyline predicted from DEM + published pose   "
                        "red ticks = predicted summits (range)   "
                        "blue = skyline found in the image",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (190, 190, 190), 2)
    sheet = np.vstack([legend] + panels)
    dest = ROOT / "docs" / "figures" / "peaks_examples.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet)
    print(f"\nwrote {dest}  ({sheet.shape[1]}x{sheet.shape[0]})")




def to_png(argv: list[str]) -> None:
    """PNG band crops of every rendered overlay, for browsing.

    Works from the JPGs the audit already wrote rather than re-marching the DEM, so this
    is seconds rather than a quarter of an hour. The band is located by finding the rows
    that actually contain overlay color, which is more robust than recomputing the
    projection and has to agree with what was drawn.
    """
    src_dir = OUT / "terrain"
    dst_dir = OUT / "terrain_png"
    dst_dir.mkdir(parents=True, exist_ok=True)
    audit = {r["camera"]: r for r in
             json.loads((OUT / "terrain_audit.json").read_text())}

    for p in sorted(src_dir.glob("*.jpg")):
        if argv and not any(a in p.name for a in argv):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        band, body = img[:116], img[116:]
        b, g, r = body[:, :, 0].astype(int), body[:, :, 1].astype(int), body[:, :, 2].astype(int)
        overlay = ((r > 190) & (g > 140) & (b < 110)) | (b > 190) & (r < 110)  # orange | blue
        rows = np.where(overlay.any(axis=1))[0]
        if rows.size:
            lo = max(0, rows.min() - PAD_PX)
            hi = min(body.shape[0], rows.max() + PAD_PX)
            if hi - lo < 240:
                mid = (lo + hi) // 2
                lo, hi = max(0, mid - 120), min(body.shape[0], mid + 120)
            body = body[lo:hi]
        out = np.vstack([band, body])
        cam = p.stem
        res = audit.get(cam, {}).get("resid_median_px")
        tag = f"{abs(res):04.0f}" if res is not None else "----"
        cv2.imwrite(str(dst_dir / f"resid{tag}_{cam}.png"), out)
    n = len(list(dst_dir.glob("*.png")))
    print(f"wrote {n} PNGs to {dst_dir}  (named by |residual| so they sort worst-last)")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["png"]:
        to_png(sys.argv[2:])
    else:
        main(sys.argv[1:])
