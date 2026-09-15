"""Contact sheet: terrain-predicted peaks laid over the frames the cameras recorded.

Nothing in the overlay comes from the image. The orange line and the red summit ticks are
synthesised from published camera metadata and Copernicus DEM GLO-30 alone, so wherever
they sit off the visible ridge, the metadata is wrong -- and the blue trace is the skyline
found in the pixels, which is what the residual is measured against.

Full frames are 2048x1536 and the interesting part is a band a few hundred pixels tall, so
each example is cropped to the band spanning both skylines. Cameras are chosen to span the
range rather than to flatter it: the best fits, the median, and the worst.

Both sheets predate the star-track calibration: published pose, rectilinear lens, and the
2048x1536 units most of those frames come from have no star solve or measured lens at all.
`calibrated` and `ridges-calibrated` redraw the same cameras under the star-solved pose and
fisheye lens, on 3072-wide frames of the same cameras (see `calibrated_view`), into
`*_calibrated.png` beside the originals.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .stars.fisheye import initial_k, project_fisheye
from .terrain import Dem, horizon, project, ridges
from .viz_terrain import observed_skyline, render, render_ridges, _best_frame

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
OUT = ROOT / "out"

PANEL_W = 1500
PAD_PX = 90
# The fisheye reaches ~53 deg off-axis at the frame's side edges, past the 8 deg the DEM is
# marched beyond a 90 deg lens's nameplate half-width; ridge_feet uses the same pad.
FISHEYE_PAD_DEG = 14.0


def band_crop(camera: str, img: np.ndarray, dem: Dem, cam: dict | None = None,
              project_fn=None, half_fov_pad: float = 8.0) -> tuple[np.ndarray, dict]:
    cam = cam or json.loads((META / "cams.json").read_text())[camera]
    H, W = img.shape[:2]
    prof = horizon(cam, dem, half_fov_pad=half_fov_pad)
    if project_fn is None:
        x, y = project(cam, prof.az_deg, prof.elev_deg, W, H)
        keep = np.isfinite(y)
    else:
        x, y = project_fn(prof.az_deg, prof.elev_deg, W, H)
        keep = np.isfinite(y) & (x >= 0) & (x < 1)    # the wider pad marches past the frame edge
    vis, st = render(camera, img, dem=dem, cam=cam, project_fn=project_fn,
                     half_fov_pad=half_fov_pad)
    vis = vis[116:]                                   # drop render's own caption band

    obs = observed_skyline(img)
    rows = [v for v in (y[keep] * H)] + \
           [v for v in obs if np.isfinite(v)]
    if not rows:
        return vis, st
    lo = int(max(0, min(rows) - PAD_PX))
    hi = int(min(H, max(rows) + PAD_PX))
    if hi - lo < 200:                                  # keep thin bands readable
        mid = (lo + hi) // 2
        lo, hi = max(0, mid - 100), min(H, mid + 100)
    return vis[lo:hi], st


def audit_examples(argv: list[str]) -> tuple[list[dict], list[str]]:
    """Audit rows to show: those matching `argv`, else two best, two median, two worst."""
    audit = [r for r in json.loads((OUT / "terrain_audit.json").read_text())
             if r.get("coverage", 0) >= 0.5 and r.get("resid_median_px") is not None]
    audit.sort(key=lambda r: abs(r["resid_median_px"]))

    if argv:
        chosen = [r for r in audit if any(a in r["camera"] for a in argv)]
        return chosen, ["" for _ in chosen]
    mid = len(audit) // 2
    return ([audit[0], audit[1], audit[mid], audit[mid + 1], audit[-2], audit[-1]],
            ["best fit", "best fit", "median", "median", "worst", "worst"])


def main(argv: list[str]) -> None:
    chosen, labels = audit_examples(argv)

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
    dest = ROOT / "docs" / "figures" / "dem_skyline_peaks_examples.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet)
    print(f"\nwrote {dest}  ({sheet.shape[1]}x{sheet.shape[0]})")


# The five cameras from the skyline-vs-ridge lever-arm comparison in NOTES.md: the ridge
# field's vertical spread on each is 4-10x the single skyline's, which is *why* ridge
# matching conditions pitch/height/azimuth better -- there is no audit-residual metric for
# ridges to sort by, since nothing here fits pose against them, so these are picked to
# match that table rather than to span a range.
RIDGE_EXAMPLE_CAMS = ["bh-n-mobo-c", "sm-s-mobo-c", "vo-n-mobo-c", "stgo-e-mobo-c",
                      "wc-n-mobo-c"]


def ridge_band_crop(camera: str, img: np.ndarray, dem: Dem, cam: dict | None = None,
                    project_fn=None, half_fov_pad: float = 8.0) -> tuple[np.ndarray, dict]:
    cam = cam or json.loads((META / "cams.json").read_text())[camera]
    H, W = img.shape[:2]
    vis, st = render_ridges(camera, img, dem=dem, cam=cam, project_fn=project_fn,
                            half_fov_pad=half_fov_pad)
    vis = vis[116:]                                   # drop render_ridges's own caption band

    field = ridges(cam, dem, half_fov_pad=half_fov_pad)
    if project_fn is None:
        x, y = project(cam, field.az_deg, field.elev_deg, W, H)
    else:
        x, y = project_fn(field.az_deg, field.elev_deg, W, H)
    inframe = (x >= 0) & (x < 1) & (y >= 0) & (y < 1)
    rows = (y[inframe] * H)
    if rows.size == 0:
        return vis, st
    lo = int(max(0, rows.min() - PAD_PX))
    hi = int(min(H, rows.max() + PAD_PX))
    if hi - lo < 200:
        mid = (lo + hi) // 2
        lo, hi = max(0, mid - 100), min(H, mid + 100)
    return vis[lo:hi], st


def main_ridges(argv: list[str]) -> None:
    """Contact sheet: the full nested ridge stack, colored by range, over the same frames.

    Companion to `main()` above. Where that figure shows one predicted curve against one
    extracted skyline, this shows every layer `ridges()` finds -- the structure the ridge-
    matching approach depends on, and the reason NOTES.md argues it is better conditioned
    than skyline fitting even though nothing here has actually calibrated a pose against
    it yet.
    """
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    cams = json.loads((META / "cams.json").read_text())
    chosen = [c for c in RIDGE_EXAMPLE_CAMS if not argv or any(a in c for a in argv)]

    dem = Dem()
    panels = []
    for camera in chosen:
        seq = next((s for s in seqs.values()
                   if s["camera"] == camera and s["has_pose"]), None)
        if seq is None:
            print(f"{camera}: no posed sequence")
            continue
        img = _best_frame(seq)
        if img is None:
            print(f"{camera}: no frame")
            continue
        crop, st = ridge_band_crop(camera, img, dem)
        h = int(crop.shape[0] * PANEL_W / crop.shape[1])
        crop = cv2.resize(crop, (PANEL_W, h), interpolation=cv2.INTER_AREA)
        bar = np.zeros((46, PANEL_W, 3), np.uint8)
        cv2.putText(bar, f"{camera}   az={cams[camera]['az']}   "
                         f"{st['n_chains']} ridgelines   {st['n_summits_in_frame']} summits",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
        panels.append(np.vstack([bar, crop]))
        print(f"{camera:22s} band {crop.shape[0]:4d}px  "
              f"{st['n_chains']} chains  {st['n_summits_in_frame']} summits", flush=True)

    if not panels:
        return
    legend = np.zeros((52, PANEL_W, 3), np.uint8)
    cv2.putText(legend, "color = range to the ridge, blue (near) to red (far)   "
                        "labels = distance in km   no pixels consulted",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (190, 190, 190), 2)
    sheet = np.vstack([legend] + panels)
    dest = ROOT / "docs" / "figures" / "dem_ridge_layers_examples.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet)
    print(f"\nwrote {dest}  ({sheet.shape[1]}x{sheet.shape[0]})")


def calibrated_view(camera: str) -> tuple[np.ndarray, dict, str] | None:
    """A 3072-wide frame of `camera`, the pose and lens to draw terrain under, and a label.

    The original sheets' frames can't simply be redrawn: most are 2048x1536 units with no
    star solve and no measured lens. First choice is instead a daytime CDN block (Q4) from
    the day of the camera's latest star solve, so it cannot have moved between the solve
    and the frame. Failing that, the camera's latest 3072-wide FIgLib sequence, pre-ignition,
    under whatever the pose ledger allows for that date -- the published pose through the
    shared measured lens when no solve applies.
    """
    from . import ridge_feet as RF
    from .stars import nights

    solves = sorted((e for e in RF.LEDGER if e["camera"] == camera and e.get("frame_w") == 3072
                     and e["source"].startswith("star:hpwren_")), key=lambda e: -e["epoch"])
    for e in solves:
        day = e["source"].split("_")[1]
        imgs = [(p, cv2.imread(str(p))) for p in sorted((nights.FRAMES / camera / f"{day}_Q4").glob("*.jpg"))]
        imgs = [(p, im) for p, im in imgs if im is not None and im.shape[1] == 3072]
        if imgs:
            p, img = max(imgs, key=lambda t: float(np.isfinite(observed_skyline(t[1])).mean()))
            when = datetime.fromtimestamp(int(p.stem), ZoneInfo("America/Los_Angeles"))
            pose = {k: e[k] for k in ("d_az", "d_pitch", "d_roll", "k_ratio", "k1")}
            return img, pose, f"HPWREN CDN {when:%Y-%m-%d %H:%M}, same day as its star solve"

    seqs = [s for s in RF.SEQS.values() if s["camera"] == camera
            and (RF.SIZES.get(s["seq"].split("#")[0]) or [0])[0] == 3072]
    # Newest first, skipping a sequence whose best frame shows little sky: some fires
    # were recorded at night.
    for s in sorted(seqs, key=lambda s: -s["t0"]):
        imgs = [im for im in (RF.decode(b) for _e, _o, b in RF.frames(s["seq"])[:3]) if im is not None]
        if not imgs:
            continue
        cov, img = max(((float(np.isfinite(observed_skyline(im)).mean()), im) for im in imgs),
                       key=lambda t: t[0])
        if cov < 0.3:
            continue
        pose = RF.pose_for(camera, s["t0"], img.shape[1])
        how = ("published pose (no star solve applies), measured lens"
               if pose["rule"] == "published+lens" else f"ledger pose ({pose['rule']})")
        return img, pose, f"FIgLib {s['seq']}, pre-ignition; {how}"
    return None


def solved_lens(cam: dict, pose: dict):
    """(the camera turned by the solve's azimuth, for marching the DEM; a projector through
    the solved pose and fisheye lens, with `terrain.project`'s call shape)."""
    def fn(az, el, W, H):
        return project_fisheye(cam, az, el, W, H, pose["d_az"], pose["d_pitch"], pose["d_roll"],
                               pose["k_ratio"] * initial_k(cam, W), pose["k1"])
    return {**cam, "az": cam["az"] + pose["d_az"]}, fn


def pose_text(pose: dict) -> str:
    return (f"d_az {pose['d_az']:+.2f}  pitch {pose['d_pitch']:+.2f}  roll {pose['d_roll']:+.2f}  "
            f"k/nameplate {pose['k_ratio']:.3f}  k1 {pose['k1']:+.3f}")


def calibrated_panel(camera: str, dem: Dem, crop_fn, head) -> np.ndarray | None:
    cams = json.loads((META / "cams.json").read_text())
    view = calibrated_view(camera)
    if view is None:
        print(f"{camera}: no 3072-wide frame")
        return None
    img, pose, where = view
    cam, fn = solved_lens(cams[camera], pose)
    crop, st = crop_fn(camera, img, dem, cam=cam, project_fn=fn, half_fov_pad=FISHEYE_PAD_DEG)
    h = int(crop.shape[0] * PANEL_W / crop.shape[1])
    crop = cv2.resize(crop, (PANEL_W, h), interpolation=cv2.INTER_AREA)
    bar = np.zeros((84, PANEL_W, 3), np.uint8)
    cv2.putText(bar, f"{camera}   {head(st)}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2)
    cv2.putText(bar, f"{where}   {pose_text(pose)}", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (200, 200, 200), 1, cv2.LINE_AA)
    print(f"{camera:22s} band {crop.shape[0]:4d}px  {head(st)}  | {where}", flush=True)
    return np.vstack([bar, crop])


def write_sheet(legend_lines: list[str], panels: list[np.ndarray], name: str) -> None:
    legend = np.zeros((16 + 34 * len(legend_lines), PANEL_W, 3), np.uint8)
    for i, t in enumerate(legend_lines):
        cv2.putText(legend, t, (12, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (190, 190, 190), 2)
    sheet = np.vstack([legend] + panels)
    dest = ROOT / "docs" / "figures" / name
    cv2.imwrite(str(dest), sheet)
    print(f"\nwrote {dest}  ({sheet.shape[1]}x{sheet.shape[0]})")


def main_calibrated(argv: list[str]) -> None:
    """`main()`'s cameras, redrawn under the star-solved pose and fisheye lens."""
    chosen, labels = audit_examples(argv)
    dem = Dem()
    panels = []
    for r, tag in zip(chosen, labels):
        def head(st, r=r, tag=tag):
            res = st.get("resid_median_px")
            return (f"residual {'n/a' if res is None else f'{res:+.0f} px'}"
                    f" (published pose: {r['resid_median_px']:+.0f} px)   {st['n_peaks']} predicted peaks"
                    + (f"   [was {tag}]" if tag else ""))
        p = calibrated_panel(r["camera"], dem, band_crop, head)
        if p is not None:
            panels.append(p)
    if panels:
        write_sheet(["orange = skyline predicted from DEM + star-solved pose and fisheye lens",
                     "red ticks = predicted summits (range)   blue = skyline found in the image"],
                    panels, "dem_skyline_peaks_examples_calibrated.png")


def main_ridges_calibrated(argv: list[str]) -> None:
    """`main_ridges()`'s cameras, redrawn under the star-solved pose and fisheye lens."""
    chosen = [c for c in RIDGE_EXAMPLE_CAMS if not argv or any(a in c for a in argv)]
    dem = Dem()
    panels = []
    for camera in chosen:
        p = calibrated_panel(camera, dem, ridge_band_crop,
                             lambda st: f"{st['n_chains']} ridgelines   {st['n_summits_in_frame']} summits")
        if p is not None:
            panels.append(p)
    if panels:
        write_sheet(["color = range to the ridge, blue (near) to red (far)   labels = distance in km",
                     "star-solved pose and fisheye lens   no pixels consulted"],
                    panels, "dem_ridge_layers_examples_calibrated.png")


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
    elif sys.argv[1:2] == ["ridges"]:
        main_ridges(sys.argv[2:])
    elif sys.argv[1:2] == ["calibrated"]:
        main_calibrated(sys.argv[2:])
    elif sys.argv[1:2] == ["ridges-calibrated"]:
        main_ridges_calibrated(sys.argv[2:])
    else:
        main(sys.argv[1:])
