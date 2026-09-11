"""Draw the terrain-predicted skyline over the frame a camera actually recorded.

Everything in the overlay comes from published metadata and a public elevation model --
no pixels are consulted. So wherever the drawn line departs from the visible ridge, the
metadata is wrong, or the lens is not the rectilinear ideal the projection assumes.
Named peaks are marked with their distance, which doubles as a check that the azimuth
is right: a peak drawn on the wrong summit is a bearing error of exactly that offset.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geom import load_cams
from .terrain import Dem, horizon, project, prominent_peaks, vfov_deg

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"


def observed_skyline(img: np.ndarray) -> np.ndarray:
    """Per-column row of the sky/terrain boundary, found from the image alone.

    Taking the first strong vertical gradient down each column finds cloud edges, not
    the ridge, and clouds sit above the skyline -- so that error is large and one-sided.
    Segment sky instead: at these wavelengths sky is blue-dominant and bright while
    terrain is neither, so the *lowest* sky pixel in a column is the silhouette. The
    vignetting on these lenses darkens the corners, hence the brightness floor.
    """
    b, g, r = (c.astype(np.int16) for c in cv2.split(cv2.GaussianBlur(img, (0, 0), 2)))
    v = img.max(axis=2)
    sky = ((b - r) > 8) & (v > 90)

    H, W = sky.shape
    rows = np.full(W, np.nan)
    for x in range(W):
        col = np.where(sky[:, x])[0]
        if col.size < H * 0.02:
            continue
        # Lowest sky pixel, but ignore isolated sky showing through gaps in the terrain
        # by requiring a run of sky immediately above it.
        for yy in col[::-1]:
            if yy >= 8 and sky[yy - 8:yy, x].mean() > 0.8:
                rows[x] = yy
                break
    return rows


def render(camera: str, img: np.ndarray, pitch_deg: float = 0.0,
           roll_deg: float = 0.0, show_observed: bool = True,
           dem: Dem | None = None) -> tuple[np.ndarray, dict]:
    cams = json.loads((META / "cams.json").read_text())
    cam = cams[camera]
    H, W = img.shape[:2]
    dem = dem or Dem()
    prof = horizon(cam, dem)
    x, y = project(cam, prof.az_deg, prof.elev_deg, W, H, pitch_deg, roll_deg)

    vis = img.copy()
    pts = [(int(px * W), int(py * H)) for px, py, ok in
           zip(x, y, (x > -0.2) & (x < 1.2) & (y > -0.5) & (y < 1.5)) if ok]
    for a, b in zip(pts, pts[1:]):
        cv2.line(vis, a, b, (0, 200, 255), 3)

    # Peaks by prominence, not by local maximum: a smooth crest carries twenty local
    # maxima that are a tenth of a degree of noise, and none of them is a landmark.
    stats = {"camera": camera, "n_peaks": 0}
    for i in prominent_peaks(prof):
        if True:
            px, py = int(x[i] * W), int(y[i] * H)
            if 0 <= px < W and 0 <= py < H:
                # A tick dropped from each predicted summit: whether it lands on a real
                # one is the whole question, and a dot alone does not show that.
                cv2.line(vis, (px, max(0, py - 42)), (px, min(H - 1, py + 42)),
                         (0, 0, 255), 2)
                cv2.circle(vis, (px, py), 9, (0, 0, 255), -1)
                cv2.circle(vis, (px, py), 9, (255, 255, 255), 2)
                cv2.putText(vis, f"{prof.range_km[i]:.0f}km", (px + 12, py - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                stats["n_peaks"] += 1

    resid = None
    if show_observed:
        obs = observed_skyline(img)
        for px in range(0, W, 6):
            if not np.isnan(obs[px]):
                cv2.circle(vis, (px, int(obs[px])), 2, (255, 120, 0), -1)
        # The horizon is sampled every 0.2 deg, which lands on only ~530 of 2048
        # columns. Interpolating onto every column compares like with like instead of
        # scoring the prediction on a sparse subset of the frame.
        xs = x * W
        keep = np.isfinite(xs) & np.isfinite(y)
        order = np.argsort(xs[keep])
        pred = np.interp(np.arange(W), xs[keep][order], (y * H)[keep][order],
                         left=np.nan, right=np.nan)
        m = ~np.isnan(obs) & ~np.isnan(pred)
        if m.sum() > 50:
            resid = (pred - obs)[m]
            stats.update(
                n_cols=int(m.sum()),
                coverage=round(float(m.sum()) / W, 3),
                resid_median_px=float(np.median(resid)),
                resid_iqr_px=float(np.percentile(resid, 75) - np.percentile(resid, 25)),
                # Distortion signature: a lens that is not rectilinear leaves a residual
                # that is symmetric about frame center and grows toward the edges, so a
                # quadratic in (x-0.5) captures it where a constant offset cannot.
                quad_fit=np.polyfit(((np.where(m)[0] / W) - 0.5), resid, 2).tolist(),
            )

    band = np.zeros((116, W, 3), np.uint8)
    cv2.putText(band, f"{camera}  az={cam['az']} fov={cam['fov']} "
                      f"elev={cam['elev']}m agl={cam.get('agl')}m  "
                      f"vfov={vfov_deg(cam, W, H):.1f}",
                (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(band, "orange = terrain-predicted skyline   red dots = peaks (range)   "
                      "blue = skyline found in image",
                (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
    if resid is not None:
        cv2.putText(band, f"median residual {stats['resid_median_px']:+.0f} px   "
                          f"IQR {stats['resid_iqr_px']:.0f} px",
                    (int(W * 0.62), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return np.vstack([band, vis]), stats


def _best_frame(seq: dict, n_try: int = 3):
    """A pre-ignition frame with the most usable sky.

    Pre-ignition on purpose: no plume over the ridge. Several are tried because haze and
    low cloud make some frames useless for skyline extraction, and a camera should not be
    judged on its worst morning.
    """
    import tarfile
    p = ROOT / "data" / "tgz" / f"{seq['seq'].split('#')[0]}.tgz"
    if not p.exists():
        return None
    with tarfile.open(p, "r:gz") as tf:
        names = sorted(((int(Path(m.name).name[:-4].split("_")[1]), m.name)
                        for m in tf
                        if m.name.endswith(".jpg") and "_" in Path(m.name).name))
    best = None
    for _off, name in names[:n_try]:
        with tarfile.open(p, "r:gz") as tf:
            fh = tf.extractfile(name)
            img = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cov = float(np.isfinite(observed_skyline(img)).mean())
        if best is None or cov > best[0]:
            best = (cov, img)
    return best[1] if best else None


def main(argv: list[str]) -> None:
    """Skyline residual for every posed camera -- a pose audit, not an anecdote."""
    from .terrain import Dem

    seqs = json.loads((META / "sequences.json").read_text())
    by_cam: dict[str, dict] = {}
    for s in seqs:
        if s["has_pose"] and s["camera"] not in by_cam:
            by_cam[s["camera"]] = s
    if argv:
        by_cam = {k: v for k, v in by_cam.items() if any(a in k for a in argv)}

    out_dir = ROOT / "out" / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    dem = Dem()
    rows = []
    for k, (cam, seq) in enumerate(sorted(by_cam.items()), 1):
        img = _best_frame(seq)
        if img is None:
            print(f"[{k}/{len(by_cam)}] {cam}: no frame")
            continue
        try:
            vis, st = render(cam, img, dem=dem)
        except Exception as exc:
            print(f"[{k}/{len(by_cam)}] {cam}: FAIL {exc}")
            continue
        cv2.imwrite(str(out_dir / f"{cam}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        st["seq"] = seq["seq"]
        rows.append(st)
        print(f"[{k}/{len(by_cam)}] {cam:22s} cov {st.get('coverage')} "
              f"median {st.get('resid_median_px')} IQR {st.get('resid_iqr_px')}",
              flush=True)

    (ROOT / "out" / "terrain_audit.json").write_text(json.dumps(rows, indent=1) + "\n")
    good = [r for r in rows if r.get("coverage", 0) >= 0.5
            and r.get("resid_median_px") is not None]
    if good:
        med = np.array([abs(r["resid_median_px"]) for r in good])
        print(f"\n{len(good)} cameras with >=50% skyline coverage")
        print(f"|median residual|: p50 {np.median(med):.0f} px  "
              f"p90 {np.percentile(med, 90):.0f} px  max {med.max():.0f} px")
        print(f"within 20 px: {(med <= 20).sum()}   "
              f"within 50 px: {(med <= 50).sum()}   "
              f"within 100 px: {(med <= 100).sum()}")




def _range_colour(km: float) -> tuple:
    """Near ridges warm, far ridges cool -- the same cue haze gives the eye.

    Log scale, because the interesting spread is 1-10 km and 10-80 km equally, and a
    linear ramp would paint everything beyond the first ridge the same color.
    """
    t = float(np.clip((np.log10(max(km, 0.5)) - np.log10(0.5))
                      / (np.log10(80.0) - np.log10(0.5)), 0, 1))
    lut = cv2.applyColorMap(np.array([[int(t * 255)]], np.uint8), cv2.COLORMAP_TURBO)
    return tuple(int(v) for v in lut[0, 0])


def render_ridges(camera: str, img: np.ndarray, dem: Dem | None = None,
                  pitch_deg: float = 0.0, roll_deg: float = 0.0,
                  min_chain_pts: int = 8) -> tuple[np.ndarray, dict]:
    """Draw every visible ridgeline, colored by range, with its summits marked.

    Where `render()` draws one curve and asks whether it lands on the skyline, this draws
    the whole nested stack. That matters for two reasons. It no longer depends on finding
    the sky, which is the step that fails on haze and on the monochrome units; and each
    ridge carries a distance, so agreement is evidence about pose at a known range rather
    than a single bearing residual.
    """
    from .terrain import ridges, summits
    cams = load_cams()
    cam = cams[camera]
    H, W = img.shape[:2]
    dem = dem or Dem()
    field = ridges(cam, dem)
    x, y = project(cam, field.az_deg, field.elev_deg, W, H, pitch_deg, roll_deg)

    vis = img.copy()
    drawn = 0
    for cid in np.unique(field.layer):
        sel = np.flatnonzero(field.layer == cid)
        if sel.size < min_chain_pts:
            continue
        sel = sel[np.argsort(field.az_deg[sel])]
        col = _range_colour(float(np.median(field.range_km[sel])))
        pts = [(int(x[i] * W), int(y[i] * H)) for i in sel]
        # Break the polyline rather than bridge a vertical jump. Chain linking bounds
        # the step in *degrees*, which on a 36 deg lens is several times the pixels it
        # is on a 90 deg one; an unbroken bridge draws a spike through the frame, and at
        # 0.1 deg ray spacing a run of them fills as a solid block of color.
        max_jump = 0.025 * H
        for a, b in zip(pts, pts[1:]):
            if abs(b[1] - a[1]) > max_jump:
                continue
            if -W < a[0] < 2 * W and -H < a[1] < 2 * H:
                cv2.line(vis, a, b, col, 2, cv2.LINE_AA)
        drawn += 1

    sm = summits(field, min_chain_pts=min_chain_pts)
    shown = 0
    for i in sm:
        px, py = int(x[i] * W), int(y[i] * H)
        if not (0 <= px < W and 0 <= py < H):
            continue
        col = _range_colour(float(field.range_km[i]))
        cv2.circle(vis, (px, py), 8, col, -1)
        cv2.circle(vis, (px, py), 8, (255, 255, 255), 2)
        cv2.putText(vis, f"{field.range_km[i]:.0f}", (px + 11, py - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        shown += 1

    band = np.zeros((116, W, 3), np.uint8)
    cv2.putText(band, f"{camera}  az={cam['az']} fov={cam['fov']}  "
                      f"{drawn} ridgelines, {shown} summits in frame",
                (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(band, "color = range to the ridge (blue near ... red far), "
                      "labels in km.  No pixels consulted.",
                (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
    for k in range(9):
        km = 0.5 * (160.0 ** (k / 8.0))
        cv2.rectangle(band, (int(W * 0.66) + k * 40, 20),
                      (int(W * 0.66) + k * 40 + 38, 50), _range_colour(km), -1)
        cv2.putText(band, f"{km:.0f}", (int(W * 0.66) + k * 40, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    stats = {"camera": camera, "n_chains": drawn, "n_summits_in_frame": shown,
             "n_crest_pts": int(len(field.az_deg))}
    return np.vstack([band, vis]), stats


def main_ridges(argv: list[str]) -> None:
    seqs = json.loads((META / "sequences.json").read_text())
    by_cam: dict[str, dict] = {}
    for s in seqs:
        if s["has_pose"] and s["camera"] not in by_cam:
            by_cam[s["camera"]] = s
    if argv:
        by_cam = {k: v for k, v in by_cam.items() if any(a in k for a in argv)}
    out_dir = ROOT / "out" / "ridges"
    out_dir.mkdir(parents=True, exist_ok=True)
    dem = Dem()
    for k, (cam, seq) in enumerate(sorted(by_cam.items()), 1):
        img = _best_frame(seq)
        if img is None:
            print(f"[{k}/{len(by_cam)}] {cam}: no frame")
            continue
        try:
            vis, st = render_ridges(cam, img, dem=dem)
        except Exception as exc:
            print(f"[{k}/{len(by_cam)}] {cam}: FAIL {exc}")
            continue
        cv2.imwrite(str(out_dir / f"{cam}.png"), vis)
        print(f"[{k}/{len(by_cam)}] {cam:22s} {st['n_chains']} chains  "
              f"{st['n_summits_in_frame']} summits", flush=True)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    if a and a[0] == "ridges":
        main_ridges(a[1:])
    else:
        main(a)
