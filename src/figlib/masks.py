"""Pixel masks for a detected plume, and the one number they are for.

A detection reaches `geolocate.py` as a single scalar: the horizontal fraction `x` that
becomes a bearing. Everything else about the box is discarded. So a mask is only worth
computing if it yields a better `x` than the box does -- and there is a specific reason
to think it does.

`wind.py` already names the problem: the plume drifts, so its centroid sits downwind of
the fire, and the correction there is to take the box's upwind *edge*. But a box edge is
set by whichever pixel of the plume reaches furthest, and that pixel is almost always
high in the column, where the smoke has been aloft longest and drifted furthest. The
edge over-corrects in the same axis the centroid under-corrects.

What actually marks the source is the **foot**: the horizontal position of the lowest
visible smoke, where the plume is still attached to the ground. A box cannot express
that -- `x0` and `x1` say nothing about which rows they came from. A mask can.

Two mask sources, deliberately different in kind:

**diff** -- the temporal-differencing mask `detect_diff.py` already builds and throws
away. Training-free, so it carries no FIgLib contamination asterisk, and it is answering
the right question for a fixed camera: what changed. Its weakness is that it segments
*change*, so a plume that has been sitting in frame for thirty minutes partly merges
into its own background.

**sam** -- Segment Anything, prompted with the pyronear box. Trained on natural images,
never on FIgLib, so it adds no contamination either. Its weakness is the opposite one:
SAM was trained on objects with boundaries, and a diffuse plume against haze is the
adversarial case. When it fails it fails by snapping to the ridgeline or by returning
the box.

Neither is trusted here. Both are rendered, side by side, for the eye to judge before
anything is wired into a kilometer.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np

from .detect_yolo import read_frames
from .detect_diff import BG_FRAMES, BG_LAG_FRAMES, OPEN_K, WIDTH, _load_gray

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
TGZ = ROOT / "data" / "tgz"

# How much of the mask's vertical extent counts as "the foot". The plume's lowest
# pixels are the least drifted, but the very lowest few are also the noisiest, so this
# is a band rather than a single row.
FOOT_FRAC = 0.20

_SAM = None


# --------------------------------------------------------------------------- foot


def foot_x(mask: np.ndarray, frac: float = FOOT_FRAC) -> float | None:
    """Column centroid of the lowest `frac` of the mask's occupied rows.

    Returns a fraction of the frame width, directly comparable to the `x` that
    `geolocate.py` feeds to `offset_bearing_deg`.
    """
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return None
    lo, hi = rows[0], rows[-1]
    band = max(1, int(round((hi - lo + 1) * frac)))
    sub = mask[hi - band + 1:hi + 1]
    cols = np.where(sub.any(axis=0))[0]
    if cols.size == 0:
        return None
    # Weight by how much smoke each column holds in the band, so a single stray pixel
    # at the edge does not pull the estimate the way a bounding edge would.
    w = sub.sum(axis=0).astype(np.float64)
    return float((np.arange(mask.shape[1]) * w).sum() / w.sum()) / mask.shape[1]


def _pick_component(mask: np.ndarray, box_px: tuple[int, int, int, int]) -> np.ndarray:
    """Keep the connected component overlapping the detection box the most.

    Both methods are being asked the same question -- which pixels of *this detection*
    are plume -- so the diff mask is anchored to the box rather than allowed to pick its
    own winner globally as `detect_diff` does.
    """
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    x0, y0, x1, y1 = box_px
    inside = np.zeros_like(labels, bool)
    inside[max(0, y0):y1, max(0, x0):x1] = True
    best, best_n = None, 0
    for i in range(1, n):
        c = labels == i
        overlap = int((c & inside).sum())
        if overlap > best_n:
            best, best_n = c, overlap
    return best if best is not None else np.zeros_like(mask, bool)


# --------------------------------------------------------------------- diff mask


def diff_mask(frames: list[tuple[int, int, bytes]], idx: int,
              box_px: tuple[int, int, int, int], z_thr: float = 4.0) -> np.ndarray | None:
    """Rebuild `detect_diff`'s change mask at frame `idx`, at that module's settings.

    Deliberately re-uses the constants rather than re-tuning them: the point is to see
    the mask the existing detector already computes, not a better one.
    """
    if idx < BG_LAG_FRAMES:
        return None
    hist = [_load_gray(b) for _, _, b in frames[idx - BG_LAG_FRAMES:idx]]
    g = _load_gray(frames[idx][2])
    bg = np.median(np.stack(hist[:BG_FRAMES]), axis=0)

    g_adj = g - (np.median(g) - np.median(bg))
    diff = np.abs(g_adj - bg)
    mad = float(np.median(np.abs(diff - np.median(diff)))) or 1.0
    z = (diff - np.median(diff)) / (1.4826 * mad)

    m = ((z > z_thr) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return _pick_component(m > 0, box_px)


# ---------------------------------------------------------------------- sam mask


SAM_MODEL = "facebook/sam-vit-base"

# Set FIGLIB_SAM_DEVICE to pin the device; otherwise CUDA is used when it is present.
# The mask cache does not record which device produced it, so a cache must not be built
# half on one and half on the other -- delete `out/masks/cache/*.sam.npz` and rebuild
# rather than resuming across a device change.
_DEVICE_ENV = "FIGLIB_SAM_DEVICE"


def sam_device():
    """The device SAM will run on, decided once and reported honestly.

    Half precision is deliberately NOT used. This box's GTX 1070 is Pascal, where fp16
    arithmetic runs at a small fraction of fp32 -- the usual "fp16 is free on a GPU"
    assumption is false here, and would cost speed as well as bits.
    """
    import os

    import torch
    want = os.environ.get(_DEVICE_ENV)
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sam():
    """Load SAM once, onto whichever device is available. Deferred so `diff` needs no torch."""
    global _SAM
    if _SAM is None:
        import torch
        from transformers import SamModel, SamProcessor
        dev = sam_device()
        model = SamModel.from_pretrained(SAM_MODEL).eval().to(dev)
        _SAM = (model, SamProcessor.from_pretrained(SAM_MODEL), torch, dev)
    return _SAM


def sam_mask(img_bgr: np.ndarray, box_px: tuple[int, int, int, int]) -> np.ndarray | None:
    """Box-prompted SAM mask, returned at the image's own resolution."""
    model, proc, torch, dev = _sam()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(rgb, input_boxes=[[list(map(float, box_px))]], return_tensors="pt")
    sizes = inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, multimask_output=False)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), *sizes)
    m = masks[0][0][0].numpy().astype(bool)
    return m if m.any() else None


# ----------------------------------------------------------------- the detection


def selected_detection(seq_name: str, conf_thr: float = 0.25,
                       window_s: tuple[int, int] = (0, 2400)) -> dict | None:
    """The detection `geolocate.bearings_for_fire` would pick for this sequence.

    Duplicated rather than imported because that function returns bearings, not the
    record they came from, and the mask work needs the frame.
    """
    path = ROOT / "out" / "yolo" / f"{seq_name.split('#')[0]}.json"
    if not path.exists():
        return None
    best = None
    for rec in sorted(json.loads(path.read_text()), key=lambda r: r["offset"]):
        if not (window_s[0] <= rec["offset"] <= window_s[1]):
            continue
        for d in rec["dets"]:
            if d["conf"] >= conf_thr and (best is None or d["conf"] > best["det"]["conf"]):
                best = {"det": d, "epoch": rec["epoch"], "offset": rec["offset"]}
    return best


# ------------------------------------------------------------------------ foot

def foot_for(seq_name: str, offset: int, det: dict) -> dict:
    """Both mask feet for one detection, read off the sequence mask cache.

    There used to be a second cache here, keyed by sequence and offset, holding just the
    two scalars. It is gone: `sequence_masks` already stores the masks these feet are
    derived from, and `sequence_masks` guarantees the selected detection's frame survives
    its subsample, so the scalars can be recomputed for free from the masks themselves.
    Two caches of the same computation is two chances for them to disagree.

    Values are `None` where the method produced no mask; the caller decides what to fall
    back to, since silently substituting a box edge for a mask foot would make the
    comparison meaningless.
    """
    out = {}
    for method in ("diff", "sam"):
        try:
            m = sequence_masks(seq_name, method)["masks"].get(offset)
        except Exception:
            m = None
        out[method] = None if m is None else foot_x(m)
    return out


# ------------------------------------------------------------------- rendering

PANEL_W = 620
# BGR. Green is the official ignition bearing, as everywhere else in this repo.
C_TRUTH, C_BOX = (60, 220, 60), (60, 60, 235)
C_CENTRE, C_UPWIND, C_FOOT = (0, 235, 235), (235, 160, 40), (235, 60, 235)


def _overlay(img: np.ndarray, mask: np.ndarray | None, color) -> np.ndarray:
    out = img.copy()
    if mask is not None and mask.any():
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        tint = np.zeros_like(out)
        tint[mask] = color
        out = cv2.addWeighted(out, 0.62, tint, 0.38, 0)
        cont, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cont, -1, color, 2)
    return out


def _vline(img, xf, color, dashed=False):
    if xf is None:
        return
    x = int(round(xf * img.shape[1]))
    if not dashed:
        cv2.line(img, (x, 0), (x, img.shape[0]), color, 3)
        return
    for y in range(0, img.shape[0], 26):
        cv2.line(img, (x, y), (x, min(y + 13, img.shape[0])), color, 3)


def _label(img, text, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (18, 18, 18), -1)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return img


def render_fire(fire_id: str, conf_thr: float = 0.25, pad: float = 0.45,
                out: Path | None = None):
    """Three panels per camera: the box, the diff mask, the SAM mask.

    Cropped around the detection, because at full frame a plume forty kilometers away is
    a smudge a hundred pixels wide and nothing about a mask is visible.
    """
    from .geom import bearing_x_frac
    from .wind import upwind_x, wind_at

    cams = json.loads((META / "cams.json").read_text())
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    res = {r["fire_id"]: r for r in json.loads((META / "resolved.json").read_text())}

    fire, rec = fires.get(fire_id), res.get(fire_id)
    if not fire:
        print(f"  {fire_id}: unknown")
        return None
    truth = (rec or {}).get("truth")

    rows, report = [], []
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        sel = selected_detection(seq_name, conf_thr)
        tgz = TGZ / f"{seq_name.split('#')[0]}.tgz"
        if not sel or not tgz.exists():
            continue
        cam = cams[s["camera"]]
        d = sel["det"]

        frames = read_frames(tgz)
        idx = next((i for i, (_, o, _) in enumerate(frames) if o == sel["offset"]), None)
        if idx is None:
            continue
        img = cv2.imdecode(np.frombuffer(frames[idx][2], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        box_px = (int(d["x0"] * W), int(d["y0"] * H), int(d["x1"] * W), int(d["y1"] * H))

        # diff runs at detect_diff's working resolution, so its box is scaled to match
        gw = WIDTH
        gh = int(round(WIDTH * H / W))
        box_g = (int(d["x0"] * gw), int(d["y0"] * gh), int(d["x1"] * gw), int(d["y1"] * gh))
        try:
            md = diff_mask(frames, idx, box_g)
        except Exception as exc:
            print(f"    diff failed on {s['camera']}: {exc}")
            md = None
        try:
            ms = sam_mask(img, box_px)
        except Exception as exc:
            print(f"    sam failed on {s['camera']}: {exc}")
            ms = None

        w = wind_at(cam["lat"], cam["lon"], sel["epoch"])
        wfrom = w and w["from_deg_100m"]
        x_centre = (d["x0"] + d["x1"]) / 2
        x_upwind = upwind_x(d["x0"], d["x1"], cam["az"], wfrom)
        x_diff = foot_x(md) if md is not None else None
        x_sam = foot_x(ms) if ms is not None else None
        x_truth = bearing_x_frac(cam, truth["lat"], truth["lon"]) if truth else None

        report.append({"camera": s["camera"], "offset": sel["offset"], "conf": d["conf"],
                       "x_centre": round(x_centre, 4), "x_upwind": round(x_upwind, 4),
                       "x_foot_diff": x_diff and round(x_diff, 4),
                       "x_foot_sam": x_sam and round(x_sam, 4),
                       "x_truth": x_truth and round(x_truth, 4)})

        # crop window, in whole-frame fractions, so every x above stays comparable
        bw, bh = d["x1"] - d["x0"], d["y1"] - d["y0"]
        cx0 = max(0.0, d["x0"] - pad * bw)
        cx1 = min(1.0, d["x1"] + pad * bw)
        cy0 = max(0.0, d["y0"] - pad * bh)
        cy1 = min(1.0, d["y1"] + pad * bh)


        raw = img.copy()
        cv2.rectangle(raw, box_px[:2], box_px[2:], C_BOX, 3)

        panes = []
        for a, m, col, name, xf in (
                (raw, None, None, f"{s['camera']} t{sel['offset']:+d}s conf {d['conf']:.2f}", None),
                (img, md, C_FOOT, f"diff mask  foot x={_fmt(x_diff)}", x_diff),
                (img, ms, C_FOOT, f"SAM mask   foot x={_fmt(x_sam)}", x_sam)):
            v = _overlay(a, m, col) if col else a.copy()
            sub = v[int(cy0 * H):int(cy1 * H), int(cx0 * W):int(cx1 * W)].copy()
            for xx, c, dash in ((x_truth, C_TRUTH, False), (x_centre, C_CENTRE, True),
                                (x_upwind, C_UPWIND, True), (xf, C_FOOT, False)):
                if xx is None or not (cx0 <= xx <= cx1):
                    continue
                _vline_at(sub, int(round((xx - cx0) * W)), c, dash)
            p = cv2.resize(sub, (PANEL_W, int(PANEL_W * sub.shape[0] / sub.shape[1])))
            panes.append(_label(p, name))

        h = max(p.shape[0] for p in panes)
        panes = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 4, cv2.BORDER_CONSTANT,
                                    value=(18, 18, 18)) for p in panes]
        rows.append(np.hstack(panes))

    if not rows:
        print(f"  {fire_id}: nothing to render")
        return None
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(18, 18, 18)) for r in rows]
    grid = np.vstack(rows)

    hdr = np.zeros((84, grid.shape[1], 3), np.uint8)
    cv2.putText(hdr, f"{fire_id}" + (f"  ->  {truth['name']}" if truth else ""),
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    lx = 12
    for txt, c in (("truth", C_TRUTH), ("box center", C_CENTRE),
                   ("upwind edge", C_UPWIND), ("mask foot", C_FOOT)):
        cv2.line(hdr, (lx, 62), (lx + 26, 62), c, 4)
        cv2.putText(hdr, txt, (lx + 34, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        lx += 46 + 13 * len(txt)
    grid = np.vstack([hdr, grid])

    out = out or ROOT / "out" / "masks" / f"{fire_id}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  {fire_id}: {len(rows)} cameras -> {out.relative_to(ROOT)}")
    for r in report:
        print("   ", json.dumps(r))
    return out, report


def _fmt(v):
    return "--" if v is None else f"{v:.4f}"


def _vline_at(img, x, color, dashed=False):
    if not (0 <= x < img.shape[1]):
        return
    if not dashed:
        cv2.line(img, (x, 0), (x, img.shape[0]), color, 2)
        return
    for y in range(0, img.shape[0], 22):
        cv2.line(img, (x, y), (x, min(y + 11, img.shape[0])), color, 2)


if __name__ == "__main__":
    import sys
    for fid in (sys.argv[1:] or ["20240701_Kitchenfire"]):
        render_fire(fid)


# ------------------------------------------------------- whole-sequence masks

MASK_DIR = ROOT / "out" / "masks" / "cache"


def detected_frames(seq_name: str, conf_thr: float = 0.25,
                    window_s: tuple[int, int] = (0, 2400)) -> dict[int, dict]:
    """Offset -> the most confident detection at that offset, inside the window."""
    path = ROOT / "out" / "yolo" / f"{seq_name.split('#')[0]}.json"
    if not path.exists():
        return {}
    out = {}
    for rec in json.loads(path.read_text()):
        if not (window_s[0] <= rec["offset"] <= window_s[1]):
            continue
        good = [d for d in rec["dets"] if d["conf"] >= conf_thr]
        if good:
            out[rec["offset"]] = max(good, key=lambda d: d["conf"])
    return out


# Cap on frames per sequence, or None for every frame carrying a detection.
#
# This was 12 while SAM ran on CPU at 13 s/frame, which was a budget decision dressed up
# as a modeling one. On the GTX 1070 the same call is 0.53 s, so all 2,883 detected
# frames across the scoring set cost about twenty-five minutes and the cap comes off.
# That matters for one method specifically: `plumefit.sequence_fit` estimates a single
# shared apex across the sequence, so frames are the whole resource it has, and it was
# the only fit that came close to the box's upwind edge while starved of them.
MAX_SEQ_FRAMES = None


def sequence_masks(seq_name: str, method: str, conf_thr: float = 0.25,
                   window_s: tuple[int, int] = (0, 2400),
                   max_frames: int = MAX_SEQ_FRAMES) -> dict:
    """Every detected frame's mask for one sequence, at `detect_diff`'s resolution.

    Cached as one bit-packed npz per sequence and method. Both methods are resampled
    onto the same grid so a column index means the same thing in either -- the fits
    below compare them, and a half-pixel of resampling difference would show up as a
    difference in kilometers.

    Returns {"masks": {offset: bool array}, "horizon_y": float, "shape": (h, w)}.
    """
    stem = seq_name.split("#")[0]
    dest = MASK_DIR / f"{stem}.{method}.npz"
    if dest.exists():
        z = np.load(dest)
        h, w = int(z["h"]), int(z["w"])
        masks = {int(k[4:]): np.unpackbits(z[k]).astype(bool)[:h * w].reshape(h, w)
                 for k in z.files if k.startswith("off_")}
        return {"masks": masks, "horizon_y": float(z["horizon_y"]), "shape": (h, w)}

    dets = detected_frames(seq_name, conf_thr, window_s)
    if max_frames and len(dets) > max_frames:
        offs = sorted(dets)
        keep = {offs[int(round(i))]
                for i in np.linspace(0, len(offs) - 1, max_frames)}
        # The frame geolocate.py actually selects must survive the subsample, or the
        # single-frame fits and the sequence fit would be looking at different plumes.
        sel = selected_detection(seq_name, conf_thr, window_s)
        if sel:
            keep.add(sel["offset"])
        dets = {o: d for o, d in dets.items() if o in keep}
    tgz = TGZ / f"{stem}.tgz"
    if not dets or not tgz.exists():
        return {"masks": {}, "horizon_y": 0.0, "shape": (0, 0)}

    frames = read_frames(tgz)
    out, horizon_y, shape = {}, 0.0, (0, 0)

    if method == "diff":
        # One rolling pass over the sequence, so twenty frames of background are built
        # once rather than once per detection.
        hist: list[np.ndarray] = []
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
        for i, (_, off, blob) in enumerate(frames):
            try:
                g = _load_gray(blob)
            except ValueError:
                continue
            if len(hist) < BG_LAG_FRAMES:
                hist.append(g)
                continue
            if not horizon_y:
                from .detect_diff import estimate_horizon
                horizon_y = float(estimate_horizon(np.median(np.stack(hist), axis=0)))
            shape = g.shape
            if off in dets:
                bg = np.median(np.stack(hist[:BG_FRAMES]), axis=0)
                g_adj = g - (np.median(g) - np.median(bg))
                diff = np.abs(g_adj - bg)
                mad = float(np.median(np.abs(diff - np.median(diff)))) or 1.0
                z = (diff - np.median(diff)) / (1.4826 * mad)
                m = cv2.morphologyEx(((z > 4.0) * 255).astype(np.uint8),
                                     cv2.MORPH_OPEN, k)
                d = dets[off]
                gh, gw = g.shape
                box = (int(d["x0"] * gw), int(d["y0"] * gh),
                       int(d["x1"] * gw), int(d["y1"] * gh))
                c = _pick_component(m > 0, box)
                if c.any():
                    out[off] = c
            hist.append(g)
            hist.pop(0)

    elif method == "sam":
        by_off = {o: b for _, o, b in frames}
        for off, d in sorted(dets.items()):
            blob = by_off.get(off)
            if blob is None:
                continue
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            gh = int(round(WIDTH * H / W))
            shape = (gh, WIDTH)
            if not horizon_y:
                from .detect_diff import estimate_horizon
                g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (WIDTH, gh),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
                horizon_y = float(estimate_horizon(g))
            box = (int(d["x0"] * W), int(d["y0"] * H),
                   int(d["x1"] * W), int(d["y1"] * H))
            try:
                m = sam_mask(img, box)
            except Exception:
                m = None
            if m is not None:
                out[off] = cv2.resize(m.astype(np.uint8), (WIDTH, gh),
                                      interpolation=cv2.INTER_NEAREST) > 0
    else:
        raise ValueError(method)

    if shape != (0, 0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest, h=shape[0], w=shape[1], horizon_y=horizon_y,
            **{f"off_{o}": np.packbits(m.ravel()) for o, m in out.items()})
    return {"masks": out, "horizon_y": horizon_y, "shape": shape}


def build_all(methods=("diff", "sam")) -> None:
    """Precompute the mask cache for every sequence the geolocation set touches."""
    import sys
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    wanted = []
    for r in json.loads((META / "resolved.json").read_text()):
        if r["tier"] not in ("confirmed", "probable") or not r.get("triangulable"):
            continue
        for sn in fires[r["fire_id"]]["sequences"]:
            s = seqs.get(sn)
            if s and s["has_pose"] and sn not in wanted:
                wanted.append(sn)
    for k, sn in enumerate(wanted, 1):
        for m in methods:
            try:
                r = sequence_masks(sn, m)
            except Exception as exc:
                print(f"  FAIL {sn} {m}: {exc}", file=sys.stderr)
                continue
            print(f"[{k}/{len(wanted)}] {sn} {m}: {len(r['masks'])} masks", flush=True)


# ---------------------------------------------------------------- mask video


VID_W = 560           # per-camera panel width
TRACE_H = 74          # foot_x-against-time strip under each panel


def _crop_box(masks: dict, shape, pad: float = 0.6):
    """One fixed crop per camera, covering every mask in the sequence.

    A crop that follows the plume frame by frame makes the mask look steadier than it
    is -- the camera appears to track the smoke. Fixing the window over the union of all
    masks means any wander in the mask is wander on screen, which is the whole point of
    watching this as video rather than as stills.
    """
    h, w = shape
    ys, xs = [], []
    for m in masks.values():
        r = np.nonzero(m.any(axis=1))[0]
        c = np.nonzero(m.any(axis=0))[0]
        if r.size and c.size:
            ys += [r[0], r[-1]]
            xs += [c[0], c[-1]]
    if not xs:
        return 0.0, 0.0, 1.0, 1.0
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = max(x1 - x0, 40), max(y1 - y0, 30)
    return (max(0.0, (x0 - pad * bw) / w), max(0.0, (y0 - pad * bh) / h),
            min(1.0, (x1 + pad * bw) / w), min(1.0, (y1 + pad * bh) / h))


def _trace(hist, truth_x, width, lo, hi, height=TRACE_H):
    """foot_x against time: is the estimate steady, or is it hunting?"""
    img = np.full((height, width, 3), 22, np.uint8)
    if hi - lo < 1e-6:
        lo, hi = lo - 0.02, hi + 0.02
    def px(x):
        return int(round((x - lo) / (hi - lo) * (height - 12))) + 6
    if truth_x is not None and lo <= truth_x <= hi:
        y = height - px(truth_x)
        cv2.line(img, (0, y), (width, y), C_TRUTH, 2)
    pts = [(int(i / max(len(hist) - 1, 1) * (width - 1)), height - px(v))
           for i, v in enumerate(hist) if v is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, C_FOOT, 2)
    for p in pts[-1:]:
        cv2.circle(img, p, 4, C_FOOT, -1)
    cv2.putText(img, "foot x vs time", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (150, 150, 150), 1)
    return img


def mask_video(fire_id: str, method: str = "sam", fps: int = 4, max_cams: int = 4,
               out: Path | None = None) -> Path | None:
    """Every camera's mask for one fire, moving through the sequence together.

    The stills in `out/masks/*.jpg` show one frame each, chosen as the detector's most
    confident -- which is the late, flattened plume. What they cannot show is whether the
    mask is tracking one physical object or re-deciding what it is every two minutes.
    That distinction is what decides whether a *sequence* fit has anything to pool, so
    it is worth looking at directly.
    """
    from .geom import bearing_x_frac

    cams = json.loads((META / "cams.json").read_text())
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    res = {r["fire_id"]: r for r in json.loads((META / "resolved.json").read_text())}

    fire = fires.get(fire_id)
    if not fire:
        print(f"  {fire_id}: unknown")
        return None
    truth = (res.get(fire_id) or {}).get("truth")

    lanes = []
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        cache = sequence_masks(seq_name, method)
        if not cache["masks"]:
            continue
        tgz = TGZ / f"{seq_name.split('#')[0]}.tgz"
        if not tgz.exists():
            continue
        cam = cams[s["camera"]]
        blobs = {o: b for _, o, b in read_frames(tgz)}
        lanes.append({"camera": s["camera"], "cam": cam, "masks": cache["masks"],
                      "blobs": blobs, "shape": cache["shape"],
                      "truth_x": bearing_x_frac(cam, truth["lat"], truth["lon"])
                                 if truth else None})
        if len(lanes) >= max_cams:
            break
    if not lanes:
        print(f"  {fire_id}: no {method} masks cached")
        return None

    offsets = sorted({o for ln in lanes for o in ln["masks"]})
    for ln in lanes:
        ln["crop"] = _crop_box(ln["masks"], ln["shape"])
        feet = [v for v in (foot_x(m) for m in ln["masks"].values()) if v is not None]
        cx0, _, cx1, _ = ln["crop"]
        ln["range"] = (min(feet + [cx0]), max(feet + [cx1])) if feet else (cx0, cx1)
        ln["hist"] = []

    panel_h = None
    vw = None
    out = out or ROOT / "out" / "videos" / f"masks_{method}_{fire_id}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    for off in offsets:
        panes = []
        for ln in lanes:
            near = min(ln["masks"], key=lambda o: abs(o - off)) if ln["masks"] else None
            m = ln["masks"][near] if near is not None and abs(near - off) <= 150 else None
            blob = ln["blobs"].get(off) or ln["blobs"].get(near)
            if blob is None:
                continue
            img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            v = _overlay(img, m, C_FOOT)
            cx0, cy0, cx1, cy1 = ln["crop"]
            sub = v[int(cy0 * H):int(cy1 * H), int(cx0 * W):int(cx1 * W)].copy()
            if sub.size == 0:
                continue
            fx = foot_x(m) if m is not None else None
            ln["hist"].append(fx)
            for xx, c in ((ln["truth_x"], C_TRUTH), (fx, C_FOOT)):
                if xx is None or not (cx0 <= xx <= cx1):
                    continue
                _vline_at(sub, int(round((xx - cx0) * W)), c)
            p = cv2.resize(sub, (VID_W, int(VID_W * sub.shape[0] / sub.shape[1])))
            px = int(m.sum()) if m is not None else 0
            p = _label(p, f"{ln['camera']}  t{off:+d}s  {px} px"
                          + ("" if m is not None else "  NO MASK"))
            lo, hi = ln["range"]
            p = np.vstack([p, _trace(ln["hist"], ln["truth_x"], VID_W, lo, hi)])
            panes.append(p)
        if not panes:
            continue
        h = max(p.shape[0] for p in panes)
        panes = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 4, cv2.BORDER_CONSTANT,
                                    value=(18, 18, 18)) for p in panes]
        cols = 2 if len(panes) > 1 else 1
        rows_img = [np.hstack(panes[i:i + cols]) for i in range(0, len(panes), cols)]
        wmax = max(r.shape[1] for r in rows_img)
        rows_img = [cv2.copyMakeBorder(r, 0, 4, 0, wmax - r.shape[1],
                                       cv2.BORDER_CONSTANT, value=(18, 18, 18))
                    for r in rows_img]
        grid = np.vstack(rows_img)
        hdr = np.zeros((44, grid.shape[1], 3), np.uint8)
        cv2.putText(hdr, f"{fire_id}  {method.upper()} masks"
                         + (f"  ->  {truth['name']}" if truth else "")
                         + "     green = official ignition bearing,"
                           "  magenta = mask foot",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)
        grid = np.vstack([hdr, grid])

        if vw is None:
            panel_h = grid.shape[:2]
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (grid.shape[1], grid.shape[0]))
        if grid.shape[:2] != panel_h:
            grid = cv2.resize(grid, (panel_h[1], panel_h[0]))
        vw.write(grid)

    if vw is None:
        print(f"  {fire_id}: nothing rendered")
        return None
    vw.release()
    print(f"  {fire_id}: {len(offsets)} frames, {len(lanes)} cameras -> "
          f"{out.relative_to(ROOT)}")
    return out
