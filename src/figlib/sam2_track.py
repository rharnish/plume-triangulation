"""Sequence input: seed from motion or from a detector's box, track with SAM2's video model.

Every other model in this experiment (`open_vocab.py`, `masks.py`) looks at one frame
at a time. SAM2's video variant is built the other way -- it keeps a memory bank across
frames, so a mask given on one frame gets propagated through the rest of the sequence
instead of being re-detected from scratch each time. For a fixed FIgLib camera, that is
the natural fit: whatever starts moving usually keeps moving in roughly the same place,
and re-running an object detector on every frame throws that continuity away.

**Two ways to seed.** `masks.py` seeds SAM from the pyronear box; `open_vocab.py` seeds
Grounding DINO from a text phrase. `motion_seeds` seeds from motion itself: it reruns the
same median-background / MAD-z-score differencing `detect_diff.py` already uses to score
smoke, but asks a smaller question -- not "how smoke-like is this change" but "what
changed" -- and hands every big-enough connected component to SAM2 as its own tracked
object. That means it seeds on *anything* that moves into frame, plume or not: a truck,
a shifting cloud edge, a tree in wind. Useful as a first pass, where which of its tracks
is actually smoke is a question for the viewer's eye, same as `open_vocab.py`'s category
boxes were.

**`detector_seed` instead anchors the track in a real detection**, for the opposite
problem: pyronear's own box is trustworthy only once confidence is high, and by then the
plume has often drifted or flattened (see `geolocate.py`'s `foot_diff`/`foot_sam`
variants). It finds the frame where `falsealarm.py`'s own k-of-m persistence rule would
first call an alarm, seeds SAM2 there with *that* box -- not a motion blob -- and
propagates backward. Every earlier mask in the resulting track is therefore still the
object a trusted detector actually found at the alarm frame; SAM2's memory is doing the
work of finding that object's earlier, fainter self, not re-detecting from scratch.
Nothing here re-checks the box at earlier frames (`falsealarm.py`'s own point is that the
detector was still below threshold there), so a track that drifts onto something else
mid-sequence -- a ridge, a cloud -- isn't caught automatically; watch for it in the
viewer, e.g. as a jump in the mask centroid between frames.

`facebook/sam2.1-hiera-tiny` -- the smallest SAM2 video checkpoint, Apache-2.0, not
trained on FIgLib. Ships in the same `transformers` pin `requirements-masks.txt`
already installs; no new dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np

from .detect_yolo import read_frames
from .masks import sam_device

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "sam2track"

SAM2_VIDEO_MODEL = "facebook/sam2.1-hiera-tiny"

BG_FRAMES = 5          # frames immediately before the seed frame, used as its background
Z_THR = 4.0
MIN_AREA_FRAC = 3e-4    # ~2000 px at 3072x2048
OPEN_K = 9
MAX_OBJECTS = 6         # cap: memory cost is per object, and a crowded seed is usually noise
BOX_PAD_FRAC = 0.15     # the motion mask undershoots a plume's visible extent

# BGR, cycled if there are more objects than colors.
PALETTE = [
    (0, 0, 255), (255, 128, 0), (0, 220, 220), (0, 255, 0),
    (255, 0, 255), (0, 128, 255), (180, 0, 255), (255, 255, 0),
]

_SAM2V = None


def _sam2_video():
    global _SAM2V
    if _SAM2V is None:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        dev = sam_device()
        proc = Sam2VideoProcessor.from_pretrained(SAM2_VIDEO_MODEL)
        model = Sam2VideoModel.from_pretrained(SAM2_VIDEO_MODEL).eval().to(dev)
        _SAM2V = (model, proc, torch, dev)
    return _SAM2V


def load_sequence(tgz: Path, max_frames: int | None = None) -> list[np.ndarray]:
    """Decoded BGR frames from a FIgLib archive, in time order."""
    return [img for _, img in load_sequence_timed(tgz, max_frames)]


def load_sequence_timed(tgz: Path, max_frames: int | None = None
                         ) -> list[tuple[tuple[int, int], np.ndarray]]:
    """((epoch, offset), decoded BGR frame) per frame, in time order.

    Same as `load_sequence` but keeps the offset each frame was named with, so a seed
    frame can be picked by offset (matching `out/yolo/*.json`) rather than by assuming
    index parity with the detection records -- a decode failure here would otherwise
    silently shift every later index.
    """
    frames = read_frames(tgz)
    if max_frames:
        frames = frames[:max_frames]
    out = []
    for epoch, offset, blob in frames:
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(((epoch, offset), img))
    return out


def detector_seed(seq_name: str, tau: float = 0.4, k: int = 1, m: int = 1
                   ) -> dict | None:
    """The frame and box `falsealarm.py`'s own k-of-m rule would first alarm on.

    Reruns `alarms_single_camera` over the full sequence (not just the negatives
    `falsealarm.sweep` scores) and takes the first alarm at or after ignition -- the
    frame an operator would actually have been shown first. Returns None if the rule
    never fires post-ignition at these parameters.
    """
    from .falsealarm import frame_scores, alarms_single_camera

    scores = frame_scores(seq_name)
    if not scores:
        return None
    alarm_offsets = alarms_single_camera(scores, tau, k, m)
    trigger_off = next((o for o in alarm_offsets if o >= 0), None)
    if trigger_off is None:
        return None
    _off, epoch, _conf, dets = next(r for r in scores if r[0] == trigger_off)
    det = max(dets, key=lambda d: d["conf"])
    return {"offset": trigger_off, "epoch": epoch, "det": det, "tau": tau, "k": k, "m": m}


def detector_boxes_by_offset(seq_name: str) -> dict[int, list[dict]]:
    """Every pyronear box for this sequence, keyed by offset -- for overlaying the raw
    detector against a SAM2 track in the viewer, frame by frame rather than just at the
    seed. `seq_name` is looked up with any `#variant` suffix stripped, same convention
    `geolocate.py` and `falsealarm.py` use for `YOLO_DIR` filenames."""
    from .falsealarm import frame_scores
    return {off: dets for off, _epoch, _conf, dets in frame_scores(seq_name.split("#")[0]) if dets}


def motion_seeds(frames_bgr: list[np.ndarray], seed_idx: int, bg_frames: int = BG_FRAMES,
                  z_thr: float = Z_THR, min_area_frac: float = MIN_AREA_FRAC,
                  max_objects: int = MAX_OBJECTS) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of whatever moved into `frames_bgr[seed_idx]`, largest first."""
    if seed_idx < bg_frames:
        raise ValueError(f"need >= {bg_frames} frames before seed_idx={seed_idx}")
    h, w = frames_bgr[seed_idx].shape[:2]

    def gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    bg = np.median(np.stack([gray(f) for f in frames_bgr[seed_idx - bg_frames:seed_idx]]), axis=0)
    g = gray(frames_bgr[seed_idx])
    g_adj = g - (np.median(g) - np.median(bg))     # absorb a whole-frame brightness shift
    diff = np.abs(g_adj - bg)
    mad = float(np.median(np.abs(diff - np.median(diff)))) or 1.0
    z = (diff - np.median(diff)) / (1.4826 * mad)

    m = ((z > z_thr) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    min_area = min_area_frac * h * w
    comps = sorted((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                    if stats[i, cv2.CC_STAT_AREA] >= min_area)[::-1]

    boxes = []
    for _, i in comps[:max_objects]:
        x, y, cw, ch = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        px, py = int(cw * BOX_PAD_FRAC), int(ch * BOX_PAD_FRAC)
        boxes.append((int(max(0, x - px)), int(max(0, y - py)),
                      int(min(w, x + cw + px)), int(min(h, y + ch + py))))
    return boxes


def track(frames_bgr: list[np.ndarray], seed_idx: int,
          boxes: list[tuple[int, int, int, int]]) -> dict[int, dict[int, np.ndarray]]:
    """SAM2 video propagation from one seeded frame across the whole sequence.

    One forward pass per frame covers every seeded object at once -- the vision
    backbone runs once regardless of object count, unlike `open_vocab.detect`'s
    per-phrase passes. Propagates forward from the seed frame to the end, then
    backward to the start, so a seed picked mid-sequence still covers it all.

    Returns {frame_idx: {obj_id: mask}}.
    """
    model, proc, torch, dev = _sam2_video()
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    h, w = frames_bgr[0].shape[:2]

    # `init_video_session` resizes and stores every frame up front. Left on the default
    # (same device as inference), that is 80+ full frames processed and held on an 8 GB
    # card at once -- measured OOM here on a 3072x2048, 80-frame sequence. Processing
    # and storage stay on CPU; only the per-frame backbone features that actually need
    # the GPU move there, inside the model's own forward pass.
    session = proc.init_video_session(video=frames_rgb, inference_device=dev,
                                       processing_device="cpu", video_storage_device="cpu",
                                       dtype=torch.float32)
    obj_ids = list(range(1, len(boxes) + 1))
    proc.add_inputs_to_inference_session(
        inference_session=session, frame_idx=seed_idx, obj_ids=obj_ids,
        input_boxes=[[list(b) for b in boxes]], original_size=(h, w))

    result: dict[int, dict[int, np.ndarray]] = {}

    def _store(seg):
        masks = proc.post_process_masks([seg.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
        result[seg.frame_idx] = {oid: masks[i, 0].cpu().numpy()
                                  for i, oid in enumerate(seg.object_ids)}

    for seg in model.propagate_in_video_iterator(session, start_frame_idx=seed_idx):
        _store(seg)
    if seed_idx > 0:
        for seg in model.propagate_in_video_iterator(session, start_frame_idx=seed_idx, reverse=True):
            if seg.frame_idx != seed_idx:
                _store(seg)
    return result


def run_sequence(tgz: Path, seed_idx: int | None = None, max_frames: int | None = None,
                  seed: str = "motion", tau: float = 0.4, k: int = 1, m: int = 1) -> dict:
    """End to end: load, seed, track, save everything the viewer needs.

    `seed="motion"` (default) seeds from whatever moved into the frame at `seed_idx`
    (or a default shortly after enough background exists), same as before. `seed=
    "detector"` ignores `seed_idx` and instead seeds from `detector_seed` -- the frame
    and box where `falsealarm.py`'s own k-of-m rule would first alarm -- then tracks
    backward from *that* real detection to the start of the sequence.

    Per frame under `OUT / stem`: the untouched frame (`NNNN.orig.jpg`) and a label
    image (`NNNN.masks.png`, pixel value = object id, 0 = background) -- same
    single-file-per-frame trick `open_vocab.py` uses for its masks, so the viewer never
    fetches one file per object. One `sequence.json` describes the objects and lists
    the frame stems in order.
    """
    seq_name = tgz.stem
    timed = load_sequence_timed(tgz, max_frames)
    frames = [img for _, img in timed]
    offsets = [off for (_epoch, off), _img in timed]
    h, w = frames[0].shape[:2]

    trigger = None
    if seed == "detector":
        trigger = detector_seed(seq_name, tau=tau, k=k, m=m)
        if trigger is None:
            raise RuntimeError(f"falsealarm's k-of-m rule (tau={tau}, k={k}, m={m}) "
                                f"never alarms post-ignition on {seq_name}")
        seed_idx = min(range(len(offsets)), key=lambda i: abs(offsets[i] - trigger["offset"]))
        d = trigger["det"]
        boxes = [(int(d["x0"] * w), int(d["y0"] * h), int(d["x1"] * w), int(d["y1"] * h))]
        stem = f"{seq_name}#detector"
    else:
        if seed_idx is None:
            seed_idx = min(BG_FRAMES + 5, len(frames) - 1)
        boxes = motion_seeds(frames, seed_idx)
        if not boxes:
            raise RuntimeError(f"no motion found at frame {seed_idx} of {seq_name} -- try a different seed_idx")
        stem = seq_name

    per_frame = track(frames, seed_idx, boxes)

    det_by_offset = detector_boxes_by_offset(seq_name)
    detector_boxes = {}
    for i, off in enumerate(offsets):
        dets = det_by_offset.get(off)
        if dets:
            detector_boxes[i] = [{"box": [round(d["x0"] * w), round(d["y0"] * h),
                                           round(d["x1"] * w), round(d["y1"] * h)],
                                   "conf": d["conf"]} for d in dets]

    seq_dir = OUT / stem
    seq_dir.mkdir(parents=True, exist_ok=True)
    frame_stems = []
    for i, img in enumerate(frames):
        fstem = f"{i:04d}"
        frame_stems.append(fstem)
        cv2.imwrite(str(seq_dir / f"{fstem}.orig.jpg"), img)
        label = np.zeros(img.shape[:2], np.uint8)
        for oid, mask in per_frame.get(i, {}).items():
            label[mask] = oid
        cv2.imwrite(str(seq_dir / f"{fstem}.masks.png"), label)

    record = {
        "sequence": stem, "width": frames[0].shape[1], "height": frames[0].shape[0],
        "seed_idx": seed_idx, "seed_mode": seed, "frames": frame_stems,
        "offsets": offsets, "detector_boxes": detector_boxes,
        "objects": [{"id": i + 1, "seed_box": list(box),
                     "color": list(PALETTE[i % len(PALETTE)][::-1])}   # BGR -> RGB
                    for i, box in enumerate(boxes)],
    }
    if trigger is not None:
        record["trigger"] = {"offset": trigger["offset"], "epoch": trigger["epoch"],
                              "conf": trigger["det"]["conf"], "tau": tau, "k": k, "m": m}
    (seq_dir / "sequence.json").write_text(json.dumps(record, indent=2))

    manifest = sorted(p.name for p in OUT.glob("*") if (p / "sequence.json").exists())
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    import shutil
    viewer_src = Path(__file__).resolve().parent / "sam2_track_viewer.html"
    if viewer_src.exists():
        shutil.copy(viewer_src, OUT / "viewer.html")

    return record


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tgz", type=Path, help="FIgLib sequence archive, e.g. data/tgz/foo.tgz")
    ap.add_argument("--seed", choices=("motion", "detector"), default="motion",
                     help="motion: seed from what moved at --seed-idx. detector: seed from "
                          "falsealarm.py's own first post-ignition alarm and track backward.")
    ap.add_argument("--seed-idx", type=int, default=None,
                     help="(--seed motion only) frame to detect motion and seed objects on "
                          "(default: shortly after enough background exists)")
    ap.add_argument("--tau", type=float, default=0.4, help="(--seed detector only) alarm threshold")
    ap.add_argument("--k", type=int, default=1, help="(--seed detector only) k-of-m")
    ap.add_argument("--m", type=int, default=1, help="(--seed detector only) k-of-m")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    record = run_sequence(args.tgz, seed_idx=args.seed_idx, max_frames=args.max_frames,
                           seed=args.seed, tau=args.tau, k=args.k, m=args.m)
    print(f"{record['sequence']}: {len(record['objects'])} objects seeded at frame "
          f"{record['seed_idx']}, tracked across {len(record['frames'])} frames -> "
          f"{OUT / record['sequence']}")
    print(f"\nviewer: cd {OUT} && python3 -m http.server 8009   then open "
          f"http://localhost:8009/viewer.html")


if __name__ == "__main__":
    main()
