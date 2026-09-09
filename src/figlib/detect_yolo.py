"""Run the pyronear YOLO11s smoke detector over FIgLib sequences.

NOTE ON PROVENANCE: this model was trained on FIgLib (see models/README.md), so its
detection and timing numbers on FIgLib measure memorisation as well as skill. They are
reported as a labelled reference point, never as a generalisation claim. Its *bearings*
are still sound, and geolocation error against official coordinates tests geometry
rather than generalisation -- which is why this detector is used for the geolocation
work without an asterisk.

Frames are letterboxed rather than squashed to the network's square input. Horizontal
fraction survives either way, but letterboxing is what the model was trained on, so it
detects better -- and the horizontal fraction is what becomes a bearing.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TGZ_DIR = ROOT / "data" / "tgz"
MODEL = ROOT / "models" / "pyronear_rr_v8.1.0.onnx"
OUT_DIR = ROOT / "out" / "yolo"
IMGSZ = 1024


@dataclass
class Det:
    conf: float
    cx: float          # all box fields are fractions of the ORIGINAL frame
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class FrameDets:
    epoch: int
    offset: int
    dets: list[dict]


def letterbox(img: np.ndarray, size: int = IMGSZ):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.45) -> list[int]:
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(int(i))
        if idx.size == 1:
            break
        xx0 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = ((boxes[idx[1:], 2] - boxes[idx[1:], 0]) *
             (boxes[idx[1:], 3] - boxes[idx[1:], 1]))
        idx = idx[1:][inter / (a + b - inter + 1e-9) <= thr]
    return keep


def detect(sess, img: np.ndarray, conf_thr: float = 0.05) -> list[Det]:
    H, W = img.shape[:2]
    canvas, r, left, top = letterbox(img)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0

    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    y = y[0] if y.ndim == 3 else y
    if y.shape[0] < y.shape[1]:
        y = y.T                                   # (anchors, 5): cx cy w h conf

    conf = y[:, 4]
    m = conf > conf_thr
    if not m.any():
        return []
    y, conf = y[m], conf[m]

    xy, wh = y[:, :2], y[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)   # letterboxed px
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r             # -> original px
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r

    out = []
    for i in nms(boxes, conf):
        x0, y0, x1, y1 = boxes[i]
        out.append(Det(
            conf=round(float(conf[i]), 4),
            cx=round(float((x0 + x1) / 2 / W), 5),
            cy=round(float((y0 + y1) / 2 / H), 5),
            x0=round(float(x0 / W), 5), y0=round(float(y0 / H), 5),
            x1=round(float(x1 / W), 5), y1=round(float(y1 / H), 5)))
    return out


def read_frames(tgz: Path):
    frames = []
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            name = Path(m.name).name
            if name.endswith(".jpg") and "_" in name:
                e, o = name[:-4].split("_")
                fh = tf.extractfile(m)
                if fh:
                    frames.append((int(e), int(o), fh.read()))
    frames.sort()
    return frames


def run_sequence(sess, tgz: Path) -> list[FrameDets]:
    out = []
    for epoch, offset, blob in read_frames(tgz):
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out.append(FrameDets(epoch, offset, [asdict(d) for d in detect(sess, img)]))
    return out


def make_session(threads: int = 4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])


def main(argv: list[str]) -> None:
    import sys
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(TGZ_DIR.glob("*.tgz"))
    if argv:
        paths = [p for p in paths if any(a in p.name for a in argv)]
    sess = make_session()
    for k, p in enumerate(paths, 1):
        dest = OUT_DIR / f"{p.name[:-4]}.json"
        if dest.exists():
            continue
        try:
            recs = run_sequence(sess, p)
        except Exception as exc:
            print(f"  FAIL {p.name}: {exc}", file=sys.stderr)
            continue
        dest.write_text(json.dumps([asdict(r) for r in recs]) + "\n")
        n = sum(len(r.dets) for r in recs)
        print(f"[{k}/{len(paths)}] {p.name[:-4]}: {len(recs)} frames, {n} dets",
              flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
