"""Core ML build of the pyronear detector, for the Apple-silicon edge measurements.

Deliberately a drop-in for `detect_yolo`: same letterbox, same NMS, same `Det` fields,
same JSON on disk. Every downstream stage -- bearings, the likelihood field, evidence
accumulation, the false-alarm sweep -- therefore runs against Core ML detections without
a line changing, which is what makes "what did quantization cost?" answerable in
kilometres of geolocation error and seconds-to-alert rather than in mAP.

The exported model takes an *image* input with the 1/255 scaling folded in, so unlike
the ONNX path there is no manual normalisation here. Everything else is shared.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np

from .detect_yolo import Det, FrameDets, letterbox, nms, read_frames, IMGSZ

ROOT = Path(__file__).resolve().parents[2]
TGZ_DIR = ROOT / "data" / "tgz"
MODELS = ROOT / "models"

# Core ML names these; "ALL" lets the runtime place ops itself, which is the setting a
# deployment would use and also the one most likely to differ from what was requested.
# Never trust the request -- confirm placement against ANE wattage.
UNITS = {"cpu": "CPU_ONLY", "gpu": "CPU_AND_GPU", "ane": "CPU_AND_NE", "all": "ALL"}


def make_model(path: Path, units: str = "all"):
    import coremltools as ct
    return ct.models.MLModel(str(path),
                             compute_units=ct.ComputeUnit[UNITS[units]])


def predict_raw(model, img: np.ndarray):
    """Letterbox, hand Core ML a PIL image, return (raw output, r, left, top)."""
    from PIL import Image
    canvas, r, left, top = letterbox(img, IMGSZ)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    out = model.predict({"image": pil})
    y = next(iter(out.values()))
    return y, r, left, top


def decode(y, r, left, top, W, H, conf_thr: float = 0.05) -> list[Det]:
    y = y[0] if y.ndim == 3 else y
    if y.shape[0] < y.shape[1]:
        y = y.T                                   # (anchors, 5): cx cy w h conf
    conf = y[:, 4]
    m = conf > conf_thr
    if not m.any():
        return []
    y, conf = y[m], conf[m]
    xy, wh = y[:, :2], y[:, 2:4]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
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


def detect(model, img: np.ndarray, conf_thr: float = 0.05) -> list[Det]:
    H, W = img.shape[:2]
    y, r, left, top = predict_raw(model, img)
    return decode(np.asarray(y), r, left, top, W, H, conf_thr)


def run_sequence(model, tgz: Path) -> list[FrameDets]:
    out = []
    for epoch, offset, blob in read_frames(tgz):
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out.append(FrameDets(epoch, offset, [asdict(d) for d in detect(model, img)]))
    return out


def main(argv: list[str]) -> None:
    import sys
    variant = argv[0] if argv else "fp32"
    units = argv[1] if len(argv) > 1 else "all"
    model_path = MODELS / f"pyronear_rr_v8.1.0{'' if variant == 'fp32' else '_' + variant}.mlpackage"
    out_dir = ROOT / "out" / "coreml" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(model_path, units)
    paths = sorted(TGZ_DIR.glob("*.tgz"))
    t_start = time.time()
    for k, p in enumerate(paths, 1):
        dest = out_dir / f"{p.name[:-4]}.json"
        if dest.exists():
            continue
        try:
            recs = run_sequence(model, p)
        except Exception as exc:
            print(f"  FAIL {p.name}: {exc}", file=sys.stderr)
            continue
        dest.write_text(json.dumps([asdict(r) for r in recs]) + "\n")
        n = sum(len(r.dets) for r in recs)
        print(f"[{k}/{len(paths)}] {p.name[:-4]}: {len(recs)} frames, {n} dets, "
              f"{time.time() - t_start:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
