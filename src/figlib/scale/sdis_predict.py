"""The released pyronear model over pyro-sdis, for comparing its boxes with the labels.

Pre- and post-processing are detect_yolo.detect() itself -- the square 1024 letterbox,
BGR->RGB, /255, the confidence floor and the 0.45 NMS -- so boxes here are directly
comparable with every FIgLib detection in out/yolo/. Only the forward pass differs: the
PyTorch checkpoint the ONNX file was exported from (best.pt, whose sha256 equals the
`weights.sha256` in the model's manifest.yaml), run on the GPU behind a two-method
adapter that looks like an onnxruntime session. `--check N` runs N images through both
and reports the largest disagreement, since the ONNX path is the one results cite.

    .venv-train/bin/python -m src.figlib.scale.sdis_predict val|train [--check N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ..detect_yolo import detect
from ..provenance import code_state, sha256_file, utc_now
from .recipe import ROOT

PT = ROOT / "models" / "pyronear_rr_v8.1.0.pt"
PT_SHA256 = "01bb40dabd1f994ac220cd8f779ee7f8aa0bae88d2b7410916788a76bdc89bbb"
IMAGES = ROOT / "data" / "sdis" / "yolo" / "images"
OUT = ROOT / "out" / "scale" / "sdis_rr"
CONF_FLOOR = 0.05


class TorchSession:
    """Just enough of onnxruntime.InferenceSession for detect_yolo.detect()."""

    class _Input:
        name = "images"

    def __init__(self, pt: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(pt)).model.float().eval().cuda()

    def get_inputs(self):
        return [self._Input()]

    @torch.no_grad()
    def run(self, _outputs, feeds):
        x = torch.from_numpy(next(iter(feeds.values()))).cuda()
        y = self.model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        return [y.float().cpu().numpy()]


def check(paths: list[Path], sess) -> dict:
    import onnxruntime as ort
    onnx = ort.InferenceSession(str(ROOT / "models" / "pyronear_rr_v8.1.0.onnx"),
                                providers=["CPUExecutionProvider"])
    worst = {"n_images": len(paths), "count_mismatch": 0, "max_conf_diff": 0.0,
             "max_coord_diff": 0.0}
    for p in paths:
        img = cv2.imread(str(p))
        a, b = detect(sess, img, CONF_FLOOR), detect(onnx, img, CONF_FLOOR)
        if len(a) != len(b):
            worst["count_mismatch"] += 1
            continue
        for da, db in zip(a, b):
            worst["max_conf_diff"] = max(worst["max_conf_diff"], abs(da.conf - db.conf))
            worst["max_coord_diff"] = max(worst["max_coord_diff"], *(
                abs(getattr(da, k) - getattr(db, k)) for k in ("x0", "y0", "x1", "y1")))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("val", "train"))
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    if sha256_file(PT) != PT_SHA256:
        raise SystemExit(f"{PT} is not the pinned rapid-raccoon v8.1.0 checkpoint")
    paths = sorted((IMAGES / a.split).glob("*.jpg"))
    sess = TorchSession(PT)
    if a.check:
        rng = np.random.default_rng(0)
        sample = [paths[i] for i in rng.choice(len(paths), a.check, replace=False)]
        print(json.dumps(check(sample, sess)))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{a.split}.jsonl"
    t0, started = time.perf_counter(), utc_now()
    with open(dest, "w") as fh:
        for k, p in enumerate(paths, 1):
            img = cv2.imread(str(p))
            H, W = img.shape[:2]
            dets = [asdict(d) for d in detect(sess, img, CONF_FLOOR)]
            fh.write(json.dumps({"image": p.stem, "w": W, "h": H, "dets": dets}) + "\n")
            if k % 2000 == 0:
                print(f"{k}/{len(paths)} {k / (time.perf_counter() - t0):.1f} img/s",
                      file=sys.stderr, flush=True)
    (OUT / f"{a.split}.meta.json").write_text(json.dumps({
        "model": {"file": str(PT.relative_to(ROOT)), "sha256": PT_SHA256},
        "conf_floor": CONF_FLOOR, "imgsz": 1024, "nms_iou": 0.45,
        "images": len(paths), "started_utc": started, "finished_utc": utc_now(),
        "output_sha256": sha256_file(dest), "code": code_state(),
        "sdis_manifest_sha256": sha256_file(ROOT / "data" / "sdis" / "manifest.json"),
    }, indent=1) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
