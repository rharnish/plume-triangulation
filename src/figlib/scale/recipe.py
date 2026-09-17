"""pyronear's training recipe, and the four variants trained with it.

RECIPE is every argument in which pyronear's rapid-raccoon v8.1.0 run differs from
ultralytics 8.4.146's defaults, read from the model's manifest.yaml (HF revision
81a1f6bd060ca7100e496ff5dedeb2d327d135e0, `train_run_args`), minus paths, device and the
inference-only keys (iou, nms, tracker). Only `model` and `imgsz` vary between variants.

The manifest also records what that run trained on: pyronear/pyro-dataset
v4.0.0-corrected (16,100 image/label pairs), fetched with `dvc get`. That set includes
FIgLib, so it is deliberately *not* used here; pyro-sdis (29,537 train images, no FIgLib)
is, which is why these models cannot reproduce rapid-raccoon's numbers exactly and are
compared with each other first and with it second.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_YAML = ROOT / "data" / "sdis" / "yolo" / "data.yaml"
PRETRAINED = ROOT / "models" / "scale" / "pretrained"   # COCO yolo11{s,l}.pt, v8.3.0 assets
RUNS = ROOT / "models" / "scale"

RECIPE = {
    "epochs": 50,
    "patience": 20,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 0.0001,
    "lrf": 0.1,
    "warmup_epochs": 6,
    "degrees": 5.0,
    "translate": 0.15,
    "mixup": 0.2,
    "single_cls": True,
    "seed": 0,
    "deterministic": True,
    "close_mosaic": 10,
    "amp": True,
    "pretrained": True,
}

VARIANTS = {
    "s1024": ("yolo11s", 1024),
    "s1280": ("yolo11s", 1280),
    "l1024": ("yolo11l", 1024),
    "l1280": ("yolo11l", 1280),
}


def weights(arch: str) -> Path:
    return PRETRAINED / f"{arch}.pt"
