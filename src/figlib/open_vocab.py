"""Open-vocabulary detection: smoke, plus the scene context around it.

`masks.py` already prompts SAM with a *given* box -- the pyronear detector's. This module
gets the box itself from a text prompt instead, via Grounding DINO, and hands it to the
same SAM model for a mask. That buys two things nothing else in the repo does:

**Smoke without a trained detector.** Pyronear's box is FIgLib-trained; Grounding DINO's
is not, so a "smoke" / "wildfire smoke plume" prompt is a second, independent read on
where the plume is -- useful as a sanity check on the trained detector, or as a fallback
where it has no weights at all (a different camera family, a different spectrum).

**Everything that is not smoke.** A fixed camera's frame is mostly context: water,
skyline, the built environment. `terrain.py` derives ridgelines and peaks from the DEM
already, geometrically, from camera pose -- this is the vision-only complement to that,
useful on frames with no pose or no DEM coverage, and as a check on the geometric one.
Water bodies and landmarks (towers, roads, buildings) have no geometric counterpart in
this repo at all; a text prompt is the only way in.

Grounding DINO ("IDEA-Research/grounding-dino-tiny") is Apache-2.0 and not trained on
FIgLib, so it carries the same "adds no contamination" property `masks.py` documents for
SAM. Both models are Swin-T / ViT-B class and run at the same budget already measured for
SAM alone: about 2 frames/second on the GTX 1070 this was built on.

Needs `requirements-masks.txt` -- no new dependency, Grounding DINO ships in the same
`transformers` pin that SAM uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np

from .masks import sam_device, sam_mask

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "openvocab"

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Category -> phrases. Grounding DINO is prompted with every phrase in one pass (joined
# below), and each returned box carries back the phrase that matched it, which is how a
# detection is filed into a category.
PROMPTS: dict[str, list[str]] = {
    "smoke": ["smoke", "wildfire smoke plume"],
    "water": ["lake", "river", "pond"],
    "terrain": ["ridgeline", "mountain peak"],
    "landmark": ["tower", "road", "building"],
}

# BGR, one per category, for the overlay.
COLORS: dict[str, tuple[int, int, int]] = {
    "smoke": (0, 0, 255),
    "water": (255, 128, 0),
    "terrain": (0, 200, 200),
    "landmark": (0, 255, 0),
}

BOX_THR = 0.25
TEXT_THR = 0.20

_GDINO = None


def _grounding_dino():
    """Load Grounding DINO once, onto whichever device `masks.sam_device` picked."""
    global _GDINO
    if _GDINO is None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        dev = sam_device()
        proc = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL).eval().to(dev)
        _GDINO = (model, proc, torch, dev)
    return _GDINO


@dataclass
class Detection:
    category: str
    phrase: str
    score: float
    box_px: tuple[int, int, int, int]     # x0, y0, x1, y1
    mask: np.ndarray | None = None


def detect(img_bgr: np.ndarray, prompts: dict[str, list[str]] = PROMPTS,
           box_thr: float = BOX_THR, text_thr: float = TEXT_THR) -> list[Detection]:
    """Text-prompted boxes, one forward pass per phrase.

    All phrases in one query -- the natural reading of Grounding DINO's period-separated
    prompt -- was tried first and measured worse: with ten short, semantically
    overlapping phrases in one string ("lake", "river", "pond", "road", ...) the text
    encoder blends adjacent spans, scores collapse under 0.11 for everything including
    smoke, and labels come back as multi-phrase mush ("river pond road"). Querying one
    phrase at a time on the same frame put smoke back to 0.57 and tower to 0.33. Ten
    passes over one image is still well inside the SAM-measured ~2 frame/s budget this
    box runs at, since only the text side changes per call.
    """
    model, proc, torch, dev = _grounding_dino()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    dets = []
    for cat, phrases in prompts.items():
        for phrase in phrases:
            inputs = proc(images=rgb, text=phrase.lower() + ".", return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inputs)
            results = proc.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=box_thr, text_threshold=text_thr,
                target_sizes=[(h, w)])[0]
            for box, score in zip(results["boxes"], results["scores"]):
                x0, y0, x1, y1 = (float(v) for v in box)
                dets.append(Detection(cat, phrase, float(score),
                                      (int(x0), int(y0), int(x1), int(y1))))
    return dets


def segment(img_bgr: np.ndarray, dets: list[Detection]) -> list[Detection]:
    """SAM mask for each box, in place. Best-effort: a mask that fails to segment
    keeps its box and is still drawn, just without a fill."""
    for d in dets:
        try:
            d.mask = sam_mask(img_bgr, d.box_px)
        except Exception:
            d.mask = None
    return dets


def draw(img_bgr: np.ndarray, dets: list[Detection]) -> np.ndarray:
    img = img_bgr.copy()
    overlay = img.copy()
    for d in dets:
        color = COLORS.get(d.category, (255, 255, 255))
        if d.mask is not None:
            overlay[d.mask] = color
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for d in dets:
        color = COLORS.get(d.category, (255, 255, 255))
        x0, y0, x1, y1 = d.box_px
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, f"{d.phrase} {d.score:.2f}", (x0, max(18, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def run_image(path: Path, prompts: dict[str, list[str]] = PROMPTS,
              box_thr: float = BOX_THR, text_thr: float = TEXT_THR) -> dict:
    """Detect, segment, and save everything the viewer needs.

    Four files per image, all under `OUT`, so `viewer.html` never has to reach outside
    it: the untouched frame (`.orig.jpg`, since the source path may be a scratch dir
    that will not exist later), the fixed all-on preview (`.jpg`, unchanged from
    before), the per-detection masks as one label image (`.masks.png` -- pixel value
    `i` is detection `i - 1`, `0` is background, so the browser can toggle a mask by
    comparing pixel values instead of fetching one file per detection), and the record
    (`.json`), now carrying each detection's mask index alongside its box.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    dets = segment(img, detect(img, prompts, box_thr, text_thr))
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{path.stem}.orig.jpg"), img)
    cv2.imwrite(str(OUT / f"{path.stem}.jpg"), draw(img, dets))

    label = np.zeros(img.shape[:2], np.uint8)
    detections = []
    for i, d in enumerate(dets, start=1):
        idx = 0
        if d.mask is not None:
            label[d.mask] = i
            idx = i
        detections.append({"category": d.category, "phrase": d.phrase,
                            "score": d.score, "box_px": d.box_px, "mask_index": idx})
    cv2.imwrite(str(OUT / f"{path.stem}.masks.png"), label)

    record = {"image": f"{path.stem}.orig.jpg",
              "width": img.shape[1], "height": img.shape[0],
              "colors": {k: list(v[::-1]) for k, v in COLORS.items()},  # BGR -> RGB
              "detections": detections}
    (OUT / f"{path.stem}.json").write_text(json.dumps(record, indent=2))
    return record


def rebuild_manifest() -> list[str]:
    """List of stems with a saved record, for `viewer.html`'s image picker."""
    stems = sorted(p.stem for p in OUT.glob("*.json"))
    (OUT / "manifest.json").write_text(json.dumps(stems, indent=2))
    return stems


def main(argv: list[str] | None = None) -> None:
    import argparse
    import shutil
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", type=Path,
                     help="frames to run open-vocabulary detection on")
    ap.add_argument("--box-thr", type=float, default=BOX_THR)
    ap.add_argument("--text-thr", type=float, default=TEXT_THR)
    args = ap.parse_args(argv)

    for path in args.images:
        record = run_image(path, box_thr=args.box_thr, text_thr=args.text_thr)
        print(f"{path.name}: {len(record['detections'])} detections -> "
              f"{OUT / (path.stem + '.jpg')}")

    rebuild_manifest()
    viewer_src = Path(__file__).resolve().parent / "open_vocab_viewer.html"
    if viewer_src.exists():
        shutil.copy(viewer_src, OUT / "viewer.html")
    print(f"\nviewer: cd {OUT} && python3 -m http.server 8008   then open "
          f"http://localhost:8008/viewer.html")


if __name__ == "__main__":
    main()
