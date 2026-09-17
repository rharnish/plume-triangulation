"""pyro-sdis parquet shards -> the directory layout ultralytics trains from.

    data/sdis/parquet/*.parquet         fetched by data/fetch_sdis.sh, pinned revision
    data/sdis/yolo/images/{train,val}/  JPEG bytes written as-is, never re-encoded
    data/sdis/yolo/labels/{train,val}/  one YOLO .txt per image, empty for negatives
    data/sdis/yolo/data.yaml
    data/sdis/manifest.json             shard hashes, counts, revision

Upstream labels all carry class id 1 (33,636 images, 32,109 boxes, no other id). They
are rewritten as class 0 with nc=1, so the files are valid on their own rather than only
under single_cls=True -- which train.py still passes, matching pyronear's recipe.

    .venv-train/bin/python -m src.figlib.scale.sdis
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from ..provenance import sha256_file, utc_now

ROOT = Path(__file__).resolve().parents[3]
SDIS = ROOT / "data" / "sdis"
PARQUET = SDIS / "parquet"
YOLO = SDIS / "yolo"
MANIFEST = SDIS / "manifest.json"

REPO = "pyronear/pyro-sdis"
REVISION = "a1e553ec4d806f71fc6db744cc22bc3469487382"
# LFS oids Hugging Face lists for REVISION; a local shard that disagrees is refused.
SHARDS = {
    "train-00000-of-00006.parquet": "7cebaeff1d94d23513a59a5652ce04303d02297dd6126dbf9d4ea0d37d7eabae",
    "train-00001-of-00006.parquet": "feb69d99b41e42cf9d73582ef2ff4237c7bbf5ad3f733a5ca0418e00795bd5b5",
    "train-00002-of-00006.parquet": "79ac3a22306cdafad474b4f56288b254d400e30ef1b2ed7d1e961cefd32e6858",
    "train-00003-of-00006.parquet": "ce112887c5eb1a44fe4e5bae6843714a9574c89fa5d5215580b1e2ed4b64822e",
    "train-00004-of-00006.parquet": "1352d4d6b2443019fac83967b463bd0daab9030987363f26bae4ba8691653884",
    "train-00005-of-00006.parquet": "05c000f07253dd60858a629119a5a2d94c8f8427b54f6f2fd503bbe06f933065",
    "val-00000-of-00001.parquet": "d3929f81f755625e35c8b18f27f95cff8726bc279cedaf1c3d048b0fd35d6c57",
}


def relabel(annotations: str) -> str:
    """Upstream 'cls xc yc w h' lines -> the same boxes as class 0."""
    out = []
    for line in annotations.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"not a YOLO box line: {line!r}")
        xc, yc, w, h = (float(v) for v in parts[1:])
        out.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def build() -> Path:
    hashes = {}
    for name, want in SHARDS.items():
        got = sha256_file(PARQUET / name)
        if got != want:
            raise SystemExit(f"{name}: sha256 {got} != {want} (revision {REVISION})")
        hashes[name] = got

    counts: dict[str, Counter] = {}
    for name in SHARDS:
        split = name.split("-", 1)[0]
        c = counts.setdefault(split, Counter())
        (YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)
        t = pq.read_table(PARQUET / name, columns=["image", "annotations", "image_name"])
        for img, ann, fname in zip(t["image"].to_pylist(), t["annotations"].to_pylist(),
                                   t["image_name"].to_pylist()):
            stem = Path(fname).stem
            img_path = YOLO / "images" / split / f"{stem}.jpg"
            if img_path.exists():
                raise SystemExit(f"duplicate image name {fname} in {split}")
            img_path.write_bytes(img["bytes"])
            label = relabel(ann)
            (YOLO / "labels" / split / f"{stem}.txt").write_text(label)
            c["images"] += 1
            c["boxes"] += label.count("\n")
            c["negatives"] += not label
        print(f"{name}: {c['images']} {split} images so far",
              file=sys.stderr)

    (YOLO / "data.yaml").write_text(
        f"path: {YOLO}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['smoke']\n")
    MANIFEST.write_text(json.dumps({
        "source": f"https://huggingface.co/datasets/{REPO}",
        "revision": REVISION,
        "shards_sha256": hashes,
        "counts": {k: dict(v) for k, v in counts.items()},
        "built_utc": utc_now(),
    }, indent=1, sort_keys=True) + "\n")
    return MANIFEST


if __name__ == "__main__":
    if (YOLO / "images").exists():
        raise SystemExit(f"{YOLO} already exists; delete it to rebuild")
    print(build())
