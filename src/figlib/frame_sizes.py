"""Frame size per FIgLib archive, read from the first JPEG header in each tarball.

The same camera name has recorded two formats: 2048x1536 on the older units (every
2016-2018 fire here, pi-s-mobo-c through 2020) and 3072x2048 since. A lens or pose
measured on one format says nothing certain about the other, so the fisheye lens
(geom.py) and the pose ledger (pose_ledger.py) both key on frame width, and this is
where that width comes from. sequences.json doesn't carry it.

    python -m src.figlib.frame_sizes        # writes data/meta/frame_sizes.json
"""

from __future__ import annotations

import io
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from . import corpus as C

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "meta" / "frame_sizes.json"


def first_frame_size(tgz: Path) -> tuple[int, int] | None:
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if m.name.endswith(".jpg"):
                fh = tf.extractfile(m)
                if fh:
                    return Image.open(io.BytesIO(fh.read())).size
    return None


def build() -> dict[str, list[int]]:
    paths = [p for p in C.tgz_paths(C.CORPORA["all"]) if p.exists()]
    with ProcessPoolExecutor(4) as pool:
        sizes = dict(zip((p.name[:-4] for p in paths), pool.map(first_frame_size, paths)))
    out = {k: list(v) for k, v in sorted(sizes.items()) if v}
    DEST.write_text(json.dumps(out, indent=0) + "\n")
    return out


if __name__ == "__main__":
    from collections import Counter
    out = build()
    print(f"wrote {DEST}: {len(out)} archives, {dict(Counter(f'{w}x{h}' for w, h in out.values()))}")
