"""Training-free smoke detection by temporal differencing.

Nothing here is learned, which is the point. Every published wildfire detector worth
using -- pyronear's, SmokeyNet -- was trained on FIgLib, so scoring one of them on
FIgLib measures memorisation as much as detection. A detector with no training set
cannot be contaminated, and its numbers can be quoted without an asterisk.

It is also the right tool rather than merely the safe one. These cameras never move,
so the scene is constant and smoke is the thing that *changes*. Appearance models throw
that away and ask "does this patch look like smoke"; differencing asks "what is here
that was not here before", which is a far easier question when the background is fixed.

The work is in rejecting everything else that changes: the sun moves, clouds drift,
vegetation shakes, and cameras are bumped. Three defences, in order of importance:

1. A **rolling median background** over recent frames, which absorbs slow drift while
   staying robust to a single bad frame.
2. **Global illumination normalisation** per frame, so a passing cloud shadow that
   lifts or drops the whole scene does not read as a detection.
3. **Robust per-frame scaling** by median absolute deviation, so the threshold means
   the same thing on a hazy afternoon as on a clear morning.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TGZ_DIR = ROOT / "data" / "tgz"
OUT_DIR = ROOT / "out" / "scores"

WIDTH = 1024          # working resolution; smoke is a large, low-frequency structure
BG_LAG_FRAMES = 20    # background is drawn from ~20 min ago, not 5 min ago
BG_FRAMES = 7         # rolling median depth
MIN_AREA_FRAC = 2e-4  # ignore specks: ~200 px at 1024x768
OPEN_K = 5


@dataclass
class FrameScore:
    epoch: int
    offset: int
    score: float          # peak region strength, in MAD units
    area_frac: float      # fraction of frame occupied by the winning region
    cx: float             # region centroid, fraction across the frame (-> bearing)
    cy: float
    x0: float
    y0: float
    x1: float
    y1: float



def estimate_horizon(bg: np.ndarray) -> float:
    """Row of the terrain silhouette, as a single scalar.

    Sky is bright and smooth; terrain is darker and textured. Per column, the lowest
    row whose local vertical gradient is large marks the skyline; the median across
    columns is a robust stand-in for a horizon that is never perfectly level. Only a
    reference height is needed -- how far a region descends below it is what matters.
    """
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(bg, (0, 0), 2), cv2.CV_32F, 0, 1, ksize=5))
    rows = []
    for x in range(0, bg.shape[1], 8):
        col = gy[:, x]
        thr = col.mean() + 2.0 * col.std()
        idx = np.where(col > thr)[0]
        if idx.size:
            rows.append(idx[0])
    return float(np.median(rows)) if rows else bg.shape[0] * 0.3


def _load_gray(blob: bytes) -> np.ndarray:
    g = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError("undecodable frame")
    h = int(round(WIDTH * g.shape[0] / g.shape[1]))
    return cv2.resize(g, (WIDTH, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def score_sequence(tgz: Path) -> list[FrameScore]:
    frames: list[tuple[int, int, bytes]] = []
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            name = Path(m.name).name
            if not name.endswith(".jpg") or "_" not in name:
                continue
            stem = name[:-4]
            epoch_s, off_s = stem.split("_")
            fh = tf.extractfile(m)
            if fh:
                frames.append((int(epoch_s), int(off_s), fh.read()))
    frames.sort()

    out: list[FrameScore] = []
    history: list[np.ndarray] = []
    horizon_y = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))

    for epoch, offset, blob in frames:
        try:
            g = _load_gray(blob)
        except ValueError:
            continue

        if len(history) < BG_LAG_FRAMES:
            history.append(g)
            continue
        if horizon_y is None:
            horizon_y = estimate_horizon(np.median(np.stack(history), axis=0))

        bg = np.median(np.stack(history[:BG_FRAMES]), axis=0)

        # Illumination normalisation: match this frame's level to the background's
        # before differencing, so a global brightness shift cancels instead of firing.
        g_adj = g - (np.median(g) - np.median(bg))

        # Smoke is darker than bright hazy sky and brighter than dark terrain, so its
        # sign flips with what lies behind it. Take the magnitude, not the sign.
        diff = np.abs(g_adj - bg)
        mad = float(np.median(np.abs(diff - np.median(diff)))) or 1.0
        z = (diff - np.median(diff)) / (1.4826 * mad)

        mask = ((z > 4.0) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        px = g.shape[0] * g.shape[1]
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA_FRAC * px:
                continue
            y1 = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
            cols = np.where(mask[:, stats[i, cv2.CC_STAT_LEFT]:
                                 stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]].any(0))[0]
            # A plume is rooted: its lowest extent reaches the terrain the fire sits on.
            # Cumulus along the same skyline floats clear of it, so how far the region
            # descends below the horizon separates the two without a cloud classifier.
            rootedness = float(np.clip((y1 - horizon_y) / (0.05 * g.shape[0]), -1.0, 1.0))
            strength = (float(z[labels == i].mean()) * float(np.log1p(area))
                        * (0.15 + 0.85 * max(0.0, rootedness)))
            if best is None or strength > best[0]:
                best = (strength, i, area)

        if best is None:
            out.append(FrameScore(epoch, offset, 0.0, 0.0, 0.5, 0.5, 0, 0, 0, 0))
        else:
            strength, i, area = best
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            H, W = g.shape
            out.append(FrameScore(
                epoch, offset, round(strength, 3), round(area / px, 6),
                round(float(cents[i][0]) / W, 5), round(float(cents[i][1]) / H, 5),
                round(x / W, 5), round(y / H, 5),
                round((x + w) / W, 5), round((y + h) / H, 5)))

        history.append(g)
        history.pop(0)
    return out


def main(limit: int | None = None) -> None:
    import sys
    from dataclasses import asdict
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(TGZ_DIR.glob("*.tgz"))[:limit]
    for k, p in enumerate(paths, 1):
        dest = OUT_DIR / f"{p.name[:-4]}.json"
        if dest.exists():
            continue
        try:
            scores = score_sequence(p)
        except Exception as exc:
            print(f"  FAIL {p.name}: {exc}", file=sys.stderr)
            continue
        dest.write_text(json.dumps([asdict(s) for s in scores]) + "\n")
        print(f"[{k}/{len(paths)}] {p.name[:-4]}: {len(scores)} scored", flush=True)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
