"""Was a camera re-aimed between a fire and the star solve that would correct it?

The pose ledger carries a solve up to MAX_DAYS to one side of a fire, on the assumption that
the camera did not move in between. Two solves on either side catch a re-aim; one solve alone
cannot. mlo-s-mobo-c solved at -23.3 deg on 2026-07-21, 362 days after the Posta fire,
but its Posta and Kitchen bearings agree with the official points under the published azimuth.
Its view had shifted ~19 deg in between (NOTES.md, 2026-10-01).

This puts two daytime frames side by side: the FIgLib sequence's first frame (pre-ignition),
and a CDN frame from the solve's date at the same time of day, so the light falls the same way.
It estimates the horizontal shift between them (ORB features, median dx of the matches that
agree on one translation) and converts it to degrees near the frame's middle. Then it writes
both frames and the estimate where `reaim_viewer.html` overlays them, with an opacity slider and
a shift slider to align them by eye.

    python -m src.figlib.stars.reaim                                # the default pairs
    python -m src.figlib.stars.reaim mlo-s-mobo-c 20250724_PostaFire 20260721
    cd out/reaim && python3 -m http.server 8010   # then http://localhost:8010/viewer.html

The CDN keeps ~90 days public: a solve date older than that has no daytime frame to fetch.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from star_calibration.fisheye import K_RATIO, initial_k
from star_calibration.hpwren import cameras
from star_calibration.hpwren.nights import CDN, _get

from .. import corpus as C
from ..detect_yolo import read_frames

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "reaim"
WIDTH = 1536          # both frames are saved at this width
LOCAL = ZoneInfo("America/Los_Angeles")

# (camera, FIgLib fire, CDN date of the solve that would be applied to it): the CDN solves of
# 2026-10-01 for confirmed fires with no star pose, plus tp-w, whose re-aim is known.
PAIRS = [
    ("mlo-s-mobo-c", "20250724_PostaFire", "20260721"),
    ("ws-w-mobo-c", "20250724_PostaFire", "20260721"),
    ("starr-n-mobo-c", "20250823_Clubfire", "20260820"),
    ("starr-n-mobo-m", "20250823_Clubfire", "20260820"),
    ("mpo-w-mobo-c", "20260319_CrosleyFire", "20260704"),
    ("mp-n-mobo-m", "20251102_ScissorsFire", "20260703"),
    ("tp-w-mobo-c", "20240724_GroveFire", "20260912"),
]


def fire_frame(camera: str, fire: str) -> tuple[np.ndarray, int]:
    """The FIgLib sequence's first frame (40 min before ignition) and its epoch."""
    arch = {p.name[:-4]: p for p in C.tgz_paths(C.CORPORA["all"])}
    name = next(a for a in arch if a.startswith(fire + "_") and a.endswith(camera))
    epoch, _off, blob = min(read_frames(arch[name]), key=lambda f: f[1])
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR), epoch


def cdn_frame(camera: str, day: str, like_epoch: int) -> tuple[np.ndarray, int]:
    """The CDN frame on `day` nearest the same local time of day as `like_epoch`."""
    t = datetime.fromtimestamp(like_epoch, LOCAL)
    q = t.hour // 3 + 1
    listing = _get(f"{CDN}/hpwren-cameras/{camera}/{day[:4]}/{day}/{day}_{camera}_Q{q}.txt")
    names = [n.strip() for n in listing.decode().split() if n.strip().endswith(".jpg")]
    want = datetime(int(day[:4]), int(day[4:6]), int(day[6:]), t.hour, t.minute, tzinfo=LOCAL).timestamp()
    name = min(names, key=lambda n: abs(int(n.split(".")[0]) - want))
    blob = _get(f"{CDN}/MTA/{camera}/large/{day}/Q{q}/{name}")
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR), int(name.split(".")[0])


def scene_mask(shape) -> np.ndarray:
    """Where the scene is: inside the lens circle, below the banner and logos, clear of the
    HPWREN watermark. Those overlays sit still in every frame, so left in, they win the vote
    for zero shift however far the camera turned."""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (W // 2, H // 2), (int(0.46 * W), int(0.56 * H)), 0, 0, 360, 255, -1)
    m[: int(0.09 * H)] = 0
    m[int(0.90 * H):, int(0.80 * W):] = 0
    return m


def shift(a: np.ndarray, b: np.ndarray) -> dict:
    """Horizontal and vertical shift of b against a (pixels at WIDTH): the median
    displacement of ORB matches, kept only if most of them agree on it."""
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(ga, scene_mask(ga.shape))
    kb, db = orb.detectAndCompute(gb, scene_mask(gb.shape))
    if da is None or db is None:
        return {"dx": None, "dy": None, "n": 0, "agree": 0.0}
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    d = np.array([np.subtract(kb[x.trainIdx].pt, ka[x.queryIdx].pt) for x in m])
    if len(d) < 10:
        return {"dx": None, "dy": None, "n": len(d), "agree": 0.0}
    # the translation most matches agree on, by voting on a coarse grid, then its median
    v = np.round(d / 8).astype(int)
    keys, counts = np.unique(v, axis=0, return_counts=True)
    top = keys[np.argmax(counts)]
    near = np.all(np.abs(v - top) <= 1, axis=1)
    dx, dy = np.median(d[near], axis=0)
    return {"dx": round(float(dx), 1), "dy": round(float(dy), 1), "n": len(d),
            "agree": round(float(near.mean()), 2)}


def build(pairs=PAIRS) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    cams = cameras()
    rows = []
    for camera, fire, day in pairs:
        pid = f"{camera}__{fire}__{day}"
        try:
            a, ea = fire_frame(camera, fire)
            b, eb = cdn_frame(camera, day, ea)
        except Exception as exc:   # an expired CDN date (403), an offline camera, no archive
            print(f"{pid}: skipped ({type(exc).__name__}: {exc})")
            continue
        a, b = (cv2.resize(x, (WIDTH, round(x.shape[0] * WIDTH / x.shape[1])), interpolation=cv2.INTER_AREA)
                for x in (a, b))
        for tag, img in (("a", a), ("b", b)):
            cv2.imwrite(str(OUT / f"{pid}_{tag}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # px per degree near the middle, at WIDTH, through the shared star-measured lens
        px_per_deg = K_RATIO * initial_k(cams[camera], WIDTH) * np.pi / 180
        s = shift(a, b)
        when = lambda e: datetime.fromtimestamp(e, LOCAL).strftime("%Y-%m-%d %H:%M %Z")
        rows.append({"id": pid, "camera": camera, "fire": fire, "solve_day": day,
                     "a": {"src": f"{pid}_a.jpg", "label": f"{fire} (fire day)", "time": when(ea)},
                     "b": {"src": f"{pid}_b.jpg", "label": f"CDN {day} (solve day)", "time": when(eb)},
                     "width": WIDTH, "height": int(a.shape[0]), "px_per_deg": round(float(px_per_deg), 3),
                     "est": {**s, "deg": None if s["dx"] is None else round(s["dx"] / px_per_deg, 2)}})
        e = rows[-1]["est"]
        print(f"{pid}: shift {e['dx']} px = {e['deg']} deg  ({e['n']} matches, {e['agree']:.0%} agree)")
    (OUT / "pairs.json").write_text(json.dumps(rows, indent=1) + "\n")
    shutil.copy(Path(__file__).with_name("reaim_viewer.html"), OUT / "viewer.html")
    print(f"\nviewer: cd {OUT} && python3 -m http.server 8010   then open "
          f"http://localhost:8010/viewer.html")
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    build([tuple(args[i:i + 3]) for i in range(0, len(args), 3)] if args else PAIRS)
