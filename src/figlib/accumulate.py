"""Accumulate every detection into one posterior, instead of picking the best one.

Taking each camera's most confident detection discards almost everything observed. A
faint 0.28 at three minutes and a solid 0.74 at twelve, pointing the same way, are two
pieces of evidence for the same bearing; keeping only the second throws away the first
and, worse, throws away the *agreement* between them, which is the part that says the
bearing is real rather than a passing cloud.

Summing log-likelihoods over all detections is not enough either, because a false
positive at the wrong bearing then contributes an unbounded penalty and can drag the
peak anywhere -- and these detectors do produce confident false positives, 0.81 on a
cumulus in one case. So each detection is modeled as a mixture:

    P(detection | fire at x) = pi * Normal(bearing | bearing_to_x, sigma)
                             + (1 - pi) * Uniform(field of view)

with `pi` rising with detector confidence. A detection that agrees with the hypothesis
contributes real evidence; one that disagrees falls back on the uniform term and its
influence is *bounded* -- it stops mattering rather than dominating. This is standard
robust estimation, and it is what lets weak-but-consistent evidence outvote a strong
outlier.

Frames are not independent: a plume visible at t is visible at t+60, so a camera
detecting in forty frames has not made forty independent measurements. Per-camera
evidence is therefore scaled by n**alpha / n, i.e. it grows as n**alpha rather than n.
alpha = 0 averages (one camera, one vote), alpha = 1 sums (full independence, badly
overconfident). alpha is chosen by measurement below, not assumed.
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path

import numpy as np

from .geolocate import credible_area_km2
from .geom import bearing_grid, haversine_km, load_cams, offset_bearing_deg

from . import corpus as C

ROOT = Path(__file__).resolve().parents[2]
META = C.current().meta
# Which detection pass to score. Overridable so the Core ML variants run through this
# exact pipeline rather than a parallel one -- the point of the quantization study is a
# paired comparison, and a second implementation would be a second source of difference.
YOLO_DIR = Path(os.environ.get("FIGLIB_DETS", C.current().dets))

SIGMA_DEG = 2.0
CONF_FLOOR = 0.10       # below this a detection is treated as pure noise
PI_MAX = 0.90           # never fully trust a single detection


def pi_of(conf: float) -> float:
    """Probability a detection is a real observation, from its confidence.

    The detector is not calibrated, so this is deliberately a blunt monotone map rather
    than a fitted curve -- claiming a calibrated probability we have not measured would
    be worse than admitting a rough one.
    """
    return float(np.clip(conf, 0.0, 1.0) * PI_MAX)


def gather(fire: dict, seqs: dict, cams: dict, t_max: int,
           conf_thr: float = CONF_FLOOR) -> dict[str, list[tuple[float, float]]]:
    """Per camera, every (bearing, confidence) seen up to `t_max` after its own t0."""
    out: dict[str, list[tuple[float, float]]] = {}
    for seq_name in fire["sequences"]:
        s = seqs.get(seq_name)
        if not s or not s["has_pose"]:
            continue
        path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
        if not path.exists():
            continue
        cam = cams[s["camera"]]
        acc = out.setdefault(s["camera"], [])
        for rec in json.loads(path.read_text()):
            if not (0 <= rec["offset"] <= t_max):
                continue
            for d in rec["dets"]:
                if d["conf"] < conf_thr:
                    continue
                x = (d["x0"] + d["x1"]) / 2
                acc.append((offset_bearing_deg(cam, x), d["conf"]))
    return {k: v for k, v in out.items() if v}


def posterior(det_by_cam: dict, cams: dict, center: tuple[float, float],
              half_extent_km: float = 60.0, step_km: float = 0.4,
              sigma_deg: float = SIGMA_DEG, alpha: float = 0.5):
    lat0, lon0 = center
    dlat = step_km / 111.32
    dlon = step_km / (111.32 * math.cos(math.radians(lat0)))
    n = int(half_extent_km / step_km)
    lats = lat0 + np.arange(-n, n + 1) * dlat
    lons = lon0 + np.arange(-n, n + 1) * dlon
    LA, LO = np.meshgrid(lats, lons, indexing="ij")

    total = np.zeros_like(LA)
    norm = 1.0 / (sigma_deg * math.sqrt(2 * math.pi))

    for camera, dets in det_by_cam.items():
        cam = cams[camera]
        brg = bearing_grid(cam["lat"], cam["lon"], LA, LO)

        uniform = 1.0 / cam["fov"]
        cam_ll = np.zeros_like(LA)
        for b, conf in dets:
            d = (brg - b + 180.0) % 360.0 - 180.0
            pi = pi_of(conf)
            like = pi * norm * np.exp(-0.5 * (d / sigma_deg) ** 2) + (1 - pi) * uniform
            cam_ll += np.log(like)
        m = len(dets)
        total += cam_ll * (m ** alpha / m)      # temporal correlation discount

    i, j = np.unravel_index(np.argmax(total), total.shape)
    return lats, lons, total, float(lats[i]), float(lons[j])


def main() -> None:
    cams = load_cams()
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = json.loads((META / "resolved.json").read_text())

    scoring = [r for r in resolved
               if r["tier"] in ("confirmed", "probable") and r.get("triangulable")]

    print("alpha sweep (temporal-correlation discount), all detections to t=2400 s")
    print(f"{'alpha':>6} {'n':>4} {'median km':>10} {'p75':>8} {'<=2km':>6} {'<=5km':>6}")
    best = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        errs = []
        for r in scoring:
            fire, t = fires[r["fire_id"]], r["truth"]
            dets = gather(fire, seqs, cams, 2400)
            if len({c.split("-")[0] for c in dets}) < 2:
                continue
            c = (float(np.mean([cams[c]["lat"] for c in dets])),
                 float(np.mean([cams[c]["lon"] for c in dets])))
            _, _, _, la, lo = posterior(dets, cams, c, alpha=alpha)
            errs.append(haversine_km(la, lo, t["lat"], t["lon"]))
        e = sorted(errs)
        med = e[len(e) // 2]
        print(f"{alpha:>6.2f} {len(e):>4} {med:>10.2f} {e[3*len(e)//4]:>8.2f} "
              f"{sum(x<=2 for x in e):>6} {sum(x<=5 for x in e):>6}")
        if best is None or med < best[1]:
            best = (alpha, med)
    print(f"\nbest alpha = {best[0]} at {best[1]:.2f} km median")


if __name__ == "__main__":
    main()
