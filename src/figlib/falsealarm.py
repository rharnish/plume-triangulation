"""Seconds-to-alert against false alarms per camera-day.

mAP is the wrong scoreboard for this problem. A detector that watches 505 cameras is
asked one question all day -- "is anything burning right now?" -- and it is asked it
1,440 times per camera per day. At that volume the only threshold that matters is the
one an operator can live with, so the honest curve is **how fast a real ignition is
called at a fixed rate of false calls per camera-day**, not average precision over a
shuffled image set.

FIgLib is unusually well suited to measuring this, because every sequence ships ~40
minutes of pre-ignition frames from the identical camera under the identical light.
The negatives are therefore the *hard* ones -- the same haze, cumulus, dust plumes and
sun glint that the positives sit in -- rather than unrelated images from elsewhere.

Three decision rules are swept, in increasing order of how much they exploit:

* `single`   -- one frame over threshold.
* `k-of-m`   -- k of the last m frames over threshold. Trades latency for suppression
                of the isolated frame-scale flicker that dominates the negatives.
* `cross`    -- two sites, within a short window, whose bearings pass close together
                on the ground. Uses geometry rather than confidence, which is the only
                thing that removes a 0.81-confidence cumulus (see geolocate.py).

PROVENANCE: the detector was trained on FIgLib (models/README.md), so absolute rates
here measure memorisation as well as skill and are a labelled reference point, not a
generalisation claim. The *shape* of the tradeoff -- how much a persistence or a
cross-site requirement buys -- is the transferable result.
"""

from __future__ import annotations

import json
import os
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .geom import offset_bearing_deg


def poisson_hi(n: int, cam_days: float) -> float | None:
    """Upper 95% bound on the rate, given `n` alarms in `cam_days` of observation.

    FIgLib gives roughly forty minutes of negatives per sequence, so the whole corpus
    is only a few camera-days. Rates below about one per camera-day are therefore
    resolved by a handful of events and a bare ratio would overstate what was measured.
    The bound is the chi-square form, exact for a Poisson count.
    """
    if not cam_days:
        return None
    from scipy.stats import chi2
    return round(float(chi2.ppf(0.975, 2 * (n + 1)) / 2 / cam_days), 3)

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "meta"
# Which detection pass to score. Overridable so the Core ML variants run through this
# exact pipeline rather than a parallel one -- the point of the quantization study is a
# paired comparison, and a second implementation would be a second source of difference.
YOLO_DIR = Path(os.environ.get("FIGLIB_DETS", ROOT / "out" / "yolo"))
OUT = ROOT / "out"

# Frames within this many seconds before annotated plume appearance are discarded
# rather than counted as negatives. The annotation is a human judgement of when smoke
# became visible, so the minute before it is genuinely ambiguous and scoring a
# detection there as a false alarm would flatter the latency numbers at the expense of
# the false-alarm rate -- the exact trade this whole analysis exists to expose.
GUARD_S = 120

# One physical false alarm should be counted once, not once per frame while it drifts
# through the scene. An operator dismisses an alarm and does not want it back a minute
# later, so alarms inside this window of the previous one are suppressed.
REFRACTORY_S = 600

LATENCY_WINDOW_S = 2400          # FIgLib's post-ignition span
SECONDS_PER_DAY = 86400


def load() -> tuple[list[dict], dict, dict]:
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    cams = json.loads((META / "cams.json").read_text())
    fires = json.loads((META / "fires.json").read_text())
    return fires, seqs, cams


def frame_scores(seq_name: str) -> list[tuple[int, int, float, list]]:
    """(offset, epoch, best confidence, detections) per frame, in time order."""
    path = YOLO_DIR / f"{seq_name.split('#')[0]}.json"
    if not path.exists():
        return []
    recs = sorted(json.loads(path.read_text()), key=lambda r: r["offset"])
    return [(r["offset"], r["epoch"],
             max((d["conf"] for d in r["dets"]), default=0.0), r["dets"])
            for r in recs]


# ---------------------------------------------------------------- decision rules

def alarms_single_camera(scores, tau: float, k: int, m: int) -> list[int]:
    """Offsets at which the k-of-m rule declares an alarm, after refractory dedup."""
    hits, out, last = [], [], None
    for i, (off, _epoch, conf, _d) in enumerate(scores):
        hits.append(conf >= tau)
        if sum(hits[max(0, i - m + 1):i + 1]) < k:
            continue
        if last is not None and off - last < REFRACTORY_S:
            continue
        out.append(off)
        last = off
    return out


# ------------------------------------------------------------------- the sweep

def sweep(taus, k: int = 1, m: int = 1) -> list[dict]:
    fires, seqs, _cams = load()
    rows = []
    for tau in taus:
        fa, neg_seconds = 0, 0.0
        lat, alerted, total = [], 0, 0
        for name, s in seqs.items():
            scores = frame_scores(name)
            if not scores:
                continue
            dt = s["median_dt"] or 60.0

            neg = [f for f in scores if f[0] < -GUARD_S]
            neg_seconds += len(neg) * dt
            fa += len(alarms_single_camera(neg, tau, k, m))

            # Latency runs over the whole sequence in time order, so a pre-ignition
            # flicker can legitimately satisfy part of a persistence requirement --
            # which is what would happen in the field.
            total += 1
            first = next((o for o in alarms_single_camera(scores, tau, k, m)
                          if 0 <= o <= LATENCY_WINDOW_S), None)
            if first is not None:
                alerted += 1
                lat.append(first)

        cam_days = neg_seconds / SECONDS_PER_DAY
        rows.append(dict(
            tau=round(tau, 3), k=k, m=m,
            fa=fa, cam_days=round(cam_days, 2),
            fa_per_cam_day=round(fa / cam_days, 3) if cam_days else None,
            fa_hi95=poisson_hi(fa, cam_days),
            n_seq=total, alerted=alerted,
            recall=round(alerted / total, 3) if total else None,
            median_s=int(np.median(lat)) if lat else None,
            p25_s=int(np.percentile(lat, 25)) if lat else None,
            p75_s=int(np.percentile(lat, 75)) if lat else None))
    return rows


# ------------------------------------------------------- cross-site coincidence

def _ray_gap_km(p0, b0, p1, b1) -> float | None:
    """Closest approach of two forward rays in a local east/north plane, km.

    None when the rays diverge -- the crossing lies behind one of the cameras, which is
    not a fire either camera could be looking at.
    """
    d0 = np.array([math.sin(math.radians(b0)), math.cos(math.radians(b0))])
    d1 = np.array([math.sin(math.radians(b1)), math.cos(math.radians(b1))])
    w = np.array(p0) - np.array(p1)
    a, b, c = d0 @ d0, d0 @ d1, d1 @ d1
    d, e = d0 @ w, d1 @ w
    den = a * c - b * b
    if abs(den) < 1e-9:                      # parallel: no crossing to speak of
        return None
    t, s = (b * e - c * d) / den, (a * e - b * d) / den
    if t < 0 or s < 0:
        return None
    return float(np.linalg.norm((np.array(p0) + t * d0) - (np.array(p1) + s * d1)))


def _enu(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111.32 * math.cos(math.radians(lat0)),
            (lat - lat0) * 111.32)


def cross_site(taus, window_s: int = 180, gap_km: float = 3.0,
               require_cross: bool = True) -> list[dict]:
    """Alarm only when two *sites* agree geometrically inside a short time window.

    With `require_cross=False` the same fires, the same cameras and the same negative
    observation time are scored with a plain any-camera rule. That is the control: the
    single-camera sweep above is over all 189 sequences, so comparing it directly with
    this would confound the decision rule with the choice of fires.
    """
    fires, seqs, cams = load()
    rows = []
    for tau in taus:
        fa, neg_seconds = 0, 0.0
        lat, alerted, total = [], 0, 0
        for fire in fires:
            members = [seqs[n] for n in fire["sequences"]
                       if n in seqs and seqs[n]["has_pose"]]
            if len({s["site"] for s in members}) < 2:
                continue
            total += 1
            lat0 = np.mean([s["lat"] for s in members])
            lon0 = np.mean([s["lon"] for s in members])

            # (epoch, site, origin, bearing) for every detection over threshold
            events: list[tuple[int, str, tuple, float, int]] = []
            for s in members:
                cam = cams[s["camera"]]
                p = _enu(cam["lat"], cam["lon"], lat0, lon0)
                dt = s["median_dt"] or 60.0
                n_neg = 0
                for off, epoch, _conf, dets in frame_scores(s["seq"]):
                    if off < -GUARD_S:
                        n_neg += 1
                    for d in dets:
                        if d["conf"] < tau:
                            continue
                        x = (d["x0"] + d["x1"]) / 2
                        events.append((epoch, s["site"], p,
                                       offset_bearing_deg(cam, x), off))
                neg_seconds += n_neg * dt
            events.sort()

            fired: list[int] = []            # offsets of coincidences, both sides
            last = None
            for i, (ep, site, p, b, off) in enumerate(events):
                hit = not require_cross
                for ep2, site2, p2, b2, _off2 in events[i + 1:]:
                    if hit or ep2 - ep > window_s:
                        break
                    if site2 == site:
                        continue
                    g = _ray_gap_km(p, b, p2, b2)
                    if g is not None and g <= gap_km:
                        hit = True
                        break
                if not hit:
                    continue
                if last is not None and off - last < REFRACTORY_S:
                    continue
                fired.append(off)
                last = off

            fa += sum(1 for o in fired if o < -GUARD_S)
            first = next((o for o in fired if 0 <= o <= LATENCY_WINDOW_S), None)
            if first is not None:
                alerted += 1
                lat.append(first)

        cam_days = neg_seconds / SECONDS_PER_DAY
        rows.append(dict(
            tau=round(tau, 3),
            rule=(f"cross<={gap_km}km/{window_s}s" if require_cross else "any-camera"),
            fa=fa, cam_days=round(cam_days, 2),
            fa_per_cam_day=round(fa / cam_days, 3) if cam_days else None,
            fa_hi95=poisson_hi(fa, cam_days),
            n_fire=total, alerted=alerted,
            recall=round(alerted / total, 3) if total else None,
            median_s=int(np.median(lat)) if lat else None,
            p25_s=int(np.percentile(lat, 25)) if lat else None,
            p75_s=int(np.percentile(lat, 75)) if lat else None))
    return rows


def figure(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"single": ("#444444", "o", "single frame"),
             "2of3": ("#1f77b4", "s", "2 of 3 frames"),
             "3of5": ("#2ca02c", "^", "3 of 5 frames"),
             "any_site": ("#ff7f0e", "d", "any camera (triangulable fires)"),
             "cross": ("#d62728", "*", "two sites agree <=3 km / 180 s")}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for rule, (c, mk, label) in style.items():
        rows = [r for r in result[rule] if r["fa"] > 0 and r["recall"]]
        x = [r["fa_per_cam_day"] for r in rows]
        axes[0].plot(x, [r["recall"] for r in rows], mk + "-", color=c,
                     label=label, ms=5, lw=1.3)
        axes[1].plot(x, [r["median_s"] for r in rows], mk + "-", color=c,
                     label=label, ms=5, lw=1.3)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per camera-day")
        ax.grid(alpha=0.3, which="both")
        # Below this the corpus holds too few negatives to resolve a rate at all.
        ax.axvspan(1e-2, 1.0, color="0.85", alpha=0.6, zorder=0)
    axes[0].set_ylabel("fraction of fires alerted within 40 min")
    axes[0].set_title("Detection rate vs false-alarm budget")
    axes[1].set_ylabel("median seconds from plume appearance to alert")
    axes[1].set_title("Latency vs false-alarm budget")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].text(0.02, 0.985, "shaded: the corpus holds under\n5 camera-days of negatives,\n"
                 "so rates here rest on 0-3 events",
                 transform=axes[0].transAxes, fontsize=7.5, color="0.3", va="top")
    fig.text(0.5, 0.015,
             "pyronear yolo11s. The detector was trained on FIgLib, so absolute rates "
             "are a labelled reference point, not a generalisation claim.",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    dest = OUT / "figures" / "falsealarm.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    print(f"\nwrote {dest}")


def main() -> None:
    taus = [round(t, 3) for t in
            [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]]
    result = {
        "guard_s": GUARD_S, "refractory_s": REFRACTORY_S,
        "single": sweep(taus, k=1, m=1),
        "2of3": sweep(taus, k=2, m=3),
        "3of5": sweep(taus, k=3, m=5),
        "any_site": cross_site(taus, require_cross=False),
        "cross": cross_site(taus),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "falsealarm.json").write_text(json.dumps(result, indent=1) + "\n")
    figure(result)
    for rule in ("single", "2of3", "3of5", "any_site", "cross"):
        print(f"\n== {rule}")
        print(f"  observed over {result[rule][0]['cam_days']} camera-days of negatives")
        print(f"{'tau':>5} {'FA':>4} {'FA/cam-day':>11} {'hi95':>7} "
              f"{'recall':>7} {'median s':>9} {'p75':>6}")
        for r in result[rule]:
            print(f"{r['tau']:>5} {r['fa']:>4} {str(r['fa_per_cam_day']):>11} "
                  f"{str(r['fa_hi95']):>7} "
                  f"{str(r['recall']):>7} {str(r['median_s']):>9} "
                  f"{str(r['p75_s']):>6}")


if __name__ == "__main__":
    main()
