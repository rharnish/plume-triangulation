"""Does a star solve actually need a moonless night?

`nights.py` only ever fetched dark blocks, on the reasoning that moonlight washes out the
star field. That was assumed, never measured, and it is an expensive assumption: it throws
away three weeks in four, and with the CDN keeping only ~89 days it is most of the supply.

The test holds everything else fixed and moves only the moon. Three cameras that solve
cleanly on a dark night (hp-w, vo-e, cp-w -- 41, 33 and 34 stars on 2026-09-11) are re-solved
on a ladder of nights from full moon high overhead down to new, all the same Q1 block, all
within two weeks of each other so the season and the star field barely move. Each night gets
the moon's illuminated fraction and elevation from `moon.py`, and `moon_light` = illuminated
fraction times sin(elevation), which is 0 whenever the moon is down.

Reported per night: whether it solved, how many stars, the residual, the pose against the
dark-night pose for that same camera, and the pole-fit quality from `pole.py` -- the last
being the useful one, because it is continuous and defined even when the solve fails.

    python -m src.figlib.stars.moon_test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import moon as M
from . import pole as POLE
from . import solve as S
from .fisheye import initial_k

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out" / "sky" / "moon_exp"
DAYS = ["20260829", "20260831", "20260901", "20260902", "20260903", "20260904",
        "20260905", "20260911"]
CAMS = ["hp-w-mobo-c", "vo-e-mobo-c", "cp-w-mobo-c"]


def run() -> list[dict]:
    rows = []
    for cam in CAMS:
        for day in DAYS:
            seq = f"hpwren_{day}_Q1_{cam}"
            if seq not in S.SEQS:
                continue
            s = S.SEQS[seq]
            c = S.CAMS[cam]
            cond = M.conditions(s["t0"], c["lat"], c["lon"])
            # how far the moon sits from where the camera is looking, which decides whether
            # its glare is in frame or behind the housing
            cond["moon_off_axis"] = abs((cond["moon_az"] - c["az"] + 180) % 360 - 180)
            r = S.solve(seq)
            raw, (W, H) = S.load_tracks(seq)
            tracks = [raw[i] for i in S.prune(raw)]
            f = POLE.estimate(tracks, W, H, S.K_RATIO * initial_k(c, W), S.K1) if len(tracks) >= 8 else None
            rows.append({
                "seq": seq, "camera": cam, "day": day, **cond,
                "n_tracks_raw": r.get("n_tracks_raw"), "n_tracks": r.get("n_tracks"),
                "status": r["status"], "reason": r.get("reason"),
                "n_stars": r.get("n_stars"), "median_px": r.get("median_px"),
                "pose": r.get("pose"),
                "pole_norm": f["norm"] if f else None,
                "pole_inlier_frac": f["inlier_frac"] if f else None})
            print(f"{seq:40s} illum {cond['illum']:.2f} el {cond['moon_el']:+5.1f} "
                  f"light {cond['moon_light']:.3f}  {r['status']:7s} "
                  f"stars {str(r.get('n_stars')):>4s} tracks {r.get('n_tracks')}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "moon_ladder.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    return rows


def report(rows: list[dict]) -> None:
    print(f"\n{'camera':14s} {'day':9s} {'illum':>6s} {'el':>6s} {'light':>6s} {'off-ax':>7s} "
          f"{'tracks':>7s} {'stars':>6s} {'med px':>7s} {'|p|':>6s} {'d_az vs dark':>13s}")
    for cam in CAMS:
        g = [r for r in rows if r["camera"] == cam]
        dark = next((r for r in g if r["moon_light"] < 0.01 and r["pose"]), None)
        for r in sorted(g, key=lambda r: -r["moon_light"]):
            d = ""
            if dark and r["pose"]:
                d = f"{r['pose']['d_az'] - dark['pose']['d_az']:+.3f}"
            print(f"{cam:14s} {r['day']:9s} {r['illum']:6.2f} {r['moon_el']:+6.1f} "
                  f"{r['moon_light']:6.3f} {r['moon_off_axis']:7.1f} {r['n_tracks']:7d} "
                  f"{str(r['n_stars'] or '-'):>6s} "
                  f"{(f'{r['median_px']:.2f}' if r['median_px'] else '-'):>7s} "
                  f"{(f'{r['pole_norm']:.3f}' if r['pole_norm'] else '-'):>6s} {d:>13s}")
    ok = [r for r in rows if r["status"] == "solved"]
    print(f"\n{len(ok)}/{len(rows)} solved overall; "
          f"{sum(1 for r in ok if r['moon_light'] > 0.1)}/"
          f"{sum(1 for r in rows if r['moon_light'] > 0.1)} of the moonlit nights solved")


if __name__ == "__main__":
    report(run())
