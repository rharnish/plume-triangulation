"""Compare geolocation outputs across calibration variants, fire by fire.

Each variant is a file geolocate.py writes: the baseline `geolocation.json`, and beside it
`geolocation_fisheye.json` (FIGLIB_LENS=fisheye), `geolocation_fisheye_ledger.json` (plus
FIGLIB_POSE_LEDGER=1: star azimuths) and `geolocation_fisheye_ledger_full.json` (plus
FIGLIB_POSE_FULL=1: the whole solved camera; FIGLIB_PROFILE=calibrated). Same detections
and solver in all of them, so differences are the calibration alone.

    python -m src.figlib.compare_geolocation [variant ...]    # default: all four, center bearing
"""

from __future__ import annotations

import json
import statistics
import sys

from . import corpus as C

VARIANTS = ["", "_fisheye", "_fisheye_ledger", "_fisheye_ledger_full"]


def load(variant: str) -> dict[str, dict] | None:
    p = C.current().out / f"geolocation{C.tier_suffix(C.tier_filter())}{variant}.json"
    return {r["fire_id"]: r for r in json.loads(p.read_text())} if p.exists() else None


def main(argv: list[str]) -> None:
    tag = "center"
    variants = argv or VARIANTS
    runs = {v: load(v) for v in variants}
    missing = [v or "(baseline)" for v, r in runs.items() if r is None]
    if missing:
        print("missing:", ", ".join(missing))
    runs = {v: r for v, r in runs.items() if r is not None}
    names = [v.strip("_") or "baseline" for v in runs]
    fires = sorted(set.intersection(*(set(r) for r in runs.values())))

    def err(r, fid):
        x = r[fid].get(tag, {})
        return x.get("error_km") if x.get("status") == "solved" else None

    print(f"[{tag} bearing]  median km (fires within 2 km)")
    print(f"  {'tier':9s} {'n':>2s}  " + "  ".join(f"{n:>20s}" for n in names))
    for tier in ("confirmed", "probable", "all"):
        ids = [f for f in fires if tier in (next(iter(runs.values()))[f]["tier"], "all")
               and all(err(r, f) is not None for r in runs.values())]
        if not ids:
            continue
        cells = []
        for r in runs.values():
            e = [err(r, f) for f in ids]
            cells.append(f"{statistics.median(e):5.2f} ({sum(x <= 2 for x in e):2d} <= 2 km)")
        print(f"  {tier:9s} {len(ids):2d}  " + "  ".join(f"{c:>20s}" for c in cells))

    last = list(runs.values())[-1]
    print(f"\nper fire, {tag}: " + " -> ".join(names) + "   [ledger corrections in the last variant]")
    base = list(runs.values())[0]
    for f in sorted(fires, key=lambda f: err(base, f) or 1e9):
        es = [err(r, f) for r in runs.values()]
        corr = [f"{b['camera']} {b['pose']['d_az']:+.2f} ({b['pose']['rule']})"
                for b in last[f].get(tag, {}).get("bearings", []) if b.get("pose")]
        cells = " -> ".join("  --  " if e is None else f"{e:6.2f}" for e in es)
        print(f"  {cells}  {f} ({base[f]['tier']})  {', '.join(corr)}")


if __name__ == "__main__":
    main(sys.argv[1:])
