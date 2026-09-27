"""Star solves by sequence name: FIgLib night sequences and HPWREN CDN blocks, one namespace.

The solver is the star_calibration library (`star_calibration.solve`), which takes a `Night`
-- tracks, frame size, camera, t0 -- and reads no files. This module is the part that knows
this project's data: which sequences exist (FIgLib's index plus the CDN blocks in the shared
HPWREN cache), where their frames come from (stars.tracks), and where tracks and results are
cached (out/sky/data/star_tracks). Figures and experiments call it by sequence name.

    python -m src.figlib.stars.solve [seq ...]    # every cached sequence if none named
"""
from __future__ import annotations

import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

from star_calibration import solve as L
from star_calibration.fisheye import K1, K_RATIO  # noqa: F401  (re-exported for figures)
from star_calibration.hpwren import cameras
from star_calibration.hpwren import nights as hpwren_nights
from star_calibration.solve import (MAX_TRACKS, Night, _spread, clip,  # noqa: F401
                                    make_coincidence, prune)

ROOT = Path(__file__).resolve().parents[3]
SKY = ROOT / "out" / "sky"
CAMS = cameras()
SEQS = {s["seq"]: s for s in json.loads((ROOT / "data/meta/all/sequences.json").read_text())}
SEQS.update(hpwren_nights.sequences())
DATA = SKY / "data/star_tracks"


def load_tracks(seq: str):
    cache = DATA / f"tracks_{seq}.pkl"
    if cache.exists():
        d = pickle.load(open(cache, "rb"))
        return d["tracks"], tuple(d.get("WH", (3072, 2048)))
    from .tracks import collect   # imported lazily: it decodes whole archives
    # A whole-night directory (nights.fetch_night) is ~470 frames: keep only one decoded, and
    # close tracks lost for 5 min so a star behind cloud can't be relinked hours later.
    night = "_N_" in seq
    tracks, decoded, _ = collect(seq, keep_frames=not night, max_gap_s=300.0 if night else None)
    W, H = 3072, 2048
    if decoded:
        H, W = next(iter(decoded.values()))[1].shape[:2]
    pickle.dump({"tracks": tracks, "WH": (W, H)}, open(cache, "wb"))
    return tracks, (W, H)


def night(seq: str) -> Night:
    """A sequence as the library's solver input."""
    s = SEQS[seq]
    tracks, (W, H) = load_tracks(seq)
    return Night(camera=s["camera"], cam=CAMS[s["camera"]], t0=s["t0"], tracks=tracks,
                 W=W, H=H, label=seq)


def windowed(seq: str, window: tuple[float, float] | None = None):
    """(raw, keep, (W, H)): the cached tracks clipped to `window` (star_calibration.solve.windowed)."""
    tracks, WH = load_tracks(seq)
    return (*L.windowed(tracks, window), WH)


def scan_context(seq: str, k_ratio: float | None = None, wide: bool = True,
                 window: tuple[float, float] | None = None):
    return L.scan_context(night(seq), k_ratio=k_ratio, wide=wide, window=window)


def solve(seq: str, wide: bool = False, min_stars: int = 8,
          k_ratio: float | None = None, window: tuple[float, float] | None = None) -> dict:
    return L.solve(night(seq), wide=wide, min_stars=min_stars, k_ratio=k_ratio, window=window)


def solve_wide(seq: str, min_stars: int = 8, window: tuple[float, float] | None = None) -> dict:
    return L.solve_wide(night(seq), min_stars=min_stars, window=window)


def _run(seq: str) -> dict:
    try:
        r = solve(seq)
    except Exception as exc:   # one bad archive shouldn't sink the batch
        r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    (DATA / f"solve_{seq}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    return r


if __name__ == "__main__":
    # whole-night blocks (<day>_N) belong to stars.window_ablation, not the batch or the ledger
    seqs = sys.argv[1:] or sorted(p.name[len("tracks_"):-4] for p in DATA.glob("tracks_*.pkl")
                                  if "_N_" not in p.name)
    with Pool(4) as pool:
        results = list(pool.imap_unordered(_run, seqs))
    results.sort(key=lambda r: (r["status"] != "solved", r["seq"]))
    for r in results:
        if r["status"] == "solved":
            p = r["pose"]
            print(f"SOLVED {r['seq']:55s} {r['n_stars']:2d} stars  med {r['median_px']:.2f}px  "
                  f"d_az {p['d_az']:+6.2f} d_pitch {p['d_pitch']:+6.2f} d_roll {p['d_roll']:+6.2f}  "
                  f"k {p['k_ratio']:.3f}x k1 {p['k1']:+.3f}  runs {r['runs_agreeing']}")
        else:
            print(f"failed {r['seq']:55s} {r.get('n_tracks', '-')}/{r.get('n_tracks_raw', '-')} tracks  "
                  f"{r['reason']}")
    (DATA / "solve_summary.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    print(f"{sum(r['status'] == 'solved' for r in results)}/{len(results)} solved")
