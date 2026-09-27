"""Night-against-night agreement over this project's solves (star_calibration.cross_night).

    python -m src.figlib.stars.agree                 # the committed solves
    python -m src.figlib.stars.agree --wide          # solve_wide's results
    python -m src.figlib.stars.agree --min-stars 4   # re-solve low, then check
"""
from __future__ import annotations

import functools
import json
import sys
from multiprocessing import Pool

from star_calibration.cross_night import report

from . import solve as S

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--min-stars" in args:
        n = int(args[args.index("--min-stars") + 1])
        seqs = sorted(p.name[len("tracks_"):-4] for p in S.DATA.glob("tracks_*.pkl"))
        with Pool(4) as pool:
            res = list(pool.imap_unordered(functools.partial(S.solve_wide, min_stars=n), seqs))
    else:
        src = S.DATA / ("solve_wide_summary.json" if "--wide" in args else "solve_summary.json")
        res = json.loads(src.read_text())
    report(res, {k: v["t0"] for k, v in S.SEQS.items()}, S.CAMS)
