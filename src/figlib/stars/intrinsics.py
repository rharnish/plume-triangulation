"""The FIgLib nights' optical centres, fitted from the FIgLib nights themselves.

star_calibration fits each HPWREN unit's optical centre from all of its solved CDN nights
(star_calibration.intrinsics; hpwren/intrinsics.json), and `intrinsics.lookup` hands the
nearest segment to any date. For a FIgLib night that means a centre from 2026 applied to a
frame from as early as 2019, and a unit swapped in between gets the wrong one: lp-w-mobo-c at
Willow (2020-12-02) borrowed (1.9, 63.1) px from a single 2026 night, which bent its solved
lens to k1 -0.059 and its pitch by 2.4 deg (NOTES.md, 2026-09-30).

So the FIgLib nights get their own series, fitted by the same library code
(`star_calibration.intrinsics.build`) from their own solves, and kept here in
data/meta/figlib_intrinsics.json. The library's rules carry over: a camera's nights are split
where its centre jumps, and a segment with too little to go on (one night under 20 matched
stars) keeps the frame's middle. A night takes the centre of the segment it is in, and nothing
else: a night no segment holds (one that didn't solve) is solved at the frame's middle.

The fit starts from solves and the solves depend on the centre, so `main` alternates: fit,
re-solve every FIgLib sequence through the new centres, fit again, re-solve. Two rounds settle
it (the second moved no centre by more than 6 px, 2026-09-30), and the solves on disk end
consistent with the file.

    python -m src.figlib.stars.intrinsics           # fit, re-solve, x2
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "data" / "meta" / "figlib_intrinsics.json"
ROUNDS = 2

_rows: list[dict] | None = None


def load() -> list[dict]:
    global _rows
    if _rows is None:
        _rows = json.loads(PATH.read_text()) if PATH.exists() else []
    return _rows


def centre(seq: str, camera: str, W: int, H: int, rows: list[dict] | None = None) -> tuple[float, float]:
    """(cx, cy) of the segment holding this night, at this frame size; (0, 0) if none does."""
    for r in load() if rows is None else rows:
        if r["camera"] == camera and r["W"] == W and r["H"] == H and seq in r["nights"]:
            return r["cx"], r["cy"]
    return 0.0, 0.0


def build(dest: Path = PATH) -> list[dict]:
    """Fit every FIgLib camera's series from the current FIgLib solves, and write it."""
    global _rows
    from star_calibration import intrinsics as I

    from . import solve as S
    results = [json.loads((S.DATA / f"solve_{q}.json").read_text()) for q in S.figlib_seqs()
               if (S.DATA / f"solve_{q}.json").exists()]
    rows = I.build(results, S.night, S.CAMS, dest)
    _rows = rows
    return rows


def main() -> None:
    from . import solve as S
    seqs = S.figlib_seqs()
    for k in range(1, ROUNDS + 1):
        before = {(r["camera"], tuple(r["nights"])): (r["cx"], r["cy"]) for r in load()}
        rows = build()
        moved = [max(abs(r["cx"] - before[key][0]), abs(r["cy"] - before[key][1]))
                 for r in rows if (key := (r["camera"], tuple(r["nights"]))) in before]
        centred = [r for r in rows if r["cx"] or r["cy"]]
        print(f"round {k}: {len(rows)} segments on {len({r['camera'] for r in rows})} cameras, "
              f"{len(centred)} with a centre"
              + (f"; largest move {max(moved):.1f} px" if moved else ""), flush=True)
        S.solve_batch(seqs, quiet=k < ROUNDS)
    print(f"wrote {PATH}")


if __name__ == "__main__":
    main()
