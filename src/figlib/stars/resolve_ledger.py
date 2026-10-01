"""Rebuild a pose ledger from the same sources under the current solver and sky model.

Each FIgLib sequence behind the ledger is solved again -- same tracks, same solver it was
accepted by (`solve_wide` where the summary records a `found_by`, `solve` otherwise, falling
back to `solve_wide` if that fails). Each CDN block's entry is copied from star_calibration's
shipped ledger (hpwren/pose_ledger.json), which hpwren.calibrate solves from the shared cache;
one the library no longer has is dropped, and named. The set of sources stays the same, so
only the solver changes. The previous ledger is kept beside the results
(pose_ledger_before_<version>.json), and the shift in each part is printed.

History: written to re-solve under proper motion, precession and refraction (NOTES.md,
2026-09-25); since star-calibration v0.3.0 the CDN blocks come from the library, and every
solve carries the optical centre it assumed (`cx`, `cy`).

    python -m src.figlib.stars.resolve_ledger
    python -m src.figlib.stars.resolve_ledger out/recent/pose_ledger.json

Given another ledger file (the `recent` corpus keeps the tracked ledger plus its own solves),
entries the tracked ledger already holds under the current sky model and solver are copied
from it, not solved a second time, so the two files never hold different poses for one night.
Rebuild the tracked ledger first.
"""
from __future__ import annotations

import json
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import pose_ledger
from star_calibration import __version__, solver_id
from star_calibration import catalog as SG
from star_calibration.hpwren import ledger_path
from . import solve as S


def _one(job):
    seq, wide = job
    try:
        r = S.solve_wide(seq) if wide else S.solve(seq)
        # a few accepted solves came from solve_wide before it recorded `found_by`
        # (hpwren_20260911_Q1_vo-e-mobo-m): give a failed grid solve the wide search too
        if r["status"] != "solved" and not wide:
            r = S.solve_wide(seq)
    except Exception as exc:
        r = {"seq": seq, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    (S.DATA / f"solve_{seq}.json").write_text(json.dumps(r, indent=1, default=float) + "\n")
    return r


def main(path: Path = pose_ledger.LEDGER):
    path = Path(path).resolve()
    tracked = path == pose_ledger.LEDGER.resolve()
    old = json.loads(path.read_text())
    backup = S.DATA / (f"pose_ledger_before_{__version__}.json" if tracked
                       else f"{path.parent.name}_{path.stem}_before_{__version__}.json")
    if not backup.exists():
        shutil.copy(path, backup)
    before = {e["source"]: e for e in old}
    # A CDN block's solve is the library's: its shipped ledger, made by hpwren.calibrate from
    # the shared cache. Only the FIgLib sequences are solved here.
    shipped = {e["source"]: e for e in json.loads(ledger_path().read_text()) if current(e)}
    if not tracked:   # the recent ledger: whatever the tracked one holds, as it holds it
        shipped |= {e["source"]: e for e in json.loads(pose_ledger.LEDGER.read_text()) if current(e)}
    cdn = [e for e in old if e["source"].startswith("star:hpwren_")]
    reused = [shipped[e["source"]] for e in old if e["source"] in shipped]
    dropped = [e["source"] for e in cdn if e["source"] not in shipped]
    old = [e for e in old if e["source"] not in shipped and e not in cdn]
    print(f"{len(reused)} entries copied from {ledger_path().name} (star-calibration {__version__})"
          + ("" if tracked else f" and {pose_ledger.LEDGER.name}")
          + (f"; {len(dropped)} CDN solves it no longer has: {dropped}" if dropped else ""))
    summary_path = S.DATA / "solve_summary.json"
    summary = {r["seq"]: r for r in json.loads(summary_path.read_text())}
    seqs = [e["source"].split(":", 1)[1] for e in old]
    jobs = [(q, "found_by" in summary.get(q, {})) for q in seqs]
    print(f"re-solving {len(jobs)} ledger sequences under: {SG.model_id()}, solver {solver_id()}",
          flush=True)
    with Pool(4) as pool:
        new = {r["seq"]: r for r in pool.imap_unordered(_one, jobs)}
    summary.update(new)
    summary_path.write_text(json.dumps(sorted(summary.values(), key=lambda r: (r["status"] != "solved", r["seq"])),
                                       indent=1, default=float) + "\n")

    t0 = {s["seq"]: s["t0"] for s in json.loads((S.ROOT / "data/meta/all/sequences.json").read_text())}
    t0.update({k: v["t0"] for k, v in S.SEQS.items() if k.startswith("hpwren_")})
    rows = pose_ledger.build([new[q] for q in seqs], t0, dest=None)
    allrows = sorted(rows + reused, key=lambda e: (e["camera"], e["epoch"]))
    path.write_text(json.dumps(allrows, indent=1) + "\n")
    lost = [q for q in seqs if new[q]["status"] != "solved"]
    print(f"wrote {path}: {len(rows)} of {len(seqs)} re-solved, plus {len(reused)} copied "
          f"({len(lost)} no longer pass: {lost})")
    for name, part in (("re-solved", rows), ("copied", reused)):
        if not part:
            continue
        d = np.array([[r["d_az"] - before[r["source"]]["d_az"], r["d_pitch"] - before[r["source"]]["d_pitch"],
                       r["d_roll"] - before[r["source"]]["d_roll"]] for r in part])
        print(f"{name}: change in d_az median {np.median(d[:, 0]):+.3f}, |median| "
              f"{np.median(abs(d[:, 0])):.3f}, range {d[:, 0].min():+.3f} .. {d[:, 0].max():+.3f} deg; "
              f"d_pitch median {np.median(d[:, 1]):+.3f}, d_roll median {np.median(d[:, 2]):+.3f}")


def current(e: dict) -> bool:
    """Solved under this sky model by this version of the solver."""
    return (e.get("sky_model") == SG.model_id()
            and (e.get("solver") or "").split("+")[0] == __version__)

if __name__ == "__main__":
    main(*sys.argv[1:2])
