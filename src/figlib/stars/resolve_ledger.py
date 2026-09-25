"""Re-solve every sequence behind the pose ledger under the current sky model, and rebuild it.

The ledger's poses were solved against J2000 star positions used as if they were of date
(stars.catalog's docstring; NOTES.md, 2026-09-24). With proper motion, precession and
refraction now applied by default, each of those sequences is solved again -- same tracks,
same solver it was accepted by (`solve_wide` where the summary records a `found_by`, `solve`
otherwise, falling back to `solve_wide` if that fails) -- and the ledger is rebuilt from exactly the same set of sources, so the only
thing that changes is the sky model. The previous ledger is kept beside the new results for
comparison, and the per-camera shift is printed.

    python -m src.figlib.stars.resolve_ledger
"""
from __future__ import annotations

import json
import shutil
from multiprocessing import Pool

import numpy as np

from .. import pose_ledger
from . import catalog as SG
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


def main():
    old = json.loads(pose_ledger.LEDGER.read_text())
    backup = S.DATA / "pose_ledger_before_resolve.json"
    if not backup.exists():
        shutil.copy(pose_ledger.LEDGER, backup)
    summary_path = S.DATA / "solve_summary.json"
    summary = {r["seq"]: r for r in json.loads(summary_path.read_text())}
    seqs = [e["source"].split(":", 1)[1] for e in old]
    jobs = [(q, "found_by" in summary.get(q, {})) for q in seqs]
    print(f"re-solving {len(jobs)} ledger sequences under: {SG.model_id()}", flush=True)
    with Pool(4) as pool:
        new = {r["seq"]: r for r in pool.imap_unordered(_one, jobs)}
    summary.update(new)
    summary_path.write_text(json.dumps(sorted(summary.values(), key=lambda r: (r["status"] != "solved", r["seq"])),
                                       indent=1, default=float) + "\n")

    t0 = {s["seq"]: s["t0"] for s in json.loads((S.ROOT / "data/meta/all/sequences.json").read_text())}
    t0.update({k: v["t0"] for k, v in S.SEQS.items() if k.startswith("hpwren_")})
    rows = pose_ledger.build([new[q] for q in seqs], t0)
    lost = [q for q in seqs if new[q]["status"] != "solved"]
    print(f"wrote {pose_ledger.LEDGER}: {len(rows)} of {len(seqs)} solves "
          f"({len(lost)} no longer pass: {lost})")
    before = {e["source"]: e for e in old}
    d = np.array([[r["d_az"] - before[r["source"]]["d_az"], r["d_pitch"] - before[r["source"]]["d_pitch"],
                   r["d_roll"] - before[r["source"]]["d_roll"]] for r in rows])
    print(f"change in d_az: median {np.median(d[:, 0]):+.3f}, range {d[:, 0].min():+.3f} .. {d[:, 0].max():+.3f} deg")
    print(f"change in d_pitch: median {np.median(d[:, 1]):+.3f};  d_roll: median {np.median(d[:, 2]):+.3f}")


if __name__ == "__main__":
    main()
