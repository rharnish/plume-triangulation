"""Group sequences into distinct fires.

FIgLib names an unnamed event by date alone, so `20180603_FIRE` is not one fire --
it is every unnamed fire the network saw that day. Three cameras fired at 20:20-20:22
UTC, two more at 23:09, one at 01:24 the next morning: three separate ignitions under
one label. Grouping by event name therefore overcounts multi-camera coverage and
misattributes ground truth.

Cluster on t0 instead. Every sequence spans roughly t0 +/- 2400 s, so two views of the
same ignition must have t0 within that half-window; 1800 s is the working threshold,
sitting in a sparse valley of the observed gap distribution (mass below 900 s, a clear
tail beyond 3600 s).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import corpus as C

ROOT = Path(__file__).resolve().parents[2]
META_DIR = C.current().meta
GAP_S = 1800


def cluster(seqs: list[dict], gap_s: int = GAP_S) -> list[dict]:
    by_event: dict[str, list[dict]] = defaultdict(list)
    for s in seqs:
        by_event[s["event"]].append(s)

    fires: list[dict] = []
    for event, group in sorted(by_event.items()):
        group.sort(key=lambda s: s["t0"])
        runs: list[list[dict]] = [[group[0]]]
        for s in group[1:]:
            if s["t0"] - runs[-1][-1]["t0"] > gap_s:
                runs.append([])
            runs[-1].append(s)

        for i, run in enumerate(runs):
            posed = [s for s in run if s["has_pose"]]
            sites = sorted({s["site"] for s in posed})
            t0s = [s["t0"] for s in run]
            fires.append({
                "fire_id": event if len(runs) == 1 else f"{event}.{i + 1}",
                "event": event,
                "t0_min": min(t0s),
                "t0_max": max(t0s),
                "t0_median": sorted(t0s)[len(t0s) // 2],
                "n_seqs": len(run),
                "n_posed": len(posed),
                "sites": sites,
                "triangulable": len(sites) >= 2,
                "sequences": [s["seq"] for s in run],
            })
    return fires


def main() -> None:
    from . import provenance as P
    started = P.utc_now()
    seqs = json.loads((META_DIR / "sequences.json").read_text())
    fires = cluster(seqs)
    dest = META_DIR / "fires.json"
    dest.write_text(json.dumps(fires, indent=1) + "\n")
    P.record("fires", [dest], params={"gap_s": GAP_S}, started=started,
             extra_inputs=[META_DIR / "sequences.json"])

    tri = [f for f in fires if f["triangulable"]]
    split = [f for f in fires if "." in f["fire_id"]]
    print(f"{len(seqs)} sequences -> {len(fires)} fires "
          f"(from {len({f['event'] for f in fires})} event labels)")
    print(f"events split into multiple fires: "
          f"{len({f['event'] for f in split})} -> {len(split)} fires")
    print(f"triangulable (>=2 distinct posed sites): {len(tri)}")
    if tri:
        best = max(tri, key=lambda f: len(f["sites"]))
        print(f"best coverage: {best['fire_id']} from {len(best['sites'])} sites "
              f"{best['sites']}")
    print(f"-> {dest.relative_to(ROOT)}")

    for gap in (900, 1800, 3600):
        c = cluster(seqs, gap)
        print(f"  sensitivity gap={gap:>4}s: {len(c):>3} fires, "
              f"{sum(1 for f in c if f['triangulable']):>2} triangulable")


if __name__ == "__main__":
    main()
