"""What did quantization cost, measured in kilometres and seconds?

"INT8 lost 0.4 mAP" tells an operator nothing. The questions a deployment actually asks
are whether the fire is still located as accurately, whether it is still found as fast,
and whether the console fills up with more false alarms. All three are already
instrumented here, so the honest way to price a quantized model is to run it through the
same pipeline and read off the same numbers.

Each variant is scored in a subprocess with FIGLIB_DETS pointing at its detections,
because the detection directory is resolved at import. The FP32 reference is restricted
to the identical 93 sequences the Core ML passes cover -- comparing a subset against a
full-corpus baseline would confound quantization with the choice of fires.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out"
VARIANT_DIRS = {"fp32": OUT / "coreml" / "fp32_ref",
                "fp16": OUT / "coreml" / "fp16",
                "int8w": OUT / "coreml" / "int8w"}


def score_one() -> dict:
    """Run inside a subprocess with FIGLIB_DETS already set."""
    from .accumulate import gather, posterior
    from .falsealarm import sweep
    from .geom import haversine_km

    META = ROOT / "data" / "meta"
    cams = json.loads((META / "cams.json").read_text())
    seqs = {s["seq"]: s for s in json.loads((META / "sequences.json").read_text())}
    fires = {f["fire_id"]: f for f in json.loads((META / "fires.json").read_text())}
    resolved = json.loads((META / "resolved.json").read_text())
    scoring = [r for r in resolved
               if r["tier"] in ("confirmed", "probable") and r.get("triangulable")]

    geo = {}
    for t_max in (180, 2400):
        errs, areas = [], []
        for r in scoring:
            fire, truth = fires[r["fire_id"]], r["truth"]
            dets = gather(fire, seqs, cams, t_max)
            if len({c.split("-")[0] for c in dets}) < 2:
                continue
            c = (float(np.mean([cams[k]["lat"] for k in dets])),
                 float(np.mean([cams[k]["lon"] for k in dets])))
            _, _, _, la, lo = posterior(dets, cams, c, alpha=0.25)
            errs.append(haversine_km(la, lo, truth["lat"], truth["lon"]))
        e = sorted(errs)
        geo[t_max] = dict(n=len(e), median=round(e[len(e) // 2], 2) if e else None,
                          p75=round(e[3 * len(e) // 4], 2) if e else None,
                          max=round(e[-1], 2) if e else None,
                          within2=sum(x <= 2 for x in e), within5=sum(x <= 5 for x in e))

    fa = sweep([0.25, 0.40, 0.50], k=1, m=1)
    return dict(geo=geo, fa=fa)


def main(argv: list[str]) -> None:
    if argv and argv[0] == "_one":
        print("@@JSON@@" + json.dumps(score_one()))
        return

    results = {}
    for name, d in VARIANT_DIRS.items():
        if not d.exists() or not any(d.glob("*.json")):
            print(f"skip {name}: no detections at {d}")
            continue
        env = dict(os.environ, FIGLIB_DETS=str(d))
        r = subprocess.run([sys.executable, "-m", "src.figlib.quantization", "_one"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("@@JSON@@")]
        if not line:
            print(f"FAIL {name}:\n{r.stderr[-800:]}")
            continue
        results[name] = json.loads(line[0][8:])
        print(f"scored {name} ({len(list(d.glob('*.json')))} sequences)")

    (OUT / "quantization.json").write_text(json.dumps(results, indent=1) + "\n")

    print("\n== Geolocation, evidence accumulated over all detections (26 scoring fires)")
    print(f"{'variant':8s} {'window':>7} {'n':>3} {'median km':>10} {'p75':>7} "
          f"{'max':>7} {'<=2km':>6} {'<=5km':>6}")
    for name, r in results.items():
        for t in ("180", "2400"):
            g = r["geo"][t]
            print(f"{name:8s} {t + ' s':>7} {g['n']:>3} {str(g['median']):>10} "
                  f"{str(g['p75']):>7} {str(g['max']):>7} "
                  f"{g['within2']:>6} {g['within5']:>6}")

    print("\n== Alerting, single frame over threshold")
    print(f"{'variant':8s} {'tau':>5} {'FA/cam-day':>11} {'hi95':>7} {'recall':>7} "
          f"{'median s':>9}")
    for name, r in results.items():
        for row in r["fa"]:
            print(f"{name:8s} {row['tau']:>5} {str(row['fa_per_cam_day']):>11} "
                  f"{str(row['fa_hi95']):>7} {str(row['recall']):>7} "
                  f"{str(row['median_s']):>9}")


if __name__ == "__main__":
    main(sys.argv[1:])
