"""Measured training throughput per variant, before committing days of GPU to it.

Each case is a real ultralytics training run on pyro-sdis with the full RECIPE, cut off
after WARMUP + MEASURE optimizer batches. Per batch it records the wait for the data
loader and the GPU step separately (CUDA synchronized at both edges), so a CPU-bound
loader on this 4-core box shows up as such instead of as a slow GPU.

The batch that actually ran is read back from the trainer: ultralytics halves the batch
on a first-epoch OOM (up to three times) without failing, and a timing at batch 4 is
not a timing at batch 16.

    .venv-train/bin/python -m src.figlib.scale.bench [variant ...] [--amp both|on|off]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from .recipe import DATA_YAML, RECIPE, ROOT, VARIANTS, weights

OUT = ROOT / "out" / "scale" / "bench.json"
WARMUP, MEASURE = 10, 40


class _Done(Exception):
    pass


def run_case(variant: str, amp: bool, batch: int, deterministic: bool) -> dict:
    arch, imgsz = VARIANTS[variant]
    t = {"mark": None, "data": [], "step": [], "batch": None}

    def start(trainer):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if t["mark"] is not None:
            t["data"].append(now - t["mark"])
        t["mark"] = now

    def end(trainer):
        torch.cuda.synchronize()
        now = time.perf_counter()
        t["step"].append(now - t["mark"])
        t["mark"] = now
        t["batch"] = trainer.batch_size
        t["mem"] = torch.cuda.max_memory_reserved() / 2**30
        if len(t["step"]) >= WARMUP + MEASURE:
            raise _Done

    model = YOLO(str(weights(arch)))
    model.add_callback("on_train_batch_start", start)
    model.add_callback("on_train_batch_end", end)
    torch.cuda.reset_peak_memory_stats()
    try:
        model.train(data=str(DATA_YAML), **{**RECIPE, "amp": amp, "epochs": 1, "batch": batch,
                                                 "deterministic": deterministic}, imgsz=imgsz,
                    val=False, plots=False, workers=4, device=0,
                    project=str(ROOT / "out" / "scale" / "bench_runs"),
                    name=f"{variant}_amp{int(amp)}_b{batch}_det{int(deterministic)}", exist_ok=True)
        raise RuntimeError("epoch finished before the measurement window filled")
    except _Done:
        pass
    # step[i] pairs with data[i-1]: the wait that preceded batch i (batch 0 has none)
    step = np.array(t["step"][WARMUP:])
    data = np.array(t["data"][WARMUP - 1:WARMUP - 1 + len(step)])
    per_batch = step + data
    n_train = json.loads((ROOT / "data" / "sdis" / "manifest.json").read_text())["counts"]["train"]["images"]
    ips = t["batch"] / per_batch.mean()
    del model
    torch.cuda.empty_cache()
    return {
        "variant": variant, "arch": arch, "imgsz": imgsz, "amp": amp,
        "deterministic": deterministic, "batch_requested": batch, "batch": t["batch"], "batches_measured": len(step),
        "images_per_s": round(float(ips), 2),
        "data_wait_frac": round(float(data.sum() / per_batch.sum()), 3),
        "gpu_step_s": round(float(np.median(step)), 3),
        "loader_wait_s": round(float(np.median(data)), 3),
        "peak_reserved_gib": round(t["mem"], 2),
        "hours_per_epoch": round(n_train / ips / 3600, 2),
        "hours_50_epochs": round(50 * n_train / ips / 3600, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--amp", choices=("both", "on", "off"), default="both")
    ap.add_argument("--batch", type=int, default=RECIPE["batch"])
    ap.add_argument("--deterministic", choices=("on", "off"), default="on")
    a = ap.parse_args()
    amps = {"both": (True, False), "on": (True,), "off": (False,)}[a.amp]
    results = json.loads(OUT.read_text()) if OUT.exists() else []
    for v in a.variants:
        for amp in amps:
            det = a.deterministic == "on"
            r = run_case(v, amp, a.batch, det)
            print(json.dumps(r), flush=True)
            key = lambda x: (x["variant"], x["amp"], x.get("batch_requested"), x.get("deterministic"))
            results = [x for x in results if key(x) != key(r)] + [r]
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
