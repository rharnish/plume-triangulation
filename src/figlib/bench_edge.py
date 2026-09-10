"""Edge inference benchmark on Apple silicon: latency, dispatch, power, thermals.

Written for the M3 MacBook Air, which is a defensible stand-in for a fielded camera in
the ways that matter here -- it is ARM64 with a real NPU, it reports per-unit wattage,
and being fanless it produces a genuine sustained-throughput throttle curve rather than
a burst number. Fielded fire cameras sit in sealed enclosures on mountaintops; a laptop
that cannot dump heat is closer to that than any rented GPU.

Three things this measures that a bare latency number does not:

* **Model-only against end-to-end.** JPEG decode and letterbox are part of the real
  pipeline and they are not free. End-to-end is what bounds achievable frame rate, and
  frame rate is what the seconds-to-alert sweep showed to be partly sampling-bound.
* **Where the work actually ran.** A requested compute unit is a request. Core ML will
  silently fall back, so placement is confirmed against the compute plan and against
  ANE wattage, never against the setting we asked for.
* **Steady state, not burst.** Every condition runs for a fixed duration rather than a
  fixed iteration count, so power sampling has something to average over, and the
  thermal run holds load long enough for the fans that do not exist to not help.

Timestamps bracket every condition so an externally running `powermetrics` log can be
sliced by wall clock afterwards.
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np

from .detect_yolo import read_frames, letterbox, IMGSZ
from .detect_coreml import make_model, decode, MODELS, UNITS

ROOT = Path(__file__).resolve().parents[2]
TGZ_DIR = ROOT / "data" / "tgz"
OUT = ROOT / "out"

VARIANTS = {"fp32": "pyronear_rr_v8.1.0.mlpackage",
            "fp16": "pyronear_rr_v8.1.0_fp16.mlpackage",
            "int8w": "pyronear_rr_v8.1.0_int8w.mlpackage"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_frames(n_decoded: int = 48, n_blobs: int = 200):
    """Pre-decoded PIL images for model-only timing; raw JPEGs for end-to-end."""
    from PIL import Image
    blobs, imgs = [], []
    for tgz in sorted(TGZ_DIR.glob("*.tgz")):
        for _epoch, _off, blob in read_frames(tgz):
            blobs.append(blob)
            if len(blobs) >= n_blobs:
                break
        if len(blobs) >= n_blobs:
            break
    for blob in blobs[:n_decoded]:
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        canvas, r, left, top = letterbox(img, IMGSZ)
        imgs.append((Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)),
                     r, left, top, img.shape[1], img.shape[0]))
    return blobs, imgs


def compute_plan(path: Path) -> dict | None:
    """Per-operation device placement, if this macOS/coremltools pair exposes it.

    This is the strong form of the dispatch check. Where it is unavailable, ANE wattage
    from powermetrics is the fallback -- weaker, but still evidence about the hardware
    rather than about what we asked for.
    """
    try:
        import coremltools as ct
        from coremltools.models.compute_plan import MLComputePlan
        from coremltools.models.compute_device import (
            MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice)
        # The plan API wants a *compiled* .mlmodelc, and handed an .mlpackage it aborts
        # the process from C++ rather than raising, which no `except` here would catch.
        holder = ct.models.MLModel(str(path))
        plan = MLComputePlan.load_from_path(
            str(holder.get_compiled_model_path()), compute_units=ct.ComputeUnit.ALL)
        counts: dict[str, int] = {}
        prog = plan.model_structure.program
        for fn in prog.functions.values():
            for op in fn.block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    continue
                d = usage.preferred_compute_device
                name = ("ANE" if isinstance(d, MLNeuralEngineComputeDevice) else
                        "GPU" if isinstance(d, MLGPUComputeDevice) else
                        "CPU" if isinstance(d, MLCPUComputeDevice) else type(d).__name__)
                counts[name] = counts.get(name, 0) + 1
        return counts or None
    except Exception as exc:
        return {"unavailable": str(exc)[:120]}


def time_condition(variant: str, units: str, imgs, blobs,
                   seconds: float = 20.0, warmup: int = 8) -> dict:
    """Model-only and end-to-end latency, run for a duration rather than a count."""
    from PIL import Image
    model = make_model(MODELS / VARIANTS[variant], units)

    # The first prediction includes model load and .mlmodelc compilation; it is not
    # a latency measurement and would dominate a short run.
    for i in range(warmup):
        model.predict({"image": imgs[i % len(imgs)][0]})

    t_start = time.time()
    lat_model = []
    while time.time() - t_start < seconds:
        pil = imgs[len(lat_model) % len(imgs)][0]
        t0 = time.perf_counter()
        model.predict({"image": pil})
        lat_model.append((time.perf_counter() - t0) * 1000)
    t_model_end = time.time()

    t_e2e_start = time.time()
    lat_e2e = []
    while time.time() - t_e2e_start < seconds:
        blob = blobs[len(lat_e2e) % len(blobs)]
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        canvas, r, left, top = letterbox(img, IMGSZ)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        y = model.predict({"image": pil})
        decode(np.asarray(next(iter(y.values()))), r, left, top,
               img.shape[1], img.shape[0])
        lat_e2e.append((time.perf_counter() - t0) * 1000)
    t_end = time.time()

    def stats(v):
        v = sorted(v)
        return dict(n=len(v), median_ms=round(statistics.median(v), 2),
                    p95_ms=round(v[int(0.95 * (len(v) - 1))], 2),
                    min_ms=round(v[0], 2), fps=round(1000 / statistics.median(v), 2))

    return dict(variant=variant, units=units,
                model_only=stats(lat_model), end_to_end=stats(lat_e2e),
                t_model=[t_start, t_model_end], t_e2e=[t_e2e_start, t_end],
                started=now_iso())


def thermal_run(variant: str, units: str, minutes: float = 20.0, imgs=None) -> dict:
    """Sustained load; per-10-second throughput, to expose the throttle curve."""
    model = make_model(MODELS / VARIANTS[variant], units)
    for i in range(8):
        model.predict({"image": imgs[i % len(imgs)][0]})

    t0 = time.time()
    buckets, count, bucket_start, i = [], 0, time.time(), 0
    while time.time() - t0 < minutes * 60:
        model.predict({"image": imgs[i % len(imgs)][0]})
        i += 1
        count += 1
        if time.time() - bucket_start >= 10.0:
            buckets.append(round(count / (time.time() - bucket_start), 2))
            count, bucket_start = 0, time.time()
    return dict(variant=variant, units=units, minutes=minutes,
                fps_per_10s=buckets, t=[t0, time.time()], started=now_iso())


def main(argv: list[str]) -> None:
    mode = argv[0] if argv else "latency"
    OUT.mkdir(exist_ok=True)
    host = dict(machine=platform.machine(), mac_ver=platform.mac_ver()[0],
                chip=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True).stdout.strip(),
                power=subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                     text=True).stdout.split("'")[1::2][:1])
    print(f"host: {host}")
    blobs, imgs = load_frames()
    print(f"loaded {len(blobs)} jpegs, {len(imgs)} pre-decoded")

    if mode == "latency":
        rows = []
        for variant in VARIANTS:
            print(f"\n== compute plan, {variant}: "
                  f"{compute_plan(MODELS / VARIANTS[variant])}")
            for units in ("cpu", "gpu", "ane", "all"):
                print(f"  [{now_iso()}] {variant:6s} {units:4s} ...",
                      end="", flush=True)
                r = time_condition(variant, units, imgs, blobs)
                rows.append(r)
                print(f" model {r['model_only']['median_ms']:7.2f} ms "
                      f"({r['model_only']['fps']:5.1f} fps)   "
                      f"e2e {r['end_to_end']['median_ms']:7.2f} ms "
                      f"({r['end_to_end']['fps']:5.1f} fps)")
        (OUT / "bench_latency.json").write_text(
            json.dumps(dict(host=host, rows=rows), indent=1) + "\n")

    elif mode == "thermal":
        variant = argv[1] if len(argv) > 1 else "fp16"
        units = argv[2] if len(argv) > 2 else "all"
        mins = float(argv[3]) if len(argv) > 3 else 20.0
        print(f"[{now_iso()}] sustained {variant}/{units} for {mins} min")
        r = thermal_run(variant, units, mins, imgs)
        r["host"] = host
        (OUT / f"bench_thermal_{variant}_{units}.json").write_text(
            json.dumps(r, indent=1) + "\n")
        f = r["fps_per_10s"]
        print(f"first 30 s {statistics.mean(f[:3]):.2f} fps -> "
              f"last 30 s {statistics.mean(f[-3:]):.2f} fps "
              f"({100 * (1 - statistics.mean(f[-3:]) / statistics.mean(f[:3])):.1f}% drop)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
