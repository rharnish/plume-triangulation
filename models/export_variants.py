"""FP32 / FP16 / INT8-weight Core ML variants, landed flat in models/.

See docs/edge-m3.md, "Reproducing this" step 2, for why each step is shaped this way:
Ultralytics' half=False path does not produce true FP32 before 8.4.146 (pin matters),
and its every export lands at the same default path, which forces the FP32-stash dance
around the FP16 export below.
"""
import shutil
from pathlib import Path
import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights)
from ultralytics import YOLO

M = Path(__file__).parent
PT = M / "pyronear_rr_v8.1.0.pt"
fp32 = M / "pyronear_rr_v8.1.0.mlpackage"
fp16 = M / "pyronear_rr_v8.1.0_fp16.mlpackage"
int8 = M / "pyronear_rr_v8.1.0_int8w.mlpackage"

# FP32: Ultralytics' default output name already equals `fp32` (same stem, same dir).
if not fp32.exists():
    tmp = Path(YOLO(str(PT)).export(
        format="coreml", imgsz=1024, nms=False, half=False, int8=False))
    if tmp != fp32:
        shutil.move(str(tmp), str(fp32))

# FP16: same default output name as FP32, so stash FP32 out of the way for the
# duration of this export, then restore it.
if not fp16.exists():
    stash = M / "_fp32.stash.mlpackage"
    shutil.move(str(fp32), str(stash))
    tmp = Path(YOLO(str(PT)).export(
        format="coreml", imgsz=1024, nms=False, half=True, int8=False))
    shutil.move(str(tmp), str(fp16))
    shutil.move(str(stash), str(fp32))

# INT8 weights, applied to the FP16 program -- not a separate Ultralytics export.
if not int8.exists():
    src = ct.models.MLModel(str(fp16))
    cfg = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    linear_quantize_weights(src, config=cfg).save(str(int8))

for p in (fp32, fp16, int8):
    mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20
    print(f"{p.name:42s} {mb:7.1f} MB")
