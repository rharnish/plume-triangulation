#!/usr/bin/env bash
# Runs the whole M3 edge experiment from docs/edge-m3.md end to end: venv, checkpoint,
# Core ML export, FIgLib data, latency/thermal/detection benchmarks, power join and the
# quantization cost report. Apple silicon macOS only.
#
# Every step mirrors docs/edge-m3.md 1:1 and is idempotent (skips work already done), so
# this is safe to re-run after an interruption -- including after the powermetrics prompt
# below, which is the one step that needs a real terminal.
#
# Usage: ./run_edge_experiment.sh   (run interactively -- sudo needs to prompt once)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "error: this needs Apple silicon macOS (powermetrics, the ANE, Core ML export)" >&2
  exit 1
fi

echo "== keeping the Mac awake for the duration =="
caffeinate -dis &
CAFFEINATE_PID=$!
trap 'kill $CAFFEINATE_PID 2>/dev/null' EXIT

echo "== 0. edge venv =="
if [[ ! -d .venv-edge ]]; then
  python3 -m venv .venv-edge
fi
.venv-edge/bin/pip install -q --upgrade pip
.venv-edge/bin/pip install -q -r requirements-edge.txt

echo "== 1. checkpoint =="
if [[ ! -f models/pyronear_rr_v8.1.0.pt ]]; then
  curl -sL -o models/pyronear_rr_v8.1.0.pt \
    https://huggingface.co/pyronear/yolo11s_rapid-raccoon_v8.1.0/resolve/main/best.pt
fi

echo "== 2. Core ML export (FP32 / FP16 / INT8-weight) =="
.venv-edge/bin/python models/export_variants.py

echo "== 3. FIgLib frame sequences (~13 GB, skips archives already present) =="
./data/fetch.sh

echo "== 4a. latency benchmark (12 conditions) =="
.venv-edge/bin/python -m src.figlib.bench_edge latency

echo "== powermetrics needs your password once here; it keeps running under =="
echo "== that authorization for the rest of this script, no further prompts =="
sudo -v
sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 1000 -o ~/pm.txt &
POWERMETRICS_PID=$!
trap 'kill $CAFFEINATE_PID 2>/dev/null; sudo kill $POWERMETRICS_PID 2>/dev/null' EXIT

echo "== 4b. thermal run (20 min sustained FP16/ANE load) =="
.venv-edge/bin/python -m src.figlib.bench_edge thermal fp16 ane 20

echo "== 4c. detections, all three variants (~3 min each) =="
.venv-edge/bin/python -m src.figlib.detect_coreml fp32 ane
.venv-edge/bin/python -m src.figlib.detect_coreml fp16 ane
.venv-edge/bin/python -m src.figlib.detect_coreml int8w ane

# quantization.py reads the FP32 pass from out/coreml/fp32_ref, not out/coreml/fp32 --
# detect_coreml names its output dir after the variant string.
if [[ -d out/coreml/fp32 && ! -d out/coreml/fp32_ref ]]; then
  mv out/coreml/fp32 out/coreml/fp32_ref
fi

echo "== stopping powermetrics =="
sudo kill "$POWERMETRICS_PID" 2>/dev/null || true
trap 'kill $CAFFEINATE_PID 2>/dev/null' EXIT
sleep 1

echo "== 5. joining power to runs, pricing quantization =="
.venv-edge/bin/python -m src.figlib.power ~/pm.txt
.venv-edge/bin/python -m src.figlib.quantization

echo "== done =="
echo "out/bench_latency.json, out/bench_thermal_fp16_ane.json, out/quantization.json"
