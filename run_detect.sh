#!/usr/bin/env bash
# One thread per worker, four workers. Without pinning, ONNX Runtime, OpenBLAS and
# OpenCV each spawn their own pool and the machine thrashes -- observed load average 12
# on four cores, running slower than a single unpinned worker.
# Queue is ordered so the sequences belonging to the ground-truth scoring set finish
# first, and geolocation can start before the long tail is done.
#
# FIGLIB_CORPUS=extra runs the separately fetched archives instead (see corpus.py). The
# pass ends with one run-log entry covering every detection file and the model pin.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CORPUS="${FIGLIB_CORPUS:-core}"
mkdir -p out/logs
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$CORPUS" = core ]; then
  LOG=out/logs/detect.log
  QUEUE=data/meta/detect_queue.txt
else
  LOG="out/logs/detect_${CORPUS}.log"
  QUEUE="$(mktemp)"
  .venv/bin/python -m src.figlib.corpus stems > "$QUEUE"
fi
xargs -P 4 -I{} sh -c '.venv/bin/python -m src.figlib.detect_yolo "{}" >> '"$LOG"' 2>&1' < "$QUEUE"
echo "DETECTION PASS COMPLETE $(date -Is)" >> "$LOG"
.venv/bin/python -m src.figlib.provenance record-detect "$STARTED" >> "$LOG" 2>&1
