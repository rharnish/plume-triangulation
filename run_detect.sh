#!/usr/bin/env bash
# One thread per worker, four workers. Without pinning, ONNX Runtime, OpenBLAS and
# OpenCV each spawn their own pool and the machine thrashes -- observed load average 12
# on four cores, running slower than a single unpinned worker.
# Queue is ordered so the sequences belonging to the ground-truth scoring set finish
# first, and geolocation can start before the long tail is done.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P 4 -I{} sh -c '.venv/bin/python -m src.figlib.detect_yolo "{}" >> out/logs/detect.log 2>&1' < data/meta/detect_queue.txt
echo "DETECTION PASS COMPLETE $(date -Is)" >> out/logs/detect.log
