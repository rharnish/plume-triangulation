"""Detector scale: does a bigger pyronear-style model, or a larger input, find plumes sooner?

yolo11s vs yolo11l at imgsz 1024 vs 1280, all trained here on pyronear/pyro-sdis with
pyronear's own recipe, then run over FIgLib next to the released rapid-raccoon model.
pyro-sdis holds no FIgLib frames, so unlike the released model these four are
uncontaminated on every sequence. Runs in .venv-train (requirements-train.txt).

Parked 2026-09-17, unfinished: real training needs more GPU than the GTX 1070 here.
Written and working: sdis (parquet -> YOLO layout), recipe (pyronear's arguments and the
four variants), bench (throughput per variant), sdis_predict and label_audit (the released
model against pyro-sdis labels). Not written: train.py, the FIgLib evaluation, and any
results or NOTES.md section. Data and COCO weights are gitignored; data/fetch_sdis.sh
re-fetches the pinned dataset.
"""
