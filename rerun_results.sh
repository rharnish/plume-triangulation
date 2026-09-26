#!/usr/bin/env bash
# Every stage whose output reaches README.md or docs/, in dependency order, from one commit.
#
# Each stage appends a line to data/meta/{,all/,recent/}runs.jsonl with the commit, every
# FIGLIB_* setting as resolved (settings.py) and the hash of every output, so run it from a
# committed tree. Inputs are not re-made here: detections (run_detect.sh, `recent detect`),
# star solves and the pose ledgers, CDN frames, DEM tiles.
#
#   ./rerun_results.sh            # ~20 min on four cores
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if ! git diff --quiet HEAD -- src tests configs; then
  echo "uncommitted changes under src/, tests/ or configs/: commit first" >&2
  exit 1
fi
mkdir -p out/logs
LOG="out/logs/rerun_$(git rev-parse --short HEAD).log"
exec > >(tee "$LOG") 2>&1
m() { echo; echo "== $*"; "$PY" -m "$@"; }

# Resize a figure onto docs/figures/ (JPEG at 1600 px wide, or a straight PNG copy).
doc() {
  "$PY" - "$1" "$2" <<'EOF'
import sys
from PIL import Image
src, dst = sys.argv[1:]
if dst.endswith(".png"):
    Image.open(src).save(dst)
else:
    im = Image.open(src).convert("RGB")
    im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS).save(dst, quality=85)
print(f"{src} -> {dst}")
EOF
}

# ---- core corpus: four camera models, same detections and solver
m src.figlib.geolocate
FIGLIB_LENS=fisheye m src.figlib.geolocate
FIGLIB_PROFILE=calibrated FIGLIB_POSE_FULL=0 m src.figlib.geolocate    # star azimuths only
FIGLIB_PROFILE=calibrated m src.figlib.geolocate                       # the whole solved camera
m src.figlib.compare_geolocation
m src.figlib.calfire
m src.figlib.fig_triangulate 20171010_FIRE 20240701_Kitchenfire 20201202_WillowFire \
  20171207_FIRE.2
m src.figlib.animate_triangulate 20240701_Kitchenfire
m src.figlib.fig_offsets
m src.figlib.stars.fig_geolocation
m src.figlib.evolve
m src.figlib.coverage
m src.figlib.bias

doc out/triangulate/20171010_FIRE.png docs/figures/triangulate_portola.jpg
doc out/triangulate/20201202_WillowFire-nightime-near-CDF-HQ.png docs/figures/triangulate_willow_night.jpg
cp out/videos/triangulate_20240701_Kitchenfire.gif docs/figures/triangulate_kitchenfire.gif
doc out/figures/offsets.png docs/figures/geolocation_offsets.png

# ---- all of FIgLib: the calfire audit and the terrain range
FIGLIB_CORPUS=all m src.figlib.geolocate
FIGLIB_CORPUS=all m src.figlib.calfire
export FIGLIB_CORPUS=all
m src.figlib.terrain_range validate
for band in cap 100 60 30; do m src.figlib.terrain_range solve "$band"; done
m src.figlib.terrain_range solve 100 ledger
m src.figlib.terrain_range figure 20260722_RainbowFire 30
m src.figlib.terrain_range figure 20260722_RainbowFire 100
m src.figlib.terrain_range figure 20240724_GroveFire 100
m src.figlib.fig_bearing
unset FIGLIB_CORPUS
doc out/all/terrain_range/20260722_RainbowFire_terrain_band30.png docs/figures/terrain_range_rainbow.jpg

# ---- recent: the extra CDN cameras (recent.py sets its own corpus)
m src.figlib.recent score
FIGLIB_LENS=fisheye m src.figlib.recent score
FIGLIB_PROFILE=recent FIGLIB_POSE_FULL=0 m src.figlib.recent score
FIGLIB_PROFILE=recent m src.figlib.recent score

echo; echo "log: $LOG"
