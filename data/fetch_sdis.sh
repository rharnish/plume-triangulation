#!/usr/bin/env bash
# pyronear/pyro-sdis (Apache 2.0), 7 parquet shards, 3.3 GB, pinned to one revision.
# Hashes are checked against the pinned LFS oids by src/figlib/scale/sdis.py, which then
# unpacks the shards into data/sdis/yolo/ for training.
set -euo pipefail
REV=a1e553ec4d806f71fc6db744cc22bc3469487382
DEST="$(dirname "$0")/sdis/parquet"
mkdir -p "$DEST"
for f in train-0000{0..5}-of-00006 val-00000-of-00001; do
  [ -f "$DEST/$f.parquet" ] && continue
  curl -sfL -o "$DEST/$f.parquet.part" \
    "https://huggingface.co/datasets/pyronear/pyro-sdis/resolve/$REV/data/$f.parquet"
  mv "$DEST/$f.parquet.part" "$DEST/$f.parquet"
  echo "fetched $f"
done
