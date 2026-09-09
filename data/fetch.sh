#!/bin/bash
# Fetch the FIgLib multi-camera sequences used by this project (~13 GB, 189 archives).
# Idempotent: re-running skips archives already present and retries only failures.
#
# Data credit: HPWREN, https://www.hpwren.ucsd.edu/
# Use of this data requires a credit reference to HPWREN in derivative work.
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST="$DIR/meta/tgz_multicam.txt"
OUT="$DIR/tgz"
BASE=https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar

[ -f "$LIST" ] || { echo "error: missing $LIST" >&2; exit 1; }
mkdir -p "$OUT"
rm -f "$DIR/failed.txt"

ok=0; skip=0; fail=0
while read -r f; do
  [ -n "$f" ] || continue
  if [ -s "$OUT/$f" ]; then skip=$((skip+1)); continue; fi
  if curl -sfL --retry 3 --retry-delay 2 -o "$OUT/$f.part" "$BASE/$f"; then
    mv "$OUT/$f.part" "$OUT/$f"; ok=$((ok+1))
  else
    rm -f "$OUT/$f.part"; echo "$f" >> "$DIR/failed.txt"; fail=$((fail+1))
  fi
done < "$LIST"

echo "fetched=$ok already-present=$skip failed=$fail"
[ "$fail" -gt 0 ] && echo "failures listed in $DIR/failed.txt" >&2
du -sh "$OUT"
