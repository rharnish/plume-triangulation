#!/usr/bin/env bash
# Copernicus DEM GLO-30 (30 m DSM) tiles covering the HPWREN cameras used here.
# Open data on AWS, no credentials. ~50 MB/tile.
set -e
cd "$(dirname "$0")/dem"
for T in N31_00_W118_00 N31_00_W117_00 N31_00_W116_00 \
         N32_00_W119_00 N32_00_W118_00 N32_00_W117_00 N32_00_W116_00 \
         N33_00_W119_00 N33_00_W118_00 N33_00_W117_00 N33_00_W116_00 \
         N34_00_W119_00 N34_00_W118_00 N34_00_W117_00 N34_00_W116_00; do
  F="Copernicus_DSM_COG_10_${T}_DEM.tif"
  [ -s "$F" ] && continue
  URL="https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_${T}_DEM/${F}"
  if curl -fsSL -o "$F.part" "$URL"; then mv "$F.part" "$F"; echo "  got $T"
  else rm -f "$F.part"; echo "  MISSING $T (ocean tile, expected for some)"; fi
done
