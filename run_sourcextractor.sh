#!/usr/bin/env bash
# Runs SourceXtractor++ (conda image) on every image in samples/ (listed by images.py):
#   - Rubin coadd cutouts: pixels in HDU 1, flux in nJy (AB zero point 31.4), no weight map.
#   - Suprime-Cam mosaic frames (.fits.fz): one run per CCD (extracted to output/inputs/ by images.py),
#     with its weight map, gain, saturation level and zero point from the header.
# Optional first argument: only process images whose name starts with it. Extra arguments go to SE++.
# SAMPLES / OUTPUT (default samples / output, relative to this directory) change where images are read
# from and catalogs written to, e.g. SAMPLES=output/inject/r0 OUTPUT=output/inject/r0 for inject.py.
set -euo pipefail

cd "$(dirname "$0")"
IMAGE=sourcextractor:1.1.0-conda
PY_IMAGE=stargal-python
SAMPLES=${SAMPLES:-samples}
OUTPUT=${OUTPUT:-output}
# Run as the host user so files written to the bind mount stay owned (and writable) by them on Linux.
AS_USER=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
mkdir -p "$OUTPUT"

docker build -q -f conda_sourcextractor_1_1_0.Dockerfile -t "$IMAGE" . >/dev/null
docker build -q -f python_tools.Dockerfile -t "$PY_IMAGE" . >/dev/null

prefix=${1:-}
[[ $# -gt 0 ]] && shift

docker run --rm "${AS_USER[@]}" -v "$PWD:/work" -w /work "$PY_IMAGE" python images.py "$SAMPLES" |
while read -r name path hdu weight weight_hdu zp gain satur _scale; do
  [[ $name == "$prefix"* ]] || continue
  echo "== $name"
  extra=()
  if [[ $weight != - ]]; then
    extra+=(--weight-image "$weight[$weight_hdu]" --weight-type weight --weight-threshold 0.5)
  fi
  [[ $gain != 0.000 ]] && extra+=(--detection-image-gain "$gain")
  [[ $satur != 0 ]] && extra+=(--detection-image-saturation "$satur")
  docker run --rm --platform linux/amd64 "${AS_USER[@]}" -v "$PWD:/work" -w /work \
    "$IMAGE" sourcextractor++ \
      --detection-image "$path[$hdu]" \
      --magnitude-zero-point "$zp" \
      "${extra[@]}" \
      --output-catalog-filename "$OUTPUT/${name}_cat.fits" \
      --output-properties SourceIDs,PixelCentroid,WorldCentroid,IsophotalFlux,AutoPhotometry,ShapeParameters,SourceFlags,NDetectedPixels,PixelBoundaries \
      --check-image-segmentation "$OUTPUT/${name}_seg.fits" \
      --check-image-background "$OUTPUT/${name}_bkg.fits" \
      --check-image-variance "$OUTPUT/${name}_var.fits" \
      --thread-count 4 \
      "$@" </dev/null
done
