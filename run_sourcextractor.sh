#!/usr/bin/env bash
# Runs SourceXtractor++ (conda image) on every image in samples/ (listed by images.py):
#   - Rubin coadd cutouts: pixels in HDU 1, flux in nJy (AB zero point 31.4), no weight map.
#   - Suprime-Cam mosaic frames (.fits.fz): one run per CCD (extracted to output/inputs/ by images.py),
#     with its weight map, gain, saturation level and zero point from the header.
# Optional first argument: only process images whose name starts with it. Extra arguments go to SE++.
set -euo pipefail

cd "$(dirname "$0")"
IMAGE=sourcextractor:1.1.0-conda
PY_IMAGE=stargal-python
mkdir -p output

docker build -q -f conda_sourcextractor_1_1_0.Dockerfile -t "$IMAGE" . >/dev/null
docker build -q -f python_tools.Dockerfile -t "$PY_IMAGE" . >/dev/null

prefix=${1:-}
[[ $# -gt 0 ]] && shift

docker run --rm -v "$PWD:/work" -w /work "$PY_IMAGE" python images.py |
while read -r name path hdu weight weight_hdu zp gain satur _scale; do
  [[ $name == "$prefix"* ]] || continue
  echo "== $name"
  extra=()
  if [[ $weight != - ]]; then
    extra+=(--weight-image "$weight[$weight_hdu]" --weight-type weight --weight-threshold 0.5)
  fi
  [[ $gain != 0.000 ]] && extra+=(--detection-image-gain "$gain")
  [[ $satur != 0 ]] && extra+=(--detection-image-saturation "$satur")
  docker run --rm --platform linux/amd64 -v "$PWD:/work" -w /work \
    "$IMAGE" sourcextractor++ \
      --detection-image "$path[$hdu]" \
      --magnitude-zero-point "$zp" \
      "${extra[@]}" \
      --output-catalog-filename "output/${name}_cat.fits" \
      --output-properties SourceIDs,PixelCentroid,WorldCentroid,IsophotalFlux,AutoPhotometry,ShapeParameters,SourceFlags,NDetectedPixels,PixelBoundaries \
      --check-image-segmentation "output/${name}_seg.fits" \
      --check-image-background "output/${name}_bkg.fits" \
      --check-image-variance "output/${name}_var.fits" \
      --thread-count 4 \
      "$@" </dev/null
done
