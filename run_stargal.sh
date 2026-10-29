#!/usr/bin/env bash
# Star/galaxy separation on top of the SourceXtractor++ catalogs (run ./run_sourcextractor.sh first):
#   PSFEx PSF per image + classic SExtractor reference (config/psf_pipeline.sh), then
#   SExtractor-style SPREAD_MODEL, CLASS_STAR and MU_ for every SE++ source (spread_model.py).
set -euo pipefail

cd "$(dirname "$0")"
SE_IMAGE=sourcextractor:1.1.0-conda
PY_IMAGE=stargal-python
JOBS=${JOBS:-4}

docker build -q -f python_tools.Dockerfile -t "$PY_IMAGE" . >/dev/null

docker run --rm -v "$PWD:/work" -w /work "$PY_IMAGE" python images.py |
while read -r name _; do
  if [[ ! -f "output/psf/${name}_prepsfex.psf" || ! -f "output/${name}_sex2_cat.fits" ]]; then
    echo "$name"
  fi
done |
xargs -r -P "$JOBS" -I{} docker run --rm --platform linux/amd64 -v "$PWD:/work" -w /work "$SE_IMAGE" \
  bash config/psf_pipeline.sh {}

docker run --rm -v "$PWD:/work" -w /work "$PY_IMAGE" python spread_model.py "$@"
