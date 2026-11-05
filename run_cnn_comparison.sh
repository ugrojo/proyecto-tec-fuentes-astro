#!/usr/bin/env bash
# CNN vs SPREAD_MODEL / CLASS_STAR / MU_MAX on the Subaru frame. Run after run_sourcextractor.sh,
# run_stargal.sh and compare_truth.py:
#   1. inject.py: REALIZATIONS (default 6) copies of each CCD with 1200 synthetic stars / galaxies
#   2. run_sourcextractor.sh on the injected CCDs, cnn_dataset.py injected: measure, match, cut
#   3. cnn_dataset.py real / benchmark: cutouts of the real sources, timing of the classical methods
#   4. cnn.py train / compare: train on injected CCDs 0-7, score on held-out CCDs 8-9 and real labels
# Finished steps are skipped (delete output/cnn to start over). Benchmarks want an otherwise idle machine.
set -euo pipefail

cd "$(dirname "$0")"
REALIZATIONS=${REALIZATIONS:-6}
AS_USER=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
docker build -q -f python_tools.Dockerfile -t stargal-python . >/dev/null
docker build -q -f cnn_tools.Dockerfile -t stargal-cnn . >/dev/null
py() { docker run --rm "${AS_USER[@]}" -v "$PWD:/work" -w /work stargal-python python -W ignore "$@"; }
cnn() { docker run --rm "${AS_USER[@]}" -v "$PWD:/work" -w /work stargal-cnn python "$@"; }

for ((r = 0; r < REALIZATIONS; r++)); do
  [[ -f output/cnn/injected_r$r.npz ]] && continue
  py inject.py --realization "$r"
  SAMPLES="output/inject/r$r/images" OUTPUT="output/inject/r$r" ./run_sourcextractor.sh SUPA >/dev/null
  py cnn_dataset.py injected "output/inject/r$r" --cleanup
done
[[ -f output/cnn/real.npz ]] || py cnn_dataset.py real
[[ -f output/cnn/benchmark_classic.json ]] || py cnn_dataset.py benchmark
[[ -f output/cnn/model.pt ]] || cnn cnn.py train
cnn cnn.py compare
