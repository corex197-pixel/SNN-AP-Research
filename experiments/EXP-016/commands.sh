#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# CPU mechanisms
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .local/round014-venv/bin/python -u experiments/ROUND-015/connectivity.py
# GPU backend: set exactly one authorized GPU UUID; run under total budget watchdog.
: "${CUDA_VISIBLE_DEVICES:?set exactly one authorized GPU UUID}"
timeout --signal=TERM --kill-after=5 840s .local/round014-venv/bin/python -u experiments/ROUND-015/backend.py
# The following scripts were executed as separate uniquely recorded runs after prior gates.
# Do not rerun against existing output paths without assigning a new Run ID and preserving logs.
# CUDA_VISIBLE_DEVICES=<GPU-UUID> timeout 840s .local/round014-venv/bin/python -u experiments/ROUND-015/model_bench.py
# CUDA_VISIBLE_DEVICES=<GPU-UUID> timeout 500s .local/round014-venv/bin/python -u experiments/ROUND-015/penalty_profile.py
