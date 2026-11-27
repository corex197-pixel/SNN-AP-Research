#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# CPU mechanisms
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .local/round014-venv/bin/python -u experiments/ROUND-015/connectivity.py
# GPU backend: set exactly one authorized GPU UUID; run under total budget watchdog.
: "${CUDA_VISIBLE_DEVICES:?set exactly one authorized GPU UUID}"
timeout --signal=TERM --kill-after=5 840s .local/round014-venv/bin/python -u experiments/ROUND-015/backend.py
