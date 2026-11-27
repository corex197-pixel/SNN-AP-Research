#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .local/round014-venv/bin/python -u experiments/ROUND-014/unit_tests.py
# Supplementary test uses its own committed code and distinct output artifacts.
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .local/round014-venv/bin/python -u experiments/ROUND-014/supplement.py
