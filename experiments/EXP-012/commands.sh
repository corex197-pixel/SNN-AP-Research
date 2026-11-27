#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?set exactly one authorized RTX4090 UUID}"
python -u experiments/ROUND-012/preflight.py
python -u experiments/ROUND-012/diagnostic.py
