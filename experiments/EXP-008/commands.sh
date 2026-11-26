#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?set exactly one authorized RTX 4090 UUID}"
python -u experiments/ROUND-009/preflight.py
python -u experiments/ROUND-009/train.py
