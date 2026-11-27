#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?set exactly one authorized RTX4090 UUID}"
# Requires EXP-012 gate, independent audit, and the original locked checkpoint.
python -u experiments/ROUND-012/formal.py
