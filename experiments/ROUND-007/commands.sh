#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?expose one approved GPU}"
exec timeout 900s python -u experiments/ROUND-007/diagnostics.py
