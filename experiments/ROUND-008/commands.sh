#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?expose one approved GPU}"
exec timeout 1800s python -u experiments/ROUND-008/tcap.py
