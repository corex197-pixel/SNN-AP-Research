#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?expose exactly one authorized GPU by UUID}"
exec timeout 1800s python -u experiments/ROUND-006/diagnostics.py
