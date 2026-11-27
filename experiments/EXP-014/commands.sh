#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?exactly one authorized GPU UUID required}"
timeout --signal=TERM --kill-after=10s 890s python -u experiments/ROUND-013/run.py
