#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?exactly one authorized RTX4090 UUID}"
# Use new Run IDs for retries; do not overwrite prior preflight or continuation output.
# Preflight: .local/round014-venv/bin/python -u experiments/ROUND-017/preflight.py
# Continuation: .local/round014-venv/bin/python -u experiments/ROUND-017/continue_train.py
