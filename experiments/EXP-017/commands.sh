#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_VISIBLE_DEVICES:?one authorized RTX4090 UUID}"
# Preregistered preflight already ran; each replay must use a new Run ID and budget.
# timeout 600s .local/round014-venv/bin/python -u experiments/ROUND-016/preflight.py
# Formal group execution (absolute shared budget is enforced by train.py):
exec .local/round014-venv/bin/python -u experiments/ROUND-016/train.py
