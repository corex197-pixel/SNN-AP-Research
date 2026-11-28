#!/usr/bin/env bash
# One authorized RTX 4090 by UUID; launch only after code commit and preflight.
CUDA_VISIBLE_DEVICES="${ROUND024_GPU_UUID:?must select authorized RTX4090}" .local/round014-venv/bin/python -u experiments/ROUND-024/run.py
