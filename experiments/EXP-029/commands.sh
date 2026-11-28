#!/bin/bash
# Four independent processes. Each stage requires prior authorized remote lock.
CUDA_VISIBLE_DEVICES="${ROUND027_GPU_UUID:?}" .local/round014-venv/bin/python -u experiments/ROUND-027/preflight.py
CUDA_VISIBLE_DEVICES="${ROUND027_GPU_UUID:?}" .local/round014-venv/bin/python -u experiments/ROUND-027/train.py
CUDA_VISIBLE_DEVICES="${ROUND027_GPU_UUID:?}" .local/round014-venv/bin/python -u experiments/ROUND-027/endpoint_admission.py
CUDA_VISIBLE_DEVICES="${ROUND027_GPU_UUID:?}" .local/round014-venv/bin/python -u experiments/ROUND-027/evaluate.py
