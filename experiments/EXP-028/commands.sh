#!/bin/bash
CUDA_VISIBLE_DEVICES="${ROUND026_GPU_UUID:?}" .local/round014-venv/bin/python -u experiments/ROUND-026/run.py
