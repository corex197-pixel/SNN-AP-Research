#!/bin/bash
# Launch only after committed code and verification of an authorized RTX4090.
CUDA_VISIBLE_DEVICES="${ROUND025_GPU_UUID:?}" .local/round014-venv/bin/python -u experiments/ROUND-025/run.py
