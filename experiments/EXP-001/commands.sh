#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
python experiments/EXP-001/verify.py
