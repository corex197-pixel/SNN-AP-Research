#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${1:?provide a new EXP-001-RUN-NNN ID}"
python experiments/EXP-001/verify.py --run-id "$1" --config experiments/EXP-001/config.yaml
