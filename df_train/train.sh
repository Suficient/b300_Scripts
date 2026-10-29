#!/bin/bash
# Launch the full probe training detached on all 8 GPUs; logs to /data/df_train/runs/train.log, metrics to W&B.
# Usage: train.sh [config]   (default: the full run; pass configs/df_vitl384_smoke.yaml for the 20-step trial)
set -euo pipefail
CFG=${1:-/data/df_train/configs/df_vitl384.yaml}
LOG=/data/df_train/runs/$(basename "$CFG" .yaml).log
mkdir -p /data/df_train/runs
cd /data/vjepa2
WANDB_PROJECT=vjepa-deepfake-probe setsid nohup .venv/bin/python -m evals.main --fname "$CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  > "$LOG" 2>&1 < /dev/null &
echo "started pid $! -> $LOG"
