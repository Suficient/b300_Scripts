#!/bin/bash
# Runs unattended after training: waits for the run to exit, picks the head with the best val accuracy at the last
# epoch (from W&B, as the handoff says), scores val and test per clip, writes the reports and logs the test report to
# W&B. Everything goes to /data/df_train/runs/post_train.log. It does not stop the VM (that needs the Nebius console).
set -u
D=/data/df_train; PY=/data/vjepa2/.venv/bin/python; CFG=$D/configs/df_vitl384.yaml
TAG=$(awk '/^tag:/ {print $2}' $CFG); CK=$D/runs/video_classification_frozen/$TAG/latest.pt
log() { echo "$(date -u +%T) $*"; }

log "waiting for training ($TAG) to exit"
while pgrep -f "^.venv/bin/python -m evals.main --fname $CFG" > /dev/null; do sleep 60; done
log "training exited"
grep -qE "\[ +20\] train: " $D/runs/df_vitl384.log || { log "STOP: no epoch 20 result in the log, training did not finish"; exit 1; }
for pid in $(ps -eo pid=,ppid=,args= | awk '/multiprocessing.spawn/ && $2 == 1 {print $1}'); do kill $pid; done
for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 5; done
[ -f $CK ] && [ ! -f $CK.tmp ] || { log "STOP: checkpoint missing or half-written: $CK"; exit 1; }

HEAD=$($PY - "$TAG" <<'EOF'
import sys, wandb
h = wandb.Api().run(f"discerning-agents/vjepa-deepfake-probe/{sys.argv[1]}").history(samples=100000, pandas=True)
h = h[h["epoch"].notna()].sort_values("epoch").iloc[-1]
cols = sorted(c for c in h.index if c.startswith("val_heads/head_"))
best = max(cols, key=lambda c: h[c])
print(int(best.split("_")[-1]), file=sys.stdout)
print(f"epoch {int(h['epoch'])}: best {best} = {h[best]:.3f}; all: " + " ".join(f"{h[c]:.1f}" for c in cols), file=sys.stderr)
EOF
)
[ -n "$HEAD" ] || { log "STOP: could not read the best head from W&B"; exit 1; }
log "best val head: $HEAD"

cd $D
for s in val test; do
    log "scoring $s"
    $PY score.py $CFG $CK csv/$s.csv runs/${s}_scores.csv 8 > runs/score_$s.log 2>&1 || { log "STOP: scoring $s failed, see runs/score_$s.log"; exit 1; }
    tail -1 runs/score_$s.log
done
$PY report.py runs/val_scores.csv --head $HEAD --name val --no-wandb > /dev/null 2>&1
WANDB_PROJECT=vjepa-deepfake-probe $PY report.py runs/test_scores.csv --head $HEAD --name test 2>&1 | grep -v "^wandb:"
log "DONE. Reports: runs/val_scores_report_head$(printf %02d $HEAD).txt and runs/test_scores_report_head$(printf %02d $HEAD).txt. Stop the VM in the Nebius console."
