#!/bin/bash
# Waits for the held-out generators to finish, then prepares (heldout.py), scores (score.py, final checkpoint, 8 GPUs)
# and reports (heldout_report.py, head 18). Log: /data/df_train/runs/heldout_run.log
set -u
D=/data/df_train; PY=/data/vjepa2/.venv/bin/python; CK=$D/runs/video_classification_frozen/df-vitl384-2x16-cropfirst/latest.pt
log() { echo "$(date -u +%T) $*"; }
cd $D
while pgrep -f "gen/facefusion-[a-z0-9-]*/batch.py" > /dev/null; do sleep 30; done
log "generators finished"; (cd ~/gen && python3 drive.py status 2>&1 | grep -E "^facefusion-")
$PY heldout.py 96 2>&1 | grep -v "^missing" | tail -3
log "scoring"
$PY score.py configs/df_vitl384.yaml $CK csv/heldout.csv runs/heldout_scores.csv 8 > runs/score_heldout.log 2>&1 || { log "STOP: scoring failed"; tail -5 runs/score_heldout.log; exit 1; }
tail -1 runs/score_heldout.log
$PY heldout_report.py runs/heldout_scores.csv --head 18
log "DONE"
