#!/bin/bash
# Wait until epoch N's checkpoint is fully on disk, stop the run, install the patched training transform and resume.
# "Epoch N+1" alone is not enough: ranks 1-7 log it while rank 0 is still saving. Needs eval.py's write-then-rename save. Usage: switch_at_epoch.sh NEXT_EPOCH NEW_UTILS_PY
L=/data/df_train/runs/df_vitl384.log; NEXT=$1; NEW=$2
CK=/data/df_train/runs/video_classification_frozen/$(awk '/^tag:/ {print $2}' /data/df_train/configs/df_vitl384.yaml)/latest.pt
until grep -q "Epoch $NEXT$" $L; do sleep 5; done
T0=$(date +%s); until [ -f $CK ] && [ ! -f $CK.tmp ] && [ $(stat -c %Y $CK) -ge $((T0 - 120)) ]; do sleep 5; done
echo "$(date -u +%T) saw Epoch $NEXT"; ls -la --time-style=+%T $CK
P=$(pgrep -f "^.venv/bin/python -m evals.main --fname /data/df_train/configs/df_vitl384.yaml")
kill $P $(pgrep -P $P)
for i in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ] && break; sleep 2; done
sleep 5
for pid in $(ps -eo pid=,ppid=,args= | awk '/multiprocessing.spawn/ && $2 == 1 {print $1}'); do kill $pid; done
sleep 3
echo "$(date -u +%T) stopped; gpu max mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1) MiB; orphans $(ps -eo ppid=,args= | awk '/multiprocessing.spawn/ && $1 == 1' | wc -l)"
cp "$NEW" /data/vjepa2/evals/video_classification_frozen/utils.py
mv $L /data/df_train/runs/df_vitl384_before_ep$NEXT.log
/data/df_train/train.sh
