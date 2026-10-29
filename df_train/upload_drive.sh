#!/bin/bash
# Uploads the probe's weights, code, metadata, results, dataset and raw data to Google Drive (rclone remote gdrive:,
# folder vjepa_df_probe), then md5-checks every file. Big files are hard-linked into a staging folder with the same
# layout as the Mac copy (df_train/pull.py); folders with many small files go up as uncompressed .tar files, since
# Drive is slow per file. Weights and small files upload while the tars are being made. Log: /data/upload/upload.log
set -u
RC=~/bin/rclone; ST=/data/upload/vjepa_df_probe; DST=gdrive:vjepa_df_probe; LOG=/data/upload/upload.log
CK=/data/df_train/runs/video_classification_frozen/df-vitl384-2x16-cropfirst
FLAGS="--transfers 8 --checkers 16 --drive-chunk-size 256M --retries 10 --low-level-retries 20 --stats 1m --stats-one-line"
log() { echo "$(date -u +%T) $*" >> $LOG; }
mkdir -p $ST/weights $ST/archives $ST/raw/clipfarmer $ST/raw/inputs
log "staging"
ln -f /data/df_train/runs/deploy_vitl384_head18.pt $ST/weights/deploy_vitl384_head18.pt
ln -f $CK/latest.pt $ST/weights/final_ep20_all_heads.pt
ln -f /data/df_train/runs/ckpt_ep16.pt $ST/weights/ckpt_ep16_all_heads.pt
ln -f /data/vjepa2_ckpt/vjepa2_1_vitl_dist_vitG_384.pt $ST/weights/vjepa2_1_vitl_dist_vitG_384.pt
ln -f /data/clipfarmer/clipfarmer.sqlite $ST/raw/clipfarmer/clipfarmer.sqlite
cp /data/inputs/select500.txt $ST/raw/inputs/
rsync -a --exclude=/clean/ --exclude=/degraded/ --exclude="*.pt" --exclude="*.pt.*" \
      --exclude=/runs/video_classification_frozen/df-vitl384-2x16-smoke/ /data/df_train/ $ST/df_train/
rsync -a --exclude=__pycache__/ --exclude="*.png" /tmp/vjtest/ $ST/helpers/
tar -C /data -cf $ST/archives/vjepa2_src.tar --exclude=vjepa2/.venv --exclude=__pycache__ vjepa2
log "uploading weights and small files"
$RC copy $ST $DST $FLAGS --exclude "archives/**" >> $LOG 2>&1 &
UP=$!
for d in df_train/clean df_train/degraded gen_out outputs clipfarmer/accepted inputs/pairs4 real v3; do
    out=$ST/archives/$(echo $d | tr / _).tar
    [ -f $out ] || { tar -C /data -cf $out.tmp $d && mv $out.tmp $out && log "packed $out ($(du -h $out | cut -f1))"; }
done
wait $UP; log "weights and small files: rclone exit $?"
log "uploading archives"
$RC copy $ST $DST $FLAGS >> $LOG 2>&1; log "archives: rclone exit $?"
log "checking md5 of every file against Drive"
$RC check $ST $DST --checkers 16 >> $LOG 2>&1 && log "CHECK OK: everything on Drive matches" || log "CHECK FAILED: see above"
$RC size $DST >> $LOG 2>&1
log "FINISHED"
