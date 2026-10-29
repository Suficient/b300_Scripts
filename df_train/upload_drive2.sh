#!/bin/bash
# Faster second stage for upload_drive.sh: splits archives over 8 GB into 8 GB parts (Drive uploads each file as one
# stream, ~10-20 MB/s, so parallel files are what's fast), uploads the archives 12 at a time while the first stage's
# weights upload finishes, then copies anything missed and md5-checks everything. Same log: /data/upload/upload.log
set -u
RC=~/bin/rclone; ST=/data/upload/vjepa_df_probe; DST=gdrive:vjepa_df_probe; LOG=/data/upload/upload.log
log() { echo "$(date -u +%T) $*" >> $LOG; }
cd $ST/archives
for f in *.tar; do
    [ $(stat -c %s $f) -gt 8589934592 ] || continue
    split -b 8G -d -a 2 $f $f.part && rm $f && log "split $f into $(ls $f.part* | wc -l) parts"
done
cat > README.txt <<'TXT'
Archives are uncompressed tar files. Big ones are split into 8 GB parts; join and extract with, for example:
  cat df_train_clean.tar.part* > df_train_clean.tar && tar -xf df_train_clean.tar
Each archive holds the folder under /data on the VM (df_train/clean, df_train/degraded, gen_out, outputs,
clipfarmer/accepted, inputs/pairs4, real, v3); vjepa2_src.tar is the V-JEPA 2 code with our patches.
TXT
log "uploading archives, 12 at a time"
$RC copy $ST/archives $DST/archives --transfers 12 --checkers 16 --drive-chunk-size 256M --retries 10 --low-level-retries 20 >> $LOG 2>&1
log "archives: rclone exit $?"
while pgrep -f "rclone copy $ST $DST" > /dev/null; do sleep 20; done
log "first stage finished; copying anything missed"
$RC copy $ST $DST --transfers 8 --checkers 16 --drive-chunk-size 256M --retries 10 >> $LOG 2>&1; log "catch-all: rclone exit $?"
log "checking md5 of every file against Drive"
$RC check $ST $DST --checkers 16 >> $LOG 2>&1 && log "CHECK OK: everything on Drive matches" || log "CHECK FAILED: see above"
$RC size $DST >> $LOG 2>&1
log "FINISHED"
