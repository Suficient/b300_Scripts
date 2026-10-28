# Source is B's whole frame: LivePortrait crops the face, animates it and pastes it back (max side 1280 px).
. "$(dirname "$0")/../lib.sh"
"$PY" inference.py -s "$DFB_REF_FRAME" -d "$DFB_VIDEO" -o "$DFB_TMP/out" --flag_crop_driving_video
finish "$(ls "$DFB_TMP"/out/*--*.mp4 | grep -v concat | head -1)" "$DFB_AUDIO"
