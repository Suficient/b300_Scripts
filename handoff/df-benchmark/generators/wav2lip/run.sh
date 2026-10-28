# The clip's mouth is re-driven by B's audio. wav2lip_gan (the sharper of the two released checkpoints).
. "$(dirname "$0")/../lib.sh"
"$PY" inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "$DFB_VIDEO_25" --audio "$DFB_REF_AUDIO" \
  --outfile "$DFB_TMP/out.mp4"
finish "$DFB_TMP/out.mp4" "$DFB_REF_AUDIO"
