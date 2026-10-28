# The clip's mouth is re-driven by B's audio. Upstream's inference.sh settings.
. "$(dirname "$0")/../lib.sh"
"$PY" -m scripts.inference --unet_config_path configs/unet/stage2_512.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt --inference_steps 20 --guidance_scale 1.5 --enable_deepcache \
  --video_path "$DFB_VIDEO_25" --audio_path "$DFB_REF_AUDIO" --video_out_path "$DFB_TMP/out.mp4" \
  --temp_dir "$DFB_TMP/temp" --seed "$DFB_SEED"
finish "$DFB_TMP/out.mp4" "$DFB_REF_AUDIO"
