# The clip's mouth is re-driven by B's audio. The script only saves side-by-side videos (original | result), so
# the result is cut out of the right half of *_paste_compare.mp4.
. "$(dirname "$0")/../lib.sh"
"$PY" infer_lip_sync_pipeline.py --video_path "$DFB_VIDEO" --audio_path "$DFB_REF_AUDIO" \
  --ckpt_path checkpoints/X-Dub_model.safetensors --ref_cfg_scale 2.5 --audio_cfg_scale 10.0 \
  --num_inference_steps 30 --seed "$DFB_SEED" --sample_name out --output_dir "$DFB_TMP/out"
ffmpeg -v error -y -i "$(ls "$DFB_TMP"/out/out_paste_compare*.mp4 | head -1)" -vf "crop=iw/2:ih:iw/2:0" -an \
  -c:v libx264 -crf 14 -preset veryfast -pix_fmt yuv420p "$DFB_TMP/out.mp4"
finish "$DFB_TMP/out.mp4" "$DFB_REF_AUDIO"
