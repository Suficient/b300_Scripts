# DREAMIDV_SIZE: 1280*720 (upstream's best quality) or 832*480. Output is 24 fps, so the clip is resampled to 24 fps.
. "$(dirname "$0")/../lib.sh"
ffmpeg -v error -y -i "$DFB_VIDEO" -r 24 -an -c:v libx264 -crf 14 "$DFB_TMP/driving.mp4"
"$PY" generate_dreamidv_dwpose.py --size "${DREAMIDV_SIZE:-1280*720}" --sample_steps 20 --base_seed "$DFB_SEED" \
  --ckpt_dir weights/Wan2.1-T2V-1.3B --dreamidv_ckpt weights/dreamidv.pth --offload_model False \
  --frame_num "$(frames "$DFB_DURATION" 24 4 1)" --sample_fps 24 \
  --ref_image "$DFB_REF" --ref_video "$DFB_TMP/driving.mp4" --save_file "$DFB_TMP/out.mp4"
finish "$DFB_TMP/out.mp4" "$DFB_AUDIO"
