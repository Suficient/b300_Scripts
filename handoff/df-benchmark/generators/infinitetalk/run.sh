# INFINITETALK_SIZE: infinitetalk-480 (default) or infinitetalk-720. 40 steps, no LoRA: upstream's reference
# settings. The image is B's whole frame, so the framing matches a webcam clip.
. "$(dirname "$0")/../lib.sh"
json_job "$DFB_TMP/in.json" cond_video "$DFB_REF_FRAME"
"$PY" generate_infinitetalk.py --ckpt_dir weights/Wan2.1-I2V-14B-480P --wav2vec_dir weights/chinese-wav2vec2-base \
  --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors --input_json "$DFB_TMP/in.json" \
  --size "${INFINITETALK_SIZE:-infinitetalk-480}" --sample_steps 40 --mode streaming --motion_frame 9 \
  --offload_model False --base_seed "$DFB_SEED" --audio_save_dir "$DFB_TMP/audio" --save_file "$DFB_TMP/out"
finish "$DFB_TMP/out.mp4" "$DFB_AUDIO"
