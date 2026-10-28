# One segment is 93 frames at 25 fps (3.7 s); the second continues it to 6.9 s, cut to the clip length.
# LONGCAT_RES: 480p (default) or 720p.
. "$(dirname "$0")/../lib.sh"
json_job "$DFB_TMP/in.json" cond_image "$DFB_REF_FRAME"
"$PY" -m torch.distributed.run --nproc_per_node=1 --master_port "$((29500 + RANDOM % 1000))" \
  run_demo_avatar_single_audio_to_video.py --checkpoint_dir weights/LongCat-Video-Avatar-1.5 \
  --model_type avatar-v1.5 --use_distill --stage_1 ai2v --num_segments 2 --resolution "${LONGCAT_RES:-480p}" \
  --input_json "$DFB_TMP/in.json" --output_dir "$DFB_TMP/out"
finish "$DFB_TMP/out/video_continue_2.mp4" "$DFB_AUDIO"
