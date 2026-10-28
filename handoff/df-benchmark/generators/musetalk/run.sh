# The clip's mouth is re-driven by B's audio, at 25 fps (MuseTalk's training rate).
. "$(dirname "$0")/../lib.sh"
printf 'task_0:\n  video_path: "%s"\n  audio_path: "%s"\n' "$DFB_VIDEO_25" "$DFB_REF_AUDIO" > "$DFB_TMP/task.yaml"
"$PY" -m scripts.inference --inference_config "$DFB_TMP/task.yaml" --result_dir "$DFB_TMP/out" \
  --unet_model_path models/musetalkV15/unet.pth --unet_config models/musetalkV15/musetalk.json --version v15 \
  --ffmpeg_path "$(dirname "$(command -v ffmpeg)")" --output_vid_name out.mp4
finish "$DFB_TMP/out/v15/out.mp4" "$DFB_REF_AUDIO"
