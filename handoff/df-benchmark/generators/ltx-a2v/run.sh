# B's frame is the first frame; the clip's audio drives the video and is kept as the sound track.
# LTX_SIZE: WIDTHxHEIGHT of the output (default 1280x704; both must be multiples of 64).
. "$(dirname "$0")/../lib.sh"
M=models/ltx-2.5
size=${LTX_SIZE:-1280x704}
ffmpeg -v error -y -i "$DFB_REF_FRAME" -vf "scale=${size%x*}:${size#*x}:force_original_aspect_ratio=increase,crop=${size%x*}:${size#*x}" \
  "$DFB_TMP/first.png"
"$PY" -m ltx_pipelines.a2vid_two_stage \
  --transformer-path $M/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --distilled-lora $M/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  --text-encoder-path $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path $M/vae/ltx-2.5-video-vae-bf16.safetensors --audio-vae-path $M/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --image "$DFB_TMP/first.png" 0 1.0 --audio-path "$DFB_AUDIO" --frame-rate 24 \
  --num-frames "$(frames "$DFB_DURATION" 24 8 1)" --width "${size%x*}" --height "${size#*x}" \
  --prompt "$DFB_PROMPT" --seed "$DFB_SEED" --output-path "$DFB_TMP/out.mp4"
finish "$DFB_TMP/out.mp4" "$DFB_AUDIO"
