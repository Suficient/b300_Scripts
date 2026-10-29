#!/usr/bin/env bash
# Kandinsky 6.0 Video Pro (distilled, 5 s): 29B CrossDiT generating video and 44 kHz audio together from a first
# frame and text. MIT. Runs through diffusers' Kandinsky6TI2VAPipeline.
. "$(dirname "$0")/../common.sh"
mkdir -p "$DIR" && cd "$DIR"
venv
install diffusers==0.42.1 "transformers>=4.57" accelerate sentencepiece "imageio[ffmpeg]" av qwen-vl-utils soundfile
hf kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers --revision 80d5e6c7f646c5f512b684e2da0765618d805894 \
  --local-dir weights --exclude "assets/*"
done_msg
