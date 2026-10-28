#!/usr/bin/env bash
# LongCat-Video-Avatar 1.5: audio-driven avatar on Meituan's own 13.6B LongCat-Video DiT, DMD-distilled to 8 steps.
. "$(dirname "$0")/../common.sh"
clone https://github.com/meituan-longcat/LongCat-Video
venv
flash_attn
reqs requirements.txt requirements_avatar.txt
install librosa
# the avatar script loads the tokenizer, text encoder, VAE and scheduler from ../LongCat-Video
hf meituan-longcat/LongCat-Video --include "tokenizer/*" "text_encoder/*" "vae/*" "scheduler/*" \
  --local-dir weights/LongCat-Video
hf meituan-longcat/LongCat-Video-Avatar-1.5 --exclude "base_model_int8/*" --local-dir weights/LongCat-Video-Avatar-1.5
done_msg
