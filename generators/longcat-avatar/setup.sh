#!/usr/bin/env bash
# LongCat-Video-Avatar 1.5: audio-driven avatar on Meituan's 13.6B LongCat-Video DiT. MIT.
. "$(dirname "$0")/../common.sh"
clone https://github.com/meituan-longcat/LongCat-Video 6b3f4b8582a8bc3f20f795735f5383716c4ba794
venv
flash_attn
SKIP="libsndfile1|tritonserverclient|sympy" reqs requirements.txt requirements_avatar.txt   # a system library;
# an unpublished serving client; sympy 1.13.1 is below torch's and audio-separator's floor
install librosa
# the avatar script loads the tokenizer, text encoder, VAE and scheduler from LongCat-Video
hf meituan-longcat/LongCat-Video --revision 03b55529b1d1d4045f5fbe14d65c8c6e8116b278 --include "tokenizer/*" --include "text_encoder/*" \
  --include "vae/*" --include "scheduler/*" \
  --local-dir weights/LongCat-Video
hf meituan-longcat/LongCat-Video-Avatar-1.5 --revision 92016c71d5d318d0f5d84e4db30015a571484ab6 \
  --exclude "base_model_int8/*" --exclude "*.msgpack" --exclude "*.h5" --exclude "*.bin" --local-dir weights/LongCat-Video-Avatar-1.5
done_msg
