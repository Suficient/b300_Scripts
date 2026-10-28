#!/usr/bin/env bash
# HunyuanVideo-Avatar: audio-driven avatar on the 13B HunyuanVideo DiT. Tencent Hunyuan Community License:
# not licensed for use in the EU, UK or South Korea.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar
venv
flash_attn
reqs requirements.txt
hf tencent/HunyuanVideo-Avatar --exclude "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
  --local-dir weights
done_msg
