#!/usr/bin/env bash
# InfiniteTalk: audio-driven talking video on Wan2.1-I2V-14B (480P weights, 480 and 720 output).
. "$(dirname "$0")/../common.sh"
clone https://github.com/MeiGen-AI/InfiniteTalk
venv
flash_attn   # before reqs: xfuser depends on it
reqs requirements.txt
install librosa "misaki[en]"
hf Wan-AI/Wan2.1-I2V-14B-480P --local-dir weights/Wan2.1-I2V-14B-480P
hf TencentGameMate/chinese-wav2vec2-base --local-dir weights/chinese-wav2vec2-base
hf TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir weights/chinese-wav2vec2-base
hf MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors --local-dir weights/InfiniteTalk
done_msg
