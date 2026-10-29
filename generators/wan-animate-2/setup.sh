#!/usr/bin/env bash
# Wan2.2-Animate-2-14B (Aug 2026), animation mode: B's image moved by A's clip. Apache-2.0. Full model, not the
# distilled one. Upstream pins torch 2.7.0 cu126; runs on the platform torch.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Wan-Video/Wan-Animate-2 3ad2fef7d61d6200c9c653e0fe47be7616b323f3
venv
flash_attn
SKIP="triton" reqs requirements.txt   # a pip freeze; triton comes with torch
install -e . --no-deps
# results.mp4 is written by moviepy at its default crf 23 inside spawned workers: write it at crf 17
sed -i 's/\.write_videofile($/.write_videofile(ffmpeg_params=["-crf", "17", "-pix_fmt", "yuv420p"], filename=/' \
  pipelines/wan_animate_2_pipeline.py
hf Wan-AI/Wan2.2-Animate-2-14B --revision 6e8f1973bf0abc2aafd517992e8b6d88c3c46e69 --local-dir ckpts \
  --exclude "wan_animate_2/wan_animate_2_bf16_distillation.safetensors"
done_msg
