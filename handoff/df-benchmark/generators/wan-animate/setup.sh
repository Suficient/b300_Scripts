#!/usr/bin/env bash
# Wan2.2-Animate-14B in replacement mode: the person in the clip is replaced by the reference character.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Wan-Video/Wan2.2
venv
flash_attn
reqs requirements.txt requirements_animate.txt
SAM2_BUILD_CUDA=0 install "git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266"
hf Wan-AI/Wan2.2-Animate-14B --local-dir Wan2.2-Animate-14B
done_msg
