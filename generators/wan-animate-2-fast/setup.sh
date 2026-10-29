#!/usr/bin/env bash
# Wan2.2-Animate-2, the authors' distilled model (10 steps, guide 1). Same install as wan-animate-2.
. "$(dirname "$0")/../common.sh"
reuse wan-animate-2
hf Wan-AI/Wan2.2-Animate-2-14B --revision 6e8f1973bf0abc2aafd517992e8b6d88c3c46e69 \
  wan_animate_2/wan_animate_2_bf16_distillation.safetensors --local-dir ckpts
done_msg
