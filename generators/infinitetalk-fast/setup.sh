#!/usr/bin/env bash
# InfiniteTalk image-to-video, the README's 4-step lightx2v path. Same install as infinitetalk.
. "$(dirname "$0")/../common.sh"
reuse infinitetalk
hf Kijai/WanVideo_comfy --revision 8260d429d19fd7a72304cad059160b95d843913f \
  Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors --local-dir weights
done_msg
