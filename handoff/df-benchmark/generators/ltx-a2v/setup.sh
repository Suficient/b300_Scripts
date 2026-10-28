#!/usr/bin/env bash
# LTX-2.5 audio-to-video (two-stage, image-conditioned): a whole new video of B speaking the clip's audio.
# The weights are gated: accept the terms on huggingface.co/Lightricks/LTX-2.5 and `hf auth login` first.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Lightricks/LTX-2
uv sync --extra natten   # torch and natten from the cu132 indexes, which have ARM64 builds
M=models/ltx-2.5
hf Lightricks/LTX-2.5 diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors --local-dir $M
done_msg
