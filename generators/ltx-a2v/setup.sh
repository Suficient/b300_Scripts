#!/usr/bin/env bash
# LTX-2.5 audio-to-video (ltx_pipelines.a2vid_two_stage): a new video of B (first frame) speaking A's audio.
# LTX-2 Community License (free under $10M revenue; its AUP bans deepfakes of real people without consent).
# Gated: accept the terms on huggingface.co/Lightricks/LTX-2.5 and put a token in ~/.cache/huggingface/token.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Lightricks/LTX-2 e98f6d927d83c92ef2be6aec76c8440b19306e39
export UV_NO_CONFIG=1   # else uv takes the repo's pyproject [tool.uv] indexes, which hide the cu130 torch
venv
# the workspace packages on the platform torch, not uv sync's lockfile (torch 2.13 cu132); natten extra not used
install -e packages/ltx-core -e packages/ltx-pipelines "transformers>=5.8,<5.15"
M=models/ltx-2.5
hf Lightricks/LTX-2.5 --revision 49b093a39c19169a8a81721dcc541e07e945232f diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors vae/ltx-2.5-audio-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors --local-dir $M
done_msg
