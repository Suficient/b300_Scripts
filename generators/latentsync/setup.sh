#!/usr/bin/env bash
# LatentSync 1.6: lip sync by SD-1.5 latent inpainting with Whisper audio features, 512x512 mouth region.
# Code Apache-2.0, weights OpenRAIL++.
. "$(dirname "$0")/../common.sh"
clone https://github.com/bytedance/LatentSync a229c3948406bc2cf6eaf4873e662e70c6a04746
venv
reqs requirements.txt
hf ByteDance/LatentSync-1.6 --revision c42c7e6c8e9c213626389fa7d9a3c444b8536353 whisper/tiny.pt latentsync_unet.pt \
  --local-dir checkpoints
hf stabilityai/sd-vae-ft-mse --revision 31f26fdeee1355a5c34592e401dd41e45d25a493 config.json \
  diffusion_pytorch_model.safetensors --local-dir checkpoints/sd-vae-ft-mse
# insightface fetches buffalo_l (detector + 106-point landmarks) into checkpoints/auxiliary/models on first use
"$PY" -c "from insightface.app import FaceAnalysis; FaceAnalysis(root='checkpoints/auxiliary')" > /dev/null
done_msg
