#!/usr/bin/env bash
# daVinci-MagiHuman (GAIR, 15B single-stream): video + speech from a reference image and text, 540p super-resolution
# stage. Apache-2.0. Off Hopper its DiT uses FlashAttention 2 (no FA3 / MagiAttention needed; MagiAttention is only
# for the 1080p stage). Needs gated google/t5gemma-9b-9b-ul2 and stabilityai/stable-audio-open-1.0 (HF token).
. "$(dirname "$0")/../common.sh"
clone https://github.com/GAIR-NLP/daVinci-MagiHuman 209209b7086eba2020c5439265221495a8357322
venv 3.12   # MagiCompiler requires Python >= 3.12 (flash_attn builds once for it)
flash_attn
install "git+https://github.com/SandAI-org/MagiCompiler@825847a386d9913fcc70046ef9403f7d74105928"
reqs requirements.txt
install --no-deps -r requirements-nodeps.txt
C=checkpoints
hf GAIR/daVinci-MagiHuman --revision 7fe95e50c11bd92bdadf94cd51dbec18427f8e0c --local-dir $C \
  --include "base/*" --include "540p_sr/*" --include "turbo_vae/*"
hf Wan-AI/Wan2.2-TI2V-5B --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e Wan2.2_VAE.pth --local-dir $C/Wan2.2-TI2V-5B
hf google/t5gemma-9b-9b-ul2 --revision 7071a2ace9c8353e12e723e5382cc7256ee41d9f --local-dir $C/t5gemma-9b-9b-ul2
hf stabilityai/stable-audio-open-1.0 --revision f21265c1e2710b3bd2386596943f0007f55f802e --local-dir $C/stable-audio-open-1.0 --exclude "*.ckpt"
done_msg
