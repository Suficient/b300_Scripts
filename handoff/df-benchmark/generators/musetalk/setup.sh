#!/usr/bin/env bash
# MuseTalk 1.5: single-step latent inpainting of the mouth (real-time class), DWPose landmarks via OpenMMLab.
. "$(dirname "$0")/../common.sh"
clone https://github.com/TMElyralab/MuseTalk
venv
reqs requirements.txt
mmlab 2.1.0 3.2.0 1.3.2
install gdown
hf TMElyralab/MuseTalk musetalkV15/musetalk.json musetalkV15/unet.pth --local-dir models
hf stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin --local-dir models/sd-vae
hf openai/whisper-tiny config.json pytorch_model.bin preprocessor_config.json --local-dir models/whisper
hf yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir models/dwpose
mkdir -p models/face-parse-bisent
[ -f models/face-parse-bisent/79999_iter.pth ] ||
  .venv/bin/gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
curl -fsSL https://download.pytorch.org/models/resnet18-5c106cde.pth -o models/face-parse-bisent/resnet18-5c106cde.pth
done_msg
