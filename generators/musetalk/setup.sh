#!/usr/bin/env bash
# MuseTalk 1.5: single-step latent inpainting of the mouth (real-time class), DWPose landmarks via OpenMMLab
# (mmcv 2.1.0 built for sm_100 by common.sh's mmlab). Pins torch 2.0.1 cu118 upstream; runs on the platform torch.
. "$(dirname "$0")/../common.sh"
clone https://github.com/TMElyralab/MuseTalk 0a89dec45a0192b824e3cf4daf96c239440c5ed8
venv
SKIP="tensorflow|tensorboard|gradio|numpy" reqs requirements.txt   # TF/gradio: training and web UI only
mmlab 2.1.0 3.2.0 1.3.2
install gdown "numpy<2"
hf TMElyralab/MuseTalk --revision 2bcb936e2fddb4d86db4c62fd45b387d0c061571 musetalkV15/musetalk.json musetalkV15/unet.pth \
  --local-dir models
hf stabilityai/sd-vae-ft-mse --revision 31f26fdeee1355a5c34592e401dd41e45d25a493 config.json diffusion_pytorch_model.bin \
  --local-dir models/sd-vae
hf openai/whisper-tiny --revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af config.json pytorch_model.bin \
  preprocessor_config.json --local-dir models/whisper
hf yzd-v/DWPose --revision 1a7144101628d69ee7a3768d1ee3a094070dc388 dw-ll_ucoco_384.pth --local-dir models/dwpose
mkdir -p models/face-parse-bisent
[ -f models/face-parse-bisent/79999_iter.pth ] ||
  .venv/bin/gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
fetch https://download.pytorch.org/models/resnet18-5c106cde.pth models/face-parse-bisent
done_msg
