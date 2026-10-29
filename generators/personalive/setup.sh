#!/usr/bin/env bash
# PersonaLive (CVPR 2026): SD-1.5 UNet portrait animation from X-NeMo-style motion, built for live streaming.
# No xformers (broken on Blackwell; diffusers' SDPA processors instead) and no TensorRT engine (offline path only).
. "$(dirname "$0")/../common.sh"
clone https://github.com/GVCLab/PersonaLive abdd112e01dcf7d89122c2e5efa29fcff0669740
venv
SKIP="numpy|polygraphy|fastapi|starlette|uvicorn.*|python-multipart|markdown2" reqs requirements_base.txt  # webcam UI/TRT
install "numpy==1.26.4" "jax<0.4.31" "jaxlib<0.4.31" "ml-dtypes<0.5" "opencv-contrib-python<4.11"  # numpy 1 (1.26.3 pin vs decord2 >= 1.26.4); mediapipe deps for numpy 1
# tools/download_weights.py's lambdalabs/sd-image-variations-diffusers was emptied (Oct 2026); lambda/ is the same model
hf lambda/sd-image-variations-diffusers --revision 42bc0ee1726b141d49f519a6ea02ccfbf073db2e \
  unet/config.json unet/diffusion_pytorch_model.bin image_encoder/config.json image_encoder/pytorch_model.bin \
  model_index.json --local-dir pretrained_weights/sd-image-variations-diffusers
hf stabilityai/sd-vae-ft-mse --revision 31f26fdeee1355a5c34592e401dd41e45d25a493 config.json diffusion_pytorch_model.bin \
  --local-dir pretrained_weights/sd-vae-ft-mse
hf huaichang/PersonaLive --revision b247581c0361b8bee0aa7232eea6e3fdd6e38e60 --local-dir . \
  --include "pretrained_weights/personalive/*.pth"
done_msg
