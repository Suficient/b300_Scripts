#!/usr/bin/env bash
# CanonSwap (ICCV 2025): face swap by warping in a canonical space (LivePortrait-based, no diffusion). Weights:
# Google Drive (ResearchRAIL-M, non-commercial). Pinned torch 2.3 cu118 upstream; runs on the platform torch.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Pixel-Talk/CanonSwap dd5100f6348edb4db32c1e5b3e99ce8b7d5f422a
venv
SKIP="gradio" reqs requirements.txt
install gdown
W=pretrained_weights M=pretrained_weights/insightface/models
mkdir -p $M
gd() { [ -f "$2" ] || .venv/bin/gdown -q "$1" -O "$2"; }
gd 1uDWiIam1jziU918iOZY2ATE2dw9aqYAr $W/combined_weights.pth
gd 1lDpbmvc7__cIfWU9rTTKNW5OXeeqohUJ $W/arcface_checkpoint.tar
gd 1uuee7ebWr9lBYfCmIPk8c4fk_YDbiNHz $W/landmark.onnx
[ -d $M/antelope ] || { gd 1yXQs6Nd0_hp97UGCvceyGehlCVTPs1ZD $W/antelope.zip && unzip -q -o $W/antelope.zip -d $M && rm $W/antelope.zip; }
[ -d $M/buffalo_l ] || { fetch https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip $W &&
  unzip -q -o $W/buffalo_l.zip 2d106det.onnx det_10g.onnx -d $M/buffalo_l && rm $W/buffalo_l.zip; }
hf jonathandinu/face-parsing --revision 758b82e15a0178c9db39c1ff666a8b56e3a550c8 config.json model.safetensors \
  preprocessor_config.json   # into the HF cache, where the pipeline loads it from
done_msg
