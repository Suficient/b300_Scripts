#!/usr/bin/env bash
# DreamID-V: video face swap on Wan2.1-1.3B (DiT), DWPose variant.
. "$(dirname "$0")/../common.sh"
clone https://github.com/bytedance/DreamID-V
venv
flash_attn   # before reqs: xfuser depends on it
reqs requirements.txt
hf Wan-AI/Wan2.1-T2V-1.3B --local-dir weights/Wan2.1-T2V-1.3B
hf XuGuo699/DreamID-V dreamidv.pth --local-dir weights
hf XuGuo699/DreamID-V dw-ll_ucoco_384.onnx yolox_l.onnx --local-dir pose/models
done_msg
