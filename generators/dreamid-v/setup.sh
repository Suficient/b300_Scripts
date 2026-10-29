#!/usr/bin/env bash
# DreamID-V: video face swap on Wan2.1-T2V-1.3B (DiT), DWPose variant (not Faster). Apache-2.0, but the README
# limits it to academic/demo use.
. "$(dirname "$0")/../common.sh"
clone https://github.com/bytedance/DreamID-V 9b589940577559c91481fb3a13bae000a55f97a1
venv
flash_attn   # before reqs: xfuser depends on it
reqs requirements.txt
install decord2   # imported as decord by dreamidv_wan, missing from requirements.txt
hf Wan-AI/Wan2.1-T2V-1.3B --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a --local-dir weights/Wan2.1-T2V-1.3B
hf XuGuo699/DreamID-V --revision 61dcef2a38a3d544ae6e6c7318724b856ae0c882 dreamidv.pth --local-dir weights
hf XuGuo699/DreamID-V --revision 61dcef2a38a3d544ae6e6c7318724b856ae0c882 dw-ll_ucoco_384.onnx yolox_l.onnx \
  --local-dir pose/models
done_msg
