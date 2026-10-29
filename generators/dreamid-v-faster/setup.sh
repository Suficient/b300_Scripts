#!/usr/bin/env bash
# DreamID-V Faster: the authors' faster model (no pose input, 16 steps). Same install as dreamid-v.
. "$(dirname "$0")/../common.sh"
reuse dreamid-v
hf XuGuo699/DreamID-V --revision 61dcef2a38a3d544ae6e6c7318724b856ae0c882 dreamidv_faster.pth --local-dir weights
done_msg
