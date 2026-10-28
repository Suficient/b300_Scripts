#!/usr/bin/env bash
# FaceFusion: per-frame one-shot face swap (ONNX swapper models). No torch: everything runs in onnxruntime.
. "$(dirname "$0")/../common.sh"
clone https://github.com/facefusion/facefusion
venv 3.12 notorch   # its pins (numpy 2.4, scipy 1.18) need Python >= 3.12
reqs requirements.txt
"$PY" facefusion.py force-download --download-scope full
done_msg
