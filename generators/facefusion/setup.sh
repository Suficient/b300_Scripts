#!/usr/bin/env bash
# FaceFusion 3.9.2: per-frame ONNX face swap, no torch. Shared by facefusion (HyperSwap 1A), facefusion-inswapper
# and facefusion-alphaface.
. "$(dirname "$0")/../common.sh"
clone https://github.com/facefusion/facefusion 3.9.2
venv 3.12 notorch   # numpy 2.5 / scipy 1.18 pins need Python >= 3.12; nothing here needs the 3.11 wheels
SKIP="gradio(-rangeslider)?" reqs requirements.txt   # gradio is the web UI only
"$PY" "$(dirname "$(readlink -f "$0")")/ff.py"  # models
done_msg
