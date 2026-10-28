#!/usr/bin/env bash
# PersonaLive (CVPR 2026): SD-1.5 UNet portrait animation from X-NeMo motion, built for live streaming.
. "$(dirname "$0")/../common.sh"
clone https://github.com/GVCLab/PersonaLive
venv
reqs requirements_base.txt
"$PY" tools/download_weights.py
done_msg
