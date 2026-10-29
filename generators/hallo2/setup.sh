#!/usr/bin/env bash
# Hallo2: audio-driven portrait animation on an SD-1.5 UNet with AnimateDiff motion modules, 512x512 face crop.
# MIT code and weights (SD-1.5 parts CreativeML OpenRAIL-M).
. "$(dirname "$0")/../common.sh"
clone https://github.com/fudan-generative-vision/hallo2 58a9aa6c9f66817a6e084f3874cfc01ac24fed3e
venv
SKIP="bitsandbytes|isort|pylint|pre-commit|mlflow" reqs requirements.txt
# xformers (no Blackwell build) is imported unconditionally but only used when enabled, which inference never does
sed -i 's/^import xformers$/xformers = None/; /^import xformers.ops$/d' hallo/models/motion_module.py
hf fudan-generative-ai/hallo2 --revision c198a7586497c4f63d8b5bfb154b1d00c6b0fc46 --local-dir pretrained_models
done_msg
