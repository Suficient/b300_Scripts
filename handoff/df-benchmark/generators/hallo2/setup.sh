#!/usr/bin/env bash
# Hallo2: audio-driven portrait animation on an SD-1.5 UNet with AnimateDiff motion modules, 512x512.
. "$(dirname "$0")/../common.sh"
clone https://github.com/fudan-generative-vision/hallo2
venv
SKIP="bitsandbytes|isort|pylint|pre-commit|mlflow" reqs requirements.txt
hf fudan-generative-ai/hallo2 --local-dir pretrained_models
done_msg
