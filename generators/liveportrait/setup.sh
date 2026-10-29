#!/usr/bin/env bash
# LivePortrait: implicit-keypoint warping (no diffusion), B's frame driven by A's face motion.
# No FasterLivePortrait / TensorRT.
. "$(dirname "$0")/../common.sh"
clone https://github.com/KlingAIResearch/LivePortrait 9b294b3d0536135442ea73cb01e6cb3ca7029dd3
venv
SKIP="gradio" reqs requirements_base.txt requirements.txt
hf KlingTeam/LivePortrait --revision 82a4fa6735ca58432b6ce39301b4b9ee066dea47 --local-dir pretrained_weights --include "liveportrait/*" --include "insightface/*"
done_msg
