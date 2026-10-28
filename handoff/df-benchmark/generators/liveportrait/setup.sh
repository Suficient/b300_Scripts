#!/usr/bin/env bash
# LivePortrait: implicit-keypoint warping (no diffusion), driven by the clip's face motion.
. "$(dirname "$0")/../common.sh"
clone https://github.com/KwaiVGI/LivePortrait
venv
reqs requirements_base.txt requirements.txt
hf KlingTeam/LivePortrait --local-dir pretrained_weights --include "liveportrait/*" "insightface/*"
done_msg
