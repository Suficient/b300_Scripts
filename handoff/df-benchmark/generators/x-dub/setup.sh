#!/usr/bin/env bash
# X-Dub (ICML 2026, Kling): mask-free lip sync by video editing, public release on Wan2.2-TI2V-5B.
. "$(dirname "$0")/../common.sh"
clone https://github.com/KlingAIResearch/X-Dub
venv
reqs requirements.txt
mmlab 2.1.0 3.2.0 1.3.2
install -e . --no-deps
hf KlingTeam/X-Dub --local-dir checkpoints
mkdir -p dwpose_tools/models
if [ -d checkpoints/dwpose_tools/models ]; then
  cp -r checkpoints/dwpose_tools/models/. dwpose_tools/models/ && rm -rf checkpoints/dwpose_tools
fi
done_msg
