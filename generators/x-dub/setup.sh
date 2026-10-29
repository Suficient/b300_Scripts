#!/usr/bin/env bash
# X-Dub (ICML 2026, Kling): mask-free lip sync by video editing, public release on Wan2.2-TI2V-5B. A's clip with
# B's audio.
. "$(dirname "$0")/../common.sh"
clone https://github.com/KlingAIResearch/X-Dub d4dac9603f80a35e525a121259096cd08994baa7
venv
SKIP="opencv-python" reqs requirements.txt
install opencv-python==4.11.0.86   # pinned 4.13 needs numpy 2; mmcv 2.1 needs numpy 1.26
mmlab 2.1.0 3.2.0 1.3.2
install -e . --no-deps
hf KlingTeam/X-Dub --revision c84e9612b7c1160f06ea609882e12efd83cacbbe --local-dir checkpoints \
  --exclude "wav2vec2-base-960h/*.h5" --exclude "wav2vec2-base-960h/*.bin"
mkdir -p dwpose_tools/models
if [ -d checkpoints/dwpose_tools/models ]; then
  cp -r checkpoints/dwpose_tools/models/. dwpose_tools/models/ && rm -rf checkpoints/dwpose_tools
fi
done_msg
