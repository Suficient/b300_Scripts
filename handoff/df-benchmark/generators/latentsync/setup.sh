#!/usr/bin/env bash
# LatentSync 1.6: lip sync by SD-1.5 latent inpainting with Whisper audio features, 512x512 mouth region.
. "$(dirname "$0")/../common.sh"
clone https://github.com/bytedance/LatentSync
venv
reqs requirements.txt
hf ByteDance/LatentSync-1.6 whisper/tiny.pt latentsync_unet.pt --local-dir checkpoints
done_msg
