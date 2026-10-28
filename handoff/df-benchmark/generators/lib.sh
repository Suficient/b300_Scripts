# Shared by every generators/*/run.sh. Sourced, not run. run.sh runs in the generator's install directory with
# these set by df-benchmark (all paths absolute):
#   DFB_VIDEO      the clip (person A)                 DFB_REF        B's face, square crop, 512 px
#   DFB_VIDEO_25   the clip at 25 fps                  DFB_REF_FRAME  the whole frame DFB_REF was cut from
#   DFB_AUDIO      the clip's audio, 16 kHz mono WAV   DFB_REF_AUDIO  B's audio, 16 kHz mono, clip length
#   DFB_DURATION   clip length in seconds              DFB_PROMPT     text prompt, for models that take one
#   DFB_SEED       seed                                DFB_TMP        scratch directory, deleted afterwards
#   DFB_OUT        where the fake goes (.mp4)
# Every model loads, runs and exits inside run.sh, so its time includes the model load.

set -euo pipefail
PY="$PWD/.venv/bin/python"
GEN_ROOT=$(dirname "$PWD")
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
# torch >= 2.6 loads checkpoints with weights_only=True; older repos' checkpoints hold pickled objects
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# CUDA/cuDNN libraries from the nvidia-* pip wheels, for onnxruntime-gpu in repos that don't load them first
export LD_LIBRARY_PATH="$("$PY" -c 'import glob,site; print(":".join(sorted(glob.glob(site.getsitepackages()[0]+"/nvidia/*/lib"))))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# finish VIDEO AUDIO: write DFB_OUT from the generator's video with AUDIO as its sound track, cut to the clip
# length. The video stream is copied, not re-encoded.
finish() {
  ffmpeg -v error -y -i "$1" -i "$2" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k \
    -t "$DFB_DURATION" "$DFB_OUT"
}

# frames SECONDS FPS STEP OFFSET: the largest frame count <= SECONDS*FPS of the form STEP*k + OFFSET
# (video DiTs want 4k+1 or 8k+1 frames); never more frames than the clip has.
frames() {
  python3 -c "import math,sys; s,f,st,o=map(float,sys.argv[1:]); print(int(math.floor((s*f-o)/st)*st+o))" "$@"
}

# json_job FILE KEY_IMAGE IMAGE: the {"prompt", KEY_IMAGE, "cond_audio": {"person1"}} input file that
# InfiniteTalk and LongCat take.
json_job() {
  python3 -c "import json,os,sys; json.dump({'prompt': os.environ['DFB_PROMPT'], sys.argv[2]: sys.argv[3],
    'cond_audio': {'person1': os.environ['DFB_AUDIO']}}, open(sys.argv[1], 'w'))" "$@"
}
