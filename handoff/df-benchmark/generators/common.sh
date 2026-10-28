# Shared by every generators/*/setup.sh. Sourced, not run.
#
# Each generator gets its own clone and .venv under $DIR (default ~/df-gen/<name>, first argument overrides).
# Everything targets the DGX Spark: Linux ARM64, CUDA 13, GB10 (sm_121), Python 3.11 (the one version with ARM64
# CUDA wheels for torch, onnxruntime-gpu, mediapipe and decord2 alike) unless a repo needs newer.

set -euo pipefail

NAME=$(basename "$(dirname "$(readlink -f "${BASH_SOURCE[1]}")")")
DIR=$(readlink -m "${1:-$HOME/df-gen/$NAME}")
GEN_ROOT=$(dirname "$DIR")
WHEELS="$GEN_ROOT/.wheels"   # wheels built from source once, shared by all generators
PY="$DIR/.venv/bin/python"
TORCH_INDEX=https://download.pytorch.org/whl/cu130
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
export UV_HTTP_TIMEOUT=600 HF_XET_HIGH_PERFORMANCE=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}   # nvcc, for the flash-attn and mmcv builds
export PATH="$CUDA_HOME/bin:$PATH"

# clone URL [COMMIT]: the upstream repo into $DIR, then cd there.
clone() {
  [ -d "$DIR/.git" ] || git clone --recursive "$1" "$DIR"
  cd "$DIR"
  if [ -n "${2:-}" ]; then git fetch -q origin "$2" && git checkout -q "$2"; fi
}

# venv [PYTHON] [notorch]: .venv with Python 3.11 (or PYTHON) and the CUDA 13 builds of torch, torchvision,
# torchaudio (unless notorch).
venv() {
  [ -x "$PY" ] || uv venv -q --python "${1:-3.11}" "$DIR/.venv"
  [ "${2:-}" = notorch ] || uv pip install -q --python "$PY" torch torchvision torchaudio --index-url "$TORCH_INDEX"
  uv pip install -q --python "$PY" pip setuptools wheel ninja packaging
}

# install PACKAGE...: install into the venv. Never touches torch (the CUDA 13 builds stay).
install() { uv pip install -q --python "$PY" --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH_INDEX" "$@" --override <(printf 'torch\ntorchvision\ntorchaudio\n'); }

# reqs FILE...: requirements files minus what has no ARM64 build or must not be replaced: torch*, xformers,
# flash-attn, decord (-> decord2, same import), onnxruntime (-> onnxruntime-gpu), mediapipe pins (-> 0.10.14, the
# newest with the legacy solutions API and ARM64 wheels), -r/-e lines, and anything matching $SKIP (a regex).
reqs() {
  local f out="$DIR/.reqs-filtered.txt"
  local drop="torch|torchvision|torchaudio|xformers|flash[-_]?attn|decord|onnxruntime(-gpu)?|mediapipe${SKIP:+|$SKIP}"
  : > "$out"
  for f in "$@"; do
    sed 's/#.*//' "$f" | grep -viE "^\s*($drop)\b|^\s*-[-a-z]" | grep -v '^\s*$' >> "$out" || true
    if grep -qiE '^\s*decord\b' "$f"; then echo decord2 >> "$out"; fi
    if grep -qiE '^\s*mediapipe\b' "$f"; then echo "mediapipe==0.10.14" >> "$out"; fi
  done
  install -r "$out"
  ort_gpu
}

# ort_gpu: onnxruntime-gpu and not the CPU onnxruntime (both install the same module; dependencies such as
# audio-separator pull in the CPU one).
ort_gpu() {
  uv pip uninstall -q --python "$PY" onnxruntime 2>/dev/null || true
  install --reinstall-package onnxruntime-gpu "onnxruntime-gpu[cuda,cudnn]>=1.30"
}

# flash_attn: FlashAttention 2 for the GB10 (sm_120 code, which runs on sm_121). Built once into $WHEELS
# (about an hour on a Spark), then reused.
flash_attn() {
  local whl
  whl=$(ls "$WHEELS"/flash_attn-*-cp311-*.whl 2>/dev/null | head -1 || true)
  if [ -z "$whl" ]; then
    mkdir -p "$WHEELS"
    TORCH_CUDA_ARCH_LIST="12.0" FLASH_ATTN_CUDA_ARCHS="120" MAX_JOBS="${MAX_JOBS:-8}" \
      "$PY" -m pip wheel --no-build-isolation --no-deps -w "$WHEELS" "flash-attn==2.8.3"
    whl=$(ls "$WHEELS"/flash_attn-*-cp311-*.whl | head -1)
  fi
  install "$whl"
}

# mmlab MMCV MMDET MMPOSE: mmengine, mmcv (built with CUDA ops for sm_120), mmdet, mmpose, for the repos that
# detect landmarks with DWPose through OpenMMLab. mmcv is built once into $WHEELS.
mmlab() {
  local mmcv_version=$1 mmdet_version=$2 mmpose_version=$3 whl
  install "mmengine==0.10.7" chumpy==0.70 --no-build-isolation
  whl=$(ls "$WHEELS"/mmcv-"$mmcv_version"-cp311-*.whl 2>/dev/null | head -1 || true)
  if [ -z "$whl" ]; then
    mkdir -p "$WHEELS"
    MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS="${MAX_JOBS:-8}" \
      "$PY" -m pip wheel --no-build-isolation --no-deps -w "$WHEELS" "mmcv==$mmcv_version"
    whl=$(ls "$WHEELS"/mmcv-"$mmcv_version"-cp311-*.whl | head -1)
  fi
  install "$whl" "mmdet==$mmdet_version" "mmpose==$mmpose_version"
}

# fetch URL DEST_DIR: download one file (resumable), for weights whose repo layout doesn't match the target.
fetch() {
  local f="$2/$(basename "$1")"
  [ -f "$f" ] && return
  mkdir -p "$2" && curl -fL -C - --retry 5 -o "$f.part" "$1" && mv "$f.part" "$f"
}

# hf ARGS...: `hf download ARGS...`, run by uvx so the venv's huggingface_hub version doesn't matter.
hf() { uvx -q --from huggingface_hub hf download "$@"; }

done_msg() { echo "$NAME installed in $DIR"; }
