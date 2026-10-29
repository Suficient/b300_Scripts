# Shared by every generators/*/setup.sh. Sourced, not run.
#
# Each generator gets its own clone and .venv in $DIR (default $GEN_ROOT/<name> = /data/gen/<name>; the setup.sh's
# first argument overrides). drive.py runs workers as $GEN_ROOT/<name>/.venv/bin/python with cwd $GEN_ROOT/<name>.
# Target: x86_64, 8x B300 (sm_103; torch and our builds carry sm_100 code, which runs on it), CUDA 13 driver,
# uv-managed Python 3.11 unless a repo needs another.

set -euo pipefail

NAME=$(basename "$(dirname "$(readlink -f "${BASH_SOURCE[1]}")")")
DIR=$(readlink -m "${1:-${GEN_ROOT:-/data/gen}/$NAME}")
GEN_ROOT=$(dirname "$DIR")
PY="$DIR/.venv/bin/python"
TORCH_INDEX=https://download.pytorch.org/whl/cu130
TORCH_PINS=(torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130)   # newest torch with torchaudio
# The CUDA 13 onnxruntime-gpu wheels (1.25+) have no sm_100 code or PTX ("no kernel image" on the B300); 1.24.4
# (CUDA 12 + cuDNN 9) ships compute_90 PTX, which the driver JIT-compiles for sm_103.
ORT="onnxruntime-gpu[cuda,cudnn]==1.24.4"
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
export UV_HTTP_TIMEOUT=600 HF_XET_HIGH_PERFORMANCE=1

# clone URL [COMMIT]: the upstream repo into $DIR (LFS files not fetched; weights come from `hf`), then cd there.
clone() {
  [ -d "$DIR/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --recursive "$1" "$DIR"
  cd "$DIR"
  if [ -n "${2:-}" ]; then
    git fetch -q origin "$2" && git checkout -q "$2" && git submodule -q update --init --recursive
  fi
}

# reuse OTHER: this method runs from OTHER's install ($GEN_ROOT/<name> becomes a symlink to $GEN_ROOT/OTHER).
reuse() {
  [ -d "$GEN_ROOT/$1/.venv" ] || { echo "install $1 first" >&2; return 1; }
  [ -e "$DIR" ] || ln -s "$1" "$DIR"
  cd "$DIR"
}

# venv [PYTHON] [notorch]: .venv with Python 3.11 (or PYTHON) and torch/torchvision/torchaudio $TORCH_PINS
# (unless notorch).
venv() {
  [ -x "$PY" ] || uv venv -q --python "${1:-3.11}" "$DIR/.venv"
  [ "${2:-}" = notorch ] || uv pip install -q --python "$PY" "${TORCH_PINS[@]}" --index-url "$TORCH_INDEX"
  uv pip install -q --python "$PY" pip "setuptools<81" wheel ninja packaging   # 81 drops pkg_resources
}

# install PACKAGE...: install into the venv. Never replaces torch*: any pin of them is overridden by $TORCH_PINS.
# A CPU onnxruntime pulled in as a dependency is swapped for $ORT.
install() {
  _uv "$@"
  if uv pip show -q --python "$PY" onnxruntime >/dev/null 2>&1; then ort_gpu; fi
}
_uv() { uv pip install -q --python "$PY" --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH_INDEX" "$@" --override <(printf '%s\n' "${TORCH_PINS[@]}"); }

# reqs FILE...: requirements files minus what must not be replaced or has no working build here: torch*,
# xformers (pins drag torch; `install xformers==0.0.35 --no-deps` runs on sm_103), flash-attn (use flash_attn),
# decord (-> decord2, same import), onnxruntime* (-> $ORT), mediapipe (-> 0.10.14, the newest with the legacy
# mp.solutions API), -r/-e/-- lines, and anything matching $SKIP (regex).
reqs() {
  local f out="$DIR/.reqs-filtered.txt" ort=
  local drop="torch|torchvision|torchaudio|xformers|flash[-_]?attn|decord|onnxruntime(-gpu)?|mediapipe${SKIP:+|$SKIP}"
  : > "$out"
  for f in "$@"; do
    sed 's/#.*//' "$f" | grep -viE "^\s*($drop)\s*([<>=!~;[]|$)|^\s*-[-a-z]" | grep -v '^\s*$' >> "$out" || true
    if grep -qiE '^\s*decord\b' "$f"; then echo decord2 >> "$out"; fi
    if grep -qiE '^\s*mediapipe\b' "$f"; then echo "mediapipe==0.10.14" >> "$out"; fi
    if grep -qiE '^\s*onnxruntime' "$f"; then ort=1; fi
  done
  install -r "$out"
  if [ -n "$ort" ]; then ort_gpu; fi
}

# ort_gpu: $ORT and not the CPU onnxruntime (same module; dependencies such as insightface pull the CPU one in).
ort_gpu() {
  uv pip uninstall -q --python "$PY" onnxruntime 2>/dev/null || true
  _uv --reinstall-package onnxruntime-gpu "$ORT" nvidia-cublas
}

# Wheels built from source once, shared by every venv with the same Python and torch.
wheels() { echo "$GEN_ROOT/.wheels/$("$PY" -c 'import sys,torch; print(f"cp{sys.version_info[0]}{sys.version_info[1]}-torch{torch.__version__}")')"; }

# flash_attn: FlashAttention 2.8.3.post2 built for sm_100 (runs on the B300's sm_103). Built once (~6 min), then
# reused. No trustworthy prebuilt wheel exists for torch 2.11 + CUDA 13.
flash_attn() {
  local w whl; w=$(wheels)
  whl=$(ls "$w"/flash_attn-*.whl 2>/dev/null | head -1 || true)
  if [ -z "$whl" ]; then
    mkdir -p "$w"
    PATH="$DIR/.venv/bin:$PATH" FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="10.0" FLASH_ATTN_CUDA_ARCHS="100" NVCC_THREADS=2 \
      MAX_JOBS="${MAX_JOBS:-32}" "$PY" -m pip wheel -q --no-build-isolation --no-deps -w "$w" "flash-attn==2.8.3.post2"
    whl=$(ls "$w"/flash_attn-*.whl | head -1)
  fi
  install "$whl" einops
}

# mmlab MMCV MMDET MMPOSE: mmengine, mmcv (built with CUDA ops for sm_100), mmdet, mmpose, for the repos that
# detect landmarks with DWPose through OpenMMLab. mmcv is built once (~2 min), then reused.
mmlab() {
  local w whl; w=$(wheels)
  install "mmengine==0.10.7" chumpy==0.70 --no-build-isolation
  whl=$(ls "$w"/mmcv-"$1"-*.whl 2>/dev/null | head -1 || true)
  if [ -z "$whl" ]; then
    mkdir -p "$w"
    PATH="$DIR/.venv/bin:$PATH" MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="10.0" MAX_JOBS="${MAX_JOBS:-32}" \
      "$PY" -m pip wheel -q --no-build-isolation --no-deps -w "$w" "mmcv==$1"
    whl=$(ls "$w"/mmcv-"$1"-*.whl | head -1)
  fi
  install "$whl" "mmdet==$2" "mmpose==$3" cython
  # xtcocotools' wheel is built against numpy 1 and fails to import under numpy 2; built here it works with both
  install --reinstall-package xtcocotools --no-binary xtcocotools --no-build-isolation xtcocotools==1.14.3
}

# fetch URL DEST_DIR: download one file (resumable), for weights that aren't on Hugging Face.
fetch() {
  local f="$2/$(basename "$1")"
  [ -f "$f" ] && return
  mkdir -p "$2" && curl -fsSL -C - --retry 5 -o "$f.part" "$1" && mv "$f.part" "$f"
}

# hf ARGS...: `hf download ARGS...` with the hf CLI that setup_b300.sh installs (uv tool), whatever the venv has.
hf() { command hf download "$@"; }

# done_msg: last line of every setup.sh. Prints dependency conflicts (uv pip install doesn't keep earlier installs'
# constraints, e.g. a later package can lift protobuf past mediapipe's <5); fix them in the setup.sh.
done_msg() { uv pip check -q --python "$PY" || true; echo "$NAME installed in $DIR"; }
