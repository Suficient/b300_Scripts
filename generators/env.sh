# Sourced by setup_b300.sh, common.sh and drive.py (for workers). Points every cache and download location at
# $GEN_ROOT (default /data/gen). HF_HOME stays where it was, so an `hf auth login` token is still found; only the
# Hugging Face caches move.
export GEN_ROOT="${GEN_ROOT:-/data/gen}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="$GEN_ROOT/.cache"   # uv, pip, torch hub, whisper, matplotlib, ...
export HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub" HF_XET_CACHE="$XDG_CACHE_HOME/huggingface/xet" \
  HF_ASSETS_CACHE="$XDG_CACHE_HOME/huggingface/assets" TORCH_HOME="$XDG_CACHE_HOME/torch" \
  TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv" \
  MODELSCOPE_CACHE="$XDG_CACHE_HOME/modelscope" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" \
  UV_PYTHON_INSTALL_DIR="$GEN_ROOT/.python" UV_PYTHON_PREFERENCE=only-managed \
  UV_LINK_MODE=hardlink   # venvs share torch & co. with the uv cache (same disk), ~6 GB saved per venv
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"   # nvcc for source builds and runtime JIT
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
