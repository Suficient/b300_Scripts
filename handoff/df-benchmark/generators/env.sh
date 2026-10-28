# Sourced by setup_spark.sh, common.sh and lib.sh with GEN_ROOT set. Points every cache and download location the
# generators use at $GEN_ROOT, so uninstall_spark.sh removes them with it. HF_HOME stays where it was, so an
# existing `hf auth login` is still found; only the Hugging Face caches move.
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export XDG_CACHE_HOME="$GEN_ROOT/.cache"   # uv, pip, torch hub, Playwright browsers, whisper, matplotlib
export HF_HUB_CACHE="$XDG_CACHE_HOME/huggingface/hub" HF_XET_CACHE="$XDG_CACHE_HOME/huggingface/xet" \
  HF_ASSETS_CACHE="$XDG_CACHE_HOME/huggingface/assets" TORCH_HOME="$XDG_CACHE_HOME/torch" \
  TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv" \
  MODELSCOPE_CACHE="$XDG_CACHE_HOME/modelscope" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" \
  UV_PYTHON_INSTALL_DIR="$GEN_ROOT/.python" UV_LINK_MODE=copy
