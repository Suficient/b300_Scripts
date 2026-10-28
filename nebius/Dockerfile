# BFS head swap (LTX-2.5, bf16) batch image for an 8x B300 VM.
#
# One container runs one ComfyUI per GPU plus the batch runner (bfs/runner.py).
# Models are not in the image: `bfsctl models` downloads them to /data/models on the VM disk.
#
#   docker build -t squishydolphin/bfs-ltx25:b300 .

FROM nvidia/cuda:13.0.3-base-ubuntu24.04

# Pinned versions (2026-10-28). ComfyUI v0.43.0; node packs at their latest commits, as the BFS
# LTX-2.5 workflow asks for current ComfyUI and BFS nodes.
ARG COMFYUI_REF=426f5eea52a7ece1dd599fff4cb6cd1f352ddd24
ARG BFSNODES_REF=1e76045b19a99882f1331c9d322a26fac60ae9b8
ARG KJNODES_REF=2371136d2fa6fd0914e514cd99b36dabe5db9adf
ARG VHS_REF=4d907bee61e92c2e65af3bd6383a4e4d356126d1
# CUDA 13 wheels include Blackwell Ultra (B300). torchaudio stops at 2.11, so torch is held there too.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    COMFY_DIR=/opt/ComfyUI \
    DATA_DIR=/data \
    HF_HUB_DISABLE_TELEMETRY=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        git ca-certificates curl ffmpeg libgl1 libglib2.0-0 tini procps \
    && rm -rf /var/lib/apt/lists/*

# torch first, then hold it with a constraints file so no later requirements.txt can swap it
RUN python3 -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip wheel setuptools \
    && pip install --index-url "$TORCH_INDEX_URL" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
    && pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /opt/torch-constraints.txt \
    && cat /opt/torch-constraints.txt

# Shallow-fetch a repo at an exact commit and install its requirements without touching torch
RUN printf '%s\n' \
      '#!/usr/bin/env bash' \
      'set -euo pipefail' \
      'url="$1"; dest="$2"; ref="$3"' \
      'git init -q "$dest"' \
      'git -C "$dest" remote add origin "$url"' \
      'git -C "$dest" fetch -q --depth 1 origin "$ref"' \
      'git -C "$dest" checkout -q FETCH_HEAD' \
      'if [ -f "$dest/requirements.txt" ]; then pip install -r "$dest/requirements.txt" -c /opt/torch-constraints.txt; fi' \
      > /usr/local/bin/git-at && chmod +x /usr/local/bin/git-at

RUN git-at https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" "$COMFYUI_REF"

RUN cd "$COMFY_DIR/custom_nodes" \
    && git-at https://github.com/alisson-anjos/ComfyUI-BFSNodes.git ComfyUI-BFSNodes "$BFSNODES_REF" \
    && git-at https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI-KJNodes "$KJNODES_REF" \
    && git-at https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git ComfyUI-VideoHelperSuite "$VHS_REF"

# BFSNodes' pyproject lists deps that are not all in its requirements.txt; opencv-python and
# opencv-python-headless both get pulled in, so make sure the headless build wins.
RUN pip install -c /opt/torch-constraints.txt librosa scenedetect "transformers>=4.55.0" "huggingface_hub[hf_xet]>=0.34" \
    && pip uninstall -y opencv-python opencv-python-headless || true \
    && pip install -c /opt/torch-constraints.txt opencv-python-headless \
    && pip check || true \
    && python -c "import torch, torchvision, torchaudio, cv2; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'archs', torch._C._cuda_getArchFlags())"

COPY bfs/ /opt/bfs/
RUN chmod +x /opt/bfs/*.sh /opt/bfs/bfsctl && ln -s /opt/bfs/bfsctl /usr/local/bin/bfsctl

EXPOSE 8188
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/opt/bfs/start.sh"]
