#!/usr/bin/env bash
# Runs on the Nebius VM (vm.sh copies it there). Safe to run again.
#   vm_setup.sh prepare   create /data for the upload
#   vm_setup.sh install   check GPUs, Docker and the NVIDIA runtime, pull the image, download the models
#                         (HF_TOKEN on stdin), start the `bfs` container, run the self-check
set -euo pipefail

IMAGE="${IMAGE:-squishydolphin/bfs-ltx25:b300}"
DATA=/data
SUDO=""
[[ $EUID -ne 0 ]] && SUDO="sudo"

prepare() {
    command -v rsync >/dev/null 2>&1 || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq rsync; }
    $SUDO mkdir -p "$DATA/inputs/pairs4" "$DATA/outputs"
    $SUDO chown "$(id -u):$(id -g)" "$DATA" "$DATA/inputs" "$DATA/inputs/pairs4"
    echo "[setup] $DATA ready, $(df -h --output=avail "$DATA" | tail -1 | tr -d ' ') free"
}

docker_cmd() {
    if docker info >/dev/null 2>&1; then echo docker; else echo "$SUDO docker"; fi
}

install_container_toolkit() {
    echo "[setup] installing the NVIDIA container toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq nvidia-container-toolkit
    $SUDO nvidia-ctk runtime configure --runtime=docker
    $SUDO systemctl restart docker
}

install() {
    local token
    token="$(tr -d '[:space:]')"   # a pasted token can carry a stray CR or space
    if [[ "$token" != hf_* ]]; then
        echo "[setup] the Hugging Face token should start with hf_ (got ${#token} characters)"
        exit 1
    fi
    prepare

    echo "[setup] GPUs"
    if ! nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; then
        echo "[setup] nvidia-smi failed: the VM needs the 'Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 13)' boot image"
        exit 1
    fi
    # 8-GPU HGX boards need the NVSwitch fabric manager before CUDA can start
    if systemctl list-unit-files 2>/dev/null | grep -q '^nvidia-fabricmanager'; then
        if ! systemctl is-active --quiet nvidia-fabricmanager; then
            echo "[setup] starting nvidia-fabricmanager"
            $SUDO systemctl enable --now nvidia-fabricmanager
        fi
    fi

    if ! command -v docker >/dev/null 2>&1; then
        echo "[setup] installing Docker"
        curl -fsSL https://get.docker.com | $SUDO sh
    fi
    $SUDO systemctl enable --now docker >/dev/null 2>&1 || true   # also brings the container back after a reboot
    local DOCKER
    DOCKER="$(docker_cmd)"

    echo "[setup] pulling $IMAGE"
    if command -v script >/dev/null 2>&1; then
        # a pseudo-terminal makes docker show its per-layer progress bars over ssh
        script -qfec "$DOCKER pull $IMAGE" /dev/null < /dev/null
    else
        $DOCKER pull "$IMAGE"
    fi

    if ! $DOCKER run --rm --gpus all "$IMAGE" nvidia-smi -L >/dev/null 2>&1; then
        install_container_toolkit
        DOCKER="$(docker_cmd)"
    fi
    echo "[setup] GPUs inside a container: $($DOCKER run --rm --gpus all "$IMAGE" nvidia-smi -L | grep -c '^GPU ')"

    echo "[setup] models -> $DATA/models (74.1 GB; progress every 30 s)"
    ( while sleep 30; do
        echo "[setup] models: $($SUDO du -sb "$DATA/models" 2>/dev/null | awk '{printf "%.1f", $1/1e9}') of 74.1 GB on disk"
      done ) &
    local monitor=$!
    if ! printf '%s' "$token" | $DOCKER run --rm -i -e HF_HUB_DISABLE_PROGRESS_BARS=1 -v "$DATA:$DATA" "$IMAGE" \
            bash -c 'HF_TOKEN="$(cat)" exec bfsctl models'; then
        kill "$monitor" 2>/dev/null
        exit 1
    fi
    kill "$monitor" 2>/dev/null

    echo "[setup] starting the bfs container (restarts by itself after a reboot or preemption)"
    $DOCKER rm -f bfs >/dev/null 2>&1 || true
    $DOCKER run -d --name bfs --restart unless-stopped --gpus all --network host --ipc host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -e BFS_GPU_PRICE="${BFS_GPU_PRICE:-0.99}" -e COMFYUI_ARGS="${COMFYUI_ARGS:-}" \
        -v "$DATA:$DATA" "$IMAGE" >/dev/null

    $SUDO tee /usr/local/bin/bfsctl >/dev/null <<'EOF'
#!/usr/bin/env bash
if docker info >/dev/null 2>&1; then exec docker exec -i bfs bfsctl "$@"; else exec sudo docker exec -i bfs bfsctl "$@"; fi
EOF
    $SUDO chmod +x /usr/local/bin/bfsctl

    echo "[setup] waiting for the ComfyUI workers"
    local n=0
    for _ in $(seq 1 60); do
        n=$($DOCKER exec bfs bash -c 'n=$(cat /data/work/workers 2>/dev/null || echo 0); up=0;
            for ((i=0;i<n;i++)); do curl -sf http://127.0.0.1:$((8188+i))/system_stats >/dev/null && up=$((up+1)); done;
            echo "$up/$n"' 2>/dev/null || echo "0/0")
        [[ "$n" != 0/* && "${n%/*}" == "${n#*/}" ]] && break
        sleep 5
    done
    echo "[setup] workers answering: $n"
    bfsctl selfcheck
}

case "${1:-}" in
    prepare) prepare ;;
    install) install ;;
    *) echo "usage: vm_setup.sh prepare|install" >&2; exit 2 ;;
esac
