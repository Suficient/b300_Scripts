#!/usr/bin/env bash
# Container entrypoint: one ComfyUI per GPU on ports 8188, 8189, ... (each restarted if it exits),
# then the batch runner. With no GPU visible it starts a single CPU worker, for testing the image.
set -uo pipefail

DATA_DIR="${DATA_DIR:-/data}"
COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
BASE_PORT="${COMFYUI_BASE_PORT:-8188}"
mkdir -p "$DATA_DIR"/{models,inputs/pairs4,outputs,runs,logs,work}

if [[ -n "${BFS_WORKERS:-}" ]]; then
    n="$BFS_WORKERS"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    n=$(nvidia-smi -L | grep -c '^GPU ')
else
    n=0
fi
mode_args="--highvram"
if (( n == 0 )); then
    echo "[start] no GPU visible: starting 1 CPU worker (testing only; start the container with --gpus all)"
    n=1
    mode_args="--cpu"
fi
echo "$n" > "$DATA_DIR/work/workers"
echo "[start] $n ComfyUI worker(s), ports ${BASE_PORT}-$((BASE_PORT + n - 1))"

worker() {
    local i=$1 port=$((BASE_PORT + $1)) dir="$DATA_DIR/work/gpu$1"
    mkdir -p "$dir"/{output,temp,user}
    while true; do
        echo "[start] gpu$i: starting ComfyUI on port $port (log: $DATA_DIR/logs/comfyui_gpu$i.log)"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$i python "$COMFY_DIR/main.py" \
            --listen 127.0.0.1 --port "$port" --disable-auto-launch \
            --extra-model-paths-config /opt/bfs/extra_model_paths.yaml \
            --input-directory "$DATA_DIR/inputs" --output-directory "$dir/output" \
            --temp-directory "$dir/temp" --user-directory "$dir/user" \
            $mode_args ${COMFYUI_ARGS:-} >> "$DATA_DIR/logs/comfyui_gpu$i.log" 2>&1
        echo "[start] gpu$i: ComfyUI exited with code $?, restarting in 10 s"
        sleep 10
    done
}

for ((i = 0; i < n; i++)); do
    worker "$i" &
done

while true; do
    python /opt/bfs/runner.py daemon
    echo "[start] runner exited with code $?, restarting in 10 s"
    sleep 10
done
