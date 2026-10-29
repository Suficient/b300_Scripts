#!/usr/bin/env bash
# Drive the BFS run on the Nebius VM from this PC.
#
#   set -x BFS_VM user@VM_IP        (fish; in bash: export BFS_VM=user@VM_IP)
#   ./vm.sh setup         upload the pair inputs, set up Docker + GPUs, download models, start, self-check
#   ./vm.sh test          10 pairs at 720p and at 1080p
#   ./vm.sh full 1080     every pair at 1080p (or: ./vm.sh full 720)
#   ./vm.sh status        progress, seconds per job, time and cost left
#   ./vm.sh pull          copy finished videos (BFS, and generators into nebius_out/gen/) every 2 min (Ctrl-C to stop)
#   ./vm.sh stop          finish the jobs in progress, start nothing new
#   ./vm.sh logs          follow the runner log
#   ./vm.sh ui            open GPU 0's ComfyUI at http://localhost:8188 (for debugging)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"
VM="${BFS_VM:?set BFS_VM to user@VM_IP first (fish: set -x BFS_VM user@1.2.3.4)}"
OUT="${BFS_OUT:-$PROJECT/nebius_out}"
# One shared connection: the key passphrase is asked once, not by every ssh/scp/rsync (setup runs two at once)
SSH_OPTS=(-o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/bfs-%C" -o ControlPersist=30m)

vm() { ssh "${SSH_OPTS[@]}" "$VM" "$@"; }

upload_inputs() {
    echo "[vm.sh] uploading pair inputs (guide clip, face, caption) from $PROJECT/pairs4"
    rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
        --include='*/' --include='a_guide.mp4' --include='b_face_1024.png' --include='caption.txt' \
        --include='job.json' --exclude='*' "$PROJECT/pairs4/" "$VM:/data/inputs/pairs4/"
}

case "${1:-}" in
    setup)
        token="${HF_TOKEN:-}"
        if [[ -z "$token" ]]; then
            read -rsp "Hugging Face token (hf_...): " token
            echo
            token="${token//[[:space:]]/}"
        fi
        scp "${SSH_OPTS[@]}" -q "$HERE/vm_setup.sh" "$VM:vm_setup.sh"
        vm bash vm_setup.sh prepare
        # model download on the VM and the input upload from here run at the same time
        printf '%s' "$token" | vm "BFS_GPU_PRICE=${BFS_GPU_PRICE:-0.99} COMFYUI_ARGS='${COMFYUI_ARGS:-}' bash vm_setup.sh install" &
        install_pid=$!
        upload_inputs
        echo "[vm.sh] upload done, waiting for the VM setup to finish"
        wait "$install_pid"
        vm bfsctl status
        ;;
    upload) upload_inputs ;;
    test) vm bfsctl start test ;;
    full)
        [[ "${2:-}" == 720 || "${2:-}" == 1080 ]] || { echo "usage: ./vm.sh full 720|1080" >&2; exit 2; }
        vm bfsctl start full "$2"
        ;;
    status) vm bfsctl status --price "${BFS_GPU_PRICE:-0.99}" ;;
    stop) vm bfsctl stop ;;
    logs) vm tail -n 50 -f /data/logs/runner.log ;;
    ui)
        echo "ComfyUI on GPU 0: http://localhost:8188  (Ctrl-C to close)"
        ssh "${SSH_OPTS[@]}" -N -L 8188:127.0.0.1:8188 "$VM"
        ;;
    pull)
        mkdir -p "$OUT"
        echo "[vm.sh] copying /data/outputs to $OUT and /data/gen_out to $OUT/gen every 2 min (Ctrl-C to stop)"
        while true; do
            if rsync -a --partial --exclude='*.part.mp4' --exclude='*.part' -e "ssh ${SSH_OPTS[*]} -o ConnectTimeout=20" \
                "$VM:/data/outputs/" "$OUT/" &&
               rsync -a --partial --exclude='*.part' -e "ssh ${SSH_OPTS[*]} -o ConnectTimeout=20" \
                "$VM:/data/gen_out/" "$OUT/gen/"; then
                echo "[vm.sh] $(date +%H:%M) $(find "$OUT" -name '*.mp4' -not -path '*/inputs/*' | wc -l) videos here"
            else
                echo "[vm.sh] $(date +%H:%M) VM not reachable, trying again in 2 min"
            fi
            sleep 120
        done
        ;;
    *)
        sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
