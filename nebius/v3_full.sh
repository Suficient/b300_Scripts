#!/usr/bin/env bash
# Full LTX-2.3 (BFS V3) run on the VM, with ACTION prompts from a local vision model first.
#
#   ./v3_full.sh start    launch on the VM, detached (survives your SSH dropping):
#                           1. Qwen3.5-27B on all 8 GPUs writes ACTION for every pair (caption_v3.txt)
#                           2. 8 LTX-2.3 containers run the 10 test pairs; if any fails, it stops there
#                           3. the remaining pairs; results in /data/outputs/full_v3 (./vm.sh pull copies them)
#   ./v3_full.sh status   progress
#   ./v3_full.sh logs     follow the log
#   ./v3_full.sh stop     stop everything
# Running start again resumes: written ACTIONs and finished videos are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"
VM="${BFS_VM:?set BFS_VM to user@VM_IP first (fish: set -x BFS_VM user@1.2.3.4)}"
SSH_OPTS=(-o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/bfs-%C" -o ControlPersist=30m)
vm() { ssh "${SSH_OPTS[@]}" "$VM" "$@"; }

case "${1:-}" in
    start)
        vm mkdir -p bfs_v3 bfs_actions
        scp "${SSH_OPTS[@]}" -q "$PROJECT/bfs_v3/run_pairs.py" "$PROJECT/bfs_v3/bfs_v3_headswap_api.json" "$VM:bfs_v3/"
        scp "${SSH_OPTS[@]}" -q "$HERE/actions/actions.py" "$VM:bfs_actions/"
        vm 'cat > bfs_v3/pipeline.sh' <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
V3_IMAGE=squishydolphin/bfs-v3-comfyui:latest
LTX25_IMAGE=squishydolphin/bfs-ltx25:b300
TEST="0530 0615 0723 1641 1745 1842 1982 2429 2746 2889"
OUT=/data/outputs/full_v3
D="docker"; docker info >/dev/null 2>&1 || D="sudo docker"
log() { echo "$(date '+%H:%M:%S') [pipeline] $*"; }

log "freeing the GPUs"
$D stop bfs >/dev/null 2>&1 || true
for i in 0 1 2 3 4 5 6 7; do $D rm -f "bfs23-$i" >/dev/null 2>&1 || true; done
$D rm -f bfs-actions >/dev/null 2>&1 || true

log "step 1: ACTION prompts with Qwen3.5-27B on all GPUs"
$D run --rm --name bfs-actions --gpus all --ipc host -v /data:/data -v "$HOME/bfs_actions:/opt/actions:ro" \
    "$LTX25_IMAGE" python /opt/actions/actions.py \
    || log "some ACTION prompts were not written; those pairs use their FACE-only caption.txt"

log "linking pair folders (caption.txt -> caption_v3.txt where it exists)"
sudo mkdir -p /data/v3 && sudo chown "$(id -u):$(id -g)" /data/v3
rm -rf /data/v3/full /data/v3/gate
mkdir -p /data/v3/full /data/v3/gate
n3=0
for d in /data/inputs/pairs4/*/; do
    p=$(basename "$d")
    [[ -f "$d/a_guide.mp4" ]] || continue
    mkdir -p "/data/v3/full/$p"
    ln -s "$d/a_guide.mp4" "/data/v3/full/$p/a_guide.mp4"
    ln -s "$d/b_face_1024.png" "/data/v3/full/$p/b_face_1024.png"
    if [[ -f "$d/caption_v3.txt" ]]; then ln -s "$d/caption_v3.txt" "/data/v3/full/$p/caption.txt"; n3=$((n3 + 1))
    else ln -s "$d/caption.txt" "/data/v3/full/$p/caption.txt"; fi
done
for p in $TEST; do ln -s "/data/v3/full/$p" "/data/v3/gate/$p"; done
log "$n3 pairs have FACE + ACTION prompts"

log "step 2: starting 8 LTX-2.3 containers"
for i in 0 1 2 3 4 5 6 7; do
    sudo mkdir -p "/data/v3/ws$i"
    $D run -d --name "bfs23-$i" --gpus "device=$i" -p "127.0.0.1:$((8188 + i)):8188" \
        -e ENABLE_OLLAMA=0 -e SKIP_MODEL_DOWNLOAD=1 -v "/data/v3/ws$i:/workspace" "$V3_IMAGE" >/dev/null
    # CUDA 12.8's runtime compiler can't target the B300; 12.9's can
    $D exec "bfs23-$i" pip install -q --no-deps "nvidia-cuda-nvrtc-cu12==12.9.86" >/dev/null 2>&1 \
        || log "gpu$i: NVRTC update failed"
    $D restart "bfs23-$i" >/dev/null
done
sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"

log "step 3: the 10 test pairs first"
python3 "$HOME/bfs_v3/run_pairs.py" /data/v3/gate "$OUT" --gpus 8
for p in $TEST; do
    if [[ ! -f "$OUT/$p.mp4" ]]; then
        log "STOPPED: test pair $p failed, so the full run was not started. See the errors above."
        exit 1
    fi
done
log "test pairs OK; step 4: all pairs"
python3 "$HOME/bfs_v3/run_pairs.py" /data/v3/full "$OUT" --gpus 8
log "done: $(ls "$OUT" | grep -vc _compare) of $(ls /data/v3/full | wc -l) pairs have a result"
EOF
        # kept in its own command: a pkill pattern must not match the shell running it
        vm 'pkill -f "[b]fs_v3/pipeline[.]sh"; sleep 1; true'
        vm 'chmod +x ~/bfs_v3/pipeline.sh
            setsid nohup bash ~/bfs_v3/pipeline.sh > ~/bfs_v3/pipeline.log 2>&1 < /dev/null &
            sleep 3; head -n 5 ~/bfs_v3/pipeline.log; echo "started; log: ~/bfs_v3/pipeline.log on the VM"'
        echo "Follow it with ./v3_full.sh logs (Ctrl-C only stops watching). Results: ./vm.sh pull"
        ;;
    status)
        vm 'tail -n 15 bfs_v3/pipeline.log; echo; echo "results: $(ls /data/outputs/full_v3 2>/dev/null | grep -c "^[0-9]*\.mp4$") of $(ls /data/inputs/pairs4 | wc -l)";
            echo "ACTION prompts: $(ls /data/inputs/pairs4/*/action.txt 2>/dev/null | wc -l)"'
        ;;
    logs) vm tail -n 40 -f bfs_v3/pipeline.log ;;
    stop)
        vm 'D="docker"; docker info >/dev/null 2>&1 || D="sudo docker";
            pkill -f "[b]fs_v3/pipeline.sh"; pkill -f "[b]fs_v3/run_pairs.py";
            $D rm -f bfs-actions >/dev/null 2>&1; for i in 0 1 2 3 4 5 6 7; do $D rm -f "bfs23-$i" >/dev/null 2>&1; done;
            echo "stopped (finished videos and prompts stay on the disk)"'
        ;;
    *)
        sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
