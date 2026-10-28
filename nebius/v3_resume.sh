#!/usr/bin/env bash
# Finish the LTX-2.3 (BFS V3) run on a fresh VM: only the pairs without a result in nebius_out/full_v3.
#
#   set -x BFS_VM user@NEW_IP
#   ./v3_resume.sh plan     how many pairs are left (nothing is sent anywhere)
#   ./v3_resume.sh start    set up the VM (Docker, NVIDIA runtime, the LTX-2.3 image), upload the remaining
#                           pairs with their saved ACTION prompts (nebius_out/captions_v3), then run them
#                           detached: 8 pairs first as a check, then the rest. Results go to
#                           /data/outputs/full_v3, so ./vm.sh pull adds them to nebius_out/full_v3.
#   ./v3_resume.sh status | logs | stop
# Safe to run start again on the same VM: uploads and finished videos are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"
PAIRS="$PROJECT/pairs4"
DONE="$PROJECT/nebius_out/full_v3"
CAPS="$PROJECT/nebius_out/captions_v3"
SSH_OPTS=(-o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/bfs-%C" -o ControlPersist=30m)
vm() { ssh "${SSH_OPTS[@]}" "$VM" "$@"; }

remaining() {
    for d in "$PAIRS"/*/; do
        p=$(basename "$d")
        [[ -f "$d/a_guide.mp4" && ! -f "$DONE/$p.mp4" ]] && echo "$p"
    done
}

case "${1:-}" in
    plan)
        left=$(remaining | wc -l)
        echo "$left of $(ls -d "$PAIRS"/*/ | wc -l) pairs left; $(ls "$CAPS"/*/caption_v3.txt 2>/dev/null | wc -l) saved ACTION prompts"
        echo "at ~60 s per pair on 8 GPUs: ~$(( left * 60 / 8 / 60 )) min of generation, plus ~25 min of setup and model loading"
        ;;
    start)
        VM="${BFS_VM:?set BFS_VM to user@NEW_IP first (fish: set -x BFS_VM user@1.2.3.4)}"
        list=$(mktemp)
        remaining > "$list"
        n=$(wc -l < "$list")
        if (( n == 0 )); then echo "nothing left: every pair has a result in $DONE"; exit 0; fi
        echo "[resume] $n pairs left"

        vm mkdir -p bfs_v3   # opens the shared connection: the key passphrase is asked here, once
        scp "${SSH_OPTS[@]}" -q "$PROJECT/bfs_v3/run_pairs.py" "$PROJECT/bfs_v3/bfs_v3_headswap_api.json" "$VM:bfs_v3/"
        vm 'cat > bfs_v3/setup.sh' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
V3_IMAGE=squishydolphin/bfs-v3-comfyui:latest
SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
command -v rsync >/dev/null 2>&1 || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq rsync; }
$SUDO mkdir -p /data/inputs/pairs4 /data/outputs
$SUDO chown "$(id -u):$(id -g)" /data /data/inputs /data/inputs/pairs4
echo "[setup] GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
    || { echo "[setup] nvidia-smi failed: use the 'Ubuntu 24.04 LTS for NVIDIA GPUs (CUDA 13)' boot image"; exit 1; }
if systemctl list-unit-files 2>/dev/null | grep -q '^nvidia-fabricmanager'; then
    systemctl is-active --quiet nvidia-fabricmanager || $SUDO systemctl enable --now nvidia-fabricmanager
fi
command -v docker >/dev/null 2>&1 || { echo "[setup] installing Docker"; curl -fsSL https://get.docker.com | $SUDO sh; }
$SUDO systemctl enable --now docker >/dev/null 2>&1 || true
D="docker"; docker info >/dev/null 2>&1 || D="$SUDO docker"
echo "[setup] pulling $V3_IMAGE (~58 GB)"
if command -v script >/dev/null 2>&1; then script -qfec "$D pull $V3_IMAGE" /dev/null < /dev/null; else $D pull "$V3_IMAGE"; fi
if ! $D run --rm --gpus all "$V3_IMAGE" nvidia-smi -L >/dev/null 2>&1; then
    echo "[setup] installing the NVIDIA container toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq nvidia-container-toolkit
    $SUDO nvidia-ctk runtime configure --runtime=docker && $SUDO systemctl restart docker
fi
echo "[setup] GPUs inside a container: $($D run --rm --gpus all "$V3_IMAGE" nvidia-smi -L | grep -c '^GPU ')"
EOF
        vm 'cat > bfs_v3/resume.sh' <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
V3_IMAGE=squishydolphin/bfs-v3-comfyui:latest
OUT=/data/outputs/full_v3
D="docker"; docker info >/dev/null 2>&1 || D="sudo docker"
log() { echo "$(date '+%H:%M:%S') [resume] $*"; }

log "linking pair folders (caption.txt -> caption_v3.txt where it exists)"
sudo mkdir -p /data/v3 && sudo chown "$(id -u):$(id -g)" /data/v3
rm -rf /data/v3/full /data/v3/gate
mkdir -p /data/v3/full /data/v3/gate
n3=0
for d in /data/inputs/pairs4/*/; do
    p=$(basename "$d")
    [[ -f "$d/a_guide.mp4" && -f "$d/b_face_1024.png" ]] || continue
    mkdir -p "/data/v3/full/$p"
    ln -s "$d/a_guide.mp4" "/data/v3/full/$p/a_guide.mp4"
    ln -s "$d/b_face_1024.png" "/data/v3/full/$p/b_face_1024.png"
    if [[ -f "$d/caption_v3.txt" ]]; then ln -s "$d/caption_v3.txt" "/data/v3/full/$p/caption.txt"; n3=$((n3 + 1))
    else ln -s "$d/caption.txt" "/data/v3/full/$p/caption.txt"; fi
done
log "$(ls /data/v3/full | wc -l) pairs, $n3 with FACE + ACTION prompts"
sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"
gate=$(for p in $(ls /data/v3/full); do [[ -f "$OUT/$p.mp4" ]] || echo "$p"; done | head -n 8)
for p in $gate; do ln -s "/data/v3/full/$p" "/data/v3/gate/$p"; done

log "starting 8 LTX-2.3 containers"
for i in 0 1 2 3 4 5 6 7; do
    $D rm -f "bfs23-$i" >/dev/null 2>&1 || true
    sudo mkdir -p "/data/v3/ws$i"
    $D run -d --name "bfs23-$i" --gpus "device=$i" -p "127.0.0.1:$((8188 + i)):8188" \
        -e ENABLE_OLLAMA=0 -e SKIP_MODEL_DOWNLOAD=1 -v "/data/v3/ws$i:/workspace" "$V3_IMAGE" >/dev/null
    # CUDA 12.8's runtime compiler can't target the B300; 12.9's can
    $D exec "bfs23-$i" pip install -q --no-deps "nvidia-cuda-nvrtc-cu12==12.9.86" >/dev/null 2>&1 \
        || log "gpu$i: NVRTC update failed"
    $D restart "bfs23-$i" >/dev/null
done

if [[ -n "$gate" ]]; then
    log "check: $(echo $gate | wc -w) pairs first (the first job on each GPU also loads the models, ~13 min)"
    python3 "$HOME/bfs_v3/run_pairs.py" /data/v3/gate "$OUT" --gpus 8
    for p in $gate; do
        if [[ ! -f "$OUT/$p.mp4" ]]; then
            log "STOPPED: pair $p failed, so the rest was not started. See the errors above."
            exit 1
        fi
    done
fi
log "check OK; running all remaining pairs"
python3 "$HOME/bfs_v3/run_pairs.py" /data/v3/full "$OUT" --gpus 8
log "done: $(ls "$OUT" | grep -c '^[0-9]*\.mp4$') of $(ls /data/v3/full | wc -l) pairs on this VM have a result"
EOF
        vm 'chmod +x bfs_v3/setup.sh bfs_v3/resume.sh'
        # VM setup (image pull) and the input upload from this PC run at the same time
        vm 'bash bfs_v3/setup.sh' &
        setup_pid=$!
        echo "[resume] uploading $n pairs (guide clip, face, captions) from $PAIRS"
        sed 's|$|/a_guide.mp4|' "$list" > "$list.files"
        sed 's|$|/b_face_1024.png|' "$list" >> "$list.files"
        sed 's|$|/caption.txt|' "$list" >> "$list.files"
        until vm 'test -d /data/inputs/pairs4 && test -w /data/inputs/pairs4'; do   # setup.sh creates it first
            kill -0 "$setup_pid" 2>/dev/null || { echo "[resume] VM setup stopped early (see above)"; exit 1; }
            sleep 3
        done
        rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" --files-from="$list.files" \
            "$PAIRS/" "$VM:/data/inputs/pairs4/"
        if [[ -d "$CAPS" ]]; then
            sed 's|$|/caption_v3.txt|' "$list" > "$list.caps"
            sed 's|$|/action.txt|' "$list" >> "$list.caps"
            rsync -a --ignore-missing-args -e "ssh ${SSH_OPTS[*]}" --files-from="$list.caps" "$CAPS/" "$VM:/data/inputs/pairs4/"
        else
            echo "[resume] no $CAPS: these pairs will use their FACE-only caption.txt"
        fi
        rm -f "$list" "$list.files" "$list.caps"
        echo "[resume] upload done, waiting for the VM setup to finish"
        if ! wait "$setup_pid"; then echo "[resume] VM setup failed (see above); nothing was started"; exit 1; fi
        vm 'pkill -f "[b]fs_v3/resume[.]sh"; sleep 1; true'
        vm 'setsid nohup bash ~/bfs_v3/resume.sh > ~/bfs_v3/resume.log 2>&1 < /dev/null &
            sleep 3; cat ~/bfs_v3/resume.log; echo "started; log: ~/bfs_v3/resume.log on the VM"'
        echo "Follow it with ./v3_resume.sh logs (Ctrl-C only stops watching). Results: ./vm.sh pull"
        ;;
    status)
        VM="${BFS_VM:?set BFS_VM first}"
        vm 'tail -n 12 bfs_v3/resume.log; echo;
            echo "results on this VM: $(ls /data/outputs/full_v3 2>/dev/null | grep -c "^[0-9]*\.mp4$") of $(ls /data/v3/full 2>/dev/null | wc -l)"'
        ;;
    logs)
        VM="${BFS_VM:?set BFS_VM first}"
        vm tail -n 40 -f bfs_v3/resume.log
        ;;
    stop)
        VM="${BFS_VM:?set BFS_VM first}"
        vm 'D="docker"; docker info >/dev/null 2>&1 || D="sudo docker";
            pkill -f "[b]fs_v3/resume[.]sh"; pkill -f "[b]fs_v3/run_pairs[.]py";
            for i in 0 1 2 3 4 5 6 7; do $D rm -f "bfs23-$i" >/dev/null 2>&1; done;
            echo "stopped (finished videos stay on the disk)"'
        ;;
    *)
        sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
