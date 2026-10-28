#!/usr/bin/env bash
# Run the 10 test pairs through the LTX-2.3 image (BFS V3) on the VM, to compare with the LTX-2.5 test.
#
#   ./v3_test.sh run       stop the LTX-2.5 container, start 8 LTX-2.3 containers (one per GPU), run the pairs
#   ./v3_test.sh run --auto-prompt   same, but the image's Ollama (minicpm-v:8b) writes FACE + ACTION per pair
#                                    instead of caption.txt; results go to test_v3_auto
#   ./v3_test.sh page      write nebius_out/test/compare.html (run ./vm.sh pull first)
#   ./v3_test.sh restore   remove the LTX-2.3 containers and start the LTX-2.5 container again
#
# Results land in /data/outputs/test_v3 (or test_v3_auto) on the VM; ./vm.sh pull copies them to nebius_out/.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"
VM="${BFS_VM:?set BFS_VM to user@VM_IP first (fish: set -x BFS_VM user@1.2.3.4)}"
SSH_OPTS=(-o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new
          -o ControlMaster=auto -o "ControlPath=$HOME/.ssh/bfs-%C" -o ControlPersist=30m)
V3_IMAGE="squishydolphin/bfs-v3-comfyui:latest"
PAIRS="0530 0615 0723 1641 1745 1842 1982 2429 2746 2889"

vm() { ssh "${SSH_OPTS[@]}" "$VM" "$@"; }

case "${1:-}" in
    run)
        auto=0
        [[ "${2:-}" == "--auto-prompt" ]] && auto=1
        out=/data/outputs/test_v3
        (( auto )) && out=/data/outputs/test_v3_auto
        vm mkdir -p bfs_v3
        scp "${SSH_OPTS[@]}" -q "$PROJECT/bfs_v3/run_pairs.py" "$PROJECT/bfs_v3/bfs_v3_headswap_api.json" "$VM:bfs_v3/"
        vm "V3_IMAGE=$V3_IMAGE PAIRS='$PAIRS' AUTO=$auto OUT=$out bash -s" <<'EOF'
set -euo pipefail
D="docker"; docker info >/dev/null 2>&1 || D="sudo docker"
if ! $D image inspect "$V3_IMAGE" >/dev/null 2>&1; then
    echo "[v3] $V3_IMAGE is not on this VM, pulling it (~58 GB)"
    $D pull "$V3_IMAGE"
fi
mkdir -p /data/v3/pairs
for p in $PAIRS; do ln -sfn "/data/inputs/pairs4/$p" "/data/v3/pairs/$p"; done
echo "[v3] stopping the LTX-2.5 container (frees the GPUs and ports)"
$D stop bfs >/dev/null 2>&1 || true
for i in 0 1 2 3 4 5 6 7; do
    $D rm -f "bfs23-$i" >/dev/null 2>&1 || true
    sudo mkdir -p "/data/v3/ws$i"
    # ComfyUI listens on all interfaces inside the image, so publish it on localhost only
    $D run -d --name "bfs23-$i" --gpus "device=$i" -p "127.0.0.1:$((8188 + i)):8188" \
        -e ENABLE_OLLAMA="$AUTO" -e SKIP_MODEL_DOWNLOAD=1 -v "/data/v3/ws$i:/workspace" "$V3_IMAGE" >/dev/null
done
echo "[v3] 8 LTX-2.3 containers started on ports 8188-8195"
# The image's PyTorch (CUDA 12.8) compiles some kernels at run time with NVRTC 12.8, which cannot target
# the B300 (sm_103); NVRTC 12.9 can, and is a drop-in for the same CUDA 12 PyTorch.
echo "[v3] updating NVRTC to 12.9 in each container"
for i in 0 1 2 3 4 5 6 7; do
    $D exec "bfs23-$i" pip install -q --no-deps "nvidia-cuda-nvrtc-cu12==12.9.86" >/dev/null 2>&1 \
        || echo "[v3] gpu$i: NVRTC update failed"
    $D restart "bfs23-$i" >/dev/null
done
sudo mkdir -p "$OUT"
sudo chown "$(id -u):$(id -g)" "$OUT"
flags=""
(( AUTO )) && flags="--auto-prompt"
python3 ~/bfs_v3/run_pairs.py /data/v3/pairs "$OUT" --gpus 8 $flags
EOF
        echo "[v3_test] done. Copy results with ./vm.sh pull, then: ./v3_test.sh page"
        ;;
    restore)
        vm 'D="docker"; docker info >/dev/null 2>&1 || D="sudo docker";
            for i in 0 1 2 3 4 5 6 7; do $D rm -f "bfs23-$i" >/dev/null 2>&1 || true; done;
            $D start bfs >/dev/null && echo "[v3] LTX-2.3 containers removed, LTX-2.5 container started"'
        ;;
    page)
        T="$PROJECT/nebius_out/test"
        {
            echo '<!doctype html><meta charset="utf-8"><title>LTX-2.3 vs LTX-2.5</title>'
            echo '<style>body{font:14px sans-serif;margin:16px}td,th{vertical-align:top;padding:4px}img{width:160px}video{width:360px}</style>'
            echo '<h1>BFS: LTX-2.3 (V3) vs LTX-2.5 (v1.1)</h1><p>Same pairs, same captions. LTX-2.3 output is ~5 s at 24 fps, longest side 1024.</p>'
            auto_col=0
            [[ -d "$PROJECT/nebius_out/test_v3_auto" ]] && auto_col=1
            echo -n '<table><tr><th>pair</th><th>B face</th><th>A guide</th><th>LTX-2.3 V3 (caption.txt)</th>'
            (( auto_col )) && echo -n '<th>LTX-2.3 V3 (Ollama FACE + ACTION)</th>'
            full_col=0
            [[ -d "$PROJECT/nebius_out/full_v3" ]] && full_col=1
            (( full_col )) && echo -n '<th>LTX-2.3 V3 (FACE + Qwen ACTION)</th>'
            echo '<th>LTX-2.5 720p</th><th>LTX-2.5 1080p</th></tr>'
            for p in $PAIRS; do
                echo "<tr><th>$p</th><td><img src=\"inputs/${p}_face.png\"></td>"
                echo "<td><video src=\"inputs/${p}_guide.mp4\" controls loop muted></video></td>"
                echo "<td><video src=\"../test_v3/$p.mp4\" controls loop muted></video></td>"
                (( auto_col )) && echo "<td><video src=\"../test_v3_auto/$p.mp4\" controls loop muted></video></td>"
                (( full_col )) && echo "<td><video src=\"../full_v3/$p.mp4\" controls loop muted></video></td>"
                echo "<td><video src=\"${p}_720p.mp4\" controls loop muted></video></td>"
                echo "<td><video src=\"${p}_1080p.mp4\" controls loop muted></video></td></tr>"
            done
            echo '</table>'
        } > "$T/compare.html"
        echo "open $T/compare.html"
        ;;
    *)
        sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
        exit 2
        ;;
esac
