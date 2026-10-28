#!/usr/bin/env bash
# comfy.sh start|stop [COMFY_DIR]: the ComfyUI server bfs-ltx23 runs on (127.0.0.1:8188). Keep it stopped while
# benchmarking other generators: it holds its models in memory.
set -euo pipefail
cd "${2:-$HOME/df-gen/bfs-ltx23}"
GEN_ROOT=$(dirname "$PWD")
. "$(dirname "$(readlink -f "$0")")/../env.sh"
up() { curl -sf localhost:8188/system_stats >/dev/null; }
case "${1:-}" in
  start)
    up && exit 0
    nohup .venv/bin/python main.py --listen 127.0.0.1 --port 8188 > comfy.log 2>&1 &
    echo $! > comfy.pid
    for _ in $(seq 120); do up && exit 0; sleep 5; done
    echo "ComfyUI didn't answer within 10 min, see $PWD/comfy.log" >&2; exit 1 ;;
  stop)
    [ -f comfy.pid ] && kill "$(cat comfy.pid)" 2>/dev/null || true
    rm -f comfy.pid ;;
  *) echo "usage: $0 start|stop [COMFY_DIR]" >&2; exit 2 ;;
esac
