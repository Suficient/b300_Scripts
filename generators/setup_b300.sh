#!/usr/bin/env bash
# Set up the B300 VM for the generators, then optionally install generators. Re-runnable: generators that
# finished are skipped, failed ones are retried. Doesn't touch docker, drivers, CUDA or the BFS containers.
#
#   ./setup_b300.sh                        system packages, uv, hf CLI, /data/gen layout
#   ./setup_b300.sh latentsync wav2lip     ... then these generators' setup.sh (logs in /data/gen/logs)
#   ./setup_b300.sh --force latentsync     reinstall even if it finished before
set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
export GEN_ROOT=$(readlink -m "${GEN_ROOT:-/data/gen}")
. "$HERE/env.sh"
LOGS=$GEN_ROOT/logs

force=; names=()
for a in "$@"; do
  case $a in
    --force) force=1 ;;
    -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 0 ;;
    *) [ -f "$HERE/$a/setup.sh" ] || { echo "unknown generator: $a" >&2; exit 2; }; names+=("$a") ;;
  esac
done

echo "== system packages"
pkgs=(git git-lfs curl rsync tmux ffmpeg build-essential cmake pkg-config libgl1 libglib2.0-0t64 libsndfile1)
missing=$(for p in "${pkgs[@]}"; do dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'ok installed' || echo "$p"; done)
if [ -n "$missing" ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y -qq --no-install-recommends $missing > /dev/null
fi
git lfs install --skip-repo > /dev/null

echo "== uv, hf"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
command -v hf > /dev/null || uv tool install -q "huggingface_hub[hf_xet]"
mkdir -p "$GEN_ROOT" "$LOGS" "$XDG_CACHE_HOME" /data/gen_out /data/gen_tmp
[ -x /usr/local/cuda/bin/nvcc ] || echo "warning: no /usr/local/cuda/bin/nvcc: flash_attn and mmlab can't build"
printf '%s, uv %s, %s, nvcc %s\n' "$(ffmpeg -version | head -1 | cut -d' ' -f1-3)" "$(uv --version | cut -d' ' -f2)" \
  "hf $(hf version 2>/dev/null | grep -o 'version: [0-9.]*' | cut -d' ' -f2)" "$(nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
df -h --output=avail "$GEN_ROOT" | tail -1 | xargs echo "free on $GEN_ROOT:"

declare -A status
for n in "${names[@]}"; do
  dir=$GEN_ROOT/$n
  if [ -f "$dir/.installed" ] && [ -z "$force" ]; then status[$n]="already installed"; continue; fi
  echo "== $n (log: $LOGS/$n.log)"
  t0=$SECONDS
  if bash "$HERE/$n/setup.sh" "$dir" > "$LOGS/$n.log" 2>&1; then
    touch "$dir/.installed"; status[$n]="ok ($(( (SECONDS - t0) / 60 )) min)"
  else
    status[$n]="FAILED after $(( (SECONDS - t0) / 60 )) min, see $LOGS/$n.log"; tail -n 5 "$LOGS/$n.log"
  fi
  echo "${status[$n]}"
done
failed=0
for n in "${names[@]}"; do
  printf '  %-24s %s\n' "$n" "${status[$n]}"; [[ ${status[$n]} != FAILED* ]] || failed=1
done
exit $failed
