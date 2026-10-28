#!/usr/bin/env bash
# Set up df-benchmark and every generator on a fresh DGX Spark (DGX OS). Re-runnable: generators that finished
# are skipped, failed ones are retried.
#
#   ./setup_spark.sh                      everything
#   ./setup_spark.sh latentsync x-dub     only these
#   ./setup_spark.sh --force latentsync   reinstall even if it finished before
#
# Takes hours (~650 GB of downloads, FlashAttention and mmcv built from source), so run it in tmux. sudo is
# needed once, at the start: system packages, Ollama, Chromium's libraries. ltx-a2v needs HF_TOKEN (or an
# `hf auth login`) for an account that accepted the LTX-2.5 terms; without one it is skipped.
# Installs go to $GEN_ROOT (default ~/df-gen), logs to $GEN_ROOT/logs. What it adds outside $GEN_ROOT (apt
# packages, uv, Ollama, files in this repo) is recorded in ~/.local/state/df-benchmark/manifest, which
# uninstall_spark.sh reads.
set -euo pipefail

REPO=$(dirname "$(readlink -f "$0")")
GEN_ROOT=$(readlink -m "${GEN_ROOT:-$HOME/df-gen}")
LOGS=$GEN_ROOT/logs
# quick installs first (they prove the toolchain), then the FlashAttention group (the first one builds the
# wheel the rest reuse), the mmcv group, and the largest downloads
ORDER=(facefusion liveportrait latentsync wav2lip personalive hallo2 dreamid-v infinitetalk wan-animate
       longcat-avatar hunyuan-avatar musetalk x-dub ltx-a2v bfs-ltx23)

force=
names=()
for a in "$@"; do
  case $a in
    --force) force=1 ;;
    -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 0 ;;
    *) names+=("$a") ;;
  esac
done
[ ${#names[@]} -gt 0 ] || names=("${ORDER[@]}")
for n in "${names[@]}"; do
  [ -f "$REPO/generators/$n/setup.sh" ] || { echo "unknown generator: $n (known: ${ORDER[*]})" >&2; exit 2; }
done
want() { [[ " ${names[*]} " == *" $1 "* ]]; }
say() { printf '\n== %s\n' "$*"; }

STATE=${XDG_STATE_HOME:-$HOME/.local/state}/df-benchmark
MANIFEST=$STATE/manifest
# note LINE: add LINE to the manifest once. Lines: gen_root=, repo=, apt=PACKAGE, created=PATH (made by the
# install), runtime=PATH (absent before the first install, may be made later by generators or the benchmark),
# ollama=installed, ollama_model=NAME (pulled into an Ollama that was already installed).
note() { grep -qxF "$1" "$MANIFEST" 2>/dev/null || echo "$1" >> "$MANIFEST"; }
if [ ! -f "$MANIFEST" ]; then
  if [ -n "$(ls -A "$GEN_ROOT" 2>/dev/null)" ]; then
    echo "$GEN_ROOT exists and isn't empty; uninstall_spark.sh deletes it whole, so pick another GEN_ROOT" >&2
    exit 1
  fi
  mkdir -p "$STATE"
  note "gen_root=$GEN_ROOT"
  note "repo=$REPO"
  # places outside $GEN_ROOT that generators (hardcoded paths) or the benchmark write to; only the ones that
  # don't exist yet are recorded, so uninstall leaves anything that was already there
  for p in "$HOME/.insightface" "$HOME/.cache/gdown" "$HOME/.cache/uv" "${XDG_DATA_HOME:-$HOME/.local/share}/uv" \
           "${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}" "$REPO/models"; do
    [ -e "$p" ] || note "runtime=$p"
  done
fi
created_if_new() { [ -e "$1" ] || note "created=$1"; }

mkdir -p "$LOGS"
exec > >(tee -a "$LOGS/setup_spark.log") 2>&1
. "$REPO/generators/env.sh"   # caches and downloads go inside $GEN_ROOT

say "checks"
[ "$(uname -m)" = aarch64 ] || echo "warning: $(uname -m), not ARM64: the generator scripts target a DGX Spark"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "warning: nvidia-smi not found"
[ -x /usr/local/cuda/bin/nvcc ] || echo "warning: no nvcc in /usr/local/cuda: FlashAttention and mmcv can't be built"
free_gb=$(df --output=avail -BG "$GEN_ROOT" | tail -1 | tr -dc 0-9)
echo "$free_gb GB free in $GEN_ROOT"
[ "$free_gb" -ge 700 ] || echo "warning: everything needs ~650 GB"

say "system packages (sudo)"
sudo -v
packages() { dpkg-query -W -f='${Package}\n' 2>/dev/null | sort || true; }
before=$(packages)
sudo apt-get update -qq
sudo apt-get install -y -qq git curl ffmpeg build-essential cmake ninja-build pkg-config python3 \
  libgl1 libglib2.0-0 libsndfile1 > /dev/null
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv > /dev/null; then
  created_if_new "$HOME/.config/uv"   # the installer's receipt
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh   # leaves shell profiles alone
  note "created=$HOME/.local/bin/uv"
  note "created=$HOME/.local/bin/uvx"
fi
if want bfs-ltx23; then
  if ! command -v ollama > /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    note "ollama=installed"
  elif ! grep -qx "ollama=installed" "$MANIFEST" && ! grep -q '^qwen3.5:9b' <<< "$(ollama list 2>/dev/null || true)"; then
    note "ollama_model=qwen3.5:9b"   # pulled by bfs-ltx23's setup into an Ollama that was already here
  fi
  uvx -q --from playwright playwright install --with-deps chromium   # for the workflow export
fi
comm -13 <(echo "$before") <(packages) | while read -r p; do
  [ -z "$p" ] || note "apt=$p"
done

declare -A status
if want ltx-a2v && [ -z "${HF_TOKEN:-}" ] && ! uvx -q --from huggingface_hub hf auth whoami > /dev/null 2>&1; then
  status[ltx-a2v]="skipped: not logged in to Hugging Face (run \`uvx --from huggingface_hub hf auth login\`, accept the LTX-2.5 terms, rerun with ltx-a2v)"
fi

say "df-benchmark"
cd "$REPO"
created_if_new "$REPO/.venv"
uv sync -q
if [ ! -f config.toml ]; then
  note "created=$REPO/config.toml"
  sed "s#^gen_root = .*#gen_root = \"$GEN_ROOT\"#" config.example.toml > config.toml
  echo "wrote config.toml (check clipfarmer_db and clipfarmer_root)"
fi

want bfs-ltx23 && created_if_new "$REPO/workflows/bfs_v3_api.json"
for n in "${names[@]}"; do
  dir=$GEN_ROOT/$n
  [ -z "${status[$n]:-}" ] || continue
  if [ -f "$dir/.dfb-installed" ] && [ -z "$force" ]; then
    status[$n]="already installed"
    continue
  fi
  say "$n (log: $LOGS/$n.log)"
  t0=$SECONDS
  if bash "$REPO/generators/$n/setup.sh" "$dir" > "$LOGS/$n.log" 2>&1; then
    touch "$dir/.dfb-installed"
    status[$n]="ok ($(( (SECONDS - t0) / 60 )) min)"
  else
    status[$n]="FAILED after $(( (SECONDS - t0) / 60 )) min, see $LOGS/$n.log"
    tail -n 5 "$LOGS/$n.log"
  fi
  echo "${status[$n]}"
done

say "summary"
failed=0
for n in "${names[@]}"; do
  printf '  %-16s %s\n' "$n" "${status[$n]}"
  [[ ${status[$n]} != FAILED* ]] || failed=1
done
du -sh "$GEN_ROOT" 2>/dev/null || true
cat <<EOF

Next:
  1. Prepare the clips where Clipfarmer's DB and clips are (this Spark, or copy work/ over from that one):
       cd $REPO && uv run df-benchmark -c config.toml prepare
  2. Benchmark (stop Clipfarmer workers and the VLM first; they skew time and memory):
       uv run df-benchmark -c config.toml run latentsync wav2lip ...
       generators/bfs-ltx23/comfy.sh start && uv run df-benchmark -c config.toml run bfs-ltx23 \\
         && generators/bfs-ltx23/comfy.sh stop
  3. uv run df-benchmark -c config.toml report
EOF
exit $failed
