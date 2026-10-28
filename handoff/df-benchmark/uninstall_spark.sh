#!/usr/bin/env bash
# Remove everything setup_spark.sh installed: $GEN_ROOT (generators, weights, caches, built wheels, logs), the
# files the install made in this repo (.venv, config.toml, the BFS workflow export), and what it added to the
# system: apt packages, uv, Ollama (or only the model it pulled, if Ollama was already there). What to remove
# comes from the manifest setup_spark.sh writes (~/.local/state/df-benchmark/manifest); anything that was on
# the Spark before the install stays. Shows the list and asks before removing anything.
#
#   ./uninstall_spark.sh          ask, then remove the install
#   ./uninstall_spark.sh --all    also the benchmark's data (clips/, work/, results/: faces) and this checkout
#   ./uninstall_spark.sh --yes    don't ask
set -euo pipefail

REPO=$(dirname "$(readlink -f "$0")")
STATE=${XDG_STATE_HOME:-$HOME/.local/state}/df-benchmark
MANIFEST=$STATE/manifest

yes=
all=
for a in "$@"; do
  case $a in
    --yes) yes=1 ;;
    --all) all=1 ;;
    -h|--help) awk 'NR > 1 && !/^#/ {exit} NR > 1' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

entries() { sed -n "s/^$1=//p" "$MANIFEST"; }
paths=()     # removed with rm -rf, as this user
packages=()  # apt packages to purge
ollama_installed=
ollama_model=
if [ -f "$MANIFEST" ]; then
  GEN_ROOT=$(entries gen_root | head -1)
  REPO=$(entries repo | head -1)
  paths+=("$GEN_ROOT")
  while read -r p; do [ -z "$p" ] || paths+=("$p"); done < <(entries created; entries runtime)
  while read -r p; do
    [ -n "$p" ] && dpkg -s "$p" > /dev/null 2>&1 && packages+=("$p")
  done < <(entries apt)
  grep -qx "ollama=installed" "$MANIFEST" && ollama_installed=1
  ollama_model=$(entries ollama_model | head -1)
else
  GEN_ROOT=$(readlink -m "${GEN_ROOT:-$HOME/df-gen}")
  echo "No install manifest ($MANIFEST). Removing $GEN_ROOT and this repo's .venv and BFS export only:"
  echo "apt packages, uv and Ollama can't be told apart from what was installed before, so they stay."
  paths+=("$GEN_ROOT" "$REPO/.venv" "$REPO/workflows/bfs_v3_api.json")
fi
[ -z "$all" ] || paths+=("$REPO/clips" "$REPO/work" "$REPO/results" "$REPO")

# drop paths that don't exist, and refuse anything that isn't a plain absolute path below / other than $HOME
existing=()
for p in "${paths[@]}"; do
  case $p in /|"$HOME"|"$HOME/"|"") echo "refusing to remove '$p'" >&2; exit 1 ;; /*) ;; *) continue ;; esac
  [ -e "$p" ] || [ -L "$p" ] && existing+=("$p")
done
paths=("${existing[@]+"${existing[@]}"}")

# apt: only if purging the recorded packages removes nothing else (something installed later may depend on them)
apt_extra=
if [ ${#packages[@]} -gt 0 ]; then
  would_purge=$(apt-get -s purge "${packages[@]}" 2>/dev/null | sed -n 's/^Purg \([^ ]*\).*/\1/p' | sort || true)
  apt_extra=$(comm -23 <(echo "$would_purge") <(printf '%s\n' "${packages[@]}" | sort) | tr '\n' ' ' || true)
  apt_extra=${apt_extra% }
fi

running=$(pgrep -af -- "$GEN_ROOT" | grep -v -e uninstall_spark -e pgrep || true)
if [ -n "$running" ]; then
  echo "Processes from $GEN_ROOT are still running; stop them first (ComfyUI: generators/bfs-ltx23/comfy.sh stop):"
  echo "$running"
  exit 1
fi

echo "Will remove:"
for p in "${paths[@]}"; do printf '  %-8s %s\n' "$(du -sh "$p" 2>/dev/null | cut -f1)" "$p"; done
if [ -n "$ollama_installed" ]; then
  echo "  Ollama: service, binary, libraries, its models in /usr/share/ollama, the ollama user and group (sudo)"
elif [ -n "$ollama_model" ]; then
  echo "  Ollama model $ollama_model (Ollama itself was there before and stays)"
fi
if [ ${#packages[@]} -gt 0 ]; then
  if [ -n "$apt_extra" ]; then
    echo "  apt packages: KEPT. Purging the ${#packages[@]} the install added would also remove packages installed"
    echo "  since, which depend on them: $apt_extra"
    echo "  To remove them anyway: sudo apt-get purge ${packages[*]}"
  else
    echo "  apt packages (sudo): ${packages[*]}"
  fi
fi
[ -f "$MANIFEST" ] && echo "  the install manifest $STATE"
[ -n "$all" ] || echo "Kept: clips/, work/, results/ and this checkout (--all removes them)."

if [ -z "$yes" ]; then
  read -r -p "Remove all of this? [y/N] " answer
  [[ $answer == [yY]* ]] || { echo "Nothing removed."; exit 1; }
fi

if [ -n "$ollama_installed" ] || [ -n "$ollama_model" ] || { [ ${#packages[@]} -gt 0 ] && [ -z "$apt_extra" ]; }; then
  sudo -v
fi
if [ -n "$ollama_installed" ]; then
  echo "removing Ollama"
  sudo systemctl stop ollama 2>/dev/null || true
  sudo systemctl disable ollama 2>/dev/null || true
  sudo rm -f /etc/systemd/system/ollama.service
  sudo systemctl daemon-reload
  for bin in /usr/local/bin/ollama /usr/bin/ollama /bin/ollama; do sudo rm -f "$bin"; done
  sudo rm -rf /usr/local/lib/ollama /usr/lib/ollama /usr/share/ollama
  sudo userdel ollama 2>/dev/null || true
  sudo groupdel ollama 2>/dev/null || true
elif [ -n "$ollama_model" ] && command -v ollama > /dev/null; then
  ollama rm "$ollama_model" || true
fi
if [ ${#packages[@]} -gt 0 ] && [ -z "$apt_extra" ]; then
  echo "purging ${#packages[@]} apt packages"
  sudo apt-get purge -y -qq "${packages[@]}"
fi
for p in "${paths[@]}"; do
  echo "removing $p"
  rm -rf -- "$p"
done
rm -rf -- "$STATE"
echo "Done."
