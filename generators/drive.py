#!/usr/bin/env python3
"""Run generators/<name>/batch.py workers (pairio) over a pair list on the B300 VM.

    drive.py start NAME [--gpus 0,1,..] [-k K] [--list FILE] [--limit N]   detached; returns at once
    drive.py status [NAME...]                                              done/failed/remaining, per worker
    drive.py stop NAME...
    drive.py smoke NAME --gpus G [--redo]                                   2 pairs in the foreground + checks

The list (default /data/inputs/select500.txt) holds pair names (resolved in /data/inputs/pairs4) or folder paths,
one per line. It is split round-robin into K workers per GPU; worker i gets pairs i, i+N, i+2N, ... Each worker is
  cd $GEN_ROOT/NAME && CUDA_VISIBLE_DEVICES=GPU .venv/bin/python generators/NAME/batch.py --pairs wNN.txt --out OUT
with env.sh's caches, the venv's nvidia/*/lib on LD_LIBRARY_PATH, MALLOC_MMAP_THRESHOLD_=4194304
MALLOC_ARENA_MAX=4 and PAIRIO_TMP=/data/gen_tmp/NAME/wNN (wiped when the worker starts). Outputs go to
$OUT_ROOT/NAME (default /data/gen_out/NAME); lists, logs and state to $OUT_ROOT/NAME/.drive. A supervisor
restarts a worker that dies (nonzero exit) up to --restarts times, moving the pair it was probably on to the end
of its list; pairio skips finished pairs, so `start` again retries failed ones.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN_ROOT = Path(os.environ.get("GEN_ROOT", "/data/gen"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/data/gen_out"))
TMP_ROOT = Path(os.environ.get("GEN_TMP", "/data/gen_tmp"))
PAIRS = Path("/data/inputs/pairs4")
LIST = Path("/data/inputs/select500.txt")
FAILED = re.compile(r" (\S+) FAILED: ")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def state_dir(name):
    return OUT_ROOT / name / ".drive"


def alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:  # zombie = dead
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def read_state(name):
    f = state_dir(name) / "state.json"
    return json.loads(f.read_text()) if f.exists() else None


def write_state(name, st):
    f = state_dir(name) / "state.json"
    f.with_suffix(".tmp").write_text(json.dumps(st, indent=1))
    f.with_suffix(".tmp").replace(f)


def pair_list(path, limit):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            p = Path(line.strip())
            out.append(str(p if p.is_absolute() else PAIRS / p))
    return out[:limit] if limit else out


def worker_env(name, gpu, tmp):
    """os.environ + env.sh + the worker's own settings."""
    out = subprocess.run(["bash", "-c", f'. "{HERE}/env.sh" && env -0'], check=True, capture_output=True,
                         env={**os.environ, "GEN_ROOT": str(GEN_ROOT)}).stdout
    env = dict(kv.split("=", 1) for kv in out.decode().split("\0") if "=" in kv)
    libs = sorted(str(p) for p in (GEN_ROOT / name / ".venv").glob("lib/python3*/site-packages/nvidia/*/lib"))
    env["LD_LIBRARY_PATH"] = ":".join(libs + [x for x in [env.get("LD_LIBRARY_PATH")] if x])
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MALLOC_MMAP_THRESHOLD_="4194304", MALLOC_ARENA_MAX="4",
               PAIRIO_TMP=str(tmp), PYTHONUNBUFFERED="1",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")  # old repos' checkpoints hold pickled objects
    # torch/OpenMP/BLAS default to every core (192); with ~20 workers that oversubscribes the CPU many times over
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(k, os.environ.get("GEN_THREADS", "8"))
    # attn_shim/sitecustomize.py: cuDNN attention instead of FlashAttention 2 (GEN_ATTN=fa2 to turn off)
    env["PYTHONPATH"] = ":".join([str(HERE / "attn_shim")] + [x for x in [env.get("PYTHONPATH")] if x])
    return env


def launch(name, w):
    """Start worker w (a state dict entry) in its generator's venv; returns the Popen."""
    sd, tmp = state_dir(name), TMP_ROOT / name / w["id"]
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    cmd = [str(GEN_ROOT / name / ".venv/bin/python"), str(HERE / name / "batch.py"),
           "--pairs", str(sd / f"{w['id']}.txt"), "--out", str(OUT_ROOT / name)]
    with open(sd / f"{w['id']}.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} drive: start on gpu {w['gpu']}\n")
        return subprocess.Popen(cmd, cwd=GEN_ROOT / name, env=worker_env(name, w["gpu"], tmp), stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def done(name, pair):
    return (OUT_ROOT / name / f"{Path(pair).name}.mp4").exists()


def supervise(name, restarts):
    """The detached process `start` leaves behind: runs the workers, restarts dead ones, exits when all end."""
    st = read_state(name)
    procs = {w["id"]: launch(name, w) for w in st["workers"]}
    for w in st["workers"]:
        w.update(pid=procs[w["id"]].pid, rc=None)
    write_state(name, st)
    while procs:
        time.sleep(5)
        for w in st["workers"]:
            p = procs.get(w["id"])
            if p is None or (rc := p.poll()) is None:
                continue
            del procs[w["id"]]
            w["rc"] = rc
            lst = state_dir(name) / f"{w['id']}.txt"
            pairs = lst.read_text().split()
            left = [x for x in pairs if not done(name, x)]
            if rc != 0 and left and w["restarts"] < restarts:
                pairs.remove(left[0])  # the pair it probably died on goes last
                lst.write_text("\n".join(pairs + left[:1]) + "\n")
                w["restarts"] += 1
                log(f"{w['id']} (gpu {w['gpu']}) died with {rc}, {len(left)} left; restart {w['restarts']}")
                procs[w["id"]] = launch(name, w)
                w.update(pid=procs[w["id"]].pid, rc=None)
            else:
                log(f"{w['id']} (gpu {w['gpu']}) exited with {rc}, {len(left)} not done")
            write_state(name, st)
    log("all workers ended")


def start(a):
    for name in a.names:
        st = read_state(name)
        if st and alive(st["supervisor"]):
            sys.exit(f"{name} is already running (supervisor {st['supervisor']}); stop it first")
        for f in (GEN_ROOT / name / ".venv/bin/python", HERE / name / "batch.py"):
            if not f.exists():
                sys.exit(f"{name}: no {f}")
        pairs = pair_list(a.list, a.limit)
        slots = [(g, j) for j in range(a.k) for g in a.gpus]
        n = min(len(slots), len(pairs))
        sd = state_dir(name)
        for f in sd.glob("w*.txt"):
            f.unlink()
        for f in sd.glob("w*.log"):
            f.replace(f.with_suffix(".log.prev"))
        sd.mkdir(parents=True, exist_ok=True)
        workers = []
        for i, (g, _) in enumerate(slots[:n]):
            w = {"id": f"w{i:02d}", "gpu": g, "pairs": len(pairs[i::n]), "restarts": 0}
            (sd / f"{w['id']}.txt").write_text("\n".join(pairs[i::n]) + "\n")
            workers.append(w)
        st = {"name": name, "list": str(a.list), "pairs": pairs, "gpus": a.gpus, "k": a.k, "workers": workers,
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_state(name, st)
        with open(sd / "supervisor.log", "a") as f:
            sup = subprocess.Popen([sys.executable, __file__, "_supervise", name, str(a.restarts)], stdout=f,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        st["supervisor"] = sup.pid
        write_state(name, st)
        print(f"{name}: {len(pairs)} pairs, {n} workers on gpus {','.join(map(str, a.gpus))}; "
              f"logs in {sd}; supervisor {sup.pid}")


def stop(a):
    for name in a.names:
        st = read_state(name)
        if not st or not alive(st.get("supervisor", 0)):
            print(f"{name}: not running")
            continue
        os.killpg(st["supervisor"], signal.SIGTERM)  # supervisor + workers + their children: one group
        for _ in range(30):
            if not any(alive(w.get("pid", 0)) for w in st["workers"]) and not alive(st["supervisor"]):
                break
            time.sleep(1)
        else:
            os.killpg(st["supervisor"], signal.SIGKILL)
        print(f"{name}: stopped")


def worker_stats(name, w):
    sd = state_dir(name)
    pairs = (sd / f"{w['id']}.txt").read_text().split()
    logf = sd / f"{w['id']}.log"
    text = logf.read_text(errors="replace") if logf.exists() else ""
    n_done = sum(done(name, p) for p in pairs)
    failed = {m for m in FAILED.findall(text)} & {Path(p).name for p in pairs if not done(name, p)}
    last = next((l for l in reversed(text.splitlines()) if l.strip()), "")
    return n_done, len(failed), len(pairs) - n_done - len(failed), last


def status(a):
    names = a.names or sorted(p.parent.parent.name for p in OUT_ROOT.glob("*/.drive/state.json"))
    for name in names:
        st = read_state(name)
        if not st:
            print(f"{name}: never started")
            continue
        run = "running" if alive(st.get("supervisor", 0)) else "not running"
        secs = [json.loads(f.read_text()).get("seconds", 0) for f in (OUT_ROOT / name).glob("[0-9]*.json")
                if f.with_suffix(".mp4").exists()]
        rows, tot = [], [0, 0, 0]
        for w in st["workers"]:
            s = worker_stats(name, w)
            tot = [t + x for t, x in zip(tot, s[:3])]
            state = "alive" if alive(w.get("pid", 0)) else "stopped" if w.get("rc") is None else f"exit {w['rc']}"
            rows.append(f"  {w['id']} gpu {w['gpu']}  {s[0]:4d} done {s[1]:3d} failed {s[2]:4d} left  {state:8s} "
                        f"restarts {w['restarts']}  | {s[3][-90:]}")
        mean = f", {sum(secs) / len(secs):.0f} s/pair per worker (mean of {len(secs)})" if secs else ""
        print(f"{name}: {run}, {tot[0]} done, {tot[1]} failed, {tot[2]} left of {len(st['pairs'])}{mean}")
        print("\n".join(rows))


def smoke(a):
    """2 pairs, one worker, in the foreground; then ffprobe each output and save a frame to look at."""
    name = a.names[0]
    pairs = pair_list(a.list, 2)
    out = OUT_ROOT / name
    if a.redo:
        for p in pairs:
            for ext in ("mp4", "json"):
                (out / f"{Path(p).name}.{ext}").unlink(missing_ok=True)
    sd = state_dir(name)
    sd.mkdir(parents=True, exist_ok=True)
    w = {"id": "smoke", "gpu": a.gpus[0]}
    (sd / "smoke.txt").write_text("\n".join(pairs) + "\n")
    log(f"{name} smoke on gpu {w['gpu']}: {' '.join(Path(p).name for p in pairs)}; log {sd / 'smoke.log'}")
    (sd / "smoke.log").unlink(missing_ok=True)
    p = launch(name, w)
    with open(sd / "smoke.log") as f:
        while True:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            elif p.poll() is not None:
                print(f.read(), end="")
                break
            else:
                time.sleep(0.5)
    ok = True
    for pair in pairs:
        n = Path(pair).name
        mp4 = out / f"{n}.mp4"
        if not mp4.exists():
            print(f"{n}: FAIL no {mp4}")
            ok = False
            continue
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                                          "-show_format", str(mp4)], capture_output=True, text=True).stdout)
        kinds = [s["codec_type"] for s in info["streams"]]
        v = next(s for s in info["streams"] if s["codec_type"] == "video") if "video" in kinds else {}
        dur = float(info["format"].get("duration", 0))
        good = "video" in kinds and "audio" in kinds and dur >= 4.9
        ok &= good
        frame = sd / f"smoke_{n}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "2.5", "-i", str(mp4), "-frames:v", "1", str(frame)])
        print(f"{n}: {'ok' if good else 'FAIL'}  {dur:.2f} s, streams {kinds}, {v.get('codec_name')} "
              f"{v.get('width')}x{v.get('height')} {v.get('pix_fmt')} {v.get('r_frame_rate')}; frame {frame}")
    sys.exit(0 if ok else 1)


def main():
    if sys.argv[1:2] == ["_supervise"]:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        return supervise(sys.argv[2], int(sys.argv[3]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start", "status", "stop", "smoke"])
    ap.add_argument("names", nargs="*", help="generator folder names")
    ap.add_argument("--gpus", type=lambda s: [int(x) for x in s.split(",")],
                    help="GPU ids, e.g. 3 or 0,1,2 (default for start: all)")
    ap.add_argument("-k", type=int, default=1, help="workers per GPU")
    ap.add_argument("--list", type=Path, default=LIST, help="pair names or folders, one per line")
    ap.add_argument("--limit", type=int, help="first N pairs of the list")
    ap.add_argument("--restarts", type=int, default=3, help="restarts per dead worker")
    ap.add_argument("--redo", action="store_true", help="smoke: delete the 2 pairs' outputs first")
    a = ap.parse_args()
    if a.cmd in ("start", "stop", "smoke") and not a.names:
        ap.error(f"{a.cmd} needs a generator name")
    if a.cmd == "smoke" and a.gpus is None:
        ap.error("smoke needs --gpus (the one GPU to use)")
    if a.gpus is None:
        a.gpus = list(range(len(subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines())))
    {"start": start, "status": status, "stop": stop, "smoke": smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
