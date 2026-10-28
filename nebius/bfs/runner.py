"""BFS batch runner: feeds pairs to one ComfyUI per GPU and collects the results.

State lives on /data so a stopped or preempted VM picks up where it left off:
  /data/control.json        the run to work on (written by `bfsctl start` / `bfsctl stop`)
  /data/inputs/pairs4/NNNN  a_guide.mp4, b_face_1024.png, caption.txt
  /data/outputs/...         finished videos (a file there means the job is done)
  /data/runs/<run>/jobs.jsonl  one line per attempt: time, worker, error
"""
import argparse
import html
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
import pipeline  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/data"))
INPUTS = DATA / "inputs" / "pairs4"
OUTPUTS = DATA / "outputs"
RUNS = DATA / "runs"
WORK = DATA / "work"
CONTROL = DATA / "control.json"
BASE_PORT = int(os.environ.get("COMFYUI_BASE_PORT", "8188"))
JOB_TIMEOUT = int(os.environ.get("BFS_JOB_TIMEOUT", "3600"))
MAX_ATTEMPTS = 3
GPU_PRICE = float(os.environ.get("BFS_GPU_PRICE", "0.99"))  # $ per GPU-hour, for estimates only

# 10 pairs for the 720p vs 1080p test: every frame size (1920, 1800 and 1440 x 1080), every frame rate
# in the batch (23.976 to 60), light to dark skin, both genders.
TEST_PAIRS = ["0530", "0615", "0723", "1641", "1745", "1842", "1982", "2429", "2746", "2889"]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(DATA / "logs" / "runner.log", "a") as fh:
        fh.write(line + "\n")


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def http(port, path, body=None, timeout=10):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def worker_count():
    try:
        return int((WORK / "workers").read_text())
    except (FileNotFoundError, ValueError):
        return 0


def all_pairs():
    return sorted(p.name for p in INPUTS.iterdir()
                  if (p / "a_guide.mp4").exists() and (p / "b_face_1024.png").exists() and (p / "caption.txt").exists()) \
        if INPUTS.exists() else []


# ---------------------------------------------------------------- runs

def run_spec(ctl):
    """Tasks of a control.json run: list of (pair, res, output path)."""
    if ctl["run"] == "test":
        return [(p, r, OUTPUTS / "test" / f"{p}_{r}p.mp4") for p in ctl["pairs"] for r in ctl["res"]]
    res = ctl["res"][0]
    return [(p, res, OUTPUTS / f"full_{res}p" / f"{p}.mp4") for p in ctl["pairs"]]


def attempts(run):
    out = []
    path = RUNS / run / "jobs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


class Scheduler:
    def __init__(self):
        self.lock = threading.Lock()
        self.ctl = None
        self.pending = []
        self.busy = set()

    def refresh(self):
        ctl = read_json(CONTROL)
        with self.lock:
            if not ctl or not ctl.get("run"):
                self.ctl, self.pending = None, []
                return
            fails = {}
            for a in attempts(ctl["run"]):
                if not a["ok"]:
                    fails[(a["pair"], a["res"])] = fails.get((a["pair"], a["res"]), 0) + 1
            self.ctl = ctl
            self.pending = [t for t in run_spec(ctl)
                            if not t[2].exists() and fails.get((t[0], t[1]), 0) < MAX_ATTEMPTS
                            and (t[0], t[1]) not in self.busy]

    def take(self):
        with self.lock:
            if not self.pending:
                return None, None
            task = self.pending.pop(0)
            self.busy.add(task[:2])
            return self.ctl["run"], task

    def done(self, task):
        with self.lock:
            self.busy.discard(task[:2])


def record(run, entry):
    (RUNS / run).mkdir(parents=True, exist_ok=True)
    with open(RUNS / run / "jobs.jsonl", "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def wait_ready(port, stop_after=None):
    t0 = time.time()
    while True:
        try:
            http(port, "/system_stats", timeout=5)
            return True
        except Exception:
            if stop_after and time.time() - t0 > stop_after:
                return False
            time.sleep(5)


def mux_audio(video, guide, out):
    """The generated video with the guide clip's original audio, cut to the video's length."""
    tmp = out.with_name(out.stem + ".part.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(guide),
                    "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-shortest", "-movflags", "+faststart",
                    str(tmp)], check=True)
    return tmp


def check_output(path, plan_):
    w, h, _, frames = pipeline.probe(path)
    if [w, h] != plan_["out"] or frames != plan_["length"]:
        raise RuntimeError(f"output is {w}x{h}, {frames} frames; expected {plan_['out']}, {plan_['length']} frames")


def run_job(worker, run, task):
    pair, res, out = task
    port = BASE_PORT + worker
    d = INPUTS / pair
    plan_ = pipeline.plan(d / "a_guide.mp4", res)
    prompt = pipeline.build(d / "a_guide.mp4", d / "b_face_1024.png", (d / "caption.txt").read_text().strip(),
                            plan_, int(pair), f"bfs/{run}/{pair}_{res}p")
    t0 = time.time()
    try:
        pid = http(port, "/prompt", {"prompt": prompt, "client_id": f"bfs-runner-{worker}"}, timeout=60)["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"prompt rejected: {e.read().decode()[:2000]}")
    unreachable_since = None
    while True:
        time.sleep(2)
        if time.time() - t0 > JOB_TIMEOUT:
            try:
                http(port, "/interrupt", {})
            except Exception:
                pass
            raise RuntimeError(f"timed out after {JOB_TIMEOUT} s")
        try:
            hist = http(port, f"/history/{pid}", timeout=10)
            unreachable_since = None
        except Exception:
            unreachable_since = unreachable_since or time.time()
            if time.time() - unreachable_since > 180:
                raise RuntimeError("ComfyUI stopped answering (crashed or restarted)")
            continue
        if pid in (hist or {}):
            entry = hist[pid]
            break
    status = entry.get("status", {})
    if status.get("status_str") != "success":
        msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
        detail = msgs[-1][1] if msgs else status
        raise RuntimeError(f"execution failed: {json.dumps(detail)[:3000]}")
    files = entry["outputs"].get(pipeline.OUTPUT_NODE, {}).get("gifs", [])
    if not files:
        raise RuntimeError("finished without an output video")
    f = files[-1]
    video = Path(f.get("fullpath") or WORK / f"gpu{worker}" / "output" / f["subfolder"] / f["filename"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = mux_audio(video, d / "a_guide.mp4", out)
    check_output(tmp, plan_)
    os.replace(tmp, out)
    video.unlink(missing_ok=True)
    try:
        http(port, "/history", {"delete": [pid]})
    except Exception:
        pass
    return time.time() - t0, plan_


def worker_loop(worker, sched, stop):
    port = BASE_PORT + worker
    first = True
    while not stop.is_set():
        run, task = sched.take()
        if task is None:
            time.sleep(5)
            continue
        try:
            if not wait_ready(port, stop_after=1800):
                raise RuntimeError(f"ComfyUI on port {port} not answering for 30 min")
            if shutil.disk_usage(DATA).free < 10e9:
                raise RuntimeError("less than 10 GB free on /data")
            sec, plan_ = run_job(worker, run, task)
            record(run, {"ts": time.time(), "pair": task[0], "res": task[1], "worker": worker, "ok": True,
                         "sec": round(sec, 1), "first_on_worker": first, "out": plan_["out"], "fps": plan_["fps"],
                         "length": plan_["length"]})
            log(f"gpu{worker} {run} {task[0]} {task[1]}p done in {sec:.0f} s")
            first = False
        except Exception as e:
            record(run, {"ts": time.time(), "pair": task[0], "res": task[1], "worker": worker, "ok": False,
                         "error": str(e)[:4000]})
            log(f"gpu{worker} {run} {task[0]} {task[1]}p FAILED: {str(e)[:500]}")
            if "not answering" in str(e) or "stopped answering" in str(e):
                first = True  # ComfyUI restarted, so its next job loads the models again
                time.sleep(60)
            elif "free on /data" in str(e):
                time.sleep(60)
        finally:
            sched.done(task)


def daemon():
    (DATA / "logs").mkdir(parents=True, exist_ok=True)
    n = worker_count()
    log(f"runner up, {n} ComfyUI workers")
    sched = Scheduler()
    stop = threading.Event()
    for w in range(n):
        threading.Thread(target=worker_loop, args=(w, sched, stop), daemon=True).start()
    last_status = 0
    while True:
        try:
            sched.refresh()
            if time.time() - last_status > 30:
                write_json(DATA / "status.json", status_dict())
                if sched.ctl and sched.ctl["run"] == "test":
                    write_test_page(sched.ctl)
                last_status = time.time()
        except Exception:
            log("runner error: " + traceback.format_exc())
        time.sleep(10)


# ---------------------------------------------------------------- status

def status_dict(price=GPU_PRICE):
    ctl = read_json(CONTROL) or {}
    out = {"control": ctl, "workers": worker_count(), "gpu_price": price}
    if not ctl.get("run"):
        return out
    tasks = run_spec(ctl)
    att = attempts(ctl["run"])
    per_res = {}
    for res in ctl["res"]:
        mine = [t for t in tasks if t[1] == res]
        ok = [a for a in att if a["ok"] and a["res"] == res]
        warm = [a["sec"] for a in ok if not a.get("first_on_worker")] or [a["sec"] for a in ok]
        fails = {}
        for a in att:
            if not a["ok"] and a["res"] == res:
                fails[a["pair"]] = fails.get(a["pair"], 0) + 1
        done = sum(1 for t in mine if t[2].exists())
        gave_up = [p for p, c in fails.items() if c >= MAX_ATTEMPTS and not (
            [t for t in mine if t[0] == p][0][2].exists())]
        info = {"total": len(mine), "done": done, "gave_up": gave_up,
                "failed_attempts": sum(fails.values()),
                "sec_per_job": round(statistics.median(warm), 1) if warm else None}
        if warm:
            gpus = max(out["workers"], 1)
            n_all = len(all_pairs()) or len(mine)
            full_h = n_all * statistics.median(warm) / 3600 / gpus
            info["full_run_estimate"] = {"pairs": n_all, "gpus": gpus, "hours": round(full_h, 1),
                                         "cost": round(full_h * gpus * price, 0)}
            left_h = (len(mine) - done - len(gave_up)) * statistics.median(warm) / 3600 / gpus
            info["remaining_hours"] = round(left_h, 2)
        per_res[str(res)] = info
    out["res"] = per_res
    open_tasks = {(t[0], t[1]) for t in tasks if not t[2].exists()}
    last_errors = [a for a in att if not a["ok"] and (a["pair"], a["res"]) in open_tasks][-3:]
    out["last_errors"] = [{"pair": a["pair"], "res": a["res"], "error": a["error"][:600]} for a in last_errors]
    return out


def print_status(price):
    s = status_dict(price)
    ctl = s["control"]
    up = []
    for w in range(s["workers"]):
        try:
            dev = http(BASE_PORT + w, "/system_stats", timeout=3)["devices"][0]["name"]
            up.append(dev)
        except Exception:
            pass
    print(f"workers: {len(up)}/{s['workers']} answering" + (f" ({up[0]})" if up else ""))
    if not ctl.get("run"):
        print("no run active. Start one with: bfsctl start test   or   bfsctl start full 720|1080")
        return
    print(f"run: {ctl['run']}  resolutions: {', '.join(f'{r}p' for r in ctl['res'])}  pairs: {len(ctl['pairs'])}")
    for res, i in s["res"].items():
        line = f"  {res}p: {i['done']}/{i['total']} done"
        if i["sec_per_job"]:
            line += f", {i['sec_per_job']:.0f} s per job on one GPU"
            line += f", ~{i['remaining_hours']:.1f} h left"
        if i["gave_up"]:
            line += f", gave up on {len(i['gave_up'])}: {' '.join(i['gave_up'][:20])}"
        print(line)
        if i.get("full_run_estimate"):
            e = i["full_run_estimate"]
            print(f"        all {e['pairs']} pairs at {res}p on {e['gpus']} GPUs: ~{e['hours']} h, ~${e['cost']:.0f}"
                  f" at ${price}/GPU-hour")
    for e in s.get("last_errors", []):
        print(f"  recent error {e['pair']} {e['res']}p: {e['error'][:300]}")


def write_test_page(ctl):
    """outputs/test/index.html: each test pair's face, guide clip and results side by side."""
    d = OUTPUTS / "test"
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    s = status_dict().get("res", {})
    rows = []
    for p in ctl["pairs"]:
        src = INPUTS / p
        for name, dst in (("a_guide.mp4", f"{p}_guide.mp4"), ("b_face_1024.png", f"{p}_face.png")):
            if (src / name).exists() and not (d / "inputs" / dst).exists():
                shutil.copy2(src / name, d / "inputs" / dst)
        cap = html.escape((src / "caption.txt").read_text().strip()) if (src / "caption.txt").exists() else ""
        cells = [f'<td><img src="inputs/{p}_face.png"></td>',
                 f'<td><video src="inputs/{p}_guide.mp4" controls loop muted></video></td>']
        for r in ctl["res"]:
            f = d / f"{p}_{r}p.mp4"
            cells.append(f'<td><video src="{f.name}" controls loop muted></video></td>' if f.exists()
                         else "<td>not done yet</td>")
        rows.append(f"<tr><th>{p}<br><small>{cap}</small></th>{''.join(cells)}</tr>")
    times = " · ".join(f"{r}p: {s[str(r)]['sec_per_job']:.0f} s/job" for r in ctl["res"]
                       if s.get(str(r), {}).get("sec_per_job"))
    heads = "".join(f"<th>{r}p</th>" for r in ctl["res"])
    (d / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>BFS 720p vs 1080p</title>
<style>body{{font:14px sans-serif;margin:16px}}td,th{{vertical-align:top;padding:4px}}img{{width:200px}}
video{{width:420px}}small{{font-weight:normal;color:#555}}</style>
<h1>BFS LTX-2.5: 720p vs 1080p</h1><p>{times}</p>
<p>Tip: the same seed is used at both resolutions, so differences come from the resolution.</p>
<table><tr><th>pair</th><th>B face</th><th>A guide</th>{heads}</tr>{''.join(rows)}</table>""")


# ---------------------------------------------------------------- self-check

def selfcheck():
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= good
        print(("  ok    " if good else "  FAIL  ") + msg)

    print("GPUs")
    smi = shutil.which("nvidia-smi")
    if smi:
        r = subprocess.run([smi, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True)
        gpus = [l for l in r.stdout.splitlines() if l.strip()]
        say(r.returncode == 0 and len(gpus) > 0, f"nvidia-smi sees {len(gpus)} GPU(s): {gpus[0] if gpus else r.stderr.strip()}")
    else:
        say(False, "nvidia-smi not found (container started without --gpus all?)")
    code = ("import torch,sys\nn=torch.cuda.device_count()\nprint(n, torch.version.cuda, torch._C._cuda_getArchFlags())\n"
            "for i in range(n):\n x=torch.randn(4096,4096,device=f'cuda:{i}',dtype=torch.bfloat16)\n"
            " y=(x@x).float().sum().item()\n q=torch.nn.functional.scaled_dot_product_attention("
            "*(torch.randn(1,8,256,128,device=f'cuda:{i}',dtype=torch.bfloat16),)*3)\n"
            " print(i, torch.cuda.get_device_name(i), 'sm_%d%d' % torch.cuda.get_device_capability(i), 'ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    lines = r.stdout.strip().splitlines()
    n = int(lines[0].split()[0]) if lines else 0
    say(r.returncode == 0 and n > 0, f"PyTorch CUDA: {lines[0] if lines else ''} {r.stderr.strip()[-600:] if r.returncode else ''}")
    for l in lines[1:]:
        print("        " + l)
    if r.returncode and "802" in r.stderr:
        print("        error 802 = NVSwitch fabric not ready: on the VM run `sudo systemctl restart nvidia-fabricmanager`")

    print("Models")
    for f, problem in models.status():
        say(problem is None, f"{Path(f[2]).name}" + (f": {problem}" if problem else ""))

    print("Inputs")
    pairs = all_pairs()
    say(len(pairs) > 0, f"{len(pairs)} pairs in {INPUTS}")
    missing = [p for p in TEST_PAIRS if p not in pairs]
    say(not missing, "test pairs present" + (f" (missing {missing})" if missing else ""))
    free = shutil.disk_usage(DATA).free / 1e9
    say(free > 50, f"{free:.0f} GB free on {DATA}")

    print("ComfyUI workers")
    n = worker_count()
    info = None
    for w in range(n):
        try:
            st = http(BASE_PORT + w, "/system_stats", timeout=5)
            dev = st["devices"][0]
            say(True, f"port {BASE_PORT + w}: {dev['name']}, {dev['vram_total'] / 2**30:.0f} GiB")
            if info is None:
                info = http(BASE_PORT + w, "/object_info", timeout=60)
        except Exception as e:
            say(False, f"port {BASE_PORT + w}: {e}")
    if info and pairs:
        d = INPUTS / pairs[0]
        for res in (720, 1080):
            plan_ = pipeline.plan(d / "a_guide.mp4", res)
            prompt = pipeline.build(d / "a_guide.mp4", d / "b_face_1024.png", "head_swap:\nFACE: test", plan_, 1, "x")
            problems = pipeline.check_against(info, prompt)
            say(not problems, f"workflow matches the installed nodes ({res}p plan for {pairs[0]}: {plan_})"
                + ("".join("\n          " + p for p in problems)))
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return ok


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(prog="bfsctl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon")
    m = sub.add_parser("models", help="download and verify the model files (needs HF_TOKEN)")
    m.add_argument("--check", action="store_true", help="only report what is there")
    sub.add_parser("selfcheck")
    s = sub.add_parser("start", help="start a run: `start test` or `start full 720|1080`")
    s.add_argument("kind", choices=["test", "full"])
    s.add_argument("res", nargs="?", type=int, choices=[720, 1080])
    s.add_argument("--pairs", help="comma-separated pair ids (default: test set, or every pair for full)")
    sub.add_parser("stop", help="finish the jobs in progress, then idle")
    st = sub.add_parser("status")
    st.add_argument("--price", type=float, default=GPU_PRICE, help="$ per GPU-hour for estimates")
    p = sub.add_parser("plan", help="show size, fps and frame count a pair would get")
    p.add_argument("pair")
    p.add_argument("res", type=int, choices=[720, 1080])
    a = ap.parse_args()

    if a.cmd == "daemon":
        daemon()
    elif a.cmd == "models":
        if a.check:
            for f, problem in models.status():
                print(f"{'ok' if problem is None else problem:>14}  {Path(f[2]).name}")
        else:
            models.download()
    elif a.cmd == "selfcheck":
        sys.exit(0 if selfcheck() else 1)
    elif a.cmd == "start":
        pairs = a.pairs.split(",") if a.pairs else (TEST_PAIRS if a.kind == "test" else all_pairs())
        have = set(all_pairs())
        missing = [p for p in pairs if p not in have]
        if missing:
            sys.exit(f"missing inputs for pairs: {' '.join(missing[:20])}")
        if a.kind == "full" and not a.res:
            sys.exit("say which resolution: bfsctl start full 720   or   bfsctl start full 1080")
        bad = [f for f, problem in models.status() if problem]
        if bad:
            sys.exit("models not ready, run `bfsctl models` first: " + ", ".join(Path(f[2]).name for f in bad))
        ctl = {"run": "test" if a.kind == "test" else f"full_{a.res}p",
               "res": [720, 1080] if a.kind == "test" else [a.res], "pairs": pairs, "started": time.time()}
        write_json(CONTROL, ctl)
        print(f"started {ctl['run']}: {len(pairs)} pairs at {', '.join(f'{r}p' for r in ctl['res'])}. "
              f"Workers pick it up within 10 s; check with `bfsctl status`.")
    elif a.cmd == "stop":
        write_json(CONTROL, {"run": None})
        print("stopped: jobs in progress finish, nothing new starts")
    elif a.cmd == "status":
        print_status(a.price)
    elif a.cmd == "plan":
        print(json.dumps(pipeline.plan(INPUTS / a.pair / "a_guide.mp4", a.res), indent=1))


if __name__ == "__main__":
    main()
