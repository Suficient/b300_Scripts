"""Run the BFS V3 head swap on a folder of pairs, using the bfs-v3-comfyui containers (one ComfyUI per GPU).

    python3 run_pairs.py PAIRS_DIR OUT_DIR [--gpus 8] [--port 8188] [--auto-prompt]

PAIRS_DIR holds pair folders (NNNN/a_guide.mp4, b_face_1024.png, caption.txt). Each GPU's ComfyUI
(ports 8188, 8189, ...) takes the next pair when it finishes one. OUT_DIR gets NNNN.mp4 (the result)
and NNNN_compare.mp4 (guide | result). Pairs with a result already in OUT_DIR are skipped, so running
again retries only the failed ones.

The workflow is the image's bfs_v3_headswap_api.json. The prompt is the pair's caption.txt; with
--auto-prompt the workflow's Ollama node writes it instead.
"""
import argparse
import copy
import json
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

WORKFLOW = Path(__file__).with_name("bfs_v3_headswap_api.json")
JOB_TIMEOUT = 1800


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def http(port, path, body=None, headers=None, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def api(port, path, obj=None):
    body = json.dumps(obj).encode() if obj is not None else None
    return json.loads(http(port, path, body, {"Content-Type": "application/json"}) or b"null")


def upload(port, src, name):
    """Put a file in ComfyUI's input folder."""
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
            f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n').encode() + src.read_bytes() + f'\r\n--{b}--\r\n'.encode()
    http(port, "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"})


def build(workflow, pair, auto_prompt):
    p = copy.deepcopy(workflow)
    p["1"]["inputs"]["video"] = f"{pair.name}_guide.mp4"
    p["2"]["inputs"]["image"] = f"{pair.name}_face.png"
    p["33"]["inputs"]["filename_prefix"] = f"BFS_V3/{pair.name}"
    p["35"]["inputs"]["filename_prefix"] = f"BFS_V3/{pair.name}_compare"
    if not auto_prompt:
        for node in ("7", "8", "9"):  # Ollama describer, trigger concat, prompt preview
            del p[node]
        p["15"]["inputs"]["text"] = (pair / "caption.txt").read_text().strip()
    return p


def run_pair(port, pair, out_dir, workflow, auto_prompt):
    t0 = time.time()
    upload(port, pair / "a_guide.mp4", f"{pair.name}_guide.mp4")
    upload(port, pair / "b_face_1024.png", f"{pair.name}_face.png")
    try:
        pid = api(port, "/prompt", {"prompt": build(workflow, pair, auto_prompt)})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"prompt rejected: {e.read().decode()[:2000]}")
    while True:
        time.sleep(5)
        if time.time() - t0 > JOB_TIMEOUT:
            api(port, "/interrupt", {})
            raise RuntimeError(f"timed out after {JOB_TIMEOUT} s")
        try:
            hist = api(port, f"/history/{pid}")
        except (urllib.error.URLError, OSError):
            continue  # ComfyUI restarting; the timeout ends the wait if the job is gone
        if pid in hist:
            entry = hist[pid]
            break
    status = entry["status"]
    if status["status_str"] != "success":
        errors = [m[1] for m in status["messages"] if m[0] == "execution_error"]
        raise RuntimeError(f"execution failed: {json.dumps(errors[-1] if errors else status)[:2000]}")
    # the result goes last: its presence marks the pair as done
    for node, suffix in (("35", "_compare"), ("33", "")):
        files = entry["outputs"].get(node, {}).get("gifs")
        if not files:
            raise RuntimeError(f"finished without a video from node {node}")
        f = files[-1]
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
        dest = out_dir / f"{pair.name}{suffix}.mp4"
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(http(port, f"/view?{q}", timeout=600))
        tmp.replace(dest)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--gpus", type=int, default=8, help="number of ComfyUI containers")
    ap.add_argument("--port", type=int, default=8188, help="port of the first one; the rest follow")
    ap.add_argument("--auto-prompt", action="store_true", help="Ollama writes the prompt instead of caption.txt")
    a = ap.parse_args()

    workflow = json.loads(WORKFLOW.read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(d for d in a.pairs_dir.iterdir() if (d / "a_guide.mp4").exists())
    todo = queue.Queue()
    for d in pairs:
        if not (a.out_dir / f"{d.name}.mp4").exists():
            todo.put(d)
    log(f"{todo.qsize()} of {len(pairs)} pairs to run on {a.gpus} GPUs")
    failed = []

    def worker(port):
        while True:
            try:
                api(port, "/system_stats")
                break
            except (urllib.error.URLError, OSError):
                log(f"port {port}: waiting for ComfyUI")
                time.sleep(30)
        while True:
            try:
                pair = todo.get_nowait()
            except queue.Empty:
                return
            try:
                sec = run_pair(port, pair, a.out_dir, workflow, a.auto_prompt)
                log(f"port {port}: {pair.name} done in {sec:.0f} s ({todo.qsize()} waiting)")
            except Exception as e:
                failed.append(pair.name)
                log(f"port {port}: {pair.name} FAILED: {e}")

    threads = [threading.Thread(target=worker, args=(a.port + i,)) for i in range(a.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"finished; {len(failed)} failed" + (f": {' '.join(sorted(failed))} (run again to retry)" if failed else ""))


if __name__ == "__main__":
    main()
