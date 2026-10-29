#!/usr/bin/env python3
# Pulls the probe's weights, code, metadata, results, dataset and raw generator outputs from the B300 VM to the Mac,
# in priority order over 4 parallel ssh streams. Big files go in 512 MiB chunks (ssh + dd) and are checked against the
# remote md5; directories go in ~1 GB rsync batches. Every unit retries after a dropped connection, and finished
# units are skipped on a rerun. Usage: VM=sufi@<VM IP> pull.py [DEST]   (default ~/vjepa_df_probe; progress in DEST/pull.log)
import hashlib, os, queue, subprocess, sys, threading, time

HOST = os.environ["VM"]
DEST = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/vjepa_df_probe")
SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3", "-c", "aes128-gcm@openssh.com"]
CHUNK = 512 << 20
CK = "/data/df_train/runs/video_classification_frozen/df-vitl384-2x16-cropfirst"
FILES = [("/data/df_train/runs/deploy_vitl384_head18.pt", "weights/deploy_vitl384_head18.pt"),   # (remote, local) in priority order
         (f"{CK}/latest.pt", "weights/final_ep20_all_heads.pt"),
         ("/data/vjepa2_ckpt/vjepa2_1_vitl_dist_vitG_384.pt", "weights/vjepa2_1_vitl_dist_vitG_384.pt")]
SMALL = [("/data/df_train/", "df_train/", ["--exclude=/clean/", "--exclude=/degraded/", "--exclude=*.pt", "--exclude=*.pt.*",
                                          "--exclude=/runs/video_classification_frozen/df-vitl384-2x16-smoke/"]),
         ("/data/vjepa2/", "vjepa2/", ["--exclude=/.venv/", "--exclude=__pycache__/"]),
         ("/tmp/vjtest/", "helpers/", ["--exclude=__pycache__/", "--exclude=*.png"])]
DIRS = [("/data/df_train/clean", "df_train/clean"), ("/data/df_train/degraded", "df_train/degraded"),
        ("/data/gen_out", "raw/gen_out"), ("/data/outputs", "raw/outputs")]

lock = threading.Lock()
def log(msg):
    with lock, open(f"{DEST}/pull.log", "a") as f: f.write(time.strftime("%H:%M:%S ") + msg + "\n")
def remote(cmd): return subprocess.run(SSH + [HOST, cmd], capture_output=True, text=True, check=True).stdout

def chunk(src, dst, i, n):
    part = f"{DEST}/{dst}.part{i:04d}"; want = min(CHUNK, n - i * CHUNK)
    if os.path.exists(part) and os.path.getsize(part) == want: return
    with open(part + ".tmp", "wb") as f:
        subprocess.run(SSH + [HOST, f"dd if={src} bs=1M skip={i * CHUNK >> 20} count={CHUNK >> 20} status=none"], stdout=f, check=True, timeout=1800)
    if os.path.getsize(part + ".tmp") != want: raise IOError(f"{part}: short read")
    os.replace(part + ".tmp", part)

def rsync(src, dst, extra, files=None):
    os.makedirs(f"{DEST}/{dst}", exist_ok=True)
    cmd = ["rsync", "-a", "--partial", "-e", " ".join(SSH)] + extra + (["--files-from=-"] if files else []) + [f"{HOST}:{src}", f"{DEST}/{dst}"]
    subprocess.run(cmd, input="\n".join(files) if files else None, text=True, check=True, timeout=7200, capture_output=True)

def worker(q):
    while True:
        try: name, fn = q.get_nowait()
        except queue.Empty: return
        for attempt in range(30):
            try: fn(); break
            except Exception as e:
                log(f"retry {attempt + 1} {name}: {str(e)[:200]}"); time.sleep(min(60, 5 * (attempt + 1)))
        else: log(f"GAVE UP {name}")
        with lock: done[0] += 1
        if done[0] % 20 == 0 or q.empty(): log(f"{done[0]}/{total} units done")

def assemble(src, dst, n):
    out = f"{DEST}/{dst}"; parts = [f"{out}.part{i:04d}" for i in range(-(-n // CHUNK))]
    if not all(os.path.exists(p) for p in parts): return log(f"NOT ASSEMBLED {dst}: chunks missing")
    md5 = hashlib.md5()
    with open(out + ".tmp", "wb") as f:
        for p in parts:
            with open(p, "rb") as g:
                while b := g.read(1 << 24): md5.update(b); f.write(b)
    want = remote(f"md5sum {src}").split()[0]
    if md5.hexdigest() != want: return log(f"MD5 MISMATCH {dst}: kept chunks, rerun to refetch")
    os.replace(out + ".tmp", out); [os.remove(p) for p in parts]; log(f"ok {dst} ({n / 1e9:.2f} GB, md5 {want})")

os.makedirs(f"{DEST}/weights", exist_ok=True); os.makedirs(f"{DEST}/env", exist_ok=True)
log(f"start -> {DEST}")
q = queue.Queue(); done = [0]
sizes = {s: int(remote(f"stat -c %s {s}")) for s, _ in FILES}
for s, d, ex in SMALL: q.put((f"rsync {s}", lambda s=s, d=d, ex=ex: rsync(s, d, ex)))
q.put(("env", lambda: open(f"{DEST}/env/vjepa2_git.txt", "w").write(remote(
    "cd /data/vjepa2 && git log -1 --format=%H && git status --short && git diff && /data/vjepa2/.venv/bin/python -c 'import importlib.metadata as m; print(chr(10).join(sorted(f\"{d.name}=={d.version}\" for d in m.distributions())))'"))))
for s, d in FILES:
    if os.path.exists(f"{DEST}/{d}") and os.path.getsize(f"{DEST}/{d}") == sizes[s]: continue
    for i in range(-(-sizes[s] // CHUNK)): q.put((f"{d} chunk {i}", lambda s=s, d=d, i=i: chunk(s, d, i, sizes[s])))
for s, d in DIRS:
    batch, b = [], 0
    for line in remote(f"cd {s} && find . -type f -printf '%s %P\\n' | sort -k2").splitlines():
        n, p = line.split(" ", 1); batch.append(p); b += int(n)
        if b > 1 << 30: q.put((f"{d} batch", lambda s=s, d=d, f=batch: rsync(s + "/", d + "/", [], f))); batch, b = [], 0
    if batch: q.put((f"{d} batch", lambda s=s, d=d, f=batch: rsync(s + "/", d + "/", [], f)))
total = q.qsize(); log(f"{total} units queued")
ts = [threading.Thread(target=worker, args=(q,)) for _ in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
for s, d in FILES:
    if not os.path.exists(f"{DEST}/{d}"): assemble(s, d, sizes[s])
log("FINISHED")
