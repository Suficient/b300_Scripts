# Build step 5: the degraded set. Picks clips per split (fakes round-robin across generators, reals spread across
# people and channels), gives each a recipe seeded from its name, and renders degraded/<split>/<name>__d.mp4 from the
# original file. Every recipe goes to meta/degraded.jsonl. Usage: degrade.py [workers] [--only N] (N clips, for a test).
import hashlib, json, random, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from pathlib import Path
import numpy as np, pandas as pd

D = Path("/data/df_train")
SIZES = {"train": 3250, "val": 350, "test": 350}                 # per class
FINAL = ["-c:v", "libx264", "-threads", "1", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-g", "50", "-keyint_min", "50", "-sc_threshold", "0", "-an", "-movflags", "+faststart"]
workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 32
only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None

# ---- selection ----------------------------------------------------------------------------------------
items = pd.read_csv(D / "meta/items.csv")
rng = np.random.default_rng(1)
items["r"] = rng.random(len(items))
picked = []
for s, n in SIZES.items():
    f = items[(items.split == s) & (items.label == 1)].sort_values("r").copy()
    f["k"] = f.groupby("gen").cumcount()                          # round-robin: k-th clip of every generator, then k+1
    picked.append(f.sort_values(["k", "r"]).head(n))
    r = items[(items.split == s) & (items.label == 0)].sort_values("r").copy()
    r["kp"] = r.groupby("shown").cumcount(); r["kc"] = r.groupby("channel").cumcount()
    picked.append(r.sort_values(["kp", "kc", "r"]).head(n))
sel = pd.concat(picked, ignore_index=True)
if only: sel = sel.groupby(["split", "label"]).head(max(1, only // 6))

# ---- recipes ------------------------------------------------------------------------------------------
def recipe(name, h0, fps0):
    g = random.Random(int(hashlib.sha256(name.encode()).hexdigest()[:16], 16))
    rc = {"kind": g.choice(["social", "webcam"])}
    if rc["kind"] == "social":
        rc["fps"] = g.choice([24, 25, 30])
        if g.random() < 0.6:
            z = g.uniform(0.0, 0.10); rc["crop"] = {"zoom": round(z, 3), "x": round(g.random(), 3), "y": round(g.random(), 3)}
        if g.random() < 0.3: rc["sharpen"] = round(g.uniform(0.3, 1.0), 2)
        rc["passes"] = []
        for _ in range(g.choice([1, 1, 2, 2, 3])):
            p = {"height": g.choice([360, 480, 540, 720, 1080]), "codec": "vp9" if g.random() < 0.15 else "h264"}
            if g.random() < 0.6: p["crf"] = g.randint(23, 38)
            else: p["kbps"] = int(g.uniform(400, 4000))
            if p["codec"] == "h264": p["preset"] = g.choice(["veryfast", "fast", "medium"])
            rc["passes"].append(p)
    else:
        rc["fps"] = g.choice([15, 20, 24, 25, 30])
        rc["passes"] = [{"height": g.choice([360, 480, 540, 720]), "codec": g.choice(["vp8", "h264"]),
                         "cbr_kbps": int(g.uniform(250, 1500))}]
        if g.random() < 0.4: rc["noise"] = g.randint(4, 15)
        if g.random() < 0.3: rc["dim"] = {"brightness": round(g.uniform(-0.15, -0.04), 3), "gamma": round(g.uniform(0.75, 0.95), 2)}
        if g.random() < 0.12: rc["corrupt"] = {"amount": g.randint(20000, 200000), "drop": round(g.uniform(0.005, 0.03), 3)}
    if g.random() < 0.4:
        rc["freeze"] = {"p": round(g.uniform(0.02, 0.15), 3),
                        "long": [[round(g.random(), 3), round(g.uniform(0.2, 0.5), 2)] for _ in range(g.choice([0, 1, 1, 2]))]}
    rc["seed"] = g.randrange(2**31)
    return rc

def enc_args(p):
    if p["codec"] == "h264":
        a = ["-c:v", "libx264", "-preset", p.get("preset", "veryfast")]
        if "cbr_kbps" in p:
            b = f"{p['cbr_kbps']}k"
            return a + ["-tune", "zerolatency", "-b:v", b, "-minrate", b, "-maxrate", b, "-bufsize", b, "-x264-params", "nal-hrd=cbr"], ".mp4"
        return a + (["-crf", str(p["crf"])] if "crf" in p else ["-b:v", f"{p['kbps']}k", "-maxrate", f"{p['kbps']}k", "-bufsize", f"{2 * p['kbps']}k"]), ".mp4"
    if p["codec"] == "vp8":
        b = f"{p['cbr_kbps']}k"
        return ["-c:v", "libvpx", "-b:v", b, "-minrate", b, "-maxrate", b, "-bufsize", b, "-deadline", "realtime", "-cpu-used", "8", "-error-resilient", "1"], ".webm"
    return ["-c:v", "libvpx-vp9", "-deadline", "good", "-cpu-used", "5", "-row-mt", "0"] + \
           (["-crf", str(p["crf"]), "-b:v", "0"] if "crf" in p else ["-b:v", f"{p['kbps']}k"]), ".webm"

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode: raise RuntimeError(f"{' '.join(map(str, cmd[:12]))}...: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r

def freeze_map(n, fps, fr, seed):
    g = random.Random(seed); src = list(range(n))
    for i in range(1, n):
        if g.random() < fr["p"]: src[i] = src[i - 1]
    for start, dur in fr["long"]:
        s = int(start * n); e = min(n, s + max(1, round(dur * fps)))
        for i in range(max(1, s), e): src[i] = src[i - 1]
    return src

def render(row):
    out = D / "degraded" / row.split / f"{row.name}__d.mp4"
    rc = recipe(row.name, row.h, row.fps)
    if out.exists(): return row, rc, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir="/data/df_train/tmp"))
    try:
        vf = []
        if "crop" in rc:
            z, x, y = rc["crop"]["zoom"], rc["crop"]["x"], rc["crop"]["y"]
            vf.append(f"crop=trunc(iw*{1 - z}/2)*2:trunc(ih*{1 - z}/2)*2:(iw-ow)*{x}:(ih-oh)*{y}")
        vf.append(f"fps={rc['fps']}")
        h = rc["passes"][0]["height"]
        vf.append(f"scale=-2:{h}:flags=bicubic")
        if "sharpen" in rc: vf.append(f"unsharp=5:5:{rc['sharpen']}")
        if "dim" in rc: vf.append(f"eq=brightness={rc['dim']['brightness']}:gamma={rc['dim']['gamma']}")
        if "noise" in rc: vf.append(f"noise=alls={rc['noise']}:allf=t+u:all_seed={rc['seed'] % 100000}")
        vf.append("format=yuv420p")
        # stage A: filters -> raw frames (frozen frames applied here) -> first encode
        info = run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", row.src, "-vf", ",".join(vf), "-frames:v", "1",
                    "-f", "rawvideo", "-"], ).stdout
        w = len(info) * 2 // 3 // h
        raw = run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", row.src, "-vf", ",".join(vf), "-filter_threads", "1",
                   "-f", "rawvideo", "-"]).stdout
        fsz = w * h * 3 // 2; n = len(raw) // fsz
        if "freeze" in rc:
            src = freeze_map(n, rc["fps"], rc["freeze"], rc["seed"])
            raw = b"".join(raw[j * fsz:(j + 1) * fsz] for j in src)
        a, ext = enc_args(rc["passes"][0]); cur = tmp / f"p0{ext}"
        run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", str(rc["fps"]),
             "-i", "-", "-threads", "1", *a, "-an", str(cur)], input=raw)
        for k, p in enumerate(rc["passes"][1:], 1):
            a, ext = enc_args(p); nxt = tmp / f"p{k}{ext}"
            run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(cur), "-vf", f"scale=-2:{p['height']}:flags=bicubic",
                 "-threads", "1", *a, "-an", str(nxt)]); cur = nxt
        if "corrupt" in rc:
            c = rc["corrupt"]; nxt = tmp / f"bad{cur.suffix}"
            run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(cur), "-c", "copy",
                 "-bsf:v", f"noise=amount={c['amount']}:drop=not(key)*lt(random(0)\\,{c['drop']})", str(nxt)]); cur = nxt
        last = rc["passes"][-1]
        if last["codec"] == "h264" and "corrupt" not in rc:
            run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(cur), "-c", "copy", "-an", "-movflags", "+faststart", str(tmp / "o.mp4")])
            rc["final"] = "remux"
        else:
            run(["ffmpeg", "-nostdin", "-v", "error", "-err_detect", "ignore_err", "-max_error_rate", "1", "-threads", "1", "-i", str(cur),
                 "-fps_mode", "cfr", "-r", str(rc["fps"]), *FINAL, str(tmp / "o.mp4")])
            rc["final"] = "x264 crf18"
        shutil.move(tmp / "o.mp4", out)
        return row, rc, "ok"
    except Exception as e:
        return row, rc, f"FAIL {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

(D / "tmp").mkdir(exist_ok=True)
t0, done, fails = time.time(), 0, 0
log = open(D / "meta" / ("degraded_test.jsonl" if only else "degraded.jsonl"), "a")
with ProcessPoolExecutor(workers) as ex:   # processes, not threads: raw frames pass through Python
    for row, rc, st in ex.map(render, [SimpleNamespace(**r) for r in sel.to_dict("records")]):
        done += 1; fails += st.startswith("FAIL")
        log.write(json.dumps({"name": row.name, "split": row.split, "label": int(row.label), "gen": row.gen, "src": row.src,
                              "path": f"/data/df_train/degraded/{row.split}/{row.name}__d.mp4", "status": st, **rc}) + "\n"); log.flush()
        if st.startswith("FAIL"): print(row.name, st, flush=True)
        if done % 250 == 0 or done == len(sel):
            el = time.time() - t0
            print(f"{done}/{len(sel)} fails {fails} elapsed {el / 60:.1f} min eta {el / done * (len(sel) - done) / 60:.1f} min", flush=True)
print("DONE", fails, "failures")
print(sel.groupby(["split", "label"]).size().unstack().to_string())
print(sel[sel.label == 1].groupby(["split", "gen"]).size().unstack(0).fillna(0).astype(int).to_string())
