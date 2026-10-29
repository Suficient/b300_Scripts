# Build step 4: re-encode every kept clip with identical x264 settings into clean/<name>.mp4 (native size, fps, length).
# Resumable: finished files are skipped; each encode writes a .tmp file and renames it when ffmpeg succeeds.
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

D = Path("/data/df_train")
OUT = D / "clean"; OUT.mkdir(exist_ok=True)
ARGS = ["-map", "0:v:0", "-c:v", "libx264", "-threads", "1", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-g", "50", "-keyint_min", "50", "-sc_threshold", "0", "-an", "-movflags", "+faststart", "-f", "mp4"]
items = pd.read_csv(D / "meta/items.csv")

def enc(row):
    out = OUT / f"{row.name}.mp4"
    if out.exists(): return row.name, "skip", 0.0
    tmp, t = out.with_suffix(".tmp"), time.time()
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1", "-i", row.src, *ARGS, str(tmp)],
                       capture_output=True, text=True)
    if r.returncode: tmp.unlink(missing_ok=True); return row.name, "FAIL " + r.stderr.strip()[-300:], time.time() - t
    tmp.rename(out); return row.name, "ok", time.time() - t

t0, done, fails = time.time(), 0, 0
with ThreadPoolExecutor(160) as ex, open(D / "meta/reencode_fail.txt", "a") as ff:
    for name, st, dt in ex.map(enc, items.itertuples(index=False)):
        done += 1
        if st.startswith("FAIL"): fails += 1; ff.write(f"{name}\t{st}\n"); ff.flush()
        if done % 500 == 0 or done == len(items):
            el = time.time() - t0
            print(f"{done}/{len(items)} fails {fails} elapsed {el / 60:.1f} min eta {el / done * (len(items) - done) / 60:.1f} min", flush=True)
print("DONE", fails, "failures")
