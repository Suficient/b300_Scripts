# Held-out generators: fakes from generators never used for training data, made on the 168 test pairs (so test
# people only). Prepares them exactly like the dataset: clean/<name> with reencode.py's x264 settings, and a degraded
# copy of every clip with degrade.py's recipes and renderer (functions loaded from those files, not copied), then
# writes csv/heldout.csv for score.py and meta/heldout.csv. Usage: heldout.py [workers]
import ast, csv, json, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

D = Path("/data/df_train")
HELD = ["facefusion-simswap", "facefusion-simswap512", "facefusion-ghost", "facefusion-blendswap", "facefusion-uniface",
        "facefusion-hififace", "facefusion-edtalk", "facefusion-dfl-m", "facefusion-dfl-f"]
SHOWN_A = {"facefusion-edtalk", "facefusion-dfl-m", "facefusion-dfl-f"}   # A's own face (lip-sync) or A's clip re-faced

def load(path, names):
    """Imports, functions and the named constants of a build script, without running its main code."""
    tree = ast.parse(Path(path).read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    g = {"__name__": "lib"}; exec(compile(ast.Module(keep, []), path, "exec"), g); return g

R = load(D / "reencode.py", {"D", "ARGS"}); G = load(D / "degrade.py", {"D", "FINAL"})

def probe(p):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", p],
                                  capture_output=True, text=True, check=True).stdout)["streams"][0]
    n, d = map(int, s["r_frame_rate"].split("/")); return int(s["width"]), int(s["height"]), n / d

def prep(row):
    out = D / "heldout/clean" / f"{row.name}.mp4"
    if not out.exists():
        tmp = out.with_suffix(".tmp")
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1", "-i", row.src, *R["ARGS"], str(tmp)],
                           capture_output=True, text=True)
        if r.returncode: return row, None, f"FAIL clean {r.stderr[-200:]}"
        tmp.rename(out)
    _, rc, st = G["render"](row)   # writes degraded/heldout/<name>__d.mp4 from the original output
    return row, rc, st

if __name__ == "__main__":
    (D / "heldout/clean").mkdir(parents=True, exist_ok=True); (D / "tmp").mkdir(exist_ok=True)
    pairs = [Path(l.strip()).name for l in open("/data/gen/lists/test_pairs.txt") if l.strip()]
    people = {}
    for q in csv.DictReader(open(D / "meta/manifest.csv")):
        if q["split"] == "test" and q["label"] == "1": people[q["pair"]] = (q["a"], q["b"])
    rows = []
    for gen in HELD:
        for p in pairs:
            src = f"/data/gen_out/{gen}/{p}.mp4"
            if not Path(src).exists(): print("missing", src); continue
            w, h, fps = probe(src); a, b = people[p]
            rows.append(SimpleNamespace(name=f"{gen}__{p}", src=src, split="heldout", gen=gen, pair=p, a=a, b=b,
                                        shown=a if gen in SHOWN_A else b, w=w, h=h, fps=fps, label=1))
    print(len(rows), "clips", flush=True)
    meta, fails = [], 0
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 96) as ex:
        for row, rc, st in ex.map(prep, rows):
            if st.startswith("FAIL"): fails += 1; print(row.name, st, flush=True); continue
            for s, path in (("clean", D / "heldout/clean" / f"{row.name}.mp4"), ("degraded", D / "degraded/heldout" / f"{row.name}__d.mp4")):
                meta.append({"path": str(path), "set": s, "name": row.name, "gen": row.gen, "pair": row.pair, "a": row.a,
                             "b": row.b, "shown": row.shown, "w": row.w, "h": row.h, "fps": round(row.fps, 3),
                             "degradation": json.dumps(rc) if s == "degraded" else ""})
    with open(D / "meta/heldout.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0])); w.writeheader(); w.writerows(meta)
    (D / "csv/heldout.csv").write_text("".join(f"{m['path']} 1\n" for m in meta))
    print(f"DONE {len(meta)} paths, {fails} failures")
