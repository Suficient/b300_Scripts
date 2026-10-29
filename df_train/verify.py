# Build step 7: checks before training. Splits are person-disjoint and pair-consistent, counts per split and generator,
# every CSV path exists and decodes (decord, the training reader: length plus first, middle and last frame), and no
# /data/real or _compare path appears anywhere.
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

D = Path("/data/df_train")
man = pd.read_csv(D / "meta/manifest.csv", keep_default_na=False)
sp = {int(k): v for k, v in json.load(open(D / "meta/split.json")).items()}
bad = 0
def check(ok, msg):
    global bad
    print(("ok   " if ok else "FAIL ") + msg); bad += not ok

for col in ("a", "b", "shown"):
    check((man[col].astype(int).map(sp) == man.split).all(), f"every clip's {col} person is in the clip's split")
persons = pd.concat([man[["a", "split"]].rename(columns={"a": "p"}), man[["b", "split"]].rename(columns={"b": "p"})])
check(persons.groupby("p").split.nunique().max() == 1, "no person appears in two splits")
check(not man.path.str.contains("/data/real/|_compare").any() and not man.src.str.contains("/data/real/|_compare").any(),
      "no /data/real or _compare paths")
check(not man.path.str.contains(r"\s").any(), "no whitespace in paths")
csvs = sorted((D / "csv").glob("*.csv"))
rows = {f.name: pd.read_csv(f, sep=" ", header=None, names=["path", "label"]) for f in csvs if not f.name.startswith("smoke")}
for n, r in rows.items():
    check(r.path.isin(man.path).all() and r.path.is_unique and set(r.label) <= {0, 1}, f"{n}: {len(r)} rows, all in the manifest, unique, labels 0/1")
lab = man.set_index("path").label
check(all((r.label.values == lab.loc[r.path].values).all() for r in rows.values()), "CSV labels match the manifest")
tr, va = set(rows["train.csv"].path), set(rows["val.csv"].path)
check(not tr & va and not tr & set(rows["test.csv"].path) and not va & set(rows["test.csv"].path), "train, val and test CSVs share no path")

print("\n== clips per split, set and class")
print(man.groupby(["split", "set", "label"]).size().unstack([1, 2]).to_string())
print("\n== fakes per generator (clean / degraded)")
g = man[man.label == 1].groupby(["gen", "split", "set"]).size().unstack([1, 2]).fillna(0).astype(int)
print(g[[c for c in [(s, k) for s in ("train", "val", "test") for k in ("clean", "degraded")] if c in g.columns]].sort_values(("train", "clean"), ascending=False).to_string())

def decodes(p):
    import decord
    try:
        vr = decord.VideoReader(p, num_threads=1); n = len(vr)
        if n < 32: return p, f"only {n} frames"
        vr.get_batch([0, n // 2, n - 1]); return p, None
    except Exception as e:
        return p, str(e)[:200]
paths = sorted(set().union(*[set(r.path) for r in rows.values()]))
with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 96) as ex:
    errs = [(p, e) for p, e in ex.map(decodes, paths, chunksize=64) if e]
for p, e in errs[:20]: print("   ", p, e)
check(not errs, f"all {len(paths)} CSV paths exist and decode with decord ({len(errs)} failures)")
print("\nALL CHECKS PASSED" if not bad else f"\n{bad} CHECKS FAILED")
sys.exit(bool(bad))
