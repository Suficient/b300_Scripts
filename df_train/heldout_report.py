# Held-out generator results: each held-out generator's fakes (heldout.py, scored by score.py) against the test reals
# of the same set (clean or degraded), next to the seen generators on the same test pairs. AUC with 95% ranges that
# resample people shown, and fakes caught at the cutoff chosen on val (as report.py).
# Usage: heldout_report.py HELDOUT_SCORES.csv --head N   (reads runs/test_scores.csv and runs/val_scores.csv)
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("--head", type=int, required=True)
ap.add_argument("--boot", type=int, default=1000); a = ap.parse_args()
D = "/data/df_train"; col = f"p{a.head:02d}"
man = pd.read_csv(f"{D}/meta/manifest.csv", keep_default_na=False)
held = pd.read_csv(a.scores)[["path", col]].merge(pd.read_csv(f"{D}/meta/heldout.csv", keep_default_na=False), on="path", validate="1:1")
test = pd.read_csv(f"{D}/runs/test_scores.csv")[["path", col]].merge(man, on="path", validate="1:1")
val = pd.read_csv(f"{D}/runs/val_scores.csv")

def auc(y, p):
    y, p = np.asarray(y), np.asarray(p); n1, n0 = (y == 1).sum(), (y == 0).sum()
    r = pd.Series(p).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def best_cutoff(y, p):
    o = np.argsort(-p); y, p = y[o], p[o]
    b = (np.cumsum(y) / y.sum() + 1 - np.cumsum(1 - y) / (1 - y).sum()) / 2
    b[np.r_[p[:-1] == p[1:], False]] = -1
    return p[np.argmax(b)]

cut = float(best_cutoff(val.label.values, val[col].values))
rng = np.random.default_rng(0)

def row(reals, fakes):
    g = pd.concat([reals.assign(label=0), fakes.assign(label=1)], ignore_index=True)
    y, p = g.label.values, g[col].values
    idx = list(g.groupby("shown").indices.values()); k = len(idx); bs = []
    for _ in range(a.boot):
        i = np.concatenate([idx[j] for j in rng.integers(0, k, k)])
        if 0 < y[i].sum() < len(i): bs.append(auc(y[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"fakes": len(fakes), "AUC": auc(y, p), "AUC_lo": lo, "AUC_hi": hi,
            "caught@val": (fakes[col] >= cut).mean(), "caught@0.5": (fakes[col] >= 0.5).mean()}

out = {}
for st in ("clean", "degraded"):
    reals = test[(test.label == 0) & (test.set == st)][[col, "shown"]]
    out[("reals called real@val", st)] = (reals[col] < cut).mean()
    h = held[held.set == st]
    for gen, f in h.groupby("gen"): out[(f"HELD-OUT {gen.replace('facefusion-', 'ff-')}", st)] = row(reals, f)
    out[("HELD-OUT all", st)] = row(reals, h)
    seen = test[(test.label == 1) & (test.set == st)]
    for gen, f in seen.groupby("gen_group"): out[(f"seen {gen}", st)] = row(reals, f)
    out[("seen all", st)] = row(reals, seen)

spec = {k: v for k, v in out.items() if not isinstance(v, dict)}
t = pd.DataFrame({k: v for k, v in out.items() if isinstance(v, dict)}).T.rename_axis(["generator", "set"])
t = t.reset_index().pivot(index="generator", columns="set")[["fakes", "AUC", "AUC_lo", "AUC_hi", "caught@val", "caught@0.5"]]
t = t.sort_values(("AUC", "clean"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
txt = (f"head {a.head}, val cutoff {cut:.4f}; reals = test reals of the same set; "
       + ", ".join(f"{k[1]} reals called real {v:.3f}" for k, v in spec.items()) + "\n\n"
       + t.to_string(float_format=lambda x: f"{x:.3f}"))
print(txt)
out_path = os.path.splitext(a.scores)[0] + f"_report_head{a.head:02d}.txt"
open(out_path, "w").write(txt + "\n"); print("\nwrote", out_path)
