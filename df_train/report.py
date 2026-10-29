# Metrics from per-clip scores (score.py): AUC and balanced accuracy overall, per generator (fast variants grouped with
# their base), clean vs degraded, native 1080p vs not, reals with glasses/headwear vs neither, 16:9 clips only, and
# reals vs the fakes made from them. 95% intervals resample people shown (a person's clips are not independent).
# Thresholded columns use 0.5, plus "@val" columns at the cutoff that maximises balanced accuracy on the val scores
# (--cutoff-from; by default val_scores.csv next to SCORES, which is where post_train.sh writes it).
# Usage: report.py SCORES.csv [SCORES2.csv ...] --head N [--name test] [--cutoff-from VAL.csv] [--no-wandb]
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("scores", nargs="+"); ap.add_argument("--head", type=int, required=True)
ap.add_argument("--name", default="test"); ap.add_argument("--no-wandb", action="store_true")
ap.add_argument("--cutoff-from"); ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()
col = f"p{a.head:02d}"
man = pd.read_csv("/data/df_train/meta/manifest.csv", keep_default_na=False)
items = pd.read_csv("/data/df_train/meta/items.csv", keep_default_na=False, usecols=["name", "labels_from"])
s = pd.concat([pd.read_csv(f) for f in a.scores]).drop_duplicates("path")
df = s[["path", col]].rename(columns={col: "p"}).merge(man, on="path", how="left", validate="1:1")
assert df.split.notna().all(), "scored paths missing from the manifest"
df = df.merge(items, on="name", how="left", validate="m:1")
df["pred"] = (df.p >= 0.5).astype(int)
df["accessory"] = df.clear_glasses.astype(str).eq("True") | df.sunglasses.astype(str).eq("True") | df.headwear.astype(str).eq("True")
df["is16x9"] = (pd.to_numeric(df.w, errors="coerce") / pd.to_numeric(df.h, errors="coerce") - 16 / 9).abs() < 0.02

def auc(y, p):
    y, p = np.asarray(y), np.asarray(p); n1, n0 = (y == 1).sum(), (y == 0).sum()
    if not n1 or not n0: return np.nan
    r = pd.Series(p).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def bal(y, pr):
    return np.mean([(pr[y == 1] == 1).mean(), (pr[y == 0] == 0).mean()]) if (y == 1).any() and (y == 0).any() else np.nan

def best_cutoff(y, p):
    # predict fake for p >= cutoff; only cut between distinct scores so ties fall on one side
    o = np.argsort(-p); y, p = y[o], p[o]
    b = (np.cumsum(y) / y.sum() + 1 - np.cumsum(1 - y) / (1 - y).sum()) / 2
    b[np.r_[p[:-1] == p[1:], False]] = -1
    return p[np.argmax(b)]

cut_src = a.cutoff_from or os.path.join(os.path.dirname(a.scores[0]), "val_scores.csv")
cut = None
if os.path.exists(cut_src) and os.path.abspath(cut_src) not in map(os.path.abspath, a.scores):
    v = pd.read_csv(cut_src); cut = float(best_cutoff(v.label.values, v[col].values))
    df["pv"] = (df.p >= cut).astype(int)
    print(f"cutoff from {cut_src}: {cut:.4f} (val balanced accuracy {bal(v.label.values, (v[col].values >= cut).astype(int)):.3f})")
elif a.cutoff_from:
    raise SystemExit(f"--cutoff-from {a.cutoff_from}: not found or same as SCORES")
ba = "bal_acc@val" if cut is not None else "bal_acc"

rng = np.random.default_rng(0)
def ci(g):
    y, p, pr = g.label.values, g.p.values, (g.pv if cut is not None else g.pred).values
    idx = list(g.groupby("shown").indices.values()); k = len(idx); out = []
    for _ in range(a.boot):
        i = np.concatenate([idx[j] for j in rng.integers(0, k, k)])
        out.append((auc(y[i], p[i]), bal(y[i], pr[i])))
    lo, hi = np.nanpercentile(np.array(out, dtype=float), [2.5, 97.5], axis=0)
    return {"AUC_lo": lo[0], "AUC_hi": hi[0], f"{ba}_lo": lo[1], f"{ba}_hi": hi[1]}

def summ(g, with_ci=False):
    y = g.label.values
    tpr = (g.pred[g.label == 1] == 1).mean() if (y == 1).any() else np.nan
    tnr = (g.pred[g.label == 0] == 0).mean() if (y == 0).any() else np.nan
    r = {"reals": int((y == 0).sum()), "fakes": int((y == 1).sum()), "AUC": auc(y, g.p),
         "bal_acc": np.nanmean([tpr, tnr]), "fake_recall": tpr, "real_specificity": tnr}
    if cut is not None:
        r["bal_acc@val"] = bal(y, g.pv.values)
        r["fake_recall@val"] = (g.pv[g.label == 1] == 1).mean() if (y == 1).any() else np.nan
        r["real_specificity@val"] = (g.pv[g.label == 0] == 0).mean() if (y == 0).any() else np.nan
    if with_ci: r.update(ci(g))
    return pd.Series(r)

tables = {}
tables["overall"] = pd.DataFrame({"all": summ(df, True), **{k: summ(g, True) for k, g in df.groupby("set")}}).T
rows = {}
for (st, gen), g in df[df.label == 1].groupby(["set", "gen_group"]):
    rows[(gen, st)] = summ(pd.concat([df[(df.label == 0) & (df.set == st)], g]), True)   # this generator's fakes vs all reals in the set
keep = ["fakes", "AUC", "AUC_lo", "AUC_hi", "fake_recall"] + (["fake_recall@val"] if cut is not None else [])
tables["per_generator"] = pd.DataFrame(rows).T.rename_axis(["generator", "set"]).reset_index().pivot(index="generator", columns="set")[keep]
tables["per_generator"] = tables["per_generator"].sort_values(("AUC", "clean"))
c = df[df.set == "clean"]
nat = {"fakes native 1080p": c[(c.label == 0) | ((c.label == 1) & (c.native_1080p.astype(str) == "True"))],
       "fakes not 1080p": c[(c.label == 0) | ((c.label == 1) & (c.native_1080p.astype(str) != "True"))]}
tables["clean_by_native_1080p"] = pd.DataFrame({k: summ(g) for k, g in nat.items()}).T
acc = {"reals with glasses/headwear": c[(c.label == 1) | ((c.label == 0) & c.accessory)],
       "reals with neither": c[(c.label == 1) | ((c.label == 0) & ~c.accessory)]}
tables["clean_by_accessory"] = pd.DataFrame({k: summ(g) for k, g in acc.items()}).T
tables["16x9_only"] = pd.DataFrame({k: summ(g, True) for k, g in df[df.is16x9].groupby("set")}).T   # reals and fakes both 16:9
ms = {}
for st, g in df.groupby("set"):
    f = g[g.label == 1]; r = g[(g.label == 0) & g.name.isin(set(f.labels_from))]
    ms[st] = summ(pd.concat([r, f[f.labels_from.isin(set(r.name))]]), True)   # reals vs only the fakes made from them
tables["reals_vs_own_fakes"] = pd.DataFrame(ms).T

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
hdr = f"head {a.head}" + (f", val cutoff {cut:.4f}" if cut is not None else "")
for k, t in tables.items(): print(f"\n== {k} ({hdr})\n" + t.to_string(float_format=lambda x: f"{x:.3f}"))
out = os.path.splitext(a.scores[0])[0] + f"_report_head{a.head:02d}.txt"
with open(out, "w") as f:
    for k, t in tables.items(): f.write(f"== {k} ({hdr})\n{t.to_string(float_format=lambda x: f'{x:.3f}')}\n\n")
print("\nwrote", out)

if not a.no_wandb and os.environ.get("WANDB_MODE") != "disabled":
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "vjepa-deepfake-probe"), name=f"{a.name}-head{a.head:02d}",
                     job_type="eval", config={"head": a.head, "scores": a.scores, "val_cutoff": cut, "cutoff_from": cut_src if cut is not None else None})
    for k, t in tables.items():
        t2 = t.copy(); t2.columns = [" ".join(map(str, x)) if isinstance(x, tuple) else str(x) for x in t2.columns]
        run.log({k: wandb.Table(dataframe=t2.reset_index())})
    o = tables["overall"]
    run.summary.update({f"{st}/{m}": o.loc[st, m] for st in o.index for m in o.columns if m not in ("reals", "fakes")})
    run.finish()
