# Build step 6: the repo-format CSVs ("path label", 0 = real, 1 = fake) and meta/manifest.csv, one row per CSV path.
import json
from pathlib import Path
import pandas as pd

D = Path("/data/df_train")
items = pd.read_csv(D / "meta/items.csv", keep_default_na=False)
deg = pd.DataFrame([json.loads(l) for l in open(D / "meta/degraded.jsonl")]).drop_duplicates("name", keep="last")
assert deg.status.isin(["ok", "skip"]).all(), deg[~deg.status.isin(["ok", "skip"])][["name", "status"]]
params = deg.set_index("name").drop(columns=["split", "label", "gen", "src", "path", "status"])
params = params.apply(lambda r: json.dumps({k: v for k, v in r.items() if isinstance(v, (dict, list)) or pd.notna(v)}), axis=1)

clean = items.assign(set="clean", path=str(D / "clean") + "/" + items.name + ".mp4", degradation="")
d = items.set_index("name").loc[deg.name].reset_index()
d = d.assign(set="degraded", path=deg.path.values, degradation=d.name.map(params))
man = pd.concat([clean, d], ignore_index=True)
man["gen_group"] = man.gen.str.removesuffix("-fast")
cols = ["path", "split", "set", "label", "gen", "gen_group", "pair", "a", "b", "shown", "channel", "name", "src",
        "clear_glasses", "sunglasses", "headwear", "native_1080p", "w", "h", "fps", "frames", "degradation"]
man[cols].to_csv(D / "meta/manifest.csv", index=False)

out = D / "csv"; out.mkdir(exist_ok=True)
def write(fname, df):
    df = df.sample(frac=1, random_state=0)
    (out / fname).write_text("".join(f"{p} {l}\n" for p, l in zip(df.path, df.label)))
    print(f"{fname}: {len(df)} ({(df.label == 0).sum()} real, {(df.label == 1).sum()} fake)")
for s in ("train", "val"): write(f"{s}.csv", man[man.split == s])
for s in ("val", "test"):
    for k in ("clean", "degraded"): write(f"{s}_{k}.csv", man[(man.split == s) & (man.set == k)])
write("test.csv", man[man.split == "test"])
