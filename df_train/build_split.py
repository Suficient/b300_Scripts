# Build steps 1-3: list the fakes, split people 80/10/10 with pairs kept together, cap reals per person.
# Writes meta/split.json (person -> split), meta/items.csv (every kept clip) and prints losses per generator.
import json, re, sqlite3, collections, random, heapq, math
from pathlib import Path
import numpy as np, pandas as pd

M = Path("/data/df_train/meta")
LIPSYNC = {"latentsync", "musetalk", "wav2lip", "x-dub", "infinitetalk-v2v", "infinitetalk-v2v-fast"}
B300_WEIGHT, K_CAP, SEEDS = 20, 12, 200
REALS_FRAC, GEN_LO, GEN_HI, REALS_HI = 0.10, 0.08, 0.11, 1.12

# ---- 1. fakes ------------------------------------------------------------------------------------------
jobs = {j["pair"]: j for j in map(json.loads, open(M / "jobs_all.jsonl"))}
rows = [("bfs-ltx23", f.stem, str(f)) for f in Path("/data/outputs/full_v3").glob("*.mp4") if re.fullmatch(r"\d+", f.stem)]
for d in sorted(Path("/data/gen_out").iterdir()):
    rows += [(d.name, f.stem, str(f)) for f in d.glob("*.mp4") if re.fullmatch(r"\d+", f.stem)]
fakes = pd.DataFrame(rows, columns=["gen", "pair", "src"])
fakes["a"] = fakes.pair.map(lambda p: jobs[p]["a_person"]); fakes["b"] = fakes.pair.map(lambda p: jobs[p]["b_person"])
fakes["a_clip"] = fakes.pair.map(lambda p: jobs[p]["a_clip"]); fakes["b_clip"] = fakes.pair.map(lambda p: jobs[p]["b_clip"])
fakes["shown"] = np.where(fakes.gen.isin(LIPSYNC), fakes.a, fakes.b)
fakes["b300"] = fakes.gen != "bfs-ltx23"
fakes["name"] = fakes.gen + "__" + fakes.pair
print(f"fakes {len(fakes)} from {fakes.gen.nunique()} sources ({fakes.b300.sum()} B300, {(~fakes.b300).sum()} head swaps)")

db = sqlite3.connect("file:/data/clipfarmer/clipfarmer.sqlite?mode=ro", uri=True)
reals = pd.read_sql("select path, person_id, channel_id, video_id, labels_json from clips where fate='accepted'", db)
reals["name"] = reals.path.str.split("/").str[-1].str.replace(r"\.mp4$", "", regex=True)
reals["src"] = "/data/clipfarmer/accepted/" + reals.name + ".mp4"
pids = sorted(set(reals.person_id) | set(fakes.a) | set(fakes.b))
n_reals = reals.person_id.value_counts().to_dict()

# The split targets use reals after the cap (computed here with all fakes; the final cap uses kept fakes).
shown_all = fakes.shown.value_counts().to_dict()
capped = {p: min(n_reals.get(p, 0), max(shown_all.get(p, 0), K_CAP)) for p in pids}
total_reals = sum(capped.values())

# ---- 2. split -----------------------------------------------------------------------------------------
gens = sorted(fakes.gen[fakes.b300].unique())
n_gen = fakes[fakes.b300].gen.value_counts().to_dict()
lo = {g: max(1, math.floor(GEN_LO * n_gen[g])) for g in gens}
hi = {g: max(2, math.ceil(GEN_HI * n_gen[g])) for g in gens}
# People linked by B300 fakes (the 500 shared pairs, groups of at most 17 people) move as one unit, so no B300
# fake is ever cut; the region growth then runs over units and can only cut head swaps.
par = {p: p for p in pids}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b in zip(fakes.a[fakes.b300], fakes.b[fakes.b300]): par[find(a)] = find(b)
unit = {p: find(p) for p in pids}
units = sorted(set(unit.values()))
u_reals = collections.Counter(); u_gen = collections.defaultdict(collections.Counter)
for p in pids: u_reals[unit[p]] += capped[p]
for a, g in zip(fakes.a[fakes.b300], fakes.gen[fakes.b300]): u_gen[unit[a]][g] += 1
w = collections.Counter()                                        # weighted edge between people
for a, b, v in zip(fakes.a, fakes.b, fakes.b300):
    if a != b: w[tuple(sorted((a, b)))] += B300_WEIGHT if v else 1
adj = collections.defaultdict(collections.Counter)               # unit -> unit, head-swap weight
for (a, b), c in w.items():
    ua, ub = unit[a], unit[b]
    if ua != ub: adj[ua][ub] += c; adj[ub][ua] += c
b300_units = sorted(u_gen)

def grow(rng, split, name):
    reals_in, gin, conn, heap, stale = 0, collections.Counter(), collections.Counter(), [], 0
    def done():
        return reals_in >= REALS_FRAC * total_reals and all(gin[g] >= lo[g] for g in gens)
    while not done():
        if stale > 3000: return False
        if not heap:
            free = [u for u in (b300_units if reals_in >= REALS_FRAC * total_reals else units) if u not in split]
            if not free: return False
            heapq.heappush(heap, (0, rng.random(), rng.choice(free)))
        _, _, u = heapq.heappop(heap)
        if u in split: continue
        d = u_gen[u]
        if reals_in + u_reals[u] > REALS_HI * REALS_FRAC * total_reals or any(gin[g] + d[g] > hi[g] for g in d):
            stale += 1; continue
        split[u] = name; reals_in += u_reals[u]; gin.update(d); stale = 0
        for n, c in adj[u].items():
            if n not in split: conn[n] += c; heapq.heappush(heap, (-conn[n], rng.random(), n))
    return True

def run(seed):
    rng, us = random.Random(seed), {}
    ok = grow(rng, us, "val") and grow(rng, us, "test")
    split = {p: us[unit[p]] for p in pids if unit[p] in us}
    loss = sum(c for (u, v), c in w.items() if split.get(u, "train") != split.get(v, "train"))
    return ok, loss, seed, split

results = [run(s) for s in range(SEEDS)]
good = [r for r in results if r[0]]
print(f"seeds meeting targets: {len(good)} of {SEEDS}")
_, loss, seed, split = min(good or results, key=lambda r: r[1])
sp = {p: split.get(p, "train") for p in pids}
print(f"chosen seed {seed}, weighted loss {loss}")

fakes["split"] = fakes.a.map(sp)
fakes["lost"] = fakes.a.map(sp) != fakes.b.map(sp)
t = fakes.groupby("gen").agg(n=("src", "size"), lost=("lost", "sum"))
for s in ("train", "val", "test"): t[s] = fakes[~fakes.lost & (fakes.split == s)].groupby("gen").size()
t = t.fillna(0).astype(int); t["val%"] = (100 * t.val / t.n).round(1); t["test%"] = (100 * t.test / t.n).round(1)
print(t.sort_values("n", ascending=False).to_string())
print(f"lost {fakes.lost.sum()} ({fakes.lost.mean():.1%}): head swaps {fakes.lost[~fakes.b300].sum()}, B300 {fakes.lost[fakes.b300].sum()}")
kf = fakes[~fakes.lost].copy()

# ---- 3. identity cap ----------------------------------------------------------------------------------
shown_kept = kf.shown.value_counts().to_dict()
priority = set(kf.a_clip.str.replace(r"\.mp4$", "", regex=True)) | set(kf.b_clip.str.replace(r"\.mp4$", "", regex=True))
rng = np.random.default_rng(0)
reals["r"] = rng.random(len(reals))
reals["pri"] = ~reals.name.isin(priority)                         # False sorts first
reals = reals.sort_values(["person_id", "pri", "r"])
reals["rv"] = reals.groupby(["person_id", "video_id"]).cumcount()      # spread over videos...
reals["rc"] = reals.groupby(["person_id", "channel_id"]).cumcount()    # ...and channels
reals = reals.sort_values(["person_id", "pri", "rv", "rc", "r"])
reals["rank"] = reals.groupby("person_id").cumcount()
reals["cap"] = reals.person_id.map(lambda p: max(shown_kept.get(p, 0), K_CAP))
kr = reals[reals["rank"] < reals.cap].copy()
dropped_pri = (~reals.pri & (reals["rank"] >= reals.cap)).sum()
kr["split"] = kr.person_id.map(sp)
print(f"reals kept {len(kr)} of {len(reals)}; pair source clips dropped by the cap {dropped_pri}")

# ---- items.csv ---------------------------------------------------------------------------------------
probe = pd.read_csv(M / "probe.csv").set_index("file")
lab = reals.set_index("name").labels_json.map(lambda s: json.loads(s or "{}"))
flags = lambda l: (l.get("glasses", {}).get("clear", 0) > .5, l.get("glasses", {}).get("sunglasses", 0) > .5, l.get("headwear", {}).get("yes", 0) > .5)
chan = reals.set_index("name").channel_id
real_items = pd.DataFrame(dict(name=kr.name, src=kr.src, label=0, split=kr.split, gen="real", pair="", a=kr.person_id, b=kr.person_id,
                               shown=kr.person_id, channel=kr.channel_id, labels_from=kr.name))
fake_items = pd.DataFrame(dict(name=kf.name, src=kf.src, label=1, split=kf.split, gen=kf.gen, pair=kf.pair, a=kf.a, b=kf.b,
                               shown=kf.shown, channel=kf.a_clip.str.replace(r"\.mp4$", "", regex=True).map(chan),
                               labels_from=kf.a_clip.str.replace(r"\.mp4$", "", regex=True)))
items = pd.concat([real_items, fake_items], ignore_index=True)
f3 = items.labels_from.map(lambda n: flags(lab.get(n, {})))
items["clear_glasses"], items["sunglasses"], items["headwear"] = zip(*f3)
for c in ("w", "h", "fps", "frames", "codec"): items[c] = items.src.map(probe[c])
items["native_1080p"] = (items.w == 1920) & (items.h == 1080)
assert items.name.is_unique and not items.name.str.contains(r"\s").any()
items.to_csv(M / "items.csv", index=False)
json.dump({str(p): s for p, s in sp.items()}, open(M / "split.json", "w"))
print(items.groupby(["split", "label"]).size().unstack().to_string())
