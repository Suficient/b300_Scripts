"""Pairs per generator for a fixed GPU budget.

Every generator gets the same cap on GPU time; a generator whose cap buys more than --max pairs gets --max, one that
buys fewer than --floor gets --floor, and the cap is the largest that keeps the total inside the budget. Each
generator's pairs are the first n of one ordering of select500 (gender x skin groups interleaved at their share, best
score first within a group), so smaller sets are subsets of larger ones and stay balanced.

times.csv: name, sec (seconds per pair per worker), k (workers per GPU), load_s (model load per worker), basis, of.
GPU-seconds per pair = sec / k. A row whose `of` names another generator is a variant of it (an authors' fast path):
the two are allocated as one method at their blended cost, and the variant takes the last --variant-share of that
method's pairs (the parent the first ones), so the method covers the same pairs either way.

    python3 allocate.py [--budget-h 12] [--gpus 8] [--out lists]   # writes lists/<name>.txt and lists/order.txt
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def order():
    groups = defaultdict(list)
    for r in csv.DictReader(open(HERE / "select500.csv")):
        if r["picked"] == "1":
            groups[r["group"]].append(r)
    keyed = []
    for rows in groups.values():
        rows.sort(key=lambda r: -float(r["score"]))
        keyed += [((i + 0.5) / len(rows), r["pair"]) for i, r in enumerate(rows)]
    return [p for _, p in sorted(keyed)]


def counts(gens, cap, floor, top, step):
    return {g: max(floor, min(top, int(cap / t // step * step))) for g, (t, _) in gens.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--times", default=HERE / "times.csv")
    ap.add_argument("--budget-h", type=float, default=12)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.9, help="share of the budget to plan for (tails, restarts)")
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--variant-share", type=float, default=0.3)
    ap.add_argument("--out", default=HERE / "lists")
    a = ap.parse_args()

    own, basis, of = {}, {}, {}
    for r in csv.DictReader(open(a.times)):
        own[r["name"]] = (float(r["sec"]) / int(r["k"]), a.gpus * int(r["k"]) * float(r["load_s"]))
        basis[r["name"]] = r["basis"]
        if r.get("of"):
            of[r["name"]] = r["of"]
    v = a.variant_share
    gens = {g: t for g, t in own.items() if g not in of}   # methods; a parent's time blended with its variant's
    for var, par in of.items():
        gens[par] = ((1 - v) * own[par][0] + v * own[var][0], own[par][1] + own[var][1])
    budget = a.budget_h * 3600 * a.gpus * a.margin
    cost = lambda n: sum(n[g] * t + load for g, (t, load) in gens.items())

    if cost({g: a.floor for g in gens}) > budget:
        fixed = sum(load for _, load in gens.values())
        fit = int((budget - fixed) / sum(t for t, _ in gens.values()))
        raise SystemExit(f"--floor {a.floor} needs {cost({g: a.floor for g in gens}) / 3600:.0f} GPU-h of "
                         f"{budget / 3600:.0f}; the largest floor that fits is {fit}")
    lo, hi = 0.0, max(t for t, _ in gens.values()) * a.max
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cost(counts(gens, mid, a.floor, a.max, a.step)) <= budget else (lo, mid)
    n = counts(gens, lo, a.floor, a.max, a.step)

    pairs = order()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "order.txt").write_text("\n".join(pairs) + "\n")
    span = {g: (0, n[g]) for g in gens}
    for var, par in of.items():
        k = n[par] - round(v * n[par])
        span[var], span[par] = (k, n[par]), (0, k)
    print(f"{'generator':22} {'GPU-s/pair':>10} {'pairs':>6} {'GPU-h':>6}  basis")
    for g, (t, load) in sorted(own.items(), key=lambda x: x[1][0]):
        lo, hi = span[g]
        (out / f"{g}.txt").write_text("\n".join(pairs[lo:hi]) + "\n")
        print(f"{g:22} {t:10.0f} {hi - lo:6} {((hi - lo) * t + load) / 3600:6.1f}  {basis[g]}")
    print(f"total {cost(n) / 3600:.1f} GPU-h = {cost(n) / 3600 / a.gpus:.1f} h on {a.gpus} GPUs "
          f"(budget {a.budget_h:g} h, planned at {a.margin:.0%})")


if __name__ == "__main__":
    main()
