"""CLI.

  df-benchmark prepare            match every clip in clips_dir to a different-looking person -> work_dir/<clip>/
  df-benchmark run NAME...        fake every prepared clip with each generator -> results_dir/NAME/<clip>.mp4,
                                  timings.jsonl (one line per clip), summary.json
  df-benchmark report             one row per generator from the summaries
  df-benchmark list               the generators, their category and whether they are installed
"""
from __future__ import annotations

import argparse
import json
import logging

from . import config as config_mod
from .generators import GENERATORS, resolve
from .prepare import clips_in, jobs_in, prepare
from .run import run

log = logging.getLogger("df-benchmark")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="df-benchmark")
    ap.add_argument("-c", "--config", help="TOML config (see config.example.toml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("run")
    p.add_argument("generators", nargs="+")
    p.add_argument("--force", action="store_true", help="regenerate clips that already have a result")
    p.add_argument("--only", nargs="+", metavar="CLIP", help="only these clips (stems)")
    sub.add_parser("report")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config_mod.load(args.config)

    if args.cmd == "prepare":
        clips = clips_in(cfg.clips_dir)
        log.info("%d clips in %s", len(clips), cfg.clips_dir)
        jobs = prepare(cfg, clips)
        log.info("%d of %d clips prepared in %s", len(jobs), len(clips), cfg.work_dir)
    elif args.cmd == "run":
        jobs = jobs_in(cfg.work_dir)
        if args.only:
            jobs = [j for j in jobs if j.stem in args.only]
        if not jobs:
            raise SystemExit(f"no prepared clips in {cfg.work_dir}: run `df-benchmark prepare` first")
        gens = [resolve(cfg, name) for name in args.generators]  # all names checked before anything runs
        for gen in gens:
            s = run(cfg, gen, jobs, args.force)
            log.info("%s: %d/%d ok, %s s total, %s s mean", gen.name, s["ok"], s["clips"], s["total_s"], s["mean_s"])
    elif args.cmd == "report":
        report(cfg)
    elif args.cmd == "list":
        names = list(dict.fromkeys([*GENERATORS, *cfg.generators, *cfg.comfy]))
        for name in names:
            try:
                g = resolve(cfg, name)
            except KeyError as e:
                print(f"{name:18} {GENERATORS.get(name, '?'):15} not configured ({e.args[0].split(': ', 1)[-1]})")
                continue
            where = f"ComfyUI ({g.comfy.get('workflow')})" if g.comfy else (
                str(g.dir) if g.dir.is_dir() else f"not installed ({g.dir})")
            print(f"{name:18} {g.category:15} {where}")


def report(cfg):
    rows = [json.loads(p.read_text()) for p in sorted(cfg.results_dir.glob("*/summary.json"))]
    if not rows:
        raise SystemExit(f"no results in {cfg.results_dir}")
    rows.sort(key=lambda r: (r["mean_s"] is None, r["mean_s"] or 0))
    cols = [("generator", 18), ("category", 15), ("ok", 6), ("total_s", 9), ("mean_s", 8), ("first_s", 8),
            ("mean_after_first_s", 8), ("x_realtime", 8), ("peak_mem_gb", 8), ("output", 22)]
    head = {"mean_after_first_s": "rest_s", "x_realtime": "x_rt", "peak_mem_gb": "mem_gb"}
    print(" ".join(f"{head.get(c, c):>{w}}" if i > 1 else f"{head.get(c, c):{w}}" for i, (c, w) in enumerate(cols)))
    for r in rows:
        o = r["output"]
        cells = {**r, "ok": f"{r['ok']}/{r['clips']}",
                 "output": f"{o['width']}x{o['height']} {o['fps']:g}fps {o['duration']:g}s" if o else "-"}
        print(" ".join(f"{_fmt(cells[c]):>{w}}" if i > 1 else f"{_fmt(cells[c]):{w}}"
                       for i, (c, w) in enumerate(cols)))


def _fmt(v) -> str:
    return "-" if v is None else str(v)


if __name__ == "__main__":
    main()
