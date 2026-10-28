"""Run one generator over every prepared job and time it.

Each clip is a separate run: a script generator is started fresh per clip (its model load is part of the
time), a ComfyUI generator keeps its models loaded between clips (only the first clip pays the load).
summary.json reports the first clip apart from the rest so the two cases can be compared."""
from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import media
from .comfy import Comfy, bind
from .generators import Generator
from .prepare import Job

log = logging.getLogger(__name__)


def run(cfg, gen: Generator, jobs: list[Job], force: bool = False) -> dict:
    out_dir = cfg.results_dir / gen.name
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    timings = out_dir / "timings.jsonl"
    comfy = Comfy(cfg.comfy_url) if gen.comfy else None
    if comfy and not comfy.up():
        raise SystemExit(f"{gen.name}: ComfyUI isn't answering at {cfg.comfy_url}"
                         " (start it with generators/bfs-ltx23/comfy.sh start)")
    for job in jobs:
        out = out_dir / f"{job.stem}.mp4"
        if out.exists() and not force:
            log.info("%s/%s: exists, skipped (--force reruns it)", gen.name, job.stem)
            continue
        out.unlink(missing_ok=True)
        log.info("%s/%s: generating", gen.name, job.stem)
        with MemoryPeak() as mem:
            t0 = time.monotonic()
            try:
                if comfy:
                    _run_comfy(cfg, gen, comfy, job, out)
                else:
                    _run_script(cfg, gen, job, out, out_dir / "logs" / f"{job.stem}.log")
                error = None if out.exists() else "finished without writing the output"
            except Exception as e:  # noqa: BLE001 - one failed clip must not stop the benchmark
                error = f"{type(e).__name__}: {e}"
            seconds = time.monotonic() - t0
        rec = {"clip": job.stem, "seconds": round(seconds, 2), "ok": error is None, "error": error,
               "peak_mem_gb": mem.peak_gb, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "output": media.probe(out) if error is None else None}
        with timings.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("%s/%s: %s in %.0f s", gen.name, job.stem, "done" if rec["ok"] else f"FAILED ({error})", seconds)
    summary = summarize(gen, timings)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _run_script(cfg, gen: Generator, job: Job, out: Path, log_path: Path) -> None:
    if not gen.dir.is_dir():
        raise FileNotFoundError(f"{gen.dir} does not exist: run generators/{gen.script.parent.name}/setup.sh first")
    d = job.dir.resolve()
    meta = job.meta
    with tempfile.TemporaryDirectory(prefix=f"dfb-{gen.name}-") as tmp, log_path.open("w") as logf:
        env = {**os.environ, **gen.env,
               "DFB_VIDEO": str(d / "driving.mp4"), "DFB_VIDEO_25": str(d / "driving_25fps.mp4"),
               "DFB_AUDIO": str(d / "driving.wav"), "DFB_REF": str(d / "ref.png"),
               "DFB_REF_FRAME": str(d / "ref_frame.png"), "DFB_REF_AUDIO": str(d / "ref_audio.wav"),
               "DFB_DURATION": str(meta["duration"]), "DFB_PROMPT": meta["prompt"], "DFB_SEED": str(cfg.seed),
               "DFB_OUT": str(out.resolve()), "DFB_TMP": tmp}
        r = subprocess.run(["bash", str(gen.script)], cwd=gen.dir, env=env, stdout=logf, stderr=subprocess.STDOUT,
                           timeout=cfg.timeout_s)
    if r.returncode:
        raise RuntimeError(f"run.sh exited with {r.returncode}, see {log_path}")


def _run_comfy(cfg, gen: Generator, comfy: Comfy, job: Job, out: Path) -> None:
    c = gen.comfy
    workflow = json.loads(Path(c["workflow"]).expanduser().read_text())
    values = {"video": comfy.upload(job.dir / "driving.mp4", f"{job.stem}.mp4"),
              "ref": comfy.upload(job.dir / "ref.png", f"{job.stem}_ref.png"), "seed": cfg.seed}
    result = comfy.run(bind(workflow, c.get("inputs", {}), values), c.get("output_node", ""), cfg.timeout_s)
    comfy.download(result, out)


def summarize(gen: Generator, timings: Path) -> dict:
    """Latest record per clip, in the order clips first ran."""
    latest: dict[str, dict] = {}
    for line in timings.read_text().splitlines():
        rec = json.loads(line)
        latest[rec["clip"]] = rec
    ok = [r for r in latest.values() if r["ok"]]
    secs = [r["seconds"] for r in ok]
    rt = [r["seconds"] / r["output"]["duration"] for r in ok if r["output"] and r["output"]["duration"]]
    mem = [r["peak_mem_gb"] for r in ok if r["peak_mem_gb"] is not None]
    return {"generator": gen.name, "category": gen.category, "clips": len(latest), "ok": len(ok),
            "failed": [r["clip"] for r in latest.values() if not r["ok"]],
            "total_s": round(sum(secs), 1),
            "mean_s": round(statistics.mean(secs), 1) if secs else None,
            "median_s": round(statistics.median(secs), 1) if secs else None,
            "first_s": secs[0] if secs else None,
            "mean_after_first_s": round(statistics.mean(secs[1:]), 1) if len(secs) > 1 else None,
            "x_realtime": round(statistics.mean(rt), 1) if rt else None,  # generation seconds per output second
            "peak_mem_gb": max(mem) if mem else None,
            "output": ok[0]["output"] if ok else None}


class MemoryPeak:
    """Peak system memory in use above the level at the start, sampled every 0.5 s from /proc/meminfo. On a DGX
    Spark the GPU allocates from the same pool, so this covers the model too, and anything else running."""

    def __init__(self):
        self.peak_gb: float | None = None
        self._stop = threading.Event()

    def __enter__(self):
        base = _used_bytes()
        if base is None:
            return self
        self._base, self._peak = base, base
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(0.5):
            self._peak = max(self._peak, _used_bytes() or 0)

    def __exit__(self, *exc):
        if hasattr(self, "_thread"):
            self._stop.set()
            self._thread.join()
            self.peak_gb = round((self._peak - self._base) / 1e9, 1)


def _used_bytes() -> int | None:
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    except OSError:
        return None
    kb = lambda k: int(info[k].split()[0])  # noqa: E731
    return (kb("MemTotal") - kb("MemAvailable")) * 1024
