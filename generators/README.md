# Generators

One folder per method: `<name>/setup.sh` (installs it) and `<name>/batch.py` (runs it over pair folders through
`pairio.py`). Shared: `common.sh` (setup helpers), `env.sh` (cache locations), `pairio.py` (batch mode),
`drive.py` (runs workers), `setup_b300.sh` (VM installer).

## B300

The VM runs everything from `~/gen` (a copy of this folder). From the Mac:

```bash
rsync -a generators/ sufi@<VM IP>:~/gen/
```

### Layout

| Path | What |
|---|---|
| `/data/gen/<name>/` | the install: clone + `.venv` (a method that reuses another's install is a symlink to it) |
| `/data/gen/.cache`, `.python`, `.wheels/cp311-torch2.11.0+cu130/` | uv/HF/torch caches, uv's Pythons, flash-attn and mmcv wheels built once |
| `/data/gen/logs/<name>.log` | setup log |
| `/data/gen_out/<name>/NNNN.mp4`, `NNNN.json` | outputs; `.drive/` holds worker lists, logs, state and smoke frames |
| `/data/gen_tmp/<name>/wNN/` | worker scratch (`PAIRIO_TMP`), wiped when the worker starts |

Stack: Python 3.11.18 (uv-managed), torch 2.11.0+cu130 (arch list sm_75-sm_120 incl. sm_100, which runs on the
B300's sm_103), flash-attn 2.8.3.post2 and mmcv 2.1.0 built for sm_100, onnxruntime-gpu 1.24.4 (CUDA 12 build: the CUDA 13 ones lack sm_100 code;
its compute_90 PTX JITs for sm_103), decord2 3.4.0, mediapipe 0.10.14.

### Set up

```bash
~/gen/setup_b300.sh                      # apt packages (ffmpeg, git-lfs, build tools), uv, hf CLI, /data/gen
~/gen/setup_b300.sh latentsync wav2lip   # + these generators; skips finished ones (--force reinstalls)
```

Gated weights need `hf auth login` (or `HF_TOKEN`) first. A generator's `setup.sh` can also run alone:
`bash ~/gen/<name>/setup.sh` (installs into `/data/gen/<name>`).

### Run

```bash
python3 ~/gen/drive.py smoke NAME --gpus 3 [--redo]     # pairs 0142 0143 in the foreground, ffprobe check
python3 ~/gen/drive.py start NAME [--gpus 0,1,2] [-k 2] [--list FILE] [--limit 8]
python3 ~/gen/drive.py status [NAME...]
python3 ~/gen/drive.py stop NAME...
```

- `smoke` passes when both outputs have a video and an audio stream and last >= 4.9 s; it saves a frame at 2.5 s
  to `/data/gen_out/NAME/.drive/smoke_NNNN.jpg`. `--redo` deletes those two outputs first (outputs are kept and
  skipped otherwise, so redo after changing settings).
- `start` defaults: every GPU, 1 worker per GPU, all of `/data/inputs/select500.txt` (pair names, or folder
  paths). The list is split round-robin over GPUs x K workers. It returns at once; a detached supervisor runs
  the workers and restarts one that dies (up to `--restarts 3`, the pair it died on moved to the end of its list).
  A failing pair is logged and skipped; `start` again retries failed pairs and skips finished ones.
- `--gpus` takes any ids, so a smoke test or a few workers can share a GPU with other jobs.
- `status` shows per generator and per worker: done, failed, left, alive/exit code, restarts, last log line, and
  the mean seconds per pair per worker (from the `NNNN.json` files).
- Workers run `cd /data/gen/NAME && .venv/bin/python ~/gen/NAME/batch.py --pairs LIST --out /data/gen_out/NAME`
  with `CUDA_VISIBLE_DEVICES`, env.sh's caches, the venv's `nvidia/*/lib` on `LD_LIBRARY_PATH`,
  `MALLOC_MMAP_THRESHOLD_=4194304 MALLOC_ARENA_MAX=4` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
- `OUT_ROOT`, `GEN_ROOT` and `GEN_TMP` override `/data/gen_out`, `/data/gen` and `/data/gen_tmp`.
