# df-benchmark

Times deepfake generators on a DGX Spark. Every clip in `clips/` (meant for 10 six-second Clipfarmer clips) is
faked once by each generator, and each run's wall time and memory are recorded. The fakes go to
`results/<generator>/`.

For each clip of person A, `prepare` finds the most similar person B in Clipfarmer's person index who is not
A (ArcFace cosine below `same_person_sim`, A's own person id excluded), takes B's highest-scoring accepted
clip (Clipfarmer's window score), and picks its largest x sharpest face frame as the reference. Every generator
then fakes the same A/B pairs. Which 15 generators, and why: [GENERATORS.md](GENERATORS.md).

## Run

```sh
uv sync
cp config.example.toml config.toml        # paths to Clipfarmer's DB and clips, install root
cp /path/to/10/clips/*.mp4 clips/
uv run df-benchmark -c config.toml prepare            # -> work/<clip>/
uv run df-benchmark -c config.toml run latentsync wav2lip   # -> results/<generator>/
uv run df-benchmark -c config.toml report             # one row per generator
uv run df-benchmark -c config.toml list               # generators, category, installed or not
```

`run` skips clips that already have a result (`--force` redoes them, `--only CLIP...` limits to some clips).
A clip that fails is logged and recorded; the run goes on with the next one.

## What each generator gets

`prepare` writes `work/<clip>/`:

| File | What |
|---|---|
| `driving.mp4`, `driving_25fps.mp4` | A's clip, as is and at 25 fps |
| `driving.wav` | A's audio, 16 kHz mono |
| `ref.png` | B's face, square crop (face box x 2), 512 px |
| `ref_frame.png` | the whole frame `ref.png` was cut from |
| `ref_audio.wav` | B's audio from the same clip, cut or padded to A's length |
| `job.json` | the match: person ids, similarity, B's clip and its score, prompt |

| Type | Input | Output |
|---|---|---|
| face swap | A's clip + B's face | A's clip with B's face |
| reenactment | B's image + A's clip as motion | B moving like A |
| talking head | B's image + A's audio | B saying A's words |
| lip sync | A's clip + B's audio | A mouthing B's words |
| full synthesis | B's image + A's audio + prompt | a new video of B saying A's words |

Every output gets A's audio, except lip sync, which keeps the audio it was driven by (B's).

## What is measured

`results/<generator>/timings.jsonl` has one line per clip run; `summary.json` the latest run of each clip.

- **seconds**: wall time of the whole run.sh, so it includes loading the model. Script generators start
  fresh for each clip; a production loop that keeps the model loaded would be faster by about the load time.
  `first_s` vs `mean_after_first_s` shows the cold page cache on the first clip.
- **x_realtime**: generation seconds per second of output video.
- **peak_mem_gb**: peak system memory in use above the level at the start of the clip. On the Spark the GPU
  allocates from the same 128 GB pool, so this includes the model.

Both numbers are for the whole machine: run the benchmark on a Spark that isn't also running Clipfarmer
workers or the VLM, or the times and memory include theirs.

## Setup on the Spark

One script installs everything on a fresh Spark: system packages, uv, this repo's environment and
`config.toml`, then every generator. It takes hours (~650 GB of downloads, two source builds), so run it in
tmux:

```sh
rsync -a --exclude .venv --exclude work --exclude results ~/Projects/Code/df-benchmark/ <spark>:~/imprint/df-benchmark/
ssh <spark>
cd ~/imprint/df-benchmark && tmux new -s setup "HF_TOKEN=hf_... ./setup_spark.sh"
```

`HF_TOKEN` is for ltx-a2v, whose weights are gated: accept the terms on huggingface.co/Lightricks/LTX-2.5 with
the token's account. Without it ltx-a2v is skipped; log in later (`uvx --from huggingface_hub hf auth login`)
and run `./setup_spark.sh ltx-a2v`.

It asks for sudo once at the start, then runs unattended. Each generator's install log is in
`~/df-gen/logs/<name>.log`. A failed generator doesn't stop the others; the summary lists what failed. Rerun to
retry: finished generators are skipped. `./setup_spark.sh NAME...` installs only those, `--force NAME`
reinstalls one.

What each `generators/<name>/setup.sh` does (shared steps in `generators/common.sh`):

- Clones the upstream repo into `~/df-gen/<name>`. Makes a venv with Python 3.11 and the CUDA 13 ARM64 builds of
  torch (FaceFusion and ComfyUI use 3.12). Installs the repo's requirements and downloads the weights.
- Packages with no ARM64 build are dropped from requirement files and replaced where needed: `decord` ->
  `decord2` (same import), `onnxruntime` -> `onnxruntime-gpu`, `mediapipe` pins -> 0.10.14, `xformers` dropped.
- FlashAttention 2 (the Wan-based repos assert on it) and mmcv with CUDA ops (MuseTalk, X-Dub) have no ARM64
  wheels. They are built once into `~/df-gen/.wheels` and reused; the first build takes about an hour.
- `bfs-ltx23` installs ComfyUI with the four custom nodes, the LTX-2.3 + BFS weights and Ollama's
  `qwen3.5:9b`. It then exports the workflow to `workflows/bfs_v3_api.json`, applying fake-pipeline's two
  manual fixes and a 5.7 s length (137 frames, the longest that fits a 6 s clip). ComfyUI is stopped after
  the export; `generators/bfs-ltx23/comfy.sh start|stop` runs it for the benchmark. While it runs it holds its
  models in memory, so keep it stopped when benchmarking other generators.

`facefusion`, `wav2lip`, `setup_spark.sh` itself and the BFS workflow export have been run (x86, CPU). The
other generators' scripts are written from the upstream READMEs and source and haven't been run on a Spark
yet: expect to fix version pins and paths in some of them on the first install.

## Uninstall

```sh
./uninstall_spark.sh          # lists what it will remove, asks, removes
./uninstall_spark.sh --all    # also clips/, work/, results/ (faces) and this checkout
```

It removes `~/df-gen` and everything the install put outside it, as recorded in
`~/.local/state/df-benchmark/manifest`:
- apt packages that weren't installed before
- uv, if the install added it
- Ollama, or only `qwen3.5:9b` if Ollama was already there
- this repo's `.venv`, `config.toml` and BFS export
- the few home-directory paths generators write to that didn't exist before

Anything that was on the Spark before the first install stays. Every cache the generators use (uv, pip,
Hugging Face, torch hub, Playwright, Triton, CUDA) points into `~/df-gen` during install and benchmark runs
(`generators/env.sh`), so it goes with it. If purging the recorded apt packages would also remove something
installed since, the packages are kept and the command to remove them is printed.

## Adding or changing a generator

A generator is `generators/<name>/setup.sh` + `run.sh`, plus an entry in `GENERATORS` in
`df_benchmark/generators.py`. run.sh runs in the install directory with `DFB_*` variables pointing at the
job's files (listed in `generators/lib.sh`) and writes `$DFB_OUT`. Settings that vary per run are environment
variables read by run.sh (documented at its top); set them in the config:

```toml
[generators.dreamid-v-480]
script = "dreamid-v"
env = { DREAMIDV_SIZE = "832*480" }
```

## Tests

```sh
uv run pytest     # identity matching, crops, config, requirement filtering, a fake generator end to end
```

## Code

| What | Where |
|---|---|
| CLI | `df_benchmark/__main__.py` |
| Settings | `df_benchmark/config.py` |
| Match each clip to a different person, write the job folder | `df_benchmark/prepare.py` |
| Clipfarmer's person index (read-only) | `df_benchmark/identities.py` |
| Face detection + embedding (SCRFD + ArcFace, non-commercial weights), copied from fake-pipeline | `df_benchmark/faces.py` |
| Generator registry | `df_benchmark/generators.py` |
| Run and time a generator, memory sampling, summaries | `df_benchmark/run.py` |
| ComfyUI client, copied from fake-pipeline | `df_benchmark/comfy.py` |
| ffmpeg / ffprobe helpers | `df_benchmark/media.py` |
| Per-generator install and run scripts | `generators/` |
