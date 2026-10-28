# BFS LTX-2.5 batch on an 8x B300 VM

Runs the BFS LTX-2.5 v1.1 head swap on every pair in `../pairs4` with one ComfyUI per GPU.

| Path | What |
|---|---|
| `Dockerfile`, `bfs/` | The image `squishydolphin/bfs-ltx25:b300`: ComfyUI v0.43.0, BFS/KJ/VHS nodes, PyTorch 2.11 + CUDA 13. No models inside. |
| `bfs/pipeline.py` | The BFS workflow as an API prompt, plus output size, fps and frame count per clip |
| `bfs/runner.py` | Batch runner and `bfsctl` (start, stop, status, selfcheck, models) |
| `bfs/models.py` | Model files (pinned Hugging Face revisions, size + sha256 checked) |
| `vm.sh` | Run from this PC: setup, test, full run, status, pull |
| `vm_setup.sh` | Copied to the VM by `vm.sh setup` |

## Run

```fish
sudo docker push squishydolphin/bfs-ltx25:b300   # once, from this PC
set -x BFS_VM user@VM_IP
./vm.sh setup          # uploads 9.4 GB of inputs, downloads 74 GB of models on the VM, starts 8 workers, self-check
./vm.sh test           # 10 pairs at 720p and 1080p
./vm.sh pull           # copies results to ../nebius_out every 2 min; open nebius_out/test/index.html
./vm.sh full 1080      # or 720
./vm.sh status
```

## What the pipeline does with each pair

- Settings, LoRA strength (0.8), sigmas, both passes and the LTXMultipleControls values are the ones in
  `workflow_ltx2.5_v1.1_head_swap_drag_and_drop.json`. The transformer and text encoder are the bf16 files.
- Prompt: the pair's `caption.txt`. Seed: the pair number, same at both resolutions.
- Size: the first pass runs at half the output size, in multiples of 32 closest to the clip's aspect ratio;
  1920x1080 clips come out 1280x704 at 720p and 1920x1088 at 1080p.
- Frame rate: the clip's own up to 30 fps; 50/59.94/60 fps clips run at half rate. Length: the largest
  8n+1 frame count that fits in the clip (137 frames at 24 fps, 177 at 30 fps).
- Audio: the A clip's original audio is copied onto the result (the drag-and-drop workflow saves silence).
- Output: `outputs/test/NNNN_720p.mp4` / `NNNN_1080p.mp4`, or `outputs/full_1080p/NNNN.mp4`, H.264 CRF 17.

## Interruptions

The container restarts with the VM. Models, inputs and finished videos are on the boot disk, a finished
video is only written once it has passed a size and frame-count check, and the runner skips pairs that are
done. A pair that fails three times is skipped and listed by `./vm.sh status`.

## If something fails

- `./vm.sh status` shows the last errors; `./vm.sh logs` follows the runner; each GPU's ComfyUI log is
  `/data/logs/comfyui_gpuN.log` on the VM.
- `ssh $BFS_VM bfsctl selfcheck` checks GPUs, PyTorch on every GPU, models, inputs and the workflow.
- `CUDA error 802` / `system not yet initialized`: `sudo systemctl restart nvidia-fabricmanager` on the VM.
- ComfyUI flags without rebuilding: `COMFYUI_ARGS="--disable-dynamic-vram" ./vm.sh setup` recreates the
  container with them (models and finished videos stay).
