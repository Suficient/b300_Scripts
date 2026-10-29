# Deepfake generator work — handoff (2026-10-28, 16:30 IST)

Moved from the Linux PC (unstable power) to the MacBook. Lives in the b300_Scripts repo
(github.com/Suficient/b300_Scripts, the PC's `~/Projects/Code/Precomputed-ltx`). Suggested first prompt on the Mac,
from a clone of that repo:

> Read handoff/GENERATOR_HANDOFF.md and the files it points to. Check the BFS run on the VM first (section 3), then
> continue Phase 2 (section 7): spawn the five verification agents at high effort as described there.

Contents:
- `handoff/GENERATOR_HANDOFF.md`: this file.
- `handoff/research/`: the three full research reports (identity swap; audio/motion-driven; full synthesis + threat
  landscape + detection benchmarks), with every source URL. Read the one for a generator before working on it.
- `handoff/df-benchmark/`: **reference only.** Older, separate work (a DGX Spark timing harness). It matters here
  because GENERATORS.md (the original list this audit started from) and its per-generator setup.sh/run.sh (how each
  repo installs and is called) live there, and because today's `generators/pairio.py` was written into it. Build
  the B300 work in this repo (section 5); don't modify df-benchmark.
- The rest of this repo: the pair builder + Nebius VM scripts (section 5).
- Sample pairs are not in git (they hold real faces). For local tests copy a few from the VM
  (`/data/inputs/pairs4/NNNN`, after the upload in section 7 adds the missing files) or from the PC's `pairs4/`.

## 1. What this is for

The user's startup (Discerning Agents; detection of deepfakes in trust-sensitive video calls) is building a video
deepfake detection training set and benchmark. Detector: V-JEPA 2.1 features (ViT-L, ~300M; bigger doesn't help at
this data size) + probe/fine-tune. Real data: 37 h of ~6 s single-face webcam clips (YouTube/Twitch) from Clipfarmer.

Guiding evidence (ScaleDF, arXiv 2510.16320, in the vault's References): detector error falls as a power law in the
number of **distinct fake methods** (exponent 0.375) and **real domains** (0.476), no saturation at 102 methods;
keeping all methods but 1/10 of the fakes only moved AUC 0.886 → 0.876. So: breadth of generators beats volume per
generator.

## 2. Decisions the user made (don't relitigate)

- Run **every generator on the same pairs**, as nested subsets of select500: the whole generation run gets **12 h on
  8 GPUs**, split by `allocate.py` (equal GPU-time cap per generator, 50-500 pairs each, from measured seconds per
  pair in `times.csv`). Then learning curves (V-JEPA probe at 100/250/500 clips where a generator has them,
  leave-one-generator-out) decide whether any generator gets more pairs.
- Generate ~5.2 s per fake (enough for the common 5 s cut), not the whole ~6 s A clip.
- Same pair through many generators is fine; split train/test **by person/source video** (A and B identities and all
  versions of a pair in one split). Real training clips must pass the same filters as the A clips (text-free etc.).
- All fakes and reals get cut to a common length (~5 s) later, so a model only needs >= 5 s. **MAGI-2 dropped**
  (114B/307 GB, no audio input, shares Wan's VAE). **MiniMax H3 dropped** (minimax-h3 and h3-swap): compute (Ref2VA
  ~20 GPU-min per 6 s clip, estimated from the vLLM recipe), ~0.6 TB of weights, and its licence.
- Non-commercial weights are allowed (research/commercial status undecided) but must be flagged.
- Quality-first settings (reference steps/resolution, no speed LoRAs, no step caching). Seed = pair number.
- Verification agents run at **high effort**.

## 3. The B300 VM and the BFS run (check this first)

- Nebius, 8x B300 SXM (sm_103, ~275 GB usable HBM each), 192 vCPU, 2.7 TB RAM, Ubuntu 24.04 (CUDA 13 driver),
  Docker + NVIDIA toolkit, user `sufi`, passwordless sudo, data on `/data`. The Mac's key
  (`Apple@MacBook-Pro.local`) is in `~/.ssh/authorized_keys`: `ssh sufi@<VM IP>` (ask the user for the IP; on the PC
  it was in `$BFS_VM`). The VM's shell is bash.
- Billing: stopping from the Nebius console/CLI stops GPU billing; disks keep billing; never `sudo shutdown` from
  inside (Nebius reboots it and keeps charging). Delete VM + disk only after everything is pulled.
- **Running now:** BFS V3 head swap (LTX-2.3, image `squishydolphin/bfs-v3-comfyui:latest`) over the pairs that
  didn't finish on the old VM. 8 containers `bfs23-0..7` (one per GPU, ComfyUI on 127.0.0.1:8188-8195), runner
  `python3 ~/bfs_v3/run_pairs.py /data/v3/full /data/outputs/full_v3 --gpus 8`, log `~/bfs_v3/resume.log`.
  Restarted 10:30 UTC (the VM's clock is UTC; 16:00 IST) with 2,749 pairs left. Measured at 11:03-11:13 UTC:
  ~430 pairs/hour (60-65 s per pair per GPU), 2,519 left, so it should finish ~17:00 UTC (~22:30 IST).
- **Memory leak found and (probably) fixed:** each ComfyUI kept ~7.5 GB of host RAM per job (300 GB per container
  after ~39 jobs; the VM nearly ran out). Containers were recreated with
  `--memory 320g --restart unless-stopped -e MALLOC_MMAP_THRESHOLD_=4194304 -e MALLOC_ARENA_MAX=4`.
  **Unconfirmed**: check that memory is flat:
  ```bash
  D="docker"; docker info >/dev/null 2>&1 || D="sudo docker"
  $D stats --no-stream --format '{{.Name}}  {{.MemUsage}}' | sort -V; grep -c "done in" ~/bfs_v3/resume.log
  tail -n 5 ~/bfs_v3/resume.log; ls /data/outputs/full_v3 | grep -c '^[0-9]*\.mp4$'
  ```
  If a container hit the 320 GB cap it restarted itself; its worker then waits up to 30 min (JOB_TIMEOUT) on the
  lost job and logs it FAILED. When the runner prints `finished; N failed`, rerun the same runner command (it skips
  finished pairs) to retry them.
- GPUs idled ~50% (each ComfyUI does CPU work — decode, 3 ffmpeg encodes — between GPU phases). A second ComfyUI per
  GPU would fix that, but only if memory is flat. Not worth doing this late in the run.
- The Mac is already running `./vm.sh pull` (copies `/data/outputs/` into `nebius_out/` every 2 min). The PC also
  has the 146 results the old VM made, in its `nebius_out/full_v3`.
- **When BFS is done** (the pull count stops rising): `BFS_VM=... ./v3_resume.sh stop` removes `bfs23-0..7`.
- VM gotchas learned: `docker rm -f` on a container holding hundreds of GB returns before it's gone (retry until
  `docker inspect` fails); `pkill -f "[b]fs_v3/..."` bracket pattern avoids matching the caller; pasted blocks'
  last line may need Enter; `setsid ... &` prints `[1]+ Done` immediately (normal); a dead ComfyUI makes
  run_pairs.py fail every pair it picks up within milliseconds (watch for FAILED storms); images built for CUDA
  12.8 need `nvidia-cuda-nvrtc-cu12==12.9.86` to JIT kernels for sm_103.

## 4. The final generator list (23 methods)

Families matter more than individual models (artifacts differ more between families). Status is relative to
`GENERATORS.md` in df-benchmark (now superseded by this table). Details, numbers and sources: `research/`.

| # | Folder name | Method | Type | Family | Status | Key facts / flags |
|---|---|---|---|---|---|---|
| 1 | facefusion-inswapper | inswapper_128_fp16 + GFPGAN (Deep-Live-Cam setup) | face swap | per-frame GAN | keep | Most used in fraud/live tools (Deep-Live-Cam 97k★, ReActor, VisoMaster); CASIA benchmark top ID 96.9%. HF facefusion/models-3.0.0. NC |
| 2 | facefusion | HyperSwap 1A 256 | face swap | per-frame GAN | keep | FaceFusion default since **3.3.0 (Jun 2025)**, not 3.9; weights ResearchRAIL = NC; models-3.3.0, 403 MB |
| 3 | facefusion-alphaface | AlphaFace 256 | face swap | per-frame GAN | new | FaceFusion 3.9.0 (Sep 2026), models-3.9.0, 556 MB; code MIT, weights NC |
| 4 | dreamid-v | DreamID-V (DWPose or Faster variant) | face swap | Wan2.1-1.3B | keep | Only open video-diffusion swapper; HF XuGuo699/DreamID-V 12.9 GB + Wan 1.3B T5/VAE; SDPA fallback works; Apache but README says academic/demo |
| 5 | canonswap | CanonSwap (ICCV 2025) | face swap | warping, no diffusion | new | New family; Pixel-Talk/CanonSwap, weights on Google Drive, ResearchRAIL-M (NC); pinned torch 2.3 cu118 → upgrade |
| 6 | bfs-ltx23 | BFS head swap v3 | head swap | LTX-2.3 | running | Being made on the VM now (section 3); nothing to do |
| 7 | wan-animate | Wan2.2-Animate-14B replace + relighting LoRA | whole person | Wan2.2-14B | keep | 72 GB; `--replace_flag --use_relighting_lora`; ~160 s/124 frames at 480x832 on B200 (Viggle). Doc claim "only one that replaces the whole person" is outdated |
| 9 | liveportrait | LivePortrait | reenactment | keypoint warping | keep | Repo now KlingAIResearch/LivePortrait, HF KlingTeam/LivePortrait 2.1 GB; InsightFace detector NC; avoid FasterLivePortrait's TensorRT 8 path (no Blackwell) |
| 10 | personalive | PersonaLive | reenactment | SD-1.5, real-time | keep | NC (README says academic only despite HF Apache); on Blackwell use `--acceleration none` (xformers broken), rebuild any TRT engine |
| 11 | wan-animate-2 | Wan2.2-Animate-2-14B (animation only) | reenactment | Wan2.2-14B | new | Aug 2026, Apache, 33 GB; >70% preferred over Wan-Animate in its study; pins torch 2.7 cu126 → use cu130 |
| 12 | infinitetalk | InfiniteTalk image-to-video | talking head | Wan2.1-I2V-14B | keep | Most-used open talking head; Doc's "best lip sync" claim is wrong (SoulX-FlashTalk leads Sync-C); pins torch 2.4.1 → override; ~3.5 min/81-frame window on H200 (third party) |
| 13 | infinitetalk-v2v | InfiniteTalk video-to-video dubbing | lip sync (full head) | Wan2.1-I2V-14B | new | Same install; A's clip + B's audio; regenerates the frame (background can drift) |
| 14 | soulx-flashtalk | SoulX-FlashTalk-14B (4 NFE, streaming) | talking head, real-time | Wan 14B distilled | new | Live-call class; Soul-AILab/SoulX-FlashTalk-14B 54.5 GB, Apache; >64 GB on one GPU |
| 15 | longcat-avatar | LongCat-Video-Avatar 1.5 | talking head | LongCat 13.6B | keep | Best open Sync-C in the independent Kandinsky S2V table; MIT; 75 GB + LongCat-Video 83 GB; distilled path needs `--use_distill --model_type avatar-v1.5` + DMD2 LoRA (verify) |
| 16 | hunyuan-avatar | HunyuanVideo-Avatar | talking head | HunyuanVideo 13B | keep | Best CSIM; 81 GB; 129 frames = 5.2 s; licence excludes EU/UK/KR; pins torch 2.4/FA 2.6.3 → override |
| 17 | hallo2 | Hallo2 | talking head | SD-1.5 | keep, low priority | 512² face crop only; unmaintained; 2nd-hardest in TalkingHeadBench |
| 18 | latentsync | LatentSync 1.6 | lip sync | SD-1.5 inpainting | keep | Best automatic lip-sync metrics; 9.6 GB; code Apache, **weights OpenRAIL++** (fix doc); 18 GB VRAM |
| 19 | musetalk | MuseTalk 1.5 | lip sync, real-time | single-step inpainting | keep | What live tools (LiveTalking) run; 256 px; mmcv from source; pins torch 2.0.1 cu118 → override |
| 20 | wav2lip | Wav2Lip | lip sync | GAN, 96 px | keep | NC; still in live tools |
| 21 | x-dub | X-Dub | lip sync, mask-free | Wan2.2-TI2V-5B | keep | 30 GB, ~21 GB VRAM; mmcv from source; public release isn't the paper's version |
| 22 | ltx-a2v | LTX-2.5 audio-to-video | full synthesis | LTX 22B | keep, **pilot 20 pairs first** | Custom-audio lip sync may not work on 2.5 yet (HF Lightricks/LTX-2.5 discussion #44); check SyncNet; fallback LTX-2.3 A2V, or 2.5 speaking A's transcript. Gated; NVFP4 build runs natively on Blackwell |
| 23 | davinci-magihuman | daVinci-MagiHuman | full synthesis | MagiHuman 15B | new | GAIR/daVinci-MagiHuman 216 GB, Apache; ref image + text (makes its own speech); **may build Hopper-only FlashAttention kernels → B300 blocker, needs SDPA/FA2 fallback** |
| 24 | kandinsky6 | Kandinsky 6.0 Video Pro (distilled, 5 s) | full synthesis | Kandinsky CrossDiT 29B | new | kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers, 80 GB, MIT; first frame + text, generates lip-synced speech; ~480p + 1.4B 1080p upscaler |

Dropped (reasons in `research/`): MiniMax H3 and its swap LoRAs (section 2), SCAIL-2 (its replacement masks need the
gated SAM3, which we skip), MAGI-2, MoCha (AGPL), Viggle-Animate (H3 licence + repainted first frame),
HunyuanCustom, UniAnimate-DiT, Phantom, SimSwap, GHOST, Stand-In (experimental), TAVR (FA3, Hopper-only),
EMOPortraits (watch), Kandinsky S2V (no weights), AptAvatar (cheaper InfiniteTalk substitute if needed),
LiveAvatar/FlashHead/Wan2.2-S2V and other Wan talking heads (weaker or redundant). Watch list: LivingSwap,
StreamingSwap, DirectSwap, LTX Dub-It on 2.5.

Transcripts: Kandinsky 6, daVinci-MagiHuman (and the LTX fallback) speak text, so they need A's words. Plan:
one Whisper large-v3 pass over all pairs on the VM, written as `a_transcript.txt` in each pair folder.

Later passes, not generators: a video-call-quality encode of every real and fake (WebRTC-like H.264/VP8,
0.5-2.5 Mbps, 720p, drops); an inswapper live-call variant (no enhancer); open voice clone + lip sync;
synthetic identities (image model from caption → animate); CodeFormer/RestoreFormer variants; a small purchased
eval set from closed models (Veo 3.1, Kling, Seedance, Grok), which DF26 found hardest for detectors.

Compute: with the estimates in `times.csv` (not measured), 500 pairs of everything would be ~560 GPU-h (~70 h on 8
GPUs); the 12 h budget gives the slow Wan/Hunyuan/InfiniteTalk-class models ~50 pairs and the per-frame ones
325-500. Replace the estimates with the 8-pair timings, then rerun `allocate.py`.

## 5. Code state

**Where the B300 generator work goes:** this repo, in a new `generators/` folder (`generators/<name>/setup.sh` +
`batch.py`, shared helpers next to them). Never add Co-Authored-By/Claude trailers to commits.

**df-benchmark, reference only** (`handoff/df-benchmark`; the PC's `~/Projects/Code/df-benchmark` has uncommitted
work from 2026-10-07; leave it alone). Useful pieces to copy from, not to extend:
- Built for a DGX Spark (aarch64, GB10 sm_121, Python 3.11, torch cu130). `generators/<name>/setup.sh` installs into
  `$GEN_ROOT/<name>` (default `~/df-gen`) using helpers in `generators/common.sh`: `clone URL [COMMIT]`,
  `venv [PY] [notorch]`, `install PKG...` (never replaces torch), `reqs FILE...` (drops torch/xformers/flash-attn,
  decord→decord2, mediapipe→0.10.14, onnxruntime→onnxruntime-gpu), `flash_attn` (builds FA 2.8.3 for **sm_120**),
  `mmlab` (mmcv for **sm_120**), `fetch URL DIR`, `hf ARGS` (`hf download`). `env.sh` points every cache into
  `$GEN_ROOT`. On the B300 the arch lists become sm_100 (its code runs on sm_103); no Spark support is needed here.
- `generators/<name>/run.sh` is a one-clip-per-process timing harness (env contract in `generators/lib.sh`:
  DFB_VIDEO, DFB_VIDEO_25, DFB_AUDIO, DFB_REF, DFB_REF_FRAME, DFB_REF_AUDIO, DFB_PROMPT, DFB_SEED, DFB_TMP, DFB_OUT,
  DFB_DURATION). Existing run.sh files show how each repo is called (useful starting points).

**`pairio.py` (new; in `handoff/df-benchmark/generators/`, move it to this repo's `generators/`; tested on CPU with a
dummy generator):** batch mode. Each generator gets
`generators/<name>/batch.py`:
```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairio import run
SETTINGS = {...}                      # copied into each NNNN.json
def load(): return model              # once per worker process
def make(model, p):                   # p: Pair; write into p.tmp
    return video_path, audio_path     # audio_path: any file; its first audio stream becomes the fake's sound
if __name__ == "__main__":
    run(load, make, SETTINGS)
```
Started as `batch.py --pairs LIST --out OUT_DIR` with `CUDA_VISIBLE_DEVICES` = one GPU. Skips pairs with
`OUT_DIR/NNNN.mp4`, writes NNNN.mp4 atomically (video stream copied, audio as AAC, cut to the shorter of A's clip
and the video) + NNNN.json (seconds, gpu, seed, settings); logs and skips failures (rerun retries). `Pair` fields:
`name, dir, tmp, video (a_guide.mp4), ref (b_face_512.jpg), ref_1024, ref_frame (b_frame.jpg), ref_video
(b_source.mp4), seed, prompt (caption.txt), job (job.json), duration, fps`, and on-demand `video_25`, `audio`
(A 16 kHz mono wav), `ref_audio` (B 16 kHz wav cut to A's length). Helpers: `frames(seconds, fps, step, offset)`,
`ffmpeg(...)`, `finish(...)`, `probe(...)`. Optional env `PAIRIO_TMP` for scratch.

**This repo** (b300_Scripts; on the PC `~/Projects/Code/Precomputed-ltx`). `build.py` builds the A/B pairs (`pairs4/`, 3,228 pairs, 18 GB, only on
the PC; the VM has a subset under `/data/inputs/pairs4`, see 7). `select_pairs.py` picks the shared pair set
(`select500.txt`, with every candidate's numbers and drop reasons in `select500.csv`; see 7). `allocate.py` + `times.csv` = pairs per generator for the run budget
(writes `lists/<name>.txt`, first n of a group-balanced ordering `lists/order.txt`). `bfs_v3/run_pairs.py` = BFS ComfyUI batch driver.
`nebius/` = VM tooling: `vm.sh` (setup/test/full/status/pull; the LTX-2.5 BFS image), `v3_test.sh`, `v3_full.sh`,
`v3_resume.sh` (fresh-VM resume of the LTX-2.3 run), `actions/actions.py` (Qwen3.5-27B ACTION captions).

## 6. Licence flags for the user (flag, don't block)

- NC weights: inswapper, HyperSwap, AlphaFace, CanonSwap, PersonaLive, Wav2Lip, LivePortrait's InsightFace detector
  (and InsightFace ArcFace generally).
- LatentSync weights: OpenRAIL++. Hunyuan-Avatar: excludes EU/UK/South Korea. DreamID-V: README says academic/demo.
- LTX-2.x licence: free under $10M revenue; its acceptable-use policy (section 7) bans non-consensual impersonation
  ("deepfakes"). That already covers the BFS run.
- Gated on Hugging Face (user must accept terms with their account; the VM needs `HF_TOKEN`): LTX-2.5 (and the
  LTX-2.3 IC-LoRAs) at least; verify the others' `gated` flag via `https://huggingface.co/api/models/<repo>`.

## 7. Phase 2: what's left (not started — the five agents were never completed)

Goal: for every generator in section 4 (except bfs-ltx23), a verified recipe that installs it on the B300 VM,
downloads its weights, and runs it in batch mode over the pair folders, tuned for quality and B300 throughput.

**Where the work runs:** the Mac writes the scripts and checks pins/licences/gating (GitHub and HF APIs) and
dependency resolution (`uv pip compile ... --python-platform x86_64-manylinux_2_28 --python-version 3.11
--extra-index-url https://download.pytorch.org/whl/cu130`; macOS has no CUDA). Real installs, downloads and tests
happen on the VM over SSH, **while BFS keeps running**: installs and downloads only use CPU, network and /data
disk; a 2-pair smoke test can share a GPU with BFS (each BFS container uses ~50 of ~275 GB), slowing BFS a little,
or wait until BFS ends. Keep installs off `/data/v3` and `/data/outputs/full_v3`, and don't touch the `bfs23-*`
containers or `~/bfs_v3`.

Verification levels to report per generator: **L0** weights (HF repo + revision sha + files + size, gated?,
licences; non-HF links answer); **L1** install resolves/installs on x86_64 + torch cu130 (overridden pins listed);
**L2** every compiled CUDA dependency has sm_100/sm_103 code or is avoided (prefer PyTorch SDPA; FA2 with sm_100 is
fine; FA3 and sm_90a/sm_100a arch-specific cubins are not; check torch's arch list with
`torch._C._cuda_getArchFlags()`); **L3** batch.py code path exercised (CPU end-to-end for small models on a sample
pair; imports/meta-device build for big ones); **L4** GPU smoke test on the VM (2 pairs, ffprobe: video + audio,
>= 4.9 s; look at a frame).

B300 tuning rules: reference quality settings; bf16 (or the authors' released fp8); no CPU offload, no low-VRAM
modes, no parallelism unless one GPU can't hold the model; K workers per GPU when the pipeline has big CPU phases
and VRAM allows (report VRAM per worker); SDPA/FA2; no growing per-pair caches; driver sets
`MALLOC_MMAP_THRESHOLD_=4194304 MALLOC_ARENA_MAX=4`; seed = pair number; best native resolution with framing kept;
high-quality encode (libx264 crf <= 17, yuv420p); right audio (A's for swaps/reenactment/talking heads driven by A,
B's for lip sync).

**The shared pair set:** `select_pairs.py` (run on the PC, where pairs4 and the Clipfarmer snapshot are) picked
500 of the 3,082 pairs on the VM most likely to give good fakes with every generator. Hard filters: both clips
>= 50% speech (webrtcvad), A's yaw <= 25 and pitch <= 20, no hands near A's or B's face, no sunglasses or headwear,
B's crop >= 200 px and >= 0.8 ArcFace match to B, age gap <= 15 (1,852 pass). Then a z-scored quality score (A
sharpness, bitrate, face size, frontal, steady, well lit, no glasses; B crop size, identity match; age and hair
match; speech) and a greedy pick: each person at most once as A and once as B (51 A people appear twice to reach
500), gender x skin groups at their share of the candidates. The VM already had a_guide.mp4, b_face_1024.png and
caption.txt for these; the PC uploads b_source.mp4, b_frame.jpg, b_face_512.jpg and job.json for them (~0.7 GB)
plus `/data/inputs/select500.txt`. Check on the VM that `/data/inputs/pairs4/<each pair>/b_source.mp4` exists.

Agent split (each at **high effort**; only the platform agent edits the shared files in `generators/`
(`pairio.py`: bug fixes only, API frozen); generator agents own their `generators/<name>/` folders and request
shared changes as diffs; one folder per method, extra method folders reuse another install):
1. **Platform:** `generators/common.sh` for x86_64/B300, adapted from df-benchmark's helpers (torch cu130 and its
   arch list, FA2 for sm_100 (prebuilt wheel if trustworthy, else VM build recipe), mmcv arch, decord vs decord2,
   mediapipe pin, onnxruntime-gpu with cuDNN 9 on a CUDA 13 driver, nvcc for source builds); `setup_b300.sh` (VM
   installer, GEN_ROOT=/data/gen); the batch driver (round-robin per-worker lists, K workers per GPU, `--gpus`,
   `--list`/`--limit`, start/status/stop/smoke, detached, logs, outputs in OUT_ROOT/<name>/; a dead worker must not
   fail the queue; runs on the GPUs it's given so smoke tests can share a GPU with BFS); a README "B300" section.
2. **Small/per-frame models:** facefusion-inswapper, facefusion, facefusion-alphaface, canonswap, liveportrait,
   personalive, wav2lip, musetalk.
3. **Wan family swaps/animation:** dreamid-v, wan-animate, wan-animate-2, x-dub.
4. **Talking heads + diffusion lip sync:** infinitetalk, infinitetalk-v2v, soulx-flashtalk, longcat-avatar,
   hunyuan-avatar, hallo2, latentsync.
5. **Full synthesis:** ltx-a2v (+ 20-pair pilot plan and SyncNet check), davinci-magihuman, kandinsky6, and the
   Whisper transcript pass.

Each agent reports: level reached per generator, exact pins, files created/changed, settings + K + expected VRAM
and seconds/pair (with basis), blockers and fixes, shared-file diffs requested, gated repos, disk left behind.

Then: run the platform installer + smoke tests on the VM, time 8 pairs per generator (into `times.csv`),
rerun `allocate.py`, then the run over each `lists/<name>.txt` once BFS is finished.

## 8. Working with this user

- Terse; wants the smallest change that does the job; lean comments; docs describe the current state (no "used to
  be X"); flag optional extras instead of adding them.
- Prefers measured numbers over estimates; say when something is an estimate.
- Cheap reliable workarounds over long root-causing unless asked.
- PC shell is fish (no heredocs); VM shell is bash; give paste-able blocks in small parts with what to check
  between them.
- Commits: never add Co-Authored-By or Claude attribution.
