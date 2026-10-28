## Identity-replacement audit (face swap, head swap, character replacement, live tools)

Numbers are from the primary sources listed at the end. HF dl means Hugging Face downloads over the last 30 days. Anything from a single source or the vendor itself is marked.

### 1. Verdicts on the current picks

| Pick | Verdict | Evidence |
|---|---|---|
| facefusion HyperSwap 1A 256 | **Keep, but fix the row** | It is FaceFusion's default swapper, but it shipped in **FaceFusion 3.3.0 (Jun 2025)**, not 3.9. Its weights are **ResearchRAIL, i.e. non-commercial** (FaceFusion's own licenses page), and it depends on InsightFace ArcFace, also non-commercial. There is no paper and no independent metrics. 403 MB ONNX. |
| facefusion-inswapper (+GFPGAN) | **Keep (highest priority)** | Top identity score of 14 open swappers in the CASIA benchmark (May 2026): ID retrieval 96.9%, similarity 0.73. In the StreamingSwap video benchmark it has ID 0.735, FVD 138, 28.6 FPS on a 4090, beating DreamID-V on pose, expression, lighting and FVD. It is what Deep-Live-Cam (97.3k stars), ReActor, Rope/VisoMaster and DeepFaceLive's "Face Swap (Insight)" module run. Non-commercial. |
| dreamid-v | **Keep** | Still the only open-weights video-diffusion face swapper. Better options exist on paper but have no weights (LivingSwap, StreamingSwap). Independent check (StreamingSwap paper): ID 0.734, but FVD 231 and pose error 9.3 against inswapper's 6.1, at 1.8 FPS. Use the `-DWPose` or `-Faster` variant (Jan 2026). Apache-2.0, but the README says academic and demo use only. |
| bfs-ltx23 | **Keep** | BFS LTX-2.5 v1.1 LoRAs also exist (Aug 24 2026). Same family, so adding them buys no breadth. |
| wan-animate (2.2-Animate-14B, replace mode) | **Keep, but fix the evidence line** | GENERATORS.md says it is "the only one that replaces the whole person". That is no longer true (MoCha, SCAIL-2, the H3 swap LoRAs, Viggle-Animate). Its successor Wan-Animate-2 (Aug 2026) has **no replacement mode**: it is animation only. |

### 2. Ranked candidates per subcategory

**a) Per-frame face swap**

| # | Model | Quality evidence | Prevalence | Date / license | Weights | Notes |
|---|---|---|---|---|---|---|
| 1 | inswapper_128(_fp16) | see above | Deep-Live-Cam, ReActor, VisoMaster, DeepFaceLive Insight module | 2023, NC | facefusion/models-3.0.0, 555/278 MB | Exists at 128 px only. The 512 and 512_live versions are closed (sold through Picsi.ai). |
| 2 | HyperSwap 1A/1B/1C 256 | none published | FaceFusion default | Jun 2025, ResearchRAIL | models-3.3.0, 403 MB each | |
| 3 | **AlphaFace 256** (new) | FF++: ID retrieval 98.8, pose 1.24, FID 2.71, 24 ms on a 4090. Self-reported and not compared with inswapper or HyperSwap | FaceFusion 3.9.0 (Sep 2026), VisoMaster-Fusion | Jan 2026. Code MIT; weights NC per FaceFusion | models-3.9.0, 556 MB | Costs almost nothing to add, since FaceFusion is already installed. |
| 4 | SimSwap 256/512 | CASIA 90% / 0.61 | 5.2k stars, in DF40 | 2020, NC | | Old and already in public benchmarks. |
| 5 | GHOST 1/2/3 | WACV-W 2025: 91.7% vs inswapper 94.1% | low | Apache-2.0, the only commercially licensed swapper (still needs ArcFace) | | |

Drop per-frame diffusion swappers (DiffSwap: CASIA 15.6% retrieval, 46 s per image; REFace: CC-BY-NC; FaceAdapter: video ID 0.129). They are slow, flicker, and nobody uses them.

**b) Video face swap (diffusion and other)**

| # | Model | Evidence | Family / date / license | Weights | Blackwell |
|---|---|---|---|---|---|
| 1 | DreamID-V | see above | Wan2.1-1.3B, Jan 2026, Apache | XuGuo699/DreamID-V, 12.9 GB plus Wan 1.3B T5/VAE | torch ≥2.4; falls back to SDPA without flash-attn. No known issues. |
| 2 | **CanonSwap** (ICCV 2025) | ID 0.397 (DreamID-V paper), but very stable over time | **Warping in a canonical space, no diffusion.** A new family for the list | Pixel-Talk/CanonSwap, weights on Google Drive, **ResearchRAIL-M (NC)**, 133 stars | Pinned to torch 2.3 cu118, which has no Blackwell kernels, so torch must be upgraded. |
| 3 | H3 FaceSwap LoRA | no third-party tests | MiniMax H3, Sep 2026 | UntMods/FaceSwap_MiniMaxH3_REF2VA, 66 MB. Tagged Apache, but the H3 license governs it | see H3 below |
| 4 | Stand-In face swap | "experimental" per its own authors | Wan2.1-14B plus 153M adapter, Apache | BowenXue/Stand-In | |

Not released: LivingSwap (CVPR 2026, "code coming soon"), StreamingSwap (Oct 2026, real-time, no weights), HiFiVFS, DynamicFace. VividFace has 90 stars and scored ID 0.175 in the StreamingSwap benchmark.

**c) Head swap**

| # | Model | Evidence | Weights |
|---|---|---|---|
| 1 | BFS LTX-2.3 v3 / LTX-2.5 v1.1 | DF26 hardness applies to LTX in general. No head-swap metrics | Alissonerdx/BFS-Best-Face-Swap-Video, 0.65–2.47 GB, LTX-2 community license |
| 2 | **BFS-author H3 head swap v1.0 r32** (Sep 2026, "experimental") | none | `h3/minimax_h3_head_swap_v1.0_r32` (310 MB), in both Alissonerdx repos |
| 3 | BFS Wan2.2 "bernini" head swap (Jun 2026) | none. Not documented in the model card | `wan22/headswap_bernini_*` high- and low-noise LoRAs, 0.6 GB each |
| 4 | GHOST 2.0 (per-frame GAN) | DirectSwap benchmark: face 0.571, FID 21.5, tLPIPS 0.0123 (VACE: 0.664 / 18.3 / 0.0070) | GitHub releases, Apache code. EMOCA/FLAME dependencies are NC. Stale since Feb 2025 |

DirectSwap (Wan-VACE-1.3B) wins every column of that benchmark (FID 9.3), but its weights are "coming soon".

**d) Whole-character replacement**

| # | Model | Evidence | Prevalence | Date / license | Weights / speed |
|---|---|---|---|---|---|
| 1 | Wan2.2-Animate-14B (replace) | MoCha benchmark: SSIM 0.692, LPIPS 0.213 | 204k HF dl, native ComfyUI | Sep 2025, Apache | 72 GB. 160 s for 124 frames at 480×832 on a B200 (Viggle's measurement) |
| 2 | **MiniMax H3 Ref2VA + swap LoRAs** | no benchmarks. Model cards admit identity drift | **Comfy-Org/MiniMax-H3 28.2M dl.** akatz H3 Character Swap V1.5: 121k dl. Alissonerdx/Minimax-H3-ComfyUI (character swap v1.1, head swap): 196k dl | Aug 2026. **H3 Community License: excludes the US, EU, UK and Korea. Firms over $20M revenue need permission. The acceptable-use policy (clause 13) bans impersonating a person without consent** | 33B omni audio-video DiT plus a Qwen3-VL-32B encoder. About 62 GiB resident at bf16 (Viggle's figure) |
| 3 | **SCAIL-2** (zai-org) | animation FVD 283–287 vs Wan-Animate 305. Its human study says it beats Wan-Animate and MoCha at replacement (figure only, no numbers) | **Comfy-Org/SCAIL-2 624k dl**, native ComfyUI, 1.3k stars | Jun 2026. MIT weights, Apache code. Relighting LoRA for replacement released Jul 2026 | 82.5 GB. Wan2.1-14B-I2V, so the same family |
| 4 | Viggle-Animate | vendor-only: 26 s vs 160 s for Wan-Animate on a B200, 3 forward passes | 32k dl | Sep 2026, H3 license | 69 GB. Needs ≥96 GB VRAM at bf16. **Needs a repainted first frame** (e.g. BFS Qwen-Image 2.1 head swap) |
| 5 | MoCha (CVPR 2026) | SSIM 0.746 / LPIPS 0.152, beating Wan-Animate in its own paper | 974 stars, kijai wrapper | Oct 2025. **Weights tagged AGPL-3.0** | 28.6 GB plus Wan2.1-T2V-14B |

Drop HunyuanCustom (behind Wan-Animate in MoCha's table; Tencent license), UniAnimate-DiT (FVD 480) and Phantom (subject-to-video, cannot replace). Wan-Animate-2 is Apache, 33 GB, and preferred over Wan-Animate in >70% of its user study, but it only does animation. It fits the reenactment scope ("B's frame moving like A"), not replacement.

**e) Real-time and live tools**

| Tool | Model it runs | Stars / status |
|---|---|---|
| Deep-Live-Cam | inswapper_128_fp16 + GFPGAN-1024 | 97.3k, active (Oct 27), AGPL |
| FaceFusion webcam mode | HyperSwap by default, any of its 14 swappers | 30.4k, active (3.9.2, Oct 26) |
| DeepFaceLive | DFM files (need per-identity training) or the Insight module (= inswapper) | 31k, **archived Nov 2024** |
| VisoMaster-Fusion (Rope lineage) | inswapper, InStyleSwapper, SimSwap512, GhostFace, CSCS, AlphaFace, DFM; TensorRT | 981 stars (VisoMaster 2.1k, Rope 5.4k), active |
| ComfyUI-ReActor | inswapper | 1.4k |

Commercial tools that dominate real-world use are out of scope. Haotian AI is a Telegram-sold real-time tool for WhatsApp, Zoom and Teams that made about $4M (404 Media / WIRED, May 2026). One secondary source says it is built on inswapper; the 404 Media article itself does not say, so this is unverified. Kling Motion Control and Viggle's app dominate character replacement. The WEF Jan 2026 report reviewed 17 swap tools without naming them; 5 were truly real-time.

**Blackwell issues across the scope**
- **ONNX swappers:**
  - The default PyPI onnxruntime-gpu wheels are CUDA 12.x builds, which run on a CUDA 13 driver.
  - FaceFusion 3.8.1 added a `cuda@13` installer, and 3.9.2 fixed a crash on a newer NVIDIA driver.
  - Deep-Live-Cam's README says to install cuDNN 8.9.7. cuDNN 8 has no Blackwell support, so use cuDNN 9.
  - I found no reports specific to sm_103 (unverified either way).
- **PyTorch and flash-attn:**
  - Wan-Animate-2 pins torch 2.7.0 cu126 and CanonSwap pins torch 2.3 cu118. Neither has Blackwell kernels, so use cu130 builds (the user's torch 2.11 cu130 ComfyUI image is fine).
  - flash-attn 2 sm_100 binaries should run on sm_103, since CUDA binaries run on the same major version.
  - flash-attn 4 had an sm_103 assertion bug (fixed upstream, flash-attention #2572).
  - Transformer Engine leaves sm103 off its flash-attn 2 head-dim allowlist (fix proposed in TE PR #3714).
- **H3's local server path uses SGLang**, which has open B300 bugs: an FA4 assertion (#25564) and a cutedsl GEMM Xid 13 error (#34340). The native ComfyUI path avoids SGLang. Viggle benchmarked on a B200 only.

### 3. What the list misses entirely
1. **The MiniMax H3 family.** It is the most-downloaded open video model right now and has a fast-growing set of swap LoRAs (character, head, face). It is a new 33B single-stream omni audio-video backbone, unlike Wan, LTX and Hunyuan, and it keeps or regenerates the source audio. The license and acceptable-use flags above apply.
2. **Warping-based video face swap** (CanonSwap). It is a family in its own right, with no diffusion and no per-frame GAN.
3. The fact that **public benchmarks contain no modern swappers**. FaceVid-Forensics-100K uses only the FF++ face swaps (Deepfakes, FaceSwap, FaceShifter). DreamID-V, BFS and the H3 swaps appear in no detection set, which strengthens the case for including them.

### 4. Recommended picks, in priority order
1. **facefusion-inswapper**, set up the way Deep-Live-Cam runs it (fp16 + GFPGAN): the most common swapper in the wild, and what live fraud tools are built on.
2. **facefusion HyperSwap 1A**: the FaceFusion default. Correct the license to ResearchRAIL.
3. **bfs-ltx23**: already running.
4. **wan-animate replace**, with the relighting LoRA: the reference open replacement model.
5. **h3-character-swap** (H3 Ref2VA + akatz V1.5, or the Alissonerdx head swap / character v1.1), **new**: a new backbone family and huge adoption. Needs a license decision first.
6. **dreamid-v** (DWPose variant): the only open video-diffusion face swapper.
7. **scail-2 replacement**, **new**: the second most-used open replacement model, MIT. Same Wan family, so lower breadth gain.
8. Optional and cheap: **alphaface_256** (one FaceFusion flag) and **canonswap** (the warping family; NC).

Watch list: LivingSwap, StreamingSwap, DirectSwap, Wan-Animate-2-Lite (real-time), Stand-In V2.

### 5. Sources
- FaceFusion licenses: https://docs.facefusion.io/introduction/licenses
- FaceFusion swapper models: https://docs.facefusion.io/usage/cli-arguments/processors/face-swapper
- FaceFusion releases: https://github.com/facefusion/facefusion/releases
- HF facefusion/models-3.3.0 and models-3.9.0
- CASIA face-swap benchmark: https://arxiv.org/html/2605.00883v1
- StreamingSwap: https://arxiv.org/html/2610.26681
- AlphaFace: https://arxiv.org/html/2601.16429 and https://github.com/andrewyu90/Alphaface_Official
- DreamID-V: https://github.com/bytedance/DreamID-V
- LivingSwap: https://arxiv.org/abs/2512.07951
- CanonSwap: https://luoxyhappy.github.io/CanonSwap/ and https://github.com/Pixel-Talk/CanonSwap
- Stand-In: https://github.com/WeChatCV/Stand-In
- DirectSwap: https://arxiv.org/html/2512.09417v3
- GHOST 2.0: https://github.com/ai-forever/ghost-2.0
- BFS video LoRAs: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video
- BFS H3 LoRAs: https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI
- H3 swap LoRA coverage: https://comfyui-wiki.com/news/2026-09-20-h3-faceswap-lora
- MiniMax H3 (incl. LICENSE): https://huggingface.co/MiniMaxAI/MiniMax-H3
- H3 Character Swap LoRA: https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA
- Viggle-Animate: https://huggingface.co/Viggle/Viggle-Animate
- Wan-Animate-2: https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B and https://arxiv.org/abs/2608.06009
- SCAIL-2: https://arxiv.org/html/2606.10804v1, https://huggingface.co/zai-org/SCAIL-2, https://github.com/zai-org/SCAIL-2
- MoCha: https://arxiv.org/html/2601.08587 and https://github.com/Orange-3DV-Team/MoCha
- Deep-Live-Cam: https://github.com/hacksider/Deep-Live-Cam
- VisoMaster-Fusion: https://github.com/VisoMasterFusion/VisoMaster-Fusion
- WEF report: https://reports.weforum.org/docs/WEF_Unmasking_Cybercrime_Strengthening_Digital_Identity_Verification_against_Deepfakes_2026.pdf
- Haotian AI (404 Media): https://www.404media.co/hello-boss-inside-the-chinese-realtime-deepfake-software-powering-scams-around-the-world/
- InsightFace inswapper variants: https://www.insightface.ai/blog/the-evolution-of-neural-network-face-swapping-from-deepfakes-to-one-shot-innovation-with-insightface
- FaceVid-Forensics-100K: https://arxiv.org/html/2608.06865
- B300 bug reports: https://github.com/sgl-project/sglang/issues/25564, https://github.com/sgl-project/sglang/issues/34340, https://github.com/NVIDIA/TransformerEngine/pull/3714
