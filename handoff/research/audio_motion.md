## Audit: audio-driven and motion-driven faces (as of 2026-10-28)

### 1. Verdicts on the current picks
| Pick | Verdict | Evidence |
|---|---|---|
| liveportrait | **Keep** | 19.3k stars (repo is now KlingAIResearch/LivePortrait, HF KlingTeam/LivePortrait, 2.1 GB). It is in DeepSpeak v2 and TalkingHeadBench. In PersonaLive's table it has the best ID-SIM of the non-FYE methods (0.723) and FVD 557. It is the only non-diffusion warping model. The InsightFace detector is non-commercial. |
| personalive | **Keep, fix license flag** | Real-time streaming reenactment: 15.8 FPS on an H100, ~12 GB. Its own paper gives the best FVD (520.6 vs 557.2 for LivePortrait) with lower ID-SIM (0.698). 4.0k stars. **License conflict:** the HF card says Apache-2.0, but the GitHub README says "released for academic research only", so treat it as non-commercial. On Blackwell, run with `--acceleration none` because xformers is broken there. The H100 TensorRT engine must be rebuilt. |
| infinitetalk | **Keep, fix the claim** | Most used open model: 8.2k stars, 68k HF downloads/month, runs in ComfyUI via kijai. GENERATORS.md says it has the "best lip sync" in AptAvatar's table. That is wrong: SoulX-FlashTalk leads on Sync-C (7.74 vs 7.17) and Sync-D. InfiniteTalk also wasn't in that paper's user study. In the independent Kandinsky S2V table it is mid-pack (mean rank 6.3–7.7 of 11). |
| longcat-avatar | **Keep** | Independent Kandinsky S2V table (Oct 2026): best open baseline on Sync-C in all three test sets (EMTD 8.41, CelebV-HQ 7.55, HDTF 8.48) and best mean rank (2.6–3.0). This conflicts with AptAvatar's short-clip Sync-C of 4.14 (worst of six there). Each figure comes from a single paper. MIT. It is the only non-Wan backbone. |
| hallo2 | **Keep (low priority)** | TalkingHeadBench: second-hardest under generator shift (mean AUC 0.883; HiFi-Net 0.48). EMOPortraits is the hardest at 0.867. Output is a 512² face crop only. 2024, unmaintained since Feb 2025. |
| hunyuan-avatar | **Keep, note length** | Kandinsky table: highest CSIM (0.80 CelebV-HQ, 0.87 HDTF) with competitive sync. It is the only HunyuanVideo backbone. run.sh asks for 129 frames, about 5.2 s at 25 fps, so output is shorter than the 6 s clips. The license excludes the EU, UK and South Korea. |
| latentsync | **Keep, fix license** | Independent LipForcing table (HDTF): best baseline on FID (6.90) and CSIM (0.967), Sync-C 8.10, FVD 117.9. It was also the strongest baseline in X-Dub's paper. 6.2k stars, 325k HF downloads/month, in DeepSpeak v2. **The HF weights are tagged openrail++ (CreativeML OpenRAIL++-M), not Apache-2.0.** Only the code is Apache. |
| musetalk | **Keep** | The real-time class: 23 FPS in LipForcing's table, FID 9.68, Sync-C 7.94. LiveTalking (10.2k stars) streams it live to WebRTC, RTMP and virtual cameras. 256 px face. mmcv must be built from source. |
| wav2lip | **Keep** | GAN reference, 13.3k stars, and still in LiveTalking (the `wav2lip256` variant). Its Sync-C of 8.56 is above ground truth because it overfits SyncNet. FID 24.15. Non-commercial. |
| x-dub | **Keep** | Mask-free. Strongest baseline in LipForcing's user study (MOS sync 4.40), but weak automatic metrics there (Sync-C 7.58, FID 14.76, 0.91 FPS). The public Wan2.2-5B release is not the paper's version, and its own comparison is still a TODO. 30 GB, ~21 GB VRAM. |

### 2. Ranked candidates per subcategory
**Talking head (B's image + A's audio)**
| # | Model | Backbone | Evidence | License | Weights | Notes |
|---|---|---|---|---|---|---|
| 1 | LongCat-Video-Avatar 1.5 (May 2026) | LongCat 13.6B, 8-step DMD | as above; own study beats Kling Avatar 2.0 at 65.9% | MIT | meituan-longcat/LongCat-Video-Avatar-1.5 75 GB + LongCat-Video 83 GB | 480/720p, 93-frame chunks plus continuation |
| 2 | InfiniteTalk (Aug 2025) | Wan2.1-I2V-14B | AptAvatar: FID 24.8 (best baseline) | Apache-2.0 | MeiGen-AI/InfiniteTalk (168 GB, all variants) | 480/720p, any length, regenerates the full frame |
| 3 | SoulX-FlashTalk (Jan 2026) | Wan 14B, 4-NFE self-correcting distillation | AptAvatar: Sync-C 7.74 (best baseline), FID 28.0, 20% preference | Apache-2.0 | Soul-AILab/SoulX-FlashTalk-14B 54.5 GB, 27k downloads/month | 32 FPS, 0.87 s latency on 8×H800; one GPU needs >64 GB. **Real-time streaming class** |
| 4 | HunyuanVideo-Avatar (May 2025) | HunyuanVideo 13B | Kandinsky: best CSIM | Tencent community (geo-restricted) | tencent/HunyuanVideo-Avatar 81 GB | 704 px, 129 frames |
| 5 | AptAvatar (Jul 2026) | InfiniteTalk fine-tune, 2 NFE | Own table: beats every baseline (FID 24.76, Sync-C 7.87, 62% preference). Single source | Apache-2.0 (LICENSE file in the HF repo) | TaoLiveAIGC/AptAvatar 54.5 GB, 20k downloads/month | 720p. Same family as InfiniteTalk at about 60× fewer NFEs. Use it as a cheaper substitute if InfiniteTalk's compute is too high, not as an extra family |
| watch | Kandinsky S2V 1.0 (arXiv 2610.19290) | Kandinsky-5 DiT 2B/21B, Hunyuan VAE | Own table: distilled Pro ranks best overall (mean rank 2.3–2.7), ahead of LongCat 1.5 | MIT (as the paper states) | **Not found:** only "tiny" test pipes on HF and no GitHub repo | It would be a new backbone family once the weights are public |

Lower priority, all Wan-based: Wan2.2-S2V (35k downloads/month, native ComfyUI, weak sync), LiveAvatar (2.5k stars, Wan2.2-S2V plus a 4-step LoRA, FP8 fits 48 GB), SoulX-FlashHead 1.3B (512², 96 FPS Lite on an RTX 4090; Pro variant Apache, Lite has an OpenRAIL-M VAE), OmniAvatar, StableAvatar (Kandinsky Sync-C ≈3.1, poor), HuMo, EchoMimicV3, FantasyTalking, AvatarForcing. Others: SkyReels-V3-A2V (Skywork license), Sonic (CC BY-NC-SA), FLOAT (CC BY-NC-ND).

**Lip sync (A's clip + B's audio)**
| # | Model | Family | Evidence (LipForcing HDTF table unless noted) | License / weights |
|---|---|---|---|---|
| 1 | LatentSync 1.6 | SD-1.5 latent inpainting, 512 | FID 6.90, CSIM 0.967 | Apache code / OpenRAIL++ weights, 9.6 GB |
| 2 | InfiniteTalk V2V dubbing | Wan2.1-14B, regenerates the full head and body | Own paper: human rank 1.11 vs 2.32 for LatentSync. FID/CSIM are worse (26.1/0.775) because it changes the whole frame. Background can drift | Already installed |
| 3 | X-Dub | Wan2.2-5B mask-free editing | above | Apache, 30 GB |
| 4 | MuseTalk 1.5 | single-step inpainting, 256 | above | MIT |
| 5 | LTX-2.3 Dub-It IC-LoRA (formerly LipDub, May 2026) | LTX-2.3 22B joint audio-video | No numbers found. Takes A's video plus **B's transcript as text** and also clones A's voice, so it doesn't fit the B-audio input contract. LTX-2.5 port is "in development" | Gated, LTX-2 community license; Lightricks/LTX-2.3-22b-IC-LoRA-DubIt |

Also considered: Wav2Lip (keep). LipForcing (Jul 2026, Wan2.1-14B causal, FVD 107.9 best, 15 FPS; Apache code, weights license not stated; 134 stars). KeySync (SVD backbone, Apache code, weights at toninio19/keysync, 404 stars). Diff2Lip (in DeepSpeak v2). VideoReTalking (2022). OmniSync has no released weights.

**Reenactment (B's image + A's motion)**
| # | Model | Family | Evidence (PersonaLive table) | Notes |
|---|---|---|---|---|
| 1 | LivePortrait | keypoint warping | ID 0.723, FVD 557 | Pastes back onto B's still frame; any length |
| 2 | PersonaLive | SD-1.5, 4 steps, streaming | ID 0.698, FVD 521, 15.8 FPS | 512² face output; license conflict |
| 3 | Wan2.2-Animate **animation mode** | Wan 14B DiT | No independent numbers | Already installed (run.sh uses `--replace_flag` only). Full frame, 720p |
| 4 | EMOPortraits | volumetric warping (MegaPortraits line) | Hardest generator in TalkingHeadBench (mean AUC 0.863–0.953) | Apache code. Weights on Google Drive (retrained on 17 FEED IDs). Ships as a pre-packed conda env, so Blackwell support is unverified and likely needs a rebuild |
| 5 | HunyuanPortrait | SVD | ID 0.644, FVD 620 | Tencent license |

FlashPortrait and FantasyPortrait (both Wan2.1-14B) have no independent numbers, and FlashPortrait's 14B checkpoint status is unclear. EditaLive (PersonaLive's successor, LoRAs on Wan2.2-Animate, 14.5 FPS) is academic-only. X-NeMo, X-Portrait 2 and Knot Forcing have no public weights.

**Real-time tools used live**
| Tool | Model it runs | Notes |
|---|---|---|
| LiveTalking (10.2k stars) | wav2lip256, MuseTalk, ER-NeRF, Ultralight | WebRTC, RTMP or virtual camera |
| FasterLivePortrait (1.2k) | LivePortrait + JoyVASA | 30+ FPS on a 3090, webcam. **TensorRT 8.x only, which has no Blackwell support; use the ONNX path** |
| PersonaLive (4k) | its own model | webcam WebUI |
| SoulX-FlashHead / FlashTalk, LiveAvatar | Wan-distilled | streaming talking heads |
| Duix-Avatar (16k) | its own model | per-identity cloning, not one-shot |
| Commercial | Pickle (per-identity avatar lip-synced live in Zoom/Meet), HeyGen, Tavus | out of scope |

I found no primary fraud report that names a talking-head or lip-sync model. Documented live-call cases, such as MITRE ATLAS's KYC case (face swap fed in through OBS and a virtual camera), involve face swap, which is outside my scope. For one-shot avatar video, commercial tools dominate: Kling Avatar 2.0, OmniHuman-1.5, HeyGen, Hedra.

### 3. Missing from the list entirely
- **Few-step causal/streaming talking heads**, the realistic live-call attack class: SoulX-FlashTalk, FlashHead, LiveAvatar. Their distillation artifacts differ from 40–50-step models.
- **Full-head V2V dubbing**, as opposed to mouth inpainting: InfiniteTalk V2V and LTX Dub-It.
- **A DiT reenactor**: Wan-Animate's animation mode.
- **Joint audio-video "talking" synthesis with generated voice**: daVinci-MagiHuman (15B, Apache; beat LTX 2.3 in 60.9% of human comparisons) takes no audio input. It builds the Hopper-only FlashAttention kernels, so it is likely blocked on B300 (unverified). MOVA, Ovi, Hallo-Live and DreamID-Omni are similar. ltx-a2v partly covers this.
- **HeyGen TAVR** (SIGGRAPH Asia 2026, Apache, Wan2.1-14B): talking avatar from a reference video, which fits the b_source.mp4 input. It **requires FlashAttention 3 (Hopper-only), which blocks it on B300**.

**Blackwell notes:** for the Wan, LongCat and Hunyuan DiTs, FA2 built for sm_100 runs on sm_103. FA3 is Hopper-only, which hits TAVR and MagiHuman; LiveAvatar falls back to FA2. Upstream pins need overriding: InfiniteTalk torch 2.4.1, HunyuanVideo-Avatar torch 2.4 / FA 2.6.3, MuseTalk torch 2.0.1 cu118, FLOAT cu118. MuseTalk and X-Dub need mmcv built from source; the repo already does this for the Spark.

### 4. Recommended picks, in priority order
1. **latentsync**: best automatic lip-sync metrics, most downloaded, in benchmarks.
2. **infinitetalk** (I2V): the most-used open talking head.
3. **liveportrait**: most-used reenactor, non-diffusion family.
4. **longcat-avatar**: best open sync in an independent table, non-Wan backbone.
5. **infinitetalk V2V** (new): full-head dubbing at zero install cost.
6. **soulx-flashtalk** (new): real-time streaming class with the best baseline sync. FlashHead is the consumer-GPU alternative.
7. **personalive**: real-time reenactor, SD-1.5 family. License flag.
8. **hunyuan-avatar**: unique backbone, best CSIM. Geo-restricted license, 5.2 s output.
9. **x-dub**: mask-free editing, strongest baseline in a user study.
10. **musetalk** and **wav2lip**: what live tools actually run.
11. **wan-animate animation mode** (new): DiT reenactment at no install cost.
12. **hallo2**: hard to detect, but old SD-1.5 face-crop output.
- Watch list: Kandinsky S2V (weights), LTX Dub-It on 2.5, EMOPortraits (if it can be rebuilt on a modern torch).

### 5. Sources
- AptAvatar: https://arxiv.org/html/2607.24013 , https://huggingface.co/TaoLiveAIGC/AptAvatar
- Kandinsky S2V: https://arxiv.org/html/2610.19290v1
- TalkingHeadBench: https://arxiv.org/html/2505.24866v3
- LipForcing: https://arxiv.org/html/2606.11180 , https://github.com/cvlab-kaist/LipForcing
- X-Dub: https://arxiv.org/html/2512.25066 , https://github.com/KlingAIResearch/X-Dub
- InfiniteTalk: https://arxiv.org/html/2508.14033
- LongCat 1.5 report: https://arxiv.org/html/2605.26486v1
- PersonaLive: https://arxiv.org/html/2512.11253 , https://github.com/GVCLab/PersonaLive
- DeepSpeak v2: https://huggingface.co/datasets/faridlab/deepspeak_v2
- SoulX-FlashTalk: https://github.com/Soul-AILab/SoulX-FlashTalk
- SoulX-FlashHead: https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B
- LiveAvatar: https://github.com/Alibaba-Quark/LiveAvatar
- LTX Dub-It: https://docs.ltx.io/open-source-model/feature-guides/audio/dub-it-beta , https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt
- TAVR: https://github.com/heygen-com/TAVR
- daVinci-MagiHuman: https://github.com/GAIR-NLP/daVinci-MagiHuman
- EMOPortraits: https://github.com/neeek2303/EMOPortraits
- LiveTalking: https://github.com/lipku/LiveTalking
- FasterLivePortrait: https://github.com/warmshao/FasterLivePortrait
- LatentSync: https://huggingface.co/ByteDance/LatentSync-1.6
- Pickle: https://www.tomsguide.com/ai/i-used-an-ai-body-double-for-video-calls-it-even-fooled-my-spouse
- Star counts, download counts, sizes and licenses come from the GitHub and HF APIs, queried 2026-10-28.
