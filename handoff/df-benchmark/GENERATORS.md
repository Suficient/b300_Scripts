# Generators

Why these 15. The set is chosen for a detector-training dataset: every manipulation type that appears in
current benchmarks and in the wild, spread over different architecture families and model generations
(artifacts differ more between families than between models of one family), at the best quality each
type reaches with open weights. Every generator must work one-shot from a single image or audio track of the
matched person (no per-identity training) and run on a DGX Spark.

"A" is the person in the clip, "B" the matched different-looking person (see README).

| Generator | Type | Makes | Architecture | Released | Weights license |
|---|---|---|---|---|---|
| facefusion (HyperSwap 1A 256) | face swap | A's clip with B's face | per-frame GAN swapper, ONNX | FaceFusion 3.9, Sep 2026 | OpenRAIL-AS (code); model terms per model |
| facefusion-inswapper (+GFPGAN) | face swap | same, 128 px swapper | per-frame, InsightFace inswapper | 2023 | non-commercial |
| dreamid-v | face swap | A's clip with B's face | Wan2.1-1.3B DiT | Jan 2026 | Apache-2.0 |
| bfs-ltx23 | head swap | A's clip with B's head | LTX-2.3 22B DiT + BFS v3 LoRA (ComfyUI) | 2026 | LTX-2 community license |
| wan-animate | character replacement | A's clip with B's whole appearance | Wan2.2-Animate-14B DiT | Sep 2025 | Apache-2.0 |
| liveportrait | reenactment | B's frame moving like A | implicit keypoints + warping, no diffusion | Jul 2024 | MIT code; InsightFace detector non-commercial |
| personalive | reenactment | B's face moving like A | SD-1.5 UNet (X-NeMo motion) | CVPR 2026 | Apache-2.0 |
| infinitetalk | talking head | B saying A's audio | Wan2.1-I2V-14B DiT + audio | Aug 2025 | Apache-2.0 |
| longcat-avatar | talking head | B saying A's audio | LongCat-Video 13.6B DiT, 8-step distilled | v1.5, May 2026 | MIT |
| hallo2 | talking head | B saying A's audio | SD-1.5 UNet + AnimateDiff | Oct 2024 | MIT |
| hunyuan-avatar | talking head | B saying A's audio | HunyuanVideo 13B DiT | May 2025 | Tencent Hunyuan community license (excludes EU, UK, South Korea) |
| latentsync | lip sync | A's clip mouthing B's audio | SD-1.5 latent inpainting, 512 px | v1.6, Jun 2025 | Apache-2.0 |
| musetalk | lip sync | A's clip mouthing B's audio | single-step latent inpainting (real time) | v1.5, Mar 2025 | MIT code, models any use; Whisper, VAE, DWPose keep their own |
| wav2lip | lip sync | A's clip mouthing B's audio | GAN on a 96 px mouth crop | 2020 | non-commercial |
| x-dub | lip sync | A's clip mouthing B's audio | mask-free editing on Wan2.2-TI2V-5B | Mar 2026 (ICML 2026) | Apache-2.0 |
| ltx-a2v | full synthesis | a new video of B saying A's audio | LTX-2.5 22B audio-video DiT | Aug 2026 | LTX-2 community license |

Lip-sync models change A's mouth and take no image, so they get B's audio (from the same clip B's reference
frame comes from): A appears to say B's words.

## Families covered

- Per-frame GAN / ONNX swappers: facefusion, facefusion-inswapper, wav2lip
- Warping without diffusion: liveportrait
- SD-1.5 UNet: personalive, hallo2, latentsync, musetalk
- Wan DiT: dreamid-v, wan-animate, infinitetalk, x-dub
- Other DiT backbones: LTX (bfs-ltx23, ltx-a2v), LongCat (longcat-avatar), HunyuanVideo (hunyuan-avatar)

## Evidence behind the picks

- InsightFace inswapper (+GFPGAN) is what Deep-Live-Cam (97k GitHub stars), Roop and ReActor run; FaceFusion is
  the most active successor and defaults to HyperSwap since 3.x. FaceFusion is the face-swap engine of the
  DeepSpeak v2 dataset.
- DreamID-V beats the other open video swappers it compares with (CanonSwap, Stand-In, image-level DreamID) on
  identity: ArcFace similarity 0.659 vs 0.397 (CanonSwap) and 0.616 (DreamID).
- Wan2.2-Animate is the open character-replacement model with native ComfyUI and Diffusers support, and the
  only one that replaces the whole person (hair, clothes, lighting) rather than the face.
- LivePortrait is the most used reenactment model (19k stars) and is in DeepSpeak v2 and TalkingHeadBench.
  PersonaLive adds a diffusion-based reenactor from a different family.
- InfiniteTalk has the best FID (24.8) and lip sync (Sync-C 7.17) of the open audio-driven baselines in the
  AptAvatar comparison (Jul 2026: InfiniteTalk, Wan2.2-S2V, LongCat-Avatar 1.5, LiveAvatar, SoulX-FlashTalk).
  LongCat-Avatar 1.5 is close behind SoulX-FlashTalk in that paper's preference study (18% vs 20%) and is the
  one non-Wan backbone among them.
- Hallo2 is among the two hardest generators to detect in TalkingHeadBench (WACV 2026).
- LatentSync 1.6 is rated the strongest local lip-sync model in 2026 comparisons; MuseTalk and Wav2Lip are
  the speed and sync-accuracy references and are in most lip-sync benchmarks. X-Dub is mask-free: it edits the
  frame instead of inpainting a masked mouth like the other three.
- DF26 (Sep 2026): LTX-2.3 image-to-video had the lowest human detection accuracy of all its generators
  (25.2%). LTX-2.5 is its successor and conditions on audio, so B can say A's words.

## Considered and left out

From the starting list:

- **StyleGAN2-ADA, StyleGAN3, EG3D**: unconditional image generators. They make stills of people who don't
  exist; they can't take B's image or follow A's clip, and produce no video. In 2026 video benchmarks (DF26,
  FaceVid-Forensics-100K) "entire face synthesis" means image/text-to-video models; ltx-a2v covers that.
- **SadTalker** (2023): 3DMM-driven, 256 px face pasted back; superseded by the talking-head models above and
  already in DF40.
- **EchoMimicV2**: half-body animation driven by hand-pose sequences; webcam head-and-shoulder clips have no
  hands to drive it with.

Also not used:

- **Hallo3** (CogVideoX): would add a fifth DiT backbone but is among the slowest talking-head models and
  is the same lineage as hallo2.
- **SoulX-FlashTalk, LiveAvatar, Wan2.2-S2V, EchoMimicV3**: Wan-based like infinitetalk. In the comparison
  above LiveAvatar and Wan2.2-S2V are behind it on FID and sync; SoulX-FlashTalk syncs slightly better
  (Sync-C 7.74 vs 7.17) with worse FID (28.0 vs 24.8). EchoMimicV3 isn't in that comparison.
- **FlashPortrait** (CVPR 2026, Wan2.1-14B reenactment): fourth Wan model; personalive gives a different family.
- **HunyuanVideo 1.5 image-to-video**: detectors were at chance on it in DF26, but it takes no audio, so B's lips
  wouldn't match any speech.
- **DeepFaceLab**: needs thousands of images and hours of training per identity pair.
- **Commercial models** (Veo, Kling, HeyGen, Seedance): don't run on the Spark.

## Sources

- DF40 (NeurIPS 2024): https://github.com/YZY-stack/DF40
- TalkingHeadBench (WACV 2026): https://arxiv.org/abs/2505.24866
- DeepSpeak v2: https://huggingface.co/datasets/faridlab/deepspeak_v2
- DF26: https://arxiv.org/abs/2609.07369
- FaceVid-Forensics-100K: https://arxiv.org/abs/2608.06865
- AptAvatar comparison table: https://arxiv.org/abs/2607.24013
- Lip-sync comparison (2026): https://instavar.com/research/ai-video/open-source-lip-sync-models
- DreamID-V: https://arxiv.org/abs/2601.01425
- FaceFusion swapper models: https://docs.facefusion.io/usage/cli-arguments/processors/face-swapper
- Deep-Live-Cam: https://github.com/hacksider/Deep-Live-Cam
- X-Dub: https://github.com/KlingAIResearch/X-Dub
- LTX-2.5: https://huggingface.co/Lightricks/LTX-2.5
