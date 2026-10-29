# Hallo2: B's face (512x512 crop) talking with A's audio; output audio = A's. scripts/inference_long.py with
# configs/inference/long.yaml (40 steps, cfg 3.5, fp16, 16-frame clips + 2 motion frames, masked motion frames),
# models loaded once. No 4K super-resolution pass (video_sr.py), as upstream's default.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import subprocess
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from pairio import run, ffmpeg
from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from scripts.inference_long import Net, process_audio_emb

SECONDS = 5.2
cfg = OmegaConf.load("configs/inference/long.yaml")
SETTINGS = {"config": "long.yaml", "steps": cfg.inference_steps, "cfg": cfg.cfg_scale, "dtype": cfg.weight_dtype,
            "size": 512, "fps": cfg.data.export_video.fps, "seconds": SECONDS, "image": "b_face_512.jpg",
            "audio": "A (a_guide.mp4)", "crf": 17}


def load():
    dt = torch.float16
    imgs = ImageProcessor((512, 512), cfg.face_analysis.model_path)
    sep = cfg.audio_separator.model_path
    auds = AudioProcessor(16000, cfg.data.export_video.fps, cfg.wav2vec.model_path, cfg.wav2vec.features == "last",
                          str(Path(sep).parent), Path(sep).name, "/data/gen_tmp/hallo2_audio")
    kw = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    kw.update(rescale_betas_zero_snr=True, timestep_spacing="trailing", prediction_type="v_prediction")
    ref = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet")
    den = UNet3DConditionModel.from_pretrained_2d(cfg.base_model_path, cfg.motion_module_path, subfolder="unet",
                                                  unet_additional_kwargs=OmegaConf.to_container(cfg.unet_additional_kwargs),
                                                  use_landmark=False)
    net = Net(ref, den, FaceLocator(conditioning_embedding_channels=320),
              ImageProjModel(cross_attention_dim=den.config.cross_attention_dim, clip_embeddings_dim=512,
                             clip_extra_context_tokens=4),
              AudioProjModel(seq_len=5, blocks=12, channels=768, intermediate_dim=512, output_dim=768,
                             context_tokens=32))
    m, u = net.load_state_dict(torch.load(f"{cfg.audio_ckpt_dir}/net.pth", map_location="cpu"))
    assert not m and not u, "wrong checkpoint"
    net.audioproj.to("cuda", dt)
    pipe = FaceAnimatePipeline(vae=AutoencoderKL.from_pretrained(cfg.vae.model_path), reference_unet=net.reference_unet,
                               denoising_unet=net.denoising_unet, face_locator=net.face_locator,
                               scheduler=DDIMScheduler(**kw), image_proj=net.imageproj).to("cuda", dt)
    return imgs, auds, net, pipe


def make(model, p):
    imgs, auds, net, pipe = model
    clip, nm = cfg.data.n_sample_frames, cfg.data.n_motion_frames
    wav, out = p.tmp / "a.wav", p.tmp / "out.mp4"
    ffmpeg("-i", p.audio, "-t", str(SECONDS), wav)
    px, region, emb, full, face, lip = imgs.preprocess(str(p.ref), str(p.tmp), cfg.face_expand_ratio)
    audio_emb, n = auds.preprocess(str(wav), clip)
    audio_emb = process_audio_emb(audio_emb)
    px, region = px.unsqueeze(0), region.unsqueeze(0)
    emb = torch.tensor(emb.reshape(1, -1))
    full, face, lip = ([m.repeat(clip, 1) for m in ms] for ms in (full, face, lip))
    gen, res = torch.manual_seed(p.seed), []
    for t in range(audio_emb.shape[0] // clip):
        motion = px.repeat(nm, 1, 1, 1) if not res else res[-1][0].permute(1, 0, 2, 3)[-nm:] * 2.0 - 1.0
        ref = torch.cat([px, motion.to(px.dtype)], 0).unsqueeze(0)
        mv = ref[:, 1:]
        if cfg.use_mask:   # upstream: random pixel dropout on the motion frames outside the face
            b, f, c, h, w = mv.shape
            mask = (torch.rand(h, w) > cfg.mask_rate)[None, None, None].expand(b, f, c, h, w)
            ref[:, 1:] = mv * (mask | region.repeat(f, 1, 1, 1).unsqueeze(0).bool())
        a = net.audioproj(audio_emb[t * clip:(t + 1) * clip].unsqueeze(0).to("cuda", torch.float16))
        res.append(pipe(ref_image=ref, audio_tensor=a, face_emb=emb, face_mask=region, pixel_values_full_mask=full,
                        pixel_values_face_mask=face, pixel_values_lip_mask=lip, width=512, height=512,
                        video_length=clip, num_inference_steps=cfg.inference_steps, guidance_scale=cfg.cfg_scale,
                        generator=gen, motion_scale=[cfg.pose_weight, cfg.face_weight, cfg.lip_weight]).videos)
    v = torch.cat(res, 2)[0, :, :n]   # c f h w in [0, 1]
    raw = (v.permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "512x512",
                    "-r", str(cfg.data.export_video.fps), "-i", "-", "-c:v", "libx264", "-crf", "17",
                    "-pix_fmt", "yuv420p", str(out)], input=raw.tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
