# LatentSync 1.6: A's clip (25 fps) re-lipped to B's audio; output audio = B's. scripts/inference.py with upstream
# inference.sh settings (stage2_512, 20 steps, guidance 1.5) minus DeepCache (no step caching); pipeline loaded once.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from pairio import run, ffmpeg
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature

SECONDS, STEPS, GUIDANCE = 5.2, 20, 1.5
SETTINGS = {"unet": "stage2_512", "steps": STEPS, "guidance": GUIDANCE, "deepcache": False, "dtype": "fp16",
            "seconds": SECONDS, "fps": 25, "audio": "B (b_source.mp4)", "crf": 13}


def load():
    cfg = OmegaConf.load("configs/unet/stage2_512.yaml")
    audio = Audio2Feature(model_path="checkpoints/whisper/tiny.pt", device="cuda", num_frames=cfg.data.num_frames,
                          audio_feat_length=cfg.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained("checkpoints/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor, vae.config.shift_factor = 0.18215, 0
    unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model), "checkpoints/latentsync_unet.pt",
                                                   device="cpu")
    pipe = LipsyncPipeline(vae=vae, audio_encoder=audio, unet=unet.to(torch.float16),
                           scheduler=DDIMScheduler.from_pretrained("configs")).to("cuda")
    return pipe, cfg


def make(model, p):
    pipe, cfg = model
    video, wav, tmp = p.tmp / "a.mp4", p.tmp / "b.wav", p.tmp / "ls"
    ffmpeg("-i", p.video_25, "-t", str(SECONDS), "-an", "-c:v", "libx264", "-crf", "10", video)
    ffmpeg("-i", p.ref_audio, "-t", str(SECONDS), wav)
    set_seed(p.seed)
    pipe(video_path=str(video), audio_path=str(wav), video_out_path=str(p.tmp / "muxed.mp4"),
         num_frames=cfg.data.num_frames, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
         weight_dtype=torch.float16, width=cfg.data.resolution, height=cfg.data.resolution,
         mask_image_path=cfg.data.mask_image_path, temp_dir=str(tmp))
    return tmp / "video.mp4", wav   # imageio libx264 crf 13 (the muxed copy is crf 18)


if __name__ == "__main__":
    run(load, make, SETTINGS)
