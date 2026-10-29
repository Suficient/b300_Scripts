# PersonaLive offline (streaming pipeline, 4 DDIM steps, fp16, SDPA): B's 512 px face crop (b_face_512.jpg) driven
# by A's clip at 25 fps (first 5.2 s, frames a multiple of 4); 512x512 output; A's audio. inference_offline.py with
# the models loaded once, seed = pair number, and only the generated frames encoded (crf 17).
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import subprocess
import numpy as np, torch, mediapipe as mp
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from decord import VideoReader
from pairio import run, ffmpeg
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline_Stream
from src.utils.util import crop_face
from src.models.motion_encoder.encoder import MotEncoder
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider

SECONDS, FPS, SIZE = 5.2, 25, 512
SETTINGS = {"config": "configs/prompts/personalive_offline.yaml", "steps": 4, "guidance": 1.0, "fp16": True,
            "stream_gen": True, "temporal_window_size": 4, "size": SIZE, "fps": FPS, "seconds": SECONDS,
            "ref": "b_face_512.jpg", "xformers": False, "crf": 17}


def load():
    cfg = OmegaConf.load(SETTINGS["config"]); icfg = OmegaConf.load(cfg.inference_config)
    dt, dev, w = torch.float16, "cuda", cfg.denoising_unet_path
    sd = lambda name: torch.load(w.replace("denoising_unet", name), map_location="cpu")
    vae = AutoencoderKL.from_pretrained(cfg.vae_path).to(dev, dtype=dt)
    ref_unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dtype=dt)
    den_unet = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
                                                       unet_additional_kwargs=icfg.unet_additional_kwargs).to(dev, dtype=dt)
    mot_enc = MotEncoder().to(dev, dtype=dt).eval()
    pose_guider = PoseGuider().to(dev, dtype=dt)
    pose_enc = MotionExtractor(num_kp=21).to(dev, dtype=dt).eval()
    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dtype=dt)
    den_unet.load_state_dict(sd("denoising_unet"), strict=False)
    ref_unet.load_state_dict(sd("reference_unet"), strict=True)
    mot_enc.load_state_dict(sd("motion_encoder"), strict=True)
    pose_guider.load_state_dict(sd("pose_guider"), strict=True)
    den_unet.load_state_dict(sd("temporal_module"), strict=False)
    pose_enc.load_state_dict(sd("motion_extractor"), strict=False)
    pipe = Pose2VideoPipeline_Stream(vae=vae, image_encoder=image_enc, reference_unet=ref_unet, denoising_unet=den_unet,
                                     motion_encoder=mot_enc, pose_encoder=pose_enc, pose_guider=pose_guider,
                                     scheduler=DDIMScheduler(**OmegaConf.to_container(icfg.noise_scheduler_kwargs)))
    return pipe.to(dev), mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def crops(images, mesh):
    """crop_face per frame; a frame where FaceMesh finds no face (hands, blur) reuses the nearest found crop."""
    out = []
    for im in images:
        try:
            out.append(Image.fromarray(crop_face(im, mesh)).convert("RGB"))
        except TypeError:
            out.append(None)
    found = [i for i, c in enumerate(out) if c is not None]
    if len(found) < 0.8 * len(out):
        raise ValueError(f"no face in {len(out) - len(found)} of {len(out)} frames")
    return [c or out[min(found, key=lambda j: abs(j - i))] for i, c in enumerate(out)]


def make(model, p):
    pipe, mesh = model
    drv, out = p.tmp / "a.mp4", p.tmp / "out.mp4"
    ffmpeg("-i", p.video, "-t", str(SECONDS), "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "10", drv)
    control = VideoReader(str(drv))
    n = len(control) // 4 * 4
    poses = [Image.fromarray(f).convert("RGB") for f in control.get_batch(list(range(n))).asnumpy()]
    ref = Image.open(p.ref).convert("RGB")
    faces = crops(poses, mesh)
    ref_face = Image.fromarray(crop_face(ref, mesh)).convert("RGB")
    g = torch.Generator(device="cuda").manual_seed(p.seed)
    video = pipe(poses, ref, faces, ref_face, SIZE, SIZE, n, num_inference_steps=4, guidance_scale=1.0, generator=g,
                 temporal_window_size=4, temporal_adaptive_step=4).videos  # (1, c, f, h, w) in [0, 1]
    frames = (video[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
                    "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)
    return out, p.video


if __name__ == "__main__":
    run(load, make, SETTINGS)
