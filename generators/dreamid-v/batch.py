# DreamID-V (DWPose variant, dreamidv.pth on Wan2.1-T2V-1.3B): B's face swapped into A's clip; output audio = A's.
# generate_dreamidv_dwpose.py with the README's settings (20 steps, 832*480 area, aspect kept; 1280*720 cost
# ~700 s/pair) and the script's defaults (img cfg 4, shift 5, UniPC, 24 fps out). The model takes the clip's
# first frame_num frames as they are, so A is resampled to 24 fps and cut to 121 frames (5.04 s, 4k+1).
# Model loaded once. dreamid-v-faster sets FAST: the authors' Faster model (dreamidv_faster.pth, no pose input,
# the README's 16 steps).
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), ".", "pose"]   # pose/extract.py imports dwpose
import importlib
import subprocess
import torch
from dreamidv_wan.configs import SIZE_CONFIGS
from pairio import run, ffmpeg, frames
from pose.extract import process_dwpose

FPS, SIZE, FAST = 24, "832*480", False
N = frames(5.2, FPS, 4, 1)
settings = lambda: {"ckpt": "dreamidv_faster.pth" if FAST else "dreamidv.pth", "pose": None if FAST else "DWPose", "size": SIZE, "steps": steps(), "guide_scale_img": 4.0, "shift": 5.0,
            "solver": "unipc", "fps": FPS, "frames": N, "image": "b_face_512.jpg", "audio": "A (a_guide.mp4)", "crf": 17}

steps = lambda: 16 if FAST else 20


def load():
    pkg = importlib.import_module("dreamidv_wan_faster" if FAST else "dreamidv_wan")
    cfg = importlib.import_module(pkg.__name__ + ".configs").WAN_CONFIGS["swapface"]
    cfg.sample_fps = FPS
    return pkg.DreamIDV(config=cfg, checkpoint_dir="weights/Wan2.1-T2V-1.3B",
                        dreamidv_ckpt="weights/dreamidv_faster.pth" if FAST else "weights/dreamidv.pth", device_id=0, rank=0, t5_fsdp=False,
                                 dit_fsdp=False, use_usp=False, t5_cpu=False)


def make(model, p):
    src, pose, mask, out = (p.tmp / f for f in ("a.mp4", "a_pose.mp4", "a_mask.mp4", "out.mp4"))
    ffmpeg("-i", p.video, "-r", str(FPS), "-frames:v", str(N), "-an", "-c:v", "libx264", "-crf", "10", src)
    process_dwpose(str(src), str(pose), str(mask))
    refs = [str(src), str(mask), str(p.ref)] + ([] if FAST else [str(pose)])
    v = model.generate("chang face", refs, size=SIZE_CONFIGS[SIZE],
                       frame_num=N, shift=5.0, sample_solver="unipc", sampling_steps=steps(), guide_scale_img=4.0,
                       seed=p.seed, offload_model=False)   # "chang face": upstream's fixed prompt
    v = ((v + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu().numpy()  # C T H W
    h, w = v.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
                    "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.tobytes(), check=True)
    return out, p.audio


if __name__ == "__main__":
    run(load, make, settings())
