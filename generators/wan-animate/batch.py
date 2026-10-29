# Wan2.2-Animate-14B, replacement mode + relighting LoRA: the person in A's clip replaced by B (B's whole frame as the
# character reference); output audio = A's. preprocess_data.py (--replace_flag, 1280x720 area, 30 fps, mask
# iterations 3, k 7, w/h_len 1) then generate.py --task animate-14B with its config (20 steps, shift 5, guide 1,
# 77-frame clips, refert_num 1); both pipelines loaded once. A is cut to 5.2 s first.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), ".", "wan/modules/animate/preprocess"]
import subprocess
import torch
import wan
from wan.configs import WAN_CONFIGS
from pairio import run, ffmpeg
from process_pipepline import ProcessPipeline

SECONDS, CK = 5.2, "Wan2.2-Animate-14B"
cfg = WAN_CONFIGS["animate-14B"]
PRE = dict(resolution_area=[1280, 720], fps=30, iterations=3, k=7, w_len=1, h_len=1, retarget_flag=False,
           use_flux=False, replace_flag=True)
SETTINGS = {"mode": "replace", "relighting_lora": True, "preprocess": PRE, "steps": cfg.sample_steps,
            "shift": cfg.sample_shift, "guide": cfg.sample_guide_scale, "clip_len": cfg.frame_num, "refert_num": 1,
            "seconds": SECONDS, "image": "b_frame.jpg", "audio": "A (a_guide.mp4)", "crf": 17}


def load():
    pre = ProcessPipeline(det_checkpoint_path=f"{CK}/process_checkpoint/det/yolov10m.onnx",
                          pose2d_checkpoint_path=f"{CK}/process_checkpoint/pose2d/vitpose_h_wholebody.onnx",
                          sam_checkpoint_path=f"{CK}/process_checkpoint/sam2/sam2_hiera_large.pt",
                          flux_kontext_path=None)
    model = wan.WanAnimate(config=cfg, checkpoint_dir=CK, device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                           use_sp=False, t5_cpu=False, convert_model_dtype=False, use_relighting_lora=True)
    return pre, model


def make(m, p):
    pre, model = m
    src, proc, out = p.tmp / "a.mp4", p.tmp / "process", p.tmp / "out.mp4"
    ffmpeg("-i", p.video, "-t", str(SECONDS), "-an", "-c:v", "libx264", "-crf", "10", src)
    proc.mkdir(exist_ok=True)
    pre(video_path=str(src), refer_image_path=str(p.ref_frame), output_path=str(proc), **PRE)
    v = model.generate(src_root_path=str(proc), replace_flag=True, refert_num=1, clip_len=cfg.frame_num,
                       shift=cfg.sample_shift, sample_solver="unipc", sampling_steps=cfg.sample_steps,
                       guide_scale=cfg.sample_guide_scale, seed=p.seed, offload_model=False)
    v = ((v + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu().numpy()  # C T H W
    h, w = v.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", str(cfg.sample_fps), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                    str(out)], input=v.tobytes(), check=True)
    return out, p.audio


if __name__ == "__main__":
    run(load, make, SETTINGS)
