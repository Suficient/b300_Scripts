# Wan2.2-Animate-2-14B (full model, not the distilled one), animation: B's whole frame moved by A's clip; output
# audio = A's. infer/wan_animate_2_demo.py's defaults (40 steps, guide 3, 81-frame clips) at Wan's native 16 fps and
# 832x480 area (turned to B's frame's orientation; the reference is letterboxed into it and the result cropped back),
# so 5.06 s is one clip: the demo's 24 fps at 720x1280 took ~50 min/pair. Pipeline built once. setup.sh makes the
# pipeline's writer use crf 17. wan-animate-2-fast sets FAST: the authors' distilled model (10 steps, guide 1).
import os, sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path("infer").resolve()), "."]
import socket
from PIL import Image
from pairio import run, ffmpeg
from core import build_object_from_config_file

SECONDS, FPS, FAST = 81 / 16, 16, False
ARGS = dict(fps=FPS, clip_len=81, sample_guide_scale=3.0, step=40, prompt="", prompt_ref="人物动作的参考视频")
settings = lambda: {**args(), "model": "distillation" if FAST else "full", "area": "832x480", "seconds": SECONDS, "image": "b_frame.jpg", "audio": "A (a_guide.mp4)",
            "crf": 17}

args = lambda: {**ARGS, **(dict(sample_guide_scale=1.0, step=10) if FAST else {})}


def load():
    os.chdir("infer")   # the config's checkpoint paths are relative to infer/
    with socket.socket() as s:   # its default port 12345 clashes with another worker on the VM
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    return build_object_from_config_file("wan_animate_2_distillation.yaml" if FAST else "wan_animate_2.yaml")


def make(pipe, p):
    src = p.tmp / "a.mp4"
    ffmpeg("-i", p.video, "-t", str(SECONDS), "-c:v", "libx264", "-crf", "10", "-c:a", "aac", src)
    w, h = Image.open(p.ref_frame).size
    w, h = (832, 480) if w >= h else (480, 832)
    (p.tmp / "out").mkdir()   # the pipeline writes tpl_audio.wav there before creating it
    out = pipe(refer_img_path=str(p.ref_frame), tpl_video_path=str(src), output_path=str(p.tmp / "out"), width=w,
               height=h, seed=p.seed, **args())
    return Path(out), p.audio


if __name__ == "__main__":
    run(load, make, settings())
