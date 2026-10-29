# CanonSwap: B's face (b_face_1024.png) swapped onto A's clip (first 5.2 s, native fps) by inference_canswap.py's
# settings (fp32, no stitching, paste-back); A's audio. Pipeline built once. Its visible watermark (watermark.png
# at 20% opacity, bottom right) is turned off: it would be a trivial detector cue. Encoded at crf 17, A's exact fps.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import numpy as np, torch
from pairio import run, ffmpeg
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
import src.can_swap_pipeline_e2e as cse
from src.utils.video import images2video

SECONDS = 5.2
SETTINGS = {"repo": "Pixel-Talk/CanonSwap", "source": "b_face_1024.png", "seconds": SECONDS, "fp16": False,
            "flag_stitching": False, "flag_pasteback": True, "watermark": False, "crf": 17}
FPS = [25.]
cse.add_watermark_to_frame_list = lambda frames, *a, **k: frames
cse.images2video = lambda images, wfp, **kw: None if wfp.endswith("_concat.mp4") else \
    images2video(images, wfp, crf=17, **{**kw, "fps": FPS[0]})


def fields(cls, d):
    return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


def load():
    a = ArgumentConfig()
    inf = fields(InferenceConfig, a.__dict__)
    inf.flag_stitching, inf.flag_pasteback, inf.flag_use_half_precision = False, True, False
    return cse.CanSwapPipeline(inference_cfg=inf, crop_cfg=fields(CropConfig, a.__dict__))


def make(pipe, p):
    np.random.seed(p.seed); torch.manual_seed(p.seed); torch.cuda.manual_seed_all(p.seed)
    drv, FPS[0] = p.tmp / "a.mp4", p.fps
    ffmpeg("-i", p.video, "-t", str(SECONDS), "-an", "-c:v", "copy", drv)
    (p.tmp / "out").mkdir()
    pipe.execute(ArgumentConfig(source=str(p.ref_1024), driving=str(drv), output_dir=str(p.tmp / "out")))
    return next((p.tmp / "out").glob("*--a.mp4")), p.video


if __name__ == "__main__":
    run(load, make, SETTINGS)
