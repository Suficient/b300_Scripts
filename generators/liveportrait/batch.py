# LivePortrait (human): B's whole frame (b_frame.jpg, full resolution) driven by A's clip (first 5.2 s, native fps),
# face cropped, animated (fp16, relative motion, stitching, expression-friendly: the defaults) and pasted back; A's
# audio. Pipeline built once; outputs written by its images2video at crf 17 (its concat preview is skipped).
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
from pairio import run, ffmpeg
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
import src.live_portrait_pipeline as lpp
from src.utils.video import images2video

SECONDS = 5.2
OPTS = dict(flag_crop_driving_video=True, flag_pasteback=True, source_max_dim=1920, flag_use_half_precision=True)
SETTINGS = {"repo": "KlingAIResearch/LivePortrait", "source": "b_frame.jpg", "seconds": SECONDS, **OPTS, "crf": 17}
FPS = [25.]
# the side-by-side preview is skipped; the result is encoded at crf 17 and A's exact fps (LivePortrait rounds it down)
lpp.images2video = lambda images, wfp, **kw: None if wfp.endswith("_concat.mp4") else \
    images2video(images, wfp, crf=17, **{**kw, "fps": FPS[0]})


def fields(cls, d):
    return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


def load():
    a = ArgumentConfig(**OPTS)
    return lpp.LivePortraitPipeline(inference_cfg=fields(InferenceConfig, a.__dict__),
                                    crop_cfg=fields(CropConfig, a.__dict__))


def make(pipe, p):
    drv, FPS[0] = p.tmp / "a.mp4", p.fps
    ffmpeg("-i", p.video, "-t", str(SECONDS), "-an", "-c:v", "copy", drv)
    wfp, _ = pipe.execute(ArgumentConfig(source=str(p.ref_frame), driving=str(drv), output_dir=str(p.tmp / "out"), **OPTS))
    return wfp, p.video


if __name__ == "__main__":
    run(load, make, SETTINGS)
