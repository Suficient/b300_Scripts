# LTX-2.5 audio-to-video (ltx_pipelines.a2vid_two_stage): a new video of B (B's frame, resized and center-cropped,
# as the first frame) driven by A's audio; output audio = A's. The module's main() with its default steps and
# guidance (two stages: dev transformer, then the distilled LoRA + x2 latent upscaler), 1280x704 (704x1280 for
# portrait frames), 121 frames at 24 fps; the pipeline is built once (its class is swapped for a cache).
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import functools
import json
from PIL import Image
from pairio import run, ffmpeg, frames
import ltx_pipelines.a2vid_two_stage as a2v

FPS, M = 24, "models/ltx-2.5"
N = frames(5.2, FPS, 8, 1)
MODELS = ["--transformer-path", f"{M}/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
          "--distilled-lora", f"{M}/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
          "--text-encoder-path", f"{M}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
          "--video-vae-path", f"{M}/vae/ltx-2.5-video-vae-bf16.safetensors",
          "--audio-vae-path", f"{M}/vae/ltx-2.5-audio-vae-bf16.safetensors",
          "--spatial-upsampler-path", f"{M}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"]
SETTINGS = {"pipeline": "a2vid_two_stage (module defaults)", "size": "1280x704", "frames": N, "fps": FPS,
            "image": "b_frame.jpg (first frame)", "audio": "A (a_guide.mp4)", "prompt": "B's face + talking", "crf": 17}

_cls, _pipe = a2v.A2VidPipelineTwoStage, []
a2v.encode_video = functools.partial(a2v.encode_video, crf=17)   # its default is 19
a2v.A2VidPipelineTwoStage = lambda **kw: _pipe[0] if _pipe else _pipe.append(_cls(**kw)) or _pipe[0]


def prompt(p):
    face = json.loads((p.dir / "job.json").read_text())["face"].rstrip(".")
    return (f"{face}, talks to the camera in the same place, clothes and lighting as in the first frame, with natural "
            f"lip movements matching the speech and small natural head movements. Static camera, natural light.")


def load():
    return None


def make(_, p):
    img, wav, out = p.tmp / "first.png", p.tmp / "a.wav", p.tmp / "out.mp4"
    w, h = Image.open(p.ref_frame).size
    w, h = (1280, 704) if w >= h else (704, 1280)
    ffmpeg("-i", p.ref_frame, "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", img)
    ffmpeg("-i", p.video, "-vn", "-ac", "2", "-ar", "48000", "-t", f"{N / FPS + 0.2:.2f}", wav)
    sys.argv = ["a2vid_two_stage", *MODELS, "--image", str(img), "0", "1.0", "--audio-path", str(wav),
                "--frame-rate", str(FPS), "--num-frames", str(N), "--width", str(w), "--height", str(h),
                "--prompt", prompt(p), "--seed", str(p.seed), "--output-path", str(out)]
    a2v.main()
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
