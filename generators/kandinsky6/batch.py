# Kandinsky 6.0 Video Pro (distilled, 5 s): B's whole frame as the first frame, B speaking A's words (Whisper
# transcript, a_transcript.txt) in a voice the model makes; output audio = the model's own 44 kHz track.
# Kandinsky6TI2VAPipeline with the checkpoint's settings (10 steps, guidance 1, 121 frames at 24 fps = 5.04 s,
# 480x864 turned to B's frame's orientation; the frame is resized and center-cropped), all on the GPU, no prompt
# expansion, no super-resolution pass.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import subprocess
import numpy as np
import soundfile as sf
import torch
from diffusers import Kandinsky6TI2VAPipeline
from PIL import Image
from pairio import run

FPS, N, STEPS = 24, 121, 10
SETTINGS = {"steps": STEPS, "guidance": 1.0, "frames": N, "fps": FPS, "size": "480x864", "image": "b_frame.jpg",
            "speech": "A's words (a_transcript.txt), voice generated", "audio": "generated", "crf": 17}


def prompt(p):
    words = (p.dir / "a_transcript.txt").read_text().strip()
    return (f'The person in the image talks to the camera and says: "{words}" Their lips move in sync with the '
            f"speech; small natural head movements, same place and lighting. Audio: only this person's clear voice "
            f"saying exactly these words, quiet room tone. No music, no other voices, no text or logos.")


def load():
    return Kandinsky6TI2VAPipeline.from_pretrained("weights", torch_dtype=torch.bfloat16).to("cuda")


def make(pipe, p):
    wav, out = p.tmp / "audio.wav", p.tmp / "out.mp4"
    img = Image.open(p.ref_frame).convert("RGB")
    h, w = (480, 864) if img.width >= img.height else (864, 480)
    r = pipe(prompt=prompt(p), image=img, height=h, width=w, num_frames=N, frame_rate=FPS, num_inference_steps=STEPS,
             guidance_scale=1.0, generator=torch.Generator("cuda").manual_seed(p.seed), output_type="np")
    v = (np.asarray(r.frames[0]).clip(0, 1) * 255).round().astype(np.uint8)   # f h w c in [0, 1]
    a = r.audio[0]
    a = a.float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a, np.float32)
    sf.write(wav, a.T if a.ndim == 2 else a, pipe.audio_sample_rate)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{v.shape[2]}x{v.shape[1]}",
                    "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
