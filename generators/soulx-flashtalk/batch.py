# SoulX-FlashTalk-14B: B's frame (cropped to 416:720 around B's face, then resized to upstream's 720x416) talking
# with A's audio; output audio = A's. generate_video.py's stream mode (the real-time path: 33-frame chunks, 5 motion
# frames, 4 steps) with upstream's prompt; pipeline loaded once (torch.compile on, as upstream).
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import json
import subprocess
from collections import deque
import cv2
import librosa
import numpy as np
import torch
from pairio import run, ffmpeg
from flash_talk.inference import get_pipeline, get_base_data, get_audio_embedding, run_pipeline, infer_params

SECONDS = 5.2
PROMPT = "A person is talking. Only the foreground characters are moving, the background remains static."
SETTINGS = {**infer_params, "mode": "stream", "prompt": PROMPT, "seconds": SECONDS, "image": "b_frame.jpg, face-centred 416:720 crop",
            "audio": "A (a_guide.mp4)", "crf": 17}


def load():
    return get_pipeline(world_size=1, ckpt_dir="models/SoulX-FlashTalk-14B", wav2vec_dir="models/chinese-wav2vec2-base")


def portrait(p, out):
    """B's frame cut to the model's 416:720 around B's face (b_face_512.jpg found in it by template matching), so
    upstream's center crop doesn't cut the face off. A frame that is already narrow enough is used as it is."""
    f = cv2.imread(str(p.ref_frame))
    H, W = f.shape[:2]
    w = round(H * infer_params["width"] / infer_params["height"])
    if w >= W:
        return p.ref_frame
    s, k = json.loads((p.dir / "job.json").read_text())["b_crop_native_px"], 0.25
    t, g = cv2.resize(cv2.imread(str(p.ref)), (round(s * k),) * 2), cv2.resize(f, None, fx=k, fy=k)
    cx = (cv2.minMaxLoc(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED))[3][0] / k + s / 2
          if t.shape[0] < min(g.shape[:2]) else W / 2)
    x = int(min(max(cx - w / 2, 0), W - w))
    cv2.imwrite(str(out), f[:, x:x + w], [cv2.IMWRITE_JPEG_QUALITY, 98])
    return out


def make(pipe, p):
    sr, fps, n, motion = (infer_params[k] for k in ("sample_rate", "tgt_fps", "frame_num", "motion_frames_num"))
    wav, out = p.tmp / "a.wav", p.tmp / "out.mp4"
    ffmpeg("-i", p.audio, "-t", str(SECONDS), wav)
    get_base_data(pipe, input_prompt=PROMPT, cond_image=str(portrait(p, p.tmp / "b.jpg")), base_seed=p.seed)
    speech, _ = librosa.load(str(wav), sr=sr, mono=True)
    step = (n - motion) * sr // fps
    speech = np.concatenate([speech, np.zeros(-len(speech) % step, speech.dtype)]).reshape(-1, step)
    cached = sr * infer_params["cached_audio_duration"]
    end = infer_params["cached_audio_duration"] * fps
    dq, frames = deque([0.0] * cached, maxlen=cached), []
    for chunk in speech:
        dq.extend(chunk.tolist())
        emb = get_audio_embedding(pipe, np.array(dq), end - n, end)
        frames.append(run_pipeline(pipe, emb)[motion:].to(torch.uint8).cpu())
    v = torch.cat(frames)[:round(SECONDS * fps)].numpy()   # whole chunks overrun the audio
    h, w = v.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                    "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
