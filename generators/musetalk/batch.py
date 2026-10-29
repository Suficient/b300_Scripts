# MuseTalk 1.5 (fp32, the default): A's clip at 25 fps (MuseTalk's training rate, first 5.2 s) mouthing B's audio;
# output audio = B's. scripts/inference.py's v15 path (bbox_shift 0, extra_margin 10, jaw parsing, cheek width 90,
# batch 8) with the models loaded once and frames encoded straight to libx264 crf 17.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import copy, glob, subprocess
import cv2, numpy as np, torch
from transformers import WhisperModel
from pairio import run, ffmpeg
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder

SECONDS, FPS, MARGIN, BATCH = 5.2, 25, 10, 8
SETTINGS = {"version": "v15", "fp16": False, "fps": FPS, "seconds": SECONDS, "extra_margin": MARGIN,
            "parsing_mode": "jaw", "cheek_width": 90, "batch_size": BATCH, "audio": "B (b_source.mp4)", "crf": 17}


def load():
    dev = torch.device("cuda")
    vae, unet, pe = load_all_model(unet_model_path="models/musetalkV15/unet.pth", vae_type="sd-vae",
                                   unet_config="models/musetalkV15/musetalk.json", device=dev)
    pe, vae.vae, unet.model = pe.to(dev), vae.vae.to(dev), unet.model.to(dev)
    whisper = WhisperModel.from_pretrained("models/whisper").to(dev, dtype=unet.model.dtype).eval()
    whisper.requires_grad_(False)
    return vae, unet, pe, whisper, AudioProcessor(feature_extractor_path="models/whisper"), \
        FaceParsing(left_cheek_width=90, right_cheek_width=90)


@torch.no_grad()
def make(model, p):
    vae, unet, pe, whisper, ap, fp = model
    dev, dt = unet.model.device, unet.model.dtype
    fdir, wav, out = p.tmp / "frames", p.tmp / "b.wav", p.tmp / "out.mp4"
    fdir.mkdir()
    ffmpeg("-i", p.video, "-t", str(SECONDS), "-r", str(FPS), "-start_number", "0", f"{fdir}/%08d.png")
    ffmpeg("-i", p.ref_video, "-t", str(SECONDS), "-vn", "-ac", "1", "-ar", "16000", wav)
    feats, n_audio = ap.get_audio_feature(str(wav))
    chunks = ap.get_whisper_chunk(feats, dev, dt, whisper, n_audio, fps=FPS,
                                  audio_padding_length_left=2, audio_padding_length_right=2)
    coords, frames = get_landmark_and_bbox(sorted(glob.glob(f"{fdir}/*.png")), 0)
    latents = []
    for (x1, y1, x2, y2), f in zip(coords, frames):
        if (x1, y1, x2, y2) == coord_placeholder:
            continue
        crop = cv2.resize(f[y1:min(y2 + MARGIN, f.shape[0]), x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents.append(vae.get_latents_for_unet(crop))
    frames_c, coords_c, latents_c = frames + frames[::-1], coords + coords[::-1], latents + latents[::-1]
    t0 = torch.tensor([0], device=dev)
    res = []
    for wb, lb in datagen(whisper_chunks=chunks, vae_encode_latents=latents_c, batch_size=BATCH, delay_frame=0, device=dev):
        res += list(vae.decode_latents(unet.model(lb.to(dtype=dt), t0, encoder_hidden_states=pe(wb)).sample))
    h, w = frames[0].shape[:2]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                           stdin=subprocess.PIPE)
    for i, r in enumerate(res):
        x1, y1, x2, y2 = coords_c[i % len(coords_c)]
        f = copy.deepcopy(frames_c[i % len(frames_c)])
        y2 = min(y2 + MARGIN, h)
        try:
            r = cv2.resize(r.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:  # no face in this frame: keep it as is (inference.py drops it)
            enc.stdin.write(f.tobytes()); continue
        enc.stdin.write(get_image(f, r, [x1, y1, x2, y2], mode="jaw", fp=fp).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("ffmpeg encode failed")
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
