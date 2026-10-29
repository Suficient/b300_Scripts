# HunyuanVideo-Avatar: B's whole frame talking with A's audio; output audio = A's. hymm_sp/sample_gpu_poor.py (the
# single-GPU path) with upstream's sampling settings (704 px, 129 frames = 5.16 s, cfg 7.5, shift 5) but 30 steps
# instead of 50 (for time), on the authors' fp8 transformer, without CPU offload or DeepCache; models loaded once.
# hunyuan-avatar-fast sets FAST: DeepCache on, as upstream's single-GPU script runs it.
import os, sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
os.environ.update(MODEL_BASE="./weights", DISABLE_SP="1")
import csv
import subprocess
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, WhisperModel
from pairio import run, ffmpeg
from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler
from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
from hymm_sp.data_kits.face_align import AlignImage
from hymm_sp.modules.fp8_optimization import fp8_activation_dequant

SECONDS, CK, FAST = 5.2, "weights/ckpts", False
PROMPT = "A person is talking."
ARGS = ["--input", "-", "--ckpt", f"{CK}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt", "--use-fp8",
        "--sample-n-frames", "129", "--image-size", "704", "--cfg-scale", "7.5", "--infer-steps", "30",
        "--flow-shift-eval-video", "5.0", "--save-path", "/data/gen_tmp/hunyuan-avatar"]
cli = lambda: ARGS + ["--use-deepcache", "1" if FAST else "0"]
settings = lambda: {"args": " ".join(cli()[4:]), "dtype": "fp8", "prompt": PROMPT, "seconds": SECONDS,
            "image": "b_frame.jpg", "audio": "A (a_guide.mp4)", "crf": 17}


def load():
    sys.argv = ["sample_gpu_poor.py"] + cli()
    args = parse_args()
    sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=torch.device("cuda"))
    # upstream's fp8 linears dequantize their weights to bf16 on every call; the same bf16 weights, made once here
    for m in sampler.model.modules():
        if hasattr(m, "fp8_scale"):
            m.weight = torch.nn.Parameter(fp8_activation_dequant(m.weight, m.fp8_scale.cuda(), m.fp8_scale.dtype),
                                          requires_grad=False)
            m.forward = m.original_forward
    whisper = WhisperModel.from_pretrained(f"{CK}/whisper-tiny/").to("cuda", torch.float32).requires_grad_(False)
    fe = AutoFeatureExtractor.from_pretrained(f"{CK}/whisper-tiny/")
    return sampler, whisper, fe, AlignImage("cuda", det_path=f"{CK}/det_align/detface.pt")


def make(model, p):
    sampler, whisper, fe, align = model
    args = sampler.args
    wav, meta, out = p.tmp / "a.wav", p.tmp / "in.csv", p.tmp / "out.mp4"
    ffmpeg("-i", p.audio, "-t", str(SECONDS), wav)
    with open(meta, "w", newline="") as f:
        csv.writer(f).writerows([["videoid", "image", "audio", "prompt", "fps"], ["out", p.ref_frame, wav, PROMPT, 25]])
    data = VideoAudioTextLoaderVal(image_size=args.image_size, meta_file=str(meta), text_encoder=sampler.text_encoder,
                                   text_encoder_2=sampler.text_encoder_2, feature_extractor=fe)
    batch = next(iter(DataLoader(data, batch_size=1)))
    args.seed = p.seed
    sample = sampler.predict(args, batch, whisper, fe, align)["samples"][0][:, :batch["audio_len"][0]]
    v = (rearrange(sample, "c f h w -> f h w c") * 255.).cpu().numpy().astype(np.uint8)   # upstream's conversion
    h, w = v.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
                    "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, settings())
