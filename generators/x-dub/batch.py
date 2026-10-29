# X-Dub: A's clip (25 fps) re-lipped to B's audio, mask-free; output audio = B's. infer_lip_sync_pipeline.py with the
# README's settings (30 steps, ref cfg 2.5, audio cfg 10), every stage on the GPU (upstream's vram_config parks
# weights on the CPU), pipeline loaded once. The script only saves side-by-side videos, so the pasted-back result
# is taken from its paste_compare call.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import argparse
import subprocess
import numpy as np
import torch
from pairio import run, ffmpeg
import infer_lip_sync_pipeline as xd

SECONDS = 5.2
ARGS = dict(ref_cfg_scale=2.5, audio_cfg_scale=10.0, audio_feat_window_size=0, num_inference_steps=30)
SETTINGS = {**ARGS, "dtype": "bf16", "offload": False, "fps": 25, "seconds": SECONDS, "audio": "B (b_source.mp4)",
            "crf": 17}
GPU = {k: (torch.bfloat16 if k.endswith("dtype") else "cuda") for k in xd.vram_config}
result = {}
xd.concat_pil_videos_horizontally = lambda a, b: b
xd.save_video_with_audio = lambda frames, audio, name, out_dir: result.__setitem__(name.rsplit("_", 2)[-2], frames)


def load():
    args = argparse.Namespace(**ARGS, seed=0, ckpt_path=xd.DEFAULT_DIT_PATH, output_dir="/data/gen_tmp/x-dub",
                              sample_name=None)
    c = xd.ModelConfig
    pipe = xd.LipSyncPipeline().from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[c(path=args.ckpt_path, **GPU), c(path=xd.DEFAULT_TEXT_ENCODER_PATH, **GPU),
                       c(path=xd.DEFAULT_VAE_PATH, **GPU)],
        tokenizer_config=c(path=xd.DEFAULT_TOKENIZER_PATH), args=args, whisper_ckpt_path=xd.DEFAULT_WHISPER_PATH,
        wav2vec_ckpt_path=xd.DEFAULT_WAV2VEC_PATH)
    pipe.whisper_processor.to(dtype=torch.float32)
    pipe.wav2vec_processor.to(dtype=torch.float32)
    return pipe, args


def make(model, p):
    pipe, args = model
    video, wav, out = p.tmp / "a.mp4", p.tmp / "b.wav", p.tmp / "out.mp4"
    ffmpeg("-i", p.video_25, "-t", str(SECONDS), "-an", "-c:v", "libx264", "-crf", "10", video)
    ffmpeg("-i", p.ref_audio, "-t", str(SECONDS), wav)
    args.seed, args.output_dir, args.sample_name = p.seed, str(p.tmp), "s"
    args.video_path, args.audio_path = str(video), str(wav)
    result.clear()
    xd.infer_one_sample(pipe, xd.preprocess_inputs(str(video), str(wav), args), "s", args)
    v = np.stack([np.asarray(f.convert("RGB")) for f in result["paste"]])
    h, w = v.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
                    "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
