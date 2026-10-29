# LongCat-Video-Avatar 1.5: B's whole frame talking with A's audio; output audio = A's. run_demo_avatar_single_audio_
# to_video.py as upstream runs v1.5 (--stage_1 ai2v --use_distill: DMD LoRA, 8 steps, guidance 1/1; 480p default;
# vocals separated with Kim_Vocal_2 first), 2 segments (93 + 80 frames at 25 fps) cut to 5.2 s; models loaded once
# in a one-process torch.distributed group (its context-parallel mesh needs one).
import os, sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import math
import socket
import subprocess
import librosa
import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import load_image
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel
from audio_separator.separator import Separator
from pairio import run, ffmpeg
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor

SECONDS, FPS, N, COND, SEGMENTS, STEPS, RES = 5.2, 25, 93, 13, 2, 8, "480p"
CK, BASE = "weights/LongCat-Video-Avatar-1.5", "weights/LongCat-Video"
VOCALS = Path("/data/gen_tmp/longcat-avatar/vocals")
PROMPT = "A person is talking."
NEG = ("Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
       "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn "
       "hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
       "three legs, many people in the background, walking backwards")   # upstream's
SETTINGS = {"model_type": "avatar-v1.5", "stage_1": "ai2v", "distill": True, "steps": STEPS, "guidance": [1.0, 1.0],
            "resolution": RES, "segments": SEGMENTS, "ref_img_index": 10, "mask_frame_range": 3, "prompt": PROMPT,
            "seconds": SECONDS, "image": "b_frame.jpg", "audio": "A (a_guide.mp4), vocals separated", "crf": 17}


def load():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(s.getsockname()[1]), RANK="0", WORLD_SIZE="1")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl")
    context_parallel_util.init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)
    bf = torch.bfloat16
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(CK, subfolder="base_model", cp_split_hw=[1, 1],
                                                               torch_dtype=bf)
    dit.load_lora(f"{CK}/lora/dmd_lora.safetensors", "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
    dit.enable_loras(["dmd"])
    pipe = LongCatVideoAvatarPipeline(
        tokenizer=AutoTokenizer.from_pretrained(BASE, subfolder="tokenizer", torch_dtype=bf),
        text_encoder=UMT5EncoderModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=bf),
        vae=AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=bf),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(CK, subfolder="scheduler", torch_dtype=bf),
        dit=dit, audio_encoder=get_audio_encoder(f"{CK}/whisper-large-v3", "avatar-v1.5").to(0),
        audio_feature_extractor=get_audio_feature_extractor(f"{CK}/whisper-large-v3", "avatar-v1.5"),
        model_type="avatar-v1.5")
    pipe.to(0)
    sep = Separator(output_dir=str(VOCALS), output_single_stem="vocals",
                    model_file_dir=f"{CK}/vocal_separator")
    sep.load_model("Kim_Vocal_2.onnx")
    return pipe, sep


def make(model, p):
    pipe, sep = model
    wav, vocal, out = p.tmp / f"{p.name}.wav", p.tmp / "vocal.wav", p.tmp / "out.mp4"   # stems are named after it
    ffmpeg("-i", p.audio, "-t", str(SECONDS), wav)
    stems = sep.separate(str(wav))
    assert stems, "no vocals separated"
    (VOCALS / stems[0]).replace(vocal)   # the separator writes where it was loaded
    speech, sr = librosa.load(str(vocal), sr=16000)
    total = (N + (SEGMENTS - 1) * (N - COND)) / FPS
    speech = np.append(speech, np.zeros(max(0, math.ceil((total - len(speech) / sr) * sr)), speech.dtype))
    emb = pipe.get_audio_embedding(speech, fps=FPS, device=0, sample_rate=sr, model_type="avatar-v1.5")
    assert not torch.isnan(emb).any(), "audio embedding has NaN"
    idx = torch.arange(5) - 2
    window = lambda start: emb[torch.clamp(torch.arange(start, start + N).unsqueeze(1) + idx, 0, emb.shape[0] - 1)][None]
    g = torch.Generator(device=0).manual_seed(p.seed)
    kw = dict(prompt=PROMPT, negative_prompt=NEG, num_frames=N,
              num_inference_steps=STEPS, text_guidance_scale=1.0, audio_guidance_scale=1.0, generator=g,
              output_type="both", use_distill=True)
    out_np, latent = pipe.generate_ai2v(image=load_image(str(p.ref_frame)), resolution=RES, audio_emb=window(0), **kw)
    frames = (out_np[0] * 255).astype(np.uint8)
    ref_latent, cur, start = latent[:, :, :1].clone(), frames, 0
    for _ in range(1, SEGMENTS):
        start += N - COND
        h, w = cur.shape[1:3]
        out_np, latent = pipe.generate_avc(video=[Image.fromarray(f) for f in cur], video_latent=latent,
                                           height=h, width=w, num_cond_frames=COND, use_kv_cache=True,
                                           offload_kv_cache=False, enhance_hf=False, audio_emb=window(start),
                                           ref_latent=ref_latent, ref_img_index=10, mask_frame_range=3, **kw)
        cur = (out_np[0] * 255).astype(np.uint8)
        frames = np.concatenate([frames, cur[COND:]])
    v = frames[:round(SECONDS * FPS)]
    h, w = v.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
                    "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
