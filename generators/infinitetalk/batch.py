# InfiniteTalk image-to-video: B's whole frame talking with A's audio; output audio = A's. generate_infinitetalk.py
# with upstream's reference settings (480 bucket of the 480P weights, 40 steps, shift 7, text/audio cfg 5/4,
# streaming with 9 motion frames, no TeaCache/APG/LoRA, no offload); pipeline and wav2vec loaded once.
# infinitetalk-v2v imports this with V2V = True: A's clip re-generated speaking B's audio. The -fast variants set
# FAST = True: the README's 4-step path (lightx2v cfg/step-distill LoRA, text/audio cfg 1/2, shift 2).
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import subprocess
import torch
import soundfile as sf
import wan
from pairio import run, ffmpeg
from wan.configs import WAN_CONFIGS
import generate_infinitetalk as gi
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor

SECONDS, V2V, FAST = 5.2, False, False
LORA = "weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"
PROMPT = "A person is talking. Only the foreground characters are moving, the background remains static."
ARGS = ["--ckpt_dir", "weights/Wan2.1-I2V-14B-480P", "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--infinitetalk_dir", "weights/InfiniteTalk/single/infinitetalk.safetensors", "--input_json", "-",
        "--size", "infinitetalk-480", "--mode", "streaming", "--motion_frame", "9", "--offload_model", "False"]


def cli():
    return ARGS + (["--lora_dir", LORA, "--lora_scale", "1.0", "--sample_steps", "4", "--sample_text_guide_scale", "1.0",
                    "--sample_audio_guide_scale", "2.0", "--sample_shift", "2"] if FAST else ["--sample_steps", "40"])


def settings():
    return {"args": " ".join(cli()[6:]), "prompt": PROMPT, "seconds": SECONDS,
            "cond": "A's clip (25 fps)" if V2V else "b_frame.jpg", "audio": "B (b_source.mp4)" if V2V else "A (a_guide.mp4)",
            "crf": 17}


def load():
    sys.argv = ["generate_infinitetalk.py"] + cli()
    args = gi._parse_args()
    pipe = wan.InfiniteTalkPipeline(config=WAN_CONFIGS[args.task], checkpoint_dir=args.ckpt_dir, quant_dir=None,
                                    device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
                                    lora_dir=args.lora_dir, lora_scales=args.lora_scale, quant=None, dit_path=None,
                                    infinitetalk_dir=args.infinitetalk_dir)
    # gi.custom_init, with the eager attention older transformers defaulted to (it asks for attention outputs)
    enc = Wav2Vec2Model.from_pretrained(args.wav2vec_dir, local_files_only=True, attn_implementation="eager")
    enc.feature_extractor._freeze_parameters()
    return pipe, args, (Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir, local_files_only=True), enc)


def make(model, p):
    pipe, args, (fe, enc) = model
    wav, out = p.tmp / "audio.wav", p.tmp / "out.mp4"
    ffmpeg("-i", p.ref_audio if V2V else p.audio, "-t", str(SECONDS), wav)
    cond = p.tmp / "cond.mp4" if V2V else p.ref_frame
    if V2V:
        ffmpeg("-i", p.video_25, "-t", str(SECONDS), "-an", "-c:v", "libx264", "-crf", "10", cond)
    speech = gi.audio_prepare_single(str(wav))
    torch.save(gi.get_embedding(speech, fe, enc), p.tmp / "1.pt")
    sf.write(p.tmp / "sum.wav", speech, 16000)
    clip = {"prompt": PROMPT, "cond_video": str(cond), "cond_audio": {"person1": str(p.tmp / "1.pt")},
            "video_audio": str(p.tmp / "sum.wav")}
    args.base_seed = p.seed
    v = pipe.generate_infinitetalk(clip, size_buckget=args.size, motion_frame=args.motion_frame,
                                   frame_num=args.frame_num, shift=args.sample_shift, sampling_steps=args.sample_steps,
                                   text_guide_scale=args.sample_text_guide_scale,
                                   audio_guide_scale=args.sample_audio_guide_scale, seed=p.seed,
                                   offload_model=False, max_frames_num=args.max_frame_num,
                                   color_correction_strength=args.color_correction_strength, extra_args=args)
    v = ((v + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).cpu().numpy()  # C T H W
    h, w = v.shape[1:3]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
                    "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, settings())
