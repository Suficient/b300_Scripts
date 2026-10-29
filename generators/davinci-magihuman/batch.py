# daVinci-MagiHuman: B's whole frame as the reference image, B speaking A's words (Whisper transcript) in a voice the
# model makes; output audio = the model's own track. example/sr_540p (base at 448x256 then the 540p SR model to
# 896x512, turned to B's frame's orientation; 5 s; its config's steps and cfg) without CPU offload, models loaded once
# in a one-process distributed group. The prompt follows example/assets/prompt.txt: a description with the quoted
# speech.
import os, sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import json
import socket
import subprocess
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from pairio import run

SECONDS, CK = 5, str(Path("checkpoints").resolve())
SETTINGS = {"config": "example/sr_540p/config.json", "seconds": SECONDS, "base": "448x256", "sr": "896x512",
            "image": "b_frame.jpg", "speech": "A's words (a_transcript.txt), voice generated", "audio": "generated",
            "crf": 17}


def prompt(p):
    face = json.loads((p.dir / "job.json").read_text())["face"].rstrip(".")
    words = (p.dir / "a_transcript.txt").read_text().strip()
    return (f"{face}, looks into the camera and talks, the same person, clothes, place and lighting as in the "
            f"reference image. The camera holds a static medium shot with natural light and colour. The person "
            f'speaks in a clear, natural, conversational tone, "{words}" with small natural head and hand movements.')


def load():
    cfg = json.loads(Path("example/sr_540p/config.json").read_text().replace("/path/to/checkpoints", CK))
    path = Path("/data/gen_tmp/davinci-magihuman/config.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(s.getsockname()[1]), RANK="0", LOCAL_RANK="0",
                          WORLD_SIZE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    sys.argv = ["entry.py", "--config-load-path", str(path)]
    from inference.common import parse_config
    from inference.infra import initialize_infra
    from inference.model.dit import get_dit
    from inference.pipeline import MagiPipeline
    initialize_infra()
    config = parse_config()
    return MagiPipeline(get_dit(config.arch_config, config.engine_config), config.evaluation_config)


def make(pipe, p):
    wav, out = p.tmp / "audio.wav", p.tmp / "out.mp4"
    img = Image.open(p.ref_frame).convert("RGB")
    (bw, bh), (sw, sh) = ((448, 256), (896, 512)) if img.width >= img.height else ((256, 448), (512, 896))
    ec = pipe.evaluation_config
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.random.manual_seed(p.seed)
        v, a = pipe.evaluator.evaluate(prompt(p), img, None, seconds=SECONDS, br_width=bw, br_height=bh, sr_width=sw,
                                       sr_height=sh, br_num_inference_steps=ec.num_inference_steps,
                                       sr_num_inference_steps=ec.sr_num_inference_steps)
    sf.write(wav, a, pipe.evaluator.audio_vae.sample_rate)
    v = np.asarray(v)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{v.shape[2]}x{v.shape[1]}",
                    "-r", str(ec.fps), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                   input=v.astype(np.uint8).tobytes(), check=True)
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
