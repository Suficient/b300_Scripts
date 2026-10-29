"""A's words for the models that speak text: Whisper large-v3 over each pair's a_guide.mp4 audio, written as
a_transcript.txt (plain text) and a_language.txt (ISO code) in the pair folder. Skips pairs that have both.

    transcribe.py LIST [--model DIR] [--batch 16]    (LIST: pair names or folder paths, one per line)
"""
import argparse
import time
from pathlib import Path

import av
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PAIRS = Path("/data/inputs/pairs4")


def audio16k(path):
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
        out = [f.to_ndarray().ravel() for fr in c.decode(audio=0) for f in rs.resample(fr)]
        out += [f.to_ndarray().ravel() for f in rs.resample(None)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("list", type=Path)
    ap.add_argument("--model", default="/data/gen/whisper/models/whisper-large-v3")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    dirs = [Path(l) if "/" in l else PAIRS / l for l in (l.strip() for l in a.list.read_text().splitlines()) if l]
    todo = [d for d in dirs if not ((d / "a_transcript.txt").exists() and (d / "a_language.txt").exists())]
    print(f"{len(todo)} of {len(dirs)} pairs to transcribe", flush=True)
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    lang_ids = {i: t[2:-2] for t, i in model.generation_config.lang_to_id.items()}
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        feats = proc([audio16k(d / "a_guide.mp4") for d in chunk], sampling_rate=16000, return_tensors="pt")
        x = feats.input_features.cuda().half()
        with torch.inference_mode():
            langs = [lang_ids[int(t)] for t in model.detect_language(x).view(-1)]
            # per-clip language, beam search, no timestamps (clips are ~6 s, one 30 s window)
            ids = model.generate(x, language=langs, task="transcribe", num_beams=5)
        texts = [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        for d, l, t in zip(chunk, langs, texts):
            (d / "a_transcript.txt").write_text(t + "\n")
            (d / "a_language.txt").write_text(l + "\n")
        print(f"{i + len(chunk)}/{len(todo)} {time.time() - t0:.0f} s  {chunk[-1].name} [{langs[-1]}] {texts[-1][:80]}",
              flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
