"""Write the ACTION half of each pair's BFS V3 prompt with a local vision model, one process per GPU.

Runs inside the bfs-ltx25 image on the VM (v3_full.sh starts it):
  docker run --rm --gpus all -v /data:/data -v ~/bfs_actions:/opt/actions squishydolphin/bfs-ltx25:b300 \
      python /opt/actions/actions.py

For each /data/inputs/pairs4/NNNN the model sees 4 frames of a_guide.mp4 and writes action.txt. Then
caption_v3.txt gets the V3 prompt: "head_swap:\nFACE:\n<FACE text from caption.txt>\n\nACTION:\n<action>".
Pairs that already have action.txt are skipped, so running again only fills in what is missing.
"""
import argparse
import multiprocessing as mp
import os
import queue
import re
import subprocess
import time
from pathlib import Path

PAIRS = Path(os.environ.get("DATA_DIR", "/data")) / "inputs" / "pairs4"
MODEL = "Qwen/Qwen3.5-27B"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
TIMES = (0.5, 2.0, 3.5, 5.0)  # seconds into the 6 s clip
FRAME_WIDTH = 512

# The ACTION part of the BFS V3 auto-prompt instruction
INSTRUCTION = (
    "These are 4 frames from one video clip, in order. Write one or two sentences describing the clip. "
    'Refer to the performer only as "a person". Describe clothing, body pose, movement, hand actions, '
    "objects being handled, camera framing and the setting. Never describe the performer's face, hair, "
    "head, makeup or expression.\n\n"
    "Use plain, factual language. No title, no preamble, no markdown, no extra commentary."
)


def frames(video):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    out = []
    for t in TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (FRAME_WIDTH, round(h * FRAME_WIDTH / w / 2) * 2), interpolation=cv2.INTER_AREA)
        out.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not out:
        raise RuntimeError(f"no frames read from {video}")
    return out


def clean(text):
    text = re.sub(r"(?s)<think>.*?</think>", "", text).replace("<think>", "").replace("</think>", "")
    text = re.sub(r"^\s*(\*\*)?ACTION:?(\*\*)?\s*", "", text.strip(), flags=re.I)
    text = " ".join(text.split())
    sentences, seen = [], set()
    for s in re.split(r"(?<=[.!?])\s+", text):  # drop repeated sentences, keep at most 3
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            sentences.append(s)
    text = " ".join(sentences[:3])
    return text[:1].upper() + text[1:]


def face_text(pair):
    cap = (pair / "caption.txt").read_text()
    m = re.search(r"FACE:\s*(.*)", cap, flags=re.S)
    return " ".join((m.group(1) if m else cap.replace("head_swap:", "")).split())


def write(pair, action):
    for name, text in (("action.txt", action + "\n"),
                       ("caption_v3.txt", f"head_swap:\nFACE:\n{face_text(pair)}\n\nACTION:\n{action}\n")):
        tmp = pair / (name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, pair / name)


def worker(gpu, device, model_dir, pairs, batch, max_tokens, done_q):
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_dir)
    proc.tokenizer.padding_side = "left"
    # no device_map: that needs `accelerate`, which the image does not have
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16).to(device).eval()
    for i in range(0, len(pairs), batch):
        group = [Path(p) for p in pairs[i:i + batch]]
        try:
            texts, images = [], []
            for p in group:
                imgs = frames(p / "a_guide.mp4")
                content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": INSTRUCTION}]
                texts.append(proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False))
                images.append(imgs)
            inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
            answers = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for p, a in zip(group, answers):
                a = clean(a)
                if a:
                    write(p, a)
                    done_q.put(("ok", p.name, a))
                else:
                    done_q.put(("empty", p.name, ""))
        except Exception as e:  # report and go on; a rerun retries these pairs
            for p in group:
                done_q.put(("error", p.name, f"{type(e).__name__}: {str(e)[:300]}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--gpus", type=int, help="processes (default: every visible GPU)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--pairs", help="comma-separated pair ids (default: every pair without action.txt)")
    a = ap.parse_args()

    if a.pairs:
        todo = [PAIRS / p for p in a.pairs.split(",")]
    else:
        todo = [p for p in sorted(PAIRS.iterdir()) if (p / "a_guide.mp4").exists() and not (p / "action.txt").exists()]
    print(f"[actions] {len(todo)} pairs to describe", flush=True)
    if not todo:
        return

    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    model_dir = snapshot_download(a.model, revision=a.revision,
                                  local_dir=Path(os.environ.get("DATA_DIR", "/data")) / "hf" / a.model.split("/")[-1],
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    print(f"[actions] model ready: {a.model} @ {a.revision[:12]}", flush=True)

    n = a.gpus or (int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
                   if a.device == "cuda" else 1)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    shards = [[str(p) for p in todo[i::n]] for i in range(n)]
    procs = [ctx.Process(target=worker, args=(i, a.device, model_dir, shards[i], a.batch, a.max_tokens, q))
             for i in range(n) if shards[i]]
    for p in procs:
        p.start()
    t0, done, bad, last = time.time(), 0, 0, 0
    while any(p.is_alive() for p in procs) or not q.empty():
        try:
            status, pair, text = q.get(timeout=5)
        except queue.Empty:
            continue
        if status == "ok":
            done += 1
            if done <= 3:
                print(f"[actions] example {pair}: {text}", flush=True)
        else:
            bad += 1
            print(f"[actions] {pair} {status}: {text}", flush=True)
        if time.time() - last > 30:
            rate = done / max(time.time() - t0, 1)
            print(f"[actions] {done}/{len(todo)} done, {bad} failed, {rate * 60:.0f}/min", flush=True)
            last = time.time()
    for p in procs:
        p.join()
    crashed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    missing = len(todo) - done - bad
    if crashed:
        print(f"[actions] worker(s) {crashed} crashed; {missing} pairs not attempted (see the traceback above)", flush=True)
    print(f"[actions] finished: {done} written, {bad + missing} not written (run again to retry)", flush=True)
    raise SystemExit(1 if crashed or done == 0 else 0)


if __name__ == "__main__":
    main()
