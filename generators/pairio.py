"""Batch mode: one generator over many pair folders, model loaded once. Standard library only, so every generator's
venv can import it. Each generators/<name>/batch.py does:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pairio import run

    SETTINGS = {...}                  # what the outputs were made with; copied into each NNNN.json

    def load():                       # once per worker process
        return model

    def make(model, p):               # p is a Pair (below)
        ...write a video into p.tmp...
        return video_path, audio_path # audio_path: any file whose first audio stream becomes the fake's sound

    if __name__ == "__main__":
        run(load, make, SETTINGS)

and is started as  batch.py --pairs LIST --out OUT_DIR  (LIST: pair folder paths, one per line), with
CUDA_VISIBLE_DEVICES set to one GPU. For every pair without OUT_DIR/NNNN.mp4 it calls make(), copies the video
stream, adds the returned audio as AAC, cuts to A's clip length, and writes NNNN.mp4 (atomically) and NNNN.json.
A failing pair is logged and skipped; running again retries it.

Pair folders (Precomputed-ltx pairs4/NNNN) hold a_guide.mp4 (A's clip, with A's audio), b_face_1024.png and
b_face_512.jpg (B's face crop), b_frame.jpg (the frame it was cut from), b_source.mp4 (B's own clip, with B's
audio), caption.txt and job.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from functools import cached_property
from pathlib import Path


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


class Pair:
    """Inputs of one pair. Derived files (16 kHz audio, 25 fps video) are made in tmp on first use."""

    def __init__(self, folder, tmp):
        self.dir = Path(folder)
        self.name = self.dir.name
        self.tmp = Path(tmp)
        self.video = self.dir / "a_guide.mp4"       # A's clip, original fps and size, with A's audio
        self.ref = self.dir / "b_face_512.jpg"      # B's face, square crop
        self.ref_1024 = self.dir / "b_face_1024.png"
        self.ref_frame = self.dir / "b_frame.jpg"   # the whole frame B's face was cut from
        self.ref_video = self.dir / "b_source.mp4"  # B's own clip, with B's audio
        self.seed = int(self.name)

    @cached_property
    def prompt(self):
        return (self.dir / "caption.txt").read_text().strip()

    @cached_property
    def job(self):
        return json.loads((self.dir / "job.json").read_text())

    @cached_property
    def _info(self):
        return probe(self.video)

    @cached_property
    def duration(self):
        return float(self._info["format"]["duration"])

    @cached_property
    def fps(self):
        v = next(s for s in self._info["streams"] if s["codec_type"] == "video")
        num, den = v["r_frame_rate"].split("/")
        return float(num) / float(den)

    @cached_property
    def video_25(self):
        """A's clip at 25 fps."""
        out = self.tmp / "a_25fps.mp4"
        ffmpeg("-i", self.video, "-r", "25", "-c:v", "libx264", "-crf", "12", "-c:a", "copy", out)
        return out

    @cached_property
    def audio(self):
        """A's audio, 16 kHz mono WAV."""
        out = self.tmp / "a_16k.wav"
        ffmpeg("-i", self.video, "-vn", "-ac", "1", "-ar", "16000", out)
        return out

    @cached_property
    def ref_audio(self):
        """B's audio, 16 kHz mono WAV, cut to A's clip length."""
        out = self.tmp / "b_16k.wav"
        ffmpeg("-i", self.ref_video, "-vn", "-ac", "1", "-ar", "16000", "-t", f"{self.duration:.3f}", out)
        return out


def frames(seconds, fps, step, offset):
    """Largest frame count <= seconds*fps of the form step*k + offset (video DiTs want 4k+1 or 8k+1)."""
    return int((seconds * fps - offset) // step * step + offset)


def finish(video, audio, out, duration):
    """out = video's stream (copied) + audio's first audio stream as AAC, cut to duration, the video's length or the
    audio's, whichever is shortest, so neither runs past the other."""
    duration = min(duration, *(float(probe(f)["format"]["duration"]) for f in (video, audio)))
    ffmpeg("-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{duration:.3f}", "-f", "mp4", out)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run(load, make, settings=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True, help="text file: pair folder paths, one per line")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = [Path(l.strip()) for l in a.pairs.read_text().splitlines() if l.strip()]
    todo = [d for d in pairs if not (a.out / f"{d.name}.mp4").exists()]
    tag = Path(sys.argv[0]).resolve().parent.name
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    log(f"{tag} gpu {gpu}: {len(todo)} of {len(pairs)} pairs to make; loading")
    t0 = time.time()
    model = load()
    log(f"{tag} gpu {gpu}: loaded in {time.time() - t0:.0f} s")
    failed = 0
    scratch = os.environ.get("PAIRIO_TMP")
    if scratch:
        os.makedirs(scratch, exist_ok=True)
    for d in todo:
        dest = a.out / f"{d.name}.mp4"
        if dest.exists():
            continue
        tmp = Path(tempfile.mkdtemp(prefix=f"{tag}-{d.name}-", dir=scratch))
        t = time.time()
        try:
            p = Pair(d, tmp)
            video, audio = make(model, p)
            part = dest.with_name(dest.name + ".part")
            finish(video, audio, part, p.duration)
            sec = time.time() - t
            (a.out / f"{d.name}.json").write_text(json.dumps(
                {"pair": d.name, "generator": tag, "seconds": round(sec, 1), "gpu": gpu, "seed": p.seed,
                 "settings": settings or {}}, indent=1))
            part.replace(dest)
            log(f"{tag} gpu {gpu}: {d.name} done in {sec:.0f} s")
        except Exception as e:
            failed += 1
            log(f"{tag} gpu {gpu}: {d.name} FAILED: {e!r}")
            traceback.print_exc()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    log(f"{tag} gpu {gpu}: finished, {failed} failed")
