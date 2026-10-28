"""ffmpeg / ffprobe helpers."""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path


def probe(path: Path) -> dict:
    """Duration (s), width, height, fps and whether there is an audio stream."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    video = next((s for s in d["streams"] if s["codec_type"] == "video"), {})
    return {"duration": round(float(d["format"].get("duration", 0)), 3),
            "width": video.get("width"), "height": video.get("height"),
            "fps": round(float(Fraction(video["r_frame_rate"])), 3) if video.get("r_frame_rate") else None,
            "audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def wav(src: Path, dest: Path, duration: float) -> None:
    """16 kHz mono WAV of src's audio, cut or padded with silence to exactly duration seconds."""
    ffmpeg("-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-af", f"apad=whole_dur={duration}",
           "-t", f"{duration:.3f}", str(dest))


def at_fps(src: Path, dest: Path, fps: int) -> None:
    """Re-encode src at a fixed frame rate, keeping its resolution and audio."""
    ffmpeg("-i", str(src), "-r", str(fps), "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(dest))
