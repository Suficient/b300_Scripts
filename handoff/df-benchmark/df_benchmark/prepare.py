"""Turn each input clip into a job folder: the clip, its audio, and a reference image + audio of the closest
different-looking person in Clipfarmer's index. Every generator then runs on the same jobs."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import media
from .faces import Faces
from .identities import Pool

log = logging.getLogger(__name__)

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm")


@dataclass
class Job:
    """One prepared clip. Files are in dir; see prepare() for what each one is."""
    dir: Path

    @property
    def stem(self) -> str:
        return self.dir.name

    @property
    def meta(self) -> dict:
        return json.loads((self.dir / "job.json").read_text())


def prepare(cfg, clips: list[Path]) -> list[Job]:
    """Writes work_dir/<clip stem>/:
      driving.mp4        the clip, unchanged
      driving_25fps.mp4  the clip at 25 fps (what most lip-sync and portrait models are trained on)
      driving.wav        the clip's audio, 16 kHz mono
      ref.png            the match's face, square crop (face box x ref_crop_scale), ref_size px
      ref_frame.png      the whole frame ref.png was cut from
      ref_audio.wav      the match's clip audio, 16 kHz mono, cut or padded to the clip's length
      job.json           who was matched and why
    Clips without a face, or without anyone to match, are skipped."""
    faces = Faces(cfg.models_dir, cfg.face_det_min)
    pool = Pool(cfg.clipfarmer_db, cfg.clipfarmer_root)
    jobs = []
    for clip in clips:
        scan = faces.scan(clip, cfg.scan_frames)
        if scan is None:
            log.warning("%s: no face found, skipped", clip.name)
            continue
        own = pool.person_of(clip)
        match = pool.nearest(scan[0], cfg.same_person_sim, exclude={own} if own is not None else set())
        if match is None:
            log.warning("%s: no one in the identity pool to match", clip.name)
            continue
        pid, sim = match
        src, score = pool.best_clip(pid)
        ref_scan = faces.scan(src, cfg.ref_frames)
        if ref_scan is None:
            log.warning("%s: no face in person %d's clip %s, skipped", clip.name, pid, src)
            continue
        _, img, face = ref_scan

        d = cfg.work_dir / clip.stem
        d.mkdir(parents=True, exist_ok=True)
        info = media.probe(clip)
        shutil.copyfile(clip, d / "driving.mp4")
        media.at_fps(clip, d / "driving_25fps.mp4", 25)
        media.wav(clip, d / "driving.wav", info["duration"])
        media.wav(src, d / "ref_audio.wav", info["duration"])
        cv2.imwrite(str(d / "ref.png"), cv2.resize(square_crop(img, face.box, cfg.ref_crop_scale),
                                                   (cfg.ref_size, cfg.ref_size), interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(d / "ref_frame.png"), img)
        (d / "job.json").write_text(json.dumps(
            {"clip": str(clip), "clip_person_id": own, "duration": info["duration"], "fps": info["fps"],
             "match_person_id": pid, "similarity": round(sim, 4), "match_clip": str(src), "match_clip_score": score,
             "prompt": cfg.prompt}, indent=2))
        log.info("%s -> person %d (sim %.2f) from %s", clip.name, pid, sim, src.name)
        jobs.append(Job(d))
    return jobs


def jobs_in(work_dir: Path) -> list[Job]:
    return [Job(p.parent) for p in sorted(work_dir.glob("*/job.json"))]


def clips_in(clips_dir: Path) -> list[Path]:
    return sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def square_crop(img: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
    """A square around the face box, side = longer box side x scale. Shifted to stay inside the frame;
    only shrunk if the frame itself is smaller than the square."""
    H, W = img.shape[:2]
    x, y, w, h = box
    side = int(min(max(w, h) * scale, H, W))
    x0 = int(np.clip(x + w / 2 - side / 2, 0, W - side))
    y0 = int(np.clip(y + h / 2 - side / 2, 0, H - side))
    return img[y0:y0 + side, x0:x0 + side]
