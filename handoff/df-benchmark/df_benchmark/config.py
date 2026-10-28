"""Every setting in one place. Override any field from a TOML file (see config.example.toml)."""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- paths ---
    clipfarmer_db: Path = Path("~/imprint/clipfarmer/data/clipfarmer.sqlite")
    clipfarmer_root: Path = Path("~/imprint/clipfarmer")  # clip paths in the DB are relative to this
    clips_dir: Path = Path("clips")  # the clips to fake
    work_dir: Path = Path("work")  # prepared inputs, one folder per clip
    results_dir: Path = Path("results")  # results/<generator>/<clip stem>.mp4
    models_dir: Path = Path("models")  # face detection + embedding weights
    gen_root: Path = Path("~/df-gen")  # generator installs: <gen_root>/<generator>

    # --- identity matching ---
    same_person_sim: float = 0.45  # at or above this, a "match" is probably the same person (Clipfarmer notes.md)
    face_det_min: float = 0.5
    scan_frames: int = 8  # frames sampled per clip to embed its face
    ref_frames: int = 24  # frames sampled from the match's clip to pick the reference image
    ref_crop_scale: float = 2.0  # ref.png = face box scaled by this, square
    ref_size: int = 512  # ref.png is resized to this

    # --- generation ---
    seed: int = 42
    prompt: str = ("A person talks to the camera in a webcam video, head and shoulders in frame, natural indoor "
                   "lighting, realistic.")
    timeout_s: float = 7200  # per clip

    # --- per generator: script (folder in generators/), dir (install), env (extra variables) ---
    generators: dict = field(default_factory=dict)

    # --- ComfyUI generators: [comfy.<name>] with workflow, output_node, inputs ---
    comfy_url: str = "http://127.0.0.1:8188"
    comfy: dict = field(default_factory=dict)


def load(path: str | Path | None) -> Config:
    """Defaults overridden by the TOML file. Unknown keys raise, so a typo fails at startup."""
    cfg = Config()
    raw = tomllib.loads(Path(path).read_text()) if path else {}
    for k, v in raw.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key: {k}")
        setattr(cfg, k, v)
    for f in dataclasses.fields(cfg):
        if f.type == "Path":
            setattr(cfg, f.name, Path(getattr(cfg, f.name)).expanduser())
    for name, g in cfg.generators.items():
        if unknown := set(g) - {"script", "dir", "env"}:
            raise KeyError(f"unknown keys in [generators.{name}]: {sorted(unknown)}")
    for name, c in cfg.comfy.items():
        if unknown := set(c) - {"workflow", "output_node", "inputs"}:
            raise KeyError(f"unknown keys in [comfy.{name}]: {sorted(unknown)}")
    return cfg
