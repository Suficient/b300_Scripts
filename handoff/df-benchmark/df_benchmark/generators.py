"""The generators and how each one is run. GENERATORS.md has why each was picked."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "generators"

# name -> category. A name is a folder in generators/ (setup.sh + run.sh) unless it is run through ComfyUI.
GENERATORS = {
    "facefusion": "face swap",
    "dreamid-v": "face swap",
    "bfs-ltx23": "face swap",  # ComfyUI
    "wan-animate": "face swap",
    "liveportrait": "reenactment",
    "personalive": "reenactment",
    "infinitetalk": "talking head",
    "longcat-avatar": "talking head",
    "hallo2": "talking head",
    "hunyuan-avatar": "talking head",
    "latentsync": "lip sync",
    "musetalk": "lip sync",
    "wav2lip": "lip sync",
    "x-dub": "lip sync",
    "ltx-a2v": "full synthesis",
}


@dataclass
class Generator:
    name: str
    category: str
    script: Path | None = None  # run.sh, for script generators
    dir: Path | None = None  # its install, the working directory run.sh runs in
    env: dict = field(default_factory=dict)
    comfy: dict | None = None  # [comfy.<name>] config, for ComfyUI generators


def resolve(cfg, name: str) -> Generator:
    """A generator by name: one of GENERATORS, or an alias from [generators.<name>] that points at a script
    (e.g. facefusion with a different swapper model)."""
    g = cfg.generators.get(name, {})
    script = g.get("script", name)
    if script not in GENERATORS and name not in cfg.comfy:
        raise KeyError(f"unknown generator {name!r} (known: {', '.join(GENERATORS)})")
    category = GENERATORS.get(script, "face swap")
    if name in cfg.comfy:
        return Generator(name, category, comfy=cfg.comfy[name])
    run_sh = SCRIPTS / script / "run.sh"
    if not run_sh.exists():
        raise KeyError(f"{name}: needs either generators/{script}/run.sh or a [comfy.{name}] section in the config")
    install = Path(g["dir"]).expanduser() if "dir" in g else cfg.gen_root / script
    return Generator(name, category, run_sh, install, {k: str(v) for k, v in g.get("env", {}).items()})
