"""Model files for the BFS LTX-2.5 workflow: download to /data/models and verify."""
import hashlib
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODELS_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "models"

LTX = ("Lightricks/LTX-2.5", "49b093a39c19169a8a81721dcc541e07e945232f")
BFS = ("Alissonerdx/BFS-Best-Face-Swap-Video", "d0adb5394c6a1b05f2245ec33cb91051a8000853")

# (repo, revision, path in repo, ComfyUI folder, size in bytes, sha256)
FILES = [
    (*LTX, "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", "diffusion_models",
     42018190584, "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4"),
    (*LTX, "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", "text_encoders",
     26263858182, "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1"),
    (*LTX, "vae/ltx-2.5-video-vae-bf16.safetensors", "vae",
     1472223346, "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3"),
    (*LTX, "vae/ltx-2.5-audio-vae-bf16.safetensors", "vae",
     364866540, "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5"),
    (*LTX, "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "latent_upscale_models",
     995778752, "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8"),
    (*BFS, "ltx-2.5/head_swap_ltx25_r128_v1.1.safetensors", "loras",
     2466809120, "0f0338394e426ea3d988401d6303fdedcc93d226a504236d2001533e1c8c1de0"),
]


def dest(f):
    return MODELS_DIR / f[3] / Path(f[2]).name


def _marker(path):
    return path.with_name(path.name + ".sha256ok")


def status():
    """List of (file, problem or None)."""
    out = []
    for f in FILES:
        p = dest(f)
        if not p.exists():
            out.append((f, "missing"))
        elif p.stat().st_size != f[4]:
            out.append((f, f"size {p.stat().st_size} != {f[4]}"))
        elif not _marker(p).exists():
            out.append((f, "not verified"))
        else:
            out.append((f, None))
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch(f, token):
    from huggingface_hub import hf_hub_download

    repo, rev, path, _, size, sha = f
    target = dest(f)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target.exists() and target.stat().st_size == size):
        print(f"[models] fetch   {path} ({size / 1e9:.1f} GB)", flush=True)
        staging = MODELS_DIR / ".download"
        got = Path(hf_hub_download(repo, path, revision=rev, local_dir=staging, token=token))
        if got.stat().st_size != size:
            raise RuntimeError(f"{path}: downloaded {got.stat().st_size} bytes, expected {size}")
        os.replace(got, target)
    if not _marker(target).exists():
        print(f"[models] verify  {target.name}", flush=True)
        digest = _sha256(target)
        if digest != sha:
            target.unlink()
            raise RuntimeError(f"{target.name}: sha256 mismatch, file deleted; run again")
        _marker(target).write_text(digest + "\n")
    print(f"[models] ok      {target.name}", flush=True)


def download(token=None):
    token = token or os.environ.get("HF_TOKEN") or None
    need = sum(f[4] for f, problem in status() if problem in ("missing",) or (problem or "").startswith("size"))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(MODELS_DIR).free
    if need > free - 5e9:
        sys.exit(f"[models] need {need / 1e9:.0f} GB, only {free / 1e9:.0f} GB free on {MODELS_DIR}")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    errors = []
    with ThreadPoolExecutor(len(FILES)) as pool:
        for f, fut in [(f, pool.submit(_fetch, f, token)) for f in FILES]:
            try:
                fut.result()
            except Exception as e:  # report every file, then fail
                errors.append(f"{f[2]}: {e}")
    shutil.rmtree(MODELS_DIR / ".download", ignore_errors=True)
    if errors:
        for e in errors:
            print(f"[models] FAILED  {e}", flush=True)
        if any("401" in e or "403" in e or "gated" in e.lower() for e in errors):
            print("[models] a gated repo refused the token: accept the terms on its Hugging Face page "
                  "with the same account, then run again", flush=True)
        sys.exit(1)
    print(f"[models] all {len(FILES)} files present and verified in {MODELS_DIR}", flush=True)
