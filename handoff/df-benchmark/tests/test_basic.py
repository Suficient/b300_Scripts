import json
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from df_benchmark import config as config_mod
from df_benchmark import generators, media
from df_benchmark.generators import resolve
from df_benchmark.identities import Pool
from df_benchmark.prepare import Job, square_crop
from df_benchmark.run import run

GENERATORS_DIR = Path(__file__).resolve().parent.parent / "generators"


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def pool(tmp_path):
    db = tmp_path / "cf.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE persons (id INTEGER PRIMARY KEY, emb_sum BLOB)")
    c.execute("CREATE TABLE clips (id INTEGER PRIMARY KEY, person_id INTEGER, fate TEXT, path TEXT, results_json TEXT)")
    people = {1: [1, 0, 0], 2: [0.8, 0.6, 0], 3: [0.3, 0.95, 0], 4: [0, 0, 1], 5: [0.6, 0.8, 0]}
    for pid, v in people.items():
        c.execute("INSERT INTO persons VALUES (?, ?)", (pid, (unit(v) * 3).tobytes()))  # sums, not means
    for pid, fate, path, score in [(1, "accepted", "a.mp4", 1.0), (2, "accepted", "b.mp4", 2.1),
                                   (2, "accepted", "b2.mp4", 2.9), (2, "accepted", "b3.mp4", None),
                                   (3, "accepted", "c.mp4", None), (4, "accepted", "d.mp4", 1.0),
                                   (5, "tossed", "e.mp4", 9.0)]:
        c.execute("INSERT INTO clips (person_id, fate, path, results_json) VALUES (?, ?, ?, ?)",
                  (pid, fate, path, json.dumps({"score": score})))
    c.commit()
    return Pool(db, tmp_path)


def test_nearest_skips_same_person_and_excluded(pool):
    # person 1 is the same person (sim 1.0); person 5 has no accepted clip
    assert pool.nearest(unit([1, 0, 0]), 0.95)[0] == 2
    assert pool.nearest(unit([1, 0, 0]), 0.95, exclude={2})[0] == 3
    assert pool.nearest(unit([1, 0, 0]), -1.0) is None


def test_best_clip_is_highest_score_unscored_last(pool, tmp_path):
    assert pool.best_clip(2) == (tmp_path / "b2.mp4", 2.9)
    assert pool.best_clip(3) == (tmp_path / "c.mp4", None)


def test_person_of_matches_by_file_name(pool, tmp_path):
    assert pool.person_of(tmp_path / "elsewhere" / "c.mp4") == 3
    assert pool.person_of(tmp_path / "_.mp4") is None  # "_" is a LIKE wildcard; must not match "a.mp4"


def test_square_crop_stays_inside_the_frame():
    img = np.zeros((1080, 1920, 3), np.uint8)
    assert square_crop(img, np.array([10, 10, 100, 120]), 2.0).shape == (240, 240, 3)  # shifted, not clipped
    assert square_crop(img, np.array([900, 500, 600, 600]), 2.5).shape == (1080, 1080, 3)  # shrunk to the frame


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[generators.x]\nscrip = 'facefusion'\n")
    with pytest.raises(KeyError):
        config_mod.load(p)


def test_resolve_aliases_and_comfy(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("gen_root = '/opt/gen'\n[generators.ff-inswapper]\nscript = 'facefusion'\nenv = { FF_MODEL = 'x' }\n"
                 "[comfy.bfs-ltx23]\nworkflow = 'w.json'\n")
    cfg = config_mod.load(p)
    g = resolve(cfg, "ff-inswapper")
    assert (g.script.parent.name, g.dir, g.env) == ("facefusion", Path("/opt/gen/facefusion"), {"FF_MODEL": "x"})
    assert resolve(cfg, "bfs-ltx23").comfy == {"workflow": "w.json"}
    with pytest.raises(KeyError):
        resolve(cfg, "nope")


def test_every_generator_has_setup_and_script_ones_run():
    for name in generators.GENERATORS:
        assert (GENERATORS_DIR / name / "setup.sh").exists(), name
        assert name == "bfs-ltx23" or (GENERATORS_DIR / name / "run.sh").exists(), name


def test_setup_spark_installs_every_generator():
    order = (GENERATORS_DIR.parent / "setup_spark.sh").read_text().split("ORDER=(", 1)[1].split(")", 1)[0].split()
    assert sorted(order) == sorted(generators.GENERATORS)


def test_reqs_filter(tmp_path):
    """common.sh's reqs(): drops what has no ARM64 build, swaps in replacements, keeps the rest."""
    gen = tmp_path / "generators" / "demo"
    gen.mkdir(parents=True)
    (tmp_path / "generators" / "common.sh").symlink_to(GENERATORS_DIR / "common.sh")
    (gen / "requirements.txt").write_text(
        "torch==2.2.2\ntorchvision==0.17.2\n--extra-index-url https://x\n-r base.txt\n-e git+https://y#egg=z\n"
        "xformers==0.0.25\nflash_attn\ndecord==0.6.0\nonnxruntime-gpu==1.18.0\nmediapipe[vision]==0.10.11\n"
        "diffusers==0.32.2  # pinned\nbitsandbytes==0.43.1\nonnx==1.16.1\n\n")
    (gen / "setup.sh").write_text('. "$(dirname "$0")/../common.sh"\ninstall() { cat "$2"; }\nort_gpu() { :; }\n'
                                  'SKIP="bitsandbytes" reqs "$(dirname "$0")/requirements.txt"\n')
    (tmp_path / "install").mkdir()  # the clone
    out = subprocess.run(["bash", str(gen / "setup.sh"), str(tmp_path / "install")], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["diffusers==0.32.2", "onnx==1.16.1", "decord2", "mediapipe==0.10.14"]


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    media.ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2", "-f", "lavfi", "-i",
                 "sine=frequency=440:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(p))
    return p


def test_wav_is_padded_to_length(clip, tmp_path):
    media.wav(clip, tmp_path / "a.wav", 3.0)
    assert media.probe(tmp_path / "a.wav")["duration"] == pytest.approx(3.0, abs=0.01)


def test_run_times_a_script_generator(clip, tmp_path, monkeypatch):
    """A fake generator whose run.sh checks every DFB_ variable and copies the clip to DFB_OUT."""
    scripts = tmp_path / "generators"
    (scripts / "facefusion").mkdir(parents=True)
    (scripts / "facefusion" / "run.sh").write_text(
        "set -eu\nfor v in VIDEO VIDEO_25 AUDIO REF REF_FRAME REF_AUDIO DURATION PROMPT SEED OUT TMP; do\n"
        '  eval "test -n \\"\\$DFB_$v\\""; done\n'
        'test -d "$DFB_TMP"; test "$FF_MODEL" = x; cp "$DFB_VIDEO" "$DFB_OUT"\n')
    monkeypatch.setattr(generators, "SCRIPTS", scripts)
    job = tmp_path / "work" / "clip"
    job.mkdir(parents=True)
    for f in ("driving.mp4", "driving_25fps.mp4"):
        (job / f).write_bytes(clip.read_bytes())
    (job / "job.json").write_text(json.dumps({"duration": 2.0, "prompt": "p"}))
    p = tmp_path / "c.toml"
    p.write_text(f"results_dir = '{tmp_path / 'results'}'\ngen_root = '{tmp_path}'\n"
                 "[generators.ff]\nscript = 'facefusion'\nenv = { FF_MODEL = 'x' }\n")
    (tmp_path / "facefusion").mkdir()  # the "install"
    cfg = config_mod.load(p)
    s = run(cfg, resolve(cfg, "ff"), [Job(job)])
    assert (s["ok"], s["clips"], s["failed"]) == (1, 1, [])
    assert s["output"]["width"] == 320 and s["output"]["audio"]
    assert (tmp_path / "results" / "ff" / "clip.mp4").exists()
    # a second run skips the finished clip; --force reruns it and the summary keeps one record per clip
    assert run(cfg, resolve(cfg, "ff"), [Job(job)])["clips"] == 1
    assert run(cfg, resolve(cfg, "ff"), [Job(job)], force=True)["clips"] == 1
    assert len((tmp_path / "results" / "ff" / "timings.jsonl").read_text().splitlines()) == 2


def test_failed_clip_is_recorded_not_raised(clip, tmp_path, monkeypatch):
    scripts = tmp_path / "generators"
    (scripts / "wav2lip").mkdir(parents=True)
    (scripts / "wav2lip" / "run.sh").write_text("echo boom; exit 3\n")
    monkeypatch.setattr(generators, "SCRIPTS", scripts)
    job = tmp_path / "work" / "clip"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"duration": 2.0, "prompt": "p"}))
    (tmp_path / "wav2lip").mkdir()
    p = tmp_path / "c.toml"
    p.write_text(f"results_dir = '{tmp_path / 'results'}'\ngen_root = '{tmp_path}'\n")
    cfg = config_mod.load(p)
    s = run(cfg, resolve(cfg, "wav2lip"), [Job(job)])
    assert s["ok"] == 0 and s["failed"] == ["clip"]
    assert "boom" in (tmp_path / "results" / "wav2lip" / "logs" / "clip.log").read_text()
