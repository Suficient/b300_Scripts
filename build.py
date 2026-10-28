"""BFS LTX-2.5 face-swap pairs from the clipfarmer-data snapshot.

Every clip that can be a guide (A) gets the nearest different person (B) by ArcFace embedding who also looks alike
enough for a head swap, and B's face image from B's best clip. Claude Haiku (through `claude -p`, so it runs on the
Claude subscription) reads every face's gender, skin tone, hair and age, which decide the match, and writes the FACE
prompt the BFS workflow expects.

Three stages, each saved in <cache>/cache.sqlite as it goes, so a run that stops resumes where it left off:
1. Scan (CPU, in parallel): every clip is checked for text once, and each person gets one face image (their B
   identity image).
2. Label (Haiku): every face crop that can be used, once.
3. Match and write: each A clip not yet in the batch gets a B. Pairs land in <batch>/NNNN/, with the batch's
   index.html, pairs.csv and rejected.csv. Pairs already in the batch stay as they are.

Filters:
- Clipfarmer's labels (A and B): no glasses, nothing on the head. B's face image also needs no hands near the face
  (a hand over B's face spoils the identity image; A's hands are allowed).
- Text: none anywhere in A's frame (LTX regenerates the whole frame and garbles text), none in B's face crop.
- Haiku: head not covered and looks 21 or older (A and B); same gender, skin tone within SKIN_STEPS, similar hair and
  age gap within MAX_AGE_GAP (A vs B).

Run with Clipfarmer's venv (it has the face models):
  ~/Projects/Code/Clipfarmer/.venv/bin/python build.py --batch pairs4 \
      --exclude pairs.csv pairs2/pairs.csv pairs3/pairs.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import multiprocessing
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np

CLAUDE_ENV = dict(os.environ)  # `claude` gets the user's environment, not the PATH, XDG_* and TMPDIR clipfarmer sets
CLIPFARMER = Path.home() / "Projects/Code/Clipfarmer"
DATA = Path.home() / "Projects/Code/clipfarmer-data"  # snapshot: clipfarmer.sqlite + data/clips/accepted
sys.path.insert(0, str(CLIPFARMER))
import clipfarmer.models as cf_models  # noqa: E402
from clipfarmer.models import InsightFaces, sharpness  # noqa: E402

OUT = Path(__file__).resolve().parent
SAME_PERSON_SIM = 0.45  # at or above this a "match" is probably the same person (df-benchmark, Clipfarmer notes)
TIMES = (0.5, 3.0, 5.5)  # where Clipfarmer's recheck samples every clip
TEXT_TIMES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)  # A frames checked for text: captions can appear mid-clip
TEXT_MODEL = OUT / "models/text_detection_en_ppocrv3_2023may.onnx"  # PP-OCRv3 DB detector, OpenCV zoo, Apache-2.0
CROP_SCALE = 2.0  # face image = square of the face box's longer side x this (df-benchmark's ref crop)
ID_SIZE = 1024  # the BFS workflow resizes the identity image to 1024x1024
CAPTION_SIZE = 512  # what Haiku sees: 19x19 patches = 361 image tokens

SKIN = ("very light", "light", "medium", "tan", "brown", "dark")
TEXTURE = ("straight", "wavy", "curly", "coily")  # neighbours count as similar
HAIR = (*TEXTURE, "braids/locs", "bald", "covered")
LENGTH = ("short", "medium", "long")
FIELDS = ("gender", "skin", "hair", "length", "age")
SKIN_STEPS = 1  # largest skin-tone difference between A and B, in steps of SKIN
MAX_AGE_GAP = 20
MIN_AGE = 21  # by Haiku's estimate; Clipfarmer's age gate draws the line at 21 too
MAX_B_USES = 10  # past this many pairs a B person is picked only when no other B fits
LABEL_TRIES = 3  # Haiku calls a face gets before it counts as unlabelled
MAX_WAIT = 6 * 3600  # how long the label stage waits out failing calls (usage limit, outage) before it stops
THREADS = 2  # ONNX Runtime and OpenCV threads per scan worker

SYSTEM = "You label face images for a dataset. Follow the output format exactly."
# The BFS LTX-2.5 v1.1 workflow's prompt template, reworded for many images per call (one numbered line per image
# instead of a head_swap block each, traits that aren't clearly visible left out), plus the fields matching uses.
PROMPT = """Each of these images shows one person's face and head, and has its number printed in a black box in its
top-left corner and in the text just before it.
Describe every image on its own. Never compare images or refer to another image.

For each image output exactly one line of seven fields separated by " | ":
<number printed in the image> | <gender> | <skin> | <hair> | <length> | <age> | <face>
- gender: male or female
- skin: skin tone, one of: very light, light, medium, tan, brown, dark
- hair: one of: straight, wavy, curly, coily, braids/locs, bald, covered (a hat, cap, hood, headwrap or veil
  hides most of the hair)
- length: hair length, one of: short, medium, long. Use short for bald.
- age: apparent age in years, a single number such as 35
- face: one FACE description:
  - Describe only the most important visible identity traits of the face and head: apparent gender, approximate age, skin tone, eye color, hair color, basic hairstyle, and one distinctive facial feature only if clearly relevant.
  - Leave out any trait that isn't clearly visible. Do not mention absent features such as "no facial hair".
  - Do not describe clothing, body, pose, expression, background, lighting, camera, or minor facial anatomy.
  - One short English sentence, approximately 8-20 words.
  Example: Female, pale skin, approximately 20-30 years old, red eyes, long black hair.

For example: 001 | male | light | straight | short | 35 | Male, light skin, approximately 30-40 years old, brown eyes, short dark brown hair.
Output nothing else."""
CLAUDE = ["claude", "-p", "--model", "haiku", "--system-prompt", SYSTEM, "--tools", "", "--strict-mcp-config",
          "--no-session-persistence", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]

# A cache made with other settings would mix two kinds of results
SETTINGS = {"scan": dict(filters="glasses.none>=0.5 headwear.yes<0.5, B hands_near_face.yes<0.5", times=TIMES,
                         text_times=TEXT_TIMES, text_model=TEXT_MODEL.name, text_side=(960, 640), crop_scale=CROP_SCALE,
                         caption_size=CAPTION_SIZE),
            "label": dict(model="haiku", prompt=hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest()[:16])}
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
-- one row per clip the scan checked as A. status: ok (can be A), text, short (a frame missing), noface
CREATE TABLE IF NOT EXISTS scan (clip INTEGER PRIMARY KEY, person INTEGER, status TEXT, text_at REAL,
    t REAL, box TEXT, crop_px INTEGER, crop_text INTEGER, sim REAL);
-- each scanned person's face image (clip NULL: no clip without hands has one without text); b_rejects: crops
-- passed over for text
CREATE TABLE IF NOT EXISTS ident (person INTEGER PRIMARY KEY, clip INTEGER, t REAL, box TEXT, crop_px INTEGER,
    sim REAL);
CREATE TABLE IF NOT EXISTS b_rejects (clip INTEGER PRIMARY KEY, person INTEGER);
CREATE TABLE IF NOT EXISTS labels (clip INTEGER PRIMARY KEY, gender TEXT, skin TEXT, hair TEXT, length TEXT,
    age INTEGER, face TEXT, tries INTEGER NOT NULL DEFAULT 0, call INTEGER);
CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, at REAL, n INTEGER, s REAL, cost REAL, usage TEXT,
    result TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS pairs (batch TEXT, n INTEGER, a_clip INTEGER, b_person INTEGER, b_clip INTEGER,
    sim REAL, PRIMARY KEY (batch, n), UNIQUE (batch, a_clip));
"""


def clean(labels: dict) -> bool:
    """No glasses or headwear by Clipfarmer's labels. An unreadable answer fails."""
    try:
        return labels["glasses"]["none"] >= 0.5 and labels["headwear"]["yes"] < 0.5
    except (KeyError, TypeError):
        return False


def hands(labels: dict) -> float:
    try:
        return float(labels["hands_near_face"]["yes"])
    except (KeyError, TypeError, ValueError):
        return 1.0


def load(db: Path):
    """Accepted clips that pass the label filters, and the mean embeddings of their people."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    clips = []
    for i, p, v, pid, e, s, lj in conn.execute(
            "SELECT id, path, video_id, person_id, emb, json_extract(results_json, '$.score'), labels_json"
            " FROM clips WHERE fate='accepted' AND path != '' AND person_id IS NOT NULL AND emb IS NOT NULL"):
        labels = json.loads(lj or "{}")
        if clean(labels):
            clips.append(dict(id=i, path=str(DATA / p), video=v, person=pid, emb=np.frombuffer(e, np.float32),
                              score=s if s is not None else -1e9, hands=hands(labels)))
    persons = {c["person"] for c in clips}
    pids, embs = [], []
    for pid, b in conn.execute("SELECT id, emb_sum FROM persons"):
        if pid in persons:
            pids.append(pid)
            embs.append(np.frombuffer(b, np.float32))
    m = np.stack(embs)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return clips, np.array(pids), m


def frames(path: Path, times):
    """(t, frame) for each of `times`, from one sequential decode. The clips have one keyframe, so seeking to each
    time decodes from the start every time."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        want = {round(t * fps): t for t in times}
        i, last = 0, max(want)
        while i <= last and cap.grab():
            if i in want:
                ok, img = cap.retrieve()
                if ok:
                    yield want[i], img
            i += 1
    finally:
        cap.release()


def square(shape: tuple, box, scale: float) -> tuple[int, int, int]:
    """x0, y0, side of a square around the face box: side = longer box side x scale, shifted to stay inside the
    frame."""
    H, W = shape[:2]
    x, y, w, h = box
    side = int(min(max(w, h) * scale, H, W))
    return int(np.clip(x + w / 2 - side / 2, 0, W - side)), int(np.clip(y + h / 2 - side / 2, 0, H - side)), side


def square_crop(img: np.ndarray, box, scale: float) -> np.ndarray:
    x0, y0, side = square(img.shape, box, scale)
    return img[y0:y0 + side, x0:x0 + side]


def resized(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


class TextFinder:
    """PP-OCRv3's DB text detector through OpenCV. Detections centred inside a face box are dropped: there it fires
    on eyes, teeth and brows, and Clipfarmer's overlay gate already tossed clips with text on the face."""

    def __init__(self, path: Path):
        self.net = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(path)))
        self.net.setBinaryThreshold(0.3)
        self.net.setPolygonThreshold(0.5)
        self.net.setUnclipRatio(2.0)
        self.net.setMaxCandidates(200)

    def find(self, img: np.ndarray, face_boxes: list, side: int) -> list[np.ndarray]:
        """Text polygons in img's pixels. The detector sees img with its longer side resized to `side`."""
        H, W = img.shape[:2]
        s = side / max(H, W)
        size = (max(32, round(W * s / 32) * 32), max(32, round(H * s / 32) * 32))
        self.net.setInputSize(size)
        self.net.setInputMean((123.675, 116.28, 103.53))  # the OpenCV zoo demo's normalisation
        self.net.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
        polys, _ = self.net.detect(cv2.resize(img, size))
        out = []
        for p in polys:
            p = np.asarray(p, np.float32) * np.float32([W / size[0], H / size[1]])
            cx, cy = p.mean(0)
            if not any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in face_boxes):
                out.append(p)
        return out


def outlined(img: np.ndarray, polys: list[np.ndarray], width: int) -> np.ndarray:
    img = img.copy()
    cv2.polylines(img, [p.astype(np.int32) for p in polys], True, (0, 0, 255), 3)
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])))


# ---------------------------------------------------------------- scan (runs in worker processes)

W: dict = {}  # per scan worker: face models, text detector, cache folder


def init_worker(cache: str):
    import onnxruntime as ort

    def session(path):  # Clipfarmer's sessions take every core; here the workers share them
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = THREADS
        return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    cf_models._onnx_session = session
    cv2.setNumThreads(THREADS)
    W.update(faces=InsightFaces(CLIPFARMER / "models", 0.5), finder=TextFinder(TEXT_MODEL), cache=Path(cache))


def look(path: Path, a_check: bool):
    """One decode of the clip. With a_check, the TEXT_TIMES frames are checked for text outside a face, stopping at
    the first that has some. Returns (that frame's time or None, TEXT_TIMES frames checked, the sharpest TIMES
    frame's largest face by Laplacian variance as (t, frame, face) or None)."""
    best, seen = None, 0
    for t, img in frames(path, sorted({*TIMES, *TEXT_TIMES}) if a_check else TIMES):
        found = W["faces"].detect(img)
        if a_check and t in TEXT_TIMES:
            seen += 1
            if polys := W["finder"].find(img, [f.box for f in found], 960):
                cv2.imwrite(str(W["cache"] / "rejected" / f"A_{path.stem}.jpg"), outlined(img, polys, 640))
                return t, seen, None
        if t in TIMES and found:
            face = max(found, key=lambda f: f.box[3])
            s = sharpness(img, face.box, 64)
            if best is None or s > best[0]:
                best = (s, t, img, face)
    return None, seen, best[1:] if best else None


def face_image(clip: dict, best: tuple, p_emb: np.ndarray) -> dict:
    """Where the face crop is, whether it has text outside the face (then `thumb` has it outlined), and how close
    its face is to the person's mean embedding. Saves the crop Haiku reads to <cache>/faces/<clip>.jpg."""
    t, img, face = best
    x0, y0, side = square(img.shape, face.box, CROP_SCALE)
    crop = img[y0:y0 + side, x0:x0 + side]
    x, y, w, h = face.box
    polys = W["finder"].find(crop, [(x - x0, y - y0, w, h)], 640)
    cv2.imwrite(str(W["cache"] / "faces" / f"{clip['id']}.jpg"), resized(crop, CAPTION_SIZE),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dict(t=t, box=json.dumps([float(v) for v in face.box]), crop_px=side, crop_text=bool(polys),
                sim=float(W["faces"].embed(img, face) @ p_emb), thumb=outlined(crop, polys, 320) if polys else None)


def scan_person(person: int, clips: list[dict], p_emb: np.ndarray) -> dict:
    """Checks each of a person's clips as A, then picks their face image: from their clips without hands near the
    face, best-scored first, the face crop of the sharpest TIMES frame, if it has no text in it."""
    try:
        rows, imgs = [], {}
        for c in clips:
            text_at, seen, best = look(Path(c["path"]), True)
            status = ("text" if text_at is not None else "short" if seen < len(TEXT_TIMES)
                      else "noface" if best is None else "ok")
            row = dict(clip=c["id"], status=status, text_at=text_at)
            if status == "ok":
                imgs[c["id"]] = face_image(c, best, p_emb)
                row |= {k: v for k, v in imgs[c["id"]].items() if k != "thumb"}
            rows.append(row)
        ident, rejects = None, []
        for c in sorted((c for c in clips if c["hands"] < 0.5), key=lambda c: -c["score"]):
            got = imgs.get(c["id"])
            if got is None:
                best = look(Path(c["path"]), False)[2]
                if best is None:
                    continue
                got = face_image(c, best, p_emb)
            if got["crop_text"]:
                cv2.imwrite(str(W["cache"] / "rejected" / f"B_{Path(c['path']).stem}.jpg"), got["thumb"])
                rejects.append(c["id"])
                continue
            ident = dict(clip=c["id"], **{k: got[k] for k in ("t", "box", "crop_px", "sim")})
            break
        return dict(person=person, rows=rows, ident=ident, rejects=rejects)
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the scan; the person is retried next run
        return dict(person=person, error=f"{type(e).__name__}: {e}")


def scan(db: sqlite3.Connection, clips: list[dict], pids: np.ndarray, m: np.ndarray, cache: Path, workers: int,
         persons: int, seed: int):
    by_person: dict[int, list[dict]] = {}
    for c in clips:
        by_person.setdefault(c["person"], []).append(c)
    todo = sorted(by_person)
    if persons:
        random.Random(seed).shuffle(todo)
        todo = sorted(todo[:persons])
    done = {p for (p,) in db.execute("SELECT person FROM ident")}
    todo = sorted((p for p in todo if p not in done), key=lambda p: -len(by_person[p]))  # big ones first
    row_of = {int(p): i for i, p in enumerate(pids)}
    print(f"scan: {len(todo)} people to scan ({len(done)} done before), "
          f"{sum(len(by_person[p]) for p in todo)} clips", flush=True)
    t0, n_done, errors = time.time(), 0, 0
    for _ in range(3):  # a worker that dies takes the pool down; start a new one for what is left
        try:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=init_worker, initargs=(str(cache),)) as ex:
                futs = [ex.submit(scan_person, p, by_person[p], m[row_of[p]]) for p in todo]
                for f in as_completed(futs):
                    r = f.result()
                    if "error" in r:
                        errors += 1
                        print(f"scan: person {r['person']} failed, retried next run: {r['error']}", flush=True)
                        continue
                    with db:
                        db.executemany(
                            "INSERT OR REPLACE INTO scan VALUES (:clip, :person, :status, :text_at, :t, :box,"
                            " :crop_px, :crop_text, :sim)",
                            [dict(dict(t=None, box=None, crop_px=None, crop_text=None, sim=None), person=r["person"],
                                  **x) for x in r["rows"]])
                        db.executemany("INSERT OR REPLACE INTO b_rejects VALUES (?, ?)",
                                       [(c, r["person"]) for c in r["rejects"]])
                        i = r["ident"] or dict(clip=None, t=None, box=None, crop_px=None, sim=None)
                        db.execute("INSERT OR REPLACE INTO ident VALUES (:person, :clip, :t, :box, :crop_px, :sim)",
                                   dict(i, person=r["person"]))
                    n_done += 1
                    if n_done % 100 == 0 or n_done == len(todo):
                        print(f"scan: {n_done}/{len(todo)} people, {time.time() - t0:.0f} s", flush=True)
            break
        except BrokenProcessPool as e:
            done = {p for (p,) in db.execute("SELECT person FROM ident")}
            todo = [p for p in todo if p not in done]
            print(f"scan: worker pool died ({e}); {len(todo)} people left", flush=True)
    else:
        sys.exit("scan: the worker pool died 3 times; stopping")
    return errors


# ---------------------------------------------------------------- label

def parse(line: str) -> dict | None:
    """A label line '<n> | gender | skin | hair | length | age | face' as a dict, or None if a field is off."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 7 or not (mt := re.fullmatch(r"(?i)(?:image\s*)?(\d+)(?:\.jpg)?:?", parts[0])):
        return None
    _, gender, skin, hair, length, age, face = parts
    gender, skin, hair, length = (s.lower() for s in (gender, skin, hair, length))
    age = re.search(r"\d+", age)
    if gender not in ("male", "female") or skin not in SKIN or hair not in HAIR or length not in LENGTH \
            or not age or not face:
        return None
    return dict(key=int(mt.group(1)), gender=gender, skin=skin, hair=hair, length=length, age=int(age.group()),
                face=face)


def haiku(faces_dir: Path, clip_ids: list[int], tmp: Path) -> dict:
    """One `claude -p` call over these clips' face crops, sent in the message, each with its number printed on it
    (with numbers in file names alone, Haiku attached 7 of 20 captions to the next image:
    captions_raw_run1_unlabeled.json). Returns {"labels": {clip: fields}, ...} or {"error": why}."""
    t0 = time.time()
    try:
        return _haiku(faces_dir, clip_ids, tmp, t0)
    except Exception as e:  # noqa: BLE001 - counts as a failed call
        return dict(error=f"{type(e).__name__}: {e}", s=time.time() - t0)


def _haiku(faces_dir: Path, clip_ids: list[int], tmp: Path, t0: float) -> dict:
    content = []
    for i, cid in enumerate(clip_ids, 1):
        key = f"{i:03d}"
        img = cv2.imread(str(faces_dir / f"{cid}.jpg"))
        cv2.rectangle(img, (0, 0), (22 + 18 * len(key), 38), (0, 0, 0), -1)
        cv2.putText(img, key, (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        data = base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]).decode()
        content += [{"type": "text", "text": f"Image {key}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}]
    content.append({"type": "text", "text": PROMPT})
    msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}})
    try:
        r = subprocess.run(CLAUDE, input=msg + "\n", capture_output=True, text=True, cwd=tmp, timeout=900,
                           env=dict(CLAUDE_ENV, TMPDIR=str(tmp)))
    except subprocess.TimeoutExpired:
        return dict(error="timed out after 900 s", s=time.time() - t0)
    res = None
    for line in r.stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict) and e.get("type") == "result":
            res = e
    if not res or res.get("is_error") or res.get("subtype") != "success":
        why = (res or {}).get("result") or (r.stderr or r.stdout)[-2000:] or f"exit code {r.returncode}"
        return dict(error=str(why), s=time.time() - t0)
    labels = {}
    for line in (res.get("result") or "").splitlines():
        f = parse(line)
        if f and 1 <= f["key"] <= len(clip_ids):
            labels.setdefault(clip_ids[f["key"] - 1], f)
    return dict(labels=labels, result=res.get("result"), s=time.time() - t0, cost=res.get("total_cost_usd"),
                usage=res.get("modelUsage"))


def tried(db: sqlite3.Connection, clip: int, fields: dict | None, call: int | None):
    """Counts one Haiku try for the face, with its fields if the try labelled it."""
    db.execute("INSERT INTO labels (clip, gender, skin, hair, length, age, face, tries, call)"
               " VALUES (?,?,?,?,?,?,?,1,?) ON CONFLICT (clip) DO UPDATE SET gender=excluded.gender,"
               " skin=excluded.skin, hair=excluded.hair, length=excluded.length, age=excluded.age,"
               " face=excluded.face, tries=tries+1, call=excluded.call",
               (clip, *(tuple(fields[k] for k in (*FIELDS, "face")) if fields else (None,) * 6), call))


def label(db: sqlite3.Connection, keys: list[int], cache: Path, chunk: int, parallel: int) -> bool:
    """Haiku's fields for each of these clips' face crops, saved as each call returns. A face a call skipped or
    answered in the wrong format goes into a later call (in groups of 10, so one bad image costs few others), up to
    LABEL_TRIES calls. After a round with failed calls, one-face calls tell which way: if one works, the failed calls
    count as tries; if none does (usage limit, outage), this waits with backoff, up to MAX_WAIT in all. Returns False
    if it stopped with faces left; the next run labels those."""
    tmp, faces = cache / "tmp", cache / "faces"
    keys = sorted(k for k in keys if (faces / f"{k}.jpg").exists())
    random.Random(0).shuffle(keys)  # a mix of people in each call, as in the batches checked by eye
    backoff, waited = 60, 0
    while True:
        labelled = {c for (c,) in db.execute("SELECT clip FROM labels WHERE gender IS NOT NULL")}
        tries = dict(db.execute("SELECT clip, tries FROM labels"))
        todo = [k for k in keys if k not in labelled and tries.get(k, 0) < LABEL_TRIES]
        if not todo:
            return True
        fresh = [k for k in todo if not tries.get(k)]
        again = [k for k in todo if tries.get(k)]
        groups = [fresh[i:i + chunk] for i in range(0, len(fresh), chunk)] + \
                 [again[i:i + 10] for i in range(0, len(again), 10)]
        print(f"label: {len(todo)} faces in {len(groups)} calls", flush=True)
        failed, streak, outage, left = [], 0, False, len(todo)
        with ThreadPoolExecutor(parallel) as ex:
            futs = {ex.submit(haiku, faces, g, tmp): g for g in groups}
            for f in as_completed(futs):
                if f.cancelled():
                    continue
                g, r = futs[f], f.result()
                with db:
                    call = db.execute("INSERT INTO calls (at, n, s, cost, usage, result, error) VALUES (?,?,?,?,?,?,?)",
                                      (time.time(), len(g), r.get("s"), r.get("cost"), json.dumps(r.get("usage")),
                                       r.get("result"), r.get("error"))).lastrowid
                    if "error" not in r:
                        for cid in g:
                            tried(db, cid, r["labels"].get(cid), call)
                if "error" in r:
                    failed.append(g)
                    streak += 1
                    print(f"label: call failed: {r['error'][:300]}", flush=True)
                    if streak >= 2 * parallel and not outage:  # down or out of quota: stop the round
                        outage = True
                        for other in futs:
                            other.cancel()
                    continue
                streak = 0
                left -= len(r["labels"])
                print(f"label: {len(r['labels'])}/{len(g)} in {r['s']:.0f} s, {left} left", flush=True)
        if not failed:
            continue
        # One face on its own tells bad requests from an outage or usage limit: a face labelled before, else the
        # first face of up to three failed calls
        works = False
        known = db.execute("SELECT clip FROM labels WHERE gender IS NOT NULL LIMIT 1").fetchone()
        for cid in ([known[0]] if known else []) + [g[0] for g in failed[:3]]:
            r = haiku(faces, [cid], tmp)
            with db:
                call = db.execute("INSERT INTO calls (at, n, s, cost, usage, result, error) VALUES (?,?,?,?,?,?,?)",
                                  (time.time(), 0 if known and cid == known[0] else 1, r.get("s"), r.get("cost"),
                                   json.dumps(r.get("usage")), r.get("result"), r.get("error"))).lastrowid
                if "error" not in r and not (known and cid == known[0]):
                    tried(db, cid, r["labels"].get(cid), call)
            if works := "error" not in r:
                break
        if works:  # Haiku works: the failed calls were bad requests, or a blip they can retry
            labelled = {c for (c,) in db.execute("SELECT clip FROM labels WHERE gender IS NOT NULL")}
            with db:
                for g in failed:
                    for cid in g:
                        if cid not in labelled:
                            tried(db, cid, None, None)
            backoff = 60
            continue
        if waited >= MAX_WAIT:
            print(f"label: stopped after waiting {waited // 60} min for calls to work", flush=True)
            return False
        print(f"label: calls are failing; trying again in {backoff} s", flush=True)
        time.sleep(backoff)
        waited += backoff
        backoff = min(2 * backoff, 1800)


# ---------------------------------------------------------------- match

def unusable(f: dict | None) -> str | None:
    """Why a face can't be used on either side, by Haiku's labels."""
    if f is None:
        return "no Haiku label"
    if f["hair"] == "covered":
        return "head covered"
    if f["age"] < MIN_AGE:
        return f"looks {f['age']}"
    return None


def mismatch(a: dict, b: dict) -> str | None:
    """Why B's head doesn't suit A's body, by Haiku's labels."""
    if a["gender"] != b["gender"]:
        return "gender"
    if abs(SKIN.index(a["skin"]) - SKIN.index(b["skin"])) > SKIN_STEPS:
        return f"skin {a['skin']} vs {b['skin']}"
    if abs(a["age"] - b["age"]) > MAX_AGE_GAP:
        return f"age {a['age']} vs {b['age']}"
    ha, hb = a["hair"], b["hair"]
    if "bald" in (ha, hb):  # a bald head suits short hair of any texture
        ok = ha == hb or (a if hb == "bald" else b)["length"] == "short"
    elif "braids/locs" in (ha, hb):
        ok = ha == hb and abs(LENGTH.index(a["length"]) - LENGTH.index(b["length"])) <= 1
    else:
        ok = (abs(TEXTURE.index(ha) - TEXTURE.index(hb)) <= 1
              and abs(LENGTH.index(a["length"]) - LENGTH.index(b["length"])) <= 1)
    return None if ok else f"hair {a['length']} {ha} vs {b['length']} {hb}"


def match(db: sqlite3.Connection, batch: str, by_id: dict[int, dict], pids: np.ndarray, m: np.ndarray,
          labels: dict[int, dict], a_ids: list[int], seed: int, n_max: int) -> list[dict]:
    """Gives each A clip not yet in the batch a B: of the people whose face image fits A by Haiku's labels, the
    closest by ArcFace, preferring one not yet paired with A's person and used fewer than MAX_B_USES times. Saves
    the new pairs; returns the A clips it turned down."""
    old = list(db.execute("SELECT a_clip, b_person FROM pairs WHERE batch=?", (batch,)))
    uses = Counter(b for _, b in old)
    combos = {(by_id[a]["person"], b) for a, b in old}
    n = db.execute("SELECT coalesce(max(n), 0) FROM pairs WHERE batch=?", (batch,)).fetchone()[0]
    row_of = {int(p): i for i, p in enumerate(pids)}
    bs = [(p, c) for p, c in db.execute("SELECT person, clip FROM ident WHERE clip IS NOT NULL")
          if p in row_of and not unusable(labels.get(c))]
    b_pid = np.array([p for p, _ in bs])
    b_m = m[[row_of[p] for p, _ in bs]]
    b_lab = [labels[c] for _, c in bs]
    b_gender = np.array([f["gender"] for f in b_lab])
    b_skin = np.array([SKIN.index(f["skin"]) for f in b_lab])
    b_age = np.array([f["age"] for f in b_lab])
    paired = {a for a, _ in old}
    todo = sorted(a for a in a_ids if a not in paired)
    random.Random(seed).shuffle(todo)
    rejected, new = [], []
    for a in todo:
        if n_max and n >= n_max:
            break
        fa, c = labels.get(a), by_id[a]
        if why := unusable(fa):
            rejected.append(dict(clip=a, role="A", reason=why))
            continue
        sims = b_m @ (c["emb"] / (np.linalg.norm(c["emb"]) + 1e-9))
        ok = np.flatnonzero((b_pid != c["person"]) & (sims < SAME_PERSON_SIM) & (b_gender == fa["gender"])
                            & (np.abs(b_skin - SKIN.index(fa["skin"])) <= SKIN_STEPS)
                            & (np.abs(b_age - fa["age"]) <= MAX_AGE_GAP))
        best = None
        for i in ok:
            if mismatch(fa, b_lab[i]):
                continue
            p = int(b_pid[i])
            key = ((c["person"], p) in combos, uses[p] >= MAX_B_USES, -float(sims[i]))
            if best is None or key < best[0]:
                best = (key, i)
        if best is None:
            rejected.append(dict(clip=a, role="A", reason="no B fits"))
            continue
        i = best[1]
        p = int(b_pid[i])
        n += 1
        uses[p] += 1
        combos.add((c["person"], p))
        new.append((batch, n, a, p, bs[i][1], float(sims[i])))
    with db:
        db.executemany("INSERT INTO pairs VALUES (?,?,?,?,?,?)", new)
    print(f"match: {len(new)} new pairs, {len(rejected)} A clips turned down", flush=True)
    return rejected


# ---------------------------------------------------------------- write

def frame(path: str, t: float) -> np.ndarray:
    return next(img for _, img in frames(Path(path), [t]))


def write_pair(job: dict) -> str | None:
    """One pair's folder, written under a temporary name and renamed at the end, so a folder that exists is
    complete. Returns an error message or None."""
    out, tmp = Path(job["out"]), Path(job["out"] + ".part")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        a, b = job["a"], job["b"]
        shutil.copyfile(a["path"], tmp / "a_guide.mp4")
        shutil.copyfile(b["path"], tmp / "b_source.mp4")
        cv2.imwrite(str(tmp / "a_frame.jpg"), frame(a["path"], a["t"]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        shutil.copyfile(a["face"], tmp / "a_face.jpg")
        img = frame(b["path"], b["t"])
        cv2.imwrite(str(tmp / "b_frame.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(tmp / "b_face_1024.png"), resized(square_crop(img, json.loads(b["box"]), CROP_SCALE), ID_SIZE))
        shutil.copyfile(b["face"], tmp / "b_face_512.jpg")
        (tmp / "caption.txt").write_text(job["row"]["prompt"] + "\n")
        (tmp / "job.json").write_text(json.dumps(job["row"], indent=2))
        tmp.rename(out)
        return None
    except Exception as e:  # noqa: BLE001 - reported, and the folder is written on the next run
        shutil.rmtree(tmp, ignore_errors=True)
        return f"{out.name}: {type(e).__name__}: {e}"


def reason_kind(r: dict) -> str:
    return f"{r['role']} " + re.sub(r"(text at).*|(looks).*", lambda mt: next(g for g in mt.groups() if g), r["reason"])


def write_index(batch: Path, cache: Path, rows: list[dict], calls: tuple, rejected: list[dict]):
    n_calls, n_faces, secs, cost = calls
    rel = os.path.relpath(cache / "rejected", batch)

    def fields(r, side):
        return " &middot; ".join(html.escape(str(r[f"{side}_{k}"])) for k in FIELDS)

    trs = []
    for r in rows:
        p = r["pair"]
        trs.append(f"""<tr>
<td class=n>{p}</td>
<td><img loading=lazy src="{p}/a_face.jpg"><br><video src="{p}/a_guide.mp4" controls muted preload=none></video>
  <div class=meta>A: person {r['a_person']}<br>{fields(r, 'a')}<br>{html.escape(r['a_clip'])}</div></td>
<td><img loading=lazy src="{p}/b_face_512.jpg"><br><video src="{p}/b_source.mp4" controls muted preload=none></video>
  <div class=meta>B: person {r['b_person']}, frame at {r['b_frame_s']}s<br>{fields(r, 'b')}<br>
  {html.escape(r['b_clip'])}<br>native crop {r['b_crop_native_px']}px &rarr; 1024</div></td>
<td class=num>{r['person_sim']:.2f}<div class=meta>B image vs B person: {r['b_face_vs_b_person_sim']:.2f}<br>
  hands near A's face: {r['a_hands']:.2f}</div></td>
<td><pre>{html.escape(r['prompt'])}</pre></td>
</tr>""")
    counts = Counter(reason_kind(r) for r in rejected)
    rej = "".join(f"<figure><img loading=lazy src=\"{html.escape(rel)}/{html.escape(f.name)}\"><figcaption>"
                  f"{html.escape(f.stem)}</figcaption></figure>" for f in sorted((cache / "rejected").glob("*.jpg")))
    (batch / "index.html").write_text(f"""<!doctype html>
<meta charset=utf-8><title>BFS precompute: {batch.name}</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222 }}
table {{ border-collapse: collapse; width: 100% }}
td, th {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; text-align: left }}
td img {{ width: 220px; height: 220px; object-fit: cover; border-radius: 4px }}
video {{ width: 220px; margin-top: 4px }}
pre {{ white-space: pre-wrap; font: 13px ui-monospace, monospace; margin: 0; max-width: 360px }}
.meta {{ color: #666; font-size: 12px; margin-top: 4px; word-break: break-all; max-width: 220px }}
.n, .num {{ font-variant-numeric: tabular-nums }}
.rej {{ display: flex; flex-wrap: wrap; gap: 8px }}
figure {{ margin: 0 }} figure img {{ height: 140px; border-radius: 4px }}
figcaption {{ font-size: 11px; color: #666; max-width: 250px; word-break: break-all }}
</style>
<h1>BFS LTX-2.5 precompute: {batch.name}</h1>
<p>{len(rows)} clips (A), the nearest matching person for each (B), B's face image and the FACE prompt Claude Haiku
wrote for it. Haiku labelled {n_faces} faces for the cache this batch draws on: {n_calls} call(s), {secs:.0f} s,
${cost:.3f} at API list price (paid from the Claude subscription). Fields: gender &middot; skin &middot; hair &middot;
length &middot; age.</p>
<table>
<tr><th>#</th><th>A: guide clip</th><th>B: identity image (from B's clip)</th><th>ArcFace sim A&harr;B</th>
<th>Prompt</th></tr>
{''.join(trs)}
</table>
<h2>Rejected</h2>
<p>{html.escape(", ".join(f"{k}: {v}" for k, v in counts.most_common()))}. Every reason is in rejected.csv.
Text found in A frames and B face crops, outlined in red:</p>
<div class=rej>{rej}</div>
""")


# ---------------------------------------------------------------- main

def open_cache(path: Path) -> sqlite3.Connection:
    for d in ("faces", "rejected", "tmp"):
        (path / d).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path / "cache.sqlite")
    db.executescript(SCHEMA)
    for k, v in SETTINGS.items():
        v = json.dumps(v, sort_keys=True)
        old = db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        if old is None:
            with db:
                db.execute("INSERT INTO meta VALUES (?, ?)", (k, v))
        elif old[0] != v:
            sys.exit(f"{path} was made with other {k} settings ({old[0]}); use another --cache")
    return db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="output folder, next to this file")
    ap.add_argument("--n", type=int, default=0, help="most pairs the batch holds (default: as many as fit)")
    ap.add_argument("--seed", type=int, default=4, help="order in which A clips pick their B")
    ap.add_argument("--exclude", nargs="*", default=[], help="pairs.csv files whose A clips to skip")
    ap.add_argument("--cache", default=str(OUT / "cache"), help="where scan results, face crops and labels are kept")
    ap.add_argument("--workers", type=int, default=8, help="scan and write processes")
    ap.add_argument("--chunk", type=int, default=50, help="images per Haiku call")
    ap.add_argument("--parallel", type=int, default=4, help="Haiku calls at a time")
    ap.add_argument("--persons", type=int, default=0, help="scan only this many people, picked at random (a trial)")
    args = ap.parse_args()

    cache, batch = Path(args.cache).resolve(), OUT / args.batch
    db = open_cache(cache)
    if not db.execute("SELECT 1 FROM meta WHERE key=?", (f"batch:{args.batch}",)).fetchone():
        if batch.exists() and any(batch.iterdir()):
            sys.exit(f"{batch} has files this cache didn't write; pick another --batch")
        with db:
            db.execute("INSERT INTO meta VALUES (?, ?)", (f"batch:{args.batch}", str(batch)))
    batch.mkdir(exist_ok=True)

    clips, pids, m = load(DATA / "clipfarmer.sqlite")
    by_id = {c["id"]: c for c in clips}
    print(f"{len(clips)} clips pass the label filters, {len(pids)} people", flush=True)
    scan_errors = scan(db, clips, pids, m, cache, args.workers, args.persons, args.seed)

    scanned = {p for (p,) in db.execute("SELECT person FROM ident")}
    a_ids = [c for (c,) in db.execute("SELECT clip FROM scan WHERE status='ok'") if c in by_id]
    b_ids = [c for (c,) in db.execute("SELECT clip FROM ident WHERE clip IS NOT NULL") if c in by_id]
    keys = sorted(set(a_ids) | set(b_ids))
    complete = label(db, keys, cache, args.chunk, args.parallel)
    labels = {c: dict(zip(FIELDS + ("face",), v, strict=True)) for c, *v in db.execute(
        "SELECT clip, gender, skin, hair, length, age, face FROM labels WHERE gender IS NOT NULL")}

    skip = {r["a_clip"] for f in args.exclude for r in csv.DictReader(open(OUT / f))}
    a_pool = [a for a in a_ids if Path(by_id[a]["path"]).name not in skip]
    rejected = match(db, args.batch, by_id, pids, m, labels, a_pool, args.seed, args.n)

    scan_row = {c: dict(t=t, box=box, crop_px=px, sim=sim) for c, t, box, px, sim in
                db.execute("SELECT clip, t, box, crop_px, sim FROM scan WHERE status='ok'")}
    ident = {p: dict(clip=c, t=t, box=box, crop_px=px, sim=sim) for p, c, t, box, px, sim in
             db.execute("SELECT person, clip, t, box, crop_px, sim FROM ident WHERE clip IS NOT NULL")}
    pairs = list(db.execute("SELECT n, a_clip, b_person, b_clip, sim FROM pairs WHERE batch=? ORDER BY n",
                            (args.batch,)))
    rows, jobs = [], []
    for n, a, bp, bc, sim in pairs:
        ca, cb, fa, fb, sa, ib = by_id[a], by_id[bc], labels[a], labels[bc], scan_row[a], ident[bp]
        key = f"{n:04d}"
        r = {"pair": key, "a_clip": Path(ca["path"]).name, "a_person": ca["person"], "b_person": bp,
             "person_sim": round(sim, 3), "b_clip": Path(cb["path"]).name, "b_frame_s": ib["t"],
             "b_crop_native_px": ib["crop_px"], "b_face_vs_b_person_sim": round(ib["sim"], 3),
             "a_hands": round(ca["hands"], 3), **{f"a_{k}": fa[k] for k in FIELDS},
             **{f"b_{k}": fb[k] for k in FIELDS}, "face": fb["face"], "prompt": f"head_swap:\nFACE: {fb['face']}"}
        rows.append(r)
        if not (batch / key).exists():
            jobs.append(dict(out=str(batch / key), row=r,
                             a=dict(path=ca["path"], t=sa["t"], face=str(cache / "faces" / f"{a}.jpg")),
                             b=dict(path=cb["path"], t=ib["t"], box=ib["box"],
                                    face=str(cache / "faces" / f"{bc}.jpg"))))
    write_errors = 0
    for d in batch.glob("*.part"):  # left by a run that stopped mid-write
        shutil.rmtree(d)
    if jobs:
        print(f"write: {len(jobs)} pair folders", flush=True)
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            for k, err in enumerate(ex.map(write_pair, jobs, chunksize=8), 1):
                if err:
                    write_errors += 1
                    print(f"write: failed, retried next run: {err}", flush=True)
                if k % 250 == 0:
                    print(f"write: {k}/{len(jobs)}", flush=True)

    name = {c["id"]: Path(c["path"]).name for c in clips}
    for c, p, status, text_at in db.execute("SELECT clip, person, status, text_at FROM scan WHERE status != 'ok'"):
        why = {"text": f"text at {text_at}s", "short": "clip too short", "noface": "no face"}[status]
        rejected.append(dict(clip=c, person=p, role="A", reason=why))
    for c, p in db.execute("SELECT clip, person FROM b_rejects"):
        rejected.append(dict(clip=c, person=p, role="B", reason="text in face crop"))
    for p, c in db.execute("SELECT person, clip FROM ident"):
        if c is None:
            rejected.append(dict(clip=None, person=p, role="B", reason="no face image without hands or text"))
        elif why := unusable(labels.get(c)):
            rejected.append(dict(clip=c, person=p, role="B", reason=why))
    for r in rejected:
        r["person"] = r.get("person", by_id[r["clip"]]["person"] if r["clip"] in by_id else "")
        r["clip"] = name.get(r["clip"], "")
    with open(batch / "rejected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "person", "role", "reason"])
        w.writeheader()
        w.writerows(rejected)
    if rows:
        with open(batch / "pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "prompt"])
            w.writeheader()
            w.writerows({k: v for k, v in r.items() if k != "prompt"} for r in rows)
    calls = db.execute("SELECT count(*), coalesce(sum(s), 0), coalesce(sum(cost), 0) FROM calls").fetchone()
    write_index(batch, cache, rows, (calls[0], len(labels), calls[1], calls[2]), rejected)
    done = sum((batch / r["pair"]).exists() for r in rows)
    print(f"{done} pair folders in {batch.name} ({done * 6 / 3600:.2f} h of guide clips), {len(scanned)} people "
          f"scanned, {len(labels)} faces labelled; scan errors {scan_errors}, write errors {write_errors}, labels "
          f"{'complete' if complete else 'incomplete (run again)'}; rejected: "
          + ", ".join(f"{k} {v}" for k, v in Counter(reason_kind(r) for r in rejected).most_common()), flush=True)


if __name__ == "__main__":
    main()
