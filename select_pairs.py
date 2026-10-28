"""Pick the pairs most likely to make good deepfakes with every generator: the shared set all generators run on.

    uv run --with webrtcvad-wheels select_pairs.py [--n 500]

Candidates are the pairs4 pairs the B300 VM has (all but the ones already in nebius_out/full_v3, which the old VM
made). Per pair it uses Clipfarmer's stored measurements of A's and B's clips (face size, yaw/pitch, sharpness,
jitter, bitrate, lighting/glasses/hands labels), job.json (B's crop resolution, A/B age and hair) and a voice
activity check of A's and B's audio (talking heads speak A's audio, lip sync speaks B's).

Hard filters drop pairs a generator is likely to fail on; the rest are ranked by a quality score and picked
greedily with every person at most once as A and once as B, with gender x skin groups kept at their share of
the candidates. Writes select<N>.txt (pair names) and select<N>.csv (every candidate, its numbers, why it was dropped).
"""
import argparse
import csv
import json
import math
import sqlite3
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "pairs4"
DONE = HERE / "nebius_out/full_v3"
CLIPFARMER_DB = Path.home() / "Projects/Code/clipfarmer-data/clipfarmer.sqlite"

# hard filters
MIN_SPEECH = 0.5        # share of 30 ms frames with voice, in A's clip and in B's clip
MAX_A_YAW, MAX_A_PITCH = 25, 20
MIN_B_CROP = 200        # px of B's face in its source frame
MIN_B_ID = 0.8          # B's reference frame vs B's identity (ArcFace)
MAX_AGE_GAP = 15
MAX_P = 0.5             # label probability above which hands near face / sunglasses / headwear count as present

# score weights (each feature is z-scored over the candidates first)
WEIGHTS = {"a_sharp": 1.0, "a_bpp": 0.5, "a_size": 1.0, "a_yaw": -1.0, "a_pitch": -0.5, "a_jitter": -0.5,
           "a_lit": 0.5, "a_glasses": -0.5, "b_crop": 1.0, "b_id": 0.5, "b_yaw": -0.25, "b_glasses": -0.5,
           "age_gap": -0.5, "same_length": 0.5, "person_sim": 0.25, "speech": 0.5}


def speech_share(path):
    """Share of 30 ms frames webrtcvad (mode 3) calls speech, over the clip's audio at 16 kHz mono."""
    import webrtcvad
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    vad, n = webrtcvad.Vad(3), 960  # 30 ms of 16-bit samples
    frames = [pcm[i:i + n] for i in range(0, len(pcm) - n + 1, n)]
    return sum(vad.is_speech(f, 16000) for f in frames) / len(frames) if frames else 0.0


def speech_pair(d):
    return speech_share(d / "a_guide.mp4"), speech_share(d / "b_source.mp4")


def clip_info(db):
    """{clip file name: (results, labels)} from Clipfarmer's accepted clips."""
    out = {}
    for path, res, lab in db.execute("select path, results_json, labels_json from clips where fate = 'accepted'"):
        out[Path(path).name] = (json.loads(res or "{}"), json.loads(lab or "{}"))
    return out


def feats(job, a, b, speech):
    (ar, al), (br, bl) = a, b
    aw, bw = ar.get("window", {}), br.get("window", {})
    p = lambda lab, q, k: lab.get(q, {}).get(k, 0.0)
    return {
        "a_sharp": aw.get("sharp", 0.0), "a_bpp": math.log(max(ar.get("bpp", 1e-4), 1e-4)),
        "a_size": aw.get("size", 0.0), "a_yaw": aw.get("yaw", 90.0), "a_pitch": abs(aw.get("pitch", 90.0)),
        "a_jitter": aw.get("jitter", 1.0),
        "a_lit": 1 - p(al, "brightness", "dim") - p(al, "light_type", "colored") - p(al, "light_type", "backlit"),
        "a_glasses": p(al, "glasses", "clear"), "a_sunglasses": p(al, "glasses", "sunglasses"),
        "a_hands": max(p(al, "hands_near_face", "yes"), job.get("a_hands", 0.0)),
        "a_headwear": p(al, "headwear", "yes"),
        "b_crop": job["b_crop_native_px"], "b_id": job["b_face_vs_b_person_sim"], "b_yaw": bw.get("yaw", 90.0),
        "b_glasses": p(bl, "glasses", "clear"), "b_sunglasses": p(bl, "glasses", "sunglasses"),
        "b_hands": p(bl, "hands_near_face", "yes"),
        "age_gap": abs(job["a_age"] - job["b_age"]), "same_length": float(job["a_length"] == job["b_length"]),
        "person_sim": job["person_sim"], "a_speech": speech[0], "b_speech": speech[1],
        "speech": min(speech),
    }


def dropped(f):
    """Why a pair fails the hard filters ('' if it passes)."""
    checks = [(f["a_speech"] < MIN_SPEECH, "A has little speech"), (f["b_speech"] < MIN_SPEECH, "B has little speech"),
              (f["a_yaw"] > MAX_A_YAW, "A turned away"), (f["a_pitch"] > MAX_A_PITCH, "A tilted"),
              (f["a_hands"] > MAX_P, "hands near A's face"), (f["a_sunglasses"] > MAX_P, "A in sunglasses"),
              (f["a_headwear"] > MAX_P, "A wears headwear"), (f["b_crop"] < MIN_B_CROP, "B's face too small"),
              (f["b_id"] < MIN_B_ID, "B's frame unlike B"),
              (f["b_sunglasses"] > MAX_P, "B in sunglasses"), (f["b_hands"] > MAX_P, "hands near B's face"),
              (f["age_gap"] > MAX_AGE_GAP, "age gap")]
    return "; ".join(why for bad, why in checks if bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    done = {p.stem for p in DONE.glob("*.mp4") if p.stem.isdigit()}
    dirs = sorted(d for d in PAIRS.iterdir() if (d / "job.json").exists() and d.name not in done)
    db = sqlite3.connect(f"file:{CLIPFARMER_DB}?mode=ro", uri=True)
    info = clip_info(db)
    print(f"{len(dirs)} candidate pairs (on the VM); checking speech in A's and B's audio")
    with ProcessPoolExecutor() as ex:
        speech = dict(zip(dirs, ex.map(speech_pair, dirs, chunksize=16)))

    rows = []
    for d in dirs:
        job = json.loads((d / "job.json").read_text())
        if job["a_clip"] not in info or job["b_clip"] not in info:
            rows.append({"pair": d.name, "dropped": "clip not in the Clipfarmer snapshot"})
            continue
        f = feats(job, info[job["a_clip"]], info[job["b_clip"]], speech[d])
        rows.append({"pair": d.name, "a_person": job["a_person"], "b_person": job["b_person"],
                     "group": f"{job['a_gender']}/{job['a_skin']}", "dropped": dropped(f), **f})

    ok = [r for r in rows if not r["dropped"]]
    for k in WEIGHTS:
        vals = [r[k] for r in ok]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        for r in ok:
            r["score"] = r.get("score", 0.0) + WEIGHTS[k] * (r[k] - mean) / sd
    ok.sort(key=lambda r: -r["score"])

    # greedy pick by score: each person at most once as A and once as B, each gender x skin group up to its share;
    # then without the group shares; then (if still short) a person may appear twice, with a different clip
    groups = {}
    for r in ok:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    quota = {g: math.floor(a.n * c / len(ok)) for g, c in groups.items()}
    as_a, as_b, picked = {}, {}, []
    for capped, limit in ((True, 1), (False, 1), (False, 2)):
        taken = {g: sum(r["group"] == g for r in picked) for g in groups}
        for r in ok:
            if len(picked) >= a.n:
                break
            if r.get("picked") or as_a.get(r["a_person"], 0) >= limit or as_b.get(r["b_person"], 0) >= limit:
                continue
            if capped and taken[r["group"]] >= quota[r["group"]]:
                continue
            r["picked"] = 1
            picked.append(r)
            as_a[r["a_person"]] = as_a.get(r["a_person"], 0) + 1
            as_b[r["b_person"]] = as_b.get(r["b_person"], 0) + 1
            taken[r["group"]] += 1

    out = HERE / f"select{a.n}.txt"
    out.write_text("".join(sorted(r["pair"] + "\n" for r in picked)))
    keys = ["pair", "picked", "score", "dropped", "group", "a_person", "b_person"] + \
        [k for k in rows[0] if k not in ("pair", "dropped", "group", "a_person", "b_person")]
    with open(HERE / f"select{a.n}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (-r.get("picked", 0), -r.get("score", -99))):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    reasons = {}
    for r in rows:
        for why in filter(None, r["dropped"].split("; ")):
            reasons[why] = reasons.get(why, 0) + 1
    print(f"{len(ok)} pass the hard filters; dropped for: " +
          ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])))
    print(f"picked {len(picked)}: " +
          ", ".join(f"{g} {sum(r['group'] == g for r in picked)}" for g in sorted(groups)))
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
