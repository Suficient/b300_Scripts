# BFS LTX-2.5 precompute test

Face-swap jobs built from the clipfarmer-data snapshot, in four batches:

- **Batch 1:** 20 pairs in `pairs/`. Its `index.html`, `pairs.csv` and caption files are at the top level.
- **Batch 2:** 100 pairs in `pairs2/`, with its own `index.html`, `pairs.csv`, caption files, and `review.csv` (my check of every pair).
- **Batch 3:** 50 pairs in `pairs3/`, built with filters for glasses, hands, headwear, text and look-alike matching (see Batch 3 below).
- **Batch 4:** `pairs4/`, every clip in the snapshot that can be a guide and isn't in batches 1–3 (see Batch 4 below).

## Files

| Path | What |
|---|---|
| `index.html` | All 20 pairs: A's face and clip, B's face and source clip, ArcFace similarity, prompt |
| `pairs.csv` | Same data as a table |
| `pairs/NN/a_guide.mp4` | A's clip: the guide video |
| `pairs/NN/a_frame.jpg`, `a_face.jpg` | A's sharpest sample frame and its face crop (review only) |
| `pairs/NN/b_face_1024.png` | B's face: the identity image for the BFS workflow |
| `pairs/NN/b_face_512.jpg` | The same crop at 512 px, which is what Haiku captioned |
| `pairs/NN/b_frame.jpg`, `b_source.mp4` | The frame and clip B's face was cut from |
| `pairs/NN/caption.txt` | The positive prompt: `head_swap:` / `FACE: …` |
| `pairs/NN/job.json` | Everything above for one pair |
| `caption_input/` | The numbered images Haiku read |
| `captions_raw.json` | Haiku's prompt, output and token usage |
| `captions_raw_run1_unlabeled.json` | First run, before the numbers were printed on the images (see findings) |
| `pairs2/NNN/` | Batch 2 pairs, same files as `pairs/NN/` |
| `pairs2/index.html`, `pairs.csv`, `caption_input/`, `captions_raw.json` | Batch 2's versions of the files above |
| `pairs2/review.csv` | My check of each batch 2 pair: caption errors, things asserted but not visible, image problems, mismatched pairs |
| `selected/` | Copies of the 75 batch 2 pairs that passed review (no mismatch, usable B image, no clear caption error), with their own index, CSVs and README |
| `pairs3/` | Batch 3: pair folders as in `pairs2/`, plus `rejected.csv` (every clip turned down and why) and `rejected/` (the text found, outlined in red) |
| `pairs4/NNNN/` | Batch 4 pairs, same files as `pairs2/NNN/` |
| `pairs4/index.html`, `pairs.csv`, `rejected.csv` | Batch 4's review page and tables |
| `pairs4.log` | Batch 4's build log |
| `cache/` | What `build.py` keeps between runs: `cache.sqlite` (scan results, Haiku labels and calls, pairs), `faces/` (the 512 px face crops Haiku reads), `rejected/` (the text found, outlined in red) |
| `models/text_detection_en_ppocrv3_2023may.onnx` | PaddleOCR's PP-OCRv3 text detector from the [OpenCV model zoo](https://huggingface.co/opencv/text_detection_ppocr) (Apache-2.0) |
| `build.py` | Builds a batch |

## How batches 1 and 2 were made

- **A:** 20 random accepted clips (seed 42), no two from the same person or video.
- **B:** the person whose mean ArcFace embedding is closest to A's clip embedding, skipping A's own person and anyone at 0.45 or above (probably the same person).
- **B's clip:** the one with the best Clipfarmer window score.
- **B's face:** the sharpest of the three frames Clipfarmer samples (0.5 s, 3.0 s, 5.5 s). The crop is a square of 2× the face box: 467–1080 px natively, resized to 1024.
- **Prompt:** the BFS v1.1 template, numbered one line per image. It also tells Haiku to leave out traits that aren't clearly visible. One `claude -p --model haiku` call covered all 20: 20 s, $0.045 at API list price, paid from the Claude subscription.

## Findings

1. **Captions can land on the wrong image.** In the first run the files were named `01.jpg`–`20.jpg` with no number printed on them. Haiku gave 7 of the 20 captions (13–19) the description of the next image. In this run each image has its number printed in a corner, and all 20 captions match their image.
2. **Remaining caption errors:**
   - 13 gives "short dark hair" for a man with grey hair.
   - Every caption states an eye colour, including the masked face in 15, so the "leave out what isn't clearly visible" instruction isn't followed for eyes.
3. **Bad identity images:**
   - 15: B wears a face mask.
   - 02: B's frame has burned-in subtitles and a watermark.
   - These two have the lowest similarity between B's face image and B's person embedding (0.80 and 0.79; the rest are 0.85–0.98). In batch 2 that score caught only 2 of 7 bad images (see below).
4. **Nearest by ArcFace isn't the same gender, skin tone or age:**
   - 06: a woman's clip is paired with a man.
   - 17: a darker-skinned woman with braids is paired with a lighter-skinned woman with auburn hair.
   - 10: a bald older man is paired with a young man with curly hair.
   - A↔B similarity is 0.17–0.33 for all 20 pairs.
   - Whether that matters for the dataset is a design call. If it does, filter on attributes the database doesn't store yet (gender, and hair from a caption of A).

## Batch 2: 100 pairs

Built the same way, with three differences:
- Seed 7.
- None of batch 1's A people.
- Haiku captioned 50 faces per call, with 3-digit numbers printed on the images.

Extraction took 187 s on this PC. Captioning was 2 calls, 69 s, $0.19 at API list price ($0.0019 per face). I checked every caption against its face and every A/B pairing by eye; `review.csv` has the per-pair calls. These are my judgements, not ground truth.

| Check | Result |
|---|---|
| Captions on the right image | 99 of 99 distinct B faces (one B was used twice) |
| Captions with a clear error | 7: hair colour (18, 19, 76), skin tone (45), age (76, 90), and "expression", which the template forbids (70, 72). Another 7 are borderline. |
| Gender in caption | 99 of 99 correct |
| Hair described under a hat, cap, hood or veil | 9 of the 10 covered heads |
| Eye colour stated when the eyes aren't visible | 11 (sunglasses, or eyes closed or looking down). Every caption states one. |
| Format | 11–15 words each; no "no beard"-style absent features; never mentions headwear |
| Glasses vs Clipfarmer's labels | Caption mentions glasses for 19 of the 28 B clips Clipfarmer labels as glasses, and never for one it labels as none |
| B face image unusable | 7 distinct images (8 pairs): sunglasses (28, 85), veil and hand (57), blurry extreme close-up (66), film still with costume hood and title text (70/77), text overlays (12, 72) |
| B face image with a minor problem | 32: headwear, headphones, small overlays, hands near the face, low resolution |
| B face upscaled to 1024 | 72; 11 are below 512 px natively |
| Pair mismatched | 16: gender 8, skin tone 6, age gap of 20+ years 3 |
| Pair usable (no mismatch, usable B image) | 78; 75 also have no clear caption error |
| To check | 71: B looks young; confirm they're an adult |

How well the available signals catch the 7 unusable images:
- **Similarity of B's image to B's person embedding:** catches 57 and 66 (0.61 and 0.46). Below 0.82 it also flags 10 images that are fine.
- **Clipfarmer's existing labels for B's clip** (glasses = sunglasses, or headwear): catch 28, 57, 70, 72 and 85.
- **Both together:** 6 of 7 (all but 12, the app overlay). The headwear label also flags 19 clips with only a cap or headphones; BFS would carry a cap over onto A.

## Batch 3: 50 pairs

Seed 3, none of batch 1's or batch 2's A people. Differences from batch 2:
- **Label filters:** only clips Clipfarmer labels as no glasses, no hands near the face and nothing on the head, for A and B. 12,993 of the 22,421 clips pass.
- **Text:** PaddleOCR's PP-OCRv3 detector (`models/`) checks 6 frames of A (one per second, the whole frame) and B's face crop. It ignores detections centred inside a face box, where it fires on eyes, teeth and brows. Clipfarmer's overlay gate already covers text on the face.
- **Matching:** Haiku labels each face's gender, skin tone (6 steps), hair (straight, wavy, curly, coily, braids/locs, bald, covered), hair length and age, in the same call that writes B's caption.
  - A pair needs the same gender, skin tone within 1 step, similar hair and an age gap of 20 years or less.
  - Both faces need an uncovered head and an age of 21 or more.
  - Each A tries its 10 nearest people by ArcFace, 2 per Haiku round.

| Check | Result |
|---|---|
| Build time | About 12 min on this PC |
| A clips checked | 229: 176 had text, 3 found no match, 50 were paired |
| B face crops with text | 103, each rejected; the person's next clip was tried |
| Haiku | 209 faces, 9 calls, 177 s, $0.065 at API list price |
| B candidates tried per pair | 1 for 28 pairs, 2–5 for 21, 10 for 1 |
| Distinct B people | 47 (3 used twice) |

Problems I found looking at every pair (my judgement, by eye):
- **B image unusable or doubtful:**
  - 06: a black-and-white photo
  - 17: a phone in front of the chin
  - 30: "URSD" in large outlined letters across the top, which the detector missed
  - 35: hands holding a compact and a brush at the face
  - 43: script text on the shirt, also missed
- **A clip:**
  - 08: a heart sticker overlay; graphics without letters aren't detected
  - 46: eating on camera
- **Pair 45:** B's skin looks darker than A's (Haiku says tan vs medium). Borderline.
- **Skin tone skew:** 44 of the 50 A people are "light". The 3 A clips dropped for no match were brown or dark skinned. Their 10 nearest people by ArcFace were mostly lighter skinned or of the other gender.

## Batch 4

Built by the current `build.py` (seed 4), from every clip that can be a guide, skipping the A clips of batches 1–3.
Differences from batch 3:
- **Several clips per person:** every clip that passes the filters is an A once. Each person can be A many times.
- **Hands near the face are allowed on A.** B's face image still comes from a clip without them: in a trial, a person
  whose only clip had a hand over the face got that frame as their identity image.
- **B is picked from everyone who fits**, not from A's 10 nearest people by ArcFace. Of the people whose face image
  matches A's gender, skin tone, hair and age, the closest by ArcFace wins. A person not yet paired with A's person,
  and used fewer than 10 times, is preferred.
- **Each face is labelled once.** Haiku labels every face crop that can be used (A clips and each person's B image)
  before matching starts, and the labels are kept in `cache/`.

| Check | Result |
|---|---|
| Clips scanned | 14,376 (no glasses, nothing on the head), from 1,872 people: 56 min on this PC |
| Can be A (no text in 6 frames) | 3,298 clips (23%), from 576 people |
| B face images | 1,543 people. 329 have no clip without hands whose face crop is free of text; 1,387 crops were passed over for text |
| Haiku | 4,442 faces, 90 calls, none failed, 11 min, $0.60 at API list price |
| Pairs | 3,228 (about 5.4 h of guide clips). Of the other A clips, 66 were in batches 1–3 and 4 look under 21; every other A found a B |
| A people | 573. Skin: light 2,597, tan 207, brown 142, dark 115, medium 109, very light 58 |
| B people | 1,278; none used more than 10 times, 379 used once. 3,222 of the 3,228 pairs have an A person / B person combination of their own |
| Disk | 18 GB in `pairs4/`, 1 GB in `cache/` |

I looked at 16 random pairs (12 of any skin tone, 4 brown or dark): all match on gender, skin tone, hair and age. In
1935, B holds something white near the chin.

## Build a batch

```
~/Projects/Code/Clipfarmer/.venv/bin/python build.py --batch pairs4 --exclude pairs.csv pairs2/pairs.csv pairs3/pairs.csv
```

Reads `~/Projects/Code/clipfarmer-data` and Clipfarmer's face models. Runs in three stages, each saved to
`cache/cache.sqlite` as it goes:
1. **Scan** (8 processes): every clip is decoded once and checked for text, and each person gets one face image.
   About an hour for the whole snapshot on this PC.
2. **Label:** Haiku labels each face crop not labelled before, 50 per `claude -p` call, 4 calls at a time. The images
   go in the message, with no tools and a one-line system prompt. If calls keep failing (usage limit, outage), it
   waits with backoff for up to 6 hours, then stops.
3. **Match and write:** pairs A clips not yet in the batch and writes their folders, then the batch's `index.html`,
   `pairs.csv` and `rejected.csv`.

A run that stops for any reason picks up where it left off when started again with the same arguments. Pairs already
in a batch keep their B. Options:
- `--exclude` takes `pairs.csv` files whose A clips to skip.
- `--n` caps the batch's size.
- `--chunk` and `--parallel` set the images per Haiku call and the calls at a time.
- `--workers` sets the scan and write processes.
- `--persons` scans only that many people, for a trial; use it with its own `--cache`.

A cache refuses settings other than the ones it was made with (filters, frame times, text detector, prompt), and a
batch folder must be new or made by the same cache.
