"""SCRFD + ArcFace (insightface buffalo_l; the weights are non-commercial). Copied from fake-pipeline's faces.py."""
from __future__ import annotations

import logging
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)

BUFFALO_L = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


@dataclass
class Face:
    box: np.ndarray  # x, y, w, h in pixels
    lm: np.ndarray  # (5, 2): the two eyes, nose, the two mouth corners
    score: float


class Faces:
    def __init__(self, models_dir: Path, min_score: float):
        pack = _ensure_buffalo(models_dir)
        self.det = _onnx_session(pack / "det_10g.onnx")
        self.rec = _onnx_session(pack / "w600k_r50.onnx")
        log.info("face models on %s", self.det.get_providers()[0])
        self.det_in = self.det.get_inputs()[0].name
        self.rec_in = self.rec.get_inputs()[0].name
        self.min_score = min_score
        self._centers: dict = {}

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        size = min(640, math.ceil(max(h, w) / 32) * 32)
        scale = size / max(h, w)
        canvas = np.zeros((size, size, 3), np.uint8)
        nh, nw = round(h * scale), round(w * scale)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        return decode_scrfd(self.det.run(None, {self.det_in: blob}), size, scale, self.min_score, self._centers)

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        m, _ = cv2.estimateAffinePartial2D(face.lm, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        crop = cv2.warpAffine(img, m, (112, 112), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        f = self.rec.run(None, {self.rec_in: blob})[0].flatten()
        return f / (np.linalg.norm(f) + 1e-9)

    def scan(self, path: Path, n: int) -> tuple[np.ndarray, np.ndarray, Face] | None:
        """Largest face in n evenly spaced frames -> (mean L2-normalised embedding, best frame, its face).
        Best = largest x sharpest. None if no frame has a face."""
        embs, best, best_score = [], None, -1.0
        for img in sample_frames(path, n):
            faces = self.detect(img)
            if not faces:
                continue
            f = max(faces, key=lambda f: f.box[2] * f.box[3])
            embs.append(self.embed(img, f))
            score = float(f.box[3]) * sharpness(img, f.box)
            if score > best_score:
                best, best_score = (img, f), score
        if not embs:
            return None
        e = np.mean(embs, axis=0)
        return e / (np.linalg.norm(e) + 1e-9), *best


def sample_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    return frames


def face_crop(img: np.ndarray, box: np.ndarray, scale: float) -> np.ndarray:
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * scale / 2
    H, W = img.shape[:2]
    return img[int(max(cy - s, 0)):int(min(cy + s, H)), int(max(cx - s, 0)):int(min(cx + s, W))]


def sharpness(img: np.ndarray, box: np.ndarray, out_h: int = 112) -> float:
    """Laplacian variance on a face crop resized to a fixed height, so faces of different sizes compare."""
    crop = face_crop(img, box, 1.0)
    if not crop.size:
        return 0.0
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * out_h / crop.shape[0])), out_h))
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def decode_scrfd(outs: list, size: int, scale: float, min_score: float, centers: dict | None = None) -> list[Face]:
    """Turn SCRFD's raw outputs into faces. outs = 3 score maps, 3 box maps, 3 keypoint maps (strides 8, 16, 32,
    two anchors per location); boxes and keypoints are distances from each anchor centre, in stride units."""
    centers = {} if centers is None else centers
    boxes, kpss, scores = [], [], []
    for i, stride in enumerate((8, 16, 32)):
        sc = outs[i].reshape(-1)
        keep = np.flatnonzero(sc >= min_score)
        if not len(keep):
            continue
        n = size // stride
        if (n, stride) not in centers:
            c = np.stack(np.mgrid[:n, :n][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            centers[(n, stride)] = np.repeat(c, 2, axis=0)
        c = centers[(n, stride)][keep]
        d = outs[i + 3].reshape(-1, 4)[keep] * stride
        k = outs[i + 6].reshape(-1, 10)[keep] * stride
        boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], 1))
        kpss.append(np.stack([c[:, 0:1] + k[:, 0::2], c[:, 1:2] + k[:, 1::2]], -1))
        scores.append(sc[keep])
    if not boxes:
        return []
    boxes, kpss, scores = np.concatenate(boxes) / scale, np.concatenate(kpss) / scale, np.concatenate(scores)
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, scores.tolist(), min_score, 0.4)
    return [Face(np.array(xywh[i], np.float32), kpss[i].astype(np.float32), float(scores[i]))
            for i in np.array(idx).reshape(-1)]


def _onnx_session(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        ort.preload_dlls()  # CUDA/cuDNN come from pip wheels, not the system
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(path), opts, providers=providers)


def _ensure_buffalo(models_dir: Path) -> Path:
    path = models_dir / "buffalo_l"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", BUFFALO_L)
        r = requests.get(BUFFALO_L, timeout=600)
        r.raise_for_status()
        tmp = models_dir / "buffalo_l.zip"
        tmp.write_bytes(r.content)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(path)
        tmp.unlink()
    return path
