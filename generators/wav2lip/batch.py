# Wav2Lip GAN: A's clip (25 fps, Wav2Lip's training rate) mouthing B's audio; output audio = B's. inference.py's
# pipeline (S3FD boxes padded 0 10 0 0, smoothed over 5 frames; 96 px mouth pasted back), model loaded once and
# frames encoded straight to libx264 instead of its DIVX .avi.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), "."]
import subprocess
import numpy as np, torch, cv2
from pairio import run, ffmpeg, probe
import audio, face_detection
from models import Wav2Lip

SECONDS, FPS, PADS, SIZE = 5.2, 25, (0, 10, 0, 0), 96
SETTINGS = {"checkpoint": "wav2lip_gan.pth", "fps": FPS, "seconds": SECONDS, "pads": PADS, "smooth": 5,
            "audio": "B (b_source.mp4)", "crf": 17}


def load():
    net = Wav2Lip()
    sd = torch.load("checkpoints/wav2lip_gan.pth", map_location="cpu")["state_dict"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    det = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device="cuda")
    return net.cuda().eval(), det


def make(model, p):
    net, det = model
    wav, out = p.tmp / "b.wav", p.tmp / "out.mp4"
    ffmpeg("-i", p.ref_video, "-t", str(SECONDS + 0.2), "-vn", "-ac", "1", "-ar", "16000", wav)
    v = next(s for s in probe(p.video)["streams"] if s["codec_type"] == "video")
    w, h = v["width"], v["height"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p.video), "-t", str(SECONDS), "-r", str(FPS),
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], check=True, capture_output=True).stdout
    frames = list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy())

    mel = audio.melspectrogram(audio.load_wav(str(wav), 16000))
    chunks, i = [], 0
    while True:
        s = int(i * 80. / FPS)
        if s + 16 > mel.shape[1]:
            chunks.append(mel[:, -16:])
            break
        chunks.append(mel[:, s:s + 16]); i += 1
    frames = frames[:len(chunks)]

    rects = []
    for i in range(0, len(frames), 16):
        rects += det.get_detections_for_batch(np.array(frames[i:i + 16]))
    if any(r is None for r in rects):
        raise ValueError("face not detected in some frame")
    boxes = np.array([[max(0, r[0] - PADS[2]), max(0, r[1] - PADS[0]), min(w, r[2] + PADS[3]), min(h, r[3] + PADS[1])]
                      for r in rects])
    boxes = np.array([boxes[min(i, len(boxes) - 5):][:5].mean(0) for i in range(len(boxes))]).astype(int)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(out)],
                           stdin=subprocess.PIPE)
    for b in range(0, len(chunks), 128):
        idx = [i % len(frames) for i in range(b, min(b + 128, len(chunks)))]
        faces = np.array([cv2.resize(frames[i][y1:y2, x1:x2], (SIZE, SIZE))
                          for i, (x1, y1, x2, y2) in ((i, boxes[i]) for i in idx)])
        masked = faces.copy(); masked[:, SIZE // 2:] = 0
        img = torch.from_numpy(np.concatenate((masked, faces), 3) / 255.).float().permute(0, 3, 1, 2).cuda()
        mels = torch.from_numpy(np.array(chunks[b:b + len(idx)])[:, None]).float().cuda()
        with torch.no_grad():
            pred = (net(mels, img).cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        for k, i in enumerate(idx):
            f = frames[i].copy(); x1, y1, x2, y2 = boxes[i]
            f[y1:y2, x1:x2] = cv2.resize(pred[k], (x2 - x1, y2 - y1))
            enc.stdin.write(f.tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("ffmpeg encode failed")
    return out, wav


if __name__ == "__main__":
    run(load, make, SETTINGS)
