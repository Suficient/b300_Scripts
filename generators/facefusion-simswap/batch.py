# FaceFusion 3.9.2, SimSwap 256: B's face (b_face_1024.png) onto A's clip, A's audio. Settings in ../facefusion/ff.py.
# Held out of training: used only to test the probe on swap models it has not seen.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent.parent / "facefusion")]
from pairio import run
import ff

M = "simswap"
SETTINGS = {"facefusion": "3.9.2", "args": ff.COMMON + ff.METHODS[M], "seconds": ff.SECONDS, "ref": "b_face_1024.png"}

if __name__ == "__main__":
    run(lambda: ff.load(M), ff.make, SETTINGS)
