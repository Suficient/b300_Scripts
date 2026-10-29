# FaceFusion 3.9.2, EDTalk 256 lip-sync: A's clip mouthing B's audio (pairio's ref_audio), B's audio. Settings in ../facefusion/ff.py.
# Held out of training: used only to test the probe on generators it has not seen.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent.parent / "facefusion")]
from pairio import run
import ff

M = "edtalk"
SETTINGS = {"facefusion": "3.9.2", "args": ff.COMMON + ff.METHODS[M], "seconds": ff.SECONDS,
            "ref": "b_source.mp4 audio"}

if __name__ == "__main__":
    run(lambda: ff.load(M), ff.make, SETTINGS)
