# FaceFusion 3.9.2, deep swapper druuzil/cillian_murphy_320 (DeepFaceLab model) on A's clip, A's audio. Settings in ../facefusion/ff.py.
# Held out of training: used only to test the probe on generators it has not seen.
import sys; from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent.parent / "facefusion")]
from pairio import run
import ff

M = "dfl_m"
SETTINGS = {"facefusion": "3.9.2", "args": ff.COMMON + ff.METHODS[M], "seconds": ff.SECONDS,
            "ref": "none (fixed identity)"}

if __name__ == "__main__":
    run(lambda: ff.load(M), ff.make, SETTINGS)
