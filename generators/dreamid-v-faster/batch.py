# DreamID-V Faster: the authors' faster model (no pose input, 16 steps). Same install as dreamid-v.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dreamid-v"))
import batch as m

m.FAST = True

if __name__ == "__main__":
    m.run(m.load, m.make, m.settings())
