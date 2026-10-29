# Wan2.2-Animate-2, the authors' distilled model (10 steps, guide 1). Same install as wan-animate-2.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wan-animate-2"))
import batch as m

m.FAST = True

if __name__ == "__main__":
    m.run(m.load, m.make, m.settings())
