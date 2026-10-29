# InfiniteTalk video-to-video dubbing, the README's 4-step lightx2v path. Same install as infinitetalk.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infinitetalk"))
import batch as m

m.V2V = m.FAST = True

if __name__ == "__main__":
    m.run(m.load, m.make, m.settings())
