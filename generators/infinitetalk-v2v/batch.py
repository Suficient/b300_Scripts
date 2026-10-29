# InfiniteTalk video-to-video dubbing: A's clip (25 fps) re-generated speaking B's audio; output audio = B's.
# Same code and settings as infinitetalk/batch.py.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infinitetalk"))
import batch as it

it.V2V = True

if __name__ == "__main__":
    it.run(it.load, it.make, it.settings())
