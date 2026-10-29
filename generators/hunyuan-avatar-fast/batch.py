# HunyuanVideo-Avatar with DeepCache, as upstream's single-GPU script runs it. Same install as hunyuan-avatar.
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hunyuan-avatar"))
import batch as m

m.FAST = True

if __name__ == "__main__":
    m.run(m.load, m.make, m.settings())
