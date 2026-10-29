# Per-clip scores: P(fake) from every probe head for every clip of a CSV, using the same encoder, val transforms and
# segment sampling as eval.py. One process per GPU, each takes every Nth clip; a clip that fails to load is an error
# (the repo's dataset would silently swap in a random other clip).
# Usage: score.py CONFIG CKPT CSV OUT.csv [num_gpus]
import os, sys
sys.path.insert(0, "/data/vjepa2")
import numpy as np, pandas as pd, torch, torch.multiprocessing as mp, torch.nn.functional as F, yaml


class One(torch.utils.data.Dataset):
    def __init__(self, ds, idx): self.ds, self.idx = ds, idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]; r = self.ds.get_item_video(j)
        if r is None: raise RuntimeError(f"failed to load {self.ds.samples[j]}")
        return (*r, j)


def work(rank, ngpu, cfg_path, ckpt_path, csv_path, out):
    torch.cuda.set_device(rank)
    import evals.video_classification_frozen.eval as E
    from evals.video_classification_frozen.models import init_module
    from src.models.attentive_pooler import AttentiveClassifier
    cfg = yaml.safe_load(open(cfg_path)); exp = cfg["experiment"]; d = exp["data"]; mk = cfg["model_kwargs"]
    dev = torch.device(f"cuda:{rank}")
    enc = init_module(module_name=mk["module_name"], frames_per_clip=d["frames_per_clip"], resolution=d["resolution"],
                      checkpoint=mk["checkpoint"], model_kwargs=mk["pretrain_kwargs"], wrapper_kwargs=mk["wrapper_kwargs"], device=dev)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["classifiers"]
    heads = []
    for s in sd:
        c = AttentiveClassifier(embed_dim=enc.embed_dim, num_heads=exp["classifier"]["num_heads"], depth=exp["classifier"]["num_probe_blocks"],
                                num_classes=d["num_classes"], use_activation_checkpointing=False).to(dev)
        c.load_state_dict({k.removeprefix("module."): v for k, v in s.items()}); c.eval(); heads.append(c)
    loader, _ = E.make_dataloader(root_path=[csv_path], batch_size=1, world_size=1, rank=0, img_size=d["resolution"],
                                  frames_per_clip=d["frames_per_clip"], frame_step=d.get("frame_step"), eval_duration=d.get("clip_duration"),
                                  num_segments=d["num_segments"], num_views_per_segment=d["num_views_per_segment"],
                                  allow_segment_overlap=True, training=False, num_workers=0, normalization=d.get("normalization"))
    ds = loader.dataset
    idx = list(range(rank, len(ds), ngpu))
    dl = torch.utils.data.DataLoader(One(ds, idx), batch_size=16, num_workers=12, pin_memory=True)
    rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=exp["optimization"].get("use_bfloat16", True)):
        for k, (buf, lab, cidx, j) in enumerate(dl):
            clips = [[v.to(dev, non_blocking=True) for v in seg] for seg in buf]
            outs = enc(clips, [c.to(dev) for c in cidx])
            p = torch.stack([sum(F.softmax(h(o).float(), dim=1) for o in outs)[:, 1] / len(outs) for h in heads], 1).cpu().numpy()
            for jj, ll, pp in zip(j.tolist(), lab.tolist(), p):
                rows.append({"path": ds.samples[jj], "label": int(ll), **{f"p{h:02d}": float(x) for h, x in enumerate(pp)}})
            if rank == 0 and k % 20 == 0: print(f"rank0 batch {k}/{len(dl)}", flush=True)
    pd.DataFrame(rows).to_csv(f"{out}.part{rank}", index=False)


if __name__ == "__main__":
    cfg, ckpt, csv, out = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else torch.cuda.device_count()
    mp.spawn(work, args=(n, cfg, ckpt, csv, out), nprocs=n)
    df = pd.concat([pd.read_csv(f"{out}.part{r}") for r in range(n)]).sort_values("path")
    expect = sum(1 for _ in open(csv))
    assert len(df) == expect and df.path.is_unique, f"scored {len(df)} clips, CSV has {expect}"
    df.to_csv(out, index=False)
    for r in range(n): os.remove(f"{out}.part{r}")
    print(f"wrote {out}: {len(df)} clips x {sum(c.startswith('p') and c[1:].isdigit() for c in df.columns)} heads")
