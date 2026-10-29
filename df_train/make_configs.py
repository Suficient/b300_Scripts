# Training config (and a 20-step trial config) from the repo's V-JEPA 2.1 ViT-L 384 SSv2 probe recipe.
import copy, yaml
base = yaml.safe_load(open("/data/vjepa2/configs/eval_2_1/vitl-384/ssv2.yaml"))
c = copy.deepcopy(base)
c.update(folder="/data/df_train/runs", tag="df-vitl384-2x16-cropfirst", nodes=1, tasks_per_node=8, num_workers=22)
d = c["experiment"]["data"]
d.update(dataset_train="/data/df_train/csv/train.csv", dataset_val="/data/df_train/csv/val.csv", num_classes=2,
         frame_step=None, clip_duration=2.14, random_resize_scale=[0.5, 1.0], reprob=0.0)
c["experiment"]["optimization"].update(batch_size=32, num_epochs=20)
assert len(c["experiment"]["optimization"]["multihead_kwargs"]) == 20
c["model_kwargs"]["checkpoint"] = "/data/vjepa2_ckpt/vjepa2_1_vitl_dist_vitG_384.pt"
yaml.safe_dump(c, open("/data/df_train/configs/df_vitl384.yaml", "w"), sort_keys=False)
s = copy.deepcopy(c)
s.update(tag="df-vitl384-2x16-smoke", resume_checkpoint=False)
s["experiment"]["data"].update(dataset_train="/data/df_train/csv/smoke_train.csv", dataset_val="/data/df_train/csv/smoke_val.csv")
s["experiment"]["optimization"]["num_epochs"] = 1
yaml.safe_dump(s, open("/data/df_train/configs/df_vitl384_smoke.yaml", "w"), sort_keys=False)
print("written")
