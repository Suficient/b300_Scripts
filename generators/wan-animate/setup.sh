#!/usr/bin/env bash
# Wan2.2-Animate-14B in replacement mode with the relighting LoRA: the person in A's clip becomes B. Apache-2.0.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Wan-Video/Wan2.2 1ea34ff48f87168174e12956e200b1d908b1c5ff
venv
flash_attn
reqs requirements.txt requirements_animate.txt
install librosa moviepy   # wan/__init__ imports the S2V modules too
SAM2_BUILD_CUDA=0 install "git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266"
# its CUDA op (only fills <= 8 px holes in the low-res masks) asserts on every input with this torch: skip that
# step, which SAM2 documents as optional
sed -i 's/"++model.fill_hole_area=8"/"++model.fill_hole_area=0"/' wan/modules/animate/preprocess/sam_utils.py
# the git install leaves sam2_configs without its yaml files; the preprocessor needs sam2_hiera_l.yaml
curl -sfL -o "$("$PY" -c 'import os, sam2_configs; print(os.path.dirname(sam2_configs.__file__))')/sam2_hiera_l.yaml" \
  https://raw.githubusercontent.com/facebookresearch/sam2/0e78a118995e66bb27d78518c4bd9a3e95b4e266/sam2_configs/sam2_hiera_l.yaml
# xlm-roberta-large is only the CLIP tokenizer; relighting_lora/ duplicates relighting_lora.ckpt
hf Wan-AI/Wan2.2-Animate-14B --revision cb93a225fbaf1ca100f54e79da8f994995b689b3 --local-dir Wan2.2-Animate-14B \
  --exclude "relighting_lora/*" --exclude "xlm-roberta-large/*.bin" \
  --exclude "xlm-roberta-large/*.msgpack" --exclude "xlm-roberta-large/*.h5" --exclude "xlm-roberta-large/model.safetensors"
done_msg
