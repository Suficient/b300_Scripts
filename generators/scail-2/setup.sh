#!/usr/bin/env bash
# SCAIL-2 (zai-org, Jun 2026) on Wan2.1-14B-I2V, replacement mode with the relighting LoRA. MIT weights.
. "$(dirname "$0")/../common.sh"
clone https://github.com/zai-org/SCAIL-2 78fe19576bb06be96c2375e088574a262a300edb
venv
flash_attn
reqs requirements.txt
hf zai-org/SCAIL-2 --revision 150cc0ca4e98e50e60b9295dacde39442fdccab2 --local-dir SCAIL-2
# the release is a sat (FSDP) checkpoint; generate.py on the wan branch wants safetensors. torch >= 2.6 loads
# weights only by default, which this pickle (the official release) needs off
sed -i 's/checkpoint = torch.load(pt_file_path)$/checkpoint = torch.load(pt_file_path, weights_only=False)/' convert.py
[ -f SCAIL-2.safetensors ] || "$PY" convert.py --scail-dir SCAIL-2 --save-path SCAIL-2.safetensors
done_msg
