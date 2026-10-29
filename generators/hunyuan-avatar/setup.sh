#!/usr/bin/env bash
# HunyuanVideo-Avatar: audio-driven avatar on the 13B HunyuanVideo DiT (the authors' fp8 weights). Tencent Hunyuan
# Community License: not licensed for use in the EU, UK or South Korea. Upstream pins torch 2.4 / FA 2.6.3.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar 8c31d0d489df8418fa36a2a709ebdf0b47c7bf06
venv
flash_attn
SKIP="numpy" reqs requirements.txt
install transformers==4.41.2   # the README's pin (requirements.txt says >=4.50; newer Llava code breaks its text encoder)
install "numpy>=1.26.4,<2"   # pinned 1.24.4 is below decord2's floor; opencv 4.9 needs <2
hf tencent/HunyuanVideo-Avatar --revision 43884f06f848cbc1c2bbfb8c32b366782538b953 --local-dir weights \
  --exclude "*/mp_rank_00_model_states.pt" --exclude "ckpts/text_encoder_2/*.h5" \
  --exclude "ckpts/text_encoder_2/*.msgpack" --exclude "ckpts/text_encoder_2/*.bin"
done_msg
