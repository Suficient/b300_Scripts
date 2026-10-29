#!/usr/bin/env bash
# SoulX-FlashTalk-14B: real-time (4 NFE, self-forcing distilled) talking head on Wan 14B, B's frame + A's audio.
# Apache-2.0. Upstream pins torch 2.7.1 cu128 + flash_attn 2.8.0; runs on the platform torch + FA 2.8.3.
. "$(dirname "$0")/../common.sh"
clone https://github.com/Soul-AILab/SoulX-FlashTalk 2ba2e1e801d59a9f526212a5aae5d374db8f2978
venv
flash_attn
reqs requirements.txt
install xformers==0.0.35 --no-deps   # the audio cross-attention's BlockDiagonalMask; 0.0.35 runs on sm_103
install "transformers>=4.49,<5" diffusers==0.35.2   # 5.x drops the custom Wav2Vec2Model's hidden_states;
# diffusers 0.36+ needs huggingface_hub 1.x, which transformers 4 refuses
hf Soul-AILab/SoulX-FlashTalk-14B --revision f3fda6499a6022862cadd39fc41ae63b64706e1b --local-dir models/SoulX-FlashTalk-14B
hf TencentGameMate/chinese-wav2vec2-base --revision 3991242c806928916fff4a8c0e4f76acf661b743 \
  --exclude chinese-wav2vec2-base-fairseq-ckpt.pt --local-dir models/chinese-wav2vec2-base
done_msg
