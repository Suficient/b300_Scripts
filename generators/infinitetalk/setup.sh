#!/usr/bin/env bash
# InfiniteTalk: audio-driven talking video on Wan2.1-I2V-14B (480P weights), image-to-video from B's frame with
# A's audio. Apache-2.0. Upstream pins torch 2.4.1; runs on the platform torch. infinitetalk-v2v reuses this install.
. "$(dirname "$0")/../common.sh"
clone https://github.com/MeiGen-AI/InfiniteTalk 50aa0a94184315407a991ae804d9b58d6d311ba8
venv
flash_attn   # before reqs: xfuser depends on it
reqs requirements.txt
install xformers==0.0.35 --no-deps   # the audio cross-attention's BlockDiagonalMask; 0.0.35 runs on sm_103
# xformers 0.0.35 has no unmasked operator here: unmasked calls (both, without sequence parallel) are plain
# attention, so they use SDPA ((B M H K) layout)
sed -i 's/x = xformers.ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=None, op=None,)/x = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, encoder_k, encoder_v))).transpose(1, 2)/' wan/modules/attention.py
sed -i 's/x = xformers.ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=attn_bias, op=None,)/x = xformers.ops.memory_efficient_attention(q, encoder_k, encoder_v, attn_bias=attn_bias, op=None,) if attn_bias is not None else torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, encoder_k, encoder_v))).transpose(1, 2)/' wan/modules/attention.py
install "transformers>=4.49,<5" diffusers==0.35.2   # 5.x drops the custom Wav2Vec2Model's hidden_states;
# diffusers 0.36+ needs huggingface_hub 1.x, which transformers 4 refuses
install librosa "misaki[en]"
sed -i '/^from inspect import ArgSpec$/d' wan/multitalk.py   # unused; gone in Python 3.11
hf Wan-AI/Wan2.1-I2V-14B-480P --revision 6b73f84e66371cdfe870c72acd6826e1d61cf279 --local-dir weights/Wan2.1-I2V-14B-480P
hf TencentGameMate/chinese-wav2vec2-base --revision 3991242c806928916fff4a8c0e4f76acf661b743 \
  --exclude chinese-wav2vec2-base-fairseq-ckpt.pt --local-dir weights/chinese-wav2vec2-base
hf TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir weights/chinese-wav2vec2-base
hf MeiGen-AI/InfiniteTalk --revision d59847ebdacf19245bfca3fb23311c0cada8378a single/infinitetalk.safetensors \
  --local-dir weights/InfiniteTalk
done_msg
