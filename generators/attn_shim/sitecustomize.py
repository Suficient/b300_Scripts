# Imported at interpreter start by drive.py's workers (this folder is on PYTHONPATH; GEN_ATTN=fa2 turns it off).
# On the B300, cuDNN's fused attention (via torch SDPA) is ~3.5x faster than our FlashAttention 2 sm_100 build at
# video-DiT sizes (33k tokens: 31 vs 110 ms; 110k: 493 vs 1787 ms) and gives the same result (bf16 rounding). So:
# - torch SDPA tries cuDNN first (it falls through to the next backend for inputs cuDNN doesn't take);
# - flash_attn's dense and varlen functions are rerouted to SDPA, except for features SDPA doesn't have (dropout,
#   windows, softcap, ALiBi, returned probabilities/LSE, paged KV), which keep flash_attn.
import os

if os.environ.get("GEN_ATTN", "cudnn") == "cudnn":
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend
        torch._C._set_sdp_priority_order([int(b) for b in (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                                                            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)])
    except Exception:
        torch = None

    def _plain(dropout_p=0.0, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, return_attn_probs=False,
               block_table=None, **_):
        return (not dropout_p and tuple(window_size) == (-1, -1) and not softcap and alibi_slopes is None
                and not return_attn_probs and block_table is None)

    def _sdpa(q, k, v, scale, causal):   # (B, S, H, D) in and out
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale, enable_gqa=k.shape[1] != q.shape[1])
        return o.transpose(1, 2)

    def _patch():
        import flash_attn
        import flash_attn.flash_attn_interface as fi
        orig = {n: getattr(fi, n) for n in ("flash_attn_func", "flash_attn_varlen_func", "flash_attn_qkvpacked_func",
                                            "flash_attn_kvpacked_func", "flash_attn_varlen_kvpacked_func")}

        def dense(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, *a, **kw):
            if a or not _plain(dropout_p, **kw):
                return orig["flash_attn_func"](q, k, v, dropout_p, softmax_scale, causal, *a, **kw)
            return _sdpa(q, k, v, softmax_scale, causal)

        # flash_attn's own parameter names: callers (Wan, InfiniteTalk) pass them as keywords
        def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                   causal=False, *a, **kw):
            if a or not _plain(dropout_p, **kw):
                return orig["flash_attn_varlen_func"](q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                                      dropout_p, softmax_scale, causal, *a, **kw)
            out, cq, ck = q.new_empty(q.shape[:2] + v.shape[2:]), cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
            for i in range(len(cq) - 1):
                out[cq[i]:cq[i + 1]] = _sdpa(q[None, cq[i]:cq[i + 1]], k[None, ck[i]:ck[i + 1]],
                                             v[None, ck[i]:ck[i + 1]], softmax_scale, causal)[0]
            return out

        def qkvpacked(qkv, dropout_p=0.0, softmax_scale=None, causal=False, *a, **kw):
            return dense(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], dropout_p, softmax_scale, causal, *a, **kw)

        def kvpacked(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, *a, **kw):
            return dense(q, kv[:, :, 0], kv[:, :, 1], dropout_p, softmax_scale, causal, *a, **kw)

        def varlen_kvpacked(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                            softmax_scale=None, causal=False, *a, **kw):
            return varlen(q, kv[:, 0], kv[:, 1], cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                          softmax_scale, causal, *a, **kw)

        new = {"flash_attn_func": dense, "flash_attn_varlen_func": varlen, "flash_attn_qkvpacked_func": qkvpacked,
               "flash_attn_kvpacked_func": kvpacked, "flash_attn_varlen_kvpacked_func": varlen_kvpacked}
        for m in (flash_attn, fi):
            for n, f in new.items():
                setattr(m, n, f)

    if torch is not None:
        try:
            _patch()
        except ImportError:
            pass
