import json, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sn_loop_common import load
tok, model = load("offline")
# log what the full-attention layers actually pass to SDPA
orig = torch.nn.functional.scaled_dot_product_attention
seen = []
def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if len(seen) < 1:
        seen.append(dict(q=tuple(q.shape), k=tuple(k.shape), q_stride_last=q.stride(-1), mask=None if attn_mask is None else (tuple(attn_mask.shape), str(attn_mask.dtype)),
                         is_causal=is_causal, extra=list(kw)))
    return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
torch.nn.functional.scaled_dot_product_attention = spy
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids = tok(text * 3, return_tensors="pt").input_ids[:, :2048].to("cuda")
for name, ctx in [("默认", None), ("强制 EFFICIENT", SDPBackend.EFFICIENT_ATTENTION)]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    try:
        with torch.inference_mode(), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if ctx is None:
                out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
            else:
                with sdpa_kernel(ctx):
                    out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        msgs = sorted({str(x.message)[:160] for x in w if "sdp" in str(x.message).lower() or "kernel" in str(x.message).lower()})
        print(f"{name}: prompt 2048 处理峰值增量 {(torch.cuda.max_memory_allocated() - base) / 1e9:.2f} GB", msgs[:3])
        if ctx is None: ref = out.logits.float()
        else: print("  与默认结果的最大 logit 差:", (out.logits.float() - ref).abs().max().item())
    except Exception as e:
        print(f"{name}: 失败 -> {str(e)[:300]}")
print("SDPA 实际收到的参数:", seen[0])
