import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from sn_loop_common import load
tok, model = load("offline")
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 3, return_tensors="pt").input_ids
orig = torch.nn.functional.scaled_dot_product_attention
seen = {}
def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    key = (tuple(q.shape), None if attn_mask is None else (tuple(attn_mask.shape), str(attn_mask.dtype)), is_causal, tuple(kw))
    seen[key] = seen.get(key, 0) + 1
    return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
torch.nn.functional.scaled_dot_product_attention = spy
for T, cache in [(4096, True), (8192, True)]:
    ids = ids_all[:, :T].to("cuda"); seen.clear()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        torch.cuda.synchronize(); t = time.time()
        if cache:
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        else:
            out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    print(f"T={T} {'generate(带cache)' if cache else 'forward(无cache)'}: 峰值增量 {(torch.cuda.max_memory_allocated() - base) / 1e9:.2f} GB, {time.time() - t:.1f}s", flush=True)
    for k, v in seen.items(): print("   SDPA 调用:", k, "x", v)
