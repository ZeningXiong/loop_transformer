"""端到端：16K 下"注意力直接读量化 KV"开/关的速度与一致性。"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import (load, make_fixed_cache, reset_cache,  # noqa: E402
                            use_fused_kv_attention)

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 24
dev = "cuda"
tok, model = load("offline", int8=[])
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()


@torch.inference_mode()
def run(n=N_GEN):
    reset_cache(cache)
    t0 = time.perf_counter()
    h = None
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    torch.cuda.synchronize()
    pre = time.perf_counter() - t0
    x = model.lm_head(h[:, -1:]).float()[0, -1].argmax().view(1, 1)
    out, pos = [], ids.shape[1]
    t1 = time.perf_counter()
    for _ in range(n):
        out.append(x.item())
        h = model.model(input_ids=x, position_ids=torch.tensor([[pos]], device=dev),
                        past_key_values=cache, use_cache=True).last_hidden_state
        x = model.lm_head(h).float()[0, -1].argmax().view(1, 1)
        pos += 1
    torch.cuda.synchronize()
    return out, pre, time.perf_counter() - t1


print(f"16K（{N_CTX} token 上下文），贪心 {N_GEN} token：")
use_fused_kv_attention(False)
a, pre_a, dec_a = run()
print(f"  逐 token 解量化到 bf16（原）  prefill {pre_a:5.1f}s | 解码 {dec_a:5.2f}s | {N_GEN / dec_a:5.2f} tok/s")
use_fused_kv_attention(True)
b, pre_b, dec_b = run()
print(f"  注意力直接读量化 KV（新）      prefill {pre_b:5.1f}s | 解码 {dec_b:5.2f}s | {N_GEN / dec_b:5.2f} tok/s")
use_fused_kv_attention(False)
same = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), min(len(a), len(b)))
print(f"  相同前缀：{same}/{N_GEN}")
if same < N_GEN:
    print(f"    原 {a[max(0, same-2):same+4]}")
    print(f"    新 {b[max(0, same-2):same+4]}")
