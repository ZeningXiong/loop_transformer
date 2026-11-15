"""Fixed KV buffer + folded GQA attention: correctness vs the previous path, then a 32k-token prompt.

Everything runs under VramGuard: if whole-card VRAM reaches the physical limit, the test stops.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sn_loop_common import (VramExceeded, VramGuard, generate_chunked, load, make_fixed_cache,  # noqa: E402
                            use_gqa_fold_attention, warm_up)

sys.stdout.reconfigure(encoding="utf-8")
CTX = 32768
NEW = 32
CHUNK = int(os.environ.get('CHUNK', '512'))
SKIP_CHECK = os.environ.get('SKIP_CHECK') == '1'


def card():
    torch.cuda.synchronize()
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 1e9


def kl(a, b):
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return (la.exp() * (la - lb)).sum().item()


tok, model = load("offline")
guard = VramGuard()
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 14, return_tensors="pt").input_ids
assert ids_all.shape[1] >= CTX, ids_all.shape
ids_all = ids_all[:, :CTX].to("cuda")
try:
    # 1. correctness: previous path (DynamicCache + repeat-K/V SDPA) vs fixed buffer + folded attention
    use_gqa_fold_attention(True)
    ids = ids_all[:, :1500]
    if SKIP_CHECK:
        raise StopIteration
    use_gqa_fold_attention(False)
    ref, s_ref = generate_chunked(model, ids, NEW, chunk_size=512, guard=guard)
    use_gqa_fold_attention(True)
    small = make_fixed_cache(model, 1500 + NEW)
    new, s_new = generate_chunked(model, ids, NEW, chunk_size=512, guard=guard, cache=small)
    first_diff = next((i for i, (x, y) in enumerate(zip(ref, new)) if x != y), len(new))
    print(f"[正确性] prompt 1500：首 token 分布 KL {kl(s_ref['first_logits'], s_new['first_logits']):.2e}，"
          f"{NEW} 个生成 token 中前 {first_diff} 个相同", flush=True)
    del small
    torch.cuda.empty_cache()

    pass
except StopIteration:
    pass
try:
    # 2. warm-up on a tiny cache, then allocate the 32k buffers once
    t = time.time()
    warm_up(model)
    print(f"[预热] {time.time() - t:.0f}s，之后整卡 {card():.2f} GB", flush=True)
    before = card()
    cache = make_fixed_cache(model, CTX + NEW)
    after = card()
    print(f"[分配] {CTX + NEW} token 的固定 KV 缓冲区 {cache._fixed_bytes / 1e9:.2f} GB；"
          f"整卡 {before:.2f} → {after:.2f} GB", flush=True)
    guard.check()

    # 3. the 32k run (and 8k for comparison with the old 7.42 GB), same preallocated buffers
    for T in [8192, CTX]:
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        guard.peak = 0
        out, st = generate_chunked(model, ids_all[:, :T], NEW, chunk_size=CHUNK, guard=guard, cache=cache)
        print(f"[prompt {T}, 每块 {CHUNK}] 计算临时显存 {(torch.cuda.max_memory_allocated() - base_alloc) / 1e9:.2f} GB，整卡峰值 {guard.peak / 1e9:.2f}/{guard.total / 1e9:.2f} GB，PyTorch 张量峰值 "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB；处理 prompt {st['prefill_s']:.1f}s "
              f"（{T / st['prefill_s']:.0f} token/s），生成 {st['decode_tokens'] / st['decode_s']:.1f} tok/s；"
              f"续写开头：{tok.decode(out[:20], skip_special_tokens=True)!r}", flush=True)
except StopIteration:
    pass
except VramExceeded as e:
    print("停止：", e)
finally:
    guard.close()
