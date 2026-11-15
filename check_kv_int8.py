"""int8 KV cache: accuracy vs bf16 KV (same fixed-buffer path), then the 32k-token run under VramGuard."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sn_loop_common import (VramExceeded, VramGuard, generate_chunked, load, make_fixed_cache,  # noqa: E402
                            use_gqa_fold_attention, warm_up)

sys.stdout.reconfigure(encoding="utf-8")
CTX, NEW = 32768, 32
CHUNK = int(os.environ.get('CHUNK', '512'))
SKIP_ACC = os.environ.get('SKIP_ACC') == '1'
QT = 4096  # tokens used for the accuracy comparison


def card():
    torch.cuda.synchronize()
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 1e9


tok, model = load("offline")
use_gqa_fold_attention(True)
guard = VramGuard()
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 14, return_tensors="pt").input_ids[:, :CTX].to("cuda")
try:
    # A. accuracy: bf16 KV and int8 KV caches fed the same 4096 tokens chunk by chunk, compared per position
    ids = ids_all[:, :QT]
    if SKIP_ACC:
        raise StopIteration
    ca = make_fixed_cache(model, QT, kv_int8=False)
    cb = make_fixed_cache(model, QT, kv_int8=True)
    buckets = {(0, 1024): [0.0, 0, 0], (1024, 2048): [0.0, 0, 0], (2048, 4096): [0.0, 0, 0]}
    with torch.inference_mode():
        for s in range(0, QT, CHUNK):
            e = min(s + CHUNK, QT)
            pos = torch.arange(s, e, device="cuda")[None]
            ha = model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=ca, use_cache=True).last_hidden_state
            hb = model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=cb, use_cache=True).last_hidden_state
            for r in range(0, e - s, 128):
                la = torch.log_softmax(model.lm_head(ha[:, r:r + 128]).float(), -1)[0]
                lb = torch.log_softmax(model.lm_head(hb[:, r:r + 128]).float(), -1)[0]
                kl = (la.exp() * (la - lb)).sum(-1)
                top = (la.argmax(-1) == lb.argmax(-1))
                for i in range(kl.shape[0]):
                    p = s + r + i
                    for (lo, hi), acc in buckets.items():
                        if lo <= p < hi:
                            acc[0] += kl[i].item(); acc[1] += int(top[i]); acc[2] += 1
            guard.check()
    for (lo, hi), (k, t, n) in buckets.items():
        print(f"[精度] 位置 {lo}-{hi}：int8 KV 相对 bf16 KV 的 KL {k / n:.2e}，首选一致 {t / n * 100:.1f}%", flush=True)
    del ca, cb
    torch.cuda.empty_cache()
    sa = make_fixed_cache(model, 1500 + NEW, kv_int8=False)
    sb = make_fixed_cache(model, 1500 + NEW, kv_int8=True)
    ga, _ = generate_chunked(model, ids_all[:, :1500], NEW, chunk_size=CHUNK, guard=guard, cache=sa)
    gb, _ = generate_chunked(model, ids_all[:, :1500], NEW, chunk_size=CHUNK, guard=guard, cache=sb)
    print(f"[精度] prompt 1500 贪心生成 {NEW} 个 token：前 {next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), len(gb))} 个相同", flush=True)
    del sa, sb
    torch.cuda.empty_cache()
except StopIteration:
    pass
try:
    # B. warm-up, allocate the 32k int8 cache once, then run
    t = time.time()
    warm_up(model, kv_int8=True)
    print(f"[预热] {time.time() - t:.0f}s，之后整卡 {card():.2f} GB", flush=True)
    before = card()
    cache = make_fixed_cache(model, CTX + NEW, kv_int8=True)
    print(f"[分配] {CTX + NEW} token：int8 KV {cache._fixed_bytes / 1e9:.2f} GB + 还原用临时缓冲区 "
          f"{cache._scratch_bytes / 1e9:.2f} GB；整卡 {before:.2f} → {card():.2f} GB", flush=True)
    guard.check()
    for T in [8192, CTX]:
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        guard.peak = 0
        out, st = generate_chunked(model, ids_all[:, :T], NEW, chunk_size=CHUNK, guard=guard, cache=cache)
        print(f"[prompt {T}, 每块 {CHUNK}] 计算临时显存 {(torch.cuda.max_memory_allocated() - base) / 1e9:.2f} GB，"
              f"整卡峰值 {guard.peak / 1e9:.2f}/{guard.total / 1e9:.2f} GB；处理 prompt {st['prefill_s']:.1f}s "
              f"（{T / st['prefill_s']:.0f} token/s），生成 {st['decode_tokens'] / st['decode_s']:.1f} tok/s；"
              f"续写开头：{tok.decode(out[:16], skip_special_tokens=True)!r}", flush=True)
except VramExceeded as e:
    print("停止：", e)
finally:
    guard.close()
