"""图重放到底花在哪：同一份缓存、同一状态，比"裸重放"和"eager 前向"的纯 GPU 时间。

单一进程、单一缓存（只建 Graph2Decoder 的缓存），符合测量纪律。
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_loop_common import load  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
tok, model = load("offline", int8=[])
dev = "cuda"
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()
max_len = n_ctx + 40
dec = Graph2Decoder(model, max_len, kv_int8=True, qlen=2)
snap = _Snapshot(dec.cache)


@torch.inference_mode()
def prefill():
    dec.reset()
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=dec.cache, use_cache=True)


def timed(fn, reps=8, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


with torch.inference_mode():
    prefill()
    snap.save()
    toks = torch.tensor([[1000, 2000]], device=dev)

    # 捕获（含预热）
    t_cap = dec.capture(dec.bucket_for(n_ctx + 2)) if False else None
    bucket = dec.bucket_for(n_ctx + 2)
    t0 = time.perf_counter()
    dec.step(toks, n_ctx)          # 首次会捕获
    torch.cuda.synchronize()
    cap_ms = (time.perf_counter() - t0) * 1e3
    snap.restore()
    dec.advance(0)

    g = dec.graphs[bucket]
    buf = dec.bufs[bucket]

    # 1) 纯重放
    rep_ms = timed(lambda: g.replay())
    # 2) 完整 dec.step（含 embedding 读、掩码填充、索引填充、重放）
    step_ms = timed(lambda: (dec.step(toks, n_ctx), snap.restore()))

    # 3) 同样状态下的 eager 前向（同一个缓存，不使用图）
    def eager():
        p = torch.arange(n_ctx, n_ctx + 2, device=dev)[None]
        h = model.model(inputs_embeds=dec.emb(toks), position_ids=p,
                        attention_mask={"full_attention": None, "linear_attention": None},
                        past_key_values=dec.cache, use_cache=True)
        return model.lm_head(h.last_hidden_state)

    eager_ms = timed(lambda: (eager(), snap.restore()))

print(f"上下文 {n_ctx}，2 个 token 一次前向，同一份缓存")
print(f"  捕获一次：{cap_ms:.0f} ms")
print(f"  纯图重放（不含任何 Python 准备）：{rep_ms:.1f} ms")
print(f"  完整 dec.step（准备 + 重放）：{step_ms:.1f} ms")
print(f"  eager 前向（同一状态）：{eager_ms:.1f} ms")
print(f"  → 重放本身比 eager {'慢' if rep_ms > eager_ms else '快'} "
      f"{abs(rep_ms - eager_ms) / eager_ms * 100:.0f}%")
