"""同一进程内比较：普通解码 / 验证 2 个 / 验证 4 个。

用法: python cmp_spec.py [上下文] [生成 token 数]
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

import bench_spec4 as B4  # noqa: E402  (reuses its model, MTP head, cache and 4-token loop)
from sn_spec import generate_speculative  # noqa: E402

model, mtp, cache, tap, ids, dev = B4.model, B4.mtp, B4.cache, B4.tap, B4.ids, B4.dev
n_ctx = ids.shape[1]
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 16


@torch.inference_mode()
def plain(n):
    B4.reset_cache(cache)
    mtp.cache.reset()
    T = ids.shape[1]
    t0 = time.perf_counter()
    for s in range(0, T, 256):
        e = min(s + 256, T)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    h = tap.post
    nxt = model.lm_head(h[:, -1:]).float().argmax(-1).reshape(1, 1)
    torch.cuda.synchronize()
    pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    for i in range(n):
        hh = model.model(input_ids=nxt, position_ids=torch.tensor([[T + i]], device=dev),
                         past_key_values=cache, use_cache=True).last_hidden_state
        nxt = model.lm_head(hh[:, -1:]).float().argmax(-1).reshape(1, 1)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t1), pre


with torch.inference_mode():
    plain(4)                                          # warm up the 1-token path
    tps1, pre = plain(n_gen)
    print(f"普通解码：{tps1:.2f} tok/s（{1e3 / tps1:.0f} ms/token），prefill {pre:.1f}s", flush=True)

    toks2, st2 = generate_speculative(model, mtp, ids, n_gen, cache, chunk_size=256)
    tps2 = st2["tokens"] / st2["decode_s"]
    print(f"验证 2 个（现有实现）：{tps2:.2f} tok/s，×{tps2 / tps1:.2f}；"
          f"步数 {st2['steps']}，命中 {st2['accepted']}，回退 {st2['redo']}", flush=True)

    toks4, st4 = B4.run(3, n_gen)
    tps4 = st4["tokens"] / st4["decode_s"]
    print(f"验证 4 个（3 个链式草稿）：{tps4:.2f} tok/s，×{tps4 / tps1:.2f}；"
          f"步数 {st4['steps']}，平均每步提交 {st4['tokens'] / st4['steps']:.2f} 个，"
          f"逐位命中 {st4['acc']}", flush=True)

print(f"\n一致性检查：两种投机都必须是主模型的贪心输出（这里只看前 12 个是否互相吻合）")
print("  验证2:", toks2[:12])
print("  验证4:", toks4[:12])
print(f"整卡峰值 {B4.guard.peak / 1e9:.2f} GB")
B4.guard.close()
