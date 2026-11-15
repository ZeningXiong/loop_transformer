"""单独跑 16K + 4-token 验证，报告绝对数值（同一轮里带一个 1-token 单步参考）。

用法: python spec4_16k.py [上下文] [预热 token 数] [计时 token 数]
"""
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
n_warm = int(sys.argv[2]) if len(sys.argv) > 2 else 8
n_time = int(sys.argv[3]) if len(sys.argv) > 3 else 48
sys.argv = [sys.argv[0], str(n_ctx), str(n_warm + n_time + 8)]   # bench_spec4 用 argv 决定缓存长度

import bench_spec4 as B4  # noqa: E402  载入模型 / MTP 头 / 16K 缓存

model, mtp, cache, tap, ids, dev = B4.model, B4.mtp, B4.cache, B4.tap, B4.ids, B4.dev


@torch.inference_mode()
def one_token_cost(reps=8):
    """同一轮里的 1-token 单步耗时，作为绝对数值的参照。"""
    kv = [l for l in cache.layers if hasattr(l, "cur")]
    base = [l.cur for l in kv]
    x = torch.tensor([[1000]], device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hh = model.model(input_ids=x, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
        model.lm_head(hh).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        for layer, c in zip(kv, base):
            layer.cur = c
            layer._set_views()
    return statistics.median(ts) * 1e3


print(f"上下文 {n_ctx}，全 NF4 权重，KV int8，每步验证 4 个（3 个链式草稿）", flush=True)
print("预热一轮…", flush=True)
B4.run(3, n_warm)
torch.cuda.empty_cache()

t1 = one_token_cost()
print(f"同一轮 1-token 单步：{t1:.1f} ms（{1e3 / t1:.2f} tok/s）", flush=True)

t0 = time.perf_counter()
toks, st = B4.run(3, n_time)
wall = time.perf_counter() - t0
tps = st["tokens"] / st["decode_s"]
print(f"\n4-token 验证（含 prefill 的总墙钟 {wall:.1f}s，其中 prefill {st['prefill_s']:.1f}s）")
print(f"  生成 {st['tokens']} 个 token / {st['decode_s']:.2f}s = **{tps:.2f} tok/s**"
      f"（{1e3 / tps:.1f} ms/token）", flush=True)
print(f"  步数 {st['steps']}，平均每步提交 {st['tokens'] / st['steps']:.2f} 个 token，"
      f"每步 {st['decode_s'] / st['steps'] * 1e3:.0f} ms")
print(f"  逐位命中（共 {st['steps']} 步）：{st['acc']} → "
      + "、".join(f"第 {i + 1} 位 {100 * a / st['steps']:.0f}%" for i, a in enumerate(st["acc"])))
print(f"  相对同一轮 1-token 单步：×{tps * t1 / 1e3:.2f}")
print(f"整卡峰值 {B4.guard.peak / 1e9:.2f} GB")
B4.guard.close()
