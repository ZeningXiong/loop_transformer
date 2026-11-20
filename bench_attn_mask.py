"""16K 下注意力带掩码 vs 不带掩码的差距：图路径为什么追不上 eager。"""
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from cudagraph_decode import masked_fold_attention  # noqa: E402
from sn_loop_common import gqa_fold_attention  # noqa: E402

dev = "cuda"
L, Hq, Hkv, D, Tq = 16464, 16, 4, 256, 2
q = torch.randn(1, Hq, Tq, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, Hkv, L, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, Hkv, L, D, device=dev, dtype=torch.bfloat16)
mask = torch.ones(1, 1, Tq, L, dtype=torch.bool, device=dev)


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def no_mask():
    gqa_fold_attention(None, q, k, v, None)


def with_mask():
    masked_fold_attention(None, q, k, v, mask)


a = bench(no_mask)
b = bench(with_mask)
print(f"16K 前缀、2 个查询：")
print(f"  折叠 GQA（无掩码）        {a:6.2f} ms")
print(f"  折叠 GQA（带 16K 掩码）   {b:6.2f} ms   （×{b / a:.2f}，每 token 14 次调用多 {(b - a) * 14:.0f} ms）")
