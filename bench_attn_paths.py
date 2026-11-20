"""16K 下三种注意力实现的开销对比：折叠 GQA（生产路径）/ 广播 KV / enable_gqa。"""
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from sn_loop_common import gqa_fold_attention  # noqa: E402

dev = "cuda"
L, Hq, Hkv, D, Tq = 16440, 16, 4, 256, 2
q = torch.randn(1, Hq, Tq, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, Hkv, L, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, Hkv, L, D, device=dev, dtype=torch.bfloat16)


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


def fold():
    gqa_fold_attention(None, q, k, v, None)


def broadcast():
    kk = k.repeat_interleave(Hq // Hkv, dim=1)
    vv = v.repeat_interleave(Hq // Hkv, dim=1)
    torch.nn.functional.scaled_dot_product_attention(q, kk, vv)


def gqa_native():
    torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)


for name, fn in (("折叠 GQA（生产）", fold), ("广播 KV", broadcast), ("enable_gqa", gqa_native)):
    try:
        print(f"{name:<18}{bench(fn):7.3f} ms")
    except Exception as e:
        print(f"{name:<18}失败: {type(e).__name__} {str(e)[:60]}")
