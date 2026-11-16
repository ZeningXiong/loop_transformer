"""延迟编码能省多少"编码"时间：标量 int8 vs 256 码字查表，单 token、14 个槽位。"""
import statistics
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8")
dev = "cuda"
shapes = [(1, 4, 1, 256)] * 14
x = [torch.randn(s, device=dev) for s in shapes]
cb = torch.randn(256, 4, device=dev)
G = 4


def scalar():
    for t in x:
        s = t.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
        (t / s).round_()


def codebook():
    for t in x:
        v = t.reshape(-1, G)
        d = (v[:, None, :] - cb[None, :, :]).pow(2).sum(-1)
        d.argmin(-1)


def bench(fn, n=50, warm=10):
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


a, b = bench(scalar), bench(codebook)
print(f"一个 token 的 KV 编码（14 个槽位，每槽 1024 个元素）：")
print(f"  标量 int8（除法+取整）     {a:6.3f} ms")
print(f"  256 码字最近邻搜索         {b:6.3f} ms   （×{b / a:.1f}）")
