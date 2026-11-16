"""LoopVQ 可行性第一步：2-bit 码本反量化在 16K decode 下的算子代价。

只测算子，不碰模型。对比三种"每层把整段前缀还原成 bf16"的做法：
  int8   ：现在用的  torch.mul(k_q, k_s, out=kb)        （8.12 bit/元素）
  2bit-TR：Triton 按码本 gather（1 字节一个 4 元素组 = 2 bit/元素）
  2bit-PT：纯 PyTorch 的 gather 版本（兜底实现有多慢）

16K 的前缀是 [1, 4, 16440, 256] = 4.2M 组；循环段 12 个槽位用它，另 2 个单访问槽位仍用 int8。
"""
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

sys.stdout.reconfigure(encoding="utf-8")
dev = "cuda"
L, H, D = 16440, 4, 256
n_elem = H * L * D
n_grp = n_elem // 4
print(f"每层前缀 {H}×{L}×{D} = {n_elem / 1e6:.1f}M 元素 = {n_grp / 1e6:.2f}M 个 4 元素组")


@triton.jit
def _deq2(codes, cb, out, n_g, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    gi = pid * BLOCK + tl.arange(0, BLOCK)
    mask = gi < n_g
    c = tl.load(codes + gi, mask=mask, other=0).to(tl.int32)[:, None]
    j = tl.arange(0, 4)[None, :]
    v = tl.load(cb + c * 4 + j, mask=mask[:, None], other=0.0)
    tl.store(out + gi[:, None] * 4 + j, v, mask=mask[:, None])


k_q = torch.randint(-127, 127, (1, H, L, D), dtype=torch.int8, device=dev)
k_s = torch.rand(1, H, L, 1, device=dev) * 0.01
out = torch.zeros(1, H, L, D, dtype=torch.bfloat16, device=dev)
codes = torch.randint(0, 255, (n_grp,), dtype=torch.uint8, device=dev)
cb = torch.randn(256, 4, device=dev)
out2 = torch.zeros(n_grp * 4, dtype=torch.float32, device=dev)
grid = (triton.cdiv(n_grp, 4096),)


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


def int8_deq():
    torch.mul(k_q[:, :, :L], k_s, out=out)


def deq2_triton():
    _deq2[grid](codes, cb, out2, n_grp, BLOCK=4096, num_warps=8)


def deq2_torch():
    out2.view(-1, 4).copy_(cb[codes.long()])


t8 = bench(int8_deq)
tt = bench(deq2_triton)
tp = bench(deq2_torch)

print(f"\n每层、16K 前缀、还原成 bf16：")
print(f"  int8      torch.mul            {t8:7.3f} ms")
print(f"  2bit      Triton 码本 gather   {tt:7.3f} ms   （×{tt / t8:.2f} vs int8）")
print(f"  2bit      纯 PyTorch gather    {tp:7.3f} ms   （×{tp / t8:.2f}）")

print(f"\n一步 decode 的 KV 反量化总代价（12 个循环槽位 + 2 个单访问槽位）：")
print(f"  现状全 int8        ：14 × {t8:.2f} = {14 * t8:6.1f} ms")
print(f"  12 槽 2-bit + 2 int8：12 × {tt:.2f} + 2 × {t8:.2f} = {12 * tt + 2 * t8:6.1f} ms")
print(f"  差值（2-bit 替换）  ：{(12 * tt + 2 * t8) - 14 * t8:+.1f} ms/步")
print(f"\n参考：当前 16K 一步 189 ms（2-token 验证，含 14 槽 int8 反量化）")
