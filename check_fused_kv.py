"""验证并测速：融合的量化 KV 注意力 vs "解量化到 bf16 scratch + SDPA"。"""
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from fused_kv_attn import attend  # noqa: E402
from int4_dequant import dequant4  # noqa: E402
from sn_loop_common import gqa_fold_attention  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
Hkv, Hq, D = 4, 16, 256
G = Hq // Hkv
dev = "cuda"
torch.manual_seed(0)


def make_layer(bits, L, scale=0.02):
    if bits == 4:
        k_q = torch.randint(0, 256, (1, Hkv, L, D // 2), device=dev, dtype=torch.uint8)
        v_q = torch.randint(0, 256, (1, Hkv, L, D // 2), device=dev, dtype=torch.uint8)
    else:
        k_q = torch.randint(-127, 128, (1, Hkv, L, D), device=dev, dtype=torch.int8)
        v_q = torch.randint(-127, 128, (1, Hkv, L, D), device=dev, dtype=torch.int8)
    k_s = torch.rand(1, Hkv, L, 1, device=dev, dtype=torch.float32) * scale + 1e-4
    v_s = torch.rand(1, Hkv, L, 1, device=dev, dtype=torch.float32) * scale + 1e-4
    return k_q, k_s, v_q, v_s


_buf = {}


def reference(q_folded, k_q, k_s, v_q, v_s, L, bits, tq):
    """现有路径：解量化到复用缓冲再走折叠 SDPA（缓冲预分配，模拟生产里的 scratch）。"""
    key = (bits, L)
    if key not in _buf:
        _buf[key] = (torch.empty(1, Hkv, L, D, dtype=torch.bfloat16, device=dev),
                     torch.empty(1, Hkv, L, D, dtype=torch.bfloat16, device=dev))
    kb, vb = _buf[key]
    if bits == 4:
        dequant4(k_q, k_s, kb)
        dequant4(v_q, v_s, vb)
    else:
        torch.mul(k_q, k_s, out=kb)
        torch.mul(v_q, v_s, out=vb)
    q = q_folded.reshape(1, Hkv, G * tq, D)
    out, _ = gqa_fold_attention(None, q.reshape(1, Hq, tq, D), kb, vb, None)   # [1, tq, Hq, D]
    return out.transpose(1, 2).reshape(1, Hq * tq, D).reshape(Hkv, G * tq, D)


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


for bits in (8, 4):
    L = N
    k_q, k_s, v_q, v_s = make_layer(bits, L)
    for tq in (1, 2):
        q = torch.randn(Hkv, G * tq, D, device=dev, dtype=torch.bfloat16) * 0.5
        got = attend(q, k_q, k_s, v_q, v_s, L, bits, G, tq)
        ref = reference(q, k_q, k_s, v_q, v_s, L, bits, tq)
        d = (got.float() - ref.float()).abs()
        rel = (d / ref.float().abs().clamp_min(1e-3)).mean().item()
        t_got = bench(lambda: attend(q, k_q, k_s, v_q, v_s, L, bits, G, tq))
        t_ref = bench(lambda: reference(q, k_q, k_s, v_q, v_s, L, bits, tq))
        print(f"int{bits} Tq={tq}: 最大差 {d.max().item():.2e} 平均相对差 {rel:.2e} | "
              f"融合 {t_got:5.2f} ms | 现路径 {t_ref:5.2f} ms", flush=True)
