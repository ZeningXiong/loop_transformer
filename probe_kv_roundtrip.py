"""量化 KV 反量化的精度：int8/int4 × 每-token scale 用 bf16 表示会丢多少，和 fp32 比差多少。"""
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
dev = "cuda"
torch.manual_seed(0)

for bits, qmax in ((8, 127), (4, 7)):
    n = 4 * 16384 * 256
    q = torch.randint(-qmax, qmax + 1, (n,), device=dev, dtype=torch.int16)
    scale = torch.rand(n // 256, device=dev, dtype=torch.float32) * 0.02 + 1e-3
    scale = scale.repeat_interleave(256)
    exact = q.float() * scale                     # 真实值（fp32）
    via_bf16 = exact.to(torch.bfloat16).float()   # 当前做法：反量化到 bf16 缓冲
    rel = ((via_bf16 - exact).abs() / exact.abs().clamp_min(1e-12))
    print(f"int{bits}: bf16 表示的最大相对误差 {rel.max().item():.2e}，"
          f"平均 {rel.mean().item():.2e}，完全相同比例 {(via_bf16 == exact).float().mean().item() * 100:.2f}%")
