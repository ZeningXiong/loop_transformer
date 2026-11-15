"""Validate and time nf4_gemv against bitsandbytes' own 4-bit GEMV on real checkpoint layers.

python test_nf4_gemv.py
"""
import json
import os
import statistics
import sys
import time

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from nf4_gemv import nf4_gemv  # noqa: E402
from sn_loop_common import load  # noqa: E402

LAYERS = ["model.language_model.layers.7.self_attn.q_proj",
          "model.language_model.layers.7.mlp.down_proj",
          "model.language_model.layers.11.mlp.gate_proj",
          "lm_head"]

tok, model = load("offline", int8=[])
code = bnb.functional.get_4bit_type("nf4", device="cuda")
print("NF4 code:", [round(v, 4) for v in code.tolist()], flush=True)


def module_of(name):
    parent, _, attr = name.rpartition(".")
    return getattr(model.get_submodule(parent), attr)


print(f"\n{'层':<46}{'形状':<16}{'bnb µs':>9}{'ours µs':>9}{'加速':>7}{'相对误差':>10}")
faster, total_bnb, total_ours = [], 0.0, 0.0
for name in LAYERS:
    mod = module_of(name)
    qs = mod.quant_state
    n, k = qs.shape
    bs = qs.blocksize
    packed = mod.weight.data.reshape(n, k // 2)
    absmax = qs.absmax.reshape(n, k // bs).contiguous()
    k2 = k // 2
    x = torch.randn(k2 * 2, dtype=torch.bfloat16, device="cuda") * 0.05

    with torch.inference_mode():
        w_ref = bnb.functional.dequantize_4bit(packed, qs).to(torch.float32)
        y_ref = w_ref @ x.float()
        y_ours = nf4_gemv(x, packed, absmax, code, blocksize=bs)
        y_bnb = mod(x.unsqueeze(0)).reshape(-1)
        torch.cuda.synchronize()

    err_ours = ((y_ours.float() - y_ref).norm() / y_ref.norm()).item()
    err_bnb = ((y_bnb.float() - y_ref).norm() / y_ref.norm()).item()

    def bench(fn, iters=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
        return statistics.median(t) * 1e6

    with torch.inference_mode():
        b = bench(lambda: mod(x.unsqueeze(0)))
        o = bench(lambda: nf4_gemv(x, packed, absmax, code, blocksize=bs))
    faster.append(b / o)
    total_bnb += b
    total_ours += o
    print(f"{name.split('language_model.')[-1]:<46}{f'[{n}, {k2 * 2}]':<16}"
          f"{b:>9.1f}{o:>9.1f}{b / o:>6.2f}x{max(err_ours, err_bnb):>10.2e}", flush=True)

print(f"\n平均加速 {sum(faster) / len(faster):.2f}x；4 层合计 {total_bnb:.0f} → {total_ours:.0f} µs")
print(f"按 244 个 4-bit 线性层折算：{total_bnb / 4 * 244 / 1e3:.1f} ms/步 → "
      f"{total_ours / 4 * 244 / 1e3:.1f} ms/步")
