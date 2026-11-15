"""验证并测速 k 行 NF4 GEMV：一次读权重 vs bitsandbytes 的 k 次单行调用。

python test_nf4_gemv_k.py
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

from nf4_gemv_k import nf4_gemv_k  # noqa: E402
from sn_loop_common import load  # noqa: E402

LAYERS = ["model.language_model.layers.7.self_attn.q_proj",
          "model.language_model.layers.7.mlp.down_proj",
          "lm_head"]

tok, model = load("offline", int8=[])
code = bnb.functional.get_4bit_type("nf4", device="cuda")


def module_of(name):
    parent, _, attr = name.rpartition(".")
    return getattr(model.get_submodule(parent), attr)


def bench(fn, iters=100):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


print(f"{'层':<34}{'k':>3}{'相对误差':>11}{'bnb×k µs':>11}{'ours µs':>10}{'加速':>7}")
for name in LAYERS:
    mod = module_of(name)
    qs = mod.quant_state
    n, k = qs.shape
    bs = qs.blocksize
    packed = mod.weight.data.reshape(n, k // 2)
    absmax = qs.absmax.reshape(n, k // bs).contiguous()
    with torch.inference_mode():
        w_ref = bnb.functional.dequantize_4bit(mod.weight.data, qs).to(torch.float32)
    for kr in (1, 2, 4):
        x = (torch.randn(kr, k, dtype=torch.bfloat16, device="cuda") * 0.05)
        with torch.inference_mode():
            ref = x.float() @ w_ref.T
            mine = nf4_gemv_k(x, packed, absmax, code, blocksize=bs)
            err = ((mine.float() - ref).norm() / ref.norm()).item()
            bnb_k = bench(lambda: [mod(x[i:i + 1]) for i in range(kr)])
            ours = bench(lambda: nf4_gemv_k(x, packed, absmax, code, blocksize=bs))
        print(f"{name.split('language_model.')[-1]:<34}{kr:>3}{err:>11.2e}"
              f"{bnb_k:>11.1f}{ours:>10.1f}{bnb_k / ours:>6.2f}x", flush=True)
    del w_ref
    torch.cuda.empty_cache()
    print()
