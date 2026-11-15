"""把 k 行 NF4 GEMV 接进 bitsandbytes：2~4 行时一次读权重，1 行仍走 bnb 的快路径。

用法: python test_k_row_patch.py [上下文]
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
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
PLAN = "mixed_plan_1.0.json"
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))["int8"])
CODE = bnb.functional.get_4bit_type("nf4", device="cuda")
_patched = [False]


def k_row_gemv(enable=True):
    """Route 2-4 row forwards through the multi-row kernel (same math, weight read once)."""
    cls = bnb.nn.Linear4bit
    if not hasattr(cls, "_orig_forward"):
        cls._orig_forward = cls.forward

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        if 2 <= rows <= 4 and self.bias is None:
            qs = self.weight.quant_state
            n, k = qs.shape
            packed = self.weight.data.reshape(n, k // 2)
            absmax = qs.absmax.reshape(n, k // qs.blocksize)
            y = nf4_gemv_k(x.reshape(rows, k), packed, absmax, CODE, blocksize=qs.blocksize)
            return y.reshape(*x.shape[:-1], n)
        return cls._orig_forward(self, x)

    cls.forward = forward if enable else cls._orig_forward
    _patched[0] = enable


texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
max_len = n_ctx + 64
cache = make_fixed_cache(model, max_len, kv_int8=True)
dev = "cuda"


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)


@torch.inference_mode()
def step_cost(k, reps=12):
    kv = [l for l in cache.layers if hasattr(l, "cur")]
    base = [l.cur for l in kv]
    toks = torch.randint(1000, 20000, (1, k), device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    out = None
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = model.model(input_ids=toks, position_ids=pos, past_key_values=cache,
                        use_cache=True).last_hidden_state
        out = model.lm_head(h).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        for layer, c in zip(kv, base):
            layer.cur = c
            layer._set_views()
    return statistics.median(ts) * 1e3, out


print(f"上下文 {n_ctx}，+1.0GB int8 方案（NF4 层走这套内核）", flush=True)
with torch.inference_mode():
    k_row_gemv(False)
    prefill()
    t1, _ = step_cost(1)
    t2, log2 = step_cost(2)
    t4, log4 = step_cost(4)
    k_row_gemv(True)
    prefill()
    t1p, _ = step_cost(1)
    t2p, log2p = step_cost(2)
    t4p, log4p = step_cost(4)

print(f"{'':<14}{'原路径':>10}{'k 行内核':>10}{'提升':>8}")
print(f"{'2 token':<14}{t2:>9.1f}ms{t2p:>9.1f}ms{t2 / t2p:>7.2f}x")
print(f"{'4 token':<14}{t4:>9.1f}ms{t4p:>9.1f}ms{t4 / t4p:>7.2f}x")
print(f"{'1 token（对照）':<14}{t1:>9.1f}ms{t1p:>9.1f}ms{t1 / t1p:>7.2f}x")


def cmp(a, b, tag):
    d = (a - b).abs().max().item()
    p, q = torch.softmax(a, -1), torch.softmax(b, -1)
    kl = (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1).mean().item()
    same = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"  2/4 行 logits {tag}：最大绝对差 {d:.2e}，KL {kl:.2e}，首选一致 {100 * same:.1f}%")


cmp(log2, log2p, "2 行")
cmp(log4, log4p, "4 行")
print(f"验证比例：2 token ×{t2 / t1:.2f} → ×{t2p / t1p:.2f}；4 token ×{t4 / t1:.2f} → ×{t4p / t1p:.2f}")
