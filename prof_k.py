"""Where does the extra time of a 2-token (and 4-token) forward come from? Profile 1 vs 2 vs 4 and diff.

python prof_k.py [上下文]
"""
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
cache = make_fixed_cache(model, n_ctx + 40, kv_int8=True)
dev = "cuda"
kv = [l for l in cache.layers if hasattr(l, "cur")]


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)


@torch.inference_mode()
def profile(k, reps=4):
    from torch.profiler import ProfilerActivity, profile as tprof

    toks = torch.randint(1000, 20000, (1, k), device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    base = [l.cur for l in kv]

    def one():
        model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True)
        for l, c in zip(kv, base):
            l.cur = c
            l._set_views()

    for _ in range(3):
        one()
    torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            one()
        torch.cuda.synchronize()
    return {r.key: r.self_device_time_total / reps / 1e3 for r in p.key_averages()}


prefill()
t = {}
for k in (1, 2, 4):
    t[k] = profile(k)
print(f"上下文 {n_ctx}：每个前向的 GPU 时间（µs，profiler 下）")
tot = {k: sum(v.values()) for k, v in t.items()}
print(f"  合计：1 token {tot[1]:.0f} µs | 2 token {tot[2]:.0f} µs | 4 token {tot[4]:.0f} µs")

keys = set(t[1]) | set(t[2]) | set(t[4])
rows = []
for key in keys:
    d2 = t[2].get(key, 0) - t[1].get(key, 0)
    d4 = t[4].get(key, 0) - t[2].get(key, 0)
    rows.append((d2, d4, key, t[1].get(key, 0), t[2].get(key, 0), t[4].get(key, 0)))
rows.sort(key=lambda r: -r[0])
print(f"\n{'kernel':<52}{'1 token':>9}{'2 token':>9}{'4 token':>9}{'Δ(1→2)':>9}{'Δ(2→4)':>9}")
for d2, d4, key, a, b, c in rows[:16]:
    print(f"{key[:50]:<52}{a:>9.0f}{b:>9.0f}{c:>9.0f}{d2:>+9.0f}{d4:>+9.0f}")

groups = {"4bit GEMM": r"gemm_4bit|dequant|quantize_4bit", "Sinkhorn/编译区": r"Torch-Compiled|sinkhorn",
          "FLA/线性注意力": r"fla|chunk|delta|wy_fast|gated|recurrent", "卷积": r"conv",
          "注意力": r"attention|softmax|sdpa|flash"}
print()
for g, pat in groups.items():
    s = {k: sum(v for key, v in t[k].items() if re.search(pat, key, re.I)) for k in (1, 2, 4)}
    print(f"{g:<14} 1 token {s[1]:7.0f} µs | 2 token {s[2]:7.0f} µs | 4 token {s[4]:7.0f} µs")
