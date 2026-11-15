"""Does the `small_batch_gemv` patch help or hurt? bnb 0.50.2's native multi-row path may already win.

python test_gemv_patch_ab.py [上下文]
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

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot, small_batch_gemv  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
cache = make_fixed_cache(model, n_ctx + 40, kv_int8=True)
dev = "cuda"


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)


@torch.inference_mode()
def step_cost(k, snap, reps=10):
    toks = torch.randint(1000, 20000, (1, k), device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    kv = [l for l in cache.layers if hasattr(l, "cur")]
    ts = []
    for _ in range(reps):
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = model.model(input_ids=toks, position_ids=pos, past_key_values=cache,
                        use_cache=True).last_hidden_state
        model.lm_head(h).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3


snap = _Snapshot(cache)
orig = bnb.nn.Linear4bit.forward
print(f"上下文 {n_ctx}", flush=True)

for label, apply in (("带 small_batch_gemv 补丁", True), ("无补丁（bnb 原生多行）", False)):
    if apply:
        small_batch_gemv()
    else:
        bnb.nn.Linear4bit.forward = orig
    prefill()
    t1 = step_cost(1, snap)
    prefill()
    t2 = step_cost(2, snap)
    prefill()
    t4 = step_cost(4, snap)
    print(f"{label:<24} 1 token {t1:6.1f} ms | 2 token {t2:6.1f} ms (x{t2 / t1:.2f}) | "
          f"4 token {t4:6.1f} ms (x{t4 / t1:.2f})", flush=True)
