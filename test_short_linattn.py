"""验证 + 测速：线性注意力短序列特化（以及去掉 small_batch_gemv 补丁）对 2/4 token 验证的影响。

python test_short_linattn.py [上下文]
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

from short_linattn import use_recurrent_for_short  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot, small_batch_gemv  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
cache = make_fixed_cache(model, n_ctx + 40, kv_int8=True)
dev = "cuda"
snap = _Snapshot(cache)
ORIG = bnb.nn.Linear4bit.forward


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)


@torch.inference_mode()
def measure(k, reps=10):
    toks = torch.randint(1000, 20000, (1, k), device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    ts, out = [], None
    for _ in range(reps):
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = model.model(input_ids=toks, position_ids=pos, past_key_values=cache,
                        use_cache=True).last_hidden_state
        out = model.lm_head(h).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3, out


def kl(a, b):
    p, q = torch.softmax(a, -1), torch.softmax(b, -1)
    return (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1).mean().item(), \
        (a.argmax(-1) == b.argmax(-1)).float().mean().item()


print(f"上下文 {n_ctx}", flush=True)
configs = []
# A: 现状（带 small_batch_gemv 补丁，走 chunk）
small_batch_gemv()
use_recurrent_for_short(False)
configs.append(("A 现状（补丁 + chunk）", None))

# B: 去掉补丁
prefill()
res = {}
for name, setup in (("B 无补丁 + chunk", lambda: bnb.nn.Linear4bit.forward == ORIG),):
    pass

# 依次测三组，每组前重置补丁/特化状态
def run(label, patch_gemv, short):
    bnb.nn.Linear4bit.forward = ORIG
    if patch_gemv:
        small_batch_gemv()
    use_recurrent_for_short(short)
    prefill()
    t1, _ = measure(1)
    prefill()
    t2, l2 = measure(2)
    prefill()
    t4, l4 = measure(4)
    print(f"{label:<26} 1 token {t1:6.1f} | 2 token {t2:6.1f} (x{t2 / t1:.2f}) | "
          f"4 token {t4:6.1f} (x{t4 / t1:.2f})", flush=True)
    return l2, l4


l2a, l4a = run("A 现状（补丁 + chunk）", True, False)
l2b, l4b = run("B 无补丁 + chunk", False, False)
l2c, l4c = run("C 无补丁 + 短序列递推", False, True)

for tag, x, y in (("B vs A", l2a, l2b), ("C vs A", l2a, l4c if False else l2c)):
    pass
print()
for tag, a, b in (("2 token: 特化 vs 现状", l2a, l2c), ("4 token: 特化 vs 现状", l4a, l4c)):
    k, t = kl(a, b)
    print(f"{tag:<24} KL {k:.2e}，首选一致 {100 * t:.1f}%")
use_recurrent_for_short(False)
bnb.nn.Linear4bit.forward = ORIG
