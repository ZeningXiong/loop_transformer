"""VramGuard 本身会不会拖慢解码？对比：无 guard / 有 guard（后台轮询）/ 每步再查一次。"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

tok, model = load("offline", int8=[])
dev = "cuda"
n_ctx = 200
cache = make_fixed_cache(model, 256)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)


@torch.inference_mode()
def steps(n, guard=None, inloop=False):
    nxt = torch.tensor([[1000]], device=dev)
    ts = []
    for _ in range(n):
        if inloop and guard is not None:
            guard.check()
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = model.model(input_ids=nxt, position_ids=torch.tensor([[n_ctx]], device=dev),
                        past_key_values=cache, use_cache=True)
        model.lm_head(o.last_hidden_state[:, -1:]).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3


with torch.inference_mode():
    prefill()
    a = steps(10)                       # 无 guard
    g = VramGuard()                     # 后台线程每 20ms 查一次
    b = steps(10, g)                    # 有 guard，但循环里不查
    c = steps(10, g, inloop=True)       # 循环里每步再查一次
    g.close()
    d = steps(10)                       # 关掉 guard 之后
print(f"无 guard                ：{a:6.1f} ms/step")
print(f"有 guard（后台轮询）    ：{b:6.1f} ms/step   ×{b / a:.2f}")
print(f"有 guard + 每步 check   ：{c:6.1f} ms/step   ×{c / a:.2f}")
print(f"关掉 guard 之后         ：{d:6.1f} ms/step   ×{d / a:.2f}")
