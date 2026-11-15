"""验证 2x2 Sinkhorn 闭式解，并测它在真实一步里能省多少。"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from fast_sinkhorn import enable, sinkhorn_2x2  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"
mod = sys.modules[type(model).__module__]
orig_sinkhorn = mod.sinkhorn            # 已被 patch_sinkhorn 编译过

# ---------- 1. 数值：随机 logits（含大幅值）对比 20 轮迭代
with torch.inference_mode():
    worst = 0.0
    for scale in (0.1, 1.0, 5.0, 20.0):
        z = torch.randn(4096, 2, 2, device=dev) * scale
        ref = orig_sinkhorn(z, 20)
        fast = sinkhorn_2x2(z, 20)
        worst = max(worst, (ref - fast).abs().max().item())
    print(f"闭式解 vs 20 轮迭代：最大绝对差 {worst:.2e}（logits 幅度 0.1–20）")
    z = torch.randn(4, 2, 2, device=dev)
    print("  示例 logits:\n", z[0].tolist())
    print("  ref :", [[round(v, 6) for v in row] for row in orig_sinkhorn(z, 20)[0].tolist()])
    print("  fast:", [[round(v, 6) for v in row] for row in sinkhorn_2x2(z, 20)[0].tolist()])

# ---------- 2. 一步 decode 的耗时（同缓存同状态，只换 sinkhorn 实现）
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


def step_time(reps=15):
    toks = torch.tensor([[1000]], device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    snap.save()
    for _ in range(5):
        model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True)
        snap.restore()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3


with torch.inference_mode():
    prefill()
    snap.save()
    t_compiled = step_time()
    enable(True)
    snap.restore()
    t_fast = step_time()
    t_compiled2 = step_time() if False else None
    enable(False)
    mod.sinkhorn = orig_sinkhorn
    snap.restore()
    t_back = step_time()

print(f"\n一步 decode（200 token 上下文、1 token）")
print(f"  现状（20 轮 + torch.compile） {t_compiled:7.1f} ms")
print(f"  闭式解                         {t_fast:7.1f} ms   → 省 {t_compiled - t_fast:.1f} ms"
      f"（{(t_compiled - t_fast) / t_compiled * 100:.0f}%）")
print(f"  再换回现状（确认无漂移）       {t_back:7.1f} ms")
