"""消融：mHC 路由在真实一步里到底占多少。同一缓存、同一状态，只切换代码路径。

A 基线 → B 只换掉 sinkhorn（隔离 torch.compile 的每次调用开销）→ C 整个 router 换成常量。
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

n_ctx = 200
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"
mod = sys.modules[type(model).__module__]
MHCRouter = mod.MHCRouter
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
    ts = []
    snap.save()
    for _ in range(5):
        model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True)
        snap.restore()
    torch.cuda.synchronize()
    for _ in range(reps):
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3


orig_sinkhorn, orig_forward = mod.sinkhorn, MHCRouter.forward
with torch.inference_mode():
    prefill()
    snap.save()
    a = step_time()

    # B：sinkhorn 换成一个廉价的确定性函数（同样形状，不做 20 轮迭代、不经 torch.compile）
    def cheap_sinkhorn(logits, iterations):
        s = logits.shape[-1]
        return torch.eye(s, device=logits.device, dtype=logits.dtype).expand(*logits.shape[:-2], s, s).contiguous()

    mod.sinkhorn = cheap_sinkhorn
    snap.restore()
    b = step_time()

    # C：整个 router 换成常量（去掉 norm + fp32 GEMV + sigmoid + matmul）
    def stub_forward(self, residual):
        streams = self.streams
        shape = residual.shape[:-2]
        post = torch.ones(*shape, streams, device=residual.device, dtype=residual.dtype)
        mix = torch.eye(streams, device=residual.device, dtype=torch.float32).expand(
            *shape, streams, streams).contiguous()
        branch = residual.mean(-2)
        return branch, post, mix

    MHCRouter.forward = stub_forward
    snap.restore()
    c = step_time()

    MHCRouter.forward = orig_forward
    mod.sinkhorn = orig_sinkhorn

print(f"一步 decode（200 token 上下文，1 token）")
print(f"  A 基线                    {a:7.1f} ms")
print(f"  B 换掉 sinkhorn           {b:7.1f} ms   → sinkhorn 占 {a - b:5.1f} ms")
print(f"  C 整个 router 换常量      {c:7.1f} ms   → 路由整体占 {a - c:5.1f} ms"
      f"（其中非 sinkhorn 部分 {b - c:.1f} ms）")
