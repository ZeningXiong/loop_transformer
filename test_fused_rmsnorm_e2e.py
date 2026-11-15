"""融合 RMSNorm 的端到端一致性：先测"同一实现跑两次"的噪声，再比融合版。

如果"原实现 vs 原实现"本身就不是逐位相同，那么之前的 KL 差异来自模型/内核的非确定性，
而不是融合。
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

import fused_rmsnorm  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"
n_ctx = 200
cache = make_fixed_cache(model, 256)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
toks = torch.tensor([[1000, 2000]], device=dev)
pos = torch.tensor([[n_ctx]], device=dev)


def kl(a, b):
    p, q = torch.softmax(a, -1), torch.softmax(b, -1)
    return (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1).mean().item()


@torch.inference_mode()
def one():
    snap.save()
    h = model.model(input_ids=toks, position_ids=pos, past_key_values=cache,
                    use_cache=True).last_hidden_state
    out = model.lm_head(h).float()
    snap.restore()
    return out


with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    snap.save()
    a = one()
    b = one()
    fused_rmsnorm.enable(True)
    c = one()
    d = one()
    fused_rmsnorm.enable(False)

bit = lambda x, y: (x.view(torch.int32) == y.view(torch.int32)).all().item()  # noqa: E731
print("同一实现（原版）跑两次： 逐位相同", bit(a, b), f" KL {kl(a, b):.3e}")
print("融合版跑两次：           逐位相同", bit(c, d), f" KL {kl(c, d):.3e}")
print("原版 vs 融合版：         逐位相同", bit(a, c), f" KL {kl(a, c):.3e}")
