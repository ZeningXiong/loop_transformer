"""确认：闭式解是精确极限（20 轮迭代在大 logits 下还没收敛），并检查端到端输出差异。"""
import json
import os
import sys

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
orig = mod.sinkhorn

with torch.inference_mode():
    print("收敛性检查（同一批 logits，比较 20 轮 / 200 轮 / 闭式解）")
    for scale in (0.1, 1.0, 5.0, 20.0):
        z = torch.randn(4096, 2, 2, device=dev) * scale
        d20 = (orig(z, 20) - sinkhorn_2x2(z, 20)).abs().max().item()
        d200 = (orig(z, 200) - sinkhorn_2x2(z, 200)).abs().max().item()
        print(f"  logits×{scale:<5}: 20 轮对闭式 {d20:.2e} | 200 轮对闭式 {d200:.2e}")

    # 端到端：2 个 token 的前向，原始 vs 闭式
    n_ctx = 200
    cache = make_fixed_cache(model, 256)
    snap = _Snapshot(cache)
    texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
    ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
    reset_cache(cache)
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    snap.save()
    toks = torch.tensor([[1000, 2000]], device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)

    def fwd():
        h = model.model(input_ids=toks, position_ids=pos, past_key_values=cache,
                        use_cache=True).last_hidden_state
        return model.lm_head(h).float()

    snap.save()
    a = fwd()
    snap.restore()
    enable(True)
    snap.save()
    b = fwd()
    snap.restore()
    enable(False)
    mod.sinkhorn = orig

p, q = torch.softmax(a, -1), torch.softmax(b, -1)
kl = (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1).mean().item()
same = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
print(f"\n2 token 前向：KL {kl:.2e}，最大绝对差 {(a - b).abs().max().item():.2e}，"
      f"首选一致 {100 * same:.1f}%")
