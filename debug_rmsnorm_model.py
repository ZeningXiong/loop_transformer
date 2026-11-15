"""定位：融合 RMSNorm 在模型里和模块级单测结果不一致的原因。

做法：启用融合后跑一次前向，钩住每个 RMSNorm 记下它的输入和输出；然后用原始实现重算同一批输入，
逐个比对。谁不一致，就打印它的形状/步幅/权重 dtype。
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
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf  # noqa: E402

ORIG = hf.Qwen3_5RMSNorm.forward
dev = "cuda"
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
n_ctx = 200
cache = make_fixed_cache(model, 256)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
recs = []


def hook(name):
    def f(mod, inp, out):
        recs.append((name, inp[0].detach().clone(), out.detach().clone(), mod))
    return f


def install(_on):
    for h in getattr(model, "_rn_handles", []):
        h.remove()
    model._rn_handles = []
    if _on:
        for n, m in model.named_modules():
            if isinstance(m, hf.Qwen3_5RMSNorm):
                model._rn_handles.append(m.register_forward_hook(hook(n)))


with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    snap.save()
    fused_rmsnorm.enable(True)
    install(True)
    toks = torch.tensor([[1000]], device=dev)
    model.model(input_ids=toks, position_ids=torch.tensor([[n_ctx]], device=dev),
                past_key_values=cache, use_cache=True)
    install(False)
    fused_rmsnorm.enable(False)
    snap.restore()

bad = 0
for name, x, y, mod in recs:
    with torch.inference_mode():
        ref = ORIG(mod, x)
    neq = (ref.view(torch.int16) != y.view(torch.int16)).sum().item()
    if neq:
        bad += 1
        if bad <= 6:
            print(f"[不一致] {name}")
            print(f"    x.shape {tuple(x.shape)} contiguous={x.is_contiguous()} "
                  f"stride={x.stride()} dtype={x.dtype}")
            print(f"    weight {tuple(mod.weight.shape)} {mod.weight.dtype}；eps {mod.eps}")
            print(f"    {neq}/{ref.numel()} 个元素不同，最大差 "
                  f"{(ref.float() - y.float()).abs().max().item():.3e}")
print(f"共 {len(recs)} 次 RMSNorm 调用，其中 {bad} 次和原始实现不一致")
