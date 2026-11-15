"""验证融合 RMSNorm：先逐位比对，再测单算子与整步耗时。"""
import json
import os
import statistics
import sys
import time

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
tm = model.model.language_model

# ---------- 1. 单元验证：真实模块 + 真实形状，逐位比对
print("逐位比对（真实模块的 weight / eps）")
norms = [m for m in model.modules() if isinstance(m, hf.Qwen3_5RMSNorm)]
dims = sorted({tuple(m.weight.shape) for m in norms})
print(f"  模型里 RMSNorm 共 {len(norms)} 个，权重形状 {dims}")

worst_bit = 0
total = 0
for m in norms[:6]:
    d = m.weight.shape[0]
    for T in (1, 2):
        x = (torch.randn(1, T, d, device=dev) * 2.0).to(torch.bfloat16)
        with torch.inference_mode():
            ref = ORIG(m, x)
            got = fused_rmsnorm.rmsnorm(x, m.weight, m.eps)
        neq = (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
        total += ref.numel()
        worst_bit = max(worst_bit, neq)
        if neq:
            print(f"    D={d} T={T} 首层：{neq}/{ref.numel()} 个元素不同"
                  f"，最大绝对差 {(ref.float() - got.float()).abs().max().item():.3e}")
print(f"  合计 {total} 个元素，最多一层有 {worst_bit} 个不同"
      f"（{'逐位相同 ✓' if worst_bit == 0 else '有差异 ✗'}）")

# ---------- 2. 单算子耗时
m = norms[0]
x = (torch.randn(1, 1, m.weight.shape[0], device=dev) * 2.0).to(torch.bfloat16)


def bench(fn, it=300, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


with torch.inference_mode():
    t_old = bench(lambda: ORIG(m, x))
    t_new = bench(lambda: fused_rmsnorm.rmsnorm(x, m.weight, m.eps))
print(f"\n单算子（D={m.weight.shape[0]}，T=1）：PyTorch {t_old:.1f} µs → 融合 {t_new:.1f} µs"
      f"（×{t_old / t_new:.2f}，省 {t_old - t_new:.1f} µs/次）")

# ---------- 3. 整步：替换后看耗时与输出是否变化
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


def step(reps=15):
    toks = torch.tensor([[1000, 2000]], device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    snap.save()
    for _ in range(3):
        model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True)
        snap.restore()
    torch.cuda.synchronize()
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


with torch.inference_mode():
    prefill()
    snap.save()
    t_ref, log_ref = step()
    fused_rmsnorm.enable(True)
    snap.restore()
    t_fused, log_fused = step()
    fused_rmsnorm.enable(False)
    snap.restore()
    t_back, _ = step()

p, q = torch.softmax(log_ref, -1), torch.softmax(log_fused, -1)
kl = (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1).mean().item()
same = (log_ref.view(torch.int32) == log_fused.view(torch.int32)).all().item()
print(f"\n一次 2-token 前向（200 token 上下文）")
print(f"  PyTorch   {t_ref:6.1f} ms")
print(f"  融合      {t_fused:6.1f} ms  → {'省' if t_ref > t_fused else '慢'} "
      f"{abs(t_ref - t_fused):.1f} ms")
print(f"  换回      {t_back:6.1f} ms")
print(f"  输出：logits 逐位相同 {same}，KL {kl:.2e}")
