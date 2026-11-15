"""Check fla gated-delta-rule kernels against transformers' PyTorch reference (Qwen3.5 path)."""
import sys
import time

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as m

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
dev = "cuda"
B, T, H, D = 1, 200, 32, 128  # 32 value heads, head_dim 128 (model's linear attention)

q = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, T, H, D, device=dev, dtype=torch.bfloat16)
g = -torch.rand(B, T, H, device=dev, dtype=torch.float32)
beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16)

for name in ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"]:
    wrapped = getattr(m, f"torch_{name}")
    ref_fn = wrapped.__wrapped__
    print(f"{name}: 实际使用的实现 = {getattr(wrapped, '__module__', '?')} / is_fla={wrapped is not ref_fn}")

kw = dict(initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
ref_o, ref_s = m.torch_chunk_gated_delta_rule.__wrapped__(q, k, v, g, beta, **kw)

from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

t = time.time()
o, s = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kw)
torch.cuda.synchronize()
print(f"fla chunk 首次调用（含 Triton 编译）{time.time() - t:.1f} 秒")
print(f"chunk  输出最大误差 {(o.float() - ref_o.float()).abs().max().item():.4f}，"
      f"状态最大误差 {(s.float() - ref_s.float()).abs().max().item():.4f}，输出量级 {ref_o.float().abs().mean().item():.4f}")

# one-token decode step continuing from the state
q1, k1, v1, g1, b1 = (x[:, :1] for x in (q, k, v, g, beta))
ref_o1, _ = m.torch_recurrent_gated_delta_rule.__wrapped__(
    q1, k1, v1, g1, b1, initial_state=ref_s, output_final_state=True, use_qk_l2norm_in_kernel=True)
o1, _ = fused_recurrent_gated_delta_rule(
    q1, k1, v1, g=g1, beta=b1, initial_state=ref_s, output_final_state=True, use_qk_l2norm_in_kernel=True)
print(f"recurrent 输出最大误差 {(o1.float() - ref_o1.float()).abs().max().item():.4f}，"
      f"输出量级 {ref_o1.float().abs().mean().item():.4f}")
