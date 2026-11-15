"""最小复现：FLA 的分块内核在 CUDA graph 捕获里能不能跑？"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf  # noqa: E402

tok, model = load("offline", int8=[])
la = model.model.language_model.layers[2].linear_attn
hk, hv, nh = la.head_k_dim, la.head_v_dim, la.num_v_heads
st = torch.zeros(1, nh, hk, hv, dtype=torch.float32, device="cuda")


def make(T):
    return dict(
        q=(torch.randn(1, T, nh, hk, device="cuda") * 0.1).to(torch.bfloat16),
        k=(torch.randn(1, T, nh, hk, device="cuda") * 0.1).to(torch.bfloat16),
        v=(torch.randn(1, T, nh, hv, device="cuda") * 0.1).to(torch.bfloat16),
        g=-torch.rand(1, T, nh, device="cuda") * 0.1,
        beta=torch.rand(1, T, nh, device="cuda").to(torch.bfloat16),
    )


for T, fn in ((1, hf.torch_recurrent_gated_delta_rule), (2, hf.torch_chunk_gated_delta_rule),
              (2, hf.torch_recurrent_gated_delta_rule)):
    a = make(T)
    with torch.inference_mode():
        try:
            fn(a["q"], a["k"], a["v"], g=a["g"], beta=a["beta"], initial_state=st.clone(),
               output_final_state=True, use_qk_l2norm_in_kernel=True)
            torch.cuda.synchronize()
            eager = "ok"
        except Exception as e:
            eager = f"{type(e).__name__}"
        try:
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                out = fn(a["q"], a["k"], a["v"], g=a["g"], beta=a["beta"], initial_state=st.clone(),
                         output_final_state=True, use_qk_l2norm_in_kernel=True)
                dummy = out[0].clone()
            g.replay()
            torch.cuda.synchronize()
            cap = "ok"
        except Exception as e:
            cap = f"{type(e).__name__}: {str(e)[:60]}"
    name = "recurrent" if fn is hf.torch_recurrent_gated_delta_rule else "chunk"
    print(f"T={T} {name:<10} eager {eager:<6} capture {cap}")
