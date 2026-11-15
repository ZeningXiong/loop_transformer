"""自底向上拆账：每个部件在 T=1/2/4 下的耗时 × 出现次数，对上整模型 140 / 239 / 285 ms。

python bench_parts.py [上下文]
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
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
cache = make_fixed_cache(model, n_ctx + 40, kv_int8=True)
dev = "cuda"
tc = model.config.text_config
tm = model.model.language_model


def bench(fn, it=40, warm=8):
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
    return statistics.median(ts) * 1e6                     # µs


with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)

rows = []

# 1) 4-bit GEMM（244 次/步）：两种代表形状
for name in ("layers.2.mlp.gate_proj", "layers.2.linear_attn.in_proj_qkv"):
    mod = model.get_submodule("model.language_model." + name)
    k_in = mod.in_features
    costs = {}
    for T in (1, 2, 4):
        x = torch.randn(1, T, k_in, dtype=torch.bfloat16, device=dev) * 0.05
        with torch.inference_mode():
            costs[T] = bench(lambda: mod(x))
    rows.append((f"4-bit GEMM {name.split('.')[-1]}", costs, 1, "244 个线性层合计"))

# 2) 线性注意力的 GDN 内核（42 次/步）
la = tm.layers[2].linear_attn
hk, hv = la.head_k_dim, la.head_v_dim
nh = la.num_v_heads
st = torch.zeros(1, nh, hk, hv, dtype=torch.float32, device=dev)
costs_chunk, costs_rec = {}, {}
for T in (1, 2, 4):
    q = (torch.randn(1, T, nh, hk, device=dev) * 0.1).to(torch.bfloat16)
    kk = (torch.randn(1, T, nh, hk, device=dev) * 0.1).to(torch.bfloat16)
    v = (torch.randn(1, T, nh, hv, device=dev) * 0.1).to(torch.bfloat16)
    g = -torch.rand(1, T, nh, device=dev) * 0.1          # 模型里 g 是 float32
    beta = torch.rand(1, T, nh, device=dev).to(torch.bfloat16)
    with torch.inference_mode():
        costs_chunk[T] = bench(lambda: hf.torch_chunk_gated_delta_rule(
            q, kk, v, g=g, beta=beta, initial_state=st.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True))
        costs_rec[T] = bench(lambda: hf.torch_recurrent_gated_delta_rule(
            q, kk, v, g=g, beta=beta, initial_state=st.clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True))
rows.append(("GDN 分块（现状）", costs_chunk, 42, "42 次线性注意力合计"))
rows.append(("GDN 递推", costs_rec, 42, "（对照，未启用）"))

# 3) 卷积路径（42 次/步）
cw = la.conv1d.weight.squeeze(1)
cb = la.conv1d.bias
cstate = next(l.conv_states[0] for l in cache.layers
              if hasattr(l, "conv_states") and l.conv_states[0] is not None)
costs = {}
for T in (1, 2, 4):
    mixed = torch.randn(1, cw.shape[0], T, dtype=torch.bfloat16, device=dev) * 0.05
    if T == 1:
        with torch.inference_mode():
            costs[T] = bench(lambda: hf.causal_conv1d_update(mixed, cstate.clone(), cw, cb, "silu"))
    else:
        with torch.inference_mode():
            costs[T] = bench(lambda: hf.causal_conv1d_fn(mixed, cw, cb, activation="silu"))
rows.append(("卷积（T=1 单步 / T>1 整段）", costs, 42, "42 次线性注意力合计"))

# 4) mHC / Sinkhorn（96 个编译区/步）
router = None
for n, m in model.named_modules():
    if type(m).__name__ == "MHCRouter":
        router = m
        break
if router is not None:
    costs = {}
    try:
        for T in (1, 2, 4):
            x = torch.randn(1, T, 2 * tc.hidden_size, dtype=torch.bfloat16, device=dev) * 0.05
            with torch.inference_mode():
                costs[T] = bench(lambda: router(x), it=20)
        rows.append(("mHC 路由（Sinkhorn）", costs, 96, "96 个融合区合计"))
    except Exception as e:      # 形状假设不成立就跳过，别影响其它部件
        print(f"[跳过 mHC 路由] {type(e).__name__}: {str(e)[:80]}")

# 5) 全注意力（14 次/步，KV 长度 = 上下文）
costs = {}
for T in (1, 2, 4):
    q = torch.randn(1, tc.num_attention_heads, T, tc.head_dim, dtype=torch.bfloat16, device=dev)
    kk = torch.randn(1, tc.num_key_value_heads, n_ctx, tc.head_dim, dtype=torch.bfloat16, device=dev)
    v = torch.randn_like(kk)
    from cudagraph_decode import masked_fold_attention
    with torch.inference_mode():
        costs[T] = bench(lambda: masked_fold_attention(None, q, kk, v, None), it=20)
rows.append(("全注意力（KV=上下文长度）", costs, 14, "14 次全注意力合计"))

# 6) lm_head
costs = {}
for T in (1, 2, 4):
    x = torch.randn(1, T, tc.hidden_size, dtype=torch.bfloat16, device=dev) * 0.05
    with torch.inference_mode():
        costs[T] = bench(lambda: model.lm_head(x))
rows.append(("lm_head", costs, 1, ""))

print(f"上下文 {n_ctx}，单位 µs/次")
print(f"{'部件':<34}{'T=1':>9}{'T=2':>9}{'T=4':>9}{'×1→2':>8}{'×2→4':>8}")
tot = {1: 0.0, 2: 0.0, 4: 0.0}
for name, costs, cnt, _ in rows:
    print(f"{name:<34}{costs[1]:>9.0f}{costs[2]:>9.0f}{costs[4]:>9.0f}"
          f"{costs[2] / costs[1]:>7.2f}x{costs[4] / costs[2]:>7.2f}x")
    for T in (1, 2, 4):
        tot[T] += costs[T] * cnt
print(f"\n按次数加总（µs）：T=1 {tot[1]:.0f} | T=2 {tot[2]:.0f} | T=4 {tot[4]:.0f}")
print("这四个部件只覆盖了主要内核；元素级长尾（数千个小算子）没有单列。")
