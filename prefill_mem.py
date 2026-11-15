"""Where does prefill memory go? Peak extra memory per module type at several prompt lengths."""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sn_loop_common import load

# which fused attention kernels does this Windows torch build have?
q = torch.randn(1, 16, 1024, 256, device="cuda", dtype=torch.bfloat16)
for b in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
    try:
        with sdpa_kernel(b): torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        print(f"SDPA 后端 {b.name}: 可用")
    except Exception as e:
        print(f"SDPA 后端 {b.name}: 不可用（{str(e).splitlines()[0][:90]}）")

tok, model = load("offline")
print("注意力实现:", model.config.text_config._attn_implementation if hasattr(model.config, "text_config") else "?")
stats = collections.defaultdict(float)
def pre(name):
    def f(_m, _i):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        f.base = torch.cuda.memory_allocated()
    return f
def post(name, p):
    def f(_m, _i, _o):
        torch.cuda.synchronize()
        stats[name] = max(stats[name], (torch.cuda.max_memory_allocated() - p.base) / 1e9)
    return f
for lname, layer in enumerate(model.model.language_model.layers):
    for attr, kind in [("self_attn", "全注意力"), ("linear_attn", "线性注意力"), ("mlp", "MLP"), ("mhc_attn", "mHC路由")]:
        m = getattr(layer, attr, None)
        if m is not None:
            p = pre(kind); m.register_forward_pre_hook(p); m.register_forward_hook(post(kind, p))
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 3, return_tensors="pt").input_ids
for T in [512, 1024, 2048]:
    stats.clear()
    ids = ids_all[:, :T].to("cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    whole = (torch.cuda.max_memory_allocated() - base) / 1e9
    cache = sum(t.numel() * t.element_size() for t in _ts(out.past_key_values)) / 1e9 if (_ts := (lambda c: [x for x in torch.utils._pytree.tree_flatten(c.__dict__ if hasattr(c, "__dict__") else c)[0] if torch.is_tensor(x)])) else 0
    del out
    print(f"T={T:5d}: 整个 prompt 处理的显存增量峰值 {whole:5.2f} GB | 其中保存下来的 cache {cache:5.2f} GB | 各模块单次调用的临时峰值: "
          + "，".join(f"{k} {v:.2f}" for k, v in sorted(stats.items(), key=lambda x: -x[1])), flush=True)
