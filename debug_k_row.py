"""Debug: for each patched 4-bit module, compare bnb's forward with the k-row kernel on real inputs."""
import json
import os
import sys

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from nf4_gemv_k import nf4_gemv_k  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
CODE = bnb.functional.get_4bit_type("nf4", device="cuda")
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :512].cuda()
cache = make_fixed_cache(model, 640)
dev = "cuda"

captured = []
mods = [(n, m) for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)][:4]

with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, 512, 256):
        model.model(input_ids=ids[:, s:s + 256], position_ids=torch.arange(s, s + 256, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    hooks = [m.register_forward_hook(lambda _m, inp, out, n=n: captured.append((n, inp[0].detach())))
             for n, m in mods]
    toks = torch.tensor([[1000, 2000]], device=dev)
    model.model(input_ids=toks, position_ids=torch.tensor([[512]], device=dev),
                past_key_values=cache, use_cache=True)
    for h in hooks:
        h.remove()

print(f"捕获 {len(captured)} 个模块的输入")
for (n, x), (_, mod) in zip(captured, mods):
    rows = x.numel() // x.shape[-1]
    qs = mod.weight.quant_state
    m, k = qs.shape
    packed = mod.weight.data.reshape(m, k // 2)
    absmax = qs.absmax.reshape(m, k // qs.blocksize)
    with torch.inference_mode():
        ref = mod(x)
        mine = nf4_gemv_k(x.reshape(rows, k), packed, absmax, CODE,
                          blocksize=qs.blocksize).reshape(*x.shape[:-1], m)
        w = bnb.functional.dequantize_4bit(mod.weight.data, qs).to(torch.float32)
        exact = x.reshape(rows, k).float() @ w.T
    d_ref = ((ref.float() - exact).norm() / exact.norm()).item()
    d_mine = ((mine.float() - exact).norm() / exact.norm()).item()
    print(f"{n[:52]:<54} rows={rows} qs={tuple(qs.shape)} packed={tuple(mod.weight.data.shape)} "
          f"| bnb 误差 {d_ref:.2e} | mine 误差 {d_mine:.2e}")
