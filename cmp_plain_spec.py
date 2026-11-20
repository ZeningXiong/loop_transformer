"""纯贪心 vs 生产投机（generate_speculative）：同一 prompt 下生成的 token 是否一致。"""
import os
import sys

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, generate_speculative  # noqa: E402

N_CTX, N_GEN = 16384, 24
dev = "cuda"
tok, model = load("offline", int8=[])
mtp = MTPHead(model)
mtp.new_cache(N_CTX + 64)
for name, child in list(mtp.named_modules()):
    if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
        pn, _, attr = name.rpartition(".")
        parent = mtp.get_submodule(pn) if pn else mtp
        new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=False,
                                compute_dtype=torch.bfloat16, quant_type="nf4",
                                compress_statistics=False, quant_storage=torch.uint8, device="meta")
        new.weight = bnb.nn.Params4bit(child.weight.data.detach(), requires_grad=False,
                                       quant_type="nf4", compress_statistics=False,
                                       quant_storage=torch.uint8)
        setattr(parent, attr, new.to(dev))
torch.cuda.empty_cache()
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)

msgs = [{"role": "user", "content": "用一句话说明 CUDA graph 是做什么的"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()


torch.inference_mode().__enter__()


def plain(ids, n=N_GEN):
    reset_cache(cache)
    h = None
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    out, pos = [], ids.shape[1]
    x = model.lm_head(h[:, -1:]).float()[0, -1].argmax().view(1, 1)
    for _ in range(n):
        out.append(x.item())
        h = model.model(input_ids=x, position_ids=torch.tensor([[pos]], device=dev),
                        past_key_values=cache, use_cache=True).last_hidden_state
        x = model.lm_head(h).float()[0, -1].argmax().view(1, 1)
        pos += 1
    return out


a = plain(ids)
reset_cache(cache)
mtp.cache.reset()
b, st = generate_speculative(model, mtp, ids, N_GEN, cache, chunk_size=256, adaptive=False)
n = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), min(len(a), len(b)))
print(f"相同前缀 {n}/{min(len(a), len(b))}")
print("贪心:", a)
print("投机:", b)
print("贪心文本:", tok.decode(a))
print("投机文本:", tok.decode(b))
print("统计:", st)
