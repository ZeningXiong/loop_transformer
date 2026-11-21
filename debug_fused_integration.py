"""定位集成 bug：同一状态下，第 3 层自注意力的输出（融合 vs 原路径）差多少。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import (load, make_fixed_cache, reset_cache,  # noqa: E402
                            use_fused_kv_attention)
from sn_spec import _Snapshot  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
dev = "cuda"
tok, model = load("offline", int8=[])
tm = model.model.language_model
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
snap = _Snapshot(cache)
cap = {}


def hook(name):
    def f(mod, inp, out):
        cap[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    return f


with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    l3 = tm.layers[3].self_attn
    h3 = l3.register_forward_hook(hook("attn3"))
    x = model.lm_head(h[:, -1:]).float()[0, -1].argmax().view(1, 1)
    pos = torch.tensor([[N_CTX]], device=dev)

    snap.save()
    use_fused_kv_attention(False)
    la = model.model(input_ids=x, position_ids=pos, past_key_values=cache,
                     use_cache=True).last_hidden_state
    a = cap["attn3"].clone()
    snap.restore()

    use_fused_kv_attention(True)
    lb = model.model(input_ids=x, position_ids=pos, past_key_values=cache,
                     use_cache=True).last_hidden_state
    b = cap["attn3"].clone()
    snap.restore()
    h3.remove()
    use_fused_kv_attention(False)

print(f"上下文 {N_CTX}")
print(f"  第3层注意力输出：最大差 {(a.float() - b.float()).abs().max().item():.3e}，"
      f"相对 {(a.float() - b.float()).abs().mean().item() / a.float().abs().mean().item():.3e}")
print(f"  最终 hidden 最大差 {(la.float() - lb.float()).abs().max().item():.3e}")
print(f"  logits 首选一致 "
      f"{model.lm_head(la).float().argmax().item() == model.lm_head(lb).float().argmax().item()}")
