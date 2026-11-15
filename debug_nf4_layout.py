"""Find out how bitsandbytes packs NF4: nibble order, row layout, absmax mapping."""
import os
import sys

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
from sn_loop_common import load  # noqa: E402

tok, model = load("offline", int8=[])
mod = model.get_submodule("model.language_model.layers.7.self_attn").q_proj
qs = mod.quant_state
n, k = qs.shape
bs = qs.blocksize
code = bnb.functional.get_4bit_type("nf4", device="cuda")
with torch.inference_mode():
    w = bnb.functional.dequantize_4bit(mod.weight.data, qs).to(torch.float32).reshape(n, k)
packed = mod.weight.data.reshape(n, k // 2)
absmax = qs.absmax.reshape(n, k // bs).float()

row = 0
b = packed[row]                                    # k/2 bytes
w_row = w[row]
am = absmax[row].repeat_interleave(bs)             # [k] scale per element

for name, lo_first in (("low nibble first", True), ("high nibble first", False)):
    lo = (b & 0x0F).long()
    hi = (b >> 4).long()
    if lo_first:
        ev, od = code[lo], code[hi]
    else:
        ev, od = code[hi], code[lo]
    rec = torch.empty(k, device="cuda")
    rec[0::2] = ev * am[0::2]
    rec[1::2] = od * am[1::2]
    print(f"{name:<18} 相对误差 {(rec - w_row).norm() / w_row.norm():.4f}")

# also try: absmax applied per 64 with the *pairs* in the other order
rec2 = torch.empty(k, device="cuda")
e = code[(b & 0x0F).long()]
o = code[(b >> 4).long()]
rec2[0::2] = e * am[0::2]
rec2[1::2] = o * am[1::2]
print("逐元素重建（码值 × 该元素所在块的 absmax）相对误差:",
      ((rec2 - w_row).norm() / w_row.norm()).item())
print("首 8 个: bnb", [round(v, 5) for v in w_row[:8].tolist()])
print("         mine", [round(v, 5) for v in rec2[:8].tolist()])
print("packed 首 8 字节:", [hex(v) for v in b[:8].tolist()], " absmax 首 2 块:", absmax[row][:2].tolist())
