"""端到端：第 2 遍槽位用 int4（其余 int8），16K 下的显存与投机速度。

一个进程只跑一个配置（避免两个 KV 缓存同时驻留）：
  python test_kv_int4.py 8   # 全部 int8
  python test_kv_int4.py 4   # 第 2 遍 int4
"""
import json
import os
import sys

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import VramGuard, load, make_fixed_cache  # noqa: E402
from sn_spec import MTPHead, generate_speculative  # noqa: E402

P2 = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N_CTX, N_SPEC = 16384, 24
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
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
guard = VramGuard()
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=P2)
print(f"第2遍 {P2}-bit | KV {cache._fixed_bytes / 2**20:.1f} MB", flush=True)
with torch.inference_mode():
    toks, st = generate_speculative(model, mtp, ids, N_SPEC, cache, chunk_size=256)
tps = st["tokens"] / st["decode_s"]
print(f"prefill {st['prefill_s']:.1f}s | 投机 {tps:.2f} tok/s（命中 {st['accepted']}/{st['steps']}）"
      f" | 每步 {st['decode_s'] / st['steps'] * 1e3:.0f} ms | 整卡峰值 {guard.peak / 1e9:.2f} GB",
      flush=True)
guard.close()
