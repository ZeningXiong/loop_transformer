"""这个模型到底占多少显存：分阶段打印 已分配 / 保留 / 整卡，并扣除其他程序。

用法: python probe_footprint.py [上下文] [pass2位数]
"""
import json
import os
import sys

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache  # noqa: E402
from sn_spec import MTPHead  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
P2 = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dev = "cuda"


def card():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


base = card()
print(f"启动时整卡已用 {base:.2f} GB（其他程序 + CUDA 上下文）", flush=True)


def show(tag):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    c = card()
    print(f"{tag:<28} 本进程分配 {a:5.2f} | 保留 {r:5.2f} | 整卡 {c:5.2f} | "
          f"扣掉启动占用后 {c - base:5.2f} GB", flush=True)


tok, model = load("offline", int8=[])
show("加载全 NF4 权重后")
mtp = MTPHead(model)
mtp.new_cache(N_CTX + 64)
show("MTP 头（bf16）就位")
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
show("MTP 头量化成 NF4 后")
torch.cuda.empty_cache()
show("empty_cache 之后")
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=P2)
print(f"缓存里 KV 字节数 {cache._fixed_bytes / 2**20:.1f} MB + scratch {cache._scratch_bytes / 2**20:.1f} MB",
      flush=True)
show("KV 缓存就位")
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
with torch.inference_mode():
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
show(f"prefill {N_CTX} 后（峰值）")
print(f"本进程峰值（allocated） {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
