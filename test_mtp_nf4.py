"""把 MTP 头量化成 NF4：省多少显存、16K 能不能放下、命中率和速度掉多少。

用法: MTP_NF4=1 python test_mtp_nf4.py [上下文] [生成 token 数]
"""
import json
import os
import sys
import time

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import VramGuard, load, make_fixed_cache  # noqa: E402
from sn_spec import MTPHead, generate_speculative  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 24
QUANT = os.environ.get("MTP_NF4") == "1"
ALL_NF4 = os.environ.get("ALL_NF4") == "1"

dev = "cuda"
plan = [] if ALL_NF4 else json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                        encoding="utf-8"))["int8"]
tok, model = load("offline", int8=plan)


def vram(tag):
    a = torch.cuda.memory_allocated() / 1e9
    free, total = torch.cuda.mem_get_info()
    print(f"  {tag:<26} 本进程 {a:5.2f} GB | 整卡已用 {(total - free) / 1e9:5.2f} GB", flush=True)
    return a


def quantize_mtp(mtp):
    """把 MTP 头里的 nn.Linear 换成 bnb 的 NF4 Linear4bit（原地替换）。"""
    n = 0
    for name, child in list(mtp.named_modules()):
        if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
            parent_name, _, attr = name.rpartition(".")
            parent = mtp.get_submodule(parent_name) if parent_name else mtp
            new = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                    bias=child.bias is not None, compute_dtype=torch.bfloat16,
                                    quant_type="nf4", compress_statistics=False,
                                    quant_storage=torch.uint8, device="meta")
            new.weight = bnb.nn.Params4bit(child.weight.data.detach(),
                                           requires_grad=False, quant_type="nf4",
                                           compress_statistics=False, quant_storage=torch.uint8)
            if child.bias is not None:
                new.bias = torch.nn.Parameter(child.bias.data.detach().clone(), requires_grad=False)
            setattr(parent, attr, new.to(dev))
            n += 1
    torch.cuda.empty_cache()
    gc.collect() if False else None
    print(f"  量化了 {n} 个线性层", flush=True)


import gc  # noqa: E402

print(f"配置：{'全 NF4 权重' if ALL_NF4 else '+1.0GB int8 权重'}，"
      f"MTP 头 {'NF4' if QUANT else 'bf16'}，上下文 {n_ctx}", flush=True)
vram("加载模型后")
mtp = MTPHead(model)
mtp.new_cache(n_ctx + n_gen + 32)
vram("MTP 头（bf16）就位")
if QUANT:
    quantize_mtp(mtp)
    vram("MTP 头量化成 NF4 后")

guard = VramGuard()
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()
cache = make_fixed_cache(model, n_ctx + n_gen + 32, kv_int8=True)
vram("KV 缓冲（16K int8）就位")

with torch.inference_mode():
    t0 = time.perf_counter()
    toks, st = generate_speculative(model, mtp, ids, n_gen, cache, chunk_size=256)
    dt = time.perf_counter() - t0
tps = st["tokens"] / st["decode_s"]
print(f"\n自投机：{st['tokens']} token / {st['decode_s']:.2f}s = **{tps:.2f} tok/s**；"
      f"步数 {st['steps']}，命中 {st['accepted']}，回退 {st['redo']}，普通补算 {st['plain_tokens']}")
print(f"命中率 {st['accepted'] / max(st['steps'], 1) * 100:.0f}% | 整卡峰值 {guard.peak / 1e9:.2f} GB")
guard.close()
