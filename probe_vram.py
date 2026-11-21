"""模型自己占多少显存：区分"本进程峰值"与"整卡（含其他程序）"。"""
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    used = int(out.split("MiB")[0].strip())
    return used


N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
print(f"启动前整卡已用（桌面/浏览器等）：{smi()} MiB", flush=True)

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache, use_fused_kv_attention  # noqa: E402
from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_spec import MTPHead, _Snapshot  # noqa: E402

import bitsandbytes as bnb  # noqa: E402

use_fused_kv_attention(True)
tok, model = load("offline", int8=[])
print(f"加载权重后：整卡 {smi()} MiB，torch 已分配 {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

mtp = MTPHead(model)
mtp.new_cache(N_CTX + 80)
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
        setattr(parent, attr, new.to("cuda"))
torch.cuda.empty_cache()

dec = Graph2Decoder(model, N_CTX + 80, kv_int8=True, qlen=2, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()

with torch.inference_mode():
    dec.reset()
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device="cuda")[None],
                        past_key_values=dec.cache, use_cache=True).last_hidden_state
    snap = dec._snap()
    for _ in range(4):
        snap.save()
        dec.step(torch.tensor([[1000, 2000]], device="cuda"), N_CTX)
        dec.advance(2)
    torch.cuda.synchronize()

alloc = torch.cuda.max_memory_allocated() / 2**30
res = torch.cuda.max_memory_reserved() / 2**30
used = smi()
print(f"上下文 {N_CTX}：")
print(f"  本进程 torch 峰值分配  {alloc:.2f} GiB")
print(f"  本进程缓存池（reserved）{res:.2f} GiB   ← 模型在这张卡上实际占的量")
print(f"  整卡已用               {used / 1024:.2f} GiB")
print(f"  其他程序（桌面/浏览器等）{used / 1024 - res:.2f} GiB")
