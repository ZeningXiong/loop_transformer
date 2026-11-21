"""验证 MTP 图：同一状态下，图草稿的 token 必须与 eager 一致；再测两者耗时。"""
import os
import statistics
import sys
import time

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from mtp_graph import MTPGraph  # noqa: E402
from sn_loop_common import load  # noqa: E402
from sn_spec import MTPHead  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
dev = "cuda"
tok, model = load("offline", int8=[])
mtp = MTPHead(model)
mtp.new_cache(N + 64)
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

D = mtp.tc.hidden_size
hid = torch.randn(1, 1, D, device=dev, dtype=torch.bfloat16)
toks = torch.tensor([[1000]], device=dev)
pos = torch.tensor([[N]], device=dev)

hd = mtp.layer.__class__  # 仅为占位
mg = None


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


with torch.inference_mode():
    # eager 参考
    mtp.cache.layer.cur = N
    lg = mtp(hid, toks, pos)
    a = lg[0, -1].argmax().item()
    t_e_draft = bench(lambda: mtp(hid, toks, pos))
    t_e_feed = bench(lambda: mtp(hid, toks, pos, logits=False))
    mtp.cache.layer.cur = N

    mg = MTPGraph(mtp, N + 64)
    mg.capture(N)
    mg.layer.cur = N
    b = int(mg.draft(hid, toks, N).item())
    t_g_draft = bench(lambda: mg.draft(hid, toks, N))
    t_g_feed = bench(lambda: mg.feed(hid, toks, N))

print(f"上下文 {N}：")
print(f"  草稿 eager {t_e_draft:5.2f} ms | 图 {t_g_draft:5.2f} ms | token {a} vs {b} 一致 {a == b}")
print(f"  回填 eager {t_e_feed:5.2f} ms | 图 {t_g_feed:5.2f} ms")
print(f"  每步合计 eager {t_e_draft + t_e_feed:.2f} → 图 {t_g_draft + t_g_feed:.2f} ms")
