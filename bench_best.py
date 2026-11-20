"""当前最快组合的实测：16K 上下文下 纯贪心 / 投机 的速度与一致性。

同一进程、同一份缓存，先跑纯贪心作为参照，再跑生产投机（强制开启），逐 token 比对。
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

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, generate_speculative  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
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
guard = VramGuard()
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()


@torch.inference_mode()
def plain(n=N_GEN):
    reset_cache(cache)
    h = None
    t0 = time.perf_counter()
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    torch.cuda.synchronize()
    pre = time.perf_counter() - t0
    x = model.lm_head(h[:, -1:]).float()[0, -1].argmax().view(1, 1)
    out, pos = [], ids.shape[1]
    t1 = time.perf_counter()
    for _ in range(n):
        out.append(x.item())
        h = model.model(input_ids=x, position_ids=torch.tensor([[pos]], device=dev),
                        past_key_values=cache, use_cache=True).last_hidden_state
        x = model.lm_head(h).float()[0, -1].argmax().view(1, 1)
        pos += 1
    torch.cuda.synchronize()
    return out, pre, time.perf_counter() - t1


@torch.inference_mode()
def spec(n=N_GEN):
    reset_cache(cache)
    mtp.cache.reset()
    out, st = generate_speculative(model, mtp, ids, n, cache, chunk_size=256, adaptive=True)
    return out, st["prefill_s"], st["decode_s"], st


print(f"16K（{ids.shape[1]} token 上下文），生成 {N_GEN} token\n")
toks_a, pre_a, dec_a = plain()
print(f"纯贪心   prefill {pre_a:5.1f}s | 解码 {dec_a:5.2f}s | {N_GEN / dec_a:5.2f} tok/s")
guard.check()
toks_b, pre_b, dec_b, st = spec()
print(f"投机     prefill {pre_b:5.1f}s | 解码 {dec_b:5.2f}s | {N_GEN / dec_b:5.2f} tok/s "
      f"| 命中 {st['accepted']}/{st['steps']} | 整卡 {guard.peak / 1e9:.2f} GB")
n = next((i for i, (p, q) in enumerate(zip(toks_a, toks_b)) if p != q), min(len(toks_a), len(toks_b)))
print(f"\n与贪心相同前缀：{n}/{N_GEN}")
if n < N_GEN:
    print("  贪心:", toks_a[max(0, n - 3):n + 4])
    print("  投机:", toks_b[max(0, n - 3):n + 4])
guard.close()
