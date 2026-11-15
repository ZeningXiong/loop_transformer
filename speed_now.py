"""当前配置的实测速度：全 NF4 权重 + NF4 MTP 头 + KV int8 + sinkhorn 闭式解，16K。

单一进程、单一模型、单一缓存；先测普通解码作参照，再测 2-token 自投机。
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

import fast_sinkhorn  # noqa: E402
from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, generate_speculative  # noqa: E402
from test_mtp_nf4 import quantize_mtp  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
n_plain, n_spec = 16, 24
dev = "cuda"

tok, model = load("offline", int8=[])          # 全 NF4；load_offline 里默认启用闭式解 sinkhorn
mod = sys.modules[type(model.model.language_model).__module__]
print(f"sinkhorn 实现：{'闭式解' if mod.sinkhorn.__name__ == 'dispatch' else mod.sinkhorn.__name__}")

mtp = MTPHead(model)
max_len = n_ctx + 64
mtp.new_cache(max_len)
quantize_mtp(mtp)
guard = VramGuard()
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()
cache = make_fixed_cache(model, max_len, kv_int8=True)


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    mtp.cache.reset()
    out = None
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                          past_key_values=cache, use_cache=True)
    return out.last_hidden_state


with torch.inference_mode():
    h = prefill()
    nxt = model.lm_head(h[:, -1:]).float().argmax(-1).reshape(1, 1)
    times = []
    for i in range(n_plain):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = model.model(input_ids=nxt, position_ids=torch.tensor([[n_ctx + i]], device=dev),
                        past_key_values=cache, use_cache=True)
        nxt = model.lm_head(o.last_hidden_state[:, -1:]).float().argmax(-1).reshape(1, 1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times) * 1e3
    print(f"普通解码：{med:.1f} ms/token = {1e3 / med:.2f} tok/s", flush=True)
    guard.check()

    toks, st = generate_speculative(model, mtp, ids, n_spec, cache, chunk_size=256)
tps = st["tokens"] / st["decode_s"]
print(f"2-token 自投机：{st['tokens']} token / {st['decode_s']:.2f}s = **{tps:.2f} tok/s**；"
      f"命中 {st['accepted']}/{st['steps']}，回退 {st['redo']}，普通补算 {st['plain_tokens']}")
print(f"相对普通解码 ×{tps / (1e3 / med):.2f} | 整卡峰值 {guard.peak / 1e9:.2f} GB")
guard.close()
