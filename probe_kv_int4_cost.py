"""干净的单步计时：每档重建缓存、按真实位置递增、每次调用都同步。

只比较"一次前向"里 KV 写入 + 反量化的差别，不跑投机、不跑 MTP。
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

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

dev = "cuda"
N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
tok, model = load("offline", int8=[])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()


@torch.inference_mode()
def one(p2bits, k, reps=10):
    cache = make_fixed_cache(model, N_CTX + 128, kv_int8=True, pass2_bits=p2bits)
    reset_cache(cache)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    tk = torch.tensor([[1000] * k], device=dev)
    cur = N_CTX

    def fwd(i):
        return model.model(input_ids=tk, position_ids=torch.tensor([[cur + i]], device=dev),
                           past_key_values=cache, use_cache=True)

    for i in range(3):                                   # 预热
        fwd(i)
    torch.cuda.synchronize()
    ts = []
    for i in range(3, 3 + reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd(i)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    kv = cache._fixed_bytes / 2**20
    del cache
    torch.cuda.empty_cache()
    return statistics.median(ts) * 1e3, kv


with torch.inference_mode():
    for label, p2 in (("全部 int8", 8), ("第2遍 int4", 4)):
        for k in (1, 2):
            t, kv = one(p2, k)
            print(f"{label:<10} k={k} | KV {kv:6.1f} MB | 前向 {t:6.1f} ms", flush=True)
