"""拆 Graph2Decoder.step：embedding 读、缓冲拷贝、掩码填充、索引填充、重放、lm_head 各多少。"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_loop_common import load  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev = "cuda"
tok, model = load("offline", int8=[])
dec = Graph2Decoder(model, N_CTX + 80, kv_int8=True, qlen=2, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()

with torch.inference_mode():
    dec.reset()
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=dec.cache, use_cache=True)
    tk = torch.tensor([[1000, 2000]], device=dev)
    buf = dec._buffers(dec.bucket_for(N_CTX + 2))
    dec.step(tk, N_CTX)                          # 触发捕获
    g = dec.graphs[dec.bucket_for(N_CTX + 2)]
    cur = N_CTX


torch.inference_mode().__enter__()  # 后续所有调用都在推理模式下


def bench(fn, n=20, warm=5):
    with torch.inference_mode():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def emb_only():
    dec.emb(tk)


def emb_copy():
    buf["emb"].copy_(dec.emb(tk))


def mask_fill():
    buf["mask"].fill_(False)
    for i in range(2):
        buf["mask"][0, 0, i, :cur + i + 1] = True


def idx_fill():
    dec.cache.idx.fill_(cur)
    for l in dec.cache._full:
        l.idx2.copy_(torch.arange(cur, cur + 2, device=dev))


def replay():
    g.replay()


def head():
    model.lm_head(buf["out"])


parts = [("embedding（SSD 读 2 行）", emb_only), ("embedding + 写入缓冲", emb_copy),
         ("掩码填充（16K 布尔）", mask_fill), ("idx 填充（14 层）", idx_fill),
         ("图重放", replay), ("lm_head（2 行 × 248k）", head)]
res = {}
for name, fn in parts:
    res[name] = bench(fn)
    print(f"{name:<24}{res[name]:7.2f} ms", flush=True)
print(f"\n合计 {sum(res.values()):.1f} ms")
