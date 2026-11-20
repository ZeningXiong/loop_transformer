"""1-token 图重放 vs 2-token 图重放：失配时改成重做 1 个 token 到底能省多少。"""
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
REPS = 20
dev = "cuda"
tok, model = load("offline", int8=[])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()


def bench(fn, n=REPS, warm=3):
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


for qlen in (2, 1):
    dec = Graph2Decoder(model, N_CTX + 80, kv_int8=True, qlen=qlen, pass2_bits=4)
    with torch.inference_mode():
        dec.reset()
        for s in range(0, N_CTX, 256):
            e = min(s + 256, N_CTX)
            model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=dec.cache, use_cache=True)
        tk = torch.tensor([[1000, 2000][:qlen]], device=dev)
        dec.step(tk, N_CTX)                      # 捕获
        g = dec.graphs[dec.bucket_for(N_CTX + qlen)]
        rep = bench(lambda: g.replay())
        step = bench(lambda: dec.step(tk, N_CTX))
        print(f"qlen={qlen}: 纯重放 {rep:6.1f} ms | 完整 step {step:6.1f} ms", flush=True)
    del dec
    torch.cuda.empty_cache()
