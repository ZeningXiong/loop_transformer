"""图重放 162 ms 里到底是什么：对 replay 本身做 kernel 级剖析。"""
import collections
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_loop_common import load  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
REPS = 5
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
    dec.step(tk, N_CTX)                                   # 捕获
    bucket = dec.bucket_for(N_CTX + 2)
    g = dec.graphs[bucket]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(REPS):
            g.replay()
        torch.cuda.synchronize()

rows = [(e.self_device_time_total / REPS / 1000.0, e.count / REPS, e.key) for e in prof.key_averages()]
rows.sort(reverse=True)
total = sum(r[0] for r in rows)
print(f"上下文 {N_CTX}，一次图重放（2 token）：kernel 时间合计 {total:.1f} ms")
print(f"{'ms':>7} {'次数':>6}  算子")
for ms, cnt, key in rows[:16]:
    print(f"{ms:7.2f} {cnt:6.1f}  {key[:74]}")
