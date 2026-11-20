"""16K 下按 kernel 名归并的一步耗时，用来确认元素级长尾到底占多少、由谁构成。"""
import collections
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev = "cuda"
tok, model = load("offline", int8=[])
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
snap = _Snapshot(cache)

with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    tk = torch.tensor([[1000]], device=dev)
    pos = torch.tensor([[N_CTX]], device=dev)
    for _ in range(3):
        snap.save()
        model.model(input_ids=tk, position_ids=pos, past_key_values=cache, use_cache=True)
        snap.restore()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    snap.save()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(5):
            model.model(input_ids=tk, position_ids=pos, past_key_values=cache, use_cache=True)
    snap.restore()

evts = prof.key_averages()
rows = [(e.self_device_time_total / 5.0, e.count / 5.0, e.key) for e in evts]
rows.sort(reverse=True)
total = sum(r[0] for r in rows)
print(f"上下文 {N_CTX}，1-token 前向，CUDA 时间合计 {total:.1f} ms/步")
print(f"{'ms/步':>8} {'次数':>6}  算子")
for ms, cnt, key in rows[:18]:
    print(f"{ms:8.2f} {cnt:6.0f}  {key[:80]}")
