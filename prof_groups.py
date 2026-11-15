"""Which kernels make up one decode step? Print every kernel, grouped, to find where FLA/conv time goes."""
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                 encoding="utf-8"))["int8"])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :200].cuda()
cache = make_fixed_cache(model, 512)

with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device="cuda")[None],
                          past_key_values=cache, use_cache=True)
    nxt = model.lm_head(out.last_hidden_state[:, -1:]).float().argmax(-1).reshape(1, 1)
    for i in range(4):
        hh = model.model(input_ids=nxt, position_ids=torch.tensor([[200 + i]], device="cuda"),
                         past_key_values=cache, use_cache=True).last_hidden_state
        nxt = model.lm_head(hh[:, -1:]).float().argmax(-1).reshape(1, 1)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(4, 8):
            hh = model.model(input_ids=nxt, position_ids=torch.tensor([[200 + i]], device="cuda"),
                             past_key_values=cache, use_cache=True).last_hidden_state
            nxt = model.lm_head(hh[:, -1:]).float().argmax(-1).reshape(1, 1)
        torch.cuda.synchronize()

rows = prof.key_averages()
tot = sum(r.self_device_time_total for r in rows)
groups = {"FLA / 线性注意力": r"fla|chunk|delta|wy_fast|recurrent|gated|solve_tril|fwd_prepare",
          "卷积": r"conv", "4bit GEMM": r"gemm_4bit|dequant|quantize_4bit",
          "注意力/softmax": r"attention|softmax|sdpa|flash", "Sinkhorn/编译区": r"Torch-Compiled|sinkhorn"}
seen, bucket = set(), {k: [0.0, 0] for k in groups}
print(f"\n一步 decode 的 GPU 时间合计 {tot / 4 / 1e3:.1f} ms（4 步平均）\n")
print(f"{'kernel':<58}{'ms/步':>9}{'占比':>8}{'次数/步':>8}")
for r in sorted(rows, key=lambda r: -r.self_device_time_total):
    if r.self_device_time_total / 4 / 1e3 < 0.15:
        continue
    for g, pat in groups.items():
        if re.search(pat, r.key, re.I):
            bucket[g][0] += r.self_device_time_total / 4 / 1e3
            bucket[g][1] += r.count / 4
            break
    print(f"{r.key[:56]:<58}{r.self_device_time_total / 4 / 1e3:>9.2f}"
          f"{100 * r.self_device_time_total / tot:>7.1f}%{r.count / 4:>8.0f}")
print()
for g, (ms, n) in bucket.items():
    print(f"{g:<16}{ms:>8.2f} ms/步{100 * ms * 4 / tot:>7.1f}%{n:>8.0f} 次/步")
