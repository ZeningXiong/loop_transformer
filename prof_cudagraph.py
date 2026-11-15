"""Where do the 110 ms of a decode step actually go?  Kernel-level profile of one decode step.

python prof_cudagraph.py [n_ctx] [graph|base]
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_decode import GraphDecoder, use_masked_attention  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

PLAN = "mixed_plan_1.0.json"
CHUNK = 256
n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 200
mode = sys.argv[2] if len(sys.argv) > 2 else "base"

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))["int8"])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()


def prefill(cache):
    for s in range(0, ids.shape[1], CHUNK):
        e = min(s + CHUNK, ids.shape[1])
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device="cuda")[None],
                          past_key_values=cache, use_cache=True)
    return out.last_hidden_state


with torch.inference_mode():
    if mode == "base":
        cache = make_fixed_cache(model, 512)
        reset_cache(cache)
        h = prefill(cache)
        nxt = model.lm_head(h[:, -1:]).float().argmax(-1).reshape(1, 1)

        def step(i):
            hh = model.model(input_ids=nxt, position_ids=torch.tensor([[n_ctx + i]], device="cuda"),
                             past_key_values=cache, use_cache=True).last_hidden_state
            return model.lm_head(hh[:, -1:]).float()
    else:
        dec = GraphDecoder(model, 512)
        use_masked_attention(True)
        dec.reset()
        h = prefill(dec.cache)
        nxt = model.lm_head(h[:, -1:]).float().argmax(-1).reshape(1, 1)
        dec.step(nxt, n_ctx)            # capture
        dec.advance(1)

        def step(i):
            out = dec.step(nxt, n_ctx + 1 + i, logits_head=True).float()
            dec.advance(1)
            return out

    for i in range(4):                  # warm up
        step(i)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(5):
            step(4 + i)
        torch.cuda.synchronize()

rows = sorted(prof.key_averages(), key=lambda r: -r.self_device_time_total)[:22]
total = sum(r.self_device_time_total for r in prof.key_averages())
print(f"\n{mode}：一次 decode 步的 GPU 时间合计 {total / 5 / 1e3:.1f} ms（5 步平均）")
print(f"{'kernel':<62}{'ms/step':>9}{'占比':>8}{'次数/步':>9}")
for r in rows:
    print(f"{r.key[:60]:<62}{r.self_device_time_total / 5 / 1e3:>9.3f}"
          f"{100 * r.self_device_time_total / total:>7.1f}%{r.count / 5:>9.0f}")
