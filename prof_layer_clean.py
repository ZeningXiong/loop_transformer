"""单层剖析（不带快照开销）：折叠注意力修好后，16K 下一个全注意力层还剩什么。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
REPS = 8
dev = "cuda"
tok, model = load("offline", int8=[])
tm = model.model.language_model
cache = make_fixed_cache(model, N_CTX + 128, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()

with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    x = torch.randn(1, 1, tm.config.hidden_size, device=dev, dtype=torch.bfloat16)
    pos = torch.tensor([[N_CTX]], device=dev)
    pe = tm.rotary_emb(x, pos.expand(3, -1, -1))
    layer = tm.layers[3]
    full = [i for i, l in enumerate(tm.layers) if hasattr(l, "self_attn")][0]
    lin = [i for i, l in enumerate(tm.layers) if hasattr(l, "linear_attn")][0]

    def call(layer, pos):
        return layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
                     past_key_values=cache, use_cache=True)

    for _ in range(3):
        call(layer, pos)
    torch.cuda.synchronize()

    import time
    t0 = time.perf_counter()
    for i in range(REPS):
        call(layer, torch.tensor([[N_CTX + i]], device=dev))
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / REPS * 1e3

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(REPS):
            call(layer, torch.tensor([[N_CTX + 100 + i]], device=dev))
        torch.cuda.synchronize()

rows = [(e.self_device_time_total / REPS / 1000.0, e.count / REPS, e.key) for e in prof.key_averages()]
rows.sort(reverse=True)
print(f"全注意力层 L{full}，16K，一次前向：墙钟 {wall:.2f} ms，kernel 合计 {sum(r[0] for r in rows):.2f} ms")
for ms, cnt, key in rows[:10]:
    print(f"{ms:7.3f} ms {cnt:5.1f}×  {key[:66]}")

with torch.inference_mode():
    for _ in range(3):
        call(tm.layers[lin], pos)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(REPS):
        call(tm.layers[lin], torch.tensor([[N_CTX + 200 + i]], device=dev))
    torch.cuda.synchronize()
    print(f"线性注意力层 L{lin}：墙钟 {(time.perf_counter() - t0) / REPS * 1e3:.2f} ms")
