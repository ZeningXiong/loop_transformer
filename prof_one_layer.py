"""单个全注意力层（16K）的 kernel 级 profile：把那 2.8 ms 找出来。"""
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
REPS = 10
dev = "cuda"
tok, model = load("offline", int8=[])
tm = model.model.language_model
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
    x = torch.randn(1, 1, tm.config.hidden_size, device=dev, dtype=torch.bfloat16)
    pos = torch.tensor([[N_CTX]], device=dev)
    pe = tm.rotary_emb(x, pos.expand(3, -1, -1))
    layer = tm.layers[3]                       # 全注意力层

    def call():
        snap.save()
        layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
              past_key_values=cache, use_cache=True)
        snap.restore()

    for _ in range(3):
        call()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(REPS):
            call()
        torch.cuda.synchronize()

rows = [(e.self_device_time_total / REPS / 1000.0, e.count / REPS, e.key) for e in prof.key_averages()]
rows.sort(reverse=True)
total = sum(r[0] for r in rows)
print(f"全注意力层 L3，16K，一次前向：kernel 合计 {total:.2f} ms")
for ms, cnt, key in rows[:14]:
    print(f"{ms:7.3f} ms {cnt:5.1f}×  {key[:72]}")
