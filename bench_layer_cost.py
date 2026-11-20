"""单层耗时：16K 下一次全注意力层 / 一次线性注意力层各多少毫秒，加起来能不能解释 162 ms。"""
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
from sn_spec import _Snapshot  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev = "cuda"
tok, model = load("offline", int8=[])
tm = model.model.language_model
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
snap = _Snapshot(cache)


def bench(fn, n=10, warm=3):
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


with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    x = torch.randn(1, 1, tm.config.hidden_size, device=dev, dtype=torch.bfloat16)
    pos = torch.tensor([[N_CTX]], device=dev)
    pe = tm.rotary_emb(x, pos.expand(3, -1, -1))
    snap.save()

    def make_call(i, layer):
        def f():
            snap.save()
            layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
                  past_key_values=cache, use_cache=True)
            snap.restore()
        return f

    full = [i for i, l in enumerate(tm.layers) if hasattr(l, "self_attn")]
    lin = [i for i, l in enumerate(tm.layers) if hasattr(l, "linear_attn")]
    bench(make_call(full[0], tm.layers[full[0]]), 3, 1)     # 预热
    t_full = bench(make_call(full[0], tm.layers[full[0]]))
    t_lin = bench(make_call(lin[0], tm.layers[lin[0]]))
    print(f"上下文 {N_CTX}，单层一次前向（1 token）：")
    print(f"  全注意力层（L{full[0]}）   {t_full:6.2f} ms   × 14 次执行 = {t_full * 14:6.1f} ms")
    print(f"  线性注意力层（L{lin[0]}）  {t_lin:6.2f} ms   × 42 次执行 = {t_lin * 42:6.1f} ms")
    print(f"  合计估算                                   {t_full * 14 + t_lin * 42:6.1f} ms")
