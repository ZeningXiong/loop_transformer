"""线性注意力层里那 13 次 copy_ 到底在拷什么（带 shape）。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
REPS = 6
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
    lin = [i for i, l in enumerate(tm.layers) if hasattr(l, "linear_attn")][0]

    def call(i):
        return tm.layers[lin](x, position_embeddings=pe, attention_mask=None,
                              position_ids=torch.tensor([[N_CTX + i]], device=dev),
                              past_key_values=cache, use_cache=True)

    for i in range(3):
        call(i)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for i in range(REPS):
            call(10 + i)
        torch.cuda.synchronize()

print(f"线性注意力层 L{lin}，16K，copy_ 明细：")
for e in prof.key_averages(group_by_input_shape=True):
    if "copy_" in e.key or e.key in ("aten::cat", "aten::conv1d", "aten::unfold"):
        print(f"{e.self_device_time_total / REPS / 1000.0:7.3f} ms {e.count / REPS:5.1f}×  "
              f"{e.key[:28]}  {e.input_shapes}")
