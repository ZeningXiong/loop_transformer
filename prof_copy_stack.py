"""打印那个 [1,4,4,16385,256] 拷贝的 Python 调用栈。"""
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
REPS = 4
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
    layer = tm.layers[3]

    def call():
        snap.save()
        layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
              past_key_values=cache, use_cache=True)
        snap.restore()

    for _ in range(2):
        call()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        for _ in range(REPS):
            call()
        torch.cuda.synchronize()

for e in prof.key_averages(group_by_input_shape=True, group_by_stack_n=6):
    if "copy_" not in e.key:
        continue
    shapes = str(e.input_shapes)
    if "16385" not in shapes:
        continue
    print(f"{e.self_device_time_total / REPS / 1000.0:.3f} ms  {e.key}  {shapes}")
    for s in (e.stack or [])[:8]:
        print("    ", s.replace("\n", " ")[:160])
