"""2-token 验证前向 vs 两次 1-token 前向：logits 是否一致（投机正确性的根）。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 512
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
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    snap.save()
    l0 = model.lm_head(h[:, -1:]).float()[0, -1]
    x = l0.argmax().view(1, 1)
    d = torch.tensor([[l0.topk(2).indices[1].item()]], device=dev)   # 故意选第二个候选当草稿

    pos = N_CTX
    # A: 一次喂两个 token
    ha = model.model(input_ids=torch.cat([x, d], 1),
                     position_ids=torch.tensor([[pos, pos + 1]], device=dev),
                     past_key_values=cache, use_cache=True).last_hidden_state
    la = model.lm_head(ha).float()
    snap.restore()

    # B: 先喂 x，再喂 d
    h1 = model.model(input_ids=x, position_ids=torch.tensor([[pos]], device=dev),
                     past_key_values=cache, use_cache=True).last_hidden_state
    lb1 = model.lm_head(h1).float()[0, -1]
    h2 = model.model(input_ids=d, position_ids=torch.tensor([[pos + 1]], device=dev),
                     past_key_values=cache, use_cache=True).last_hidden_state
    lb2 = model.lm_head(h2).float()[0, -1]
    snap.restore()

    da = (la[0, 0] - lb1).abs().max().item()
    db = (la[0, 1] - lb2).abs().max().item()
    print(f"第 1 个位置 logits 最大差 {da:.3e}，首选一致 {la[0, 0].argmax().item() == lb1.argmax().item()}")
    print(f"第 2 个位置 logits 最大差 {db:.3e}，首选一致 {la[0, 1].argmax().item() == lb2.argmax().item()}")
    print(f"hidden 第 2 行差 {(ha[0, 1] - h2[0, -1]).abs().max().item():.3e}")
