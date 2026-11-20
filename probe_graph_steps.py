"""逐帧对比：图 vs eager 在同样的 token 序列上，第几步开始分叉。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 64
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = "cuda"
tok, model = load("offline", int8=[])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()

dec = Graph2Decoder(model, N_CTX + 120, kv_int8=True, qlen=2, pass2_bits=4)
cache_e = make_fixed_cache(model, N_CTX + 120, kv_int8=True, pass2_bits=4)

with torch.inference_mode():
    dec.reset()
    reset_cache(cache_e)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        pos = torch.arange(s, e, device=dev)[None]
        h = model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=dec.cache,
                        use_cache=True).last_hidden_state
        model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=cache_e, use_cache=True)
    lg0 = model.lm_head(h[:, -1:]).float()[0, -1]
    x = lg0.argmax().view(1, 1)
    d = lg0.topk(2).indices[1].view(1, 1)
    bad = -1
    for i in range(STEPS):
        toks = torch.cat([x, d], 1)
        Lg = dec.step(toks, N_CTX + 2 * i)
        Le = model.lm_head(model.model(
            input_ids=toks,
            position_ids=torch.tensor([[N_CTX + 2 * i, N_CTX + 2 * i + 1]], device=dev),
            past_key_values=cache_e, use_cache=True).last_hidden_state).float()
        d0 = (Lg[0, 0] - Le[0, 0]).abs().max().item()
        d1 = (Lg[0, 1] - Le[0, 1]).abs().max().item()
        print(f"  步{i}: 第1位差 {d0:.3e} 第2位差 {d1:.3e} | 首选一致 "
              f"{Lg[0,0].argmax().item() == Le[0,0].argmax().item()}/"
              f"{Lg[0,1].argmax().item() == Le[0,1].argmax().item()}", flush=True)
        if d0 > 1e-2 or d1 > 1e-2:
            bad = i
            break
        dec.advance(2)
        nxt = Le[0, 1]
        x = nxt.argmax().view(1, 1)
        d = nxt.topk(2).indices[1].view(1, 1)
    print(f"上下文 {N_CTX}：{'全对' if bad < 0 else f'第 {bad} 步开始分叉'}")
