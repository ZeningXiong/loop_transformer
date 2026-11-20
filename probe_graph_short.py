"""短上下文图 bug：第一帧就比，图 step 的 logits vs eager 同状态前向。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 64
dev = "cuda"
tok, model = load("offline", int8=[])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()

dec = Graph2Decoder(model, N_CTX + 80, kv_int8=True, qlen=2, pass2_bits=4)
with torch.inference_mode():
    dec.reset()
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=dec.cache, use_cache=True).last_hidden_state
    x = model.lm_head(h[:, -1:]).float()[0, -1].argmax().view(1, 1)
    d = model.lm_head(h[:, -1:]).float()[0, -1].topk(2).indices[1].view(1, 1)
    toks = torch.cat([x, d], 1)

    snap = dec._snap()
    snap.save()
    Lg = dec.step(toks, N_CTX)                 # 图
    snap.restore()

    # eager 同状态
    cache_e = make_fixed_cache(model, N_CTX + 80, kv_int8=True, pass2_bits=4)
    reset_cache(cache_e)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache_e, use_cache=True)
    he = model.model(input_ids=toks, position_ids=torch.tensor([[N_CTX, N_CTX + 1]], device=dev),
                     past_key_values=cache_e, use_cache=True).last_hidden_state
    Le = model.lm_head(he).float()

    print(f"上下文 {N_CTX}：")
    print(f"  第1位 logits 最大差 {(Lg[0,0]-Le[0,0]).abs().max().item():.3e} | 首选一致 "
          f"{Lg[0,0].argmax().item() == Le[0,0].argmax().item()}")
    print(f"  第2位 logits 最大差 {(Lg[0,1]-Le[0,1]).abs().max().item():.3e} | 首选一致 "
          f"{Lg[0,1].argmax().item() == Le[0,1].argmax().item()}")
    print(f"  图 pad={dec.bucket_for(N_CTX + 2)}，cur={N_CTX}")
    lay = dec.cache._full[0]
    print(f"  第一层 keys 视图长度 {lay.keys.shape[2]}，pad={lay.pad}，cur={lay.cur}")
