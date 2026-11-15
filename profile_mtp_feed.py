"""MTP 回填为什么 21 ms：把 1 行前向拆到子算子级别，并和 bf16 头对比。"""
import json
import os
import statistics
import sys
import time

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, get_tap  # noqa: E402

n_ctx = 16384
QUANT = os.environ.get("MTP_NF4", "1") == "1"
dev = "cuda"
tok, model = load("offline", int8=[])
tap = get_tap(model)
mtp = MTPHead(model)
mtp.new_cache(n_ctx + 64)
if QUANT:
    for name, child in list(mtp.named_modules()):
        if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
            pn, _, attr = name.rpartition(".")
            parent = mtp.get_submodule(pn) if pn else mtp
            new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=False,
                                    compute_dtype=torch.bfloat16, quant_type="nf4",
                                    compress_statistics=False, quant_storage=torch.uint8,
                                    device="meta")
            new.weight = bnb.nn.Params4bit(child.weight.data.detach(), requires_grad=False,
                                           quant_type="nf4", compress_statistics=False,
                                           quant_storage=torch.uint8)
            setattr(parent, attr, new.to(dev))
cache = make_fixed_cache(model, n_ctx + 64, kv_int8=True)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()


def bench(fn, n=20, warm=5):
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
    mtp.cache.reset()
    T = ids.shape[1]
    prev_h = None
    for s in range(0, T, 256):
        e = min(s + 256, T)
        pos = torch.arange(s, e, device=dev)[None]
        model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=cache, use_cache=True)
        h = tap.post
        if prev_h is not None:
            mtp(prev_h, ids[:, s:s + 1], pos[:, :1], logits=False)
        if e - s > 1:
            mtp(h[:, :-1], ids[:, s + 1:e], pos[:, 1:], logits=False)
        prev_h = h[:, -1:]
    print(f"MTP 头：{'NF4' if QUANT else 'bf16'}，缓存已填到 {mtp.cache.layer.cur}", flush=True)

    tok1 = torch.tensor([[1000]], device=dev)
    pos1 = torch.tensor([[n_ctx]], device=dev)
    hidden = prev_h

    def full():                       # 完整的一次回填（等价 draft_step(logits=False)）
        mtp(hidden, tok1, pos1, logits=False)

    e1 = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(tok1))
    e2 = mtp.pre_fc_norm_hidden(hidden)
    cat = torch.cat([e1, e2] if mtp.embed_first else [e2, e1], dim=-1)

    def fc_only():
        mtp.fc(cat)

    x = mtp.fc(cat)
    pe = mtp.tm.rotary_emb(x, pos1[None].expand(3, -1, -1))

    def rot_only():
        mtp.tm.rotary_emb(x, pos1[None].expand(3, -1, -1))

    def layer_only():
        mtp.layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos1,
                  past_key_values=mtp.cache, use_cache=True)

    # 只做 QKV 投影 + 注意力（手工走一遍 attention 子模块）
    att = mtp.layer.self_attn

    def attn_only():
        h2 = att.input_layernorm(x) if hasattr(att, "input_layernorm") else x
        q = att.q_proj(h2)
        k = att.k_proj(h2)
        v = att.v_proj(h2)
        return q, k, v

    t_all = bench(full)
    t_fc = bench(fc_only)
    t_rot = bench(rot_only)
    t_layer = bench(layer_only)
    t_qkv = bench(attn_only)
    t_ln = bench(lambda: mtp.pre_fc_norm_hidden(hidden))
    t_emb = bench(lambda: mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(tok1)))

print(f"\n完整回填一次          {t_all:6.2f} ms")
print(f"  ├ pre_fc_norm_embed {t_emb:6.2f} ms")
print(f"  ├ pre_fc_norm_hidden{t_ln:6.2f} ms")
print(f"  ├ fc（8192→4096）   {t_fc:6.2f} ms")
print(f"  ├ rotary            {t_rot:6.2f} ms")
print(f"  ├ 整层（含注意力）  {t_layer:6.2f} ms")
print(f"  └ q/k/v 三个投影    {t_qkv:6.2f} ms")
print(f"  其余（cat/launch 等）{t_all - t_fc - t_rot - t_layer:6.2f} ms")
