"""把生产的 int8 KV 路径接进 Graph2Decoder，端到端测 2-token 投机。

对照：同一份模型、同一段 16K 上下文
  eager  : 生产路径 make_fixed_cache(int8) + generate_speculative
  graph  : Graph2Decoder(int8, qlen=2) 做验证前向，投机逻辑与 eager 相同
每步固定提交 2 个 token（失配时提交模型的纠正 token），因此输出应与贪心解码逐字一致。
"""
import json
import os
import sys
import time

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, get_tap, generate_speculative  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
N_GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
dev = "cuda"
tok, model = load("offline", int8=[])
mtp = MTPHead(model)
mtp.new_cache(N_CTX + 80)
for name, child in list(mtp.named_modules()):
    if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
        pn, _, attr = name.rpartition(".")
        parent = mtp.get_submodule(pn) if pn else mtp
        new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=False,
                                compute_dtype=torch.bfloat16, quant_type="nf4",
                                compress_statistics=False, quant_storage=torch.uint8, device="meta")
        new.weight = bnb.nn.Params4bit(child.weight.data.detach(), requires_grad=False,
                                       quant_type="nf4", compress_statistics=False,
                                       quant_storage=torch.uint8)
        setattr(parent, attr, new.to(dev))
torch.cuda.empty_cache()
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
guard = VramGuard()


@torch.inference_mode()
def draft(hidden, token, pos):
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(token))
    h = mtp.pre_fc_norm_hidden(hidden)
    x = mtp.fc(torch.cat([e, h] if mtp.embed_first else [h, e], dim=-1))
    pe = mtp.tm.rotary_emb(x, torch.tensor([[pos]], device=dev)[None].expand(3, -1, -1))
    x = mtp.layer(x, position_embeddings=pe, attention_mask=None,
                  position_ids=torch.tensor([[pos]], device=dev),
                  past_key_values=mtp.cache, use_cache=True)
    return mtp.lm_head(mtp.norm(x))[0, -1], x


@torch.inference_mode()
def feed(hidden, token, pos):
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(token))
    h = mtp.pre_fc_norm_hidden(hidden)
    x = mtp.fc(torch.cat([e, h] if mtp.embed_first else [h, e], dim=-1))
    pe = mtp.tm.rotary_emb(x, torch.tensor([[pos]], device=dev)[None].expand(3, -1, -1))
    mtp.layer(x, position_embeddings=pe, attention_mask=None,
              position_ids=torch.tensor([[pos]], device=dev),
              past_key_values=mtp.cache, use_cache=True)


@torch.inference_mode()
def feed2(h_prev, h0, t0, t1, pos0):
    """一次处理两个 token 的 MTP 回填（原来要调用两次）。"""
    hidden = torch.cat([h_prev, h0], dim=1)
    tokens = torch.cat([t0, t1], dim=1)
    pos = torch.arange(pos0, pos0 + 2, device=dev)[None]
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(tokens))
    h = mtp.pre_fc_norm_hidden(hidden)
    x = mtp.fc(torch.cat([e, h] if mtp.embed_first else [h, e], dim=-1))
    pe = mtp.tm.rotary_emb(x, pos[None].expand(3, -1, -1))
    mtp.layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
              past_key_values=mtp.cache, use_cache=True)


@torch.inference_mode()
def run_graph():
    import sn_loop_common as _slc

    _slc.use_fused_kv_attention(os.environ.get("SENSENOVA_FUSED") == "1")
    mg = None
    if os.environ.get("SENSENOVA_MTP_GRAPH") == "1":
        from mtp_graph import MTPGraph

        mg = MTPGraph(mtp, N_CTX + 80)
        mg.open_prefill()
    dec = Graph2Decoder(model, N_CTX + 80, kv_int8=True, qlen=2, pass2_bits=4)
    dec.reset()
    mtp.cache.reset()
    prev_h = None
    t0 = time.perf_counter()
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        pos = torch.arange(s, e, device=dev)[None]
        out = model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=dec.cache,
                          use_cache=True).last_hidden_state
        if prev_h is not None:
            mtp(prev_h, ids[:, s:s + 1], torch.tensor([[s]], device=dev), logits=False)
        if e - s > 1:
            mtp(out[:, :-1], ids[:, s + 1:e], torch.arange(s + 1, e, device=dev)[None], logits=False)
        prev_h = out[:, -1:]
    torch.cuda.synchronize()
    pre = time.perf_counter() - t0

    if mg is not None:
        mg.capture(N_CTX)
    h_last = prev_h
    lg = model.lm_head(h_last).float()
    x = lg[0, -1].argmax().view(1, 1)
    out_toks, steps, acc = [], 0, 0
    snap = dec._snap()
    seg = dict(snap=0.0, draft=0.0, step=0.0, head=0.0, feed=0.0)
    td = time.perf_counter()
    while len(out_toks) < N_GEN:
        t0 = time.perf_counter()
        snap.save()
        base = mtp.cache.layer.cur
        t1 = time.perf_counter()
        if mg is not None:
            d = mg.draft(h_last, x, N_CTX + steps * 2).view(1, 1)
        else:
            dlg, _ = draft(h_last, x, N_CTX)
            d = dlg.argmax().view(1, 1)
        toks2 = torch.cat([x, d], 1)
        t2 = time.perf_counter()
        L = dec.step(toks2, N_CTX + steps * 2)
        H = dec.bufs[dec.bucket_for(N_CTX + steps * 2 + 2)]["out"].clone()
        p0 = L[0, 0].argmax().view(1, 1)
        if d.item() == p0.item():
            commit, nxt, hidden0, hidden1 = [x, d], L[0, 1].argmax().view(1, 1), H[:, 0:1], H[:, 1:2]
            acc += 1
        else:
            snap.restore()
            L2 = dec.step(torch.cat([x, p0], 1), N_CTX + steps * 2)
            H2 = dec.bufs[dec.bucket_for(N_CTX + steps * 2 + 2)]["out"].clone()
            commit, nxt, hidden0, hidden1 = [x, p0], L2[0, 1].argmax().view(1, 1), H2[:, 0:1], H2[:, 1:2]
        t3 = time.perf_counter()
        dec.advance(2)
        out_toks.extend(t.item() for t in commit)
        steps += 1
        # 保持 MTP 缓存同步：喂 (隐藏, token) 两对
        mtp.cache.layer.cur = base
        mtp.cache.layer._set_views()
        feed2(h_last, hidden0, commit[0], commit[1], N_CTX + (steps - 1) * 2)
        x, h_last = nxt, hidden1
        t4 = time.perf_counter()
        guard.check()
        seg["snap"] += t1 - t0
        seg["draft"] += t2 - t1
        seg["step"] += t3 - t2
        seg["feed"] += t4 - t3
    torch.cuda.synchronize()
    dec_s = time.perf_counter() - td
    cap_s = sum(dec.capture_s.values())
    steady = (dec_s - cap_s) / max(steps, 1) * 1e3
    print("  分段（ms/步）：" + "  ".join(f"{k} {v / max(steps, 1) * 1e3:.1f}" for k, v in seg.items()),
          flush=True)
    del dec
    torch.cuda.empty_cache()
    return out_toks[:N_GEN], dict(prefill_s=pre, decode_s=dec_s, steps=steps, acc=acc,
                                  cap_s=cap_s, steady_ms=steady)


@torch.inference_mode()
def run_eager():
    cache = make_fixed_cache(model, N_CTX + 80, kv_int8=True, pass2_bits=4)
    toks, st = generate_speculative(model, mtp, ids, N_GEN, cache, chunk_size=256)
    del cache
    torch.cuda.empty_cache()
    return toks, st


print(f"16K，{N_GEN} token，全 NF4 + NF4 MTP + int8 KV")
with torch.inference_mode():
    tg, st_g = run_graph()
    guard.check()
    print(f"[graph] prefill {st_g['prefill_s']:.1f}s | 捕获 {st_g['cap_s']:.1f}s | "
          f"稳态 {st_g['steady_ms']:.0f} ms/步 = {2e3 / st_g['steady_ms']:.2f} tok/s | "
          f"总 {N_GEN / st_g['decode_s']:.2f} tok/s | 命中 {st_g['acc']}/{st_g['steps']} | "
          f"整卡 {guard.peak / 1e9:.2f} GB", flush=True)
    te, st_e = run_eager()
    print(f"[eager] prefill {st_e['prefill_s']:.1f}s | 解码 {st_e['decode_s']:.2f}s | "
          f"{st_e['tokens'] / st_e['decode_s']:.2f} tok/s | 整卡 {guard.peak / 1e9:.2f} GB", flush=True)
print(f"token 一致：{tg == te[:len(tg)]}（graph {len(tg)} / eager {len(te)}）")
print("  graph 文本:", tok.decode(tg)[:160].replace("\n", " "))
print("  eager 文本:", tok.decode(te)[:160].replace("\n", " "))
guard.close()
