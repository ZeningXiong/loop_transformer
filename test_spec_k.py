"""16K 下对比 2 / 4 / 6 token 验证（nda 次草稿）：全 NF4 权重 + NF4 MTP 头 + KV int8 + 闭式解 sinkhorn。

每次尝试前重新 prefill，同一个进程、同一份模型和缓存；显存超限立即停止（VramGuard）。
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

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot, get_tap  # noqa: E402

n_ctx, n_gen = 16384, 24
dev = "cuda"
tok, model = load("offline", int8=[])          # 全 NF4；sinkhorn 闭式解由 load_offline 默认启用
tap = get_tap(model)
mtp = MTPHead(model)
max_len = n_ctx + 64
mtp.new_cache(max_len)

# MTP 头量化成 NF4
n_q = 0
for name, child in list(mtp.named_modules()):
    if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
        parent_name, _, attr = name.rpartition(".")
        parent = mtp.get_submodule(parent_name) if parent_name else mtp
        new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=False,
                                compute_dtype=torch.bfloat16, quant_type="nf4",
                                compress_statistics=False, quant_storage=torch.uint8, device="meta")
        new.weight = bnb.nn.Params4bit(child.weight.data.detach(), requires_grad=False,
                                       quant_type="nf4", compress_statistics=False,
                                       quant_storage=torch.uint8)
        setattr(parent, attr, new.to(dev))
        n_q += 1

guard = VramGuard()
cache = make_fixed_cache(model, max_len, kv_int8=True)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()


@torch.inference_mode()
def draft_step(hidden, token, pos, logits=True):
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(token))
    hgn = mtp.pre_fc_norm_hidden(hidden)
    x = mtp.fc(torch.cat([e, hgn] if mtp.embed_first else [hgn, e], dim=-1))
    pe = mtp.tm.rotary_emb(x, pos[None].expand(3, -1, -1))
    x = mtp.layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
                  past_key_values=mtp.cache, use_cache=True)
    if not logits:
        return None, x
    return mtp.lm_head(mtp.norm(x))[0, -1:1], x


@torch.inference_mode()
def main_forward(tokens, pos):
    p = torch.arange(pos, pos + tokens.shape[1], device=dev)[None]
    h = model.model(input_ids=tokens, position_ids=p, past_key_values=cache,
                    use_cache=True).last_hidden_state
    return tap.post, model.lm_head(h).float()


@torch.inference_mode()
def run(nd, n_max):
    reset_cache(cache)
    mtp.cache.reset()
    T = ids.shape[1]
    t0 = time.perf_counter()
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
    logits = model.lm_head(tap.post[:, -1:]).float()
    torch.cuda.synchronize()
    pre_s = time.perf_counter() - t0

    x = logits[0, -1].argmax().view(1, 1)
    h_last, pos = prev_h, T
    out, steps, acc = [], 0, [0] * max(nd, 1)
    per = []
    td = time.perf_counter()
    while len(out) < n_max:
        t_step = time.perf_counter()
        snap.save()
        base = mtp.cache.layer.cur
        drafts, hh, tk, p = [], h_last, x, pos
        for _ in range(nd):
            lg, hh = draft_step(hh, tk, torch.tensor([[p]], device=dev))
            tk = lg.argmax().view(1, 1)
            drafts.append(tk)
            p += 1
        H, L = main_forward(torch.cat([x] + drafts, 1), pos)
        preds = [L[0, i].argmax().view(1, 1) for i in range(1 + nd)]
        j = 0
        while j < nd and preds[j].item() == drafts[j].item():
            j += 1
        steps += 1
        for i in range(j):
            acc[i] += 1
        if j == nd:
            commit, nxt, Hc = [x] + drafts, preds[nd], H
        else:
            snap.restore()
            prefix = torch.cat([x] + drafts[:j] + [preds[j]], 1)
            Hc, L2 = main_forward(prefix, pos)
            commit, nxt = prefix.split(1, dim=1), L2[0, -1].argmax().view(1, 1)
        out.extend(t.item() for t in commit)
        torch.cuda.synchronize()
        per.append((time.perf_counter() - t_step, len(commit)))
        mtp.cache.layer.cur = base
        mtp.cache.layer._set_views()
        for i, tk in enumerate(commit):
            hidden = h_last if i == 0 else Hc[:, i - 1:i]
            draft_step(hidden, tk, torch.tensor([[pos + i]], device=dev), logits=False)
        x, h_last, pos = nxt, Hc[:, len(commit) - 1:len(commit)], pos + len(commit)
        guard.check()
        if out and out[-1] in (248045, 248046):
            break
    torch.cuda.synchronize()
    return out[:n_max], dict(prefill_s=pre_s, decode_s=time.perf_counter() - td,
                             steps=steps, acc=acc, tokens=len(out[:n_max]), per=per)


print(f"配置：全 NF4 + NF4 MTP（{n_q} 层）+ KV int8 + 闭式解 sinkhorn，上下文 {n_ctx}，"
      f"整卡上限 {guard.total / 1e9:.2f} GB", flush=True)
for nd in (1, 2, 3, 5):
    toks, st = run(nd, n_gen)
    tps = st["tokens"] / st["decode_s"]
    print(f"\n验证 {1 + nd} 个 token（{nd} 草稿）：{st['tokens']} token / {st['decode_s']:.2f}s = "
          f"**{tps:.2f} tok/s**（{st['decode_s'] / st['steps'] * 1e3:.0f} ms/步）")
    per = st["per"]
    if len(per) > 4:
        warm = per[:3]
        steady = per[3:]
        st_tok = sum(n for _, n in steady)
        st_s = sum(t for t, _ in steady)
        print(f"  逐位命中 {st['acc']}，提交/步 {st['tokens'] / st['steps']:.2f}")
        print(f"  前 3 步 {1000 * sum(t for t, _ in warm) / len(warm):.0f} ms/步（预热）→ "
              f"稳态 {1000 * st_s / len(steady):.0f} ms/步 = **{st_tok / st_s:.2f} tok/s**")
    print(f"  prefill {st['prefill_s']:.1f}s，整卡峰值 {guard.peak / 1e9:.2f} GB", flush=True)
print(f"\n全程整卡峰值 {guard.peak / 1e9:.2f} GB / {guard.total / 1e9:.2f} GB")
guard.close()
