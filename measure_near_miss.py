"""失配时草稿离"正确 token"有多近？——为"近似接受"提供数据。

跑 3 草稿的自投机若干步，在每个验证位置记录 p(草稿)/p(模型首选) 的比值，
统计不同阈值下能"捞回"多少失配（也就是能省掉多少次重算）。
"""
import json
import os
import sys

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot, get_tap  # noqa: E402

n_ctx, n_tok, ND = 16384, 24, 3
dev = "cuda"
tok, model = load("offline", int8=[])
tap = get_tap(model)
mtp = MTPHead(model)
mtp.new_cache(n_ctx + 64)
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
cache = make_fixed_cache(model, n_ctx + 64, kv_int8=True)
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
def prefill():
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
    logits = model.lm_head(tap.post[:, -1:]).float()
    return prev_h, logits[0, -1].argmax().view(1, 1)


with torch.inference_mode():
    h_last, x = prefill()
    pos = n_ctx
    out, steps = [], 0
    ratios = []          # (position_index, ratio)
    while len(out) < n_tok and steps < 12:
        steps += 1
        snap.save()
        base = mtp.cache.layer.cur
        drafts, hh, tk, p = [], h_last, x, pos
        for _ in range(ND):
            lg, hh = draft_step(hh, tk, torch.tensor([[p]], device=dev))
            tk = lg.argmax().view(1, 1)
            drafts.append(tk)
            p += 1
        H, L = main_forward(torch.cat([x] + drafts, 1), pos)
        preds = [L[0, i].argmax().view(1, 1) for i in range(1 + ND)]
        # 概率比：p(草稿) / p(模型首选)
        for i in range(ND):
            pr = torch.softmax(L[0, i], -1)
            ratios.append((i, (pr[drafts[i].item()] / pr[preds[i].item()]).item()))
        j = 0
        while j < ND and preds[j].item() == drafts[j].item():
            j += 1
        if j == ND:
            commit, nxt, Hc = [x] + drafts, preds[ND], H
        else:
            snap.restore()
            prefix = torch.cat([x] + drafts[:j] + [preds[j]], 1)
            Hc, L2 = main_forward(prefix, pos)
            commit, nxt = prefix.split(1, dim=1), L2[0, -1].argmax().view(1, 1)
        out.extend(t.item() for t in commit)
        mtp.cache.layer.cur = base
        mtp.cache.layer._set_views()
        for i, tk in enumerate(commit):
            hidden = h_last if i == 0 else Hc[:, i - 1:i]
            draft_step(hidden, tk, torch.tensor([[pos + i]], device=dev), logits=False)
        x, h_last, pos = nxt, Hc[:, len(commit) - 1:len(commit)], pos + len(commit)

mis = [r for _, r in ratios if r < 0.999]
print(f"\n{steps} 步，{len(ratios)} 个草稿位，其中失配 {len(mis)} 个")
if mis:
    mis.sort()
    import statistics
    print(f"  失配时的概率比 p(草稿)/p(首选)：中位数 {statistics.median(mis):.3f}，"
          f"最大 {max(mis):.3f}，最小 {min(mis):.2e}")
    for a in (0.9, 0.7, 0.5, 0.3, 0.1):
        k = sum(1 for r in mis if r >= a)
        print(f"  比值 ≥ {a:<4}: {k:3d}/{len(mis)} = {100 * k / len(mis):5.1f}% 的失配可被“近似接受”")
