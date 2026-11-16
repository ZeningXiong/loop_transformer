"""把自投机的一步拆成 5 段计时：快照 / 草稿 / 验证 / 失配重算 / MTP 回填。

用来解释"3-token 每步为什么比 2-token 贵 135 ms"。
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

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot, get_tap  # noqa: E402

n_ctx, n_max = 16384, 24
dev = "cuda"
tok, model = load("offline", int8=[])
tap = get_tap(model)
mtp = MTPHead(model)
max_len = n_ctx + 64
mtp.new_cache(max_len)
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
    return prev_h


@torch.inference_mode()
def run(nd, n_tok):
    h_last = prefill()
    logits = model.lm_head(tap.post[:, -1:]).float()
    x = logits[0, -1].argmax().view(1, 1)
    T = n_ctx
    pos = T
    out = []
    ph = dict(snap=0.0, draft=0.0, verify=0.0, redo=0.0, feed=0.0)
    steps = 0
    redos = 0
    while len(out) < n_tok:
        t0 = time.perf_counter()
        snap.save()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        base = mtp.cache.layer.cur
        drafts, hh, tk, p = [], h_last, x, pos
        for _ in range(nd):
            lg, hh = draft_step(hh, tk, torch.tensor([[p]], device=dev))
            tk = lg.argmax().view(1, 1)
            drafts.append(tk)
            p += 1
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        H, L = main_forward(torch.cat([x] + drafts, 1), pos)
        preds = [L[0, i].argmax().view(1, 1) for i in range(1 + nd)]
        j = 0
        while j < nd and preds[j].item() == drafts[j].item():
            j += 1
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if j == nd:
            commit, nxt, Hc = [x] + drafts, preds[nd], H
        else:
            redos += 1
            snap.restore()
            prefix = torch.cat([x] + drafts[:j] + [preds[j]], 1)
            Hc, L2 = main_forward(prefix, pos)
            commit, nxt = prefix.split(1, dim=1), L2[0, -1].argmax().view(1, 1)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        mtp.cache.layer.cur = base
        mtp.cache.layer._set_views()
        for i, tk in enumerate(commit):
            hidden = h_last if i == 0 else Hc[:, i - 1:i]
            draft_step(hidden, tk, torch.tensor([[pos + i]], device=dev), logits=False)
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        out.extend(t.item() for t in commit)
        x, h_last, pos = nxt, Hc[:, len(commit) - 1:len(commit)], pos + len(commit)
        steps += 1
        ph["snap"] += t1 - t0
        ph["draft"] += t2 - t1
        ph["verify"] += t3 - t2
        ph["redo"] += t4 - t3
        ph["feed"] += t5 - t4
    return ph, steps, redos, len(out), (nd + 1), len(commit)


with torch.inference_mode():
    for nd in (1, 3):
        ph, steps, redos, ntok, per_step, last_commit = run(nd, n_max)
        tot = sum(ph.values())
        print(f"\n验证 {1 + nd} 个 token：{steps} 步，提交 {ntok}，其中失配重算 {redos} 次")
        print(f"  总 {tot * 1e3:.0f} ms = {tot * 1e3 / steps:.0f} ms/步")
        for k in ("snap", "draft", "verify", "redo", "feed"):
            print(f"    {k:<7}{ph[k] * 1e3 / steps:7.1f} ms/步  ({100 * ph[k] / tot:4.0f}%)")
        print(f"  速度 {ntok / tot:.2f} tok/s", flush=True)
