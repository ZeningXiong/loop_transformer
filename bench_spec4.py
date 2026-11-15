"""一次验证 4 个 token（3 个链式草稿）的自投机解码，与现有"1 草稿 / 验证 2"对比。

用法: python bench_spec4.py [上下文 token 数] [生成 token 数]
环境变量 ALL_NF4=1 用全 NF4 权重（8GB 内才放得下 16K + MTP）。
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot, get_tap  # noqa: E402

PLAN = "mixed_plan_1.0.json"
CHUNK = 256
n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 32
ND = 3                                        # drafts per step -> 1 + ND tokens verified

ALL_NF4 = os.environ.get("ALL_NF4") == "1"
plan = [] if ALL_NF4 else json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))["int8"]
tok, model = load("offline", int8=plan)
guard = VramGuard()
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()
max_len = n_ctx + n_gen + 32
mtp = MTPHead(model)
mtp.new_cache(max_len)
cache = make_fixed_cache(model, max_len, kv_int8=True)
tap = get_tap(model)
dev = "cuda"


@torch.inference_mode()
def main_forward(tokens, pos):
    p = torch.arange(pos, pos + tokens.shape[1], device=dev)[None]
    h = model.model(input_ids=tokens, position_ids=p, past_key_values=cache, use_cache=True).last_hidden_state
    return tap.post, model.lm_head(h).float()


@torch.inference_mode()
def draft_step(hidden, token, pos):
    """One MTP step, also returning its hidden state so drafts can be chained EAGLE-style."""
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(token))
    h = mtp.pre_fc_norm_hidden(hidden)
    x = mtp.fc(torch.cat([e, h] if mtp.embed_first else [h, e], dim=-1))
    pe = mtp.tm.rotary_emb(x, pos[None].expand(3, -1, -1))
    x = mtp.layer(x, position_embeddings=pe, attention_mask=None, position_ids=pos,
                  past_key_values=mtp.cache, use_cache=True)
    return mtp.lm_head(mtp.norm(x))[0, -1:1], x


@torch.inference_mode()
def run(nd, n_max):
    reset_cache(cache)
    mtp.cache.reset()
    T = ids.shape[1]
    t0 = time.perf_counter()
    prev_h = None
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
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

    snap = _Snapshot(cache)
    x = logits[0, -1].argmax().view(1, 1)
    h_last, pos = prev_h, T
    out, steps, acc, accepted_total = [], 0, [0] * max(nd, 1), 0
    td = time.perf_counter()
    while len(out) < n_max:
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
        accepted_total += j
        for i in range(j):
            acc[i] += 1
        if j == nd:                                   # every draft accepted
            commit, nxt, Hc = [x] + drafts, preds[nd], H
        else:                                         # first mismatch: roll back, redo the committed prefix
            snap.restore()
            prefix = torch.cat([x] + drafts[:j] + [preds[j]], 1)
            Hc, L2 = main_forward(prefix, pos)
            commit, nxt = prefix.split(1, dim=1), L2[0, -1].argmax().view(1, 1)
        out.extend(t.item() for t in commit)
        # keep the MTP cache in step: rewind, then consume (hidden at q-1, token at q) for each committed token
        mtp.cache.layer.cur = base
        mtp.cache.layer._set_views()
        for i, tk in enumerate(commit):
            hidden = h_last if i == 0 else Hc[:, i - 1:i]
            draft_step(hidden, tk, torch.tensor([[pos + i]], device=dev))
        x, h_last, pos = nxt, Hc[:, len(commit) - 1:len(commit)], pos + len(commit)
        if out and out[-1] in (248045, 248046):       # rough EOS guard
            break
    torch.cuda.synchronize()
    return out[:n_max], dict(prefill_s=pre_s, decode_s=time.perf_counter() - td,
                             steps=steps, acc=acc, tokens=len(out[:n_max]))


if __name__ == "__main__":
    print(f"上下文 {n_ctx}，KV int8，{'全 NF4' if ALL_NF4 else '+1.0GB int8'}，"
          f"每步验证 {1 + ND} 个 token", flush=True)
    toks, st = run(ND, n_gen)
    tps = st["tokens"] / st["decode_s"]
    print(f"结果：{st['tokens']} token / {st['decode_s']:.2f}s = {tps:.2f} tok/s；"
          f"步数 {st['steps']}，平均每步提交 {st['tokens'] / st['steps']:.2f} 个，"
          f"逐位命中 {st['acc']}；prefill {st['prefill_s']:.1f}s", flush=True)
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB")
    guard.close()
