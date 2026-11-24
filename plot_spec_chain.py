"""测量并绘制"多 token 链式草稿"的逐位准确率。

与真实投机一致的做法（每一步）：
  1. 用 MTP **链式**草稿 K 个 token：第 k 个用第 k-1 个草稿的隐藏状态；
  2. 主模型一次前向验证 [x, d1..dK]（K+1 个位置），得到每个位置的 argmax；
  3. 逐位比较 d_i 与 logits[i-1]，统计"条件于前缀全对"的接受率；
  4. 回滚，按"已接受前缀 + 1 个奖励 token"提交，继续。

输出：assets/spec_chain.png
"""
import json
import os
import sys

import bitsandbytes as bnb
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
dev = "cuda"

tok, model = load("offline", int8=[])
mtp = MTPHead(model)
HEAD = 1024
mtp.new_cache(N_CTX + HEAD)
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

cache = make_fixed_cache(model, N_CTX + HEAD, kv_int8=True, pass2_bits=4)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()


@torch.inference_mode()
def mtp_step(hidden, token, pos):
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(token))
    h = mtp.pre_fc_norm_hidden(hidden)
    z = mtp.fc(torch.cat([e, h] if mtp.embed_first else [h, e], dim=-1))
    pe = mtp.tm.rotary_emb(z, torch.tensor([[pos]], device=dev)[None].expand(3, -1, -1))
    z = mtp.layer(z, position_embeddings=pe, attention_mask=None,
                  position_ids=torch.tensor([[pos]], device=dev),
                  past_key_values=mtp.cache, use_cache=True)
    return mtp.lm_head(mtp.norm(z))[0, -1], z


with torch.inference_mode():
    reset_cache(cache)
    mtp.cache.reset()
    prev_h = None
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
        if prev_h is not None:
            mtp(prev_h, ids[:, s:s + 1], torch.tensor([[s]], device=dev), logits=False)
        if e - s > 1:
            mtp(h[:, :-1], ids[:, s + 1:e], torch.arange(s + 1, e, device=dev)[None], logits=False)
        prev_h = h[:, -1:]

    x = model.lm_head(prev_h).float()[0, -1].argmax().view(1, 1)
    h_last, pos = prev_h, N_CTX
    hit, tried = [0] * K, [0] * K
    steps_done = 0

    STEPS = min(STEPS, (HEAD - 2 * K) // 2)
    for _ in range(STEPS):
        snap.save()
        base = mtp.cache.layer.cur

        # 1) 链式草稿 K 个
        drafts, h_chain, cur_tok = [], h_last, x
        for k in range(K):
            lg, z = mtp_step(h_chain, cur_tok, pos + k)
            cur_tok = lg.argmax().view(1, 1)
            drafts.append(cur_tok.item())
            h_chain = z
        mtp.cache.layer.cur = base
        mtp.cache.layer._set_views()

        # 2) 一次前向验证 [x, d1..dK]
        seq = torch.tensor([[x.item()] + drafts], device=dev)
        h = model.model(input_ids=seq, position_ids=torch.arange(pos, pos + K + 1, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
        lg = model.lm_head(h).float()
        pred = [lg[0, i].argmax().item() for i in range(K + 1)]

        # 3) 逐位统计（条件于前缀全对）
        accepted = 0
        for i in range(K):
            tried[i] += 1
            if drafts[i] == pred[i]:
                hit[i] += 1
                accepted += 1
            else:
                break
        snap.restore()

        # 4) 提交：x + 已接受前缀 + 奖励 token，并重放以得到正确 KV
        commit = [x.item()] + drafts[:accepted] + [pred[accepted]]
        H = model.model(input_ids=torch.tensor([commit], device=dev),
                        position_ids=torch.arange(pos, pos + len(commit), device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
        pos += len(commit)
        h_last = H[:, -1:]
        x = model.lm_head(h_last).float()[0, -1].argmax().view(1, 1)
        steps_done += 1

        # MTP 缓存同步：(h_j, token_j)
        mtp.cache.layer.cur = pos - len(commit)
        mtp.cache.layer._set_views()
        for j, t in enumerate(commit):
            mtp(H[:, j:j + 1], torch.tensor([[t]], device=dev),
                torch.tensor([[pos - len(commit) + j]], device=dev), logits=False)

per_pos = [h / t if t else float("nan") for h, t in zip(hit, tried)]
reach, prod = [], 1.0
for p in per_pos:
    prod *= p
    reach.append(prod)

print(f"16K、K={K}、{steps_done} 步")
print("逐位尝试次数:", tried)
print("逐位接受次数:", hit)
print("逐位接受率:", [round(p, 3) for p in per_pos])
print("链式到达率:", [round(r, 3) for r in reach])
valid_reach = [r for r in reach if r == r]
print("期望提交 ≈ 1 + Σ到达率 =", round(1 + sum(valid_reach), 2), "token/步")
json.dump({"K": K, "steps": steps_done, "tried": tried, "hit": hit,
           "per_pos": per_pos, "reach": reach},
          open(os.path.join(HERE, "logs", "spec_chain.json"), "w", encoding="utf-8"), ensure_ascii=False)

os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
xs = list(range(1, K + 1))
fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=160)
heights = [(0 if p != p else p * 100) for p in per_pos]
bars = ax.bar(xs, heights, width=0.55, color="#4C78A8",
              label="逐位接受率（条件于前 k-1 个全对）")
for b, p, t in zip(bars, per_pos, tried):
    txt = "未采到" if p != p else f"{p * 100:.0f}%"
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{txt}\n(n={t})",
            ha="center", va="bottom", fontsize=8)
ax.plot(xs, [r * 100 for r in reach], "o-", color="#E45756", lw=2, ms=6, label="链式到达率（前 k 个全对）")
for x_, r in zip(xs, reach):
    if r == r:
        ax.annotate(f"{r * 100:.0f}%", (x_, r * 100), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=8, color="#E45756")
ax.set_xticks(xs)
ax.set_xticklabels([f"第 {i} 个草稿" for i in xs])
ax.set_ylim(0, 112)
ax.set_ylabel("百分比 (%)")
ax.set_title(f"多 token 链式草稿的准确率（16K 上下文，K={K}，{steps_done} 步）", fontsize=11)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8, loc="upper right")
valid = [r for r in reach if r == r]
ax.text(0.02, 0.04, f"期望提交 ≈ 1 + Σ到达率 = {1 + sum(valid):.2f} token/步",
        transform=ax.transAxes, fontsize=8.5)
fig.tight_layout()
out = os.path.join(HERE, "assets", "spec_chain.png")
fig.savefig(out)
print("图已保存:", out)
