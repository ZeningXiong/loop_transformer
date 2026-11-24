"""从 logs/spec_chain.json 重新绘制链式草稿准确率图（不重新测量）。"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

d = json.load(open(os.path.join(HERE, "logs", "spec_chain.json"), encoding="utf-8"))
per_pos, reach, tried, K = d["per_pos"], d["reach"], d["tried"], d["K"]
valid = [r for r in reach if r == r]
exp_tok = 1 + sum(valid)

xs = list(range(1, K + 1))
heights = [0 if p != p else p * 100 for p in per_pos]
fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=170)
bars = ax.bar(xs, heights, width=0.55, color="#4C78A8",
              label="逐位接受率（条件于前 k-1 个全对）")
for b, p, t in zip(bars, per_pos, tried):
    txt = "未采到" if p != p else f"{p * 100:.0f}%"
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{txt}\n(n={t})",
            ha="center", va="bottom", fontsize=8.5)
line_x = [x_ for x_, r in zip(xs, reach) if r == r]
line_y = [r * 100 for r in reach if r == r]
ax.plot(line_x, line_y, "o-", color="#E45756", lw=2.2, ms=7,
        label="链式到达率（前 k 个全对）")
for x_, r in zip(line_x, line_y):
    if abs(r * 100 - next((p * 100 for p, xx in zip(per_pos, xs) if xx == x_), 0)) < 3:
        continue          # 和柱子标签重合就不重复标
    ax.annotate(f"{r * 100:.0f}%", (x_, r * 100), textcoords="offset points", xytext=(22, 4),
                fontsize=8.5, color="#E45756")

ax.set_xticks(xs)
ax.set_xticklabels([f"第 {i} 个草稿" for i in xs])
ax.set_ylim(0, 126)
ax.set_ylabel("百分比 (%)")
ax.set_title(f"多 token 链式草稿的准确率（16K 上下文，K={K}，{d['steps']} 步）", fontsize=11.5)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8.5, loc="upper right")
ax.text(0.30, 0.965, f"期望提交 ≈ 1 + Σ到达率 = {exp_tok:.2f} token/步",
        transform=ax.transAxes, fontsize=9, color="#333333")
ax.text(0.30, 0.895, "拐点在第 2 个草稿：第 3 个之后基本无收益",
        transform=ax.transAxes, fontsize=8.5, color="#666666")
fig.tight_layout()
out = os.path.join(HERE, "assets", "spec_chain.png")
fig.savefig(out)
print("已重绘:", out, "| 期望提交", round(exp_tok, 2))
