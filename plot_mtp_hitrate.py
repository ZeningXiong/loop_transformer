"""MTP 逐位命中率图：左图条件命中率（到达该位后命中多少），右图链路到达率。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 两次完整对比（16K、全 NF4、NF4 MTP、KV int8、闭式解 sinkhorn、链式草稿）里逐位被接受的次数
# (被接受次数, 到达该位的步数)
COND = {
    1: (69, 74),   # 12/12 + 8/9 + 7/8 + 7/8 两次
    2: (12, 45),   # 2/8 + 3/7 + 2/7 两次
    3: (4, 8),     # 1/3 + 1/2 两次（样本很少）
    4: (0, 2),
    5: (0, 1),
}
pos = list(COND)
rate = [100 * c / n for c, n in COND.values()]
n = [n for _, n in COND.values()]

# 链路到达率：从第 1 位一路命中到第 k 位的概率
reach, acc = [], 1.0
for r in rate:
    acc *= r / 100
    reach.append(100 * acc)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

import math


def wilson(k, nn, z=1.96):
    if nn == 0:
        return 0.0, 0.0
    ph = k / nn
    d = 1 + z * z / nn
    c = (ph + z * z / (2 * nn)) / d
    h = z * math.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


err = [[r - wilson(k, nn)[0] for r, (k, nn) in zip(rate, COND.values())],
       [wilson(k, nn)[1] - r for r, (k, nn) in zip(rate, COND.values())]]
b = ax[0].bar([f"第{i}个" for i in pos], rate, color=["#2d6cdf"] + ["#9db8e8"] * 4,
              yerr=err, capsize=4, ecolor="#444")
ax[0].axhline(75, ls="--", c="#d64545", lw=1.4)
ax[0].text(4.45, 78, "多草稿划算的门槛 ≈75%", color="#d64545", ha="right", fontsize=9)
for r, (k, nn), rect in zip(rate, COND.values(), b):
    ax[0].text(rect.get_x() + rect.get_width() / 2, r + 1.5, f"{r:.0f}%\n{k}/{nn}",
               ha="center", fontsize=9)
ax[0].set_ylim(0, 108)
ax[0].set_ylabel("条件命中率（%）")
ax[0].set_title("到达该位后，草稿被接受的概率")
ax[0].grid(axis="y", alpha=0.25)

ax[1].plot([f"第{i}个" for i in pos], reach, "o-", color="#2d6cdf", lw=2)
for i, r in enumerate(reach):
    ax[1].text(i, r + 3, f"{r:.0f}%", ha="center", fontsize=9)
ax[1].set_ylim(0, 105)
ax[1].set_ylabel("链式到达该位的概率（%）")
ax[1].set_title("链式草稿的有效长度：第 2 位就掉到 1/4")
ax[1].grid(alpha=0.25)

fig.suptitle("SenseNova-Loop-9B：MTP 逐位命中率（16K、全 NF4、NF4 头、KV int8、闭式解 sinkhorn）",
             fontsize=11)
fig.text(0.5, 0.005, "口径：条件命中率 = 该位被接受次数 / 到达该位的步数；"
                     "第 3–5 位样本仅 1–2 步起算，误差棒为 95% Wilson 区间", ha="center",
         fontsize=8, color="#555")
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
out = os.path.join(HERE, "assets", "mtp_hitrate.png")
fig.savefig(out, dpi=150)
print("已保存", out)
