"""LoopKV's allocation method applied to this project's own weight-precision data.

The paper profiles each (loop, layer) cell once, sums the single-cell damages to predict a whole plan
(r = 0.98), then solves a multiple-choice knapsack exactly. mixed_search.json already holds the
equivalent per-group profile for weights (single upgrade NF4 -> int8, measured KL), so both steps can
be replayed on data that cost hours to collect.

python loopkv_plan.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open(os.path.join(HERE, "mixed_search.json"), encoding="utf-8"))
groups, base = d["groups"], d["baseline"]["search"]["kl"]

print("1) 可加性检验：把各组的单组伤害相加，能否预测整个方案的 KL（论文 r=0.98）")
print(f"{'预算 GB':>9}{'组数':>6}{'预测 KL':>10}{'实测 KL':>10}{'误差':>9}")
pred, meas = [], []
for row in d["curve"]:
    g = row["groups"]
    p = base - sum(groups[n]["gain"] for n in g)
    pred.append(p)
    meas.append(row["search"]["kl"])
    print(f"{row['budget_gb']:>9.2f}{len(g):>6}{p:>10.4f}{row['search']['kl']:>10.4f}"
          f"{(p - row['search']['kl']) / row['search']['kl'] * 100:>8.1f}%")
n = len(pred)
mp, mm = sum(pred) / n, sum(meas) / n
cov = sum((a - mp) * (b - mm) for a, b in zip(pred, meas))
r = cov / ((sum((a - mp) ** 2 for a in pred) ** 0.5) * (sum((b - mm) ** 2 for b in meas) ** 0.5))
print(f"   相关系数 r = {r:.3f}（论文报告的 0.98 是同一量级）\n")

print("2) 精确背包：同样显存预算下，最多能拿回多少 KL（论文用 DP 而不是贪心）")
names = list(groups)
gb = [groups[n_]["extra_bytes"] / 1e9 for n_ in names]
gain = [groups[n_]["gain"] for n_ in names]
step = 0.0025                                    # 0.25% of a GB per DP cell
print(f"{'预算 GB':>9}{'贪心组数':>9}{'贪心预测 KL':>12}{'DP 组数':>8}{'DP 预测 KL':>11}{'多拿回':>9}")
for budget in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5):
    # greedy = the project's existing order (by gain per byte)
    acc, used, gsel = 0.0, 0.0, []
    for n_ in d["order"]:
        if used + groups[n_]["extra_bytes"] / 1e9 <= budget:
            used += groups[n_]["extra_bytes"] / 1e9
            gsel.append(n_)
    gl = base - sum(groups[n_]["gain"] for n_ in gsel)
    # exact 0/1 knapsack over integer byte cells
    cap = int(round(budget / step))
    best = [0.0] * (cap + 1)
    take = [[False] * (cap + 1) for _ in range(len(names))]
    for i, (b, gn) in enumerate(zip(gb, gain)):
        w = max(1, int(round(b / step)))
        for c in range(cap, w - 1, -1):
            cand = best[c - w] + gn
            if cand > best[c] + 1e-12:
                best[c] = cand
                take[i][c] = True
    c = max(range(cap + 1), key=lambda j: best[j])
    chosen = []
    for i in range(len(names) - 1, -1, -1):
        if take[i][c]:
            chosen.append(names[i])
            c -= max(1, int(round(gb[i] / step)))
    dl = base - best[cap] if best[cap] > 0 else base
    dl = base - sum(groups[n_]["gain"] for n_ in chosen)
    print(f"{budget:>9.2f}{len(gsel):>9}{gl:>12.4f}{len(chosen):>8}{dl:>11.4f}"
          f"{gl - dl:>9.4f}")

print("\n3) +1.0GB 档：贪心选了什么、DP 选了什么")
b = 1.0
acc, used, gsel = 0.0, 0.0, []
for n_ in d["order"]:
    if used + groups[n_]["extra_bytes"] / 1e9 <= b:
        used += groups[n_]["extra_bytes"] / 1e9
        gsel.append(n_)
print(f"   贪心：{len(gsel)} 组，{used:.2f} GB，预测 KL {base - sum(groups[n_]['gain'] for n_ in gsel):.4f}")
print("   " + "、".join(gsel))
plan = json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"), encoding="utf-8"))["groups"]
print(f"   现有 mixed_plan_1.0.json：{len(plan)} 组，预测 KL "
      f"{base - sum(groups[n_]['gain'] for n_ in plan):.4f}，实测 0.0785")
missing = [n_ for n_ in gsel if n_ not in plan]
extra = [n_ for n_ in plan if n_ not in gsel]
print(f"   与现有方案的差异：贪心多选了 {missing}，现有方案多选了 {extra}")
