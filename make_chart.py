"""Chart for the CUDA-graph decode experiment: cudagraph_bench.json + cudagraph_batch.json."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

bench = json.load(open(os.path.join(HERE, "cudagraph_bench.json"), encoding="utf-8"))
batch = json.load(open(os.path.join(HERE, "cudagraph_batch.json"), encoding="utf-8"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
fig.suptitle("SenseNova-Loop-9B-Preview：CUDA Graph 解码与批处理效应（RTX 4070 Laptop 8GB，+1.0GB int8 方案）",
             fontsize=12.5, y=0.985)

# ---- left: tok/s by context length
xs = range(len(bench))
w = 0.36
base = [r["base_tps"] for r in bench]
graph = [r["graph_tps"] for r in bench]
b1 = ax1.bar([x - w / 2 for x in xs], base, w, label="当前解码（逐层 eager）", color="#8ea9c1")
b2 = ax1.bar([x + w / 2 for x in xs], graph, w, label="CUDA Graph 重放", color="#2f6f9f")
for x, r in zip(xs, bench):
    ax1.text(x - w / 2, r["base_tps"] + 0.15, f'{r["base_tps"]:.2f}', ha="center", fontsize=9)
    ax1.text(x + w / 2, r["graph_tps"] + 0.15, f'{r["graph_tps"]:.2f}', ha="center", fontsize=9,
             color="#1d4f77", fontweight="bold")
    ax1.text(x + w / 2, r["graph_tps"] * 0.62, f'×{r["speedup"]:.2f}', ha="center", fontsize=12,
             color="white", fontweight="bold")
    ax1.text(x + w / 2, r["graph_tps"] * 0.40, f'{r["graph_ms"]:.0f} ms', ha="center",
             fontsize=9.5, color="white")
ax1.set_xticks(list(xs))
ax1.set_xticklabels([f'上下文 {r["n_ctx"]}' for r in bench])
ax1.set_ylabel("解码速度（token/s）")
ax1.set_ylim(0, max(base + graph) * 1.25)
ax1.axhline(0, color="black", lw=0.8)
ax1.grid(axis="y", alpha=0.25)
ax1.legend(loc="upper right", frameon=False)
ax1.set_title("图重放带来的提升：每步 56 次层执行一次提交，输出逐 token 相同", fontsize=10.5)

# ---- right: cost per step vs tokens per step
ks = [r["k"] for r in batch]
step_ms = [r["ms_per_step"] for r in batch]
tok_ms = [r["ms_per_token"] for r in batch]
ax2.bar(ks, step_ms, 0.55, color="#c9d8e5", label="每步耗时（ms）")
ax2.plot(ks, step_ms, "o-", color="#2f6f9f", lw=1.8, ms=6)
for k, s, t in zip(ks, step_ms, tok_ms):
    ax2.text(k, s + 10, f"{s:.0f} ms/步", ha="center", fontsize=9, color="#1d4f77")
    ax2.text(k, s / 2, f"{t:.0f}\nms/token", ha="center", fontsize=10, color="#b3541e",
             fontweight="bold")
ax2.set_xlabel("一次前向里验证的 token 数")
ax2.set_ylabel("耗时（ms）")
ax2.set_xticks(ks)
ax2.set_ylim(0, max(step_ms) * 1.22)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("每步验证更多 token：2 个反而更贵，4 个省 30%", fontsize=10.5)
ax2.text(0.03, 0.06, "线性注意力在 T>1 时切到分块算法，第 2 个 token 并不便宜\n→ 自投机只有一次验证 4 个 token 时才划算",
         transform=ax2.transAxes, fontsize=9, color="#555555", linespacing=1.5)

fig.tight_layout(rect=[0, 0.02, 1, 0.95])
out = os.path.join(HERE, "assets", "cudagraph_compare.png")
fig.savefig(out, dpi=140)
print(out)
