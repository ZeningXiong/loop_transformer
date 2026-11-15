"""逐个测量 int8 层在 T=1/2 下的耗时，并给出"逐行调用"的对照。"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import Int8WOLinear, load  # noqa: E402

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"


def bench(fn, it=15, warm=4):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


mods = []
for name, m in model.named_modules():
    if isinstance(m, Int8WOLinear):
        mods.append((name, m))
print(f"int8 线性层共 {len(mods)} 个")

t1 = t2 = t2row = 0.0
worst = []
with torch.inference_mode():
    for name, m in mods:
        k = m.in_features
        x1 = torch.randn(1, 1, k, dtype=torch.bfloat16, device=dev) * 0.05
        x2 = torch.randn(1, 2, k, dtype=torch.bfloat16, device=dev) * 0.05
        a = bench(lambda: m(x1))
        b = bench(lambda: m(x2))
        c = bench(lambda: [m(x2[:, i:i + 1]) for i in range(2)])
        t1 += a
        t2 += b
        t2row += c
        worst.append((b / a, b - a, name, a, b, c))

print(f"\nT=1 合计 {t1 / 1e3:.1f} ms | T=2 合计 {t2 / 1e3:.1f} ms（+{(t2 - t1) / 1e3:.1f} ms）"
      f" | T=2 逐行调用 {t2row / 1e3:.1f} ms（+{(t2row - t1) / 1e3:.1f} ms）")
worst.sort(key=lambda r: -r[1])
print(f"\n{'层':<52}{'T=1':>8}{'T=2':>9}{'倍数':>7}{'逐行':>8}")
for r, d, name, a, b, c in worst[:8]:
    print(f"{name.replace('model.language_model.', '')[:50]:<52}{a:>8.0f}{b:>9.0f}{r:>6.1f}x{c:>8.0f}")
