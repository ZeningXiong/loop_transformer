"""mHC 的小算子占"元素级长尾"多少：单独计时 router / merge / enter / leave，再乘出现次数。"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load  # noqa: E402

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"
mod = sys.modules[type(model).__module__]          # modeling_sensenova_loop
MHCRouter, merge = mod.MHCRouter, mod.MHCRouter.merge
tm = model.model.language_model
routers = [m for m in model.modules() if isinstance(m, MHCRouter)]
print(f"mHC 路由器共 {len(routers)} 个；streams={routers[0].streams}，"
      f"weight {tuple(routers[0].weight.shape)}，fixed_identity={routers[0].fixed_identity}")


def bench(fn, it=200, warm=20):
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


r = routers[0]
dim = r.weight.shape[1] // r.streams
print(f"每层执行：1 个 attn 路由 + 1 个 ffn 路由 + 2 次 merge；一个 token 56 次层执行")

res = {}
for T in (1, 2):
    x = (torch.randn(1, T, r.streams, dim, device=dev) * 0.05).to(torch.bfloat16)
    with torch.inference_mode():
        branch, post, mix = r(x)
        out = merge(x, branch, post, mix)
        res[T] = dict(
            router=bench(lambda: r(x)),
            merge=bench(lambda: merge(x, branch, post, mix)),
        )
    print(f"T={T}: router {res[T]['router']:.1f} µs | merge {res[T]['merge']:.1f} µs "
          f"| branch {tuple(branch.shape)} mix {tuple(mix.shape)}")

for T in (1, 2):
    router_total = 2 * 56 * res[T]["router"] / 1e3
    merge_total = 2 * 56 * res[T]["merge"] / 1e3
    print(f"\nT={T}：一个 token 里 mHC 的 router {router_total:.1f} ms + merge {merge_total:.1f} ms "
          f"= {router_total + merge_total:.1f} ms")
print("\n参考：内核 profile 里整个元素级长尾约 60 ms/步（1 token，200 token 上下文）")
