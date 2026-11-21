"""16K 下 KV cache 到底占多少：按槽位统计实际分配的存储。"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
tok, model = load("offline", int8=[])
for bits in (8, 4):
    cache = make_fixed_cache(model, N + 64, kv_int8=True, pass2_bits=bits)
    full = [l for l in cache.layers if hasattr(l, "cur")]
    tot = sum(l.storage_bytes() for l in full)
    by_bits = {}
    for l in full:
        by_bits.setdefault(l.bits, [0, 0])
        by_bits[l.bits][0] += 1
        by_bits[l.bits][1] += l.storage_bytes()
    print(f"第2遍 {bits}-bit：{len(full)} 个槽位，KV 合计 {tot / 2**20:.1f} MiB "
          f"（{', '.join(f'{b}-bit × {c[0]}: {c[1] / 2**20:.1f} MiB' for b, c in sorted(by_bits.items()))}）")
    del cache
    torch.cuda.empty_cache()
