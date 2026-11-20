"""16K 下元素级融合的端到端收益：RMSNorm / SiLU×up 逐个打开，看一步能省多少。"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

import fused_rmsnorm  # noqa: E402
from fused_silu import enable_silu  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev = "cuda"
tok, model = load("offline", int8=[])
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
snap = _Snapshot(cache)

with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    tk = torch.tensor([[1000]], device=dev)
    pos = torch.tensor([[N_CTX]], device=dev)

    def step():
        snap.save()
        model.model(input_ids=tk, position_ids=pos, past_key_values=cache, use_cache=True)
        snap.restore()

    def bench(n=10, warm=3):
        with torch.inference_mode():
            for _ in range(warm):
                step()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                step()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e3

    fused_rmsnorm.enable(False)
    enable_silu(False)
    base = bench()
    fused_rmsnorm.enable(True)
    rn = bench()
    enable_silu(True)
    both = bench()
    enable_silu(False)
    rn_only = bench()
    # 顺序反转再测一轮：先融合后基线，排除漂移
    enable_silu(True)
    both2 = bench()
    fused_rmsnorm.enable(False)
    enable_silu(False)
    base2 = bench()

print(f"上下文 {N_CTX}，1-token 前向：")
print(f"  基线                    {base:7.1f} ms")
print(f"  + 融合 RMSNorm          {rn:7.1f} ms  （{rn - base:+.1f}，{100 * (rn - base) / base:+.1f}%）")
print(f"  + 融合 SiLU×up          {both:7.1f} ms  （{both - base:+.1f}，{100 * (both - base) / base:+.1f}%）")
print(f"  只开 SiLU（关 RMSNorm）  {rn_only:7.1f} ms")
print(f"  --- 顺序反转 ---")
print(f"  融合（先）              {both2:7.1f} ms")
print(f"  基线（后）              {base2:7.1f} ms  （融合相对它 {100 * (both2 - base2) / base2:+.1f}%）")
