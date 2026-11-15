"""16K 上下文 + 自投机解码：实测代价与收益，并回答"一次验证 4 个 token 是否更好"。

python bench_spec16k.py [上下文 token 数] [生成 token 数]
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, generate_speculative  # noqa: E402

PLAN = "mixed_plan_1.0.json"
CHUNK = 256
n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 24

ALL_NF4 = os.environ.get("ALL_NF4") == "1"
plan = [] if ALL_NF4 else json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))["int8"]
tok, model = load("offline", int8=plan)
print("权重方案：" + ("全 NF4" if ALL_NF4 else "加上 +1.0GB int8 方案"), flush=True)
guard = VramGuard()
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
pool = tok("".join(texts * 3), return_tensors="pt").input_ids
assert pool.shape[1] >= n_ctx, pool.shape
ids = pool[:, :n_ctx].cuda()

mtp = MTPHead(model)
max_len = n_ctx + n_gen + 32
mtp.new_cache(max_len)
cache = make_fixed_cache(model, max_len, kv_int8=True)
dev = "cuda"


def prefill(ids):
    t0 = time.perf_counter()
    for s in range(0, ids.shape[1], CHUNK):
        e = min(s + CHUNK, ids.shape[1])
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                          past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out.last_hidden_state


@torch.inference_mode()
def plain(n):
    reset_cache(cache)
    mtp.cache.reset()
    t_pre, h = prefill(ids)
    nxt = model.lm_head(h[:, -1:]).float().argmax(-1).reshape(1, 1)
    times = []
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hh = model.model(input_ids=nxt, position_ids=torch.tensor([[n_ctx + i]], device=dev),
                         past_key_values=cache, use_cache=True).last_hidden_state
        nxt = model.lm_head(hh[:, -1:]).float().argmax(-1).reshape(1, 1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    guard.check()
    return t_pre, statistics.median(times) * 1e3, times


@torch.inference_mode()
def verify_cost(k, reps=8):
    """Cost of the main-model forward on k tokens (what a k-token speculative verify pays)."""
    toks = torch.randint(1000, 20000, (1, k), device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    kv = [l for l in cache.layers if hasattr(l, "cur")]
    base_curs = [l.cur for l in kv]
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hh = model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
        model.lm_head(hh[:, -k:]).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        for layer, c in zip(kv, base_curs):
            layer.cur = c
            layer._set_views()
    return statistics.median(ts) * 1e3


print(f"\n=== {n_ctx} token 上下文，KV int8，MTP 头已加载 ===", flush=True)
t_pre, t1, _ = plain(n_gen)
print(f"普通解码：prefill {t_pre:.2f}s，{t1:.1f} ms/token（{1e3 / t1:.2f} tok/s）", flush=True)

with torch.inference_mode():
    reset_cache(cache)
    mtp.cache.reset()
    prefill(ids)
    t2 = verify_cost(2)
    t4 = verify_cost(4)
print(f"验证代价：1 token {t1:.1f} ms，2 token {t2:.1f} ms（x{t2 / t1:.2f}），"
      f"4 token {t4:.1f} ms（x{t4 / t1:.2f}）", flush=True)

with torch.inference_mode():
    toks, st = generate_speculative(model, mtp, ids, n_gen, cache, chunk_size=CHUNK)
dec = st["decode_s"]
spec_tok_s = st["tokens"] / dec
print(f"自投机解码（当前实现，1 草稿 = 验证 2 个）：{st['tokens']} token / {dec:.2f}s = "
      f"{spec_tok_s:.2f} tok/s；步数 {st['steps']}，命中 {st['accepted']}，回退 {st['redo']}，"
      f"普通补算 {st['plain_tokens']}", flush=True)
p = st["accepted"] / max(st["steps"], 1)

print(f"\n命中率 p = {p:.2f}；按实测代价推算各方案的理论吞吐（相对普通解码）：")
print(f"{'方案':<26}{'每步 token':>10}{'每步代价':>10}{'相对速度':>10}")
print(f"{'普通（验证 1 个）':<26}{1:>10.2f}{t1:>10.1f}{1.0:>10.2f}x")
# 2-token: commit 2 tokens, cost T2 + (1-p)*T2 when rejected (redo is another 2-token forward)
cost2 = t2 * (2 - p)
print(f"{'当前：1 草稿 / 验证 2':<26}{2:>10.2f}{cost2:>10.1f}{2 / cost2 * t1:>10.2f}x")
# 4-token: commit 1 + d1..d3 with geometric acceptance, redo on the first miss
exp4 = 1 + p + p ** 2 + p ** 3
cost4 = t4 * (1 + (1 - p ** 3) * 0.5)          # a miss near the end needs a partial redo
print(f"{'改为：3 草稿 / 验证 4':<26}{exp4:>10.2f}{cost4:>10.1f}{exp4 / cost4 * t1:>10.2f}x")
print(f"\n盈亏平衡：验证 4 个需要 p >= "
      f"{[round(x, 2) for x in [0.5, 0.6, 0.7, 0.8, 0.9] if (1 + x + x ** 2 + x ** 3) / t4 * t1 < (1 + x) / (t2 * (2 - x)) * t1][:1]}")
print(f"整卡峰值 {guard.peak / 1e9:.2f} GB")
guard.close()
