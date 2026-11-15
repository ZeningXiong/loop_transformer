"""8K 上下文下，把循环块跑 2 遍 vs 4 遍：输出与速度。

python test_loop4_8k.py [上下文] [生成 token 数]
"""
import copy
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import _fixed_kv_layer_class, load  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 24

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"
tc = model.config.text_config
tm = model.model.language_model
BASE = list(tm._loop_plan)                       # [0..3] + [4..27]*2 + [28..31]
CORE = list(range(4, 28))
PLANS = {
    "2 遍（原始）": BASE,
    "4 遍": [0, 1, 2, 3] + CORE * 4 + [28, 29, 30, 31],
}


def set_plan(plan):
    tm._loop_plan = list(plan)
    cfg = copy.deepcopy(tm.config)
    cfg.num_hidden_layers = len(plan)
    cfg.layer_types = [tm.config.layer_types[i] for i in plan]
    tm._loop_cache_config = cfg


def make_cache(max_len, kv_int8=True):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    cache = DynamicCache(config=tm._loop_cache_config)
    cache._loop_execution_plan = tm._loop_plan
    shape = (1, tc.num_key_value_heads, max_len, tc.head_dim)
    scratch = (torch.zeros(shape, dtype=torch.bfloat16, device=dev),
               torch.zeros(shape, dtype=torch.bfloat16, device=dev)) if kv_int8 else None
    FixedKVLayer = _fixed_kv_layer_class()
    n = 0
    for i, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:
            cache.layers[i] = FixedKVLayer(max_len, tc.num_key_value_heads, tc.head_dim,
                                           torch.bfloat16, dev, scratch=scratch)
            n += 1
    return cache, n


texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()


@torch.inference_mode()
def run(plan, tag):
    set_plan(plan)
    max_len = n_ctx + n_gen + 32
    cache, n_slots = make_cache(max_len)
    t0 = time.perf_counter()
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                          past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    pre = time.perf_counter() - t0
    nxt = model.lm_head(out.last_hidden_state[:, -1:]).float().argmax(-1).reshape(1, 1)
    gen, times = [], []
    for i in range(n_gen):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        o = model.model(input_ids=nxt, position_ids=torch.tensor([[n_ctx + i]], device=dev),
                        past_key_values=cache, use_cache=True)
        nxt = model.lm_head(o.last_hidden_state[:, -1:]).float().argmax(-1).reshape(1, 1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t1)
        gen.append(nxt.item())
    med = statistics.median(times) * 1e3
    print(f"\n=== {tag} ===  层执行 {len(plan)} 次，全注意力槽位 {n_slots} 个")
    print(f"  prefill {pre:.1f}s | 生成 {med:.0f} ms/token（{1e3 / med:.2f} tok/s）")
    print(f"  输出：{tok.decode(gen, skip_special_tokens=True)[:120]!r}")
    return med, tok.decode(gen, skip_special_tokens=True)


res = {}
for tag, plan in PLANS.items():
    res[tag] = run(plan, tag)
a, b = list(res.values())
print(f"\n4 遍 vs 2 遍：耗时 ×{b[0] / a[0]:.2f}；输出逐字相同：{a[1] == b[1]}")
set_plan(BASE)
