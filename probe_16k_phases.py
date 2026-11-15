"""16K 下把耗时分段：单 token 步、2-token 步、自投机整步，同一个进程同一个缓存。"""
import json
import os
import statistics
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot, generate_speculative  # noqa: E402

n_ctx = 16384
plan = json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"), encoding="utf-8"))["int8"] \
    if os.environ.get("PLAN_INT8") == "1" else []
tok, model = load("offline", int8=plan)
dev = "cuda"
mtp = MTPHead(model)
mtp.new_cache(n_ctx + 64)
if os.environ.get("MTP_NF4") == "1":
    import bitsandbytes as bnb

    n_q = 0
    for name, child in list(mtp.named_modules()):
        if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
            parent_name, _, attr = name.rpartition(".")
            parent = mtp.get_submodule(parent_name) if parent_name else mtp
            new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=False,
                                    compute_dtype=torch.bfloat16, quant_type="nf4",
                                    compress_statistics=False, quant_storage=torch.uint8, device="meta")
            new.weight = bnb.nn.Params4bit(child.weight.data.detach(), requires_grad=False,
                                           quant_type="nf4", compress_statistics=False,
                                           quant_storage=torch.uint8)
            setattr(parent, attr, new.to(dev))
            n_q += 1
    print(f"MTP 头量化为 NF4（{n_q} 个线性层）", flush=True)
cache = make_fixed_cache(model, n_ctx + 64, kv_int8=True)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()


def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,power.draw,"
                           "utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    mtp.cache.reset()
    out = None
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                          past_key_values=cache, use_cache=True)
    return out.last_hidden_state


@torch.inference_mode()
def timed(k, reps=8):
    toks = torch.randint(1000, 20000, (1, k), device=dev)
    pos = torch.tensor([[n_ctx]], device=dev)
    ts = []
    for _ in range(2):
        snap.save()
        model.model(input_ids=toks, position_ids=pos, past_key_values=cache, use_cache=True)
        snap.restore()
    torch.cuda.synchronize()
    for _ in range(reps):
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = model.model(input_ids=toks, position_ids=pos, past_key_values=cache,
                        use_cache=True).last_hidden_state
        model.lm_head(h).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3


with torch.inference_mode():
    print(f"加载后：{smi()}")
    h = prefill()
    snap.save()
    print(f"prefill 后：{smi()}", flush=True)
    t1 = timed(1)
    print(f"  1 token 前向 {t1:6.1f} ms | {smi()}", flush=True)
    t2 = timed(2)
    print(f"  2 token 前向 {t2:6.1f} ms | {smi()}", flush=True)
    toks, st = generate_speculative(model, mtp, ids, 24, cache, chunk_size=256)
tps = st["tokens"] / st["decode_s"]
print(f"\n自投机 {st['tokens']} token / {st['decode_s']:.2f}s = {tps:.2f} tok/s"
      f"（{st['decode_s'] / st['steps'] * 1e3:.0f} ms/步）| 命中 {st['accepted']}/{st['steps']}")
print(f"对比：2-token 前向 {t2:.0f} ms；按 1 次验证 + 1 次 MTP 草稿 + 1 次 MTP 回填，"
      f"理论每步约 {t2 + 25:.0f} ms")
print(f"结束时：{smi()}")
