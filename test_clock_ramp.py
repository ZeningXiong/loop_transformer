"""同一个进程里，冷启动 vs 预热后的单步耗时，同时打印 GPU 频率/温度。

用来判断之前 9.68 vs 4.22 tok/s 的差到底是"机器被别的程序占了"还是"GPU 频率没升上去"。
"""
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
from sn_spec import _Snapshot  # noqa: E402


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.current.memory,"
                          "temperature.gpu,power.draw,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return out


tok, model = load("offline", int8=[])
dev = "cuda"
n_ctx = 200
cache = make_fixed_cache(model, 256)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    for s in range(0, n_ctx, 256):
        e = min(s + 256, n_ctx)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)


@torch.inference_mode()
def steps(n=10):
    nxt = torch.tensor([[1000]], device=dev)
    ts = []
    for i in range(n):
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = model.model(input_ids=nxt, position_ids=torch.tensor([[n_ctx]], device=dev),
                        past_key_values=cache, use_cache=True)
        model.lm_head(o.last_hidden_state[:, -1:]).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3


print(f"启动时：{smi()}")
prefill()
print(f"prefill 后：{smi()}")
for r in range(6):
    ms = steps(10)
    print(f"  第 {r + 1} 轮（10 步）：{ms:6.1f} ms/step | {smi()}", flush=True)
