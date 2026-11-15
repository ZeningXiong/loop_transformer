"""二分定位：2-token 图捕获在哪个配置下失败。"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from cudagraph_spec import Graph2Decoder  # noqa: E402
from sn_loop_common import load, reset_cache  # noqa: E402

tok, model = load("offline", int8=[])
ids = tok("hello world, this is a long prompt. " * 80, return_tensors="pt").input_ids[:, :512].cuda()
ids = ids.repeat(1, 2) if ids.shape[1] < 512 else ids[:, :512]
ids = ids[:, :512].contiguous()
dev = "cuda"


@torch.inference_mode()
def try_capture(qlen, kv_int8, prefill_len=512):
    dec = Graph2Decoder(model, prefill_len + 32, kv_int8=kv_int8, qlen=qlen)
    reset_cache(dec.cache)
    for s in range(0, prefill_len, 256):
        model.model(input_ids=ids[:, s:s + 256], position_ids=torch.arange(s, s + 256, device=dev)[None],
                    past_key_values=dec.cache, use_cache=True)
    try:
        dec.step(torch.randint(1000, 20000, (1, qlen), device=dev), prefill_len)
        torch.cuda.synchronize()
        out = "ok"
    except Exception as e:
        out = f"{type(e).__name__}: {str(e)[:70]}"
    del dec
    torch.cuda.empty_cache()
    return out


for qlen in (1, 2):
    for kv8 in (False, True):
        print(f"qlen={qlen} kv_int8={kv8}: {try_capture(qlen, kv8)}", flush=True)
