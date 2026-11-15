"""One-time offline conversion: all 249 NF4-candidate linear weights -> weight-only int8 (per-output-row scale).

int8 value q = round(w / s), s = max|w_row| / 127, stored as int8 [out, in] + fp32 scale [out].
Streams one tensor (or a row chunk of lm_head) at a time, so RAM stays small.
Output: D:\\models\\SenseNova-Loop-9B-Preview-nf4\\int8-000NN.safetensors + int8_manifest.json
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding="utf-8")
SRC = r"D:\models\SenseNova-Loop-9B-Preview"
DST = r"D:\models\SenseNova-Loop-9B-Preview-nf4"
SHARD_BYTES = 512 << 20
ROW_CHUNK = 16384

nf4 = json.load(open(os.path.join(DST, "quant_manifest.json"), encoding="utf-8"))
weight_map = json.load(open(os.path.join(SRC, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
names = list(nf4["quantized"])
man = {"format": "int8-weight-only-rowwise-v1", "tensors": {}, "shards": []}
buf, nbytes, sid = {}, 0, 0


def flush():
    global buf, nbytes, sid
    if buf:
        fn = f"int8-{sid:05d}.safetensors"
        save_file(buf, os.path.join(DST, fn))
        man["shards"].append(fn)
        for k in buf:
            man["tensors"].setdefault(k.rsplit(".", 1)[0] if k.endswith(".scale") else k, {})["file"] = fn
        buf, nbytes, sid = {}, 0, sid + 1


def add(k, t):
    global nbytes
    buf[k] = t
    nbytes += t.numel() * t.element_size()
    if nbytes >= SHARD_BYTES:
        flush()


def quant_rows(w):  # w: fp32 on GPU [rows, in]
    s = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(w / s[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, s


t0 = time.time()
max_err = 0.0
by_file = {}
for n in names:
    by_file.setdefault(weight_map[n], []).append(n)
for fname, ns in sorted(by_file.items()):
    with safe_open(os.path.join(SRC, fname), framework="pt") as f:
        for n in ns:
            sl = f.get_slice(n)
            rows = sl.get_shape()[0]
            qs, ss = [], []
            for r in range(0, rows, ROW_CHUNK):
                w = sl[r:min(r + ROW_CHUNK, rows)].to("cuda", torch.float32)
                q, s = quant_rows(w)
                err = ((q.float() * s[:, None] - w).abs() / s[:, None]).max().item()  # in units of one step
                max_err = max(max_err, err)
                qs.append(q.cpu())
                ss.append(s.cpu())
                del w, q, s
            add(n, torch.cat(qs))
            add(n + ".scale", torch.cat(ss))
            man["tensors"].setdefault(n, {})["shape"] = [rows, sl.get_shape()[1]]
flush()
json.dump(man, open(os.path.join(DST, "int8_manifest.json"), "w", encoding="utf-8"), indent=1)
size = sum(os.path.getsize(os.path.join(DST, s)) for s in man["shards"])
print(f"完成 {len(names)} 个张量，{len(man['shards'])} 个分片，{size / 1e9:.2f} GB，{time.time() - t0:.0f}s；"
      f"最大舍入误差 {max_err:.3f} 个量化步长（应 ≤ 0.5）")
