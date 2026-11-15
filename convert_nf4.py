"""One-time offline conversion: SenseNova-Loop-9B-Preview (FP32 safetensors) -> NF4 checkpoint + raw bf16 embedding.

Streams one tensor at a time (read -> GPU -> quantize -> write small shard), so RAM stays low.
Every quantized tensor is checked bit-for-bit against nf4_reference.json, which recorded what
transformers' own online quantization produces; the converted model must behave identically.

Output (D:\\models\\SenseNova-Loop-9B-Preview-nf4):
  nf4-000NN.safetensors   quantized linears (uint8 packed + fp32 absmax) and all other text weights
  embed_tokens.bf16.bin   248320 x 4096 bf16, row i at byte offset i*8192 (read per token from SSD)
  quant_manifest.json     what is where, plus quantization parameters
  config / tokenizer / modeling code copied from the original directory
"""
import hashlib
import json
import os
import shutil
import sys
import threading
import time

import bitsandbytes as bnb
import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding="utf-8")
SRC = r"D:\models\SenseNova-Loop-9B-Preview"
DST = r"D:\models\SenseNova-Loop-9B-Preview-nf4"
HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_BYTES = 512 << 20
EMBED = "model.language_model.embed_tokens.weight"
COPY_FILES = ["config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json",
              "tokenizer_config.json", "modeling_sensenova_loop.py", "README.md", "README_CN.md"]

proc = psutil.Process()
peak = {"private": 0, "wset": 0}


def watch():
    while True:
        mi = proc.memory_info()
        peak["private"] = max(peak["private"], mi.private)
        peak["wset"] = max(peak["wset"], mi.rss)
        time.sleep(0.05)


threading.Thread(target=watch, daemon=True).start()


def sha(t):
    return hashlib.sha1(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


ref = json.load(open(os.path.join(HERE, "nf4_reference.json"), encoding="utf-8"))
weight_map = json.load(open(os.path.join(SRC, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
quantized = {n + ".weight": v for n, v in ref["quantized"].items()}
others = {n: v for n, v in ref["other"].items()
          if "visual" not in n and n != EMBED and n in weight_map}  # buffers (e.g. inv_freq) are not in files
skipped_buffers = [n for n, v in ref["other"].items() if "visual" not in n and n != EMBED and n not in weight_map]
DTYPES = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}

os.makedirs(DST, exist_ok=True)
for f in COPY_FILES:
    shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))

# group work by source shard so each file is opened once
todo = {}
for name in list(quantized) + list(others):
    todo.setdefault(weight_map[name], []).append(name)

manifest = {"format": "sensenova-loop-nf4-v1", "blocksize": 64, "quant_type": "nf4", "compress_statistics": False,
            "compute_dtype": "bfloat16", "quantized": {}, "other": {}, "shards": [],
            "embedding": {"file": "embed_tokens.bf16.bin", "rows": None, "dim": None, "dtype": "bfloat16"},
            "dropped": ["model.visual.*", "mtp.*"], "runtime_buffers": skipped_buffers}
buf, buf_bytes, shard_id = {}, 0, 0
mismatch = []


def flush():
    global buf, buf_bytes, shard_id
    if not buf:
        return
    fname = f"nf4-{shard_id:05d}.safetensors"
    save_file(buf, os.path.join(DST, fname))
    manifest["shards"].append(fname)
    for k in buf:
        base = k[:-len(".absmax")] if k.endswith(".absmax") else k
        section = "quantized" if base in quantized else "other"
        manifest[section].setdefault(base, {})["file"] = fname
    buf, buf_bytes, shard_id = {}, 0, shard_id + 1


def add(key, tensor):
    global buf_bytes
    buf[key] = tensor
    buf_bytes += tensor.numel() * tensor.element_size()
    if buf_bytes >= SHARD_BYTES:
        flush()


t0 = time.time()
done = 0
total = len(quantized) + len(others)
for src_file, names in sorted(todo.items()):
    with safe_open(os.path.join(SRC, src_file), framework="pt") as f:
        for name in names:
            if name in quantized:
                r = quantized[name]
                w = f.get_tensor(name).to(torch.bfloat16).to("cuda").contiguous()  # same path as from_pretrained
                packed, qs = bnb.functional.quantize_4bit(
                    w, blocksize=64, compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8)
                del w
                if sha(packed) != r["sha_data"] or sha(qs.absmax) != r["sha_absmax"]:
                    mismatch.append(name)
                add(name, packed.cpu())
                add(name + ".absmax", qs.absmax.cpu())
                manifest["quantized"].setdefault(name, {}).update(shape=list(qs.shape), packed_shape=list(packed.shape))
                del packed, qs
            else:
                dt = DTYPES[others[name]["dtype"]]
                add(name, f.get_tensor(name).to(dt).contiguous())
                manifest["other"].setdefault(name, {}).update(dtype=others[name]["dtype"])
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} 张量  {time.time() - t0:.0f}s  进程私有内存峰值 {peak['private'] / 1e9:.2f} GB", flush=True)
flush()
torch.cuda.empty_cache()

# embedding: FP32 in the source -> raw bf16 rows, written in 16k-row chunks
with safe_open(os.path.join(SRC, weight_map[EMBED]), framework="pt") as f:
    sl = f.get_slice(EMBED)
    rows, dim = sl.get_shape()
    with open(os.path.join(DST, "embed_tokens.bf16.bin"), "wb") as out:
        for s in range(0, rows, 16384):
            chunk = sl[s:min(s + 16384, rows)].to(torch.bfloat16).contiguous()
            out.write(chunk.view(torch.uint8).numpy().tobytes())
manifest["embedding"].update(rows=rows, dim=dim, row_bytes=dim * 2)

json.dump(manifest, open(os.path.join(DST, "quant_manifest.json"), "w", encoding="utf-8"), indent=1)
size = sum(os.path.getsize(os.path.join(DST, x)) for x in os.listdir(DST))
print(f"完成：{time.time() - t0:.0f}s，量化 {len(manifest['quantized'])} 层，其他 {len(manifest['other'])} 个张量，"
      f"{len(manifest['shards'])} 个分片，目录总大小 {size / 1e9:.2f} GB")
print(f"与参照逐位一致：{len(quantized) - len(mismatch)}/{len(quantized)}" + (f"，不一致: {mismatch[:5]}" if mismatch else ""))
print(f"转换期间进程私有内存峰值 {peak['private'] / 1e9:.2f} GB，工作集峰值（含文件映射页）{peak['wset'] / 1e9:.2f} GB")
