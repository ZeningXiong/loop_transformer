"""Split the whole-card peak into: tensors in use / PyTorch cached-but-free / everything else."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from sn_loop_common import VramExceeded, VramGuard, generate_chunked, load
free0, total = torch.cuda.mem_get_info()
print(f"启动前整卡已用（其他程序）{(total - free0) / 1e9:.2f} GB")
tok, model = load("offline")
guard = VramGuard()
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 6, return_tensors="pt").input_ids.to("cuda")
generate_chunked(model, ids_all[:, :600], 4, chunk_size=512, guard=guard)  # warm-up (Triton compile etc.)
try:
    for T in [4096, 8192]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); guard.peak = 0
        new, st = generate_chunked(model, ids_all[:, :T], 16, chunk_size=512, guard=guard)
        alloc, resv = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9
        stats = torch.cuda.memory_stats()
        print(f"T={T}: 整卡峰值 {guard.peak / 1e9:.2f} GB = PyTorch 预留峰值 {resv:.2f}（其中张量实际在用 {alloc:.2f}，"
              f"缓存空闲 {resv - alloc:.2f}）+ 其他 {guard.peak / 1e9 - resv:.2f}（CUDA 运行环境/其他程序）；"
              f"显存分配次数 {stats['allocation.all.allocated']}，向显卡申请新块 {stats['segment.all.allocated']} 次", flush=True)
except VramExceeded as e:
    print("停止：", e)
finally:
    guard.close()
