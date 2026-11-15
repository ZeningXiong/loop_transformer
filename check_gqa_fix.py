import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
import transformers.integrations.sdpa_attention as sa
from sn_loop_common import load
tok, model = load("offline")
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 3, return_tensors="pt").input_ids
orig = sa.use_gqa_in_sdpa
def run(T):
    ids = ids_all[:, :T].to("cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    with torch.inference_mode():
        torch.cuda.synchronize(); t = time.time()
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1); torch.cuda.synchronize()
    return out.logits.float(), (torch.cuda.max_memory_allocated() - base) / 1e9, time.time() - t
for T in [512, 2048, 4096, 8192]:
    sa.use_gqa_in_sdpa = orig
    a = run(T) if T <= 2048 else (None, float("nan"), float("nan"))
    sa.use_gqa_in_sdpa = lambda *args, **kw: False   # repeat K/V -> memory-efficient SDPA kernel
    run(64)
    b = run(T)
    diff = (a[0] - b[0]).abs().max().item() if a[0] is not None else float("nan")
    print(f"T={T:5d}: 原来 峰值增量 {a[1]:5.2f} GB, {a[2]:5.2f}s | 修复后 {b[1]:5.2f} GB, {b[2]:5.2f}s | 最后位置 logit 最大差 {diff:.3f}", flush=True)
