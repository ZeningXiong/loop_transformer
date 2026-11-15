"""Chunked prefill must give the same continuation as one-shot prefill (loop + linear-attn caches)."""
import json, os, sys, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import psutil, torch
from sn_loop_common import load
plan = sys.argv[1] if len(sys.argv) > 1 else None
names = set(json.load(open(plan, encoding="utf-8"))["int8"]) if plan else ()
tok, model = load("offline", int8=names)
peak = {"card": 0, "rss": 0}; proc = psutil.Process()
def watch():
    while True:
        f, t = torch.cuda.mem_get_info(); peak["card"] = max(peak["card"], t - f)
        peak["rss"] = max(peak["rss"], proc.memory_info().rss); time.sleep(0.02)
threading.Thread(target=watch, daemon=True).start()
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 3, return_tensors="pt").input_ids
def run(n, chunk, new=24):
    ids = ids_all[:, :n].to("cuda")
    torch.cuda.empty_cache(); peak["card"] = 0; peak["rss"] = 0
    with torch.inference_mode():
        torch.cuda.synchronize(); t = time.time()
        out = model.generate(ids, max_new_tokens=new, min_new_tokens=new, do_sample=False, prefill_chunk_size=chunk)
        torch.cuda.synchronize()
    return out[0, n:].tolist(), time.time() - t, peak["card"] / 1e9, peak["rss"] / 1e9
run(64, None, 4); run(64, 32, 4)  # warm-up
for n in [600, 1024]:
    a = run(n, None); b = run(n, 256)
    print(f"prompt {n}: 不分块 {a[1]:.1f}s 整卡 {a[2]:.2f}GB | 分块256 {b[1]:.1f}s 整卡 {b[2]:.2f}GB | 生成的 24 个 token 相同: {a[0] == b[0]}"
          + ("" if a[0] == b[0] else f"  首个不同位置 {next(i for i, (x, y) in enumerate(zip(a[0], b[0])) if x != y)}"), flush=True)
for n in [4096, 8192, 16384]:
    r = run(n, 512, 16)
    print(f"prompt {n:5d} 分块512: 用时 {r[1]:.1f}s，整卡显存峰值 {r[2]:.2f}GB，进程内存峰值 {r[3]:.2f}GB", flush=True)
