"""Whole-card VRAM peak (incl. other apps) and speed for a mixed plan at several prompt lengths."""
import gc, json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import psutil, torch
from sn_loop_common import load
plan = sys.argv[1]
names = set(json.load(open(plan, encoding="utf-8"))["int8"]) if plan != "none" else ()
peak = {"card": 0, "rss": 0}; proc = psutil.Process()
def watch():
    while True:
        f, t = torch.cuda.mem_get_info(); peak["card"] = max(peak["card"], t - f)
        peak["rss"] = max(peak["rss"], proc.memory_info().rss); time.sleep(0.02)
tok, model = load("offline", int8=names)
threading.Thread(target=watch, daemon=True).start()
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 3, return_tensors="pt").input_ids
for n in [2048, 4096, 8192]:
    ids = ids_all[:, :n].to("cuda")
    gc.collect(); torch.cuda.empty_cache(); peak["card"] = 0
    with torch.inference_mode():
        model.generate(ids, max_new_tokens=4, do_sample=False)  # warm/compile
        torch.cuda.synchronize(); t = time.time()
        out = model.generate(ids, max_new_tokens=32, min_new_tokens=32, do_sample=False)
        torch.cuda.synchronize(); dt = time.time() - t
    print(f"{plan:24s} prompt {n:5d}: 整卡显存峰值 {peak['card'] / 1e9:.2f}/{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB，"
          f"进程内存峰值 {peak['rss'] / 1e9:.2f} GB，(含 prompt 处理) 32 token 用时 {dt:.1f}s", flush=True)
