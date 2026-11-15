import os, sys, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import psutil, torch
p = psutil.Process()
peak = {"rss": 0}
def watch():
    while True:
        peak["rss"] = max(peak["rss"], p.memory_info().rss); time.sleep(0.05)
threading.Thread(target=watch, daemon=True).start()
from sn_loop_common import load
t = time.time(); tok, model = load("nf4")
print(f"加载完成 {time.time()-t:.0f}s：当前 RAM {p.memory_info().rss/1e9:.2f} GB，加载期间峰值 {peak['rss']/1e9:.2f} GB，Windows 峰值工作集 {p.memory_info().peak_wset/1e9:.2f} GB")
peak["rss"] = 0
ids = tok(tok.apply_chat_template([{"role":"user","content":"介绍一下长城。"}], tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").to("cuda")
with torch.inference_mode(): out = model.generate(**ids, max_new_tokens=64, do_sample=False)
print(f"生成期间 RAM 峰值 {peak['rss']/1e9:.2f} GB；显存峰值 {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
emb = model.model.language_model.embed_tokens.weight_cpu
print(f"其中 CPU embedding {emb.numel()*emb.element_size()/1e9:.2f} GB")
import collections
cnt = collections.Counter()
for n, prm in list(model.named_parameters()) + list(model.named_buffers()):
    cnt[(str(prm.device), str(prm.dtype))] += prm.numel() * prm.element_size()
for k, v in sorted(cnt.items(), key=lambda x: -x[1]): print("  参数", k, f"{v/1e9:.2f} GB")
