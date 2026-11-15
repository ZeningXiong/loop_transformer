"""Track VRAM used outside PyTorch's allocator, phase by phase, for an 8192-token prompt."""
import json, os, sys, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from sn_loop_common import load
tok, model = load("offline")
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids = tok(text * 6, return_tensors="pt").input_ids[:, :8192].to("cuda")
def outside():
    torch.cuda.synchronize(); f, t = torch.cuda.mem_get_info()
    return (t - f - torch.cuda.memory_reserved()) / 1e9
peak_out = {"v": 0.0, "at": ""}
phase = {"v": ""}
def watch():
    while True:
        f, t = torch.cuda.mem_get_info(); o = (t - f - torch.cuda.memory_reserved()) / 1e9
        if o > peak_out["v"]: peak_out.update(v=o, at=phase["v"])
        time.sleep(0.005)
threading.Thread(target=watch, daemon=True).start()
print(f"加载后：PyTorch 以外 {outside():.2f} GB")
cache = None
with torch.inference_mode():
    for i, s in enumerate(range(0, 8192, 512)):
        phase["v"] = f"第 {i + 1} 块 prompt（位置 {s}）"
        out = model(input_ids=ids[:, s:s + 512], position_ids=torch.arange(s, s + 512, device="cuda")[None],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        if i in (0, 3, 7, 11, 15): print(f"  第 {i + 1:2d} 块后：PyTorch 以外 {outside():.2f} GB，期间峰值 {peak_out['v']:.2f}（{peak_out['at']}）", flush=True)
    phase["v"] = "逐 token 生成"
    nxt = out.logits[:, -1].argmax(-1)
    for j in range(16):
        out = model(input_ids=nxt[:, None], position_ids=torch.tensor([[8192 + j]], device="cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values; nxt = out.logits[:, -1].argmax(-1)
print(f"生成后：PyTorch 以外 {outside():.2f} GB，全程峰值 {peak_out['v']:.2f} GB（{peak_out['at']}）")
del cache, out; torch.cuda.empty_cache(); time.sleep(1)
print(f"释放 cache 并清空缓存后：PyTorch 以外 {outside():.2f} GB")
