import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
import transformers.integrations.sdpa_attention as sa
from sn_loop_common import load
plan = set(json.load(open("mixed_plan_2.0.json", encoding="utf-8"))["int8"])
tok, model = load("offline", int8=plan)
data = []
for line in open("eval_texts_heldout.jsonl", encoding="utf-8"):
    k = json.loads(line)["key"]; r = torch.load(os.path.join(r"D:\models\_eval_bf16_ref_heldout", f"{k}.pt"))
    data.append((r["ids"].to("cuda"), r["lp"]))
@torch.inference_mode()
def ev():
    kl = top = n = 0.0
    for ids, lpr in data:
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state
        for s in range(0, h.shape[1], 64):
            lp = torch.log_softmax(model.lm_head(h[:, s:s + 64]).float(), -1)[0]
            lr = lpr[s:s + 64].to("cuda").float()
            kl += (lr.exp() * (lr - lp)).sum().item(); top += (lr.argmax(-1) == lp.argmax(-1)).sum().item(); n += lp.shape[0]
    return kl / n, top / n
orig = sa.use_gqa_in_sdpa
a = ev()
sa.use_gqa_in_sdpa = lambda *x, **k: False
b = ev()
print(f"验证集（+2.0GB 方案 vs bf16）：原注意力实现 KL {a[0]:.4f} 首选 {a[1] * 100:.1f}% | 省显存实现 KL {b[0]:.4f} 首选 {b[1] * 100:.1f}%")
