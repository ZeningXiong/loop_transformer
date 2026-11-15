"""Search a per-layer NF4/int8 mix that minimizes the error vs bf16 under a GPU-memory budget.

1. sensitivity: upgrade one group at a time (32 layers x {attn, mlp} + lm_head = 65 groups) to int8,
   measure the KL reduction on the SEARCH texts (bf16 references from eval_nf4_vs_bf16.py).
2. rank groups by KL reduction per extra byte; build the greedy prefix for several budgets and
   evaluate each on both the SEARCH texts and the HELD-OUT texts (never used for ranking).
3. spot check: for the top groups, compare int8 against bf16 to see how much int8 itself still loses.

Results -> mixed_search.json (resumable).
"""
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sn_loop_common import load, module_bytes, set_precision  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mixed_search.json")
SEARCH_TOKENS = 256
BUDGETS_GB = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
LM_CHUNK = 64


def load_set(texts_file, store, max_tokens, skip_prefix=None):
    out = []
    for line in open(os.path.join(HERE, texts_file), encoding="utf-8"):
        key = json.loads(line)["key"]
        if skip_prefix and key.startswith(skip_prefix):
            continue
        ref = torch.load(os.path.join(store, f"{key}.pt"))
        T = min(max_tokens, ref["ids"].shape[1])
        out.append((key, ref["ids"][:, :T].to("cuda"), ref["lp"][:T].contiguous()))  # refs stay in RAM (~2.5 GB total)
    return out


SEARCH = load_set("eval_texts.jsonl", r"D:\models\_eval_bf16_ref", SEARCH_TOKENS, skip_prefix="chat")
HELDOUT = load_set("eval_texts_heldout.jsonl", r"D:\models\_eval_bf16_ref_heldout", 384)

tok, model = load("offline")


@torch.inference_mode()
def evaluate(dataset):
    kl_sum = top_sum = n = 0.0
    for _key, ids, lp_ref16 in dataset:
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        for s in range(0, hidden.shape[1], LM_CHUNK):
            lp = torch.log_softmax(model.lm_head(hidden[:, s:s + LM_CHUNK]).float(), -1)[0]
            lr = lp_ref16[s:s + LM_CHUNK].to("cuda", non_blocking=True).float()
            kl_sum += (lr.exp() * (lr - lp)).sum().item()
            top_sum += (lr.argmax(-1) == lp.argmax(-1)).sum().item()
            n += lp.shape[0]
    return {"kl": kl_sum / n, "top1": top_sum / n, "positions": int(n)}


# groups
qnames = [n for n, m in model.named_modules() if type(m).__name__ == "Linear4bit"]
groups = {"lm_head": ["lm_head.weight"]}
for i in range(32):
    pre = f"model.language_model.layers.{i}."
    groups[f"L{i:02d}.attn"] = [n + ".weight" for n in qnames if n.startswith(pre) and ("_attn." in n)]
    groups[f"L{i:02d}.mlp"] = [n + ".weight" for n in qnames if n.startswith(pre) and ".mlp." in n]
assert sum(len(v) for v in groups.values()) == len(qnames) == 249, (sum(len(v) for v in groups.values()), len(qnames))


def set_group(g, prec):
    total = 0
    for name in groups[g]:
        total += set_precision(model, name, prec)
    # Params4bit <-> Linear4bit reference each other: free the old layers now, not at some later GC pass
    gc.collect()
    torch.cuda.empty_cache()
    return total


def group_nf4_bytes(g):
    return sum(module_bytes(model.get_submodule(n[:-7])) for n in groups[g])


res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
res.setdefault("groups", {})
t0 = time.time()
if "baseline" not in res:
    res["baseline"] = {"search": evaluate(SEARCH), "heldout": evaluate(HELDOUT)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
base = res["baseline"]
print(f"全 NF4 基线：搜索集 KL {base['search']['kl']:.4f} 首选 {base['search']['top1'] * 100:.1f}%；"
      f"验证集 KL {base['heldout']['kl']:.4f} 首选 {base['heldout']['top1'] * 100:.1f}%", flush=True)

# 1. sensitivity
gc.collect()
torch.cuda.empty_cache()
alloc0 = torch.cuda.memory_allocated() / 1e9
print(f"初始常驻显存 {alloc0:.2f} GB", flush=True)
for g in groups:
    if g in res["groups"]:
        continue
    nf4_b = group_nf4_bytes(g)
    int8_b = set_group(g, "int8")
    r = evaluate(SEARCH)
    set_group(g, "nf4")
    torch.cuda.empty_cache()
    gain = base["search"]["kl"] - r["kl"]
    alloc = torch.cuda.memory_allocated() / 1e9
    if alloc > alloc0 + 0.05:
        print(f"  !! 显存没有回到初始值：{alloc:.2f} GB（初始 {alloc0:.2f} GB）", flush=True)
    res["groups"][g] = {"kl": r["kl"], "gain": gain, "extra_bytes": int8_b - nf4_b, "names": groups[g]}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"  {g:9s} 升 int8：KL {r['kl']:.4f}（降 {gain:+.4f}），多占 {(int8_b - nf4_b) / 1e6:6.0f} MB，"
          f"性价比 {gain / ((int8_b - nf4_b) / 1e9):.4f}/GB  [{time.time() - t0:.0f}s]", flush=True)

# 2. greedy prefix by gain per byte
order = sorted(res["groups"], key=lambda g: -res["groups"][g]["gain"] / res["groups"][g]["extra_bytes"])
res["order"] = order
res["curve"] = []
current = []
used = 0
for budget in BUDGETS_GB:
    for g in order:
        if g in current or res["groups"][g]["gain"] <= 0:
            continue
        if used + res["groups"][g]["extra_bytes"] <= budget * 1e9:
            set_group(g, "int8")
            current.append(g)
            used += res["groups"][g]["extra_bytes"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    weights_gb = torch.cuda.memory_allocated() / 1e9
    s, h = evaluate(SEARCH), evaluate(HELDOUT)
    point = {"budget_gb": budget, "extra_gb": used / 1e9, "groups": list(current), "search": s, "heldout": h,
             "allocated_gb": weights_gb, "eval_peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    res["curve"].append(point)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"预算 +{budget:.2f}GB（实际 +{used / 1e9:.2f}，{len(current)} 组）：搜索集 KL {s['kl']:.4f} 首选 {s['top1'] * 100:.1f}%；"
          f"验证集 KL {h['kl']:.4f} 首选 {h['top1'] * 100:.1f}%；常驻显存 {weights_gb:.2f}GB", flush=True)

# 3. int8 vs bf16 spot check on the 3 most valuable non-lm_head groups (model currently holds the largest mix)
for g in current:
    set_group(g, "nf4")
torch.cuda.empty_cache()
spot = []
for g in [x for x in order if x != "lm_head"][:3]:
    set_group(g, "bf16")
    r16 = evaluate(SEARCH)
    set_group(g, "nf4")
    torch.cuda.empty_cache()
    spot.append({"group": g, "kl_bf16": r16["kl"], "kl_int8": res["groups"][g]["kl"], "kl_nf4": base["search"]["kl"]})
    print(f"抽查 {g}: 全NF4 {base['search']['kl']:.4f} | 该组 int8 {res['groups'][g]['kl']:.4f} | 该组 bf16 {r16['kl']:.4f}", flush=True)
res["spot_check_bf16"] = spot
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print(f"总耗时 {time.time() - t0:.0f}s")
