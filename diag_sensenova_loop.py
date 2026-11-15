"""Diagnose SenseNova-Loop-9B-Preview answers: quantization / thinking mode / tokenizer regex fix.

usage: python -I diag_sensenova_loop.py --set math --quant nf4 [--think] [--fix-regex] [--max-new 2048]
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script dir
from sn_loop_common import load

import threading

import psutil

_proc = psutil.Process()
_peak = {"wset": 0}


def _watch():
    while True:
        _peak["wset"] = max(_peak["wset"], _proc.memory_info().rss)
        time.sleep(0.05)


threading.Thread(target=_watch, daemon=True).start()

sys.stdout.reconfigure(encoding="utf-8")

SETS = {
    "basic": [
        ("13 乘以 17 再减去 29 等于多少？只回答数字。", "192"),
        ("用三句话介绍一下杜甫，以及他最有名的一首诗。", None),
        ("写一个 Python 函数，判断一个字符串是否是回文（忽略大小写和非字母数字字符）。只给代码。", None),
    ],
    "math": [
        ("13 乘以 17 再减去 29 等于多少？只回答数字。", "192"),
        ("48 乘以 25 等于多少？只回答数字。", "1200"),
        ("1234 加 5678 等于多少？只回答数字。", "6912"),
        ("15 乘以 15 再减去 17 等于多少？只回答数字。", "208"),
        ("391 除以 17 等于多少？只回答数字。", "23"),
        ("2 的 10 次方减去 100 等于多少？只回答数字。", "924"),
        ("37 乘以 43 等于多少？只回答数字。", "1591"),
        ("(85 - 37) 乘以 6 等于多少？只回答数字。", "288"),
    ],
}

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="math", choices=SETS)
ap.add_argument("--quant", default="nf4", choices=["offline", "nf4", "bf16"])
ap.add_argument("--think", action="store_true")
ap.add_argument("--fix-regex", action="store_true")
ap.add_argument("--max-new", type=int, default=None)
ap.add_argument("--no-compile", action="store_true", help="keep the original (slow) Sinkhorn")
ap.add_argument("--plan", default=None, help="mixed_plan_*.json: layers to load as int8")
ap.add_argument("--out", default=None)
args = ap.parse_args()
max_new = args.max_new or (2048 if args.think else 300)
plan = json.load(open(args.plan, encoding="utf-8")) if args.plan else None
tag = f"{args.set}_{args.quant}{('-mix' + str(plan['budget_gb'])) if plan else ''}_{'think' if args.think else 'nothink'}{'_fixregex' if args.fix_regex else ''}"
out_path = args.out or f"diag_{tag}.jsonl"

t0 = time.time()
tok, model = load(args.quant, fix_regex=args.fix_regex, compile_sinkhorn=not args.no_compile,
                  int8=set(plan["int8"]) if plan else ())
print(f"[{tag}] 加载 {time.time() - t0:.0f} 秒，加载期间工作集峰值 {_peak['wset'] / 1e9:.2f} GB，加载后 {_proc.memory_info().rss / 1e9:.2f} GB", flush=True)
_peak["wset"] = 0

first_dev = "cuda:0"
right = total = 0
with open(out_path, "w", encoding="utf-8") as fo:
    for q, expect in SETS[args.set]:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
            enable_thinking=args.think,
        )
        inputs = tok(prompt, return_tensors="pt").to(first_dev)
        torch.cuda.synchronize()
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        new = out[0, inputs.input_ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=True).strip()
        final = text.split("</think>")[-1].strip()
        ok = None
        if expect is not None:
            nums = re.findall(r"-?\d+(?:\.\d+)?", final.replace(",", ""))
            ok = bool(nums) and nums[-1] == expect
            total += 1
            right += ok
        rec = dict(q=q, expect=expect, ok=ok, final=final, full=text, tokens=len(new), sec=round(dt, 1),
                   tok_per_s=round(len(new) / dt, 2))
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush()
        mark = "" if ok is None else ("✅" if ok else f"❌(应为 {expect})")
        short = final if len(final) < 160 else final[:160] + "…"
        print(f"{mark} {q}\n   → {short}\n   [{len(new)} tokens, {dt:.1f} 秒, {len(new) / dt:.1f} tok/s]", flush=True)

if total:
    print(f"[{tag}] 正确 {right}/{total}")
print(f"[{tag}] 峰值显存 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB；生成期间内存工作集峰值 {_peak['wset'] / 1e9:.2f} GB")
emb = model.model.language_model.embed_tokens
if hasattr(emb, "stats") and emb.stats["rows"]:
    st = emb.stats
    print(f"[{tag}] SSD 读 embedding：{st['calls']} 次前向，{st['rows']} 行，共 {st['read_s'] * 1e3:.1f} 毫秒，平均每行 {st['read_s'] / st['rows'] * 1e3:.3f} 毫秒")
