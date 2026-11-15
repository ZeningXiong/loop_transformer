"""任务级对比：现状（20 轮 Sinkhorn + torch.compile）vs 闭式解。

8 道数学题 + 2 个开放问题，贪心解码；同一模型、同一缓存，只切换 sinkhorn 实现。

python test_sinkhorn_quality.py
"""
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from fast_sinkhorn import enable  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

MATH = [
    ("13 乘以 17 再减去 29 等于多少？只回答数字。", "192"),
    ("48 乘以 25 等于多少？只回答数字。", "1200"),
    ("1234 加 5678 等于多少？只回答数字。", "6912"),
    ("15 乘以 15 再减去 17 等于多少？只回答数字。", "208"),
    ("391 除以 17 等于多少？只回答数字。", "23"),
    ("2 的 10 次方减去 100 等于多少？只回答数字。", "924"),
    ("37 乘以 43 等于多少？只回答数字。", "1591"),
    ("(85 - 37) 乘以 6 等于多少？只回答数字。", "288"),
]
OPEN = [
    ("用三句话介绍一下杜甫，以及他最有名的一首诗。", 90),
    ("写一个 Python 函数，判断一个字符串是否是回文（忽略大小写和非字母数字字符）。只给代码。", 110),
]

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"
cache = make_fixed_cache(model, 1024)
EOS = {248045, 248046}


@torch.inference_mode()
def generate(prompt, max_new, chunk=256):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    reset_cache(cache)
    T = ids.shape[1]
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                          past_key_values=cache, use_cache=True)
    nxt = model.lm_head(out.last_hidden_state[:, -1:]).float().argmax(-1).reshape(1, 1)
    gen = []
    for i in range(max_new):
        if nxt.item() in EOS:
            break
        gen.append(nxt.item())
        o = model.model(input_ids=nxt, position_ids=torch.tensor([[T + i]], device=dev),
                        past_key_values=cache, use_cache=True)
        nxt = model.lm_head(o.last_hidden_state[:, -1:]).float().argmax(-1).reshape(1, 1)
    return tok.decode(gen, skip_special_tokens=True).strip()


def norm(s):
    return re.sub(r"[^0-9]", "", s)


def run_all(tag):
    res = []
    t0 = time.perf_counter()
    for q, want in MATH:
        out = generate(q, 16)
        res.append(("math", q, want, out, norm(out) == norm(want) if want else None))
    for q, n in OPEN:
        res.append(("open", q, None, generate(q, n), None))
    print(f"[{tag}] {len(res)} 题，用时 {time.perf_counter() - t0:.0f}s", flush=True)
    return res


enable(False)                       # 现状
cur = run_all("现状（20 轮）")
enable(True)                        # 闭式解
fast = run_all("闭式解")
enable(False)

print(f"\n{'类型':<6}{'答案一致':<9}{'都答对':<8}{'现状对/闭式对':>14}")
math_cur = [r for r in cur if r[0] == "math"]
math_fast = [r for r in fast if r[0] == "math"]
same_math = sum(1 for a, b in zip(math_cur, math_fast) if a[3] == b[3])
ok_cur = sum(1 for r in math_cur if r[4])
ok_fast = sum(1 for r in math_fast if r[4])
print(f"{'数学':<6}{f'{same_math}/8':<9}{'—':<8}{f'{ok_cur}/8 -> {ok_fast}/8':>14}")
open_cur = [r for r in cur if r[0] == "open"]
open_fast = [r for r in fast if r[0] == "open"]
same_open = sum(1 for a, b in zip(open_cur, open_fast) if a[3] == b[3])
print(f"{'开放':<6}{f'{same_open}/2':<9}{'—':<8}{'':>14}")

print("\n逐题（数学）：")
for (q, want), a, b in zip(MATH, math_cur, math_fast):
    flag = "同" if a[3] == b[3] else "不同"
    print(f"  {q[:22]:<24} 期望 {want:<5} | 现状 {a[3][:12]:<12} | 闭式 {b[3][:12]:<12} [{flag}]")
print("\n开放问题（各取前 60 字）：")
for a, b in zip(open_cur, open_fast):
    print(f"  问：{a[1][:30]}")
    print(f"    现状：{a[3][:60].replace(chr(10), ' ')}")
    print(f"    闭式：{b[3][:60].replace(chr(10), ' ')}")
