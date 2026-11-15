"""任务级对比（开思考）：现状（20 轮 Sinkhorn）vs 闭式解。

python test_sinkhorn_quality_think.py
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
OPEN = [("用三句话介绍一下杜甫，以及他最有名的一首诗。", 160)]

tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
dev = "cuda"
cache = make_fixed_cache(model, 2048)
EOS = {248045, 248046}


@torch.inference_mode()
def generate(prompt, max_new, chunk=256):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
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
    return gen, tok.decode(gen, skip_special_tokens=True).strip()


def last_num(s):
    nums = re.findall(r"\d+", s.replace(",", ""))
    return nums[-1] if nums else ""


def run_all(tag):
    out = []
    t0 = time.perf_counter()
    for q, want in MATH:
        ids, txt = generate(q, 256)
        out.append(("math", q, want, txt, last_num(txt), len(ids)))
    for q, n in OPEN:
        ids, txt = generate(q, n)
        out.append(("open", q, None, txt, "", len(ids)))
    print(f"[{tag}] {len(out)} 题，{sum(r[5] for r in out)} 个 token，用时 "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return out


enable(False)
cur = run_all("现状（20 轮）")
enable(True)
fast = run_all("闭式解")
enable(False)

m_cur = [r for r in cur if r[0] == "math"]
m_fast = [r for r in fast if r[0] == "math"]
print(f"\n{'题':<26}{'期望':>6}{'现状':>7}{'闭式':>7}{'现状对':>7}{'闭式对':>7}{'思考 token':>11}")
for a, b in zip(m_cur, m_fast):
    print(f"{a[1][:24]:<26}{a[2]:>6}{a[4]:>7}{b[4]:>7}"
          f"{('✓' if a[4] == a[2] else '✗'):>7}{('✓' if b[4] == b[2] else '✗'):>7}"
          f"{f'{a[5]}/{b[5]}':>11}")
ok_c = sum(1 for r in m_cur if r[4] == r[2])
ok_f = sum(1 for r in m_fast if r[4] == r[2])
same = sum(1 for a, b in zip(m_cur, m_fast) if a[4] == b[4])
print(f"\n正确数：现状 {ok_c}/8，闭式 {ok_f}/8；最终答案一致 {same}/8")

o_cur = [r for r in cur if r[0] == "open"][0]
o_fast = [r for r in fast if r[0] == "open"][0]
print(f"开放问题：现状 {o_cur[5]} token，闭式 {o_fast[5]} token，逐字相同 {o_cur[3] == o_fast[3]}")
print(f"  现状：{o_cur[3][:80].replace(chr(10), ' ')}")
print(f"  闭式：{o_fast[3][:80].replace(chr(10), ' ')}")
