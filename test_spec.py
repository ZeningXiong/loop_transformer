"""Self-speculative decoding test: pick the MTP wiring, then compare speed/output with plain greedy decoding."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sn_loop_common import (VramExceeded, VramGuard, generate_chunked, load, make_fixed_cache,  # noqa: E402
                            reset_cache, use_gqa_fold_attention, warm_up)
from sn_spec import MTPHead, generate_speculative, get_tap, small_batch_gemv  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
NEW = 200

tok, model = load("offline")
use_gqa_fold_attention(True)
guard = VramGuard()
tap = get_tap(model)
try:
    warm_up(model)
    cache = make_fixed_cache(model, 4096)
    mtp = MTPHead(model)
    mtp.new_cache(4096)
    print(f"MTP 头加载完成；整卡 {(guard.total - torch.cuda.mem_get_info()[0]) / 1e9:.2f} GB", flush=True)

    # 1. which wiring is right? teacher-forced acceptance on real text
    text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
    ids = tok(text, return_tensors="pt").input_ids[:, :1024].to("cuda")
    T = ids.shape[1]
    pre, post, target = [], [], []
    reset_cache(cache)
    with torch.inference_mode():
        for s in range(0, T, 256):
            pos = torch.arange(s, min(s + 256, T), device="cuda")[None]
            model.model(input_ids=ids[:, s:s + 256], position_ids=pos, past_key_values=cache, use_cache=True)
            pre.append(tap.pre.clone()); post.append(tap.post.clone())
            target.append(torch.cat([model.lm_head(tap.post[:, r:r + 64]).float().argmax(-1)
                                     for r in range(0, tap.post.shape[1], 64)], 1))
    pre, post, target = torch.cat(pre, 1), torch.cat(post, 1), torch.cat(target, 1)[0]
    best = None
    for src in ["post_norm", "pre_norm"]:
        for emb_first in [True, False]:
            mtp.hidden_source, mtp.embed_first = src, emb_first
            mtp.cache.reset()
            h = (pre if src == "pre_norm" else post)
            preds = []
            for s in range(0, T - 1, 256):
                e = min(s + 256, T - 1)
                lg = mtp(h[:, s:e], ids[:, s + 1:e + 1], torch.arange(s + 1, e + 1, device="cuda")[None], all_rows=True)
                preds.append(lg[0].float().argmax(-1))
            preds = torch.cat(preds)                     # MTP guess for token i+2, i = 0..T-2
            acc = (preds[:-1] == target[1:T - 1]).float().mean().item()  # vs main model's own choice
            print(f"  接法 hidden={src:9s} embedding在前={emb_first!s:5s}：猜中主模型选择的比例 {acc * 100:5.1f}%", flush=True)
            if best is None or acc > best[0]:
                best = (acc, src, emb_first)
    mtp.hidden_source, mtp.embed_first = best[1], best[2]
    print(f"采用：hidden={best[1]}，embedding在前={best[2]}", flush=True)

    # 2. plain greedy vs self-speculative on real chat prompts
    prompts = [
        ("中文介绍", "用三句话介绍一下杜甫，以及他最有名的一首诗。", False),
        ("写代码", "写一个 Python 函数实现快速排序，并用两三句话说明它的时间复杂度。", False),
        ("英文解释", "Explain the difference between TCP and UDP in a few short paragraphs.", False),
        ("推理(思考)", "一个水池有甲乙两个进水管，甲管单独注满要6小时，乙管单独注满要4小时。两管同时开，多久注满？", True),
    ]
    tot = {"base_t": 0.0, "base_n": 0, "spec_t": 0.0, "spec_n": 0}
    for name, q, think in prompts:
        p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=think)
        pid = tok(p, return_tensors="pt").input_ids.to("cuda")
        base, sb = generate_chunked(model, pid, NEW, guard=guard, cache=cache)
        guard.check()
        spec, ss = generate_speculative(model, mtp, pid, NEW, cache)
        guard.check()
        n = min(len(base), len(spec))
        div = next((i for i in range(n) if base[i] != spec[i]), n)
        b_tps = sb["decode_tokens"] / sb["decode_s"]
        s_tps = (ss["tokens"] - 1) / ss["decode_s"]
        tot["base_t"] += sb["decode_s"]; tot["base_n"] += sb["decode_tokens"]
        tot["spec_t"] += ss["decode_s"]; tot["spec_n"] += ss["tokens"] - 1
        print(f"[{name}] 普通 {b_tps:.1f} tok/s | 自投机 {s_tps:.1f} tok/s（×{s_tps / b_tps:.2f}），"
              f"草稿命中 {ss['accepted']}/{ss['steps']} = {ss['accepted'] / max(ss['steps'], 1) * 100:.0f}%，"
              f"其中 {ss['plain_tokens']} 个 token 退回普通生成；"
              f"两者输出前 {div}/{n} 个 token 相同", flush=True)
        if div < n:
            print(f"    普通：…{tok.decode(base[max(0, div - 8):div + 8])!r}\n    投机：…{tok.decode(spec[max(0, div - 8):div + 8])!r}")
    b, s_ = tot["base_n"] / tot["base_t"], tot["spec_n"] / tot["spec_t"]
    print(f"合计：普通 {b:.1f} tok/s，自投机 {s_:.1f} tok/s，×{s_ / b:.2f}；整卡峰值 {guard.peak / 1e9:.2f} GB")
except VramExceeded as e:
    print("停止：", e)
finally:
    guard.close()
