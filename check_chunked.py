"""Check chunked prefill: same results as one-shot prefill, and memory that no longer grows with the prompt."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sn_loop_common import VramExceeded, VramGuard, generate_chunked, load  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PLAN = sys.argv[1] if len(sys.argv) > 1 else None
plan = set(json.load(open(PLAN, encoding="utf-8"))["int8"]) if PLAN else ()
print("方案:", PLAN or "全 NF4", flush=True)
tok, model = load("offline", int8=plan)
guard = VramGuard()
text = "".join(json.loads(l)["text"] for l in open("eval_texts.jsonl", encoding="utf-8"))
ids_all = tok(text * 6, return_tensors="pt").input_ids.to("cuda")


def kl(a, b):
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return (la.exp() * (la - lb)).sum().item()


try:
    # 1. correctness
    chat = tok.apply_chat_template([{"role": "user", "content": "用三句话介绍一下杜甫，以及他最有名的一首诗。"}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cid = tok(chat, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        hf = model.generate(cid, max_new_tokens=40, do_sample=False)[0, cid.shape[1]:].tolist()
    mine, _ = generate_chunked(model, cid, 40, chunk_size=8, guard=guard)
    print(f"短对话：自写循环（每块 8 token）与 transformers generate 的 40 个 token 相同: {mine == hf[:len(mine)]}")
    print("   回答:", tok.decode(mine, skip_special_tokens=True)[:80])
    for T in [600, 1500]:
        ids = ids_all[:, :T]
        one, s1 = generate_chunked(model, ids, 32, chunk_size=T, guard=guard)
        for cs in [512, 256]:
            ch, s2 = generate_chunked(model, ids, 32, chunk_size=cs, guard=guard)
            same = sum(x == y for x, y in zip(one, ch))
            print(f"prompt {T}：一次性 vs 每块 {cs}：首 token 分布 KL {kl(s1['first_logits'], s2['first_logits']):.2e}，"
                  f"32 个生成 token 中前 {next((i for i, (x, y) in enumerate(zip(one, ch)) if x != y), len(ch))} 个完全相同（共相同 {same}）", flush=True)

    # 2. memory vs prompt length (chunk 512)
    for T in [2048, 4096, 8192, 16384, 32768]:
        if T > ids_all.shape[1]:
            break
        ids = ids_all[:, :T]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        guard.peak = 0
        new, st = generate_chunked(model, ids, 16, chunk_size=512, guard=guard)
        extra = (torch.cuda.max_memory_allocated() - base) / 1e9
        print(f"prompt {T:6d}（每块 512）：本进程显存增量峰值 {extra:.2f} GB，整卡峰值 {guard.peak / 1e9:.2f}/{guard.total / 1e9:.2f} GB，"
              f"prompt 处理 {st['prefill_s']:.1f}s，生成 {st['decode_tokens'] / st['decode_s']:.1f} tok/s", flush=True)
except VramExceeded as e:
    print("停止：", e)
finally:
    guard.close()
