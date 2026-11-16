"""测"最近 W 个 token 保持 bf16"对 16K 下 int8 KV 精度的改善。

两个阶段分开跑（各自一个进程，避免显存叠加）：
  python test_kv_window.py ref     # 精确 bf16 KV，把最后 TAIL 个位置的 logits 存盘
  python test_kv_window.py cmp     # int8 / int8+窗口，和存盘的参考比 KL
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402

dev = "cuda"
N_CTX, CHUNK, TAIL = 16384, 256, 96
REF = os.path.join(HERE, "_kv_ref.pt")
phase = sys.argv[1] if len(sys.argv) > 1 else "cmp"
only = int(sys.argv[2]) if len(sys.argv) > 2 else -1

tok, model = load("offline", int8=[])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
guard = VramGuard()
print(f"上下文 {ids.shape[1]} token，阶段 {phase}", flush=True)


@torch.inference_mode()
def run(recent, qint):
    cache = make_fixed_cache(model, N_CTX + 8, kv_int8=qint, recent=recent)
    reset_cache(cache)
    t0 = time.perf_counter()
    logits = None
    for s in range(0, N_CTX, CHUNK):
        e = min(s + CHUNK, N_CTX)
        pos = torch.arange(s, e, device=dev)[None]
        h = model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=cache,
                        use_cache=True).last_hidden_state
        if e == N_CTX:
            rows = [model.lm_head(h[:, r:r + 32]).float() for r in range(0, e - s, 32)]
            logits = torch.cat(rows, 0)[-TAIL:]
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    del cache
    torch.cuda.empty_cache()
    return logits, t


with torch.inference_mode():
    if phase == "ref":
        ref, t = run(0, False)
        torch.save(ref.to(torch.float16).cpu(), REF)
        print(f"精确 bf16 KV：prefill {t:.1f}s，参考 logits 已存盘 | 整卡 {guard.peak / 1e9:.2f} GB",
              flush=True)
    else:
        ref = torch.load(REF, weights_only=True).to(dev).float()
        lp = torch.log_softmax(ref, -1)
        pairs = ([("single", only)] if only >= 0 else
                 [("int8（现状）", 0), ("延迟编码 N=8", 8),
                  ("延迟编码 N=32", 32), ("延迟编码 N=128", 128)])
        for label, recent in pairs:
            cur, t = run(recent, True)
            lq = torch.log_softmax(cur, -1)
            kl = (lp.exp() * (lp - lq)).sum(-1).mean().item()
            top1 = (ref.argmax(-1) == cur.argmax(-1)).float().mean().item()
            print(f"{label:<18} prefill {t:5.1f}s | KL {kl:.4f} | 首选一致 {top1 * 100:.2f}% | "
                  f"整卡 {guard.peak / 1e9:.2f} GB", flush=True)
            del cur, lq
            torch.cuda.empty_cache()
            guard.check()
guard.close()
