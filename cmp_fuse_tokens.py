"""最小对照 2：分别开 RMSNorm / SiLU / 两个都开，看是哪一个让输出变了。"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

import fused_rmsnorm  # noqa: E402
from fused_silu import enable_silu  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

N_CTX, N_GEN = 16384, 4
dev = "cuda"
tok, model = load("offline", int8=[])
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()
snap = _Snapshot(cache)

with torch.inference_mode():
    reset_cache(cache)
    h = None
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    pre = model.lm_head(h[:, -1:]).float()[0, -1].clone()
    snap.save()


@torch.inference_mode()
def run(n=N_GEN):
    snap.save()
    logits = [pre.clone()]
    nxt = pre.argmax().view(1, 1)
    for i in range(n):
        h2 = model.model(input_ids=nxt, position_ids=torch.tensor([[N_CTX + i]], device=dev),
                         past_key_values=cache, use_cache=True).last_hidden_state
        cur = model.lm_head(h2).float()[0, -1]
        logits.append(cur.clone())
        nxt = cur.argmax().view(1, 1)
    snap.restore()
    return logits


def set_mode(rn, silu):
    fused_rmsnorm.enable(rn)
    enable_silu(silu)


ref = None
for name, rn, silu in (("基线", False, False), ("基线（重跑）", False, False),
                       ("只 RMSNorm", True, False),
                       ("只 SiLU", False, True), ("两个都开", True, True)):
    set_mode(rn, silu)
    lg = run()
    if ref is None:
        ref = lg
        print(f"{name:<10}（参考）")
        continue
    d = [max(abs(a - b).max().item() for a, b in zip(ref, lg)) for _ in (0,)]
    step1 = (ref[1] - lg[1]).abs().max().item()
    top = [a.argmax().item() == b.argmax().item() for a, b in zip(ref, lg)]
    print(f"{name:<10} 第 1 步 logits 最大差 {step1:.3e} | 各步首选一致 {sum(top)}/{len(top)}")
set_mode(False, False)
