"""Is the short-sequence recurrent dispatch correct, and does it actually save time?

Same starting state for both modes: prefill once, then snapshot/restore around every forward.

python check_short_linattn.py [上下文] [token 数]
"""
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from short_linattn import use_recurrent_for_short  # noqa: E402
from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import _Snapshot  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
tok, model = load("offline", int8=json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"),
                                                encoding="utf-8"))["int8"])
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts), return_tensors="pt").input_ids[:, :n_ctx].cuda()
cache = make_fixed_cache(model, n_ctx + 40, kv_int8=True)
snap = _Snapshot(cache)
dev = "cuda"


@torch.inference_mode()
def prefill():
    reset_cache(cache)
    for s in range(0, ids.shape[1], 256):
        e = min(s + 256, ids.shape[1])
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)


@torch.inference_mode()
def measure(tokens, reps=8):
    pos = torch.tensor([[n_ctx]], device=dev)
    ts, out = [], None
    for _ in range(reps):
        snap.save()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = model.model(input_ids=tokens, position_ids=pos, past_key_values=cache,
                        use_cache=True).last_hidden_state
        out = model.lm_head(h).float()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        snap.restore()
    return statistics.median(ts) * 1e3, out


prefill()
toks = torch.randint(1000, 20000, (1, k), device=dev)
use_recurrent_for_short(False)
t_chunk, log_chunk = measure(toks)
use_recurrent_for_short(True)
t_rec, log_rec = measure(toks)
use_recurrent_for_short(False)

p, q = torch.softmax(log_chunk, -1), torch.softmax(log_rec, -1)
kl = (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum(-1).mean().item()
same = (log_chunk.argmax(-1) == log_rec.argmax(-1)).float().mean().item()
print(f"{n_ctx} token 上下文，一次验证 {k} 个 token")
print(f"  chunk 路径     {t_chunk:7.1f} ms")
print(f"  递推特化       {t_rec:7.1f} ms   （快 {t_chunk / t_rec:.2f}x）")
print(f"  两者 logits：KL {kl:.2e}，最大绝对差 {(log_chunk - log_rec).abs().max().item():.2e}，"
      f"首选一致 {100 * same:.1f}%")
