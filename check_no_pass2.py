"""Is the second loop pass removable? LoopDuet-style depth ablation on SenseNova-Loop-9B.

The paper (arXiv 2611.07211) removes loop iterations from prompt positions and keeps decoding depth,
because in Huginn / OLMo retrofits the recurrent state has essentially settled after the loops they
drop (their acceptance threshold is d = ||h^P - h^S|| / ||h^S|| <= 0.24).

SenseNova runs the block 4..27 twice. Here the *whole* second pass is neutralised -- every layer of
the second visit returns the output it produced on the first visit -- which is the strongest possible
form of "delete pass 2". The output distribution is then compared with the stored bf16 reference.
If this is already catastrophic, no amount of scheduling can make the second pass removable.

usage: python check_no_pass2.py
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import VramGuard, load  # noqa: E402

STORE = r"D:\models\_eval_bf16_ref_heldout"
PLAN = "mixed_plan_1.0.json"
T = 384
CHUNK = 32
LOOPED = range(4, 28)


@torch.inference_mode()
def kl_chunked(model, ids, lp_ref):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    n = hidden.shape[1]
    kl_sum, top1, nll_sum, n_nll = 0.0, 0.0, 0.0, 0
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        lp = torch.log_softmax(model.lm_head(hidden[:, s:e]).float(), dim=-1)[0]
        ref = lp_ref[s:e].float()
        kl_sum += (ref.exp() * (ref - lp)).sum().item()
        top1 += (ref.argmax(-1) == lp.argmax(-1)).sum().item()
        k = e - s if e < n else e - s - 1
        if k > 0:
            nll_sum += -lp[:k].gather(1, ids[0, s + 1:e + 1].unsqueeze(1))[:, 0].sum().item()
            n_nll += k
        del lp
    return kl_sum / n, top1 / n, math.exp(nll_sum / max(n_nll, 1)), n


def main():
    plan = json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))["int8"]
    tok, model = load("offline", int8=plan, compile_sinkhorn=False)
    guard = VramGuard()
    layers = model.model.language_model.layers

    cache, visit, active = {}, {}, [False]
    memo = {}

    def make_hook(layer_idx):
        def hook(_mod, _args, output):
            if not active[0]:
                return None
            k = visit.get(layer_idx, 0)
            visit[layer_idx] = k + 1
            out = output[0] if isinstance(output, tuple) else output
            if k == 0:
                memo[layer_idx] = out
                return None
            return memo[layer_idx]          # second visit: contribute nothing new
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in LOOPED]
    items = [json.loads(l) for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"), encoding="utf-8")]

    res = {"base": [0.0, 0.0, 0.0, 0], "no_pass2": [0.0, 0.0, 0.0, 0]}
    for it in items:
        ref = torch.load(os.path.join(STORE, f"{it['key']}.pt"), weights_only=False)
        ids = ref["ids"][:, :T].cuda()
        lp_ref = ref["lp"][:T].cuda()
        for name in ("base", "no_pass2"):
            active[0] = name == "no_pass2"
            visit.clear()
            memo.clear()
            kl, top1, ppl, n = kl_chunked(model, ids, lp_ref)
            guard.check()
            r = res[name]
            r[0] += kl * n
            r[1] += top1 * n
            r[2] += ppl
            r[3] += n
            print(f"  {name:<9} {it['key']:<24} KL {kl:7.4f}  首选 {top1 * 100:5.1f}%", flush=True)
        del lp_ref
        torch.cuda.empty_cache()
        memo.clear()

    print(f"\n{'方案':<12}{'KL':>10}{'首选一致':>10}{'PPL':>10}")
    for name in ("base", "no_pass2"):
        kl, top1, ppl, npos = res[name]
        print(f"{name:<12}{kl / npos:>10.4f}{top1 / npos * 100:>9.1f}%{ppl / len(items):>10.3f}")
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB")
    guard.close()
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
