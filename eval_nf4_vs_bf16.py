"""Compare NF4 (offline checkpoint) against unquantized bf16, position by position.

phase "ref": run bf16 (GPU+CPU split), save full next-token log-probs (fp16) per text, plus the
             hidden state after every decoder-layer execution (56 per forward) for a few texts.
phase "cmp": run the offline NF4 model on the same tokens and compare:
             KL(P_bf16 || P_nf4) per position, top-1 agreement, perplexity of both,
             and the relative hidden-state error after each of the 56 layer executions.

usage: python -I eval_nf4_vs_bf16.py ref|cmp [texts.jsonl store_dir out.json]
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sn_loop_common import load  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = r"D:\models\_eval_bf16_ref"  # ~2.5 GB of reference log-probs, deletable afterwards
TEXTS = "eval_texts.jsonl"
OUT = "eval_nf4_vs_bf16.json"
MAX_TOKENS = 384
HIDDEN_KEYS = {"zh_tech_process", "code_json_decoder"}  # texts whose per-layer hidden states we keep
LM_CHUNK = 64

phase = sys.argv[1]
if len(sys.argv) > 2:  # optional: texts file, reference store dir, output json
    TEXTS, STORE, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
texts = [json.loads(l) for l in open(os.path.join(HERE, TEXTS), encoding="utf-8")]
os.makedirs(STORE, exist_ok=True)

tok, model = load("bf16" if phase == "ref" else "offline")
layers = model.model.language_model.layers
captured = []


def hook(_mod, _inp, out):
    h = out[0] if isinstance(out, tuple) else out
    captured.append(h.detach()[0].to("cpu", torch.bfloat16))


handles = [layer.register_forward_hook(hook) for layer in layers]


@torch.inference_mode()
def logprobs(ids):
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    out = []
    for s in range(0, hidden.shape[1], LM_CHUNK):
        logits = model.lm_head(hidden[:, s:s + LM_CHUNK]).float()
        out.append(torch.log_softmax(logits, dim=-1)[0].to("cuda"))
    return torch.cat(out)  # [T, V] fp32 on GPU


summary = []
t_all = time.time()
for item in texts:
    key = item["key"]
    ids = tok(item["text"], return_tensors="pt").input_ids[:, :MAX_TOKENS].to("cuda:0")
    T = ids.shape[1]
    captured.clear()
    t = time.time()
    lp = logprobs(ids)
    dt = time.time() - t
    target = ids[0, 1:]
    nll = -lp[:-1].gather(1, target[:, None])[:, 0]

    if phase == "ref":
        lp16 = lp.half()
        # storage noise floor: KL(fp32 || fp16-rounded) of the very same distribution
        p = lp.exp()
        floor = (p * (lp - lp16.float())).sum(-1).mean().item()
        torch.save({"ids": ids.cpu(), "lp": lp16.cpu(), "nll": nll.cpu()}, os.path.join(STORE, f"{key}.pt"))
        if key in HIDDEN_KEYS:
            torch.save(captured.copy(), os.path.join(STORE, f"{key}.hidden.pt"))
        print(f"{key:20s} T={T}  bf16 PPL {nll.mean().exp().item():7.3f}  存储噪声 KL {floor:.2e}  "
              f"层执行次数 {len(captured)}  {dt:.1f}s", flush=True)
        continue

    ref = torch.load(os.path.join(STORE, f"{key}.pt"))
    assert torch.equal(ref["ids"], ids.cpu()), "tokenization differs between runs"
    lp_ref = ref["lp"].to("cuda").float()
    p_ref = lp_ref.exp()
    kl = (p_ref * (lp_ref - lp)).sum(-1)  # [T] nats
    top1 = (lp_ref.argmax(-1) == lp.argmax(-1)).float()
    ppl_ref = ref["nll"].mean().exp().item()
    ppl_q = nll.mean().exp().item()
    rec = dict(key=key, desc=item["desc"], tokens=T, kl_mean=kl.mean().item(), kl_median=kl.median().item(),
               kl_p95=kl.quantile(0.95).item(), kl_max=kl.max().item(), frac_kl_gt_0_1=(kl > 0.1).float().mean().item(),
               top1_agree=top1.mean().item(), ppl_bf16=ppl_ref, ppl_nf4=ppl_q, ppl_change=ppl_q / ppl_ref - 1)
    if key in HIDDEN_KEYS:
        href = torch.load(os.path.join(STORE, f"{key}.hidden.pt"))
        rel = []
        for a, b in zip(href, captured):
            a, b = a.float(), b.float()
            rel.append(((b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)).mean().item())
        rec["hidden_rel_err_per_exec"] = rel
    summary.append(rec)
    print(f"{key:20s} KL 均值 {rec['kl_mean']:.4f}  中位 {rec['kl_median']:.4f}  P95 {rec['kl_p95']:.4f}  "
          f"首选一致 {rec['top1_agree'] * 100:5.1f}%  PPL {ppl_ref:.3f}→{ppl_q:.3f} ({rec['ppl_change'] * 100:+.1f}%)", flush=True)

if phase == "cmp":
    tot = sum(r["tokens"] for r in summary)
    w = lambda k: sum(r[k] * r["tokens"] for r in summary) / tot  # noqa: E731
    print(f"\n总计 {tot} 个位置：KL 均值 {w('kl_mean'):.4f}，首选一致 {w('top1_agree') * 100:.1f}%，"
          f"KL>0.1 的位置占 {w('frac_kl_gt_0_1') * 100:.1f}%，PPL 平均变化 {w('ppl_change') * 100:+.2f}%")
    json.dump(summary, open(os.path.join(HERE, OUT), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"[{phase}] 总耗时 {time.time() - t_all:.0f}s")
