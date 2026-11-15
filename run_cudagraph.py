"""Correctness and speed check for CUDA-graph decode (see cudagraph_decode.py).

  python run_cudagraph.py check                 # greedy 32 tokens: graph vs current path, token-exact?
  python run_cudagraph.py bench 128 512 2048    # tok/s and ms/token per context length, both paths

Both paths do the same chunked prefill and then one token per step; the only difference is whether the
56 layer applications of a decode step are replayed from a captured graph.
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

from cudagraph_decode import (BUCKETS, ContextFull, GraphDecoder,  # noqa: E402
                             masked_fold_attention, use_masked_attention)
from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402

PLAN = "mixed_plan_1.0.json"
CHUNK = 256


def plan_int8():
    return json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))["int8"]


def prompt_tokens(tok, n):
    texts = [json.loads(line)["text"] for line in open(os.path.join(HERE, "eval_texts.jsonl"),
                                                       encoding="utf-8")]
    ids = tok("".join(texts), return_tensors="pt").input_ids
    assert ids.shape[1] >= n, (ids.shape[1], n)
    return ids[:, :n].cuda()


def prefill(model, cache, ids):
    for s in range(0, ids.shape[1], CHUNK):
        e = min(s + CHUNK, ids.shape[1])
        out = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device="cuda")[None],
                          past_key_values=cache, use_cache=True)
    return out.last_hidden_state


# ------------------------------------------------------------------------------------------------ baseline
def run_baseline(model, tok, n_ctx, n_gen, max_len):
    cache = make_fixed_cache(model, max_len)
    ids = prompt_tokens(tok, n_ctx)
    with torch.inference_mode():
        reset_cache(cache)
        h = prefill(model, cache, ids)
        nxt = model.lm_head(h[:, -1:]).float().argmax(-1).reshape(1, 1)
        first_logits = model.lm_head(h[:, -1:]).float()[0, 0].clone()
        out_tokens, times = [nxt.item()], []
        for i in range(n_gen):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            h = model.model(input_ids=nxt, position_ids=torch.tensor([[n_ctx + i]], device="cuda"),
                            past_key_values=cache, use_cache=True).last_hidden_state
            logits = model.lm_head(h[:, -1:]).float()
            nxt = logits.argmax(-1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            out_tokens.append(nxt.item()) if i < n_gen - 1 else None
    del cache
    torch.cuda.empty_cache()
    return out_tokens, first_logits, times


# ------------------------------------------------------------------------------------------------ graph
def run_graph(model, tok, n_ctx, n_gen, max_len, kv_int8=False):
    dec = GraphDecoder(model, max_len, kv_int8=kv_int8)
    ids = prompt_tokens(tok, n_ctx)
    use_masked_attention(True)
    try:
        with torch.inference_mode():
            dec.reset()
            h = prefill(model, dec.cache, ids)
            nxt = model.lm_head(h[:, -1:]).float().argmax(-1).reshape(1, 1)
            first_logits = model.lm_head(h[:, -1:]).float()[0, 0].clone()
            out_tokens, times = [nxt.item()], []
            for i in range(n_gen):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = dec.step(nxt, n_ctx + i, logits_head=True).float()
                nxt = logits.argmax(-1)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                dec.advance(1)
                if i < n_gen - 1:
                    out_tokens.append(nxt.item())
    finally:
        use_masked_attention(False)
    return out_tokens, first_logits, times, dec.capture_s


def phase_check(model, tok):
    n_ctx, n_gen = 200, 32
    b_tok, b_logits, b_t = run_baseline(model, tok, n_ctx, n_gen, 512)
    print(f"基线：{n_ctx} token 上下文，生成 {len(b_tok) - 1} 个 token，"
          f"{statistics.median(b_t) * 1e3:.1f} ms/token", flush=True)
    g_tok, g_logits, g_t, cap = run_graph(model, tok, n_ctx, n_gen, 512)
    print(f"图解码：{statistics.median(g_t) * 1e3:.1f} ms/token；捕获耗时 "
          + "、".join(f"{k}:{v:.2f}s" for k, v in cap.items()), flush=True)
    same = b_tok == g_tok
    print(f"token 序列一致：{same}")
    if not same:
        for i, (a, b) in enumerate(zip(b_tok, g_tok)):
            if a != b:
                print(f"  第 {i} 个 token 开始不同：{b_tok[max(0, i - 3):i + 3]} vs {g_tok[max(0, i - 3):i + 3]}")
                break
    p = torch.softmax(b_logits.float(), -1)
    q = torch.softmax(g_logits.float(), -1)
    kl = (p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum().item()
    print(f"首步 logits：KL {kl:.3e}，最大绝对差 {(b_logits - g_logits).abs().max().item():.3e}")
    return same


def phase_bench(model, tok, lengths):
    guard = VramGuard()
    rows = []
    for n_ctx in lengths:
        max_len = min([b for b in BUCKETS if b >= n_ctx + 33] or [BUCKETS[-1]])
        n_gen = 24
        b_tok, _, bt = run_baseline(model, tok, n_ctx, n_gen, max_len)
        g_tok, _, gt, cap = run_graph(model, tok, n_ctx, n_gen, max_len)
        b_ms, g_ms = statistics.median(bt) * 1e3, statistics.median(gt) * 1e3
        agree = b_tok == g_tok
        rows.append(dict(n_ctx=n_ctx, max_len=max_len, base_ms=b_ms, graph_ms=g_ms,
                         base_tps=1e3 / b_ms, graph_tps=1e3 / g_ms, speedup=b_ms / g_ms,
                         capture_s=sum(cap.values()), tokens_equal=agree))
        print(f"上下文 {n_ctx:>6}：基线 {b_ms:7.1f} ms/token（{1e3 / b_ms:5.2f} tok/s）→ "
              f"图 {g_ms:7.1f} ms/token（{1e3 / g_ms:5.2f} tok/s），×{b_ms / g_ms:.2f}，"
              f"token 一致 {agree}，捕获 {sum(cap.values()):.1f}s", flush=True)
        guard.check()
    json.dump(rows, open(os.path.join(HERE, "cudagraph_bench.json"), "w", encoding="utf-8"), indent=1)
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB")
    guard.close()
    return rows


def phase_batch(model, tok, n_ctx=512, rounds=3, n_gen=8):
    """Cost of verifying k tokens in one forward, interleaved over rounds to cancel clock drift."""
    from sn_spec import small_batch_gemv

    cache = make_fixed_cache(model, 2048)
    ids = prompt_tokens(tok, n_ctx)
    times = {k: [] for k in (1, 2, 4)}
    with torch.inference_mode():
        reset_cache(cache)
        prefill(model, cache, ids)
        pos = n_ctx
        for _ in range(rounds):
            for k in (1, 2, 4):
                toks = torch.randint(1000, 20000, (1, k), device="cuda")
                for _ in range(n_gen):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    hh = model.model(input_ids=toks,
                                     position_ids=torch.arange(pos, pos + k, device="cuda")[None],
                                     past_key_values=cache, use_cache=True).last_hidden_state
                    model.lm_head(hh[:, -k:]).float()
                    torch.cuda.synchronize()
                    times[k].append(time.perf_counter() - t0)
                    pos += k
    base = statistics.median(times[1])
    rows = []
    for k in (1, 2, 4):
        ms = statistics.median(times[k]) * 1e3
        rows.append(dict(k=k, ms_per_step=ms, ms_per_token=ms / k, ratio=ms / (base * 1e3)))
        print(f"  每步 {k} 个 token：{ms:7.1f} ms/步，{ms / k:7.1f} ms/token"
              f"（成本是 1 token 的 x{ms / (base * 1e3):.2f}）", flush=True)
    del cache
    torch.cuda.empty_cache()
    json.dump(rows, open(os.path.join(HERE, "cudagraph_batch.json"), "w", encoding="utf-8"), indent=1)
    return rows


if __name__ == "__main__":
    phase, args = sys.argv[1], [int(x) for x in sys.argv[2:]]
    tok, model = load("offline", int8=plan_int8(), compile_sinkhorn=True)
    if phase == "check":
        ok = phase_check(model, tok)
    elif phase == "batch":
        phase_batch(model, tok, *(args or [512]))
    else:
        phase_bench(model, tok, args)
