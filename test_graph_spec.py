"""16K + 2-token MTP + CUDA Graph：正确性与速度。

两种模式各自单独一个进程跑，避免两个 KV 缓存同时存在把显存挤爆（否则两条路径互相拖慢）。
  python test_graph_spec.py 16384 24 graph
  python test_graph_spec.py 16384 24 eager
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
from sn_spec import MTPHead, _Snapshot, generate_speculative, get_tap  # noqa: E402

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 24
mode = sys.argv[3] if len(sys.argv) > 3 else "graph"
ALL_NF4 = os.environ.get("ALL_NF4") == "1"
plan = [] if ALL_NF4 else json.load(open(os.path.join(HERE, "mixed_plan_1.0.json"), encoding="utf-8"))["int8"]

tok, model = load("offline", int8=plan)
guard = VramGuard()
dev = "cuda"
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 3), return_tensors="pt").input_ids[:, :n_ctx].cuda()
max_len = n_ctx + n_gen + 32
mtp = MTPHead(model)
mtp.new_cache(max_len)
tap = get_tap(model)
CHUNK = 256


@torch.inference_mode()
def prefill(c):
    reset_cache(c)
    mtp.cache.reset()
    prev_h = None
    T = ids.shape[1]
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        pos = torch.arange(s, e, device=dev)[None]
        model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=c, use_cache=True)
        h = tap.post
        if prev_h is not None:
            mtp(prev_h, ids[:, s:s + 1], pos[:, :1], logits=False)
        if e - s > 1:
            mtp(h[:, :-1], ids[:, s + 1:e], pos[:, 1:], logits=False)
        prev_h = h[:, -1:]
    return model.lm_head(tap.post[:, -1:]).float()


print(f"上下文 {n_ctx}，{'全 NF4' if ALL_NF4 else '+1.0GB int8'}，KV int8，模式 {mode}", flush=True)
if mode == "eager":
    cache = make_fixed_cache(model, max_len, kv_int8=True)
    with torch.inference_mode():
        toks, st = generate_speculative(model, mtp, ids, n_gen, cache, chunk_size=CHUNK)
    print(f"逐层 eager 2-token 投机：{st['tokens']} token / {st['decode_s']:.2f}s = "
          f"**{st['tokens'] / st['decode_s']:.2f} tok/s**；命中 {st['accepted']}/{st['steps']}")
else:
    from cudagraph_spec import Graph2Decoder

    dec = Graph2Decoder(model, max_len, kv_int8=True, qlen=2)
    snap = _Snapshot(dec.cache)
    with torch.inference_mode():
        logits = prefill(dec.cache)
        x = logits[0, -1].argmax().view(1, 1)
        h_last, pos = tap.post[:, -1:], n_ctx
        out, steps, acc, redo, cap = [], 0, 0, 0, 0.0
        t0 = time.perf_counter()
        while len(out) < n_gen:
            d = mtp(h_last, x, torch.tensor([[pos]], device=dev))[0, -1].argmax().view(1, 1)
            snap.save()
            t1 = time.perf_counter()
            lg = dec.step(torch.cat([x, d], 1), pos)
            torch.cuda.synchronize()
            cap += time.perf_counter() - t1
            y1, y2 = lg[0, 0].argmax().view(1, 1), lg[0, 1].argmax().view(1, 1)
            steps += 1
            ok = y1.item() == d.item()
            if not ok:
                redo += 1
                snap.restore()
                lg = dec.step(torch.cat([x, y1], 1), pos)
                y2 = lg[0, 1].argmax().view(1, 1)
            else:
                acc += 1
            dec.advance(2)
            h_use = dec.bufs[dec.bucket_for(pos + 2)]["out"]
            tok2 = d if ok else y1
            out.extend([x.item(), tok2.item()])
            mtp(h_use[:, :1], tok2, torch.tensor([[pos + 1]], device=dev), logits=False)
            x, h_last, pos = y2, h_use[:, 1:2], pos + 2
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    print(f"图 + 2-token 投机：{len(out)} token / {dt:.2f}s = **{len(out) / dt:.2f} tok/s**；"
          f"步数 {steps}，命中 {acc}，回退 {redo}；图内 {cap * 1e3 / steps:.1f} ms/步")
print(f"整卡峰值 {guard.peak / 1e9:.2f} GB")
guard.close()
