"""投机一步的组成部分：草稿 / 验证 / 回填 / 快照 各多少毫秒（16K）。"""
import json
import os
import sys
import time

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot, get_tap  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
STEPS = 8
dev = "cuda"
tok, model = load("offline", int8=[])
mtp = MTPHead(model)
mtp.new_cache(N_CTX + 64)
for name, child in list(mtp.named_modules()):
    if isinstance(child, torch.nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
        pn, _, attr = name.rpartition(".")
        parent = mtp.get_submodule(pn) if pn else mtp
        new = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=False,
                                compute_dtype=torch.bfloat16, quant_type="nf4",
                                compress_statistics=False, quant_storage=torch.uint8, device="meta")
        new.weight = bnb.nn.Params4bit(child.weight.data.detach(), requires_grad=False,
                                       quant_type="nf4", compress_statistics=False,
                                       quant_storage=torch.uint8)
        setattr(parent, attr, new.to(dev))
torch.cuda.empty_cache()
cache = make_fixed_cache(model, N_CTX + 64, kv_int8=True, pass2_bits=4)
tap = get_tap(model)
snap = _Snapshot(cache)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()

with torch.inference_mode():
    reset_cache(cache)
    mtp.cache.reset()
    prev_h = None
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
        if prev_h is not None:
            mtp(prev_h, ids[:, s:s + 1], torch.tensor([[s]], device=dev), logits=False)
        if e - s > 1:
            mtp(h[:, :-1], ids[:, s + 1:e], torch.arange(s + 1, e, device=dev)[None], logits=False)
        prev_h = h[:, -1:]
    x = model.lm_head(prev_h).float()[0, -1].argmax().view(1, 1)
    h_last, pos = prev_h, N_CTX
    t_snap = t_draft = t_main = t_head = t_feed = 0.0
    for i in range(STEPS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        snap.save()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        d = mtp(h_last, x, torch.tensor([[pos]], device=dev))[0, -1].argmax().view(1, 1)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        H = model.model(input_ids=torch.cat([x, d], 1),
                        position_ids=torch.tensor([[pos, pos + 1]], device=dev),
                        past_key_values=cache, use_cache=True).last_hidden_state
        torch.cuda.synchronize(); t3 = time.perf_counter()
        L = model.lm_head(H).float()
        torch.cuda.synchronize(); t4 = time.perf_counter()
        ok = L[0, 0].argmax().item() == d.item()
        nxt = d if ok else L[0, 0].argmax().view(1, 1)
        if not ok:
            snap.restore()
            H = model.model(input_ids=torch.cat([x, nxt], 1),
                            position_ids=torch.tensor([[pos, pos + 1]], device=dev),
                            past_key_values=cache, use_cache=True).last_hidden_state
            L = model.lm_head(H).float()
        mtp(H[:, :1], nxt, torch.tensor([[pos + 1]], device=dev), logits=False)
        torch.cuda.synchronize(); t5 = time.perf_counter()
        t_snap += t1 - t0; t_draft += t2 - t1; t_main += t3 - t2; t_head += t4 - t3; t_feed += t5 - t4
        x, h_last, pos = L[0, 1].argmax().view(1, 1), H[:, 1:2], pos + 2

    # 对照：同状态下 1-token 前向
    torch.cuda.synchronize(); t0 = time.perf_counter()
    model.model(input_ids=x, position_ids=torch.tensor([[pos]], device=dev),
                past_key_values=cache, use_cache=True)
    torch.cuda.synchronize(); one = (time.perf_counter() - t0) * 1e3
    # 再来一次 2-token
    torch.cuda.synchronize(); t0 = time.perf_counter()
    model.model(input_ids=torch.cat([x, x], 1), position_ids=torch.tensor([[pos, pos + 1]], device=dev),
                past_key_values=cache, use_cache=True)
    torch.cuda.synchronize(); two = (time.perf_counter() - t0) * 1e3
    print(f"  对照：1-token {one:.1f} ms | 2-token {two:.1f} ms")
    ms = 1000.0 / STEPS
    print(f"投机一步（16K，2 token，{STEPS} 步平均）：")
    print(f"  快照 save     {t_snap * ms:6.1f} ms")
    print(f"  MTP 草稿      {t_draft * ms:6.1f} ms")
    print(f"  主模型 2-token {t_main * ms:6.1f} ms")
    print(f"  lm_head       {t_head * ms:6.1f} ms")
    print(f"  MTP 回填      {t_feed * ms:6.1f} ms")
    print(f"  合计          {(t_snap + t_draft + t_main + t_head + t_feed) * ms:6.1f} ms/步")
