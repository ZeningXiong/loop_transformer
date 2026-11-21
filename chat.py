"""跟优化后的 SenseNova-Loop-9B 对话。

默认走当前最快的组合（16K）：**CUDA graph + 2-token 投机 + 注意力直接读量化 KV**，
上下文不足 256 token 时自动退回纯贪心（短上下文里投机草稿太弱，反而更慢）。

    python chat.py                  # 默认：≥256 token 用图+投机，短对话用贪心
    python chat.py --eager          # 全程纯贪心（最稳，10 tok/s 左右）
    python chat.py --ctx 8192       # 更小上下文
    python chat.py --think          # 打开思考模式
    python chat.py --q "问题"        # 一次性回答

对话内命令：/reset、/think on|off、/stats、/quit。
每轮打印速度、投机命中率、整卡峰值。
"""
import argparse
import os
import sys
import time

# 用别的解释器启动时自动切到项目 venv（venv 里才有 torch/bitsandbytes）
if os.name == "nt":
    _venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    if os.path.exists(_venv) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv).lower():
        import subprocess

        sys.exit(subprocess.call([_venv, os.path.abspath(__file__), *sys.argv[1:]]))

import warnings

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", message=".*seq_len.*num_heads.*")
warnings.filterwarnings("ignore", message=".*Triton version.*")
warnings.filterwarnings("ignore", message=".*causal_conv1d.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

from transformers.utils import logging as _hf_log  # noqa: E402

_hf_log.set_verbosity_error()

from sn_loop_common import (VramGuard, load, make_fixed_cache, reset_cache,  # noqa: E402
                            use_fused_kv_attention)
from sn_spec import MTPHead, _Snapshot, get_tap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=16384, help="上下文上限（token）")
ap.add_argument("--max-new", type=int, default=256, help="每轮最多生成多少 token")
ap.add_argument("--think", action="store_true", help="打开思考模式")
ap.add_argument("--eager", action="store_true", help="全程纯贪心（不用图和投机）")
ap.add_argument("--int4", type=int, default=4, help="第 2 遍 KV 的位数（4 或 8）")
ap.add_argument("--q", type=str, default=None, help="直接回答这一句然后退出（非交互）")
args = ap.parse_args()

dev = "cuda"
print("加载模型（全 NF4 + 融合内核 + 折叠注意力）…", flush=True)
tok, model = load("offline", int8=[])
mtp = MTPHead(model)
mtp.new_cache(args.ctx + 64)
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

MAXL = args.ctx + 64
dec = None
if not args.eager:
    from cudagraph_spec import Graph2Decoder

    use_fused_kv_attention(True)
    dec = Graph2Decoder(model, MAXL, kv_int8=True, qlen=2, pass2_bits=args.int4, buckets=(MAXL,))
    cache = dec.cache
    snap = dec._snap()
else:
    cache = make_fixed_cache(model, MAXL, kv_int8=True, pass2_bits=args.int4)
    snap = _Snapshot(cache)

tap = get_tap(model)
guard = VramGuard()
eos_ids = model.generation_config.eos_token_id
eos = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
think = args.think
state = dict(ids=None, cur=0, prev_h=None)
print_f = lambda s="": print(s, end="", flush=True)


@torch.inference_mode()
def set_pad(pad):
    if dec is None:
        return
    for l in dec.cache._full:
        l.set_pad(pad)


@torch.inference_mode()
def prefill(ids):
    """只预填新增部分（保留历史 KV）；模板变短时整段重来。"""
    ids = ids.to(dev)
    T = ids.shape[1]
    if state["ids"] is not None:
        keep = min(state["cur"], T)
        if keep and torch.equal(ids[:, :keep], state["ids"][:, :keep]):
            start = keep
        else:
            reset_cache(cache)
            mtp.cache.reset()
            state.update(ids=None, cur=0, prev_h=None)
            start = 0
    else:
        reset_cache(cache)
        mtp.cache.reset()
        start = 0

    set_pad(None)                       # 预填用动态视图
    prev_h = state["prev_h"]
    for s in range(start, T, 256):
        e = min(s + 256, T)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
        if prev_h is not None:
            mtp(prev_h, ids[:, s:s + 1], torch.tensor([[s]], device=dev), logits=False)
        if e - s > 1:
            mtp(h[:, :-1], ids[:, s + 1:e], torch.arange(s + 1, e, device=dev)[None], logits=False)
        prev_h = h[:, -1:]
    state["ids"] = ids
    state["cur"] = T
    state["prev_h"] = prev_h
    return T, prev_h


@torch.inference_mode()
def _draft(hidden, token, pos):
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(token))
    h = mtp.pre_fc_norm_hidden(hidden)
    z = mtp.fc(torch.cat([e, h] if mtp.embed_first else [h, e], dim=-1))
    pe = mtp.tm.rotary_emb(z, torch.tensor([[pos]], device=dev)[None].expand(3, -1, -1))
    z = mtp.layer(z, position_embeddings=pe, attention_mask=None,
                  position_ids=torch.tensor([[pos]], device=dev),
                  past_key_values=mtp.cache, use_cache=True)
    return mtp.lm_head(mtp.norm(z))[0, -1]


@torch.inference_mode()
def _feed(hidden, token, pos):
    e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(token))
    h = mtp.pre_fc_norm_hidden(hidden)
    z = mtp.fc(torch.cat([e, h] if mtp.embed_first else [h, e], dim=-1))
    pe = mtp.tm.rotary_emb(z, torch.tensor([[pos]], device=dev)[None].expand(3, -1, -1))
    mtp.layer(z, position_embeddings=pe, attention_mask=None,
              position_ids=torch.tensor([[pos]], device=dev),
              past_key_values=mtp.cache, use_cache=True)


@torch.inference_mode()
def generate_graph(ids):
    T, h_last = prefill(ids)
    set_pad(MAXL)                       # 解码用固定 pad（图里形状静态）
    if not state.get("captured"):
        print_f("（首次捕获 CUDA graph，约 4 秒…）\n")
        dec.capture(MAXL)
        state["captured"] = True
    x = model.lm_head(h_last).float()[0, -1].argmax().view(1, 1)
    pos, out, steps, acc = T, [], 0, 0
    acc_ids, shown = [], 0

    def emit(tokens):
        nonlocal shown
        acc_ids.extend(tokens)
        full = tok.decode(acc_ids)
        if len(full) > shown:
            print_f(full[shown:])
            shown = len(full)

    t0 = time.perf_counter()
    while len(out) < args.max_new:
        snap.save()
        base = mtp.cache.layer.cur
        d = _draft(h_last, x, pos).argmax().view(1, 1)
        L = dec.step(torch.cat([x, d], 1), pos)
        H = dec.bufs[dec.bucket_for(pos + 2)]["out"].clone()
        p0 = L[0, 0].argmax().view(1, 1)
        if d.item() == p0.item():
            acc += 1
            commit, nxt, h0, h1 = [x, d], L[0, 1].argmax().view(1, 1), H[:, 0:1], H[:, 1:2]
        else:
            snap.restore()
            L2 = dec.step(torch.cat([x, p0], 1), pos)
            H2 = dec.bufs[dec.bucket_for(pos + 2)]["out"].clone()
            commit, nxt, h0, h1 = [x, p0], L2[0, 1].argmax().view(1, 1), H2[:, 0:1], H2[:, 1:2]
        dec.advance(2)
        steps += 1
        out.extend([commit[0].item(), commit[1].item()])
        stop = commit[0].item() in eos or commit[1].item() in eos
        emit([t for t in (commit[0].item(), commit[1].item()) if t not in eos])
        if stop:
            break
        mtp.cache.layer.cur = base
        mtp.cache.layer._set_views()
        _feed(h_last, commit[0], pos)
        _feed(h0, commit[1], pos + 1)
        x, h_last, pos = nxt, h1, pos + 2
        guard.check()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    state["cur"] = pos
    state["prev_h"] = h_last
    rate = acc / max(steps, 1) * 100
    print_f(f"\n\033[90m[{len(out)} token / {dt:.1f}s = {len(out) / dt:.2f} tok/s"
            f"，图+投机 {acc}/{steps}，整卡 {guard.peak / 1e9:.2f} GB]\033[0m\n")
    return tok.decode(acc_ids)


@torch.inference_mode()
def generate_eager(ids):
    T, h_last = prefill(ids)
    x = model.lm_head(h_last).float()[0, -1].argmax().view(1, 1)
    pos, n, acc_ids, shown = T, 0, [], 0
    t0 = time.perf_counter()
    while n < args.max_new:
        mtp(h_last, x, torch.tensor([[pos]], device=dev), logits=False)
        h = model.model(input_ids=x, position_ids=torch.tensor([[pos]], device=dev),
                        past_key_values=cache, use_cache=True).last_hidden_state
        lg = model.lm_head(h).float()
        n += 1
        stop = x.item() in eos
        if not stop:
            acc_ids.append(x.item())
            full = tok.decode(acc_ids)
            if len(full) > shown:
                print_f(full[shown:])
                shown = len(full)
        if stop:
            break
        x, h_last, pos = lg[0, -1].argmax().view(1, 1), h[:, -1:], pos + 1
        guard.check()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    state["cur"] = pos
    state["prev_h"] = h_last
    print_f(f"\n\033[90m[{n} token / {dt:.1f}s = {n / dt:.2f} tok/s，贪心，"
            f"整卡 {guard.peak / 1e9:.2f} GB]\033[0m\n")
    return tok.decode(acc_ids)


def build(messages):
    kw = dict(tokenize=True, add_generation_prompt=True, return_tensors="pt")
    try:
        ids = tok.apply_chat_template(messages, enable_thinking=think, **kw)
    except TypeError:
        ids = tok.apply_chat_template(messages, **kw)
    if not torch.is_tensor(ids):
        ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
        if not torch.is_tensor(ids):
            ids = torch.tensor([ids], dtype=torch.long)
    return ids.to(dev)


def respond(ids):
    """上下文够长就走图+投机，否则贪心。"""
    use_graph = dec is not None and ids.shape[1] >= 256
    return (generate_graph if use_graph else generate_eager)(ids)


messages = []
if args.q:
    ids = build([{"role": "user", "content": args.q}])
    print("模型 > ", end="", flush=True)
    respond(ids)
    guard.close()
    sys.exit(0)

print(f"上下文上限 {args.ctx}，{'图+投机+量化KV（<256 token 自动贪心）' if dec is not None else '纯贪心'}，"
      f"思考模式 {'开' if think else '关'}。输入 /quit 退出。\n", flush=True)
while True:
    try:
        line = input("\033[1m你\033[0m > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        continue
    if line in ("/quit", "/exit"):
        break
    if line == "/reset":
        messages.clear()
        reset_cache(cache)
        mtp.cache.reset()
        state.update(ids=None, cur=0, prev_h=None)
        print("（上下文已清空）")
        continue
    if line == "/stats":
        print(f"整卡峰值 {guard.peak / 1e9:.2f} GB")
        continue
    if line.startswith("/think"):
        think = line.endswith("on")
        print(f"（思考模式 {'开' if think else '关'}）")
        continue

    messages.append({"role": "user", "content": line})
    ids = build(messages)
    if ids.shape[1] > args.ctx - args.max_new:
        print("（已超出上下文上限，请 /reset）")
        messages.pop()
        continue
    print("\033[1m模型\033[0m > ", end="", flush=True)
    answer = respond(ids)
    messages.append({"role": "assistant", "content": answer})
    print()

guard.close()
