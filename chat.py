"""跟优化后的 SenseNova-Loop-9B 对话。

启用：全 NF4 权重、融合 RMSNorm/SiLU、折叠 GQA 注意力、int8 KV + 第 2 遍 int4、MTP 两 token 投机。

    python chat.py                 # 16K 上下文，最多 256 token/轮
    python chat.py --ctx 8192      # 更小上下文
    python chat.py --think         # 打开思考模式（默认关）
    python chat.py --plain         # 关掉投机，纯贪心

对话内命令：/reset 清空、/think on|off 切换思考、/stats 看显存、/quit 退出。
每轮打印速度、投机命中率、整卡峰值。

多轮对话每轮会重新预填整段历史（保证模板一致），所以很长的对话会变慢。
"""
import argparse
import os
import sys
import time

# 用的不是项目 venv 的解释器时，自动切过去（venv 里才有 torch/bitsandbytes）
if os.name == "nt":
    _venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    if os.path.exists(_venv) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv).lower():
        import subprocess

        sys.exit(subprocess.call([_venv, os.path.abspath(__file__), *sys.argv[1:]]))

import bitsandbytes as bnb
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*seq_len.*num_heads.*")
warnings.filterwarnings("ignore", message=".*Triton version.*")
warnings.filterwarnings("ignore", message=".*causal_conv1d.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache  # noqa: E402
from sn_spec import MTPHead, _Snapshot, generate_speculative, get_tap  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, default=16384, help="上下文上限（token）")
ap.add_argument("--max-new", type=int, default=256, help="每轮最多生成多少 token")
ap.add_argument("--think", action="store_true", help="打开思考模式")
ap.add_argument("--spec", action="store_true", help="启用 2-token 投机（更快，但当前实现的输出与贪心不一致，慎用）")
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

cache = make_fixed_cache(model, args.ctx + 64, kv_int8=True, pass2_bits=args.int4)
snap = _Snapshot(cache)


guard = VramGuard()
eos_ids = model.generation_config.eos_token_id
eos = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
think = args.think


@torch.inference_mode()
def prefill(ids):
    """主模型分块预填；MTP 消费 (h_i, x_{i+1}) 对。返回最后一个位置的隐藏状态。"""
    T = ids.shape[1]
    prev_h = None
    for s in range(0, T, 256):
        e = min(s + 256, T)
        h = model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
        if prev_h is not None:
            mtp(prev_h, ids[:, s:s + 1], torch.tensor([[s]], device=dev), logits=False)
        if e - s > 1:
            mtp(h[:, :-1], ids[:, s + 1:e], torch.arange(s + 1, e, device=dev)[None], logits=False)
        prev_h = h[:, -1:]
    return prev_h



@torch.inference_mode()
def generate(ids):
    """贪心 / 2-token 投机解码。逐 token 攒起来再整体解码——按 token 对分别 decode 会吃掉跨 token 的多字节字符。"""
    T = ids.shape[1]
    h_last = prefill(ids)
    logits = model.lm_head(h_last).float()
    x = logits[0, -1].argmax().view(1, 1)
    pos, n_out, steps, acc = T, 0, 0, 0
    acc_ids, shown = [], 0

    def emit(tokens):
        nonlocal shown
        acc_ids.extend(tokens)
        full = tok.decode(acc_ids)
        if len(full) > shown:
            print(full[shown:], end="", flush=True)
            shown = len(full)

    if args.spec:
        # 投机走生产实现（已验证与贪心逐 token 一致），代价是不能流式显示
        def _reset():
            reset_cache(cache)
            mtp.cache.reset()

        _reset()
        ids_out, st = generate_speculative(model, mtp, ids, args.max_new, cache, chunk_size=256)
        text = tok.decode(ids_out)
        print(text, flush=True)
        dt = st["prefill_s"] + st["decode_s"]
        print(f"\n\033[90m[{len(ids_out)} token / {dt:.1f}s = {len(ids_out) / max(st['decode_s'], 1e-6):.2f} tok/s"
              f"，投机 {st['accepted']}/{st['steps']}，整卡 {guard.peak / 1e9:.2f} GB]\033[0m", flush=True)
        return text

    t0 = time.perf_counter()
    while n_out < args.max_new:
        if (not args.spec):
            mtp(h_last, x, torch.tensor([[pos]], device=dev), logits=False)
            h = model.model(input_ids=x, position_ids=torch.tensor([[pos]], device=dev),
                            past_key_values=cache, use_cache=True).last_hidden_state
            lg = model.lm_head(h).float()
            n_out += 1
            if x.item() in eos:
                break
            emit([x.item()])
            x, h_last, pos = lg[0, -1].argmax().view(1, 1), h[:, -1:], pos + 1
            continue
        draft = mtp(h_last, x, torch.tensor([[pos]], device=dev))[0, -1].argmax().view(1, 1)
        snap.save()
        h = model.model(input_ids=torch.cat([x, draft], 1),
                        position_ids=torch.tensor([[pos, pos + 1]], device=dev),
                        past_key_values=cache, use_cache=True).last_hidden_state
        lg = model.lm_head(h).float()
        y1, y2 = lg[0, 0].argmax().view(1, 1), lg[0, 1].argmax().view(1, 1)
        steps += 1
        if y1.item() == draft.item():
            acc += 1
            t2, h_use, y_next = draft, h, y2
        else:
            snap.restore()
            h = model.model(input_ids=torch.cat([x, y1], 1),
                            position_ids=torch.tensor([[pos, pos + 1]], device=dev),
                            past_key_values=cache, use_cache=True).last_hidden_state
            lg = model.lm_head(h).float()
            t2, h_use, y_next = y1, h, lg[0, 1].argmax().view(1, 1)
        n_out += 2
        if x.item() in eos or t2.item() in eos:
            break
        emit([x.item(), t2.item()])
        mtp(h_use[:, :1], t2, torch.tensor([[pos + 1]], device=dev), logits=False)
        x, h_last, pos = y_next, h_use[:, 1:2], pos + 2
        guard.check()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    rate = acc / max(steps, 1) * 100
    tag = "" if (not args.spec) else f"，投机 {acc}/{steps} = {rate:.0f}%"
    print(f"\n\033[90m[{len(acc_ids)} token / {dt:.1f}s = {len(acc_ids) / dt:.2f} tok/s{tag}"
          f"，整卡 {guard.peak / 1e9:.2f} GB]\033[0m", flush=True)
    return tok.decode(acc_ids)


def build(messages):
    """直接用模板拿 token id，避免再走一遍分词器的 encode 路径。"""
    kw = dict(tokenize=True, add_generation_prompt=True, return_tensors="pt")
    try:
        ids = tok.apply_chat_template(messages, enable_thinking=think, **kw)
    except TypeError:
        ids = tok.apply_chat_template(messages, **kw)
    if isinstance(ids, (list, tuple)):
        ids = torch.tensor([ids], dtype=torch.long)
    if not torch.is_tensor(ids):
        ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
        if not torch.is_tensor(ids):
            ids = torch.tensor([ids], dtype=torch.long)
    return ids.to(device="cuda")


messages = []
if args.q:
    msgs = [{"role": "user", "content": args.q}]
    ids = build(msgs)
    print("模型 > ", end="", flush=True)
    generate(ids)
    guard.close()
    sys.exit(0)
print(f"上下文上限 {args.ctx}，第 2 遍 KV {args.int4}-bit，思考模式 {'开' if think else '关'}。输入 /quit 退出。\n",
      flush=True)
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
    reset_cache(cache)
    mtp.cache.reset()
    print("\033[1m模型\033[0m > ", end="", flush=True)
    answer = generate(ids)
    messages.append({"role": "assistant", "content": answer})
    print()

guard.close()
