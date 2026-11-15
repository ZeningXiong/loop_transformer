"""Time each piece of one speculative step against a plain 1-token decode step."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import torch
from sn_loop_common import load, make_fixed_cache, reset_cache, use_gqa_fold_attention, warm_up, VramGuard, VramExceeded
from sn_spec import MTPHead, get_tap, small_batch_gemv, _Snapshot
tok, model = load("offline"); use_gqa_fold_attention(True)
guard = VramGuard(); tap = get_tap(model)
warm_up(model)
cache = make_fixed_cache(model, 2048); mtp = MTPHead(model); mtp.new_cache(2048)
ids = torch.randint(1000, 20000, (1, 300), device="cuda")
with torch.inference_mode():
    model.model(input_ids=ids, position_ids=torch.arange(300, device="cuda")[None], past_key_values=cache, use_cache=True)
    mtp(tap.post[:, :-1], ids[:, 1:], torch.arange(1, 300, device="cuda")[None])
snap = _Snapshot(cache)
def timeit(name, fn, n=20, restore=True):
    with torch.inference_mode():
        for _ in range(3): snap.save(); fn(); snap.restore() if restore else None
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            if restore: snap.save()
            fn()
            if restore: snap.restore()
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / n * 1e3
    print(f"{name:42s} {ms:7.1f} ms", flush=True); return ms
P = 300
one = torch.tensor([[5000]], device="cuda"); two = torch.tensor([[5000, 6000]], device="cuda")
def fwd(tokens):
    return model.model(input_ids=tokens, position_ids=torch.arange(P, P + tokens.shape[1], device="cuda")[None], past_key_values=cache, use_cache=True).last_hidden_state
h = tap.post[:, -1:]
try:
    a = timeit("主模型 1 个 token 前向（不含 lm_head）", lambda: fwd(one))
    b = timeit("主模型 2 个 token 前向（验证，不含 lm_head）", lambda: fwd(two))
    timeit("lm_head 1 行", lambda: model.lm_head(h), restore=False)
    timeit("lm_head 2 行", lambda: model.lm_head(torch.cat([h, h], 1)), restore=False)
    mtp_pos = torch.tensor([[P]], device="cuda")
    def mtp_no_head():
        mtp.cache.layer.cur = P - 1; mtp.cache.layer._set_views()
        e = mtp.pre_fc_norm_embedding(mtp.tm.embed_tokens(one)); hh = mtp.pre_fc_norm_hidden(h)
        x = mtp.fc(torch.cat([e, hh], -1)); pe = mtp.tm.rotary_emb(x, mtp_pos[None].expand(3, -1, -1))
        return mtp.layer(x, position_embeddings=pe, attention_mask=None, position_ids=mtp_pos, past_key_values=mtp.cache, use_cache=True)
    def mtp_full():
        mtp.cache.layer.cur = P - 1; mtp.cache.layer._set_views(); return mtp(h, one, mtp_pos)
    timeit("MTP 头（含 lm_head）", mtp_full, restore=False)
    timeit("MTP 头（不含 lm_head）", mtp_no_head, restore=False)
    timeit("保存 + 恢复线性注意力状态", lambda: None)
    print(f"验证一次 2 个 token / 单 token 前向 = {b / a:.2f} 倍")
    # where does the 2-token forward spend more? compare linear-attn layer alone
    la = model.model.language_model.layers[0].linear_attn
    x1 = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16); x2 = torch.randn(1, 2, 4096, device="cuda", dtype=torch.bfloat16)
    from sn_loop_common import _fixed_kv_layer_class
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
except VramExceeded as e:
    print("停止：", e)
finally:
    guard.close()
