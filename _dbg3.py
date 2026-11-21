import os, sys, json, torch
sys.path.insert(0, r"D:\ClaudeCode\project\loop_transformer")
sys.stdout.reconfigure(encoding="utf-8")
from sn_loop_common import load, make_fixed_cache, reset_cache, use_fused_kv_attention, QuantKV, gqa_fold_attention
from fused_kv_attn import attend
N = 4096
dev = "cuda"
tok, model = load("offline", int8=[])
cache = make_fixed_cache(model, N + 64, kv_int8=True, pass2_bits=4)
HERE = r"D:\ClaudeCode\project\loop_transformer"
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"), encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N].cuda()
with torch.inference_mode():
    reset_cache(cache)
    for s0 in range(0, N, 256):
        e = min(s0 + 256, N)
        h = model.model(input_ids=ids[:, s0:e], position_ids=torch.arange(s0, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    x = model.lm_head(h[:, -1:]).float()[0, -1].argmax().view(1, 1)
    use_fused_kv_attention(True)
    # 手动抓 q：用 hook 抓 attention 模块输入不方便，直接从缓存的最后一层更新后取 k/v，
    # 再用模型自己的 q_proj 复现 q（同一 hidden 状态）
    use_fused_kv_attention(False)
    reset_cache(cache)
    for s0 in range(0, N, 256):
        e = min(s0 + 256, N)
        h = model.model(input_ids=ids[:, s0:e], position_ids=torch.arange(s0, e, device=dev)[None],
                        past_key_values=cache, use_cache=True).last_hidden_state
    lay = cache.layers[3]
    use_fused_kv_attention(True)
    model.model(input_ids=x, position_ids=torch.tensor([[N]], device=dev), past_key_values=cache, use_cache=True)
    kv = cache.layers[3]
    L = kv.cur
    kbf = (kv.k_q[:, :, :L].float() * kv.k_s[:, :, :L]).to(torch.bfloat16)
    vbf = (kv.v_q[:, :, :L].float() * kv.v_s[:, :, :L]).to(torch.bfloat16)
    # 用同一份 k/v 和随机 q 做精确 torch 参考
    q = torch.randn(1, 16, 1, 256, device=dev, dtype=torch.bfloat16)
    qf = q.reshape(4, 4, 256)
    exact = torch.empty(4, 4, 256, device=dev, dtype=torch.float32)
    for hv in range(4):
        kk = kbf[0, hv].float(); vv = vbf[0, hv].float()
        for gi in range(4):
            s = (q[0, hv*4+gi, 0].float() @ kk.T) / 16.0
            p = torch.softmax(s, -1)
            exact[hv, gi] = p @ vv
    ref_fold = gqa_fold_attention(None, q, kbf, vbf, None)[0].reshape(1, 16, 1, 256).transpose(1,2)
    for ns in (1, 8):
        got = attend(qf, kv.k_q, kv.k_s, kv.v_q, kv.v_s, L, kv.bits, 4, 1, n_splits=ns)
        print(f"ns={ns}: vs 精确 {(got.float() - exact).abs().max().item():.3e}")
    print(f"折叠SDPA: vs 精确 {(ref_fold.float().reshape(4,4,256) - exact).abs().max().item():.3e}")
    print(f"k 最大 {kbf.abs().max().item():.2f} q 最大 {q.abs().max().item():.2f}")
