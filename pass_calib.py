"""Loop-aware (two-pass merged) activation calibration for the weight-only int8 layers.

Motivation: layers 4-27 of SenseNova-Loop are executed twice per token with the same weights but
different input distributions (pass 1 / pass 2). Weight-only int8 uses one absmax scale per output
row, derived from the weights alone -- it never looks at activations, and every calibration-free
scheme that does look at activations in a single-pass model would only see one of the two passes.
This script tests whether the per-row scale should be fitted to the *union* of the two passes.

Phases (run separately, each writes into OUT):
  python pass_calib.py capture   # load model, record E[x_k^2] of every int8-layer input, per loop pass
  python pass_calib.py solve     # no model: fit per-row scales under 4 importance hypotheses
  python pass_calib.py kl        # load model, install each scale variant, KL vs stored bf16 reference
"""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

OUT = r"D:\models\_pass_calib"
os.makedirs(OUT, exist_ok=True)
PLAN = "mixed_plan_1.0.json"          # the +1.0 GB plan: 106 weight-only-int8 tensors
CAPTURE_LEN = 512                      # tokens per text used for the activation statistics
CAP_CHUNK = 128                        # 512-token chunks OOM the 8 GB card once int8 layers are added
KL_LEN = 384                           # the stored bf16 reference covers the first 384 tokens
GRID = [round(1.0 - 0.05 * i, 2) for i in range(15)]   # candidate scales as a fraction of absmax
VARIANTS = ["A", "P1", "P2", "M"]      # absmax / pass1-fit / pass2-fit / merged-fit


def plan_int8():
    return json.load(open(os.path.join(HERE, PLAN), encoding="utf-8"))["int8"]


def texts(path):
    return [json.loads(line)["text"] for line in open(os.path.join(HERE, path), encoding="utf-8")]


SRC = r"D:\models\SenseNova-Loop-9B-Preview"   # original FP32 checkpoint: the unquantized weights
_wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]


def true_weight(name, device="cuda", chunk=8192):
    """Read one unquantized weight [out, in] fp32 from the original checkpoint, row chunk at a time."""
    from safetensors import safe_open

    with safe_open(os.path.join(SRC, _wm[name]), framework="pt") as f:
        sl = f.get_slice(name)
        rows, cols = sl.get_shape()
        w = torch.empty((rows, cols), dtype=torch.float32, device=device)
        for r in range(0, rows, chunk):
            w[r:r + chunk] = sl[r:min(r + chunk, rows)].to(device, torch.float32)
    return w


# ---------------------------------------------------------------------------------------------
# Phase 1: capture E[x^2] per input channel, separately for the two loop passes
# ---------------------------------------------------------------------------------------------
def phase_capture():
    from sn_loop_common import VramGuard, load, make_fixed_cache, reset_cache, use_gqa_fold_attention

    names = plan_int8()
    tok, model = load("offline", int8=names, compile_sinkhorn=False)
    use_gqa_fold_attention(True)
    guard = VramGuard()
    cache = make_fixed_cache(model, CAPTURE_LEN)

    tm = model.model.language_model
    loop_plan = list(tm._loop_plan)
    vis = {n: loop_plan.count(int(n.split(".")[3])) for n in names}
    mods = {n: model.get_submodule(n[: -len(".weight")]) for n in names}
    print(f"{len(names)} 个 int8 张量；访问次数分布 {sorted(set(vis.values()))}；"
          f"其中两遍循环的 {sum(v == 2 for v in vis.values())} 个", flush=True)

    acc = {n: [torch.zeros(mods[n].in_features, dtype=torch.float32, device="cuda") for _ in range(2)]
           for n in names}
    cnt = {n: [0, 0] for n in names}
    seen = {}

    def make_hook(n):
        def hook(_mod, args):
            p = seen.get(n, 0)
            seen[n] = p + 1
            if p > 1:
                return None
            x = args[0]
            acc[n][p].add_(x.float().pow(2).sum(dim=tuple(range(x.dim() - 1))))
            cnt[n][p] += x.shape[-2] * x.shape[0]
            return None

        return hook

    handles = [mods[n].register_forward_pre_hook(make_hook(n)) for n in names]
    stats = {}
    for tag, path in (("calib", "eval_texts.jsonl"), ("held", "eval_texts_heldout.jsonl")):
        for key, text in zip([json.loads(l)["key"] for l in open(os.path.join(HERE, path), encoding="utf-8")],
                             texts(path)):
            ids = tok(text, return_tensors="pt").input_ids[:, :CAPTURE_LEN].cuda()
            for s in range(0, ids.shape[1], CAP_CHUNK):
                e = min(s + CAP_CHUNK, ids.shape[1])
                seen.clear()
                with torch.inference_mode():
                    if s == 0:
                        reset_cache(cache)     # the cache holds inference tensors: reset under inference_mode
                    model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device="cuda")[None],
                                past_key_values=cache, use_cache=True)
                guard.check()
            print(f"  [{tag}] {key} {ids.shape[1]} token", flush=True)
        stats[tag] = {n: {"H1": (acc[n][0] / max(cnt[n][0], 1)).cpu(),
                          "H2": (acc[n][1] / max(cnt[n][1], 1)).cpu(),
                          "n1": cnt[n][0], "n2": cnt[n][1], "look_plan": vis[n]} for n in names}
        for n in names:  # reset for the next split
            acc[n][0].zero_()
            acc[n][1].zero_()
            cnt[n] = [0, 0]
    for h in handles:
        h.remove()
    torch.save(stats, os.path.join(OUT, "acts.pt"))
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB", flush=True)
    guard.close()


# ---------------------------------------------------------------------------------------------
# Phase 2: fit per-row scales.  error(s) = sum_k H_k (w_k - round(w_k / s) s)^2
# ---------------------------------------------------------------------------------------------
def phase_solve():
    from sn_loop_common import _get

    names = plan_int8()
    stats = torch.load(os.path.join(OUT, "acts.pt"), weights_only=False)
    fit = stats["calib"]                      # fitted on the calibration texts
    ev = stats["held"]                        # judged on held-out activations
    out, rows = {}, []
    denom = {(1,): 0.0, (2,): 0.0}
    errs = {(p, v): 0.0 for p in (1, 2) for v in ("A", "P1", "P2", "M", "ORACLE")}

    for i, n in enumerate(names):
        W = true_weight(n)                                  # unquantized [out, in] fp32
        s_abs = W.abs().amax(1).clamp_min(1e-12) / 127.0    # the scale weight-only int8 uses
        H1 = fit[n]["H1"].cuda()[None, :]
        H2 = fit[n]["H2"].cuda()[None, :]
        e1 = ev[n]["H1"].cuda()[None, :]
        e2 = ev[n]["H2"].cuda()[None, :]
        looped = fit[n]["n2"] > 0
        Hm, em = (H1 + H2, e1 + e2) if looped else (H1, e1)
        hyp = {"A": None, "P1": H1, "P2": H2 if looped else H1, "M": Hm, "ORACLE": em}

        chosen = {}
        for v, H in hyp.items():
            if H is None:                                   # absmax: g = 1
                chosen[v] = torch.ones_like(s_abs)
                continue
            best_g, best_e = torch.ones_like(s_abs), None
            for g in GRID:
                step = (s_abs * g)[:, None]
                d = W - torch.round(W / step).clamp_(-127, 127) * step
                e = (d * d * H).sum(1)
                if best_e is None:
                    best_e, best_g = e, torch.full_like(s_abs, g)
                else:
                    m = e < best_e
                    best_e = torch.where(m, e, best_e)
                    best_g = torch.where(m, torch.full_like(s_abs, g), best_g)
            chosen[v] = best_g
        sfull = {v: (s_abs * chosen[v]).cpu() for v in chosen}
        out[n] = {v: sfull[v] for v in VARIANTS}

        for v in ("A", "P1", "P2", "M", "ORACLE"):
            step = sfull[v].cuda()[:, None]
            d = W - torch.round(W / step).clamp_(-127, 127) * step
            for p, H in ((1, e1), (2, e2)):
                errs[(p, v)] += (d * d * H).sum().item()
        denom[(1,)] += (e1 * W * W).sum().item()
        denom[(2,)] += (e2 * W * W).sum().item()

        rows.append(dict(name=n, looped=looped, out=W.shape[0],
                         g_p1=chosen["P1"].mean().item(), g_p2=chosen["P2"].mean().item(),
                         g_m=chosen["M"].mean().item(),
                         g_shift_p1_vs_m=(chosen["P1"] - chosen["M"]).abs().mean().item(),
                         H_shift=((H1 - H2).abs().sum() / H1.sum()).item() if looped else 0.0))
        del W, s_abs
        if i % 20 == 0:
            torch.cuda.empty_cache()

    torch.save(out, os.path.join(OUT, "scales.pt"))

    # ---- report: weighted output error relative to the unquantized weights, on held-out activations
    print(f"\n拟合：校准集激活（{sum(1 for r in rows if r['looped'])} 个两遍层张量 / {len(rows)} 个）")
    print("评测：验证集激活上的加权输出误差  E[(Δw·x)^2] / E[(w·x)^2]  （越小越好）")
    print(f"{'方案':<12}{'第1遍':>13}{'第2遍':>13}{'两遍合计':>13}")
    tbl = {}
    for v in ("A", "P1", "P2", "M", "ORACLE"):
        r1 = errs[(1, v)] / denom[(1,)]
        r2 = errs[(2, v)] / denom[(2,)]
        rm = (errs[(1, v)] + errs[(2, v)]) / (denom[(1,)] + denom[(2,)])
        tbl[v] = dict(pass1=r1, pass2=r2, merged=rm)
        label = {"A": "absmax", "P1": "只用第1遍", "P2": "只用第2遍", "M": "两遍合并",
                 "ORACLE": "合并(上限)"}[v]
        print(f"{label:<12}{r1:>13.3e}{r2:>13.3e}{rm:>13.3e}")
    print("\n相对 absmax 的误差下降：")
    a = tbl["A"]
    for v in ("P1", "P2", "M"):
        d = lambda k: 100 * (1 - tbl[v][k] / a[k]) if a[k] else 0.0  # noqa: E731
        print(f"  {v:<3} 两遍合计 {d('merged'):+6.2f}%（第1遍 {d('pass1'):+6.2f}%，第2遍 {d('pass2'):+6.2f}%）")
    loops = [r for r in rows if r["looped"]]
    print(f"\n两遍之间输入分布差异：平均 |H1-H2|/sum(H1) = {sum(r['H_shift'] for r in loops) / len(loops):.3f}；"
          f"拟合缩放 |g_P1-g_M| 平均 {sum(r['g_shift_p1_vs_m'] for r in loops) / len(loops):.4f}；"
          f"平均 g：P1 {sum(r['g_p1'] for r in loops) / len(loops):.3f}，"
          f"P2 {sum(r['g_p2'] for r in loops) / len(loops):.3f}，M {sum(r['g_m'] for r in loops) / len(loops):.3f}")
    json.dump(dict(table=tbl, rows=rows), open(os.path.join(OUT, "solve.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------------------------
# Phase 3: end-to-end KL against the stored bf16 reference, one variant at a time
# ---------------------------------------------------------------------------------------------
@torch.inference_mode()
def kl_chunked(model, ids, lp_ref, chunk=32):
    """KL(bf16 || variant), top-1 agreement and NLL, accumulated 32 positions at a time.

    A full [T, vocab] fp32 log-prob tensor (0.38 GB at T=384) plus its temporaries is what makes the
    8 GB card spill, so nothing that size is ever materialized.
    """
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    T = hidden.shape[1]
    kl_sum, top1, nll_sum, n, n_nll = 0.0, 0, 0.0, 0, 0
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lp = torch.log_softmax(model.lm_head(hidden[:, s:e]).float(), dim=-1)[0]
        ref = lp_ref[s:e].float()
        kl_sum += (ref.exp() * (ref - lp)).sum().item()
        top1 += (ref.argmax(-1) == lp.argmax(-1)).sum().item()
        n += e - s
        k = e - s if e < T else e - s - 1          # last position has no next token
        if k > 0:
            nll_sum += -lp[:k].gather(1, ids[0, s + 1:e + 1].unsqueeze(1))[:, 0].sum().item()
            n_nll += k
        del lp
    return kl_sum / n, top1 / n, nll_sum / max(n_nll, 1)


def install_int8(model, name, q_new, s_new):
    from sn_loop_common import Int8WOLinear

    mod_name = name[: -len(".weight")]
    parent_name, _, attr = mod_name.rpartition(".")
    parent = model.get_submodule(parent_name)
    old = getattr(parent, attr)
    setattr(parent, attr, Int8WOLinear(q_new, s_new))
    del old


def install_scaled(model, name, s_new, chunk=4096):
    """Quantize the unquantized weight at scale `s_new` and swap the layer, one row chunk at a time.

    Holding a whole fp32 weight (up to 268 MB) on the GPU next to the model pushed the card past
    8 GB during the swap, so the codes are built chunk by chunk and only the int8 copy stays.
    """
    from safetensors import safe_open

    s_cuda = s_new.cuda()
    with safe_open(os.path.join(SRC, _wm[name]), framework="pt") as f:
        sl = f.get_slice(name)
        rows, cols = sl.get_shape()
        q = torch.empty((rows, cols), dtype=torch.int8, device="cuda")
        for r in range(0, rows, chunk):
            w = sl[r:min(r + chunk, rows)].to("cuda", torch.float32)
            q[r:r + chunk] = torch.round(w / s_cuda[r:r + chunk, None]).clamp_(-127, 127).to(torch.int8)
            del w
    install_int8(model, name, q, s_cuda)


def phase_kl():
    from sn_loop_common import VramGuard, load, use_gqa_fold_attention

    names = plan_int8()
    scales = torch.load(os.path.join(OUT, "scales.pt"), weights_only=False)
    tok, model = load("offline", int8=names, compile_sinkhorn=False)
    use_gqa_fold_attention(True)
    guard = VramGuard()
    STORE = r"D:\models\_eval_bf16_ref_heldout"
    items = [json.loads(l) for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"), encoding="utf-8")]

    refs, ppl_bf16 = {}, 0.0            # ids + bf16 log-probs stay in RAM, only one text on the GPU at a time
    for it in items:
        ref = torch.load(os.path.join(STORE, f"{it['key']}.pt"), weights_only=False)
        refs[it["key"]] = (ref["ids"][:, :KL_LEN], ref["lp"][:KL_LEN], ref["nll"].mean().exp().item())
        ppl_bf16 += refs[it["key"]][2]
    torch.cuda.empty_cache()
    print(f"参考：bf16 的 8 段文本，共 {sum(v[0].shape[1] for v in refs.values())} 个位置", flush=True)

    res = {v: {"kl": 0.0, "top1": 0, "nll": 0.0, "n": 0, "n_nll": 0} for v in VARIANTS}
    for v in VARIANTS:
        for i, n in enumerate(names):   # quantize the *unquantized* weights at this variant's scale
            install_scaled(model, n, scales[n][v])
            if i % 16 == 15:
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[{v}] 安装完成，整卡 {(total - free) / 1e9:.2f} GB", flush=True)
        for key, (ids_cpu, lp_ref_cpu, _) in refs.items():
            ids, lp_ref = ids_cpu.cuda(), lp_ref_cpu.cuda()
            kl, top1, nll = kl_chunked(model, ids, lp_ref)
            guard.check()
            r = res[v]
            T = ids.shape[1]
            r["kl"] += kl * T
            r["top1"] += top1 * T
            r["nll"] += nll * (T - 1)
            r["n"] += T
            r["n_nll"] += T - 1
            print(f"  {v:<3} {key:<24} KL {kl:.4f}", flush=True)
            del ids, lp_ref
        torch.cuda.empty_cache()

    print(f"\n{'方案':<12}{'KL':>9}{'首选一致':>9}{'PPL':>9}")
    base = None
    for v in VARIANTS:
        r = res[v]
        kl = r["kl"] / r["n"]
        base = base if base is not None else kl
        label = {"A": "absmax", "P1": "只用第1遍", "P2": "只用第2遍", "M": "两遍合并"}[v]
        extra = "基准" if v == "A" else f"{100 * (kl / base - 1):+.1f}% vs absmax"
        print(f"{label:<12}{kl:>9.4f}{r['top1'] / r['n'] * 100:>8.1f}%"
              f"{math.exp(r['nll'] / r['n_nll']):>9.3f}   {extra}")
    print(f"{'bf16':<12}{'—':>9}{'—':>9}{ppl_bf16 / len(items):>9.3f}")
    json.dump(res, open(os.path.join(OUT, "kl.json"), "w", encoding="utf-8"), indent=1)
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB", flush=True)
    guard.close()


if __name__ == "__main__":
    {"capture": phase_capture, "solve": phase_solve, "kl": phase_kl}[sys.argv[1]]()
