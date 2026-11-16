"""Low-cost feasibility test of LoopVQ (arXiv 2611.05304) on SenseNova-Loop-9B.

Question: does coding the two loop passes of a full-attention layer jointly ("loop" grouping: G/2 channels x 2
passes per 8-bit code) beat coding each pass alone ("channel" grouping: G channels x 1 pass) at the same bitwidth?
Plain k-means only (no Fisher weighting, no codebook fine-tuning), as in the paper's Figure 2c.

Phases (run separately, each writes to OUT):
  python check_loopvq.py capture   # load model, record pre-RoPE K and V of every full-attention visit
  python check_loopvq.py kmeans    # no model; correlations + k-means NMSE on held-out text; writes dequantized KV
  python check_loopvq.py kl        # load model, replace looped-pass K/V with the dequantized ones, KL vs exact KV
"""
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
OUT = r"D:\models\_loopvq_tmp"
os.makedirs(OUT, exist_ok=True)
CHUNK, MAXLEN, SINK = 512, 16384, 8
WIN, TOTAL = 0, 0  # 最近 WIN 个 token 保持精确（0 = 全部量化）
SEL = (0, 1)       # 哪些循环遍的格子被量化（逐格分配用）
KCODE = 256  # 8-bit codes, as in the paper
GS = (4, 8, 16, 32)  # 2, 1, 0.5, 0.25 bits per element
KL_CFGS = ["int8", "int4", "loop_G4", "chan_G4", "loop_G8", "chan_G8", "loop_G16", "chan_G16"]


def texts(path):
    return [json.loads(l)["text"] for l in open(path, encoding="utf-8")]


# --------------------------------------------------------------------------- model-side helpers
def attn_modules(model):
    tm = model.model.language_model
    plan = list(tm._loop_plan)
    full = [i for i, l in enumerate(tm.layers) if hasattr(l, "self_attn")]
    visits = {i: plan.count(i) for i in full}
    return tm, full, visits


class Hooks:
    """Count calls of each full-attention layer's k_norm / v_proj within one forward -> loop pass index.
    mode 'capture': store outputs. mode 'replace': swap outputs of looped passes with given tensors."""

    def __init__(self, model):
        self.tm, self.full, self.visits = attn_modules(model)
        self.mode, self.store, self.repl, self.span = None, {}, None, None
        self.cnt = {}
        self.handles = []
        for i in self.full:
            a = self.tm.layers[i].self_attn
            self.handles.append(a.k_norm.register_forward_hook(self._mk(i, "k")))
            self.handles.append(a.v_proj.register_forward_hook(self._mk(i, "v")))

    def begin(self, span):
        self.cnt = {}
        self.span = span  # (s, e) token positions of this chunk

    def _mk(self, layer, kv):
        def hook(mod, inp, out):
            key = (layer, kv)
            p = self.cnt.get(key, 0)
            self.cnt[key] = p + 1
            if self.mode == "capture":
                x = out.detach().reshape(out.shape[0], out.shape[1], 4, -1)[0]  # [T, H, D]
                self.store.setdefault((layer, p, kv), []).append(x.to("cpu", torch.bfloat16))
            elif self.mode == "replace" and self.visits[layer] == 2 and p in SEL:
                s, e = self.span
                lo = max(0, SINK - s)  # keep sink tokens exact
                hi = e - s if WIN <= 0 else max(0, min(e, TOTAL - WIN) - s)  # [hi, ) 留精确
                if hi <= lo:
                    return None
                r = self.repl[(layer, p, kv)][s:e].to(out.device, out.dtype)  # [T, H, D]
                new = out.clone()
                new.view(1, e - s, 4, -1)[0, lo:hi] = r[lo:hi]
                return new
            return None
        return hook


def run_seq(model, hooks, ids, cache):
    from sn_loop_common import reset_cache
    hs = []
    with torch.inference_mode():
        reset_cache(cache)
        for s in range(0, ids.shape[1], CHUNK):
            e = min(s + CHUNK, ids.shape[1])
            hooks.begin((s, e))
            pos = torch.arange(s, e, device="cuda")[None]
            h = model.model(input_ids=ids[:, s:e], position_ids=pos, past_key_values=cache,
                            use_cache=True).last_hidden_state
            hs.append(h[0].to("cpu"))
    return torch.cat(hs)


def load_model():
    from sn_loop_common import VramGuard, load, make_fixed_cache, use_gqa_fold_attention
    tok, model = load("offline")
    use_gqa_fold_attention(True)
    guard = VramGuard()
    cache = make_fixed_cache(model, MAXLEN)
    return tok, model, guard, cache


def tokenize(tok, path):
    return [tok(t, return_tensors="pt").input_ids[:, :MAXLEN].to("cuda") for t in texts(path)]


def phase_capture():
    tok, model, guard, cache = load_model()
    hooks = Hooks(model)
    print("全注意力层（物理编号）及访问次数：", hooks.visits, flush=True)
    hooks.mode = "capture"
    for name, path in (("calib", "_lq_calib.jsonl"), ("heldout", "_lq_held.jsonl")):
        seqs = []
        for ids in tokenize(tok, path):
            hooks.store = {}
            run_seq(model, hooks, ids, cache)
            guard.check()
            seqs.append({k: torch.cat(v) for k, v in hooks.store.items()})
        n = sum(next(iter(s.values())).shape[0] for s in seqs)
        torch.save(seqs, os.path.join(OUT, f"kv_{name}.pt"))
        print(f"{name}: {len(seqs)} 段，共 {n} token", flush=True)
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB", flush=True)
    guard.close()


# --------------------------------------------------------------------------- k-means side
def kmeans(X, K=KCODE, iters=25, seed=0):
    """Batched k-means. X [P, N, d] fp32 cuda -> codebooks [P, K, d]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    P, N, d = X.shape
    C = X[:, torch.randperm(N, generator=g, device="cuda")[:K]].clone()
    x2 = (X * X).sum(-1, keepdim=True)
    for _ in range(iters):
        a = (x2 - 2 * X @ C.transpose(1, 2) + (C * C).sum(-1)[:, None]).argmin(-1)  # [P, N]
        S = torch.zeros_like(C).scatter_add_(1, a[..., None].expand(-1, -1, d), X)
        cnt = torch.zeros(P, K, device="cuda").scatter_add_(1, a, torch.ones_like(a, dtype=torch.float32))
        C = torch.where(cnt[..., None] > 0, S / cnt.clamp_min(1)[..., None], C)
    return C


def encode(X, C):
    a = ((X * X).sum(-1, keepdim=True) - 2 * X @ C.transpose(1, 2) + (C * C).sum(-1)[:, None]).argmin(-1)
    return torch.gather(C, 1, a[..., None].expand(-1, -1, C.shape[-1]))


def split(x1, x2, G, mode):
    """x1, x2 [N, D] (pass 1, pass 2) -> sub-vector problems [P, N, G]."""
    N, D = x1.shape
    if mode == "chan":
        return torch.cat([x1.view(N, D // G, G), x2.view(N, D // G, G)], 1).transpose(0, 1)
    c = G // 2
    return torch.cat([x1.view(N, D // c, c), x2.view(N, D // c, c)], -1).transpose(0, 1)


def merge(Z, G, mode, N, D):
    if mode == "chan":
        Z = Z.transpose(0, 1)  # [N, 2D/G, G]
        return Z[:, :D // G].reshape(N, D), Z[:, D // G:].reshape(N, D)
    c = G // 2
    Z = Z.transpose(0, 1)  # [N, D/c, G]
    return Z[..., :c].reshape(N, D), Z[..., c:].reshape(N, D)


def scalar_q(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(x / s).clamp(-qmax, qmax) * s


def stack(seqs, key):
    return torch.cat([s[key] for s in seqs]).float()  # [N, H, D]


def phase_kmeans():
    calib = torch.load(os.path.join(OUT, "kv_calib.pt"), weights_only=False)
    held = torch.load(os.path.join(OUT, "kv_heldout.pt"), weights_only=False)
    lens = [next(iter(s.values())).shape[0] for s in held]
    looped = sorted({l for (l, p, kv) in calib[0] if p == 1})
    print(f"循环段全注意力层：{looped}；校准 {sum(next(iter(s.values())).shape[0] for s in calib)} token，"
          f"测试 {sum(lens)} token", flush=True)
    H = 4
    res = {"corr": {}, "nmse": {}}

    # 1. correlation (paper Fig. 2b) and pass-2-from-pass-1 residual energy, on held-out tokens
    for kv in "kv":
        same, other, resid = [], [], []
        for l in looped:
            a, b = stack(held, (l, 0, kv)).cuda(), stack(held, (l, 1, kv)).cuda()  # [N, H, D]
            for h in range(H):
                x1, x2 = a[:, h], b[:, h]
                z1 = (x1 - x1.mean(0)) / x1.std(0).clamp_min(1e-6)
                z2 = (x2 - x2.mean(0)) / x2.std(0).clamp_min(1e-6)
                same.append((z1 * z2).mean(0).abs().mean().item())
                D = x1.shape[1]
                zg = z1.view(-1, D // 16, 16)
                cm = torch.einsum("ngi,ngj->gij", zg, zg) / zg.shape[0]
                off = ~torch.eye(16, dtype=torch.bool, device="cuda")
                other.append(cm.abs()[:, off].mean().item())
                resid.append(((x2 - x1) ** 2).sum().item() / (x2 ** 2).sum().item())
        res["corr"][kv] = {"same_channel_across_passes": sum(same) / len(same),
                           "other_channels_same_pass": sum(other) / len(other),
                           "pass2_minus_pass1_nmse": sum(resid) / len(resid)}
        print(f"[相关性 {kv.upper()}] 同通道跨两遍 |r|={res['corr'][kv]['same_channel_across_passes']:.3f}；"
              f"同一遍内 16 通道组里不同通道 |r|={res['corr'][kv]['other_channels_same_pass']:.3f}；"
              f"直接用第 1 遍代替第 2 遍的 NMSE={res['corr'][kv]['pass2_minus_pass1_nmse']:.3f}", flush=True)

    # 2. k-means NMSE at equal bitwidth (fit on calib, measure on held-out) + scalar references
    deq = {c: {} for c in KL_CFGS}
    for G in GS:
        for mode in ("chan", "loop"):
            t0 = time.perf_counter()
            err, tot, err_p = {"k": 0.0, "v": 0.0}, {"k": 0.0, "v": 0.0}, {"k": [0.0, 0.0], "v": [0.0, 0.0]}
            name = f"{mode}_G{G}"
            for l in looped:
                for kv in "kv":
                    ca, cb = stack(calib, (l, 0, kv)).cuda(), stack(calib, (l, 1, kv)).cuda()
                    ha, hb = stack(held, (l, 0, kv)).cuda(), stack(held, (l, 1, kv)).cuda()
                    N, _, D = ha.shape
                    ra, rb = torch.empty_like(ha), torch.empty_like(hb)
                    for h in range(H):
                        C = kmeans(split(ca[:, h], cb[:, h], G, mode))
                        Z = encode(split(ha[:, h], hb[:, h], G, mode), C)
                        ra[:, h], rb[:, h] = merge(Z, G, mode, N, D)
                    for p, (x, r) in enumerate(((ha, ra), (hb, rb))):
                        e = ((x - r) ** 2).sum().item()
                        err[kv] += e; tot[kv] += (x ** 2).sum().item(); err_p[kv][p] += e
                    if name in deq:
                        off = 0
                        for si, n in enumerate(lens):
                            deq[name].setdefault(si, {})[(l, 0, kv)] = ra[off:off + n].to("cpu", torch.bfloat16)
                            deq[name][si][(l, 1, kv)] = rb[off:off + n].to("cpu", torch.bfloat16)
                            off += n
            res["nmse"][name] = {kv: err[kv] / tot[kv] for kv in "kv"}
            print(f"[k-means {16 / G * 0.5:.3g} bit/元素] {'跨两遍' if mode == 'loop' else '单遍按通道'} G={G}："
                  f"K NMSE={res['nmse'][name]['k']:.4f}，V NMSE={res['nmse'][name]['v']:.4f}"
                  f"（{time.perf_counter() - t0:.0f}s）", flush=True)
            torch.cuda.empty_cache()
    for bits in (8, 4, 2):
        err, tot = {"k": 0.0, "v": 0.0}, {"k": 0.0, "v": 0.0}
        name = f"int{bits}"
        for l in looped:
            for kv in "kv":
                for p in (0, 1):
                    x = stack(held, (l, p, kv)).cuda()
                    r = scalar_q(x, bits)
                    err[kv] += ((x - r) ** 2).sum().item(); tot[kv] += (x ** 2).sum().item()
                    if name in deq:
                        off = 0
                        for si, n in enumerate(lens):
                            deq[name].setdefault(si, {})[(l, p, kv)] = r[off:off + n].to("cpu", torch.bfloat16)
                            off += n
        res["nmse"][name] = {kv: err[kv] / tot[kv] for kv in "kv"}
        print(f"[标量 int{bits}，每 token 每头一个缩放，约 {bits + 32 / 256:.3g} bit/元素] "
              f"K NMSE={res['nmse'][name]['k']:.4f}，V NMSE={res['nmse'][name]['v']:.4f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "nmse.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for c in KL_CFGS:
        torch.save([deq[c][i] for i in range(len(lens))], os.path.join(OUT, f"deq_{c}.pt"))


def phase_kl():
    tok, model, guard, cache = load_model()
    hooks = Hooks(model)
    seqs = tokenize(tok, "_lq_held.jsonl")
    ref = [run_seq(model, hooks, ids, cache) for ids in seqs]  # mode None: exact KV
    res = {}
    for cfg in KL_CFGS:
        hooks.mode = "replace"
        kl_sum, top_sum, n = 0.0, 0, 0
        deq = torch.load(os.path.join(OUT, f"deq_{cfg}.pt"), weights_only=False)
        for si, ids in enumerate(seqs):
            hooks.repl = deq[si]
            hq = run_seq(model, hooks, ids, cache)
            guard.check()
            with torch.inference_mode():
                for r in range(SINK, hq.shape[0], 128):
                    la = torch.log_softmax(model.lm_head(ref[si][r:r + 128].cuda()).float(), -1)
                    lb = torch.log_softmax(model.lm_head(hq[r:r + 128].cuda()).float(), -1)
                    kl_sum += (la.exp() * (la - lb)).sum().item()
                    top_sum += (la.argmax(-1) == lb.argmax(-1)).sum().item()
                    n += la.shape[0]
        del deq
        gc.collect()
        res[cfg] = {"kl": kl_sum / n, "top1": top_sum / n}
        print(f"[KL] {cfg}: KL={kl_sum / n:.3e}，首选一致 {top_sum / n * 100:.1f}%（{n} 个位置）", flush=True)
    json.dump(res, open(os.path.join(OUT, "kl.json"), "w", encoding="utf-8"), indent=1)
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB", flush=True)
    guard.close()


def phase_gen():
    """16K 提示词上的生成对比：exact KV vs loop_G4（2-bit）。新 token 的 KV 两边都是精确的，
    所以这里隔离的是"提示词侧 KV 被压到 2-bit"对生成内容的影响。"""
    tok, model, guard, cache = load_model()
    hooks = Hooks(model)
    ids = tokenize(tok, "_lq_held.jsonl")[0]
    n_gen = 24
    out = {}
    lps = {}
    for tag in ("exact", "int8", "loop_G4"):
        hooks.mode = "replace" if tag != "exact" else None
        if tag != "exact":
            hooks.repl = torch.load(os.path.join(OUT, f"deq_{tag}.pt"), weights_only=False)[0]
        h = run_seq(model, hooks, ids, cache)
        hooks.mode = None
        with torch.inference_mode():
            lg = model.lm_head(h[-1:].cuda()).float()
            toks, lps[tag] = [], []
            for i in range(n_gen):
                lps[tag].append(torch.log_softmax(lg[0, -1], -1))
                nxt = lg[0, -1].argmax().view(1, 1)
                toks.append(int(nxt))
                pos = torch.tensor([[ids.shape[1] + i]], device="cuda")
                o = model.model(input_ids=nxt, position_ids=pos, past_key_values=cache, use_cache=True)
                lg = model.lm_head(o.last_hidden_state).float()
        out[tag] = dict(tokens=toks, text=tok.decode(toks, skip_special_tokens=True))
        guard.check()
        print(f"\n[{tag}] 生成 {n_gen} token：{out[tag]['text'][:300]!r}", flush=True)
    same = next((i for i, (a, b) in enumerate(zip(out["exact"]["tokens"], out["loop_G4"]["tokens"]))
                 if a != b), n_gen)
    kl = [(a.exp() * (a - b)).sum().item() for a, b in zip(lps["exact"], lps["loop_G4"])]
    print(f"\n[生成对比] 一致前缀 {same}/{n_gen} 个 token；"
          f"KL 均值 {sum(kl[:max(same, 1)]) / max(same, 1):.4f}，最大 {max(kl):.4f}")
    json.dump({"n_gen": n_gen, "same": same, "kl": kl,
               "exact": out["exact"]["text"], "loop_G4": out["loop_G4"]["text"]},
              open(os.path.join(OUT, "gen.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"整卡峰值 {guard.peak / 1e9:.2f} GB", flush=True)
    guard.close()


def _long_ids(tok):
    """16K 的单条长序列（不存在就现生成）。"""
    path = "_lq_held.jsonl"
    if not os.path.exists(path):
        ts = [json.loads(l)["text"] for l in open("eval_texts_heldout.jsonl", encoding="utf-8")]
        ids0 = tok("".join(ts * 60), add_special_tokens=False)["input_ids"][:16300]
        json.dump({"text": tok.decode(ids0)}, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return tokenize(tok, path)[0]


def phase_ref96():
    tok, model, guard, cache = load_model()
    hooks = Hooks(model)
    ids = _long_ids(tok)
    h = run_seq(model, hooks, ids, cache)
    with torch.inference_mode():
        lg = model.lm_head(h[-96:].cuda()).float()
    torch.save(lg.half().cpu(), os.path.join(OUT, "ref96.pt"))
    print(f"ref96 已保存 | 整卡 {guard.peak / 1e9:.2f} GB", flush=True)
    guard.close()


def phase_win96():
    global WIN, TOTAL
    tok, model, guard, cache = load_model()
    hooks = Hooks(model)
    ids = _long_ids(tok)
    TOTAL = int(ids.shape[1])
    ref = torch.load(os.path.join(OUT, "ref96.pt"), weights_only=True).cuda().float()
    lp = torch.log_softmax(ref, -1)
    deq = torch.load(os.path.join(OUT, "deq_loop_G4.pt"), weights_only=False)[0]
    for W in (0, 256, 1024):
        WIN = W
        hooks.mode, hooks.repl = "replace", deq
        h = run_seq(model, hooks, ids, cache)
        with torch.inference_mode():
            lg = model.lm_head(h[-96:].cuda()).float()
            lq = torch.log_softmax(lg, -1)
            kl = (lp.exp() * (lp - lq)).sum(-1).mean().item()
            top1 = (ref.argmax(-1) == lg.argmax(-1)).float().mean().item()
        print(f"loop_G4 + 最近 {W:<5} KL {kl:.4f} | 首选一致 {top1 * 100:.2f}% | "
              f"整卡 {guard.peak / 1e9:.2f} GB", flush=True)
        del h
        guard.check()
    guard.close()


def phase_alloc96():
    """逐格分配的第一问：第 1 遍 vs 第 2 遍，哪一边的 KV 压到 2-bit 更便宜。"""
    global SEL, WIN
    tok, model, guard, cache = load_model()
    hooks = Hooks(model)
    ids = _long_ids(tok)
    WIN = 0
    ref = torch.load(os.path.join(OUT, "ref96.pt"), weights_only=True).cuda().float()
    lp = torch.log_softmax(ref, -1)
    deq = torch.load(os.path.join(OUT, "deq_loop_G4.pt"), weights_only=False)[0]
    hooks.mode, hooks.repl = "replace", deq
    d8 = torch.load(os.path.join(OUT, "deq_int8.pt"), weights_only=False)[0]
    d4 = torch.load(os.path.join(OUT, "deq_int4.pt"), weights_only=False)[0]
    for label, repl, sel in (
            ("部署方案：第1遍 int8 + 第2遍 int4",
             {k: v for k, v in d8.items() if k[1] == 0} | {k: v for k, v in d4.items() if k[1] == 1},
             (0, 1)),
            ("对照：全部 int8", d8, (0, 1)),
            ("只压第 2 遍 int4", d4, (1,)),
            ("只压第 2 遍 2bit", torch.load(os.path.join(OUT, "deq_loop_G4.pt"),
                                          weights_only=False)[0], (1,))):
        SEL = sel
        hooks.repl = repl
        h = run_seq(model, hooks, ids, cache)
        with torch.inference_mode():
            lg = model.lm_head(h[-96:].cuda()).float()
            lq = torch.log_softmax(lg, -1)
            kl = (lp.exp() * (lp - lq)).sum(-1).mean().item()
            top1 = (ref.argmax(-1) == lg.argmax(-1)).float().mean().item()
        print(f"{label:<20} KL {kl:.4f} | 首选一致 {top1 * 100:.2f}% | 整卡 {guard.peak / 1e9:.2f} GB",
              flush=True)
        del h
        guard.check()
    guard.close()


if __name__ == "__main__":
    {"capture": phase_capture, "kmeans": phase_kmeans, "kl": phase_kl,
     "gen": phase_gen, "ref96": phase_ref96, "win96": phase_win96,
     "alloc96": phase_alloc96}[sys.argv[1]]()
