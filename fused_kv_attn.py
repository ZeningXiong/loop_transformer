"""解码期注意力直接读量化 KV（int8 / int4），省掉"整段 KV 解量化到 bf16 scratch"的往返。

形状按本模型固定：Hkv=4、D=256、G=Hq/Hkv=4；查询长度 Tq=1（纯解码）或 2（投机验证）。

数值路径刻意贴合现有实现：反量化 -> bf16 -> QK^T / PV（与 SDPA 的 bf16 张量核一致），
softmax 在 fp32 里做，因此结果应与"解量化到 bf16 scratch + SDPA"基本一致。

实现是 split-L 的 flash 解码：每个 (kv_head, split) 一个 program，online softmax 累积；
partial 用 torch 那一侧合并（张量很小），避免第二个 kernel 的复杂度。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_kernel(q_ptr, kq_ptr, ks_ptr, vq_ptr, vs_ptr, acc_ptr, m_ptr, l_ptr, mask_ptr,
                  L, n_splits, sh_k, st_k, sh_s, st_s, sh_m, st_m,
                  G: tl.constexpr, TQ: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                  BLOCK_L: tl.constexpr, BITS: tl.constexpr, SCALE: tl.constexpr,
                  HAS_MASK: tl.constexpr):
    pid = tl.program_id(0)
    hv = pid // n_splits
    sp = pid % n_splits

    per = tl.cdiv(L, n_splits)
    lo = sp * per
    hi = tl.minimum(lo + per, L)

    r = tl.arange(0, ROWS)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + hv * ROWS * D + r[:, None] * D + d[None, :])   # [ROWS, D] bf16
    # 因果上限：第 r 行查询对应位置 L-TQ + (r % TQ)；填充行（r >= G*TQ）给 -1，永不入选
    lim = (L - TQ) + (r % TQ)
    lim = tl.where(r < G * TQ, lim, -1)

    m_i = tl.full([ROWS], -float("inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    DP: tl.constexpr = D // 2 if BITS == 4 else D

    for start in range(lo, hi, BLOCK_L):
        offs = start + tl.arange(0, BLOCK_L)
        valid = (offs < hi) & (offs < L)
        base = hv * sh_k + offs[:, None] * st_k + tl.arange(0, DP)[None, :]
        if BITS == 4:
            pk = tl.load(kq_ptr + base, mask=valid[:, None], other=8)
            hi4 = ((pk >> 4) & 0xF).to(tl.int32) - 8
            lo4 = (pk & 0xF).to(tl.int32) - 8
            kq = tl.interleave(hi4, lo4).to(tl.float32)
        else:
            kq = tl.load(kq_ptr + base, mask=valid[:, None], other=0).to(tl.float32)
        ks = tl.load(ks_ptr + hv * sh_s + offs * st_s, mask=valid, other=0.0)
        k = (kq * ks[:, None]).to(tl.bfloat16)

        s = tl.dot(q, tl.trans(k)) * SCALE                        # [ROWS, BLOCK_L]
        if HAS_MASK:
            # mask 形状 [Tq, L]：折叠后的第 r 行对应原第 (r % TQ) 行
            mrow = r % TQ
            mvals = tl.load(mask_ptr + mrow[:, None] * sh_m + offs[None, :] * st_m,
                            mask=valid[None, :], other=0)
            vis = valid[None, :] & mvals
        else:
            vis = valid[None, :] & (offs[None, :] <= lim[:, None])
        s = tl.where(vis, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        if BITS == 4:
            pv = tl.load(vq_ptr + base, mask=valid[:, None], other=8)
            hi4v = ((pv >> 4) & 0xF).to(tl.int32) - 8
            lo4v = (pv & 0xF).to(tl.int32) - 8
            vq = tl.interleave(hi4v, lo4v).to(tl.float32)
        else:
            vq = tl.load(vq_ptr + base, mask=valid[:, None], other=0).to(tl.float32)
        vs = tl.load(vs_ptr + hv * sh_s + offs * st_s, mask=valid, other=0.0)
        v = (vq * vs[:, None]).to(tl.bfloat16)

        acc += tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    ob = (hv * n_splits + sp) * ROWS
    tl.store(acc_ptr + ob * D + r[:, None] * D + d[None, :], acc)
    tl.store(m_ptr + ob + r, m_i)
    tl.store(l_ptr + ob + r, l_i)


def attend(q_folded, k_q, k_s, v_q, v_s, L, bits, g, tq, n_splits=8, block_l=64, mask=None):
    """q_folded [Hkv, g*tq, D] bf16 -> out [Hkv, g*tq, D] bf16（直接读量化 KV）。

    k_q/v_q: [1, Hkv, max_len, D or D/2]；k_s/v_s: [1, Hkv, max_len, 1] fp32。
    """
    Hkv, rows, D = q_folded.shape
    ROWS = 16
    qp = torch.zeros(Hkv, ROWS, D, dtype=torch.bfloat16, device=q_folded.device)
    qp[:, :rows] = q_folded
    ns = max(1, min(n_splits, triton.cdiv(L, block_l)))
    acc = torch.empty(Hkv * ns, ROWS, D, dtype=torch.float32, device=q_folded.device)
    m = torch.empty(Hkv * ns, ROWS, dtype=torch.float32, device=q_folded.device)
    low = torch.empty(Hkv * ns, ROWS, dtype=torch.float32, device=q_folded.device)
    if mask is None:
        mask_arg, sh_m, st_m = qp, 1, 1
    else:
        mask_arg = mask if mask.dim() == 2 else mask.reshape(mask.shape[-2], mask.shape[-1])
        sh_m, st_m = mask_arg.stride(0), mask_arg.stride(1)
    _split_kernel[(Hkv * ns,)](
        qp, k_q, k_s, v_q, v_s, acc, m, low, mask_arg,
        L, ns, k_q.stride(1), k_q.stride(2), k_s.stride(1), k_s.stride(2), sh_m, st_m,
        G=g, TQ=tq, D=D, ROWS=ROWS, BLOCK_L=block_l, BITS=bits,
        SCALE=1.0 / (D ** 0.5), HAS_MASK=mask is not None, num_warps=4, num_stages=2)
    acc = acc.view(Hkv, ns, ROWS, D)
    m = m.view(Hkv, ns, ROWS)
    low = low.view(Hkv, ns, ROWS)
    mmax = m.max(dim=1, keepdim=True).values
    w = torch.exp(m - mmax)
    out = (acc * w[..., None]).sum(1) / (low * w).sum(1)[..., None]
    return out[:, :rows].to(torch.bfloat16)
