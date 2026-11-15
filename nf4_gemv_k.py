"""NF4 multi-row GEMV: y[0..k-1] = W x[0..k-1] with W stored as bitsandbytes NF4.

The point is the k-row dimension: the packed weight is read **once** per launch and applied to every
row, instead of the k separate single-row passes that bitsandbytes' fast path (and the
`small_batch_gemv` patch) end up doing. That is what makes speculative verification cheap.

Layout (verified against bnb.functional.dequantize_4bit in debug_nf4_layout.py):
  weight [N, K] -> packed uint8 [N, K/2], high nibble = even element, low nibble = odd element,
  every run of 64 weights of a row shares one fp32 absmax.

Rows are unrolled explicitly (one accumulator each, k <= 4): the weight load and the codebook lookup are
shared, only the small multiply-add is repeated. Triton 3.1 has no list comprehensions in kernels and a
3-D broadcast over the row axis was 2.5x slower, so this is the workable shape.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _nf4_gemv_k_kernel(x_ptr, q_ptr, a_ptr, c_ptr, y_ptr, N, K,
                       BLOCK_N: tl.constexpr, KR: tl.constexpr, BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    k2 = K // 2
    nb = K // BLOCKSIZE
    HALF: tl.constexpr = BLOCKSIZE // 2
    j = tl.arange(0, HALF)
    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for b in range(0, nb):
        cols = b * HALF + j
        packed = tl.load(q_ptr + rows[:, None] * k2 + cols[None, :], mask=row_mask[:, None], other=0)
        sc = tl.load(a_ptr + rows * nb + b, mask=row_mask, other=0.0)
        even = tl.load(c_ptr + (packed >> 4).to(tl.int32)) * sc[:, None]   # scale once per block
        odd = tl.load(c_ptr + (packed & 0x0F).to(tl.int32)) * sc[:, None]
        pos = b * BLOCKSIZE + 2 * j
        xe = tl.load(x_ptr + pos).to(tl.float32)
        xo = tl.load(x_ptr + pos + 1).to(tl.float32)
        acc0 += tl.sum(even * xe[None, :] + odd * xo[None, :], axis=1)
        if KR > 1:
            xe1 = tl.load(x_ptr + K + pos).to(tl.float32)
            xo1 = tl.load(x_ptr + K + pos + 1).to(tl.float32)
            acc1 += tl.sum(even * xe1[None, :] + odd * xo1[None, :], axis=1)
        if KR > 2:
            xe2 = tl.load(x_ptr + 2 * K + pos).to(tl.float32)
            xo2 = tl.load(x_ptr + 2 * K + pos + 1).to(tl.float32)
            acc2 += tl.sum(even * xe2[None, :] + odd * xo2[None, :], axis=1)
        if KR > 3:
            xe3 = tl.load(x_ptr + 3 * K + pos).to(tl.float32)
            xo3 = tl.load(x_ptr + 3 * K + pos + 1).to(tl.float32)
            acc3 += tl.sum(even * xe3[None, :] + odd * xo3[None, :], axis=1)
    tl.store(y_ptr + rows, acc0.to(y_ptr.dtype.element_ty), mask=row_mask)
    if KR > 1:
        tl.store(y_ptr + N + rows, acc1.to(y_ptr.dtype.element_ty), mask=row_mask)
    if KR > 2:
        tl.store(y_ptr + 2 * N + rows, acc2.to(y_ptr.dtype.element_ty), mask=row_mask)
    if KR > 3:
        tl.store(y_ptr + 3 * N + rows, acc3.to(y_ptr.dtype.element_ty), mask=row_mask)


def nf4_gemv_k(x, packed, absmax, code, blocksize=64, block_n=64, warps=4):
    """x: [KR, K] with KR <= 4. packed: uint8 [N, K/2]. absmax: fp32 [N, K/blocksize]. -> [KR, N]."""
    kr, k = x.shape
    n = packed.shape[0]
    y = torch.empty((kr, n), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, block_n),)
    _nf4_gemv_k_kernel[grid](x.contiguous(), packed, absmax.reshape(n, -1), code, y, n, k,
                             BLOCK_N=block_n, KR=kr, BLOCKSIZE=blocksize, num_warps=warps)
    return y
