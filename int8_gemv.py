"""Triton GEMV for weight-only int8 linear layers, one token at a time (decode).

y[n] = scale[n] * sum_k x[k] * q[n, k]      x: bf16 [K], q: int8 [N, K] row-major, scale: fp32 [N]
Reads the int8 weights exactly once, converts in registers, accumulates in fp32.
Generic: works for any model's linear layer; only the (N, K) shape changes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _int8_gemv_kernel(x_ptr, q_ptr, s_ptr, y_ptr, N, K,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        col_mask = cols < K
        x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        w = tl.load(q_ptr + rows[:, None] * K + cols[None, :],
                    mask=row_mask[:, None] & col_mask[None, :], other=0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    s = tl.load(s_ptr + rows, mask=row_mask, other=0.0)
    tl.store(y_ptr + rows, (acc * s).to(y_ptr.dtype.element_ty), mask=row_mask)


def int8_gemv(x, q, scale, block_n=16, block_k=512):
    """x: [..., K] with exactly one row (numel == K); returns [..., N] in x.dtype."""
    N, K = q.shape
    x1 = x.reshape(K).contiguous()
    y = torch.empty(N, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, block_n),)
    _int8_gemv_kernel[grid](x1, q, scale, y, N, K, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4)
    return y.reshape(*x.shape[:-1], N)
