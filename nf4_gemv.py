"""NF4 single-token GEMV: y = W x with W stored as bitsandbytes NF4.

Layout (verified against bnb.functional.dequantize_4bit in debug_nf4_layout.py):
  * weight [N, K] -> packed uint8 [N, K/2], row major
  * inside a byte the **high nibble is the even element**, the low nibble the odd one
    (byte 0x3c with absmax 0.0339 gives -0.0134 then +0.0149, i.e. code[3], code[12])
  * each run of 64 weights of a row shares one fp32 absmax: value = code4[nibble] * absmax[n, k // 64]

Kernel: one program per BLOCK_N rows, walks K in BK-wide steps so every load is a wide contiguous
run per row, unpacks both nibbles, looks the codes up in the 16-entry table (64 bytes, stays in L1),
and applies the block scale to the 64-wide partial sums instead of to individual weights.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _nf4_gemv_kernel(x_ptr, q_ptr, a_ptr, c_ptr, y_ptr, N, K,
                     BLOCK_N: tl.constexpr, BLOCKSIZE: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    k2 = K // 2
    nb = K // BLOCKSIZE
    HALF: tl.constexpr = BK // 2
    G: tl.constexpr = BK // BLOCKSIZE          # scale groups per step
    G32: tl.constexpr = BLOCKSIZE // 2
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for s in range(0, K, BK):
        cols = s // 2 + tl.arange(0, HALF)
        packed = tl.load(q_ptr + rows[:, None] * k2 + cols[None, :], mask=row_mask[:, None], other=0)
        even = tl.load(c_ptr + (packed >> 4).to(tl.int32))       # high nibble -> element 2i
        odd = tl.load(c_ptr + (packed & 0x0F).to(tl.int32))      # low nibble  -> element 2i+1
        i = s + 2 * tl.arange(0, HALF)
        xe = tl.load(x_ptr + i).to(tl.float32)
        xo = tl.load(x_ptr + i + 1).to(tl.float32)
        xe = tl.reshape(xe, (G, G32))[None, :, :]
        xo = tl.reshape(xo, (G, G32))[None, :, :]
        part = (tl.reshape(even, (BLOCK_N, G, G32)) * xe
                + tl.reshape(odd, (BLOCK_N, G, G32)) * xo)
        part = tl.sum(part, axis=2)                              # [BLOCK_N, G]
        sc = tl.load(a_ptr + rows[:, None] * nb + (s // BLOCKSIZE + tl.arange(0, G))[None, :],
                     mask=row_mask[:, None], other=0.0)
        acc += tl.sum(part * sc, axis=1)
    tl.store(y_ptr + rows, acc.to(y_ptr.dtype.element_ty), mask=row_mask)


def nf4_gemv(x, packed, absmax, code, blocksize=64, block_n=64, bk=512, warps=4):
    """x: [..., K] with exactly one row. packed: uint8 [N, K/2]. absmax: fp32 [N, K/blocksize]."""
    n, k2 = packed.shape
    k = k2 * 2
    y = torch.empty(n, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(n, block_n),)
    _nf4_gemv_kernel[grid](x.reshape(k), packed, absmax.reshape(n, -1), code, y, n, k,
                           BLOCK_N=block_n, BLOCKSIZE=blocksize, BK=bk, num_warps=warps)
    return y.reshape(*x.shape[:-1], n)
