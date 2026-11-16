"""4-bit KV 的融合解包：读打包字节 + 行缩放，直接写 bf16，不经任何临时张量。

布局与 FixedKVLayer._quant4 一致：每个 uint8 存两个 4-bit（高半字节在前，偏移二进制 +8），
每行（一个 token 的一个头）一个 fp32 缩放。支持带步幅的切片（真实缓存是 [1,H,max_len,D] 的视图）。
"""
import statistics
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _deq4(packed, scale, out, L, D2, pi_h, pi_t, ps_h, ps_t, po_h, po_t,
           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    h = row // L
    t = row - h * L
    cols = tl.arange(0, BLOCK)
    mask = cols < D2
    b = tl.load(packed + h * pi_h + t * pi_t + cols, mask=mask, other=0).to(tl.int32)
    s = tl.load(scale + h * ps_h + t * ps_t)
    hi = (((b >> 4) & 0xF).to(tl.float32) - 8.0) * s
    lo = ((b & 0xF).to(tl.float32) - 8.0) * s
    j = tl.arange(0, 2)[None, :]
    v = tl.where(j == 0, hi[:, None], lo[:, None])
    tl.store(out + h * po_h + t * po_t + cols[:, None] * 2 + j, v, mask=mask[:, None])


def dequant4(packed: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    """packed [1,H,L,D/2] uint8（可带步幅）, scale [1,H,L,1] fp32 -> out [1,H,L,D] bf16（原地写）。"""
    H, L, D2 = packed.shape[1], packed.shape[2], packed.shape[-1]
    _deq4[(H * L,)](packed, scale, out, L, D2,
                    packed.stride(1), packed.stride(2),
                    scale.stride(1), scale.stride(2),
                    out.stride(1), out.stride(2),
                    BLOCK=triton.next_power_of_2(D2), num_warps=1)


def _ref(packed, s):
    hi = (packed >> 4).to(torch.int16) - 8
    lo = (packed & 0xF).to(torch.int16) - 8
    v = torch.stack([hi, lo], dim=-1).reshape(packed.shape[0], packed.shape[1], packed.shape[2],
                                              packed.shape[3] * 2)
    return (v.float() * s).to(torch.bfloat16)


if __name__ == "__main__":
    dev = "cuda"
    torch.manual_seed(0)
    H, MAXLEN, D = 4, 16448, 256
    D2 = D // 2
    kp = torch.randint(0, 256, (1, H, MAXLEN, D2), dtype=torch.uint8, device=dev)
    ks = torch.rand(1, H, MAXLEN, 1, device=dev) * 0.01
    scr = torch.zeros(1, H, MAXLEN, D, dtype=torch.bfloat16, device=dev)

    for L in (256, 16384):
        out = scr[:, :, :L]
        ref = _ref(kp[:, :, :L], ks[:, :, :L])
        out.zero_()
        dequant4(kp[:, :, :L], ks[:, :, :L], out)
        neq = (ref.view(torch.int16) != out.view(torch.int16)).sum().item()
        print(f"L={L}: 逐位差异 {neq}/{ref.numel()}", flush=True)

    def bench(fn, n=15, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e3

    out = scr[:, :, :16384]
    t_fused = bench(lambda: dequant4(kp[:, :, :16384], ks[:, :, :16384], out))
    t_torch = bench(lambda: out.copy_(_ref(kp[:, :, :16384], ks[:, :, :16384])))
    print(f"16K 前缀，每层解包：融合内核 {t_fused:.3f} ms | torch.stack {t_torch:.3f} ms "
          f"（×{t_torch / t_fused:.1f}）")
