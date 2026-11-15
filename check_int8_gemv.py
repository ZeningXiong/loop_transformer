import os, sys, time, torch, bitsandbytes as bnb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
from int8_gemv import int8_gemv
F = torch.nn.functional
def bench(f, n=200):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
for o, i in [(12288, 4096), (4096, 12288), (8192, 4096), (4096, 4096), (1024, 4096), (248320, 4096)]:
    w = torch.randn(o, i, device="cuda") * 0.02
    s = (w.abs().amax(1) / 127).contiguous(); q = torch.round(w / s[:, None]).to(torch.int8).contiguous()
    x = torch.randn(1, 1, i, device="cuda", dtype=torch.bfloat16)
    ref = (x.float() @ (q.float() * s[:, None]).t())  # fp32 reference of the int8 layer
    best = None
    for bn, bk in [(8, 512), (16, 256), (16, 512), (32, 256), (4, 1024), (8, 1024)]:
        y = int8_gemv(x, q, s, bn, bk)
        err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
        t = bench(lambda: int8_gemv(x, q, s, bn, bk))
        if best is None or t < best[0]: best = (t, bn, bk, err)
    p4, qs = bnb.functional.quantize_4bit(w.bfloat16(), blocksize=64, quant_type="nf4", compress_statistics=False)
    t_n = bench(lambda: bnb.matmul_4bit(x, p4.t(), qs))
    t_e = bench(lambda: (F.linear(x, q.to(x.dtype)).float() * s).to(x.dtype))
    print(f"{o:6d}x{i:5d}: Triton int8 {best[0]:6.0f}us ({q.numel() / best[0] / 1e3:4.0f} GB/s, BLOCK_N={best[1]} BLOCK_K={best[2]}, 误差 {best[3]:.1e}) | 原版 int8 {t_e:6.0f}us | NF4 {t_n:6.0f}us")
