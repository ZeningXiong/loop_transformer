import sys, time, torch, bitsandbytes as bnb
sys.stdout.reconfigure(encoding="utf-8")
F = torch.nn.functional
def int8_lin(x, q, s):
    return (F.linear(x, q.to(x.dtype)).float() * s).to(x.dtype)
fast = torch.compile(int8_lin, dynamic=True)
def bench(f, *a, n=200):
    for _ in range(5): f(*a)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f(*a)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
for out_f, in_f in [(12288, 4096), (4096, 12288), (248320, 4096)]:
    w = torch.randn(out_f, in_f, device="cuda") * 0.02
    s = w.abs().amax(1) / 127; q = torch.round(w / s[:, None]).to(torch.int8)
    x = torch.randn(1, 1, in_f, device="cuda", dtype=torch.bfloat16)
    ref = int8_lin(x, q, s); y = fast(x, q, s)
    err = (y.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    p4, qs = bnb.functional.quantize_4bit(w.bfloat16(), blocksize=64, quant_type="nf4", compress_statistics=False)
    t_e, t_c = bench(int8_lin, x, q, s), bench(fast, x, q, s)
    t_n = bench(lambda x: bnb.matmul_4bit(x, p4.t(), qs), x)
    xp = torch.randn(1, 300, in_f, device="cuda", dtype=torch.bfloat16)  # prefill-sized input
    t_pe, t_pc = bench(int8_lin, xp, q, s, n=50), bench(fast, xp, q, s, n=50)
    gb = q.numel() / 1e9
    print(f"[{out_f}x{in_f}] 解码1 token: int8 原版 {t_e:7.0f}us | int8 编译 {t_c:6.0f}us ({gb / t_c * 1e6:5.0f} GB/s) | NF4 {t_n:6.0f}us ; "
          f"prompt 300 token: 原版 {t_pe:6.0f}us 编译 {t_pc:6.0f}us ; 相对误差 {err:.1e}")
