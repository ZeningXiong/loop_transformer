import sys, time, torch
import torch._inductor.config as ic
sys.stdout.reconfigure(encoding="utf-8")
F = torch.nn.functional
print("mixed_mm 相关配置:", {k: getattr(ic, k) for k in dir(ic) if "mixed_mm" in k})
def int8_lin(x, q, s):
    return (F.linear(x, q.to(x.dtype)).float() * s).to(x.dtype)
def int8_lin_v2(x, q, s):  # written as mul + sum: for 1 token this is a GEMV the compiler can fuse with the int8 load
    return ((x.float()[..., None, :] * q.float()).sum(-1) * s).to(x.dtype)
def bench(f, *a, n=100):
    for _ in range(5): f(*a)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f(*a)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
out_f, in_f = 12288, 4096
w = torch.randn(out_f, in_f, device="cuda") * 0.02
s = w.abs().amax(1) / 127; q = torch.round(w / s[:, None]).to(torch.int8)
x = torch.randn(1, 1, in_f, device="cuda", dtype=torch.bfloat16)
ref = int8_lin(x, q, s)
cands = {}
cands["静态形状"] = torch.compile(int8_lin, dynamic=False)
if hasattr(ic, "mixed_mm_choice"):
    ic.mixed_mm_choice = "triton"; torch._dynamo.reset()
    cands["静态 + mixed_mm=triton"] = torch.compile(int8_lin, dynamic=False)
cands["乘加写法 静态"] = torch.compile(int8_lin_v2, dynamic=False)
ic.coordinate_descent_tuning = True
cands["乘加写法 静态 + 坐标下降调优"] = torch.compile(int8_lin_v2, dynamic=False)
for k, f in cands.items():
    try:
        y = f(x, q, s); err = (y.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
        t = bench(f, x, q, s); print(f"{k:28s} {t:7.0f}us  {q.numel() / t / 1e3:5.0f} GB/s  相对误差 {err:.1e}")
    except Exception as e:
        print(k, "失败:", str(e)[:200])
