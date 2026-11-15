import sys, time, torch
sys.stdout.reconfigure(encoding="utf-8")
F = torch.nn.functional
torch._dynamo.config.cache_size_limit = 64
def int8_lin(x, q, s):
    return (F.linear(x, q.to(x.dtype)).float() * s).to(x.dtype)
def bench(f, *a, n=100):
    for _ in range(3): f(*a)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f(*a)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e6
shapes = [(12288, 4096), (4096, 12288), (8192, 4096), (4096, 4096)]
data = []
for o, i in shapes:
    q = torch.randint(-127, 127, (o, i), device="cuda", dtype=torch.int8); s = torch.rand(o, device="cuda")
    data.append((torch.randn(1, 1, i, device="cuda", dtype=torch.bfloat16), q, s))
for mode in ["dynamic=False", "dynamic=False + mark_static", "dynamic=False + automatic_dynamic 关闭"]:
    torch._dynamo.reset()
    torch._dynamo.config.automatic_dynamic_shapes = "关闭" not in mode
    f = torch.compile(int8_lin, dynamic=False)
    res = []
    for x, q, s in data:
        if "mark_static" in mode:
            torch._dynamo.mark_static(q); torch._dynamo.mark_static(x)
        res.append(f"{q.shape[0]}x{q.shape[1]}:{q.numel() / bench(f, x, q, s) / 1e3:4.0f}GB/s")
    print(f"{mode:40s}", "  ".join(res))
