import sys, time, torch
sys.stdout.reconfigure(encoding="utf-8")
def sinkhorn(logits, iterations):
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()
fast = torch.compile(sinkhorn, dynamic=True, fullgraph=True)
for shape in [(1, 1, 2, 2), (1, 37, 2, 2), (1, 1, 2, 2)]:
    x = torch.randn(*shape, device="cuda") * 3
    t = time.time(); y = fast(x, 20); torch.cuda.synchronize(); tc = time.time() - t
    ref = sinkhorn(x, 20)
    print(shape, "最大误差", (y - ref).abs().max().item(), f"首调 {tc:.1f}s")
x = torch.randn(1, 1, 2, 2, device="cuda")
for name, f in [("原版", sinkhorn), ("编译版", fast)]:
    for _ in range(3): f(x, 20)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(500): f(x, 20)
    torch.cuda.synchronize(); print(name, f"{(time.time() - t) / 500 * 1e6:.0f} 微秒/次")
