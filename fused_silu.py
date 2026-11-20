"""SiLU×up 融合：把 MLP 里的 `act_fn(gate) * up` 合成一个 kernel。

PyTorch 的顺序是：silu(g) 用 fp32 算、舍入到 bf16；再与 u 相乘（fp32 算、舍入到 bf16）。
所以内核必须显式在中间舍入一次，否则结果会差 1 ulp——用 PTX `cvt.rn.bf16.f32` 做舍入栅栏。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = g / (1.0 + tl.exp(-g))
    # 舍入栅栏：显式走一次 bf16（编译器不能把这一步优化掉）
    s = tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", "=h,r", [s], dtype=tl.bfloat16, is_pure=True, pack=1
    ).to(tl.float32)
    tl.store(out_ptr + offs, s * u, mask=mask)


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(g)
    n = g.numel()
    block = 1024
    _silu_mul_kernel[(triton.cdiv(n, block),)](g, u, out, n, BLOCK=block, num_warps=4)
    return out


def enable_silu(flag: bool = True) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    if not hasattr(hf, "_mlp_orig"):
        hf._mlp_orig = hf.Qwen3_5MLP.forward
    if not flag:
        hf.Qwen3_5MLP.forward = hf._mlp_orig
        return

    def forward(self, x):
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))

    hf.Qwen3_5MLP.forward = forward


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for shape in ((1, 1, 11008), (1, 7, 11008), (1, 512, 11008)):
        g = torch.randn(shape, device=dev, dtype=torch.bfloat16) * 2
        u = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        ref = torch.nn.functional.silu(g) * u
        got = silu_mul(g, u)
        neq = (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
        print(f"{tuple(shape)}: 逐位差异 {neq}/{ref.numel()}")
