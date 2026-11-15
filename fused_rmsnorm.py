"""RMSNorm 融合（第一步最小验证）：保持 PyTorch 的 mean 顺序，只把尾巴融成一个 Triton kernel。

PyTorch 的 Qwen3_5RMSNorm 一步是 8~9 个 kernel：
    x.float() → pow(2) → mean → +eps → rsqrt → *x → *(1+weight) → .type_as(bf16)

这里只融合后半段（+eps → rsqrt → *x → *(1+w) → 转 bf16），前面的 pow/mean 仍交给 PyTorch，
所以方差在**逐位完全相同**的意义上算出来——这正是论文里说的"把 mean 留给 PyTorch、用 fenced
Triton tail 接管其余部分"。

两条路径：
  * 连续输入（占绝大多数：每层两个 norm）走无步幅参数的快路径；
  * q_norm / k_norm 的输入是 qkv 打包张量的非连续视图（行步幅 512、列步幅 1），走带步幅的路径。
    （这个坑是靠逐位比对抓出来的：不按真实步幅寻址，第 2 行起会读错元素。）
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tail_contig(x_ptr, w_ptr, var_ptr, out_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + row)
    y = (x * tl.rsqrt(var + eps)) * (1.0 + w)
    tl.store(out_ptr + row * D + cols, y, mask=mask)


@triton.jit
def _tail_strided(x_ptr, w_ptr, var_ptr, out_ptr, D, eps, rs, cs, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * rs + cols * cs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + row)
    y = (x * tl.rsqrt(var + eps)) * (1.0 + w)
    tl.store(out_ptr + row * D + cols, y, mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [..., D] bf16（可非连续）；weight: [D]（模型里存的是偏移量，实际乘 1+weight）。"""
    shape = x.shape
    d = shape[-1]
    xf = x.reshape(-1, d)
    var = xf.float().pow(2).mean(-1, keepdim=True)      # PyTorch 自己的求和顺序
    out = torch.empty(xf.shape, dtype=x.dtype, device=x.device)
    block = max(triton.next_power_of_2(d), 128)
    warps = min(8, max(1, block // 512))
    if xf.stride(1) == 1 and xf.stride(0) == d:
        _tail_contig[(xf.shape[0],)](xf, weight, var, out, d, eps, BLOCK=block, num_warps=warps)
    else:
        _tail_strided[(xf.shape[0],)](xf, weight, var, out, d, eps, xf.stride(0), xf.stride(1),
                                      BLOCK=block, num_warps=warps)
    return out.reshape(shape)


def enable(flag: bool = True) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    if not hasattr(hf, "_rmsnorm_orig"):
        hf._rmsnorm_orig = hf.Qwen3_5RMSNorm.forward
    if not flag:
        hf.Qwen3_5RMSNorm.forward = hf._rmsnorm_orig
        return

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)

    hf.Qwen3_5RMSNorm.forward = forward
