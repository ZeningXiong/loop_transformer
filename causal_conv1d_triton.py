"""线性注意力的因果卷积更新：PyTorch 的 grouped conv1d 在 T=1、groups=8192 时走 im2col，
每层要 1.0–1.4 ms 的拷贝。这里用一个小 Triton kernel 直接算 4 个 tap。

out[c] = silu(sum_k w[c,k] * [state(c,0), state(c,1), state(c,2), x(c)][k] + b[c])
state 左移一格后写入 x(c)。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _conv_silu_kernel(state_ptr, x_ptr, w_ptr, b_ptr, out_ptr, C, HAS_BIAS: tl.constexpr,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C
    s0 = tl.load(state_ptr + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(state_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(state_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(4):
        w = tl.load(w_ptr + offs * 4 + k, mask=mask, other=0.0).to(tl.float32)
        v = s0 if k == 0 else (s1 if k == 1 else (s2 if k == 2 else xv))
        acc += w * v
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = acc / (1.0 + tl.exp(-acc))
    tl.store(out_ptr + offs, y, mask=mask)
    # 状态左移：state = [s1, s2, x]
    tl.store(state_ptr + offs * 3 + 0, s1, mask=mask)
    tl.store(state_ptr + offs * 3 + 1, s2, mask=mask)
    tl.store(state_ptr + offs * 3 + 2, xv, mask=mask)


def conv_silu_update(hidden_states, conv_state, weight, bias=None, activation="silu"):
    """hidden_states [1, C, 1]，conv_state [1, C, 3]，weight [C, 1, 4]。"""
    C = hidden_states.shape[1]
    w = weight.reshape(C, 4).contiguous()
    x = hidden_states.reshape(-1).contiguous()
    st = conv_state.reshape(-1)                      # 原地：左移并写入 x
    out = torch.empty_like(x)
    block = 256
    _conv_silu_kernel[(triton.cdiv(C, block),)](
        st, x, w, bias if bias is not None else w, out, C,
        HAS_BIAS=bias is not None, BLOCK=block, num_warps=4)
    return out.view(1, C, 1)


def enable(flag: bool = True) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    if not hasattr(hf, "_conv_orig"):
        hf._conv_orig = hf.causal_conv1d_update
    if not flag:
        hf.causal_conv1d_update = hf._conv_orig
        return

    def patched(hidden_states, conv_state, weight, bias=None, activation=None):
        if hidden_states.shape[-1] != 1 or conv_state.shape[-1] != 3 or activation != "silu":
            return hf._conv_orig(hidden_states, conv_state, weight, bias, activation)
        return conv_silu_update(hidden_states, conv_state, weight, bias, activation)

    hf.causal_conv1d_update = patched


if __name__ == "__main__":
    torch.manual_seed(0)
    C = 8192
    st = torch.randn(1, C, 3, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, C, 1, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(C, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(C, device="cuda", dtype=torch.bfloat16) * 0.1

    # 参考实现（HF 的回退路径）
    st_ref = st.clone()
    hs = torch.cat([st_ref, x], dim=-1).to(w.dtype)
    st_ref.copy_(hs[:, :, -3:])
    ref = torch.nn.functional.conv1d(hs, w, b, padding=0, groups=C)[:, :, -1:]
    ref = torch.nn.functional.silu(ref).to(x.dtype)

    st_new = st.clone()
    got = conv_silu_update(x, st_new, w, b)
    neq = (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
    print(f"conv+silu 逐位差异 {neq}/{ref.numel()}，状态一致 {torch.equal(st_new, st_ref)}")

    import time
    for name, fn in (("参考 conv1d", lambda: torch.nn.functional.silu(
            torch.nn.functional.conv1d(torch.cat([st, x], -1).to(w.dtype), w, b, groups=C)[:, :, -1:])),
                     ("Triton", lambda: conv_silu_update(x, st_new, w, b))):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        print(f"  {name:<12}{(time.perf_counter() - t0) / 50 * 1e3:.3f} ms")
