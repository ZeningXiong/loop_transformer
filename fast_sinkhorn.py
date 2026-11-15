"""2x2 Sinkhorn 的闭式解。

模型的 sinkhorn 在 log 空间做 iterations 轮「行归一化 + 列归一化」。对 2x2 矩阵，双随机矩阵必然
形如 [[p, 1-p], [1-p, p]]，而 p 由交叉比 ad/(bc) 唯一确定：

    p = sigmoid((a + d - b - c) / 2)        # a,b,c,d 是 log 域的四个元素

所以 20 轮迭代（每轮 2 次 logsumexp）可以换成一次 sigmoid + 几次拼装。streams > 2 时仍走原实现。
"""
import torch


def sinkhorn_2x2(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """logits: [..., 2, 2]，log 域。返回双随机矩阵（和原实现相同的 exp 之后的形状）。"""
    z = logits.float()
    a, b = z[..., 0, 0], z[..., 0, 1]
    c, d = z[..., 1, 0], z[..., 1, 1]
    p = torch.sigmoid(0.5 * (a + d - b - c))
    q = 1.0 - p
    out = torch.stack([torch.stack([p, q], dim=-1), torch.stack([q, p], dim=-1)], dim=-2)
    return out


def enable(flag: bool = True) -> None:
    """把模型的 module 级 sinkhorn 换成分派版本（2x2 走闭式解，其余走原实现）。"""
    import sys

    import bitsandbytes as bnb  # noqa: F401  确保模型模块已加载

    model_mod = None
    for name, m in sys.modules.items():
        if name.endswith("modeling_sensenova_loop"):
            model_mod = m
            break
    if model_mod is None:
        raise RuntimeError("modeling_sensenova_loop 还没被导入")

    if not hasattr(model_mod, "_sinkhorn_orig"):
        model_mod._sinkhorn_orig = model_mod.sinkhorn
    orig = model_mod._sinkhorn_orig

    if not flag:
        model_mod.sinkhorn = orig
        return

    def dispatch(logits, iterations):
        if logits.shape[-1] == 2:
            return sinkhorn_2x2(logits, iterations)
        return orig(logits, iterations)

    model_mod.sinkhorn = dispatch
