"""用 sync 调试模式抓出线性注意力层里的隐式同步（1 元素拷贝花 0.6 ms = 同步）。"""
import json
import os
import sys
import traceback
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from sn_loop_common import load, make_fixed_cache, reset_cache  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
dev = "cuda"
tok, model = load("offline", int8=[])
tm = model.model.language_model
cache = make_fixed_cache(model, N_CTX + 128, kv_int8=True, pass2_bits=4)
texts = [json.loads(l)["text"] for l in open(os.path.join(HERE, "eval_texts_heldout.jsonl"),
                                             encoding="utf-8")]
ids = tok("".join(texts * 60), add_special_tokens=False, return_tensors="pt").input_ids[:, :N_CTX].cuda()

# 把 CUDA 的同步点变成 Python 栈回溯
_orig = torch.Tensor.cpu


def noisy_cpu(self, *a, **k):
    if self.is_cuda and self.numel() <= 64:
        print("--- 小张量 D2H（可疑同步）shape=", tuple(self.shape))
        traceback.print_stack(limit=6)
    return _orig(self, *a, **k)


torch.Tensor.cpu = noisy_cpu

with torch.inference_mode():
    reset_cache(cache)
    for s in range(0, N_CTX, 256):
        e = min(s + 256, N_CTX)
        model.model(input_ids=ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True)
    x = torch.randn(1, 1, tm.config.hidden_size, device=dev, dtype=torch.bfloat16)
    pos = torch.tensor([[N_CTX]], device=dev)
    pe = tm.rotary_emb(x, pos.expand(3, -1, -1))
    lin = [i for i, l in enumerate(tm.layers) if hasattr(l, "linear_attn")][0]
    print("=== 开始单层调用 ===")
    tm.layers[lin](x, position_embeddings=pe, attention_mask=None, position_ids=pos,
                   past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
