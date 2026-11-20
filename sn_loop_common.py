"""Shared loader for SenseNova-Loop-9B-Preview experiments (nf4 on GPU / bf16 offload) + Sinkhorn speed patch."""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


# 原始 FP32 检查点；可用环境变量 SENSENOVA_MODEL 覆盖（模型在 D:/E: 之间搬过，取存在的那个）
MODEL = os.environ.get("SENSENOVA_MODEL") or _first_existing(
    r"D:\models\SenseNova-Loop-9B-Preview", r"E:\models\SenseNova-Loop-9B-Preview")

NF4_DEVICE_MAP = {
    "model.language_model.embed_tokens": "cpu",  # 248320 x 4096 bf16 ~2 GB stays in RAM
    "model.visual": "cpu",                       # vision tower unused in text tests
    "model.language_model.layers": 0,
    "model.language_model.norm": 0,
    "model.language_model.rotary_emb": 0,
    "lm_head": 0,
}


def load(quant="nf4", fix_regex=False, compile_sinkhorn=True, int8=()):
    """quant: "offline" = pre-quantized NF4 dir + SSD embedding (default for new work);
    "nf4" = quantize on the fly from the original FP32 files; "bf16" = unquantized, GPU+CPU split."""
    if quant == "offline":
        return load_offline(fix_regex=fix_regex, compile_sinkhorn=compile_sinkhorn, int8=int8)
    tok_kwargs = {"fix_mistral_regex": True} if fix_regex else {}
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, **tok_kwargs)
    if quant == "nf4":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map=NF4_DEVICE_MAP,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["visual", "model.visual"], llm_int8_enable_fp32_cpu_offload=True,
            ),
        ).eval()
    elif quant == "bf16":  # unquantized: what fits on the GPU, the rest in CPU RAM (slow but exact weights)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="auto", max_memory={0: "6GiB", "cpu": "16GiB"},
        ).eval()
    else:
        raise ValueError(quant)
    if quant == "nf4":
        cpu_embedding(model)
    if compile_sinkhorn:
        patch_sinkhorn(model)
    return tok, model


class CpuEmbedding(torch.nn.Module):
    """Embedding that really runs on the CPU: looks up only the needed rows, ships them to the GPU.

    accelerate's cpu "offload" instead copies the whole 2 GB table to the GPU on every forward pass.
    """

    def __init__(self, weight, out_device):
        super().__init__()
        self.weight_cpu = weight  # plain tensor, not a Parameter: keeps it out of accelerate/.to() moves
        self.out_device = out_device

    def forward(self, input_ids):
        rows = torch.nn.functional.embedding(input_ids.cpu(), self.weight_cpu)
        return rows.to(self.out_device, non_blocking=True)


def cpu_embedding(model):
    import json
    import os

    from accelerate.hooks import remove_hook_from_module
    from safetensors import safe_open

    name = "model.language_model.embed_tokens.weight"
    index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json"), encoding="utf-8"))
    with safe_open(os.path.join(MODEL, index["weight_map"][name]), framework="pt") as f:
        weight = f.get_tensor(name).to(torch.bfloat16).contiguous()
    lm = model.model.language_model
    remove_hook_from_module(lm.embed_tokens, recurse=True)
    lm.embed_tokens = CpuEmbedding(weight, out_device=torch.device("cuda:0"))


def patch_sinkhorn(model):
    """Replace the module-level 20-iteration Sinkhorn on 2x2 matrices with a fused torch.compile version.

    Mathematically identical (max diff ~1e-7 in fp32); removes thousands of tiny kernel launches per token.
    MHCRouter.forward looks up `sinkhorn` in its module globals at call time, so rebinding it is enough.
    """
    mod = sys.modules[type(model).__module__]
    if not hasattr(mod, "_orig_sinkhorn"):
        mod._orig_sinkhorn = mod.sinkhorn
        mod.sinkhorn = torch.compile(mod._orig_sinkhorn, dynamic=True, fullgraph=True)


# ---------------------------------------------------------------------------------------------
# Offline NF4 checkpoint (made by convert_nf4.py) + embedding rows read from SSD per token
# ---------------------------------------------------------------------------------------------
NF4_DIR = os.environ.get("SENSENOVA_NF4") or _first_existing(
    r"D:\models\SenseNova-Loop-9B-Preview-nf4", r"E:\models\SenseNova-Loop-9B-Preview-nf4")


class SsdEmbedding(torch.nn.Module):
    """Reads only the embedding rows a forward pass needs from a raw bf16 file on SSD (8 KB per row).

    No table is kept in RAM. Rows land in a small CPU buffer, then go to the GPU.
    `stats` tracks how much time the reads take, to check that they are negligible.
    """

    def __init__(self, path, rows, dim, out_device):
        super().__init__()
        self.path, self.rows, self.dim = path, rows, dim
        self.row_bytes = dim * 2
        self.out_device = out_device
        self.f = open(path, "rb", buffering=0)
        self.stats = {"calls": 0, "rows": 0, "read_s": 0.0}

    def forward(self, input_ids):
        import time

        ids = input_ids.reshape(-1).tolist()
        uniq = sorted(set(ids))
        host = torch.empty((len(uniq), self.dim), dtype=torch.bfloat16)
        mv = memoryview(host.view(torch.uint8).numpy()).cast("B")
        rb = self.row_bytes
        t = time.perf_counter()
        for i, tok_id in enumerate(uniq):
            self.f.seek(tok_id * rb)
            self.f.readinto(mv[i * rb:(i + 1) * rb])
        self.stats["read_s"] += time.perf_counter() - t
        self.stats["calls"] += 1
        self.stats["rows"] += len(uniq)
        rows_gpu = host.to(self.out_device)
        if len(uniq) == len(ids) and uniq == ids:
            out = rows_gpu
        else:
            pos = {tok_id: i for i, tok_id in enumerate(uniq)}
            idx = torch.tensor([pos[x] for x in ids], device=self.out_device)
            out = rows_gpu.index_select(0, idx)
        return out.view(*input_ids.shape, self.dim)


def load_offline(fix_regex=False, compile_sinkhorn=True, device="cuda:0", int8=()):
    """Build the model skeleton on the meta device and fill it tensor by tensor from the NF4 shards.

    Nothing large ever sits in RAM: each tensor goes file -> GPU. The vision tower and MTP head are
    not created/loaded (text-only), and the embedding is replaced by SsdEmbedding.
    `int8`: names of linear weights ("...proj.weight", "lm_head.weight") to load as weight-only int8.
    """
    import json
    import os

    import bitsandbytes as bnb
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import AutoConfig, GenerationConfig

    tok_kwargs = {"fix_mistral_regex": True} if fix_regex else {}
    tok = AutoTokenizer.from_pretrained(NF4_DIR, trust_remote_code=True, **tok_kwargs)
    man = json.load(open(os.path.join(NF4_DIR, "quant_manifest.json"), encoding="utf-8"))

    cfg = AutoConfig.from_pretrained(NF4_DIR, trust_remote_code=True)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, dtype=torch.bfloat16)
    model.model.visual = None  # text-only: drop the vision tower entirely

    code = bnb.functional.get_4bit_type(man["quant_type"], device=device)
    pending = {}  # packed weight and its absmax may sit in different shards

    def install(name, packed, absmax):
        mod_name = name[: -len(".weight")]
        parent_name, _, attr = mod_name.rpartition(".")
        parent = model.get_submodule(parent_name)  # "" -> the model itself (lm_head)
        old = getattr(parent, attr)
        shape = torch.Size(man["quantized"][name]["shape"])
        new = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=old.bias is not None,
                                compute_dtype=torch.bfloat16, compress_statistics=False,
                                quant_type="nf4", quant_storage=torch.uint8, device="meta")
        qs = bnb.functional.QuantState(absmax=absmax, shape=shape, code=code, blocksize=man["blocksize"],
                                       quant_type="nf4", dtype=torch.bfloat16)
        new.weight = bnb.nn.Params4bit(data=packed, requires_grad=False, quant_state=qs,
                                       blocksize=man["blocksize"], compress_statistics=False, quant_type="nf4",
                                       quant_storage=torch.uint8, module=new, bnb_quantized=True)
        new.quant_state = qs
        setattr(parent, attr, new)

    for shard in man["shards"]:
        with safe_open(os.path.join(NF4_DIR, shard), framework="pt", device=device) as f:
            for key in f.keys():
                base = key[: -len(".absmax")] if key.endswith(".absmax") else key
                if base in man["quantized"]:
                    if base in int8:
                        continue
                    slot = pending.setdefault(base, {})
                    slot["absmax" if key.endswith(".absmax") else "packed"] = f.get_tensor(key)
                    if len(slot) == 2:
                        install(base, slot["packed"], slot["absmax"])
                        del pending[base]
                else:
                    t = f.get_tensor(key)
                    set_module_tensor_to_device(model, key, device, value=t, dtype=t.dtype)
    if pending:
        raise RuntimeError(f"incomplete quantized tensors: {list(pending)[:3]}")
    for name in int8:
        set_precision(model, name, "int8", device=device)

    for name, b in list(model.named_buffers()):  # computed buffers (rotary inv_freq) were built on CPU
        if b.device.type != "cuda":
            set_module_tensor_to_device(model, name, device, value=b, dtype=b.dtype)

    emb = man["embedding"]
    model.model.language_model.embed_tokens = SsdEmbedding(
        os.path.join(NF4_DIR, emb["file"]), emb["rows"], emb["dim"], torch.device(device))

    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.device.type == "meta"]
    if left:
        raise RuntimeError(f"{len(left)} tensors were not loaded, e.g. {left[:5]}")
    model.generation_config = GenerationConfig.from_pretrained(NF4_DIR)
    model.eval()
    import gc

    gc.collect()  # drop replaced layers (Params4bit <-> Linear4bit cycles) and loader leftovers
    torch.cuda.empty_cache()
    use_memory_efficient_attention()
    if compile_sinkhorn:
        patch_sinkhorn(model)
    import fast_sinkhorn  # 2x2 Sinkhorn 的闭式解：一步 decode 省约 23%，任务级答案不变

    fast_sinkhorn.enable(True)
    # 元素级融合：两者都逐位一致（RMSNorm 单算子 ×1.23，SiLU×up 一次 kernel 替代两次），16K 一步 −5%
    if os.environ.get("SENSENOVA_FUSE", "0") == "1":  # 默认关：单步 −5%，但端到端输出长度与基线不同，需先查清
        import fused_rmsnorm

        fused_rmsnorm.enable(True)
        try:
            from fused_silu import enable_silu

            enable_silu(True)
        except Exception:  # 没有 triton 时静默回退
            pass
    return tok, model


# ---------------------------------------------------------------------------------------------
# Mixed precision: per-layer NF4 / weight-only int8 / bf16
# ---------------------------------------------------------------------------------------------
SRC_FP32 = MODEL  # original FP32 checkpoint, only needed for the bf16 option


class Int8WOLinear(torch.nn.Module):
    """Weight-only int8 linear: W = q * s (q int8 [out, in], s fp32 per output row).

    Activations stay bf16 (unlike bitsandbytes LLM.int8, which also quantizes activations).
    """

    def __init__(self, q, scale):
        super().__init__()
        self.register_buffer("q", q, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.in_features, self.out_features = q.shape[1], q.shape[0]
        self.bias = None

    def forward(self, x):
        rows = x.numel() // self.in_features
        bn, bk = (4, 1024) if self.in_features >= 8192 else (8, 512)  # tuned on the 4070 Laptop
        if rows == 1:  # single token (decode): Triton GEMV reading int8 directly, ~5x faster
            from int8_gemv import int8_gemv

            return int8_gemv(x, self.q, self.scale, bn, bk)
        if rows <= 4:
            # Speculative verification (2-4 rows). The generic path below casts the whole int8 weight
            # back to bf16 on every call (2.2GB across the model); calling the fast kernel per row is
            # ~6x cheaper at 2 rows (measured: 89ms -> 30ms for all 106 int8 layers).
            from int8_gemv import int8_gemv

            xs = x.reshape(rows, 1, self.in_features)
            out = torch.cat([int8_gemv(xs[i], self.q, self.scale, bn, bk) for i in range(rows)], dim=0)
            return out.reshape(*x.shape[:-1], self.out_features)
        return _int8_linear(x, self.q, self.scale)  # prompt: plain dequant + matmul


def _int8_linear(x, q, scale):
    y = torch.nn.functional.linear(x, q.to(x.dtype))  # int8 values are exact in bf16
    return (y.float() * scale).to(x.dtype)



_key_index = {}


def _index(kind):
    """Map tensor key -> shard file for the NF4 ("nf4") or int8 ("int8") checkpoints."""
    import json
    import os

    from safetensors import safe_open

    if kind not in _key_index:
        man = json.load(open(os.path.join(NF4_DIR, "quant_manifest.json" if kind == "nf4" else "int8_manifest.json"),
                             encoding="utf-8"))
        idx = {}
        for shard in man["shards"]:
            with safe_open(os.path.join(NF4_DIR, shard), framework="pt") as f:
                for k in f.keys():
                    idx[k] = shard
        _key_index[kind] = idx
    return _key_index[kind]


def _get(kind, key, device):
    import os

    from safetensors import safe_open

    with safe_open(os.path.join(NF4_DIR, _index(kind)[key]), framework="pt", device=device) as f:
        return f.get_tensor(key)


def module_bytes(mod):
    n = 0
    for t in list(mod.parameters()) + list(mod.buffers()):
        n += t.numel() * t.element_size()
    qs = getattr(getattr(mod, "weight", None), "quant_state", None)
    if qs is not None:
        n += qs.absmax.numel() * qs.absmax.element_size()
    return n


def set_precision(model, name, prec, device="cuda:0"):
    """Swap one linear layer ("...weight" name) to "nf4", "int8" or "bf16" in place. Returns new GPU bytes."""
    import bitsandbytes as bnb

    mod_name = name[: -len(".weight")]
    parent_name, _, attr = mod_name.rpartition(".")
    parent = model.get_submodule(parent_name)
    old = getattr(parent, attr)
    in_f, out_f = old.in_features, old.out_features
    if prec == "nf4":
        packed, absmax = _get("nf4", name, device), _get("nf4", name + ".absmax", device)
        new = bnb.nn.Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.bfloat16, compress_statistics=False,
                                quant_type="nf4", quant_storage=torch.uint8, device="meta")
        qs = bnb.functional.QuantState(absmax=absmax, shape=torch.Size([out_f, in_f]),
                                       code=bnb.functional.get_4bit_type("nf4", device=device), blocksize=64,
                                       quant_type="nf4", dtype=torch.bfloat16)
        new.weight = bnb.nn.Params4bit(data=packed, requires_grad=False, quant_state=qs, blocksize=64,
                                       compress_statistics=False, quant_type="nf4", quant_storage=torch.uint8,
                                       module=new, bnb_quantized=True)
        new.quant_state = qs
    elif prec == "int8":
        new = Int8WOLinear(_get("int8", name, device), _get("int8", name + ".scale", device))
    elif prec == "bf16":
        import json
        import os

        from safetensors import safe_open

        wm = json.load(open(os.path.join(SRC_FP32, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
        with safe_open(os.path.join(SRC_FP32, wm[name]), framework="pt") as f:
            sl = f.get_slice(name)
            rows = sl.get_shape()[0]
            w = torch.empty((rows, in_f), dtype=torch.bfloat16, device=device)
            for r in range(0, rows, 16384):
                w[r:r + 16384] = sl[r:min(r + 16384, rows)].to(device, torch.bfloat16)
        new = torch.nn.Linear(in_f, out_f, bias=False, device="meta")
        new.weight = torch.nn.Parameter(w, requires_grad=False)
    else:
        raise ValueError(prec)
    setattr(parent, attr, new)
    del old
    return module_bytes(new)


def use_memory_efficient_attention():
    """Make long prompts use O(T) attention memory instead of O(T^2).

    transformers passes enable_gqa=True to SDPA on CUDA, assuming FlashAttention exists. Windows torch
    builds have no FlashAttention, and in torch 2.5 the memory-efficient kernel does not accept
    enable_gqa, so SDPA silently falls back to the MATH kernel, which materializes the full
    16 x T x T score matrix (~1 GB at T=2048, ~16 GB at T=8192). Repeating K/V to 16 heads first
    lets SDPA use the memory-efficient kernel. Verified: held-out KL vs bf16 0.0501 -> 0.0503.
    """
    import transformers.integrations.sdpa_attention as sa

    sa.use_gqa_in_sdpa = lambda *args, **kwargs: False


# ---------------------------------------------------------------------------------------------
# Chunked prefill + own decode loop, with a whole-card VRAM guard
# ---------------------------------------------------------------------------------------------
class VramExceeded(RuntimeError):
    pass


class VramGuard:
    """Polls whole-card VRAM (all processes). Once usage reaches `limit_frac` of the physical card,
    Windows starts spilling into system RAM; the run is then aborted instead of crawling on."""

    def __init__(self, limit_frac=0.97, interval=0.02):
        import threading

        self.limit_frac, self.interval = limit_frac, interval
        self.peak = 0
        self.tripped = False
        self._stop = False
        self.total = torch.cuda.mem_get_info()[1]
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        import time

        while not self._stop:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            self.peak = max(self.peak, used)
            if used >= self.limit_frac * total:
                self.tripped = True
            time.sleep(self.interval)

    def check(self):
        if self.tripped:
            raise VramExceeded(f"显存达到 {self.peak / 1e9:.2f}/{self.total / 1e9:.2f} GB，超出 8GB 限制，停止本次测试")

    def close(self):
        self._stop = True


@torch.inference_mode()
def generate_chunked(model, input_ids, max_new_tokens, chunk_size=512, eos_token_ids=None, guard=None, cache=None):
    """Greedy generation whose prompt-processing memory depends on chunk_size, not on prompt length.

    The prompt is fed in chunks of `chunk_size` tokens with explicit text positions (for text-only input
    the 3D M-RoPE rows are identical, so plain positions are exact); linear-attention conv/recurrent
    states and the full-attention KV cache carry over between chunks. Only the last position's logits
    are computed. Returns (new_token_ids list, stats dict).
    """
    import time

    dev = input_ids.device
    T = input_ids.shape[1]
    if eos_token_ids is None:
        eos = model.generation_config.eos_token_id
        eos_token_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    if cache is not None:
        reset_cache(cache)  # reuse a preallocated fixed-size cache
    t0 = time.perf_counter()
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        out = model(input_ids=input_ids[:, s:e], position_ids=torch.arange(s, e, device=dev)[None],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        if guard is not None:
            torch.cuda.synchronize()
            guard.check()
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    last_logits = out.logits[:, -1].float()
    nxt = last_logits.argmax(-1)
    new = [nxt.item()]
    t1 = time.perf_counter()
    for i in range(max_new_tokens - 1):
        if new[-1] in eos_token_ids:
            break
        pos = torch.tensor([[T + i]], device=dev)
        out = model(input_ids=nxt[:, None], position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        new.append(nxt.item())
        if guard is not None:
            guard.check()
    t_decode = time.perf_counter() - t1
    return new, {"prefill_s": t_prefill, "decode_s": t_decode, "decode_tokens": len(new) - 1,
                 "first_logits": last_logits}


# ---------------------------------------------------------------------------------------------
# Fixed-size KV buffers + GQA attention without K/V repetition + warm-up
# ---------------------------------------------------------------------------------------------
class ContextFull(RuntimeError):
    pass


def _fixed_kv_layer_class():
    from transformers.cache_utils import DynamicLayer

    class FixedKVLayer(DynamicLayer):
        """Full-attention KV cache preallocated to `max_len` tokens. update() writes in place and returns
        views of the filled part, so the KV cache never reallocates (no growth, no fragmentation).

        int8 mode: K and V are stored as int8 with one fp32 scale per (head, token). On each update the
        filled part is dequantized into a bf16 scratch buffer shared by all layers (layers run one after
        another, so one scratch is enough) and that view is returned to the attention.
        """

        def __init__(self, max_len, n_heads, head_dim, dtype, device, scratch=None, recent=0, bits=8):
            super().__init__()
            self.max_len = max_len
            self.int8 = scratch is not None
            self.recent = int(recent) if (scratch is not None and recent) else 0
            self.bits = int(bits) if self.int8 else 8
            head_dim = head_dim
            shape = (1, n_heads, max_len, head_dim)
            if self.int8:
                if self.bits == 4:
                    shape = (1, n_heads, max_len, head_dim // 2)
                    self.k_q = torch.zeros(shape, dtype=torch.uint8, device=device)
                    self.v_q = torch.zeros_like(self.k_q)
                else:
                    self.k_q = torch.zeros(shape, dtype=torch.int8, device=device)
                    self.v_q = torch.zeros_like(self.k_q)
                self.k_s = torch.zeros((1, n_heads, max_len, 1), dtype=torch.float32, device=device)
                self.v_s = torch.zeros_like(self.k_s)
                self.head_dim = head_dim
                self.scratch = scratch  # (k_bf16, v_bf16), each [1, n_heads, max_len, head_dim]
                if self.recent:
                    wshape = (1, n_heads, self.recent, head_dim)
                    self.win_k = torch.zeros(wshape, dtype=dtype, device=device)
                    self.win_v = torch.zeros_like(self.win_k)
            else:
                self.keys_buf = torch.zeros(shape, dtype=dtype, device=device)
                self.values_buf = torch.zeros_like(self.keys_buf)
            self.dtype, self.device = dtype, device
            self.cur = 0
            self.is_initialized = True
            self._set_views()

        def _set_views(self):
            if self.int8:
                ks, vs = self.scratch
                self.keys, self.values = ks[:, :, :self.cur], vs[:, :, :self.cur]
            else:
                self.keys, self.values = self.keys_buf[:, :, :self.cur], self.values_buf[:, :, :self.cur]

        def storage_bytes(self):
            ts = (self.k_q, self.v_q, self.k_s, self.v_s) if self.int8 else (self.keys_buf, self.values_buf)
            return sum(t.numel() * t.element_size() for t in ts)

        def lazy_initialization(self, key_states, value_states):
            pass

        @staticmethod
        def _quant(x):
            s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
            return torch.round(x.float() / s).clamp_(-127, 127).to(torch.int8), s

        @staticmethod
        def _quant4(x):
            """[1,H,T,D] -> packed [1,H,T,D/2] uint8（每字节两个 4-bit，偏移二进制）+ scale。"""
            s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
            q = torch.round(x.float() / s).clamp_(-7, 7).to(torch.int16)
            q = q.view(1, x.shape[1], x.shape[2], x.shape[3] // 2, 2)
            packed = ((q[..., 0] + 8) * 16 + (q[..., 1] + 8)).to(torch.uint8)
            return packed, s

        @staticmethod
        def _dequant4(packed, s, out_view):
            hi = (packed >> 4).to(torch.int16) - 8
            lo = (packed & 0xF).to(torch.int16) - 8
            v = torch.stack([hi, lo], dim=-1)
            v = v.reshape(packed.shape[0], packed.shape[1], packed.shape[2], packed.shape[3] * 2)
            torch.mul(v.float(), s, out=out_view)

        def update(self, key_states, value_states, *args, **kwargs):
            t = key_states.shape[-2]
            a, b = self.cur, self.cur + t
            if b > self.max_len:
                raise ContextFull(f"上下文已满：{self.cur} + {t} > 上限 {self.max_len} token")
            if self.int8 and self.bits == 4:
                kq, ks = self._quant4(key_states)
                vq, vs = self._quant4(value_states)
                self.k_q[:, :, a:b].copy_(kq)
                self.k_s[:, :, a:b].copy_(ks)
                self.v_q[:, :, a:b].copy_(vq)
                self.v_s[:, :, a:b].copy_(vs)
                kb, vb = self.scratch
                from int4_dequant import dequant4  # 融合解包：不经临时张量，逐位一致
                dequant4(self.k_q[:, :, :b], self.k_s[:, :, :b], kb[:, :, :b])
                dequant4(self.v_q[:, :, :b], self.v_s[:, :, :b], vb[:, :, :b])
            elif self.int8 and self.recent:
                W = self.recent
                cold = max(0, b - W)                      # [0, cold) 走 int8，[cold, b) 走 bf16 窗口
                # 1) 之前留在窗口里、这次离开窗口的位置：从环里取出后量化
                ring_lo = max(0, a - W)
                if cold > ring_lo:
                    idx = torch.arange(ring_lo, cold, device=self.device) % W
                    src_k = self.win_k.index_select(2, idx)
                    src_v = self.win_v.index_select(2, idx)
                    kq, ks = self._quant(src_k)
                    vq, vs = self._quant(src_v)
                    self.k_q[:, :, ring_lo:cold].copy_(kq)
                    self.k_s[:, :, ring_lo:cold].copy_(ks)
                    self.v_q[:, :, ring_lo:cold].copy_(vq)
                    self.v_s[:, :, ring_lo:cold].copy_(vs)
                # 2) 新 token：落在窗口外的直接量化，落在窗口内的写进环
                new_cold = max(a, cold)
                if new_cold > a:
                    kq, ks = self._quant(key_states[:, :, :new_cold - a])
                    vq, vs = self._quant(value_states[:, :, :new_cold - a])
                    self.k_q[:, :, a:new_cold].copy_(kq)
                    self.k_s[:, :, a:new_cold].copy_(ks)
                    self.v_q[:, :, a:new_cold].copy_(vq)
                    self.v_s[:, :, a:new_cold].copy_(vs)
                if b > new_cold:
                    idx = torch.arange(new_cold, b, device=self.device) % W
                    self.win_k.index_copy_(2, idx, key_states[:, :, new_cold - a:].contiguous())
                    self.win_v.index_copy_(2, idx, value_states[:, :, new_cold - a:].contiguous())
                # 3) 反量化：冷段走 int8，窗口段拷 bf16
                kb, vb = self.scratch
                if cold:
                    torch.mul(self.k_q[:, :, :cold], self.k_s[:, :, :cold], out=kb[:, :, :cold])
                    torch.mul(self.v_q[:, :, :cold], self.v_s[:, :, :cold], out=vb[:, :, :cold])
                if b > cold:
                    idx = torch.arange(cold, b, device=self.device) % W
                    kb[:, :, cold:b].copy_(self.win_k.index_select(2, idx))
                    vb[:, :, cold:b].copy_(self.win_v.index_select(2, idx))
            elif self.int8:
                kq, ks = self._quant(key_states)
                vq, vs = self._quant(value_states)
                self.k_q[:, :, a:b].copy_(kq)
                self.k_s[:, :, a:b].copy_(ks)
                self.v_q[:, :, a:b].copy_(vq)
                self.v_s[:, :, a:b].copy_(vs)
                kb, vb = self.scratch
                torch.mul(self.k_q[:, :, :b], self.k_s[:, :, :b], out=kb[:, :, :b])  # fp32 math, rounded to bf16
                torch.mul(self.v_q[:, :, :b], self.v_s[:, :, :b], out=vb[:, :, :b])
            else:
                self.keys_buf[:, :, a:b].copy_(key_states)
                self.values_buf[:, :, a:b].copy_(value_states)
            self.cur = b
            self._set_views()
            return self.keys, self.values

        def get_seq_length(self):
            return self.cur

        def reset(self):
            self.cur = 0
            self._set_views()

    return FixedKVLayer


def _plan_pairs(model):
    """(物理层, 第一遍执行槽, 第二遍执行槽)：只对出现两次的全注意力层。"""
    lm = model.model.language_model
    types = model.config.text_config.layer_types
    seen = {}
    for ei, pi in enumerate(lm._loop_plan):
        if types[pi] == "full_attention":
            if pi in seen:
                yield pi, seen[pi], ei
            else:
                seen[pi] = ei


def make_fixed_cache(model, max_len, device="cuda:0", kv_int8=False, share_k=False, recent=0,
                     pass2_bits=8):
    """A loop-model DynamicCache whose 14 full-attention slots are FixedKVLayer(max_len) (bf16 or int8).

    share_k=True: 两遍循环共用一份 K 平面（第二遍原地覆盖第一遍写的区间），V 仍按遍独立。
    recent=W: 最近 W 个 token 的 K/V 保持 bf16（只有更早的位置走 int8），像 LoopVQ 的 serving 配方。
    pass2_bits=4: 循环段的第二遍执行槽用 4-bit 打包（8 就是现在的 int8）。
    """
    from transformers.cache_utils import DynamicCache, DynamicLayer

    FixedKVLayer = _fixed_kv_layer_class()
    tm = model.model.language_model
    tc = model.config.text_config
    cache = DynamicCache(config=tm._loop_cache_config)
    cache._loop_execution_plan = tm._loop_plan
    shape = (1, tc.num_key_value_heads, max_len, tc.head_dim)
    scratch = (torch.zeros(shape, dtype=torch.bfloat16, device=device),
               torch.zeros(shape, dtype=torch.bfloat16, device=device)) if kv_int8 else None
    second = {e2 for _pi, e1, e2 in _plan_pairs(model)}
    n = 0
    for i, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:
            bits = pass2_bits if (pass2_bits != 8 and i in second) else 8
            cache.layers[i] = FixedKVLayer(max_len, tc.num_key_value_heads, tc.head_dim, torch.bfloat16, device,
                                           scratch=scratch, recent=recent, bits=bits)
            n += 1
    assert n == 14, n
    if share_k:
        for _pi, e1, e2 in _plan_pairs(model):
            a, b = cache.layers[e1], cache.layers[e2]
            if a.int8:
                b.k_q, b.k_s = a.k_q, a.k_s
            else:
                b.keys_buf = a.keys_buf
                b._set_views()
    seen, total = set(), 0
    for layer in cache.layers:
        if isinstance(layer, FixedKVLayer):
            bufs = ((layer.k_q, layer.k_s, layer.v_q, layer.v_s) if layer.int8
                    else (layer.keys_buf, layer.values_buf))
            for t in bufs:
                if id(t) not in seen:
                    seen.add(id(t))
                    total += t.numel() * t.element_size()
    cache._fixed_bytes = total
    cache._scratch_bytes = sum(t.numel() * t.element_size() for t in scratch) if scratch else 0
    cache._scratch = scratch
    return cache


def reset_cache(cache):
    for layer in cache.layers:
        layer.reset()


def gqa_fold_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """Grouped-query attention without repeating K/V and without enable_gqa.

    The g query heads sharing one KV head are stacked along the sequence axis: [1, Hkv, g*Tq, D] against
    K/V [1, Hkv, L, D] is plain multi-head attention, which the memory-efficient SDPA kernel accepts.
    The causal mask is rebuilt for the stacked rows (queries are the last Tq of the L positions).
    The kernel is forced: if it cannot run, this raises instead of silently using the O(T^2) MATH kernel.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, Hq, Tq, D = query.shape
    Hkv, L = key.shape[1], key.shape[2]
    g = Hq // Hkv
    assert B == 1, "batch size 1 only"
    q = query.reshape(B, Hkv, g * Tq, D)
    if Tq == 1:
        mask = None
    else:
        qpos = torch.arange(L - Tq, L, device=query.device)
        allowed = torch.arange(L, device=query.device)[None, :] <= qpos[:, None]  # [Tq, L]
        mask = allowed.repeat(g, 1)[None, None]                                    # [1, 1, g*Tq, L]
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(q, key, value, attn_mask=mask, scale=scaling)
    out = out.reshape(B, Hq, Tq, D).transpose(1, 2).contiguous()
    return out, None


def use_gqa_fold_attention(enable=True):
    """Route the model's 'sdpa' attention through gqa_fold_attention (or restore the original)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import transformers.integrations.sdpa_attention as sa

    ALL_ATTENTION_FUNCTIONS["sdpa"] = gqa_fold_attention if enable else sa.sdpa_attention_forward


def warm_up(model, chunk_size=512, device="cuda:0", kv_int8=False):
    """Compile every kernel path (first chunk, continued chunk, single-token decode) on a tiny cache,
    so the one-off Triton compile/autotune spike happens before the big KV buffers are allocated."""
    import gc

    small = make_fixed_cache(model, 2 * chunk_size + 64, device, kv_int8=kv_int8)
    ids = torch.randint(1000, 20000, (1, chunk_size + 100), device=device)
    generate_chunked(model, ids, 8, chunk_size=chunk_size, cache=small)
    del small
    gc.collect()
    torch.cuda.empty_cache()
