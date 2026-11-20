"""CUDA-graph decode for SenseNova-Loop-9B-Preview.

What has to be static, and how it is handled here:
  * KV write position  -> the layer writes with `index_copy_` at an index buffer that is filled once
                          per step from Python (a device fill, outside the graph).
  * KV read length     -> the layer hands attention a view of `pad` slots instead of `:cur`, and the
                          caller supplies an explicit mask over those slots (all-True up to cur).
  * token / position   -> copied into static input buffers before each replay.
  * embedding read     -> happens outside the graph (it is a Python file read from SSD), its output is
                          copied into the static `inputs_embeds` buffer.

Buckets: one graph per padded length (256 ... 32768), captured lazily the first time a sequence needs
it. Attention then always reads `pad` keys, so the cost is bounded by the next bucket, not by 32K.
"""
import os
import sys

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


class ContextFull(RuntimeError):
    pass


class GraphKVLayer(DynamicLayer):
    """Fixed-size KV slot with a graph-friendly write path (see module docstring)."""

    def __init__(self, max_len, n_heads, head_dim, dtype, device, idx, scratch=None, bits=8):
        super().__init__()
        self.max_len = max_len
        self.int8 = scratch is not None
        self.bits = int(bits) if self.int8 else 8
        self.head_dim = head_dim
        self.shape = (1, n_heads, max_len, head_dim if self.bits == 8 else head_dim // 2)
        self.device = device
        self.idx = idx                                  # shared [1] long buffer, filled per step
        self.pad = None                                 # None -> attention sees :cur (prefill)
        if self.int8:
            if self.bits == 4:
                self.k_q = torch.zeros(self.shape, dtype=torch.uint8, device=device)
            else:
                self.k_q = torch.zeros(self.shape, dtype=torch.int8, device=device)
            self.v_q = torch.zeros_like(self.k_q)
            self.k_s = torch.zeros((1, n_heads, max_len, 1), dtype=torch.float32, device=device)
            self.v_s = torch.zeros_like(self.k_s)
            self.scratch = scratch
        else:
            self.keys_buf = torch.zeros(self.shape, dtype=dtype, device=device)
            self.values_buf = torch.zeros_like(self.keys_buf)
        self.cur = 0
        self.is_initialized = True
        self._set_views()

    # -- views -----------------------------------------------------------------------------------
    def _set_views(self):
        n = self.cur if self.pad is None else self.pad
        if self.int8:
            ks, vs = self.scratch
            self.keys, self.values = ks[:, :, :n], vs[:, :, :n]
        else:
            self.keys, self.values = self.keys_buf[:, :, :n], self.values_buf[:, :, :n]

    def set_pad(self, pad):
        self.pad = pad
        self._set_views()

    def storage_bytes(self):
        ts = ((self.k_q, self.v_q, self.k_s, self.v_s) if self.int8
              else (self.keys_buf, self.values_buf))
        return sum(t.numel() * t.element_size() for t in ts)

    def get_seq_length(self):
        return self.cur

    def lazy_initialization(self, *a, **k):
        pass

    def reset(self):
        self.cur = 0
        self._set_views()

    @staticmethod
    def _quant(x):
        s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
        return torch.round(x.float() / s).clamp_(-127, 127).to(torch.int8), s

    @staticmethod
    def _quant4(x):
        s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
        q = torch.round(x.float() / s).clamp_(-7, 7).to(torch.int16)
        q = q.view(1, x.shape[1], x.shape[2], x.shape[3] // 2, 2)
        return ((q[..., 0] + 8) * 16 + (q[..., 1] + 8)).to(torch.uint8), s

    # -- write -----------------------------------------------------------------------------------
    def update(self, key_states, value_states, *args, **kwargs):
        t = key_states.shape[-2]
        a, b = self.cur, self.cur + t
        if b > self.max_len:
            raise ContextFull(f"上下文已满：{self.cur} + {t} > {self.max_len}")
        q4 = self.int8 and self.bits == 4
        if self.int8:
            kq, ks = (self._quant4 if q4 else self._quant)(key_states)
            vq, vs = (self._quant4 if q4 else self._quant)(value_states)
            if t == 1:                                  # decode: capturable, position from the buffer
                self.k_q.index_copy_(2, self.idx, kq)
                self.k_s.index_copy_(2, self.idx, ks)
                self.v_q.index_copy_(2, self.idx, vq)
                self.v_s.index_copy_(2, self.idx, vs)
                n = self.pad if self.pad is not None else b
            else:                                       # prefill: plain slice writes
                self.k_q[:, :, a:b].copy_(kq)
                self.k_s[:, :, a:b].copy_(ks)
                self.v_q[:, :, a:b].copy_(vq)
                self.v_s[:, :, a:b].copy_(vs)
                n = b
            kb, vb = self.scratch
            if q4:
                from int4_dequant import dequant4
                dequant4(self.k_q[:, :, :n], self.k_s[:, :, :n], kb[:, :, :n])
                dequant4(self.v_q[:, :, :n], self.v_s[:, :, :n], vb[:, :, :n])
            else:
                torch.mul(self.k_q[:, :, :n], self.k_s[:, :, :n], out=kb[:, :, :n])
                torch.mul(self.v_q[:, :, :n], self.v_s[:, :, :n], out=vb[:, :, :n])
        elif t == 1:
            self.keys_buf.index_copy_(2, self.idx, key_states)
            self.values_buf.index_copy_(2, self.idx, value_states)
        else:
            self.keys_buf[:, :, a:b].copy_(key_states)
            self.values_buf[:, :, a:b].copy_(value_states)
        self.cur = b
        self._set_views()
        return self.keys, self.values


def make_graph_cache(model, max_len, device="cuda:0", kv_int8=False, pass2_bits=8):
    """Mirror of sn_loop_common.make_fixed_cache, with graph-friendly slots."""
    from sn_loop_common import _plan_pairs

    tm = model.model.language_model
    tc = model.config.text_config
    cache = DynamicCache(config=tm._loop_cache_config)
    cache._loop_execution_plan = tm._loop_plan
    idx = torch.zeros(1, dtype=torch.long, device=device)
    shape = (1, tc.num_key_value_heads, max_len, tc.head_dim)
    scratch = (torch.zeros(shape, dtype=torch.bfloat16, device=device),
               torch.zeros(shape, dtype=torch.bfloat16, device=device)) if kv_int8 else None
    second = {e2 for _pi, e1, e2 in _plan_pairs(model)}
    n_full = 0
    for i, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:                 # full attention: replace
            bits = pass2_bits if (pass2_bits != 8 and i in second) else 8
            cache.layers[i] = GraphKVLayer(max_len, tc.num_key_value_heads, tc.head_dim,
                                           torch.bfloat16, device, idx, scratch=scratch, bits=bits)
            n_full += 1
    assert n_full == 14, n_full
    cache.idx = idx
    cache._full = [l for l in cache.layers if isinstance(l, GraphKVLayer)]
    return cache


# ------------------------------------------------------------------------------------------------
# attention with an explicit mask (the stock fold-attention drops it for single-token decode)
# ------------------------------------------------------------------------------------------------
def masked_fold_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                          is_causal=None, **kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    B, Hq, Tq, D = query.shape
    Hkv, L = key.shape[1], key.shape[2]
    g = Hq // Hkv
    q = query.reshape(B, Hkv, g * Tq, D)
    if attention_mask is not None:
        mask = attention_mask                            # [1,1,1,pad] bool, True = visible
        if mask.shape[-2] == Tq and g > 1:               # prefill: repeat per folded query head
            mask = mask.unsqueeze(-3).expand(*mask.shape[:-2], g, Tq, mask.shape[-1])
            mask = mask.reshape(mask.shape[0], 1, g * Tq, mask.shape[-1])
    elif Tq == 1:
        mask = None
    else:
        qpos = torch.arange(L - Tq, L, device=query.device)
        allowed = torch.arange(L, device=query.device)[None, :] <= qpos[:, None]
        mask = allowed.repeat(g, 1)[None, None]
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(q, key, value, attn_mask=mask,
                                                               scale=scaling)
    return out.reshape(B, Hq, Tq, D).transpose(1, 2).contiguous(), None


def use_masked_attention(enable=True):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import transformers.integrations.sdpa_attention as sa

    ALL_ATTENTION_FUNCTIONS["sdpa"] = masked_fold_attention if enable else sa.sdpa_attention_forward


# ------------------------------------------------------------------------------------------------
# decode with graphs
# ------------------------------------------------------------------------------------------------
class GraphDecoder:
    def __init__(self, model, max_len, device="cuda:0", kv_int8=False, buckets=BUCKETS):
        self.model = model
        self.tm = model.model.language_model
        self.emb = self.tm.embed_tokens
        self.hidden = model.config.text_config.hidden_size
        self.cache = make_graph_cache(model, max_len, device, kv_int8)
        self.buckets = sorted(buckets)
        self.device = device
        self.graphs = {}
        self.bufs = {}
        self.capture_s = {}

    def _buffers(self, bucket):
        if bucket not in self.bufs:
            self.bufs[bucket] = dict(
                emb=torch.zeros(1, 1, self.hidden, dtype=torch.bfloat16, device=self.device),
                pos=torch.zeros(1, 1, dtype=torch.long, device=self.device),
                mask=torch.ones(1, 1, 1, bucket, dtype=torch.bool, device=self.device),
                out=torch.zeros(1, 1, self.hidden, dtype=torch.bfloat16, device=self.device),
            )
        return self.bufs[bucket]

    def bucket_for(self, length):
        for b in self.buckets:
            if b >= length:
                return b
        raise ContextFull(f"需要 {length} 个位置，超过了最大的桶 {self.buckets[-1]}")

    def _snapshot_linear(self):
        snap = []
        for layer in self.cache.layers:
            st = {}
            for attr in ("conv_states", "recurrent_states"):
                d = getattr(layer, attr, None)
                if isinstance(d, dict):
                    st[attr] = {k: v.clone() for k, v in d.items() if torch.is_tensor(v)}
            if st:
                snap.append((layer, st))
        return snap

    def _restore_linear(self, snap):
        for layer, st in snap:
            for attr, d in st.items():
                tgt = getattr(layer, attr)
                for k, v in d.items():
                    tgt[k].copy_(v)

    def capture(self, bucket):
        import time

        t0 = time.perf_counter()
        buf = self._buffers(bucket)
        for layer in self.cache._full:
            layer.set_pad(bucket)
        snap = self._snapshot_linear()
        curs = [layer.cur for layer in self.cache._full]
        self.cache.idx.fill_(curs[0])
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            out = self.model.model(inputs_embeds=buf["emb"], position_ids=buf["pos"],
                                   attention_mask={"full_attention": buf["mask"],
                                                   "linear_attention": None},
                                   past_key_values=self.cache, use_cache=True)
            buf["out"].copy_(out.last_hidden_state)
        for layer, c in zip(self.cache._full, curs):
            layer.cur = c
        self._restore_linear(snap)
        for layer in self.cache._full:
            layer._set_views()
        self.graphs[bucket] = g
        self.capture_s[bucket] = time.perf_counter() - t0
        torch.cuda.synchronize()
        return self.capture_s[bucket]

    def prepare(self, token_ids, cur):
        """Outside the graph: embedding read, position, mask, write index."""
        bucket = self.bucket_for(cur + 1)
        if bucket not in self.graphs:
            self.capture(bucket)
        buf = self._buffers(bucket)
        buf["emb"].copy_(self.emb(token_ids))
        buf["pos"].fill_(cur)
        buf["mask"].fill_(False)
        buf["mask"][..., :cur + 1] = True
        self.cache.idx.fill_(cur)
        return bucket

    def step(self, token_ids, cur, logits_head=True):
        bucket = self.prepare(token_ids, cur)
        self.graphs[bucket].replay()
        h = self.bufs[bucket]["out"]
        return self.model.lm_head(h) if logits_head else h

    def advance(self, n=1):
        for layer in self.cache._full:
            layer.cur += n
            layer._set_views()

    def reset(self):
        self.cache.reset()
