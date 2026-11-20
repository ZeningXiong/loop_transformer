"""Two-token CUDA-graph verification for MTP self-speculative decoding.

The single-token graph in cudagraph_decode.py fixes the query length at 1. A speculative verify step
needs q_len = 2 in one forward (that is the whole point: the second position must attend to the first).
This module generalises the same three tricks to q_len 2:

  * KV write  -> index_copy_ with a 2-entry index buffer filled outside the graph
  * KV read   -> the same fixed `pad` view, with a [1,1,2,pad] mask
  * inputs    -> emb / position / mask written into static buffers before replay

Everything else (buckets, capture, int8 KV) is inherited.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cudagraph_decode import GraphKVLayer, make_graph_cache  # noqa: E402


class GraphKV2Layer(GraphKVLayer):
    """Same slot, but a verify step writes qlen entries at once."""

    def __init__(self, max_len, n_heads, head_dim, dtype, device, idx, scratch=None, qlen=2, bits=8):
        super().__init__(max_len, n_heads, head_dim, dtype, device, idx, scratch=scratch, bits=bits)
        self.qlen = qlen
        self.idx2 = torch.zeros(qlen, dtype=torch.long, device=device)

    def update(self, key_states, value_states, *args, **kwargs):
        t = key_states.shape[-2]
        if t != self.qlen:
            return super().update(key_states, value_states, *args, **kwargs)
        a, b = self.cur, self.cur + t
        if b > self.max_len:
            raise RuntimeError(f"上下文已满：{self.cur} + {t} > {self.max_len}")
        ix = self.idx2
        if self.int8:
            q4 = self.bits == 4
            kq, ks = (self._quant4 if q4 else self._quant)(key_states)
            vq, vs = (self._quant4 if q4 else self._quant)(value_states)
            self.k_q.index_copy_(2, ix, kq)
            self.k_s.index_copy_(2, ix, ks)
            self.v_q.index_copy_(2, ix, vq)
            self.v_s.index_copy_(2, ix, vs)
            n = self.pad if self.pad is not None else b
            kb, vb = self.scratch
            if q4:
                from int4_dequant import dequant4
                dequant4(self.k_q[:, :, :n], self.k_s[:, :, :n], kb[:, :, :n])
                dequant4(self.v_q[:, :, :n], self.v_s[:, :, :n], vb[:, :, :n])
            else:
                torch.mul(self.k_q[:, :, :n], self.k_s[:, :, :n], out=kb[:, :, :n])
                torch.mul(self.v_q[:, :, :n], self.v_s[:, :, :n], out=vb[:, :, :n])
        else:
            self.keys_buf.index_copy_(2, ix, key_states)
            self.values_buf.index_copy_(2, ix, value_states)
        self.cur = b
        self._set_views()
        return self.keys, self.values


def make_graph_cache2(model, max_len, device="cuda:0", kv_int8=False, qlen=2, pass2_bits=8):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    tm = model.model.language_model
    tc = model.config.text_config
    cache = DynamicCache(config=tm._loop_cache_config)
    cache._loop_execution_plan = tm._loop_plan
    idx = torch.zeros(1, dtype=torch.long, device=device)
    shape = (1, tc.num_key_value_heads, max_len, tc.head_dim)
    scratch = (torch.zeros(shape, dtype=torch.bfloat16, device=device),
               torch.zeros(shape, dtype=torch.bfloat16, device=device)) if kv_int8 else None
    from sn_loop_common import _plan_pairs
    second = {e2 for _pi, e1, e2 in _plan_pairs(model)}
    n_full = 0
    for i, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:
            bits = pass2_bits if (pass2_bits != 8 and i in second) else 8
            cache.layers[i] = GraphKV2Layer(max_len, tc.num_key_value_heads, tc.head_dim,
                                            torch.bfloat16, device, idx, scratch=scratch, qlen=qlen, bits=bits)
            n_full += 1
    assert n_full == 14, n_full
    cache.idx = idx
    cache._full = [l for l in cache.layers if isinstance(l, GraphKV2Layer)]
    return cache


class Graph2Decoder:
    """Captures one graph per bucket for a fixed query length `qlen`."""

    def __init__(self, model, max_len, device="cuda:0", kv_int8=False, qlen=2, pass2_bits=8,
                 buckets=(256, 512, 1024,
                                                                                       2048, 4096, 8192,
                                                                                       16384, 32768)):
        self.model = model
        self.tm = model.model.language_model
        self.emb = self.tm.embed_tokens
        self.hidden = model.config.text_config.hidden_size
        self.qlen = qlen
        self.device = device
        self.cache = make_graph_cache2(model, max_len, device, kv_int8, qlen, pass2_bits)
        self.max_len = max_len
        # 桶不能超过 KV 缓冲区长度；把 max_len 本身也当一个桶，避免为 16K 分配 32K
        self.buckets = sorted({b for b in buckets if b < max_len} | {max_len})
        self.graphs, self.bufs, self.capture_s = {}, {}, {}

    def _buffers(self, bucket):
        if bucket not in self.bufs:
            self.bufs[bucket] = dict(
                emb=torch.zeros(1, self.qlen, self.hidden, dtype=torch.bfloat16, device=self.device),
                pos=torch.zeros(1, self.qlen, dtype=torch.long, device=self.device),
                mask=torch.ones(1, 1, self.qlen, bucket, dtype=torch.bool, device=self.device),
                out=torch.zeros(1, self.qlen, self.hidden, dtype=torch.bfloat16, device=self.device),
            )
        return self.bufs[bucket]

    def bucket_for(self, length):
        for b in self.buckets:
            if b >= length:
                return b
        raise RuntimeError(f"需要 {length} 个位置，超过最大桶 {self.buckets[-1]}")

    def _snap(self):
        import sn_spec
        return sn_spec._Snapshot(self.cache)

    def capture(self, bucket):
        import time

        t0 = time.perf_counter()
        buf = self._buffers(bucket)
        for layer in self.cache._full:
            layer.set_pad(bucket)
        snap = self._snap()
        snap.save()
        curs = snap.curs

        def arm() -> None:
            self.cache.idx.fill_(curs[0])
            for l in self.cache._full:
                l.idx2.copy_(torch.arange(curs[0], curs[0] + self.qlen, device=self.device))

        # 预热一次：让 torch.compile 的区域按这个形状编译完，否则重编译会发生在捕获过程中
        with torch.inference_mode():
            arm()
            self.model.model(inputs_embeds=buf["emb"], position_ids=buf["pos"],
                             attention_mask={"full_attention": buf["mask"],
                                             "linear_attention": None},
                             past_key_values=self.cache, use_cache=True)
            torch.cuda.synchronize()
        snap.restore()
        arm()
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            out = self.model.model(inputs_embeds=buf["emb"], position_ids=buf["pos"],
                                   attention_mask={"full_attention": buf["mask"],
                                                   "linear_attention": None},
                                   past_key_values=self.cache, use_cache=True)
            buf["out"].copy_(out.last_hidden_state)
        for layer, c in zip(self.cache._full, curs):
            layer.cur = c
        snap.restore()
        for layer in self.cache._full:
            layer._set_views()
        self.graphs[bucket] = g
        self.capture_s[bucket] = time.perf_counter() - t0
        torch.cuda.synchronize()
        return self.capture_s[bucket]

    def step(self, tokens, cur):
        """tokens: [1, qlen]. Returns logits [1, qlen, V]."""
        bucket = self.bucket_for(cur + self.qlen)
        if bucket not in self.graphs:
            self.capture(bucket)
        buf = self._buffers(bucket)
        buf["emb"].copy_(self.emb(tokens))
        buf["pos"].copy_(torch.arange(cur, cur + self.qlen, device=self.device)[None])
        buf["mask"].fill_(False)
        for i in range(self.qlen):
            buf["mask"][0, 0, i, :cur + i + 1] = True
        self.cache.idx.fill_(cur)
        for l in self.cache._full:
            l.idx2.copy_(torch.arange(cur, cur + self.qlen, device=self.device))
        self.graphs[bucket].replay()
        return self.model.lm_head(self.bufs[bucket]["out"]).float()

    def advance(self, n):
        for layer in self.cache._full:
            layer.cur += n
            layer._set_views()

    def reset(self):
        self.cache.reset()
