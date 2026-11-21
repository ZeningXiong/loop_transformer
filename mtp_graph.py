"""把 MTP 的草稿与回填各捕获成一张 CUDA graph（用于图 + 投机 + 量化 KV 的组合）。

要点：
- MTP 的 KV 缓存换成支持固定 pad 的 GraphKVLayer；预填用 pad=None，解码切回 pad=max_len；
- embedding 读盘留在图外（每步一次 H2D 拷贝）；
- **所有每步变化的量（位置、掩码、写指针）都在图外填好**，图里只读不写常量；
- 两张图：草稿（含 norm + lm_head + argmax）与回填（只写 KV，不算 lm_head）。
"""
import torch

from cudagraph_decode import GraphKVLayer, use_masked_attention


class MTPGraph:
    def __init__(self, mtp, max_len, device="cuda"):
        use_masked_attention(True)
        self.mtp = mtp
        self.max_len = max_len
        self.device = device
        tc = mtp.tc
        self.D = tc.hidden_size
        self.idx = torch.zeros(1, dtype=torch.long, device=device)
        layer = GraphKVLayer(max_len, tc.num_key_value_heads, tc.head_dim, torch.bfloat16,
                             device, self.idx)
        mtp.cache.layer = layer
        self.layer = layer
        self.emb_buf = torch.zeros(1, 1, self.D, dtype=torch.bfloat16, device=device)
        self.hid_buf = torch.zeros(1, 1, self.D, dtype=torch.bfloat16, device=device)
        self.pos_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.mask_buf = torch.zeros(1, 1, 1, max_len, dtype=torch.bool, device=device)
        self.out_buf = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.g_draft = None
        self.g_feed = None

    # -- 预填 / 解码切换 ---------------------------------------------------------------------
    def open_prefill(self):
        self.layer.set_pad(None)

    def _arm(self, pos):
        """图外准备：pad、写指针、位置、掩码。"""
        self.layer.set_pad(self.max_len)
        self.layer.cur = pos
        self.idx.fill_(pos)
        self.pos_buf.fill_(pos)
        self.mask_buf.zero_()
        self.mask_buf[0, 0, 0, :pos + 1] = True

    def _body(self, logits):
        m = self.mtp
        h = m.pre_fc_norm_hidden(self.hid_buf)
        z = m.fc(torch.cat([self.emb_buf, h] if m.embed_first else [h, self.emb_buf], dim=-1))
        pe = m.tm.rotary_emb(z, self.pos_buf[None].expand(3, -1, -1))
        z = m.layer(z, position_embeddings=pe, attention_mask=self.mask_buf,
                    position_ids=self.pos_buf, past_key_values=m.cache, use_cache=True)
        if logits:
            self.out_buf.copy_(m.lm_head(m.norm(z))[0, -1].argmax().view(1, 1))

    @torch.inference_mode()
    def capture(self, pos):
        self._arm(pos)
        self._body(True)                      # 预热，确保内核与形状定型
        torch.cuda.synchronize()
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            self._body(True)
        self.g_draft = g1
        self._arm(pos)
        self._body(True)
        torch.cuda.synchronize()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            self._body(False)
        self.g_feed = g2

    @torch.inference_mode()
    def draft(self, hidden, token, pos):
        self._arm(pos)
        self.emb_buf.copy_(self.mtp.tm.embed_tokens(token))
        self.hid_buf.copy_(hidden)
        self.g_draft.replay()
        self.layer.cur = pos + 1
        return self.out_buf

    @torch.inference_mode()
    def feed(self, hidden, token, pos):
        self._arm(pos)
        self.emb_buf.copy_(self.mtp.tm.embed_tokens(token))
        self.hid_buf.copy_(hidden)
        self.g_feed.replay()
        self.layer.cur = pos + 1
