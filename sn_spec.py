"""Self-speculative decoding with the model's own MTP (multi-token prediction) head.

The checkpoint ships one MTP block (mtp.safetensors, ~0.24B params bf16): it takes the main model's hidden
state at position i and the embedding of token i+1, and predicts token i+2. Transformers ignores these
weights, so the head is rebuilt here from the Qwen3.5 building blocks.

Greedy speculation with one draft token per step:
  1. draft d = MTP(h_last, x_next)
  2. verify: one main-model forward on [x_next, d]
  3. accept -> 2 tokens for 1 forward; reject -> restore the linear-attention states / KV write position,
     redo the forward on [x_next, y] with the correct y -> 2 tokens for 2 forwards.
Every committed token is the main model's own greedy choice, so the text matches plain greedy decoding
up to floating-point differences between the 1-token and 2-token compute paths.
"""
import os
import time

import torch
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

from sn_loop_common import _fixed_kv_layer_class, reset_cache

_MODEL_DIR = os.environ.get("SENSENOVA_MODEL") or r"D:\models\SenseNova-Loop-9B-Preview"
MTP_FILE = os.environ.get("SENSENOVA_MTP") or os.path.join(_MODEL_DIR, "mtp.safetensors")


class _MiniCache:
    """Cache object for the single MTP attention layer: one fixed-size bf16 KV buffer."""

    def __init__(self, max_len, n_heads, head_dim, device):
        self.layer = _fixed_kv_layer_class()(max_len, n_heads, head_dim, torch.bfloat16, device)

    def update(self, k, v, layer_idx, *args, **kwargs):
        return self.layer.update(k, v)

    def get_seq_length(self, *args, **kwargs):
        return self.layer.cur

    def reset(self):
        self.layer.reset()


class MTPHead(torch.nn.Module):
    def __init__(self, model, device="cuda:0", hidden_source="post_norm", embed_first=True):
        super().__init__()
        tc = model.config.text_config
        self.tc = tc
        # plain references to the shared main-model parts, kept out of this module's parameters
        self.__dict__["tm"] = model.model.language_model
        self.__dict__["lm_head"] = model.lm_head
        self.hidden_source, self.embed_first = hidden_source, embed_first
        full_idx = tc.layer_types.index("full_attention")
        with torch.device(device):
            self.fc = torch.nn.Linear(2 * tc.hidden_size, tc.hidden_size, bias=False, dtype=torch.bfloat16)
            self.layer = hf.Qwen3_5DecoderLayer(tc, full_idx).to(torch.bfloat16)
            self.norm = hf.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(torch.bfloat16)
            self.pre_fc_norm_hidden = hf.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(torch.bfloat16)
            self.pre_fc_norm_embedding = hf.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(torch.bfloat16)
        sd = {}
        with safe_open(MTP_FILE, framework="pt", device=device) as f:
            for k in f.keys():
                sd[k[len("mtp."):].replace("layers.0.", "layer.")] = f.get_tensor(k)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "rotary" not in m]
        assert not missing and not unexpected, (missing, unexpected)
        self.eval()
        self.cache = None

    def new_cache(self, max_len, device="cuda:0"):
        self.cache = _MiniCache(max_len, self.tc.num_key_value_heads, self.tc.head_dim, device)

    @torch.inference_mode()
    def forward(self, hidden, token_ids, positions, logits=True, all_rows=False):
        """hidden: [1, T, D] main-model hidden at positions-1; token_ids: [1, T] tokens at `positions`.
        Returns logits [1, T, V] predicting the token after each position (None if logits=False: only the
        MTP's own KV cache is updated)."""
        e = self.pre_fc_norm_embedding(self.tm.embed_tokens(token_ids))
        h = self.pre_fc_norm_hidden(hidden)
        x = self.fc(torch.cat([e, h] if self.embed_first else [h, e], dim=-1))
        pos3 = positions[None].expand(3, -1, -1)
        pe = self.tm.rotary_emb(x, pos3)
        x = self.layer(x, position_embeddings=pe, attention_mask=None, position_ids=positions,
                       past_key_values=self.cache, use_cache=True)
        if not logits:
            return None
        return self.lm_head(self.norm(x if all_rows else x[:, -1:]))


class _HiddenTap:
    """Captures the main model's final hidden state before and after its final norm."""

    def __init__(self, model):
        self.pre = self.post = None
        model.model.language_model.norm.register_forward_hook(self._hook)

    def _hook(self, _m, inp, out):
        self.pre, self.post = inp[0], out


def get_tap(model):
    if not hasattr(model, "_hidden_tap"):
        model._hidden_tap = _HiddenTap(model)
    return model._hidden_tap


def small_batch_gemv():
    """bitsandbytes uses its fast 4-bit GEMV only for a single row; 2-4 rows fall back to dequantizing the
    whole weight. Run small batches row by row through the fast path (same math as 1-token decode)."""
    import bitsandbytes as bnb

    cls = bnb.nn.Linear4bit
    if getattr(cls, "_small_batch_patched", False):
        return
    orig = cls.forward

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        if 1 < rows <= 4:
            xs = x.reshape(rows, 1, 1, x.shape[-1])
            return torch.cat([orig(self, xs[i]) for i in range(rows)], dim=1).reshape(*x.shape[:-1], -1)
        return orig(self, x)

    cls.forward = forward
    cls._small_batch_patched = True


def _linear_states(cache):
    out = []
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):
            out.extend(t for t in layer.conv_states.values() if t is not None) if isinstance(layer.conv_states, dict) \
                else out.extend(t for t in layer.conv_states if t is not None)
            out.extend(t for t in layer.recurrent_states.values() if t is not None) if isinstance(layer.recurrent_states, dict) \
                else out.extend(t for t in layer.recurrent_states if t is not None)
    return out


class _Snapshot:
    """Save/restore everything a 2-token verify step changes: linear-attention states (updated in place)
    and the write position of every fixed KV buffer."""

    def __init__(self, cache):
        self.cache = cache
        self.states = _linear_states(cache)
        self.saved = [torch.empty_like(t) for t in self.states]
        self.kv = [l for l in cache.layers if hasattr(l, "cur")]

    def save(self):
        for s, t in zip(self.saved, self.states):
            s.copy_(t)
        self.curs = [l.cur for l in self.kv]

    def restore(self):
        for s, t in zip(self.saved, self.states):
            t.copy_(s)
        for l, c in zip(self.kv, self.curs):
            l.cur = c
            l._set_views()


@torch.inference_mode()
def generate_speculative(model, mtp, input_ids, max_new_tokens, cache, chunk_size=512, eos_token_ids=None,
                         adaptive=True, threshold=0.6, plain_block=16):
    """Greedy self-speculative generation. Returns (tokens, stats).

    adaptive: with the measured costs (2-token verify ~1.38x a 1-token step, a rejection needs a redo),
    speculation only pays off above ~63% acceptance. When the running acceptance drops below `threshold`,
    decode `plain_block` tokens normally, then probe speculation again."""
    dev = input_ids.device
    tap = get_tap(model)
    T = input_ids.shape[1]
    if eos_token_ids is None:
        eos = model.generation_config.eos_token_id
        eos_token_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    hid = (lambda: tap.pre) if mtp.hidden_source == "pre_norm" else (lambda: tap.post)
    reset_cache(cache)
    mtp.cache.reset()

    def main(tokens, start):
        pos = torch.arange(start, start + tokens.shape[1], device=dev)[None]
        h_post = model.model(input_ids=tokens, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
        return hid(), model.lm_head(h_post).float()  # logits only for these few rows

    # prefill: main model chunk by chunk; MTP consumes (h_i, x_{i+1}) pairs for every prompt position
    t0 = time.perf_counter()
    prev_h = None
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        pos = torch.arange(s, e, device=dev)[None]
        model.model(input_ids=input_ids[:, s:e], position_ids=pos, past_key_values=cache, use_cache=True)
        h = hid()
        if prev_h is not None:  # pair the previous chunk's last hidden with this chunk's first token
            mtp(prev_h, input_ids[:, s:s + 1], pos[:, :1], logits=False)
        if e - s > 1:
            mtp(h[:, :-1], input_ids[:, s + 1:e], pos[:, 1:], logits=False)
        prev_h = h[:, -1:]
    last_logits = model.lm_head(tap.post[:, -1:]).float()
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    snap = _Snapshot(cache)
    x = last_logits[0, -1].argmax().view(1, 1)  # next token (position T), already decided by the main model
    h_last, pos = prev_h, T
    out = []
    stats = {"steps": 0, "accepted": 0, "redo": 0, "plain_tokens": 0}
    ema, plain_left = 0.8, 0       # running acceptance rate; tokens left in plain-decoding mode
    t1 = time.perf_counter()
    while len(out) < max_new_tokens:
        if adaptive and plain_left > 0:
            # plain 1-token step (speculation was not paying off); keep the MTP cache in step anyway
            mtp(h_last, x, torch.tensor([[pos]], device=dev), logits=False)
            H, L = main(x, pos)
            out.append(x.item())
            stats["plain_tokens"] += 1
            if x.item() in eos_token_ids:
                break
            x, h_last, pos = L[0, -1].argmax().view(1, 1), H[:, -1:], pos + 1
            plain_left -= 1
            if plain_left == 0:
                ema = threshold + 0.05  # probe speculation again
            continue
        # 1. draft the token after x
        d = mtp(h_last, x, torch.tensor([[pos]], device=dev))[0, -1].argmax().view(1, 1)
        # 2. verify [x, d] in one main-model forward
        snap.save()
        H, L = main(torch.cat([x, d], 1), pos)
        y1, y2 = L[0, 0].argmax().view(1, 1), L[0, 1].argmax().view(1, 1)
        stats["steps"] += 1
        ok = y1.item() == d.item()
        ema = 0.8 * ema + 0.2 * ok
        if ok:
            stats["accepted"] += 1
            nxt_tok, H_use, y_next = d, H, y2
        else:
            # 3. rejected: roll back and redo with the main model's own choice y1
            snap.restore()
            stats["redo"] += 1
            H_use, L2 = main(torch.cat([x, y1], 1), pos)
            nxt_tok, y_next = y1, L2[0, 1].argmax().view(1, 1)
        out.extend([x.item(), nxt_tok.item()])
        if x.item() in eos_token_ids or nxt_tok.item() in eos_token_ids:
            break
        # keep the MTP KV cache in step: pair (h at pos, token at pos+1); the draft for the next step
        # then uses (h at pos+1, y_next) at pos+2 and is computed at the top of the loop
        mtp(H_use[:, :1], nxt_tok, torch.tensor([[pos + 1]], device=dev), logits=False)
        x, h_last, pos = y_next, H_use[:, 1:2], pos + 2
        if adaptive and ema < threshold:
            plain_left = plain_block
    torch.cuda.synchronize()
    stats.update(prefill_s=t_prefill, decode_s=time.perf_counter() - t1, tokens=len(out))
    return out[:max_new_tokens], stats
