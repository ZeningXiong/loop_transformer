"""Short-sequence specialisation for the linear-attention (Gated DeltaNet) layers.

transformers picks the algorithm by sequence length:
    seq_len == 1  ->  fused_recurrent_gated_delta_rule   (one state update per token)
    seq_len  > 1  ->  chunk_gated_delta_rule             (64-wide chunked matmuls)

A speculative verify step is seq_len 2-4, and the chunked path costs almost the same as a long chunk,
so verification pays a large fixed cost for two extra tokens. For a *cached* decode the chunked
algorithm is also unnecessary: the recurrent kernel already accepts an initial state and walks the
sequence, which is exactly what 2-4 tokens need.

This module rebinds the module-level name the layer looks up at call time, so no site-packages edit is
needed. Only cached (initial_state is not None) forwards of length <= MAX are rerouted.
"""
MAX_SEQ = 4


def use_recurrent_for_short(enable=True, max_seq=MAX_SEQ):
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    if not hasattr(hf, "_orig_chunk_gdn"):
        hf._orig_chunk_gdn = hf.torch_chunk_gated_delta_rule
    if not enable:
        hf.torch_chunk_gated_delta_rule = hf._orig_chunk_gdn
        return

    def dispatch(query, key, value, g, beta, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kw):
        if initial_state is not None and query.shape[1] <= max_seq:
            return hf.torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens, **kw)
        return hf._orig_chunk_gdn(query, key, value, g=g, beta=beta, initial_state=initial_state,
                                  output_final_state=output_final_state,
                                  use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                                  cu_seqlens=cu_seqlens, **kw)

    hf.torch_chunk_gated_delta_rule = dispatch
