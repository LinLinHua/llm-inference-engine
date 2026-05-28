"""
Step 2: Patches HuggingFace attention layers with custom kernels.
Follows official LlamaAttention.forward signature exactly.
Reference: transformers/src/transformers/models/llama/modeling_llama.py
"""

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from src.engine.base_engine import BaseEngine


class ModelRunner(BaseEngine):

    def __init__(
        self,
        model_name:       str,
        device:           str = "cuda",
        dtype:            torch.dtype = torch.float16,
        use_flash_attn:   bool = True,
        use_flash_decode: bool = True,
    ):
        super().__init__(model_name=model_name, device=device, dtype=dtype)

        self.use_flash_attn   = use_flash_attn
        self.use_flash_decode = use_flash_decode
        self._cuda_ext        = None

        if use_flash_attn:
            self._load_cuda_kernel()

        if use_flash_attn or use_flash_decode:
            self._patch_attention_layers()

    def _load_cuda_kernel(self):
        try:
            import torch.utils.cpp_extension as cpp_ext
            import os
            src = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "src", "kernels", "cuda", "flash_attn_prefill.cu"
            )
            self._cuda_ext = cpp_ext.load(
                name="flash_attn_prefill",
                sources=[os.path.join(base, "flash_attn_prefill_bind.cpp"),os.path.join(base, "flash_attn_prefill.cu"),],
                extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_80"],
                verbose=False,
            )
            print("  [FA-2 kernel] Loaded ✓")
        except Exception as e:
            print(f"  [FA-2 kernel] Failed: {e}")
            print(f"  → Falling back to PyTorch SDPA")

    def _patch_attention_layers(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            attn._use_flash_attn   = self.use_flash_attn
            attn._use_flash_decode = self.use_flash_decode
            attn._cuda_ext         = self._cuda_ext
            attn._layer_id         = layer_idx
            attn._paged_kv_cache   = None
            attn._request_id       = 0
            attn._Hq               = self.num_heads
            attn._Hkv              = self.num_kv_heads
            attn.forward           = _patched_forward.__get__(attn, type(attn))
        print(f"  [{len(self.model.model.layers)} layers patched] ✓")

    def set_paged_kv_cache(self, paged_kv, request_id: int = 0):
        for layer in self.model.model.layers:
            layer.self_attn._paged_kv_cache = paged_kv
            layer.self_attn._request_id     = request_id


# ── Helper functions ──────────────────────────────────────────────────────────

def _patched_forward(
    self,
    hidden_states,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # ── Step 1: QKV projection ───────────────────────────────────────────────
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # ── Step 2: RoPE (matches official apply_rotary_pos_emb) ─────────────────
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ── Step 3: KV cache (matches official DynamicCache.update) ──────────────
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self._layer_id, cache_kwargs
        )

    # ── Step 3b: Write into PagedKVCache ─────────────────────────────────────
    paged_kv   = getattr(self, "_paged_kv_cache", None)
    request_id = getattr(self, "_request_id", 0)
    seq_pos    = key_states.shape[2] - 1

    if paged_kv is not None:
        paged_kv.write_kv(
            request_id=request_id,
            layer_id=self._layer_id,
            token_pos=seq_pos,
            k=key_states[0, :, seq_pos, :],
            v=value_states[0, :, seq_pos, :],
        )

    # ── Step 4: Choose kernel ─────────────────────────────────────────────────
    g         = self._Hq // self._Hkv
    seq_q     = query_states.shape[2]
    is_decode = (seq_q == 1)
    cuda_ext  = getattr(self, "_cuda_ext", None)

    if is_decode and getattr(self, "_use_flash_decode", False):
        attn_output = _flash_decode(query_states, key_states, value_states)
    elif not is_decode and cuda_ext is not None and getattr(self, "_use_flash_attn", False):
        attn_output = cuda_ext.fa2_prefill(query_states, key_states, value_states, True)
    else:
        K_exp = key_states.repeat_interleave(g, dim=1)
        V_exp = value_states.repeat_interleave(g, dim=1)
        attn_output = F.scaled_dot_product_attention(
            query_states, K_exp, V_exp, is_causal=(seq_q > 1)
        )

    # ── Step 5: Output projection (matches official reshape) ─────────────────
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None  # (attn_output, attn_weights)


def _flash_decode(Q, K, V):
    """Triton FlashDecoding kernel, fallback to SDPA."""
    try:
        from src.kernels.triton.flash_decode import flash_decode
        return flash_decode(Q, K, V, causal=True)
    except Exception:
        g = Q.shape[1] // K.shape[1]
        return F.scaled_dot_product_attention(
            Q, K.repeat_interleave(g, 1), V.repeat_interleave(g, 1),
            is_causal=False
        )