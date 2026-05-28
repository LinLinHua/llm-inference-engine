"""
src/engine/model_runner.py

Step 2: Patches HuggingFace attention layers with custom kernels.
"""

import torch
import torch.nn.functional as F
from src.engine.base_engine import BaseEngine, GenerationResult


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
                sources=[src],
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
            # Store shape info to avoid HuggingFace version differences
            attn._Hq               = self.num_heads
            attn._Hkv              = self.num_kv_heads
            attn._D                = self.head_dim
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
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = True,
    cache_position=None,
    position_embeddings=None,  # (cos, sin) passed in from HuggingFace >= 4.46
    **kwargs,
):
    B, seq_q, _ = hidden_states.shape
    Hq  = self._Hq
    Hkv = self._Hkv
    D   = self._D
    g   = Hq // Hkv

    # ── Step 1: QKV projection ───────────────────────────────────────────────
    Q = self.q_proj(hidden_states)
    K = self.k_proj(hidden_states)
    V = self.v_proj(hidden_states)

    Q = Q.view(B, seq_q, Hq,  D).transpose(1, 2)  # [B, Hq,  seq_q, D]
    K = K.view(B, seq_q, Hkv, D).transpose(1, 2)  # [B, Hkv, seq_q, D]
    V = V.view(B, seq_q, Hkv, D).transpose(1, 2)  # [B, Hkv, seq_q, D]

    # ── Step 2: RoPE ─────────────────────────────────────────────────────────
    # cos/sin shape: [B, seq_q, D] → unsqueeze(1) → [B, 1, seq_q, D]
    # Q/K shape:     [B, Hq/Hkv, seq_q, D]
    # broadcast works because head dim = 1 broadcasts to Hq/Hkv
    if position_embeddings is not None:
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)  # [B, 1, seq_q, D]
        sin = sin.unsqueeze(1)
        Q = (Q * cos) + (_rotate_half(Q) * sin)
        K = (K * cos) + (_rotate_half(K) * sin)
    elif hasattr(self, "rotary_emb"):
        cos, sin = self.rotary_emb(V, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        Q = (Q * cos) + (_rotate_half(Q) * sin)
        K = (K * cos) + (_rotate_half(K) * sin)

    # ── Step 3: KV cache ─────────────────────────────────────────────────────
    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim=2)
        V = torch.cat([past_key_value[1], V], dim=2)
    new_past_kv = (K, V) if use_cache else None

    # ── Step 3b: Write into PagedKVCache ─────────────────────────────────────
    paged_kv   = getattr(self, "_paged_kv_cache", None)
    request_id = getattr(self, "_request_id", 0)
    layer_id   = getattr(self, "_layer_id", 0)
    seq_pos    = K.shape[2] - 1

    if paged_kv is not None:
        paged_kv.write_kv(
            request_id=request_id,
            layer_id=layer_id,
            token_pos=seq_pos,
            k=K[0, :, seq_pos, :],
            v=V[0, :, seq_pos, :],
        )

    # ── Step 4: Attention ─────────────────────────────────────────────────────
    is_decode = (seq_q == 1)
    cuda_ext  = getattr(self, "_cuda_ext", None)

    if is_decode and getattr(self, "_use_flash_decode", False):
        attn_out = _flash_decode(Q, K, V)
    elif not is_decode and cuda_ext is not None and getattr(self, "_use_flash_attn", False):
        attn_out = cuda_ext.fa2_prefill(Q, K, V, True)
    else:
        K_exp    = K.repeat_interleave(g, dim=1)
        V_exp    = V.repeat_interleave(g, dim=1)
        attn_out = F.scaled_dot_product_attention(
            Q, K_exp, V_exp, is_causal=(seq_q > 1)
        )

    # ── Step 5: Output projection ─────────────────────────────────────────────
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_q, Hq * D)
    return self.o_proj(attn_out), new_past_kv


def _rotate_half(x):
    """Rotates half the hidden dims — matches HuggingFace's rotate_half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _flash_decode(Q, K, V):
    """Call Triton FlashDecoding kernel, fallback to SDPA."""
    try:
        from src.kernels.triton.flash_decode import flash_decode
        return flash_decode(Q, K, V, causal=True)
    except Exception:
        g = Q.shape[1] // K.shape[1]
        return F.scaled_dot_product_attention(
            Q, K.repeat_interleave(g, 1), V.repeat_interleave(g, 1),
            is_causal=False
        )