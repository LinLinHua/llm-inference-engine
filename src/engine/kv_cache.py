"""
Step 3: Paged KV Cache
"""

import torch
from typing import Dict, List, Tuple, Optional


class PagedKVCache:
    def __init__(
        self,
        num_blocks:   int,
        block_size:   int,
        num_kv_heads: int,
        num_layers:   int,
        head_dim:     int,
        device:       str = "cuda",
        dtype:        torch.dtype = torch.float16,
    ):
        self.num_blocks   = num_blocks
        self.block_size   = block_size
        self.num_kv_heads = num_kv_heads
        self.num_layers   = num_layers
        self.head_dim     = head_dim
        self.device       = device
        self.dtype        = dtype

        self.key_cache = [
            torch.zeros(num_blocks, num_kv_heads, head_dim, block_size,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.zeros(num_blocks, num_kv_heads, head_dim, block_size,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        self.free_blocks:  List[int]            = list(range(num_blocks))
        self.block_tables: Dict[int, List[int]] = {}
        self.seq_lens:     Dict[int, int]        = {}

        mem_gb = (2 * num_blocks * num_kv_heads * head_dim
                  * block_size * num_layers * 2) / (1024 ** 3)
        print(f"  [PagedKVCache] {num_blocks} blocks × "
              f"{block_size} tokens × {num_layers} layers = {mem_gb:.2f} GB")

    def allocate(self, request_id: int, seq_len: int = 0):
        """Register a new request and pre-allocate blocks if needed."""
        assert request_id not in self.block_tables, \
            f"Request {request_id} already exists"
        self.block_tables[request_id] = []
        self.seq_lens[request_id]     = 0
        if seq_len > 0:
            self._ensure_blocks(request_id, seq_len)

    def free(self, request_id: int):
        """Return all blocks used by this request back to the pool."""
        if request_id in self.block_tables:
            self.free_blocks.extend(self.block_tables.pop(request_id))
            self.seq_lens.pop(request_id, None)

    def _ensure_blocks(self, request_id: int, seq_len: int):
        """Allocate new physical blocks if needed."""
        needed  = (seq_len + self.block_size - 1) // self.block_size
        current = len(self.block_tables[request_id])
        for _ in range(needed - current):
            if not self.free_blocks:
                raise RuntimeError("KV cache full! Increase num_blocks.")
            self.block_tables[request_id].append(self.free_blocks.pop())

    def write_kv(
        self,
        request_id: int,
        layer_id:   int,
        token_pos:  int,
        k:          torch.Tensor,
        v:          torch.Tensor,
    ) -> None:
        """Write one token's K/V into paged cache."""
        self._ensure_blocks(request_id, token_pos + 1)
        block_idx = token_pos // self.block_size
        block_off = token_pos %  self.block_size
        phys      = self.block_tables[request_id][block_idx]
        self.key_cache  [layer_id][phys, :, :, block_off] = k
        self.value_cache[layer_id][phys, :, :, block_off] = v

    def read_kv(
        self,
        request_id: int,
        layer_id:   int,
        seq_len:    int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for one request. Returns [1, Hkv, seq_len, D]."""
        Hkv = self.num_kv_heads
        D   = self.head_dim
        bs  = self.block_size
        K_out = torch.empty(1, Hkv, seq_len, D,
                            device=self.device, dtype=self.dtype)
        V_out = torch.empty_like(K_out)
        blocks = self.block_tables[request_id]
        for t in range(seq_len):
            phys = blocks[t // bs]
            K_out[0, :, t, :] = self.key_cache  [layer_id][phys, :, :, t % bs]
            V_out[0, :, t, :] = self.value_cache[layer_id][phys, :, :, t % bs]
        return K_out, V_out

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def utilization(self) -> float:
        return 1.0 - len(self.free_blocks) / self.num_blocks