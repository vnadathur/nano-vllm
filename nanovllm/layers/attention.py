import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# Opaque to Dynamo: eliminates graph breaks from get_context(), Triton store_kvcache,
# and Flash Attention. A single custom op node replaces the whole attention + KV cache block.
@torch.library.custom_op("nanovllm::attention", mutates_args=("k_cache", "v_cache"))
def attention_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    num_heads: int, head_dim: int, scale: float, num_kv_heads: int,
) -> torch.Tensor:
    context = get_context()
    if k_cache.numel() > 0:
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
    if context.is_prefill:
        # Prefix cache: read full K/V from cache instead of just-computed tokens.
        if context.block_tables is not None:
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(q, k, v,
                                   max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                   max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                   softmax_scale=scale, causal=True, block_table=context.block_tables)
    else:
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                                    softmax_scale=scale, causal=True)
        o = o.squeeze(1)  # normalize to [N, H, D] matching prefill shape
    return o


@attention_op.register_fake
def _attention_op_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    num_heads: int, head_dim: int, scale: float, num_kv_heads: int,
) -> torch.Tensor:
    return torch.empty_like(q)


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        return torch.ops.nanovllm.attention(q, k, v, self.k_cache, self.v_cache, self.num_heads, self.head_dim, self.scale, self.num_kv_heads)
