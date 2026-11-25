# coding=utf-8
"""Attention statistics for sink calibration (M1).

Two kinds of statistics are supported:

``exact``
    Materialise the causal prefill attention matrix (from a separate eager /
    SDPA calibration process) and average it over heads and queries.  This is a
    diagnostic tool: memory grows as ``O(heads * T**2)``, so keep ``T`` small.

``proxy``
    A flash-compatible fallback that does not need attention weights.  The
    default proxy is the per-token key norm (averaged over KV heads), which is
    invariant to RoPE because RoPE is an orthogonal rotation.  ``k_proj``
    outputs can therefore be captured with a read-only forward hook in a
    calibration process even when the model runs flash attention.

All functions are pure: inputs are never modified and no global state (RNG,
dtype, backend flags) is touched.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = [
    "attention_mass_from_weights",
    "causal_attention_mass",
    "causal_attention_mass_streaming",
    "causal_mask",
    "key_norm_scores",
    "key_norm_scores_from_projection",
    "repeat_kv",
    "select_query_count",
]


def select_query_count(query_len: int, query_fraction: float = 1.0) -> int:
    """Number of *last* queries used for the token score.

    ``query_fraction=1.0`` uses all prefill queries; ``0.25`` uses the last
    quarter (at least one query).
    """
    if query_len < 1:
        raise ValueError(f"query_len must be >= 1, got {query_len}")
    if not (0.0 < query_fraction <= 1.0):
        raise ValueError(f"query_fraction must be in (0, 1], got {query_fraction}")
    return max(1, math.ceil(query_len * query_fraction))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query heads (HF ``repeat_kv`` semantics)."""
    if n_rep < 1:
        raise ValueError(f"n_rep must be >= 1, got {n_rep}")
    if n_rep == 1:
        return hidden_states
    if hidden_states.dim() != 4:
        raise ValueError(
            f"expected a [batch, heads, seq, head_dim] tensor, got {tuple(hidden_states.shape)}"
        )
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def causal_mask(
    query_len: int,
    key_len: int,
    query_offset: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Boolean ``[query_len, key_len]`` mask; ``True`` means the key is visible.

    ``query_offset`` is the absolute key position of the first query.  It
    defaults to ``key_len - query_len`` (the usual prefill/decode layout).  The
    KIVI cache uses the same convention: cached keys occupy positions
    ``[0, key_len)`` and queries continue at the end.
    """
    if query_len < 1 or key_len < 1:
        raise ValueError(
            f"query_len and key_len must be >= 1, got {query_len} and {key_len}"
        )
    if query_offset is None:
        query_offset = key_len - query_len
    if query_offset < 0 or query_offset + query_len > key_len:
        raise ValueError(
            f"query_offset={query_offset} with query_len={query_len} does not fit "
            f"key_len={key_len}"
        )
    query_positions = torch.arange(
        query_offset, query_offset + query_len, device=device
    )
    key_positions = torch.arange(key_len, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def _scaled_causal_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    query_offset: Optional[int] = None,
) -> torch.Tensor:
    """Float32 causal logits ``[batch, query_heads, query_len, key_len]``."""
    if query.dim() != 4 or key.dim() != 4:
        raise ValueError(
            "query and key must be [batch, heads, seq_len, head_dim] tensors, got "
            f"{tuple(query.shape)} and {tuple(key.shape)}"
        )
    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key must share the batch size")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head_dim must match")
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} is not a multiple of "
            f"num_kv_heads={num_kv_heads}"
        )

    key = repeat_kv(key, num_query_heads // num_kv_heads)
    scale = query.shape[-1] ** -0.5
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale

    mask = causal_mask(
        query.shape[-2], key.shape[-2], query_offset, device=logits.device
    )
    return logits.masked_fill(~mask, float("-inf"))


def causal_attention_mass(
    query: torch.Tensor,
    key: torch.Tensor,
    query_fraction: float = 1.0,
    query_offset: Optional[int] = None,
) -> torch.Tensor:
    """Exact per-key attention mass, averaged over heads and selected queries.

    Args:
        query: ``[batch, query_heads, query_len, head_dim]`` post-RoPE queries.
        key: ``[batch, kv_heads, key_len, head_dim]`` post-RoPE keys.
        query_fraction: fraction of the *last* queries to average over.
        query_offset: absolute position of the first query; defaults to
            ``key_len - query_len``.

    Returns:
        ``[batch, key_len]`` float32 scores.  Each row of the underlying
        attention matrix sums to one, so the returned scores sum to one across
        keys as well.
    """
    logits = _scaled_causal_logits(query, key, query_offset)
    probs = torch.softmax(logits, dim=-1)
    num_queries = select_query_count(query.shape[-2], query_fraction)
    return probs[:, :, query.shape[-2] - num_queries :, :].mean(dim=(1, 2))


def causal_attention_mass_streaming(
    query: torch.Tensor,
    key: torch.Tensor,
    query_fraction: float = 1.0,
    query_offset: Optional[int] = None,
    query_chunk: int = 64,
) -> torch.Tensor:
    """Exact attention mass computed in query chunks (lower peak memory).

    Mathematically identical to :func:`causal_attention_mass`; only the query
    dimension is chunked, so peak memory is
    ``O(batch * heads * query_chunk * key_len)``.
    """
    if query_chunk < 1:
        raise ValueError(f"query_chunk must be >= 1, got {query_chunk}")
    if query.dim() != 4 or key.dim() != 4:
        raise ValueError("query and key must be 4D tensors")
    query_len = query.shape[-2]
    key_len = key.shape[-2]
    num_queries = select_query_count(query_len, query_fraction)
    first_query = query_len - num_queries
    base_offset = key_len - query_len if query_offset is None else query_offset

    total = torch.zeros(
        (query.shape[0], key_len), dtype=torch.float32, device=query.device
    )
    for start in range(first_query, query_len, query_chunk):
        end = min(start + query_chunk, query_len)
        logits = _scaled_causal_logits(
            query[:, :, start:end, :],
            key,
            query_offset=base_offset + start,
        )
        total = total + torch.softmax(logits, dim=-1).sum(dim=(1, 2))

    num_heads = query.shape[1]
    return total / float(num_queries * num_heads)


def attention_mass_from_weights(
    attn_weights: torch.Tensor,
    query_fraction: float = 1.0,
) -> torch.Tensor:
    """Average an already-softmaxed attention tensor over heads and queries.

    Args:
        attn_weights: ``[batch, heads, query_len, key_len]`` attention
            probabilities (e.g. an entry of ``outputs.attentions``).
        query_fraction: fraction of the *last* queries to average over.

    Returns:
        ``[batch, key_len]`` float32 scores.
    """
    if attn_weights.dim() != 4:
        raise ValueError(
            "attn_weights must be [batch, heads, query_len, key_len], got "
            f"{tuple(attn_weights.shape)}"
        )
    query_len = attn_weights.shape[-2]
    num_queries = select_query_count(query_len, query_fraction)
    selected = attn_weights[:, :, query_len - num_queries :, :].float()
    return selected.mean(dim=(1, 2))


def key_norm_scores(key: torch.Tensor) -> torch.Tensor:
    """Per-token key norm averaged over KV heads.

    This is the default flash-compatible sink proxy.  RoPE applies an
    orthogonal rotation per position, so the norm of a key vector is unchanged
    whether it is measured before or after RoPE.

    Args:
        key: ``[batch, kv_heads, seq_len, head_dim]``.

    Returns:
        ``[batch, seq_len]`` float32 scores.
    """
    if key.dim() != 4:
        raise ValueError(
            f"key must be [batch, heads, seq_len, head_dim], got {tuple(key.shape)}"
        )
    norms = torch.linalg.vector_norm(key.float(), dim=-1)
    return norms.mean(dim=1)


def key_norm_scores_from_projection(
    k_projection: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Key-norm proxy from a captured ``k_proj`` output.

    Args:
        k_projection: ``[batch, seq_len, num_kv_heads * head_dim]``.
        num_kv_heads: number of KV heads in the layer.
        head_dim: per-head dimension.

    Returns:
        ``[batch, seq_len]`` float32 scores.
    """
    if num_kv_heads < 1 or head_dim < 1:
        raise ValueError("num_kv_heads and head_dim must be >= 1")
    if k_projection.dim() != 3:
        raise ValueError(
            "k_projection must be [batch, seq_len, num_kv_heads * head_dim], got "
            f"{tuple(k_projection.shape)}"
        )
    expected = num_kv_heads * head_dim
    if k_projection.shape[-1] != expected:
        raise ValueError(
            f"k_projection last dim is {k_projection.shape[-1]}, expected {expected}"
        )
    batch, seq_len, _ = k_projection.shape
    key = k_projection.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    return key_norm_scores(key)
