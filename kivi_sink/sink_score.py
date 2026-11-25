# coding=utf-8
"""Token sink scores, layer scores, budget statistics and rank correlation (M1).

Score conventions
-----------------
A *token score* is a non-negative ``[seq_len]`` tensor: attention mass received
by each key token (exact mode) or a proxy such as the key norm (flash-compatible
mode).  The exact scores already sum to one because every attention row is a
softmax; proxy scores are normalised here before they are compared or turned
into per-layer scores (``A_l``).

A *layer score* ``A_l`` is the normalised score mass of the top-M tokens in
layer ``l``.  It is computed per sample and then aggregated across calibration
samples (mean / median / bootstrap CI), which keeps batch samples independent.

All functions are pure tensor operations; nothing here touches global RNG
state (bootstrap sampling uses a local ``torch.Generator``).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import torch

__all__ = [
    "aggregate_layer_statistics",
    "bootstrap_mean_ci",
    "layer_sink_scores",
    "mean_pairwise_spearman",
    "normalize_token_scores",
    "spearman_correlation",
    "top_m_indices",
    "top_m_share",
]

ScoreLike = Union[torch.Tensor, Sequence[float]]


def _as_float_tensor(values: ScoreLike, name: str = "scores") -> torch.Tensor:
    tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = tensor.float()
    elif not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def _prepare_score_matrix(
    scores: ScoreLike,
    lengths: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    """Return ``(scores_2d, valid_mask, was_1d, lengths_tensor)``.

    ``scores`` may be ``[seq_len]`` or ``[batch, seq_len]``.  ``lengths`` marks
    valid (unpadded) tokens per row; padded positions are excluded from both
    selection and normalisation.
    """
    tensor = _as_float_tensor(scores).detach()
    single = tensor.dim() == 1
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 2:
        raise ValueError(
            f"scores must be 1D or 2D, got shape {tuple(tensor.shape)}"
        )
    if tensor.numel() == 0:
        raise ValueError("scores must not be empty")
    if (tensor < 0).any():
        raise ValueError("sink scores must be non-negative")

    batch_size, seq_len = tensor.shape
    if lengths is None:
        lengths_tensor = torch.full(
            (batch_size,), seq_len, dtype=torch.long, device=tensor.device
        )
    else:
        lengths_tensor = torch.as_tensor(
            lengths, dtype=torch.long, device=tensor.device
        )
        if lengths_tensor.dim() != 1 or lengths_tensor.numel() != batch_size:
            raise ValueError(
                f"lengths must have one entry per row ({batch_size}), got "
                f"{tuple(lengths_tensor.shape)}"
            )
        if (lengths_tensor < 0).any() or (lengths_tensor > seq_len).any():
            raise ValueError(f"lengths must be in [0, {seq_len}]")
    if (lengths_tensor == 0).any():
        raise ValueError("every row needs at least one valid score")

    positions = torch.arange(seq_len, device=tensor.device).unsqueeze(0)
    valid_mask = positions < lengths_tensor.unsqueeze(1)
    return tensor, valid_mask, single, lengths_tensor


def normalize_token_scores(
    scores: ScoreLike,
    lengths: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Normalise token scores to sum to one over valid tokens.

    Returns the same shape as the input (``[seq_len]`` or ``[batch, seq_len]``).
    """
    tensor, valid_mask, single, _ = _prepare_score_matrix(scores, lengths)
    masked = tensor.masked_fill(~valid_mask, 0.0)
    totals = masked.sum(dim=1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("cannot normalise a row whose valid scores sum to zero")
    normalised = masked / totals
    return normalised.squeeze(0) if single else normalised


def top_m_indices(
    scores: ScoreLike,
    m: int,
    lengths: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Indices of the top-``m`` tokens per row (descending score, stable ties).

    Padding positions (beyond ``lengths``) are never selected.  ``m`` must not
    exceed the shortest valid row, so the result always has exactly ``m``
    entries per row.
    """
    tensor, valid_mask, single, lengths_tensor = _prepare_score_matrix(
        scores, lengths
    )
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError(f"m must be an int, got {type(m).__name__}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if m > int(lengths_tensor.min()):
        raise ValueError(
            f"m={m} exceeds the shortest valid sequence ({int(lengths_tensor.min())})"
        )
    masked = tensor.masked_fill(~valid_mask, float("-inf"))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    selected = order[:, :m]
    return selected.squeeze(0) if single else selected


def top_m_share(
    scores: ScoreLike,
    m: int,
    normalize: bool = True,
    lengths: Optional[Sequence[int]] = None,
) -> Union[float, torch.Tensor]:
    """Score mass carried by the top-``m`` tokens.

    With ``normalize=True`` (default) the result is a fraction in ``[0, 1]``
    of the row's total valid score: this is the per-sample ``A_l``.  With
    ``normalize=False`` the raw summed score is returned (useful when the
    scores are already exact attention mass).

    Returns a Python ``float`` for 1D input and a ``[batch]`` tensor for 2D
    input.
    """
    tensor, valid_mask, single, lengths_tensor = _prepare_score_matrix(
        scores, lengths
    )
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError(f"m must be an int, got {type(m).__name__}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if m > int(lengths_tensor.min()):
        raise ValueError(
            f"m={m} exceeds the shortest valid sequence ({int(lengths_tensor.min())})"
        )

    masked = tensor.masked_fill(~valid_mask, float("-inf"))
    order = torch.argsort(masked, dim=1, descending=True, stable=True)
    selected_mass = torch.gather(masked, 1, order[:, :m]).sum(dim=1)

    if not normalize:
        result = selected_mass
    else:
        total = tensor.masked_fill(~valid_mask, 0.0).sum(dim=1)
        if (total <= 0).any():
            raise ValueError("cannot normalise a row whose valid scores sum to zero")
        result = selected_mass / total

    return float(result[0]) if single else result


def layer_sink_scores(
    per_layer_scores: Union[Mapping[int, torch.Tensor], Sequence[torch.Tensor], torch.Tensor],
    top_m: int,
    normalize: bool = True,
    lengths: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Compute ``A_l`` for every layer.

    Args:
        per_layer_scores: mapping ``layer -> [seq_len]``, a sequence of
            ``[seq_len]`` tensors (in layer order), or a ``[num_layers,
            seq_len]`` tensor.
        top_m: number of protected tokens per layer.
        normalize: whether to return the normalised share (default) or the raw
            top-M mass.
        lengths: optional valid-token lengths when the scores are batched.

    Returns:
        ``[num_layers]`` float32 tensor.
    """
    rows: List[torch.Tensor]
    if isinstance(per_layer_scores, Mapping):
        rows = [per_layer_scores[key] for key in sorted(per_layer_scores)]
    elif torch.is_tensor(per_layer_scores):
        if per_layer_scores.dim() == 1:
            raise ValueError(
                "a 1D tensor describes a single layer; pass a mapping, a list or "
                "a [num_layers, seq_len] tensor"
            )
        if per_layer_scores.dim() != 2:
            raise ValueError(
                "per_layer_scores tensor must be 2D [num_layers, seq_len], got "
                f"{tuple(per_layer_scores.shape)}"
            )
        rows = [per_layer_scores[index] for index in range(per_layer_scores.shape[0])]
    else:
        rows = list(per_layer_scores)

    if not rows:
        raise ValueError("per_layer_scores must contain at least one layer")
    values = [
        top_m_share(row, top_m, normalize=normalize, lengths=lengths) for row in rows
    ]
    return torch.as_tensor(values, dtype=torch.float32)


def bootstrap_mean_ci(
    values: ScoreLike,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap percentile interval for the mean of ``values``.

    Sampling uses a local ``torch.Generator`` seeded with ``seed``; the global
    RNG is never touched, so calibration results are reproducible without side
    effects.
    """
    tensor = _as_float_tensor(values, "values").detach().reshape(-1).cpu()
    if tensor.numel() < 1:
        raise ValueError("values must not be empty")
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive int, got {n_boot}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if tensor.numel() == 1:
        value = float(tensor[0])
        return value, value

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0, tensor.numel(), (n_boot, tensor.numel()), generator=generator
    )
    means = tensor[indices].mean(dim=1)
    alpha = (1.0 - ci) / 2.0
    low = float(torch.quantile(means, alpha))
    high = float(torch.quantile(means, 1.0 - alpha))
    return low, high


def _score_matrix_from_samples(
    per_sample_scores: Union[torch.Tensor, Sequence[ScoreLike]],
) -> torch.Tensor:
    if torch.is_tensor(per_sample_scores):
        matrix = _as_float_tensor(per_sample_scores, "per_sample_scores").detach()
        if matrix.dim() == 1:
            matrix = matrix.unsqueeze(0)
        if matrix.dim() != 2:
            raise ValueError(
                "per_sample_scores tensor must be [num_samples, num_layers], got "
                f"{tuple(matrix.shape)}"
            )
        return matrix
    rows = [
        _as_float_tensor(row, "per_sample_scores").detach().reshape(-1)
        for row in per_sample_scores
    ]
    if not rows:
        raise ValueError("per_sample_scores must contain at least one sample")
    widths = {row.numel() for row in rows}
    if len(widths) != 1:
        raise ValueError(f"all samples must have the same layer count, got {widths}")
    return torch.stack(rows)


def aggregate_layer_statistics(
    per_sample_layer_scores: Union[torch.Tensor, Sequence[ScoreLike]],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Dict[str, List[float]]:
    """Aggregate per-sample ``A_l`` across calibration samples.

    Args:
        per_sample_layer_scores: ``[num_samples, num_layers]`` tensor or a list
            of per-sample layer-score vectors.
        n_boot, ci, seed: forwarded to :func:`bootstrap_mean_ci`.

    Returns:
        JSON-ready dictionary with ``mean``, ``median``, ``std``, ``ci_low``,
        ``ci_high`` (one value per layer) and ``n`` (number of samples).
    """
    matrix = _score_matrix_from_samples(per_sample_layer_scores)
    num_samples, num_layers = matrix.shape
    if num_layers < 1:
        raise ValueError("per-layer score vectors must not be empty")

    mean = matrix.mean(dim=0)
    median = matrix.median(dim=0).values
    if num_samples > 1:
        std = matrix.std(dim=0, unbiased=True)
    else:
        std = torch.zeros(num_layers, dtype=matrix.dtype)

    ci_low: List[float] = []
    ci_high: List[float] = []
    for layer in range(num_layers):
        low, high = bootstrap_mean_ci(
            matrix[:, layer], n_boot=n_boot, ci=ci, seed=seed
        )
        ci_low.append(low)
        ci_high.append(high)

    return {
        "mean": [float(value) for value in mean],
        "median": [float(value) for value in median],
        "std": [float(value) for value in std],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": [int(num_samples)] * num_layers,
    }


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranks with ties assigned their average rank (1-based)."""
    n = values.numel()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty(n, dtype=torch.float64)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = rank
        start = end + 1
    return ranks


def spearman_correlation(x: ScoreLike, y: ScoreLike) -> float:
    """Spearman rank correlation between two 1D vectors.

    Ties use average ranks.  Returns ``nan`` when either vector is constant or
    has fewer than two elements.
    """
    x_tensor = _as_float_tensor(x, "x").detach().reshape(-1).cpu()
    y_tensor = _as_float_tensor(y, "y").detach().reshape(-1).cpu()
    if x_tensor.numel() != y_tensor.numel():
        raise ValueError(
            f"x and y must have the same length, got {x_tensor.numel()} and "
            f"{y_tensor.numel()}"
        )
    if x_tensor.numel() < 2:
        return float("nan")

    x_ranks = _average_ranks(x_tensor)
    y_ranks = _average_ranks(y_tensor)
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denominator = torch.sqrt(
        (x_centered * x_centered).sum() * (y_centered * y_centered).sum()
    )
    if float(denominator) == 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum() / denominator)


def _nanmean(values: Iterable[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def mean_pairwise_spearman(
    per_sample_scores: Union[torch.Tensor, Sequence[ScoreLike]],
) -> float:
    """Mean pairwise Spearman correlation across samples (stability metric).

    Input rows are per-sample score vectors (e.g. ``A_l`` across layers).  A
    high value means the ranking is stable across calibration samples.
    Returns ``nan`` with fewer than two samples or fewer than two entries per
    sample.
    """
    matrix = _score_matrix_from_samples(per_sample_scores)
    num_samples = matrix.shape[0]
    if num_samples < 2 or matrix.shape[1] < 2:
        return float("nan")
    correlations: List[float] = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            correlations.append(
                spearman_correlation(matrix[i], matrix[j])
            )
    return _nanmean(correlations)
