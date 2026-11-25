#!/usr/bin/env python3
# coding=utf-8
"""M1 calibration runner: collect attention statistics and layer sink scores.

This entry point runs in its own process and never touches the KIVI inference
path.  It loads a plain Hugging Face model, runs a single prefill forward per
calibration sample, and writes per-layer token scores / ``A_l`` statistics to
JSON (plus a CSV summary).

Modes
-----
``exact``
    Eager attention with ``output_attentions=True``.  Exact, but materialises
    the full attention matrix per layer, so keep ``--max-length`` small; treat
    it as a diagnostic and as ground truth for proxy validation.
``proxy``
    SDPA attention without attention weights; a read-only forward hook captures
    ``self_attn.k_proj`` outputs and the per-token key norm is used as the sink
    proxy.  Key norms are RoPE-invariant, so this is flash-compatible.
``both``
    Eager pass that computes exact attention and the proxy at the same time,
    plus per-layer Spearman correlations between them.

Calibration data
----------------
The default source is LongBench ``narrativeqa`` (not in the LongBench or
LongBench-E evaluation task lists, so calibration and test data stay
disjoint).  ``--calibration-jsonl`` accepts a held-out JSONL file with a
``text`` field (or ``context``/``input`` fields).  ``--synthetic`` generates
random Q/K tensors and needs no model or dataset, which makes the full pipeline
smoke-testable on a CPU-only machine.

Examples
--------
Synthetic smoke test (no model, no network)::

    python -m kivi_sink.calibration --synthetic \
        --output results/sink/calibration/synthetic.json

Exact attention + proxy correlation on a short diagnostic window::

    python -m kivi_sink.calibration \
        --model LLM-Research/Meta-Llama-3.1-8B-Instruct --mode both \
        --max-length 512 --samples 8 \
        --output results/sink/calibration/llama31_both.json

Long-context proxy pass::

    python -m kivi_sink.calibration \
        --model LLM-Research/Meta-Llama-3.1-8B-Instruct --mode proxy \
        --max-length 4096 --samples 16 \
        --output results/sink/calibration/llama31_proxy.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from kivi_sink.attention_stats import (
    attention_mass_from_weights,
    causal_attention_mass,
    key_norm_scores,
    key_norm_scores_from_projection,
)
from kivi_sink.sink_score import (
    aggregate_layer_statistics,
    layer_sink_scores,
    mean_pairwise_spearman,
    spearman_correlation,
)
from utils.device_utils import resolve_device

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "sink" / "calibration"

# Default calibration task: LongBench English long-context task that is not part
# of either the default 8-task or the LongBench-E 13-task evaluation sets.
DEFAULT_CALIBRATION_TASK = "narrativeqa"

# Tasks used by pred_long_bench.py (default set and --e set).  Calibration must
# stay disjoint from these unless --allow-eval-task is passed explicitly.
EVAL_TASKS = frozenset(
    {
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "gov_report",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_count",
        "passage_retrieval_en",
        "lcc",
        "repobench-p",
        "qmsum",
    }
)

SCHEMA_VERSION = 1
_K_PROJ_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.k_proj$")
_DTYPE_CHOICES = ("auto", "float16", "bfloat16", "float32")


@dataclass
class _SampleResult:
    """Scores collected for one calibration sample."""

    sample_id: str
    num_tokens: int
    exact: Optional[Dict[int, torch.Tensor]] = None
    proxy: Optional[Dict[int, torch.Tensor]] = None


# ---------------------------------------------------------------------------
# read-only k_proj capture (calibration process only)
# ---------------------------------------------------------------------------


class KProjCapture:
    """Capture ``self_attn.k_proj`` outputs without modifying the model.

    Supports Llama/Mistral/Qwen2-style module names
    (``...layers.<i>.self_attn.k_proj``).  Hooks only read the projection
    output and are always removed by :meth:`stop`.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._targets: List[Tuple[int, torch.nn.Module]] = self._discover()

    def _discover(self) -> List[Tuple[int, torch.nn.Module]]:
        targets: Dict[int, torch.nn.Module] = {}
        for name, module in self.model.named_modules():
            match = _K_PROJ_PATTERN.search(name)
            if match is not None:
                targets[int(match.group(1))] = module
        if not targets:
            raise RuntimeError(
                "no 'layers.<i>.self_attn.k_proj' modules found; the proxy "
                "mode currently supports Llama/Mistral/Qwen2-style models"
            )
        return sorted(targets.items())

    def _hook(self, layer: int):
        def capture(module, inputs, output):
            self.captured[layer] = output.detach()

        return capture

    def start(self) -> "KProjCapture":
        if self._handles:
            raise RuntimeError("KProjCapture is already started")
        for layer, module in self._targets:
            self._handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def stop(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear(self) -> None:
        self.captured = {}

    def __enter__(self) -> "KProjCapture":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    @property
    def num_layers(self) -> int:
        return len(self._targets)


# ---------------------------------------------------------------------------
# calibration data
# ---------------------------------------------------------------------------


def _text_from_record(record: Dict[str, object], text_field: Optional[str]) -> str:
    if text_field:
        value = record.get(text_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"record has no non-empty string field {text_field!r}; keys: "
                f"{sorted(record)}"
            )
        return value

    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text

    parts = []
    for field in ("context", "input"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    if not parts:
        raise ValueError(
            "record has no 'text' field and no non-empty 'context'/'input' "
            f"fields; keys: {sorted(record)}. Pass --text-field to select one."
        )
    return "\n\n".join(parts)


def load_calibration_records(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Load calibration texts from JSONL or LongBench.

    Returns a deterministic list of ``{"id": str, "text": str}`` dictionaries.
    LongBench records are taken in dataset order (first ``--samples`` rows), so
    repeated runs see the same samples.
    """
    if args.calibration_jsonl and args.longbench_task:
        raise ValueError("pass either --calibration-jsonl or --longbench-task, not both")

    if args.calibration_jsonl:
        path = Path(os.path.expanduser(args.calibration_jsonl))
        records: List[Dict[str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON ({exc})"
                    ) from exc
                text = _text_from_record(record, args.text_field)
                sample_id = str(
                    record.get("_id")
                    or record.get("id")
                    or f"{path.stem}-{len(records)}"
                )
                records.append({"id": sample_id, "text": text})
                if len(records) >= args.samples:
                    break
        if not records:
            raise ValueError(f"{path} contains no calibration records")
        return records

    task = args.longbench_task or DEFAULT_CALIBRATION_TASK
    if task in EVAL_TASKS and not args.allow_eval_task:
        raise ValueError(
            f"LongBench task {task!r} is part of the evaluation set; choose a "
            "task outside EVAL_TASKS or pass --allow-eval-task deliberately "
            f"(evaluation tasks: {sorted(EVAL_TASKS)})"
        )

    from utils.modelscope_hub import load_long_bench_task

    dataset = load_long_bench_task(
        task,
        data_source=args.data_source,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    records = []
    for index in range(min(args.samples, len(dataset))):
        record = dict(dataset[index])
        text = _text_from_record(record, args.text_field)
        sample_id = str(record.get("_id") or f"{task}-{index}")
        records.append({"id": sample_id, "text": text})
    if not records:
        raise ValueError(f"LongBench task {task!r} returned no samples")
    return records


def _maybe_chat_template(
    text: str, tokenizer, model_name: Optional[str], mode: str
) -> str:
    """Mirror the LongBench entry point's Llama-3-Instruct chat template."""
    if mode == "none":
        return text
    should_apply = mode == "always"
    if mode == "auto":
        lowered = (model_name or "").lower()
        should_apply = "llama-3" in lowered and "instruct" in lowered
    if not should_apply:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# model execution
# ---------------------------------------------------------------------------


def _resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    if spec == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[spec]


def _resolve_model_path(args: argparse.Namespace) -> str:
    from utils.modelscope_hub import resolve_model_path

    return resolve_model_path(
        args.model,
        model_source=args.model_source,
        cache_dir=args.cache_dir,
    )


def _load_model(path: str, dtype: torch.dtype, device: torch.device, eager: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="eager" if eager else "sdpa",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _proxy_scores(
    capture: KProjCapture, config
) -> Dict[int, torch.Tensor]:
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if not num_kv_heads:
        num_kv_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None)
    if not head_dim:
        head_dim = config.hidden_size // config.num_attention_heads
    return {
        layer: key_norm_scores_from_projection(projection, num_kv_heads, head_dim)[0]
        .float()
        .cpu()
        for layer, projection in capture.captured.items()
    }


def _collect_model_scores(
    args: argparse.Namespace,
) -> Tuple[List[_SampleResult], List[str], Dict[str, object]]:
    device = resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    path = _resolve_model_path(args)
    eager = args.mode in ("exact", "both")
    model, tokenizer = _load_model(path, dtype, device, eager=eager)

    capture = None
    if args.mode in ("proxy", "both"):
        capture = KProjCapture(model)

    texts = load_calibration_records(args)
    results: List[_SampleResult] = []
    skipped: List[str] = []

    try:
        for record in texts:
            text = _maybe_chat_template(
                record["text"], tokenizer, args.model, args.chat_template
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            num_tokens = int(input_ids.shape[1])
            if num_tokens < args.top_m:
                skipped.append(record["id"])
                print(
                    f"[calibration] skipping {record['id']}: {num_tokens} tokens "
                    f"< top_m={args.top_m}",
                    file=sys.stderr,
                )
                continue

            if capture is not None:
                capture.clear()
                capture.start()
            try:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=eager,
                    )
            finally:
                if capture is not None:
                    capture.stop()

            exact = None
            if eager:
                weights = getattr(outputs, "attentions", None)
                if not weights:
                    raise RuntimeError(
                        "model did not return attention weights; exact mode needs "
                        "a Hugging Face model with eager attention"
                    )
                exact = {
                    layer: attention_mass_from_weights(
                        weight, args.query_fraction
                    )[0]
                    .float()
                    .cpu()
                    for layer, weight in enumerate(weights)
                }

            proxy = None
            if capture is not None:
                proxy = _proxy_scores(capture, model.config)
                if exact is not None and set(proxy) != set(exact):
                    raise RuntimeError(
                        "captured k_proj layers do not match attention layers: "
                        f"proxy={sorted(proxy)} exact={sorted(exact)}"
                    )

            results.append(
                _SampleResult(
                    sample_id=record["id"],
                    num_tokens=num_tokens,
                    exact=exact,
                    proxy=proxy,
                )
            )

            del outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if capture is not None:
            capture.stop()

    if not results:
        raise RuntimeError(
            "no calibration sample was long enough for top_m; lower --top-m or "
            "use longer calibration texts"
        )

    metadata: Dict[str, object] = {
        "model_path": path,
        "num_decoder_layers": int(model.config.num_hidden_layers),
        "dtype": str(dtype),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, skipped, metadata


def _collect_synthetic_scores(
    args: argparse.Namespace,
) -> Tuple[List[_SampleResult], List[str], Dict[str, object]]:
    """Random Q/K pass; exercises the full pipeline without transformers."""
    if args.max_length < args.top_m:
        raise ValueError("--max-length must be >= --top-m")
    query_heads, kv_heads, head_dim = 4, 2, 16
    want_exact = args.mode in ("exact", "both")
    want_proxy = args.mode in ("proxy", "both")
    generator = torch.Generator()
    results: List[_SampleResult] = []
    for sample_index in range(args.samples):
        generator.manual_seed(args.seed + sample_index)
        exact: Optional[Dict[int, torch.Tensor]] = {} if want_exact else None
        proxy: Optional[Dict[int, torch.Tensor]] = {} if want_proxy else None
        for layer in range(args.synthetic_layers):
            query = torch.randn(
                1, query_heads, args.max_length, head_dim, generator=generator
            )
            key = torch.randn(
                1, kv_heads, args.max_length, head_dim, generator=generator
            )
            # Make layers mildly heterogeneous so A_l is not trivially flat.
            key = key * (0.5 + 0.5 * layer)
            if exact is not None:
                exact[layer] = causal_attention_mass(
                    query, key, args.query_fraction
                )[0]
            if proxy is not None:
                proxy[layer] = key_norm_scores(key)[0]
        results.append(
            _SampleResult(
                sample_id=f"synthetic-{sample_index}",
                num_tokens=args.max_length,
                exact=exact,
                proxy=proxy,
            )
        )
    metadata: Dict[str, object] = {
        "model_path": None,
        "num_decoder_layers": args.synthetic_layers,
        "dtype": "float32",
        "synthetic_query_heads": query_heads,
        "synthetic_kv_heads": kv_heads,
        "synthetic_head_dim": head_dim,
    }
    return results, [], metadata


# ---------------------------------------------------------------------------
# summary and output
# ---------------------------------------------------------------------------


def _version_or_none(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _metadata(args: argparse.Namespace, extra: Dict[str, object]) -> Dict[str, object]:
    dirty = _git("status", "--porcelain")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": _version_or_none("transformers"),
        "flash_attn": _version_or_none("flash-attn"),
        "device": str(resolve_device(args.device)),
    }
    metadata.update(extra)
    return metadata


def _layer_ids(results: Sequence[_SampleResult], mode: str) -> List[int]:
    layers = None
    for result in results:
        scores = getattr(result, mode)
        if scores is None:
            continue
        keys = tuple(sorted(scores))
        if layers is None:
            layers = keys
        elif keys != layers:
            raise ValueError(f"inconsistent {mode} layer sets: {layers} vs {keys}")
    if layers is None:
        raise ValueError(f"no {mode} scores were collected")
    return list(layers)


def _layer_matrix(
    results: Sequence[_SampleResult],
    mode: str,
    layer_ids: Sequence[int],
    top_m: int,
) -> torch.Tensor:
    rows = []
    for result in results:
        scores = getattr(result, mode)
        if scores is None:
            raise ValueError(f"sample {result.sample_id} has no {mode} scores")
        rows.append(
            layer_sink_scores(
                {layer: scores[layer] for layer in layer_ids},
                top_m,
                normalize=True,
            )
        )
    return torch.stack(rows)


def _nanmean(values: Sequence[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _summarize(
    args: argparse.Namespace,
    results: Sequence[_SampleResult],
    skipped: Sequence[str],
    extra_metadata: Dict[str, object],
) -> Dict[str, object]:
    has_exact = any(result.exact is not None for result in results)
    has_proxy = any(result.proxy is not None for result in results)
    exact_layers = _layer_ids(results, "exact") if has_exact else []
    proxy_layers = _layer_ids(results, "proxy") if has_proxy else []

    exact_matrix = (
        _layer_matrix(results, "exact", exact_layers, args.top_m) if has_exact else None
    )
    proxy_matrix = (
        _layer_matrix(results, "proxy", proxy_layers, args.top_m) if has_proxy else None
    )

    exact_stats = (
        aggregate_layer_statistics(
            exact_matrix, n_boot=args.n_boot, ci=args.ci, seed=args.seed
        )
        if exact_matrix is not None
        else None
    )
    proxy_stats = (
        aggregate_layer_statistics(
            proxy_matrix, n_boot=args.n_boot, ci=args.ci, seed=args.seed
        )
        if proxy_matrix is not None
        else None
    )

    token_spearman: List[float] = []
    if has_exact and has_proxy:
        if exact_layers != proxy_layers:
            raise ValueError(
                f"exact and proxy layers differ: {exact_layers} vs {proxy_layers}"
            )
        for layer in exact_layers:
            per_sample = []
            for result in results:
                per_sample.append(
                    spearman_correlation(
                        result.exact[layer], result.proxy[layer]
                    )
                )
            token_spearman.append(_nanmean(per_sample))

    layer_correlation: Dict[str, object] = {}
    if has_exact and has_proxy:
        per_sample_layer_corr = [
            spearman_correlation(exact_matrix[i], proxy_matrix[i])
            for i in range(len(results))
        ]
        layer_correlation.update(
            {
                "proxy": "key_norm",
                "token_spearman_per_layer": token_spearman,
                "token_spearman_mean": _nanmean(token_spearman),
                "layer_A_spearman_per_sample": per_sample_layer_corr,
                "layer_A_spearman_mean": _nanmean(per_sample_layer_corr),
            }
        )
    if exact_matrix is not None:
        layer_correlation["exact_layer_A_stability_spearman"] = (
            mean_pairwise_spearman(exact_matrix)
        )
    if proxy_matrix is not None:
        layer_correlation["proxy_layer_A_stability_spearman"] = (
            mean_pairwise_spearman(proxy_matrix)
        )

    layer_entries: List[Dict[str, object]] = []
    layer_count = max(len(exact_layers), len(proxy_layers))
    for index in range(layer_count):
        entry: Dict[str, object] = {"layer": None}
        if index < len(exact_layers):
            entry["layer"] = exact_layers[index]
        elif index < len(proxy_layers):
            entry["layer"] = proxy_layers[index]
        for prefix, stats in (("exact", exact_stats), ("proxy", proxy_stats)):
            if stats is None:
                continue
            entry[f"{prefix}_A_mean"] = stats["mean"][index]
            entry[f"{prefix}_A_median"] = stats["median"][index]
            entry[f"{prefix}_A_std"] = stats["std"][index]
            entry[f"{prefix}_A_ci_low"] = stats["ci_low"][index]
            entry[f"{prefix}_A_ci_high"] = stats["ci_high"][index]
        if index < len(token_spearman):
            entry["exact_token_spearman_mean"] = token_spearman[index]
        layer_entries.append(entry)

    sample_entries = []
    for sample_index, result in enumerate(results):
        entry: Dict[str, object] = {
            "id": result.sample_id,
            "num_tokens": result.num_tokens,
        }
        if exact_matrix is not None:
            entry["exact_A"] = [float(value) for value in exact_matrix[sample_index]]
        if proxy_matrix is not None:
            entry["proxy_A"] = [float(value) for value in proxy_matrix[sample_index]]
        sample_entries.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": _metadata(args, extra_metadata),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"func"}
        },
        "samples": sample_entries,
        "samples_skipped": list(skipped),
        "layers": layer_entries,
        "correlation": layer_correlation,
    }


_CSV_FIELDS = [
    "layer",
    "exact_A_mean",
    "exact_A_median",
    "exact_A_std",
    "exact_A_ci_low",
    "exact_A_ci_high",
    "proxy_A_mean",
    "proxy_A_median",
    "proxy_A_std",
    "proxy_A_ci_low",
    "proxy_A_ci_high",
    "exact_token_spearman_mean",
]


def _write_outputs(payload: Dict[str, object], output_path: Path) -> Tuple[Path, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for entry in payload["layers"]:
            writer.writerow({key: entry.get(key) for key in _CSV_FIELDS})
    return output_path, csv_path


def _save_token_scores(
    path: str, results: Sequence[_SampleResult], exact_layers, proxy_layers
) -> Path:
    import numpy as np

    arrays: Dict[str, "np.ndarray"] = {}
    for sample_index, result in enumerate(results):
        if result.exact is not None:
            for layer in exact_layers:
                arrays[f"exact_sample{sample_index}_layer{layer}"] = (
                    result.exact[layer].cpu().numpy()
                )
        if result.proxy is not None:
            for layer in proxy_layers:
                arrays[f"proxy_sample{sample_index}_layer{layer}"] = (
                    result.proxy[layer].cpu().numpy()
                )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M1 calibration: exact/proxy attention statistics and "
        "layer sink scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="Model id or local path.")
    parser.add_argument(
        "--model-source",
        default="modelscope",
        choices=("modelscope", "hf"),
        help="Where to resolve --model from.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Explicit cache root; see utils/modelscope_hub.py.",
    )
    parser.add_argument(
        "--mode",
        default="exact",
        choices=("exact", "proxy", "both"),
        help="Attention statistics mode.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random Q/K tensors instead of loading a model/dataset.",
    )
    parser.add_argument(
        "--synthetic-layers",
        type=int,
        default=4,
        help="Number of synthetic layers (--synthetic only).",
    )
    parser.add_argument(
        "--calibration-jsonl",
        default=None,
        help="Held-out JSONL file with a 'text' field (or context/input).",
    )
    parser.add_argument(
        "--longbench-task",
        default=None,
        help=f"LongBench task used as calibration data (default: "
        f"{DEFAULT_CALIBRATION_TASK}).",
    )
    parser.add_argument(
        "--text-field",
        default=None,
        help="Explicit JSONL field holding the text (default: auto-detect "
        "'text' or 'context'/'input').",
    )
    parser.add_argument(
        "--allow-eval-task",
        action="store_true",
        help="Allow a LongBench task from the evaluation set (leakage risk).",
    )
    parser.add_argument(
        "--data-source",
        default="modelscope",
        choices=("modelscope", "hf"),
        help="Where to resolve the LongBench task from.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Local LongBench data directory (reused instead of downloading).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Token truncation length of each calibration sample.",
    )
    parser.add_argument(
        "--query-fraction",
        type=float,
        default=1.0,
        help="Fraction of the last prefill queries used for token scores.",
    )
    parser.add_argument(
        "--top-m",
        type=int,
        default=4,
        help="Top-M tokens per layer used for the layer score A_l.",
    )
    parser.add_argument(
        "--chat-template",
        default="auto",
        choices=("auto", "none", "always"),
        help="Apply a chat template before tokenising (auto: Llama-3 Instruct).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=_DTYPE_CHOICES,
        help="Model dtype (auto: float16 on CUDA, float32 on CPU).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. 'cuda:1', '1', or 'cpu'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for bootstrap sampling and synthetic tensors (local RNG only).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for the A_l confidence interval.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for the bootstrap interval.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (a sibling .csv is written next to it).",
    )
    parser.add_argument(
        "--save-token-scores",
        default=None,
        help="Optional .npz path for per-sample, per-layer token scores.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.synthetic:
        if args.model:
            raise ValueError("--synthetic does not accept --model")
    else:
        if not args.model:
            raise ValueError("--model is required unless --synthetic is used")
    if args.calibration_jsonl and args.longbench_task:
        raise ValueError("pass either --calibration-jsonl or --longbench-task, not both")
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")
    if args.max_length < 2:
        raise ValueError("--max-length must be >= 2")
    if args.top_m < 1:
        raise ValueError("--top-m must be >= 1")
    if not (0.0 < args.query_fraction <= 1.0):
        raise ValueError("--query-fraction must be in (0, 1]")
    if args.n_boot < 1:
        raise ValueError("--n-boot must be >= 1")
    if not (0.0 < args.ci < 1.0):
        raise ValueError("--ci must be in (0, 1)")
    if args.synthetic_layers < 1:
        raise ValueError("--synthetic-layers must be >= 1")


def _default_output_path(args: argparse.Namespace) -> Path:
    if args.synthetic:
        stem = "synthetic"
    else:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", (args.model or "model").rstrip("/").split("/")[-1])
        stem = f"{name}_{args.mode}_{args.samples}samples_len{args.max_length}"
    return DEFAULT_OUTPUT_DIR / f"{stem}.json"


def _print_summary(payload: Dict[str, object], paths: Sequence[Path]) -> None:
    layers = payload["layers"]
    print(
        f"[calibration] samples={len(payload['samples'])} "
        f"layers={len(layers)} "
        f"skipped={len(payload['samples_skipped'])}"
    )
    for prefix in ("exact", "proxy"):
        values = [
            entry[f"{prefix}_A_mean"]
            for entry in layers
            if f"{prefix}_A_mean" in entry
        ]
        if values:
            print(
                f"[calibration] {prefix} A_l mean={sum(values) / len(values):.4f} "
                f"min={min(values):.4f} max={max(values):.4f}"
            )
    correlation = payload["correlation"]
    if "token_spearman_mean" in correlation:
        print(
            "[calibration] exact-vs-proxy token Spearman mean="
            f"{correlation['token_spearman_mean']:.4f}"
        )
    for path in paths:
        print(f"[calibration] wrote {path}")


def run_calibration(args: argparse.Namespace) -> Dict[str, object]:
    _validate_args(args)
    if args.synthetic:
        results, skipped, extra_metadata = _collect_synthetic_scores(args)
    else:
        results, skipped, extra_metadata = _collect_model_scores(args)

    payload = _summarize(args, results, skipped, extra_metadata)
    output_path = Path(args.output) if args.output else _default_output_path(args)
    json_path, csv_path = _write_outputs(payload, output_path)

    written = [json_path, csv_path]
    if args.save_token_scores:
        written.append(
            _save_token_scores(
                args.save_token_scores,
                results,
                [entry["layer"] for entry in payload["layers"] if "exact_A_mean" in entry],
                [entry["layer"] for entry in payload["layers"] if "proxy_A_mean" in entry],
            )
        )
    _print_summary(payload, written)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_calibration(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
