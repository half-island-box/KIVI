# coding=utf-8
"""Per-layer quantization sensitivity (``E_l``) probe for the KIVI cache.

Why this exists
---------------
M1 showed the per-layer attention mass ``A_l`` varies 2.15x across layers, but
the token-level story is saturated: protecting {0,1} in every layer recovers 62%
of the 2-bit PPL loss, and protecting the whole prefill only reaches 75%.  The
open question for a layer-wise *precision* budget is whether the damage from
2-bit KV is itself uneven across layers -- if it is, upgrading the sensitive
layers to 4-bit should beat upgrading an arbitrary half.

This script measures that directly.  For each probed layer it upgrades exactly
that layer from ``--base_bits`` to ``--upgrade_bits`` and re-measures the same
PPL windows; the drop relative to an all-base run is the layer's marginal value.

``k_bits`` / ``v_bits`` are read from the attention module on every forward, so
the probe only assigns attributes -- no model code changes and no re-load
between layers:

    for layer, attn in enumerate(model.model.layers):
        attn.self_attn.k_bits = ...

Measurement contract
--------------------
Identical to ``ppl_eval.py`` (decode-based, see its docstring): each window is
one 512-token prefill plus 2048 teacher-forced single-token steps, so every
scored position reads the (quantized) history.  Windows are fixed once and
reused across configs, so the deltas are paired and directly comparable.

Read the result as a *go / no-go for layer-wise bit allocation*:

- spread of the per-layer deltas >= ~1.5x  -> an allocator has something to
  exploit; compare it against "random half upgraded" before believing it;
- spread < ~1.3x -> per-layer bit allocation is like token selection: the
  lever is flat, and the effort belongs in M4 (decode-side quantization).

Usage::

    python probe_layer_bits.py \
        --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
        --device 7 --base_bits 2 --upgrade_bits 4 \
        --layers 0,4,8,12,16,20,24,28 --ceiling \
        --json_out results/sink/probe/llama31_E_l_subset.json

    # all 32 layers, no ceiling run
    python probe_layer_bits.py ... --layers all
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import ppl_eval


def _log(message):
    print(f"[probe] {message}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure each layer's marginal PPL value of one extra bit, using the "
            "decode-based WikiText continuation loss."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)

    from utils.modelscope_hub import add_model_source_argument
    from utils.device_utils import add_device_argument

    add_model_source_argument(parser)
    add_device_argument(parser)

    parser.add_argument("--base_bits", type=int, default=2, choices=(2, 4, 8),
                        help="bit width applied to every layer outside the probe")
    parser.add_argument("--upgrade_bits", type=int, default=4, choices=(2, 4, 8),
                        help="bit width given to the probed layer")
    parser.add_argument("--upgrade", type=str, default="both", choices=("k", "v", "both"),
                        help="which side of the cache the probed layer upgrades")
    parser.add_argument("--layers", type=str, default="all",
                        help="'all' or a comma separated list, e.g. 0,4,8,12")
    parser.add_argument("--ceiling", action="store_true",
                        help="also measure all-layers-at-upgrade_bits, to normalise deltas")

    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--protect_prefix", type=int, default=0)
    parser.add_argument("--prefill_length", type=int, default=512)
    parser.add_argument("--decode_length", type=int, default=2048)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=ppl_eval.DEFAULT_DATASET)
    parser.add_argument("--subset", type=str, default=ppl_eval.DEFAULT_SUBSET)
    parser.add_argument("--split", type=str, default=ppl_eval.DEFAULT_SPLIT)
    parser.add_argument("--json_out", type=str, default=None)
    return parser


def parse_layers(spec, num_layers):
    if spec.strip().lower() == "all":
        return list(range(num_layers))
    layers = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        index = int(piece)
        if index < 0 or index >= num_layers:
            raise SystemExit(f"--layers {index} is out of range for {num_layers} layers")
        layers.append(index)
    if not layers:
        raise SystemExit("--layers selected nothing")
    return sorted(set(layers))


def apply_bits(model, k_bits, v_bits):
    """Assign per-layer bit widths; the attention reads them on every forward."""
    for index, layer in enumerate(model.model.layers):
        layer.self_attn.k_bits = k_bits[index]
        layer.self_attn.v_bits = v_bits[index]


def run_ppl(model, segments, args, device):
    """Teacher-forced continuation loss over the fixed windows."""
    import torch

    total_nll = 0.0
    tokens = 0
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(0, len(segments), args.batch_size):
            batch = segments[index:index + args.batch_size]
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            total_nll += ppl_eval.segment_nll(
                model, input_ids, args.prefill_length, args.decode_length
            )
            tokens += input_ids.shape[0] * args.decode_length
    seconds = time.perf_counter() - started
    mean_nll = total_nll / max(tokens, 1)
    ppl = math.exp(mean_nll)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ppl, seconds, tokens


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.base_bits == args.upgrade_bits:
        parser.error("--base_bits and --upgrade_bits are equal; nothing to probe")

    import torch
    from utils.device_utils import resolve_device
    from utils.modelscope_hub import resolve_model_path
    from utils.tokenizer_utils import load_llama_tokenizer

    device = resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        device = torch.device(f"cuda:{device.index}")

    model_path = resolve_model_path(
        args.model_name_or_path,
        model_source=args.model_source,
        cache_dir=args.cache_dir,
    )
    tokenizer = load_llama_tokenizer(model_path, args.model_name_or_path)

    text = ppl_eval.load_corpus(args)
    ids = ppl_eval.tokenize_corpus(tokenizer, text)
    span = args.prefill_length + args.decode_length
    starts = ppl_eval.window_starts(len(ids), span, args.windows)
    if not starts:
        raise SystemExit(f"corpus has {len(ids)} tokens, fewer than one {span}-token window")
    segments = [ids[start:start + span] for start in starts]
    _log(
        f"corpus tokens: {len(ids)}, windows: {len(segments)} "
        f"({args.prefill_length} prefill + {args.decode_length} decode)"
    )

    # Load with the base width; the probe then reassigns attributes per run.
    load_args = argparse.Namespace(**vars(args))
    load_args.k_bits = args.base_bits
    load_args.v_bits = args.base_bits
    model = ppl_eval.load_model(load_args, model_path, device)

    num_layers = len(model.model.layers)
    probe_layers = parse_layers(args.layers, num_layers)
    runs = len(probe_layers) + 1 + (1 if args.ceiling else 0)
    _log(f"{num_layers} layers, probing {len(probe_layers)} of them; {runs} PPL runs")

    base_k = [args.base_bits] * num_layers
    base_v = [args.base_bits] * num_layers

    apply_bits(model, base_k, base_v)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_ppl, baseline_s, tokens = run_ppl(model, segments, args, device)
    _log(f"baseline (all {args.base_bits}-bit): ppl={baseline_ppl:.4f} in {baseline_s:.0f}s")

    ceiling_ppl = None
    if args.ceiling:
        apply_bits(model, [args.upgrade_bits] * num_layers, [args.upgrade_bits] * num_layers)
        ceiling_ppl, ceiling_s, _ = run_ppl(model, segments, args, device)
        _log(f"ceiling (all {args.upgrade_bits}-bit): ppl={ceiling_ppl:.4f} in {ceiling_s:.0f}s")

    ceiling_gap = None
    if ceiling_ppl is not None:
        ceiling_gap = baseline_ppl - ceiling_ppl

    records = []
    for order, layer in enumerate(probe_layers, 1):
        k_bits, v_bits = list(base_k), list(base_v)
        if args.upgrade in ("k", "both"):
            k_bits[layer] = args.upgrade_bits
        if args.upgrade in ("v", "both"):
            v_bits[layer] = args.upgrade_bits
        apply_bits(model, k_bits, v_bits)
        ppl, seconds, _ = run_ppl(model, segments, args, device)
        delta = baseline_ppl - ppl
        recovery = (delta / ceiling_gap) if ceiling_gap else None
        records.append({
            "layer": layer,
            "ppl": ppl,
            "delta_vs_baseline": delta,
            "fraction_of_ceiling": recovery,
            "seconds": seconds,
        })
        extra = ""
        if recovery is not None:
            extra = f" ({100 * recovery:+.1f}% of the ceiling gap)"
        _log(f"[{order}/{len(probe_layers)}] L{layer}: ppl={ppl:.4f} delta={delta:+.4f}{extra}")

    apply_bits(model, base_k, base_v)

    deltas = [r["delta_vs_baseline"] for r in records]
    positive = [d for d in deltas if d > 0]
    summary = {
        "baseline_ppl": baseline_ppl,
        "ceiling_ppl": ceiling_ppl,
        "ceiling_gap": ceiling_gap,
        "delta_min": min(deltas) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "delta_mean": sum(deltas) / len(deltas) if deltas else None,
        "spread_ratio": (max(positive) / min(positive)) if len(positive) >= 2 and min(positive) > 0 else None,
        "layers_improving": len(positive),
        "layers_probed": len(records),
        "best_layers": [r["layer"] for r in sorted(records, key=lambda r: -r["delta_vs_baseline"])[:5]],
        "worst_layers": [r["layer"] for r in sorted(records, key=lambda r: r["delta_vs_baseline"])[:5]],
    }

    print()
    print(f"baseline (all {args.base_bits}-bit): ppl={baseline_ppl:.4f}")
    if ceiling_ppl is not None:
        print(f"ceiling  (all {args.upgrade_bits}-bit): ppl={ceiling_ppl:.4f}")
    print(f"probed layers: {len(records)}, improving: {summary['layers_improving']}")
    if summary["spread_ratio"] is not None:
        print(f"spread of positive deltas: {summary['spread_ratio']:.2f}x")
    print("layer | ppl    | delta   | % of ceiling gap")
    for record in sorted(records, key=lambda r: -r["delta_vs_baseline"]):
        share = record["fraction_of_ceiling"]
        share_text = f"{100 * share:+6.1f}%" if share is not None else "     -"
        print(f"{record['layer']:5d} | {record['ppl']:.4f} | {record['delta_vs_baseline']:+.4f} | {share_text}")

    payload = {
        "metric": "continuation-loss (prefix prefill + teacher-forced decode)",
        "model": args.model_name_or_path,
        "model_source": args.model_source,
        "device": str(device),
        "base_bits": args.base_bits,
        "upgrade_bits": args.upgrade_bits,
        "upgrade": args.upgrade,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "protect_prefix": args.protect_prefix,
        "prefill_length": args.prefill_length,
        "decode_length": args.decode_length,
        "windows": len(segments),
        "tokens": tokens,
        "peak_mem_gb": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 3
            if device.type == "cuda" else None
        ),
        "baseline_seconds": baseline_s,
        "layers": records,
        "summary": summary,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.json_out:
        out_path = os.path.expanduser(args.json_out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
