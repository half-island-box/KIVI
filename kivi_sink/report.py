# coding=utf-8
"""M1 metric report: evaluate a calibration JSON against M1 acceptance metrics.

The calibration runner produces per-layer ``A_l`` statistics and exact-vs-proxy
correlations.  This module turns that artifact into the numbers M1 actually
asks for and flags obvious problems:

- layer heterogeneity: ``A_l`` range / coefficient of variation and the excess
  over the uniform reference ``top_m / T``;
- cross-sample stability: pairwise Spearman across calibration samples and the
  bootstrap CI width;
- proxy validity: per-layer token-score Spearman and per-sample layer-score
  Spearman between exact attention and the key-norm proxy;
- sample integrity: sample count and skipped records.

Thresholds are deliberately conservative defaults and can be overridden on the
command line.  They are engineering guards, not research conclusions: a report
on synthetic data validates the tooling only.

Usage::

    python -m kivi_sink.report \
        --input results/sink/calibration/llama31_both.json \
        --json-out results/sink/calibration/llama31_both.report.json

    python -m kivi_sink.report --input docs/examples/m1_synthetic_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

__all__ = ["evaluate_calibration", "main"]

DEFAULT_MIN_TOKEN_SPEARMAN = 0.5
DEFAULT_MIN_LAYER_SPEARMAN = 0.5
DEFAULT_MIN_STABILITY = 0.7
DEFAULT_MIN_SAMPLES = 3
DEFAULT_MIN_AL_RANGE = 0.0


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite_mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if _finite(value)]
    return statistics.fmean(finite) if finite else None


def _finite_median(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if _finite(value)]
    return statistics.median(finite) if finite else None


def _uniform_reference(payload: Dict) -> Optional[float]:
    config = payload.get("config") or {}
    top_m = config.get("top_m")
    if not isinstance(top_m, int) or top_m < 1:
        return None
    token_counts = [
        sample.get("num_tokens")
        for sample in payload.get("samples") or []
        if isinstance(sample.get("num_tokens"), int) and sample["num_tokens"] > 0
    ]
    if token_counts:
        return statistics.fmean(top_m / tokens for tokens in token_counts)
    max_length = config.get("max_length")
    if isinstance(max_length, int) and max_length > 0:
        return top_m / max_length
    return None


def _summarise_layers(
    layers: Sequence[Dict],
    prefix: str,
    uniform_reference: Optional[float],
) -> Optional[Dict]:
    means: List[float] = []
    ci_widths: List[float] = []
    per_layer: List[Dict] = []
    for entry in layers:
        key = f"{prefix}_A_mean"
        if key not in entry:
            continue
        mean = entry[key]
        if not _finite(mean):
            continue
        means.append(float(mean))
        width = None
        if _finite(entry.get(f"{prefix}_A_ci_low")) and _finite(
            entry.get(f"{prefix}_A_ci_high")
        ):
            width = float(entry[f"{prefix}_A_ci_high"]) - float(entry[f"{prefix}_A_ci_low"])
            ci_widths.append(width)
        per_layer.append(
            {
                "layer": entry.get("layer"),
                f"{prefix}_A_mean": float(mean),
                **(
                    {f"{prefix}_ci_width": width}
                    if width is not None
                    else {}
                ),
            }
        )
    if not means:
        return None

    mean_over_layers = statistics.fmean(means)
    spread = max(means) - min(means)
    cv = (statistics.pstdev(means) / mean_over_layers) if mean_over_layers > 0 else None
    return {
        "num_layers": len(means),
        "mean_over_layers": mean_over_layers,
        "min": min(means),
        "max": max(means),
        "range": spread,
        "cv": cv,
        "ci_width_mean": (_finite_mean(ci_widths)),
        "uniform_reference": uniform_reference,
        "excess_over_uniform": (
            mean_over_layers - uniform_reference
            if uniform_reference is not None
            else None
        ),
        "per_layer": per_layer,
    }


def _proxy_validity(correlation: Dict) -> Optional[Dict]:
    token_per_layer = correlation.get("token_spearman_per_layer") or []
    layer_per_sample = correlation.get("layer_A_spearman_per_sample") or []
    if not token_per_layer and not layer_per_sample:
        return None
    finite_token = [float(value) for value in token_per_layer if _finite(value)]
    finite_layer = [float(value) for value in layer_per_sample if _finite(value)]
    return {
        "proxy": correlation.get("proxy"),
        "token_spearman_mean": _finite_mean(token_per_layer),
        "token_spearman_median": _finite_median(token_per_layer),
        "token_spearman_positive_layers": sum(1 for value in finite_token if value > 0),
        "token_spearman_layers": len(finite_token),
        "layer_A_spearman_mean": _finite_mean(layer_per_sample),
        "layer_A_spearman_positive_samples": sum(
            1 for value in finite_layer if value > 0
        ),
        "layer_A_spearman_samples": len(finite_layer),
    }


def _check(name: str, status: str, detail: str) -> Dict:
    return {"name": name, "status": status, "detail": detail}


def _layer_score_check(name: str, summary: Optional[Dict]) -> Dict:
    if summary is None:
        return _check(name, "skip", "no per-layer scores in this artifact")
    bad = [
        entry
        for entry in summary["per_layer"]
        if not (0.0 <= entry[f"{name.split('_')[0]}_A_mean"] <= 1.0)
    ]
    if bad:
        return _check(
            name, "warn", f"{len(bad)} layer(s) outside the [0, 1] A_l range"
        )
    return _check(
        name,
        "pass",
        f"{summary['num_layers']} layers, mean={summary['mean_over_layers']:.4f}",
    )


def evaluate_calibration(
    payload: Dict,
    *,
    source: Optional[str] = None,
    min_token_spearman: float = DEFAULT_MIN_TOKEN_SPEARMAN,
    min_layer_spearman: float = DEFAULT_MIN_LAYER_SPEARMAN,
    min_stability: float = DEFAULT_MIN_STABILITY,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_al_range: float = DEFAULT_MIN_AL_RANGE,
    max_ci_width: Optional[float] = None,
) -> Dict:
    """Compute M1 metrics and a provisional pass/warn verdict for one artifact."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a calibration JSON object")
    layers = payload.get("layers") or []
    samples = payload.get("samples") or []
    correlation = payload.get("correlation") or {}
    config = payload.get("config") or {}

    uniform_reference = _uniform_reference(payload)
    exact_summary = _summarise_layers(layers, "exact", uniform_reference)
    proxy_summary = _summarise_layers(layers, "proxy", uniform_reference)
    proxy_validity = _proxy_validity(correlation)

    stability = {
        "exact": correlation.get("exact_layer_A_stability_spearman"),
        "proxy": correlation.get("proxy_layer_A_stability_spearman"),
    }
    finite_stability = [value for value in stability.values() if _finite(value)]

    num_samples = len(samples)
    skipped = payload.get("samples_skipped") or []
    checks: List[Dict] = []

    if exact_summary is not None:
        checks.append(_layer_score_check("exact_layer_scores", exact_summary))
    else:
        checks.append(_check("exact_layer_scores", "skip", "exact mode not run"))
    if proxy_summary is not None:
        checks.append(_layer_score_check("proxy_layer_scores", proxy_summary))
    else:
        checks.append(_check("proxy_layer_scores", "skip", "proxy mode not run"))

    checks.append(
        _check(
            "sample_count",
            "pass" if num_samples >= min_samples else "warn",
            f"{num_samples} usable sample(s), {len(skipped)} skipped; "
            f"recommended minimum={min_samples}",
        )
    )

    heterogeneity = exact_summary or proxy_summary
    if heterogeneity is None:
        checks.append(_check("layer_heterogeneity", "skip", "no layer scores"))
    else:
        status = "pass" if heterogeneity["range"] >= min_al_range else "warn"
        checks.append(
            _check(
                "layer_heterogeneity",
                status,
                f"A_l range={heterogeneity['range']:.4f} "
                f"(min={heterogeneity['min']:.4f}, max={heterogeneity['max']:.4f}, "
                f"threshold={min_al_range})",
            )
        )

    if proxy_validity is None:
        checks.append(
            _check("proxy_token_spearman", "skip", "exact/proxy comparison not available")
        )
        checks.append(
            _check("proxy_layer_spearman", "skip", "exact/proxy comparison not available")
        )
    else:
        token_mean = proxy_validity["token_spearman_mean"]
        checks.append(
            _check(
                "proxy_token_spearman",
                "pass"
                if token_mean is not None and token_mean >= min_token_spearman
                else "warn",
                f"mean token Spearman={_format_optional(token_mean)} "
                f"(threshold={min_token_spearman})",
            )
        )
        layer_mean = proxy_validity["layer_A_spearman_mean"]
        checks.append(
            _check(
                "proxy_layer_spearman",
                "pass"
                if layer_mean is not None and layer_mean >= min_layer_spearman
                else "warn",
                f"mean layer-A Spearman={_format_optional(layer_mean)} "
                f"(threshold={min_layer_spearman})",
            )
        )

    if not finite_stability:
        checks.append(
            _check(
                "cross_sample_stability",
                "skip",
                "needs at least two samples with per-layer scores",
            )
        )
    else:
        best = max(finite_stability)
        checks.append(
            _check(
                "cross_sample_stability",
                "pass" if best >= min_stability else "warn",
                f"best A_l Spearman={best:.4f} (threshold={min_stability})",
            )
        )

    if max_ci_width is None:
        checks.append(
            _check("ci_width", "skip", "set --max-ci-width to enable this check")
        )
    else:
        widths = [
            summary["ci_width_mean"]
            for summary in (exact_summary, proxy_summary)
            if summary is not None and summary["ci_width_mean"] is not None
        ]
        if not widths:
            checks.append(_check("ci_width", "skip", "no bootstrap CI in artifact"))
        else:
            worst = max(widths)
            checks.append(
                _check(
                    "ci_width",
                    "pass" if worst <= max_ci_width else "warn",
                    f"max mean CI width={worst:.4f} (threshold={max_ci_width})",
                )
            )

    verdict = "warn" if any(check["status"] == "warn" for check in checks) else "pass"

    return {
        "report_version": 1,
        "source": {
            "path": source,
            "created_at": (payload.get("metadata") or {}).get("created_at"),
            "model_path": (payload.get("metadata") or {}).get("model_path"),
            "mode": config.get("mode"),
            "schema_version": payload.get("schema_version"),
        },
        "coverage": {
            "num_samples": num_samples,
            "num_samples_skipped": len(skipped),
            "num_layers": len(layers),
            "has_exact": exact_summary is not None,
            "has_proxy": proxy_summary is not None,
            "token_scores_saved": bool(config.get("save_token_scores")),
        },
        "config": {
            "top_m": config.get("top_m"),
            "max_length": config.get("max_length"),
            "query_fraction": config.get("query_fraction"),
            "n_boot": config.get("n_boot"),
            "ci": config.get("ci"),
        },
        "layer_scores": {
            "uniform_reference": uniform_reference,
            "exact": exact_summary,
            "proxy": proxy_summary,
        },
        "proxy_validity": proxy_validity,
        "stability": stability,
        "thresholds": {
            "min_token_spearman": min_token_spearman,
            "min_layer_spearman": min_layer_spearman,
            "min_stability": min_stability,
            "min_samples": min_samples,
            "min_al_range": min_al_range,
            "max_ci_width": max_ci_width,
        },
        "checks": checks,
        "verdict": verdict,
        "notes": [
            "Thresholds are provisional engineering guards, not research conclusions.",
            "Synthetic artifacts validate the metric tooling only; real-model "
            "numbers require a GPU calibration run.",
        ],
    }


def _format_optional(value) -> str:
    return f"{value:.4f}" if _finite(value) else "n/a"


def _print_report(report: Dict) -> None:
    source = report["source"]["path"] or "<in-memory>"
    print(
        f"[M1 metrics] {source} mode={report['source']['mode']} "
        f"samples={report['coverage']['num_samples']} "
        f"layers={report['coverage']['num_layers']}"
    )
    for prefix in ("exact", "proxy"):
        summary = report["layer_scores"][prefix]
        if summary is None:
            print(f"[M1 metrics] {prefix}: not collected")
            continue
        print(
            f"[M1 metrics] {prefix} A_l: mean={summary['mean_over_layers']:.4f} "
            f"range={summary['range']:.4f} "
            f"excess_over_uniform="
            f"{_format_optional(summary['excess_over_uniform'])}"
        )
    validity = report["proxy_validity"]
    if validity is not None:
        print(
            f"[M1 metrics] proxy={validity['proxy']} "
            f"token_spearman_mean="
            f"{_format_optional(validity['token_spearman_mean'])} "
            f"layer_A_spearman_mean="
            f"{_format_optional(validity['layer_A_spearman_mean'])}"
        )
    stability = {
        key: value for key, value in report["stability"].items() if _finite(value)
    }
    if stability:
        formatted = ", ".join(
            f"{key}={value:.4f}" for key, value in stability.items()
        )
        print(f"[M1 metrics] stability: {formatted}")
    for check in report["checks"]:
        print(
            f"[M1 metrics] check {check['name']}: {check['status']} "
            f"({check['detail']})"
        )
    print(f"[M1 metrics] verdict: {report['verdict'].upper()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a kivi_sink calibration JSON against M1 metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Calibration JSON path.")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for the machine-readable metric report.",
    )
    parser.add_argument(
        "--min-token-spearman",
        type=float,
        default=DEFAULT_MIN_TOKEN_SPEARMAN,
        help="Minimum mean per-layer token Spearman for exact vs proxy.",
    )
    parser.add_argument(
        "--min-layer-spearman",
        type=float,
        default=DEFAULT_MIN_LAYER_SPEARMAN,
        help="Minimum mean per-sample layer-A Spearman for exact vs proxy.",
    )
    parser.add_argument(
        "--min-stability",
        type=float,
        default=DEFAULT_MIN_STABILITY,
        help="Minimum cross-sample A_l Spearman.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Recommended minimum number of calibration samples.",
    )
    parser.add_argument(
        "--min-al-range",
        type=float,
        default=DEFAULT_MIN_AL_RANGE,
        help="Minimum max-min spread of A_l across layers.",
    )
    parser.add_argument(
        "--max-ci-width",
        type=float,
        default=None,
        help="Optional maximum mean bootstrap CI width for A_l.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = Path(args.input)
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = evaluate_calibration(
        payload,
        source=str(path),
        min_token_spearman=args.min_token_spearman,
        min_layer_spearman=args.min_layer_spearman,
        min_stability=args.min_stability,
        min_samples=args.min_samples,
        min_al_range=args.min_al_range,
        max_ci_width=args.max_ci_width,
    )
    _print_report(report)
    if args.json_out:
        destination = Path(args.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[M1 metrics] wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
