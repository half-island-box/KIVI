# coding=utf-8
"""Unit tests for kivi_sink.report (M1 metric evaluation)."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from kivi_sink.report import evaluate_calibration, main


def _layer_entry(index, exact=None, proxy=None):
    entry = {"layer": index}
    if exact is not None:
        entry.update(
            {
                "exact_A_mean": exact,
                "exact_A_median": exact,
                "exact_A_std": 0.01,
                "exact_A_ci_low": exact - 0.01,
                "exact_A_ci_high": exact + 0.01,
                "exact_token_spearman_mean": 0.8,
            }
        )
    if proxy is not None:
        entry.update(
            {
                "proxy_A_mean": proxy,
                "proxy_A_median": proxy,
                "proxy_A_std": 0.01,
                "proxy_A_ci_low": proxy - 0.01,
                "proxy_A_ci_high": proxy + 0.01,
            }
        )
    return entry


def _payload(
    exact_means,
    proxy_means=None,
    token_spearman=None,
    layer_spearman=None,
    stability=(0.9, 0.85),
    n_samples=3,
    skipped=0,
    top_m=2,
    max_length=64,
):
    layers = []
    for index, exact in enumerate(exact_means):
        proxy = proxy_means[index] if proxy_means is not None else None
        layers.append(_layer_entry(index, exact=exact, proxy=proxy))
    samples = [
        {"id": f"s{index}", "num_tokens": max_length}
        for index in range(n_samples)
    ]
    correlation = {}
    if token_spearman is not None:
        correlation["token_spearman_per_layer"] = token_spearman
    if layer_spearman is not None:
        correlation["layer_A_spearman_per_sample"] = layer_spearman
    if proxy_means is not None:
        correlation["proxy"] = "key_norm"
        correlation["exact_layer_A_stability_spearman"] = stability[0]
        correlation["proxy_layer_A_stability_spearman"] = stability[1]
    return {
        "schema_version": 1,
        "metadata": {"created_at": "2026-01-01T00:00:00+00:00", "model_path": "m"},
        "config": {
            "mode": "both" if proxy_means is not None else "exact",
            "top_m": top_m,
            "max_length": max_length,
            "query_fraction": 1.0,
            "n_boot": 100,
            "ci": 0.95,
            "save_token_scores": None,
        },
        "samples": samples,
        "samples_skipped": [f"skip-{index}" for index in range(skipped)],
        "layers": layers,
        "correlation": correlation,
    }


class EvaluateCalibrationTest(unittest.TestCase):
    def test_passing_payload(self):
        payload = _payload(
            exact_means=[0.20, 0.15, 0.10, 0.12],
            proxy_means=[0.18, 0.16, 0.11, 0.13],
            token_spearman=[0.8, 0.7, 0.9, 0.6],
            layer_spearman=[0.8, 0.75, 0.7],
        )
        report = evaluate_calibration(payload, source="artifact.json")
        self.assertEqual(report["verdict"], "pass")
        self.assertTrue(report["coverage"]["has_exact"])
        self.assertTrue(report["coverage"]["has_proxy"])

        exact = report["layer_scores"]["exact"]
        self.assertAlmostEqual(exact["mean_over_layers"], 0.1425, places=6)
        self.assertAlmostEqual(exact["range"], 0.10, places=6)
        self.assertAlmostEqual(report["layer_scores"]["uniform_reference"], 0.03125)
        self.assertAlmostEqual(exact["excess_over_uniform"], 0.11125, places=6)
        self.assertAlmostEqual(exact["ci_width_mean"], 0.02, places=6)

        validity = report["proxy_validity"]
        self.assertAlmostEqual(validity["token_spearman_mean"], 0.75, places=6)
        self.assertAlmostEqual(validity["layer_A_spearman_mean"], 0.75, places=6)
        self.assertEqual(validity["token_spearman_positive_layers"], 4)
        self.assertEqual(validity["layer_A_spearman_positive_samples"], 3)

        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["exact_layer_scores"], "pass")
        self.assertEqual(statuses["proxy_layer_scores"], "pass")
        self.assertEqual(statuses["sample_count"], "pass")
        self.assertEqual(statuses["proxy_token_spearman"], "pass")
        self.assertEqual(statuses["proxy_layer_spearman"], "pass")
        self.assertEqual(statuses["cross_sample_stability"], "pass")
        self.assertEqual(statuses["ci_width"], "skip")

    def test_low_proxy_correlation_warns(self):
        payload = _payload(
            exact_means=[0.2, 0.15, 0.1],
            proxy_means=[0.2, 0.15, 0.1],
            token_spearman=[0.1, -0.2, 0.3],
            layer_spearman=[0.2, -0.1, 0.0],
        )
        report = evaluate_calibration(payload)
        self.assertEqual(report["verdict"], "warn")
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["proxy_token_spearman"], "warn")
        self.assertEqual(statuses["proxy_layer_spearman"], "warn")

    def test_exact_only_skips_proxy_checks(self):
        payload = _payload(exact_means=[0.2, 0.1, 0.15])
        report = evaluate_calibration(payload)
        self.assertIsNone(report["proxy_validity"])
        self.assertFalse(report["coverage"]["has_proxy"])
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["proxy_token_spearman"], "skip")
        self.assertEqual(statuses["proxy_layer_spearman"], "skip")
        self.assertEqual(statuses["exact_layer_scores"], "pass")
        self.assertEqual(report["verdict"], "pass")

    def test_too_few_samples_warns(self):
        payload = _payload(exact_means=[0.2, 0.1], n_samples=2)
        report = evaluate_calibration(payload, min_samples=3)
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["sample_count"], "warn")
        self.assertEqual(report["verdict"], "warn")

    def test_skipped_samples_are_reported(self):
        payload = _payload(exact_means=[0.2, 0.1, 0.15], n_samples=3, skipped=2)
        report = evaluate_calibration(payload)
        self.assertEqual(report["coverage"]["num_samples_skipped"], 2)

    def test_al_range_threshold(self):
        payload = _payload(exact_means=[0.1001, 0.1002])
        report = evaluate_calibration(payload, min_al_range=0.01)
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["layer_heterogeneity"], "warn")

    def test_ci_width_threshold(self):
        payload = _payload(exact_means=[0.2, 0.1, 0.15])
        report = evaluate_calibration(payload, max_ci_width=0.005)
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["ci_width"], "warn")

    def test_nan_metrics_are_ignored(self):
        payload = _payload(
            exact_means=[0.2, 0.1, 0.15],
            proxy_means=[0.2, 0.1, 0.15],
            token_spearman=[float("nan"), 0.6, 0.8],
            layer_spearman=[float("nan"), 0.7],
            stability=(float("nan"), 0.8),
        )
        report = evaluate_calibration(payload)
        validity = report["proxy_validity"]
        self.assertAlmostEqual(validity["token_spearman_mean"], 0.7, places=6)
        self.assertEqual(validity["token_spearman_layers"], 2)
        self.assertAlmostEqual(validity["layer_A_spearman_mean"], 0.7, places=6)
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["cross_sample_stability"], "pass")

    def test_layer_scores_out_of_range_warn(self):
        payload = _payload(exact_means=[1.5, 0.1])
        report = evaluate_calibration(payload)
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["exact_layer_scores"], "warn")

    def test_uniform_reference_falls_back_to_max_length(self):
        payload = _payload(exact_means=[0.2, 0.1], n_samples=0, max_length=32)
        report = evaluate_calibration(payload)
        self.assertAlmostEqual(report["layer_scores"]["uniform_reference"], 2 / 32)

    def test_invalid_payload(self):
        with self.assertRaises(TypeError):
            evaluate_calibration([])


class ReportCliTest(unittest.TestCase):
    def test_writes_json_report(self):
        payload = _payload(
            exact_means=[0.2, 0.15, 0.1],
            proxy_means=[0.18, 0.16, 0.12],
            token_spearman=[0.8, 0.7, 0.9],
            layer_spearman=[0.8, 0.7, 0.75],
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cal.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            destination = Path(tmp) / "report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = main(
                    ["--input", str(source), "--json-out", str(destination)]
                )
            self.assertEqual(return_code, 0)
            self.assertTrue(destination.is_file())
            report = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(report["verdict"], "pass")
            self.assertIn("verdict: PASS", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
