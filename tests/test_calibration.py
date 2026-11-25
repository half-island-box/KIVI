# coding=utf-8
"""Unit tests for the M1 calibration runner (synthetic and hook paths)."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from kivi_sink.calibration import (
    EVAL_TASKS,
    KProjCapture,
    build_arg_parser,
    load_calibration_records,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Linear(8, 8, bias=False)


class _ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ToyAttention()


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer(), _ToyLayer()])


class KProjCaptureTest(unittest.TestCase):
    def test_capture_is_read_only_and_removable(self):
        torch.manual_seed(0)
        model = _ToyModel()
        module = model.layers[0].self_attn.k_proj
        inputs = torch.randn(1, 5, 8)
        inputs_before = inputs.clone()

        capture = KProjCapture(model)
        self.assertEqual(capture.num_layers, 2)
        with capture:
            output = module(inputs)
        self.assertTrue(torch.equal(inputs, inputs_before))
        self.assertIn(0, capture.captured)
        self.assertTrue(torch.equal(capture.captured[0], output.detach()))
        self.assertEqual(capture._handles, [])

        # Without hooks the output is unchanged.
        capture.clear()
        second = module(inputs)
        self.assertTrue(torch.equal(output, second))
        self.assertEqual(capture.captured, {})

    def test_unsupported_model_raises(self):
        with self.assertRaises(RuntimeError):
            KProjCapture(torch.nn.Linear(4, 4))


class SyntheticCalibrationTest(unittest.TestCase):
    def _run(self, tmp_dir: Path, extra_args=None):
        output = tmp_dir / "synthetic.json"
        argv = [
            "--synthetic",
            "--samples",
            "2",
            "--synthetic-layers",
            "3",
            "--max-length",
            "16",
            "--top-m",
            "2",
            "--n-boot",
            "8",
            "--output",
            str(output),
        ]
        if extra_args:
            argv.extend(extra_args)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return_code = main(argv)
        self.assertEqual(return_code, 0)
        return output, stdout.getvalue(), stderr.getvalue()

    def test_writes_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, stdout, _ = self._run(Path(tmp))
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".csv").is_file())

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["samples"]), 2)
            self.assertEqual(len(payload["layers"]), 3)
            self.assertEqual(payload["samples_skipped"], [])
            self.assertTrue(payload["metadata"]["model_path"] is None)
            self.assertEqual(payload["metadata"]["dtype"], "float32")

            for entry in payload["layers"]:
                self.assertIn("exact_A_mean", entry)
                self.assertIn("proxy_A_mean", entry)
                self.assertIn("exact_token_spearman_mean", entry)
                self.assertGreaterEqual(entry["exact_A_mean"], 0.0)
                self.assertLessEqual(entry["exact_A_mean"], 1.0)
            correlation = payload["correlation"]
            self.assertIn("token_spearman_mean", correlation)
            self.assertIn("layer_A_spearman_mean", correlation)
            self.assertIn("exact_layer_A_stability_spearman", correlation)

            csv_text = output.with_suffix(".csv").read_text(encoding="utf-8")
            self.assertIn("exact_A_mean", csv_text.splitlines()[0])
            self.assertEqual(len(csv_text.strip().splitlines()), 4)  # header + 3 layers
            self.assertIn("wrote", stdout)

    def test_saves_token_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "scores.npz"
            output, _, _ = self._run(Path(tmp), ["--save-token-scores", str(npz_path)])
            self.assertTrue(output.is_file())
            self.assertTrue(npz_path.is_file())
            import numpy as np

            with np.load(npz_path) as archive:
                # 2 samples * 3 layers * 2 modes
                self.assertEqual(len(archive.files), 12)
                self.assertIn("exact_sample0_layer0", archive.files)
                self.assertIn("proxy_sample1_layer2", archive.files)
                self.assertEqual(archive["exact_sample0_layer0"].shape, (16,))

    def test_rejects_model_with_synthetic(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(ValueError):
                main(["--synthetic", "--model", "some/model"])


class CalibrationDataGuardTest(unittest.TestCase):
    def _args(self, *extra):
        parser = build_arg_parser()
        return parser.parse_args(["--model", "some/model", *extra])

    def test_eval_task_is_rejected_before_loading(self):
        for task in sorted(EVAL_TASKS):
            args = self._args("--longbench-task", task)
            with self.assertRaises(ValueError):
                load_calibration_records(args)

    def test_jsonl_loading_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "first", "id": "a"}),
                        json.dumps({"context": "ctx", "input": "q"}),
                        json.dumps({"text": "third"}),
                    ]
                ),
                encoding="utf-8",
            )
            args = self._args("--calibration-jsonl", str(path), "--samples", "2")
            records = load_calibration_records(args)
            self.assertEqual([record["id"] for record in records], ["a", "cal-1"])
            self.assertEqual(records[0]["text"], "first")
            self.assertEqual(records[1]["text"], "ctx\n\nq")

    def test_jsonl_and_longbench_are_mutually_exclusive(self):
        args = self._args(
            "--calibration-jsonl",
            "cal.jsonl",
            "--longbench-task",
            "narrativeqa",
        )
        with self.assertRaises(ValueError):
            load_calibration_records(args)


class ImportSideEffectTest(unittest.TestCase):
    def test_import_does_not_change_global_state(self):
        code = """
import torch
state = torch.get_rng_state().clone()
dtype = torch.get_default_dtype()
deterministic = torch.backends.cudnn.deterministic
benchmark = torch.backends.cudnn.benchmark
import kivi_sink
import kivi_sink.attention_stats
import kivi_sink.sink_score
import kivi_sink.calibration
assert torch.equal(state, torch.get_rng_state())
assert torch.get_default_dtype() == dtype
assert torch.backends.cudnn.deterministic == deterministic
assert torch.backends.cudnn.benchmark == benchmark
print("ok")
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
