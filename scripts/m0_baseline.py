#!/usr/bin/env python3
# coding=utf-8
"""Dependency-light M0 baseline recorder for SinkBudget.

This script supports the M0 acceptance items in
``docs/kv_cache_sink_protection_roadmap.md``:

- ``env``   record GPU / CUDA / PyTorch / Transformers / flash-attn / Triton
            versions plus git commit, branch and dirty state;
- ``check`` validate the fixed KIVI quantization configuration without a GPU;
- ``run``   run a command repeatedly and persist every attempt (successes and
            failures) as JSON, so baseline repeats are auditable.

Only the Python standard library is imported at module load, so ``check`` and
``env`` work on a CPU-only machine where the project dependencies are missing.

Examples
--------
Validate the fixed M0 KIVI-2bit configuration::

    python scripts/m0_baseline.py check \
        --k-bits 2 --v-bits 2 --group-size 32 --residual-length 128

Record the environment on the GPU host::

    python scripts/m0_baseline.py env --out results/m0_baselines/env.json

Run a LongBench smoke three times, persisting failures::

    python scripts/m0_baseline.py run --name longbench_llama2_kivi2 --repeat 3 -- \
        bash scripts/long_test.sh 0 2 2 32 128 modelscope/Llama-2-7b-ms

Raw records land under ``results/m0_baselines/`` (ignored by git); summarize
the runs into ``docs/m0_llama_baseline.md`` before committing.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "m0_baselines"

KIVI_BITS = (2, 4, 8, 16)
PEAK_MEM_RE = re.compile(r"peak mem:\s*([0-9]*\.?[0-9]+)\s*GB", re.IGNORECASE)
USED_TIME_RE = re.compile(r"used time:\s*([0-9]*\.?[0-9]+)\s*ms", re.IGNORECASE)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(args):
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


def _git_metadata():
    commit = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _git(["status", "--porcelain"])
    dirty_files = porcelain.splitlines() if porcelain else []
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:50],
        "dirty_file_count": len(dirty_files),
    }


def _package_version(name):
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - very old Python
        return None


def _torch_info():
    try:
        import torch
    except Exception as exc:
        return {"import_error": f"{type(exc).__name__}: {exc}"}

    info = {
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                }
            )
    return info


def _nvidia_smi_info():
    query = "name,driver_version,memory.total"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or "nvidia-smi failed"}
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines or None


def collect_env():
    """Collect the environment metadata required by M0.3."""
    return {
        "recorded_at": _utc_now(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "git": _git_metadata(),
        "software": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "flash-attn",
                "triton",
                "accelerate",
                "datasets",
                "modelscope",
                "tokenizers",
                "sentencepiece",
            )
        },
        "torch": _torch_info(),
        "nvidia_smi": _nvidia_smi_info(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "HF_ENDPOINT",
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "MODELSCOPE_CACHE",
            )
            if os.environ.get(key) is not None
        },
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "run"


def _truncate(text, limit):
    if not limit or text is None or len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    omitted = len(text) - 2 * half
    return (
        text[:half]
        + f"\n...[truncated {omitted} chars by m0_baseline.py]...\n"
        + text[-half:]
    )


def cmd_check(args):
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("k_bits_valid", args.k_bits in KIVI_BITS, f"k_bits={args.k_bits}, allowed={KIVI_BITS}")
    add("v_bits_valid", args.v_bits in KIVI_BITS, f"v_bits={args.v_bits}, allowed={KIVI_BITS}")
    add("group_size_positive", args.group_size > 0, f"group_size={args.group_size}")
    add(
        "residual_length_positive",
        args.residual_length > 0,
        f"residual_length={args.residual_length}",
    )
    add(
        "residual_multiple_of_group",
        args.group_size > 0 and args.residual_length % args.group_size == 0,
        f"residual_length={args.residual_length}, group_size={args.group_size}, "
        f"remainder={args.residual_length % args.group_size if args.group_size else 'n/a'}",
    )
    add(
        "residual_at_least_group",
        args.group_size > 0 and args.residual_length >= args.group_size,
        f"residual_length={args.residual_length}, group_size={args.group_size}",
    )

    kivi_path = args.k_bits < 16 and args.v_bits < 16
    payload = {
        "recorded_at": _utc_now(),
        "config": {
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "group_size": args.group_size,
            "residual_length": args.residual_length,
        },
        "kivi_path": kivi_path,
        "use_flash_required": kivi_path,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def cmd_env(args):
    payload = collect_env()
    if args.out:
        _write_json(Path(args.out), payload)
        print(f"wrote environment record to {args.out}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args):
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("error: no command given (use `-- <command>`)", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    _write_json(out_dir / "env.json", env)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe = _safe_name(args.name)
    manifest_path = out_dir / "manifest.jsonl"

    records = []
    for index in range(args.repeat):
        started_at = _utc_now()
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
            spawn_error = None
        except OSError as exc:
            returncode = None
            stdout = ""
            stderr = ""
            spawn_error = f"{type(exc).__name__}: {exc}"
        duration = time.perf_counter() - started

        peak_mem = PEAK_MEM_RE.search(stdout or "")
        used_time = USED_TIME_RE.search(stdout or "")
        record = {
            "name": args.name,
            "repeat_index": index + 1,
            "repeat_total": args.repeat,
            "command": command,
            "cwd": str(REPO_ROOT),
            "started_at": started_at,
            "duration_sec": round(duration, 3),
            "returncode": returncode,
            "success": returncode == 0,
            "spawn_error": spawn_error,
            "peak_mem_gb": float(peak_mem.group(1)) if peak_mem else None,
            "used_time_ms": float(used_time.group(1)) if used_time else None,
            "stdout": _truncate(stdout, args.max_output_chars),
            "stderr": _truncate(stderr, args.max_output_chars),
            "git": env["git"],
            "software": env["software"],
            "torch": env["torch"],
            "nvidia_smi": env["nvidia_smi"],
        }

        run_path = out_dir / f"{stamp}_{safe}_run{index + 1}.json"
        _write_json(run_path, record)
        record["record_path"] = str(run_path)
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "ok" if record["success"] else "failed"
        print(
            f"[{index + 1}/{args.repeat}] {status} "
            f"returncode={returncode} duration={record['duration_sec']}s -> {run_path}"
        )
        if not record["success"]:
            tail = (stderr or stdout or spawn_error or "").strip().splitlines()
            if tail:
                print(f"    last stderr/stdout line: {tail[-1][:300]}")
        records.append(record)

    ok_count = sum(1 for item in records if item["success"])
    print(f"summary: {ok_count}/{len(records)} runs succeeded; records in {out_dir}")
    return 0 if ok_count == len(records) else 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="M0 baseline recorder (environment, config check, repeat runs)."
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    check = sub.add_parser("check", help="validate the fixed KIVI quantization config")
    check.add_argument("--k-bits", type=int, default=2)
    check.add_argument("--v-bits", type=int, default=2)
    check.add_argument("--group-size", type=int, default=32)
    check.add_argument("--residual-length", type=int, default=128)
    check.set_defaults(func=cmd_check)

    env = sub.add_parser("env", help="record environment / versions / git state")
    env.add_argument("--out", type=str, default=None, help="write JSON to this path")
    env.set_defaults(func=cmd_env)

    run = sub.add_parser("run", help="run a command N times and persist every attempt")
    run.add_argument("--name", type=str, required=True, help="label for the run group")
    run.add_argument("--repeat", type=int, default=1, help="number of repetitions")
    run.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"record directory (default: {DEFAULT_OUT_DIR})",
    )
    run.add_argument(
        "--max-output-chars",
        type=int,
        default=200000,
        help="truncate captured stdout/stderr to this many chars (0 disables)",
    )
    run.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run after `--`",
    )
    run.set_defaults(func=cmd_run)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "repeat", 1) < 1:
        parser.error("--repeat must be >= 1")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
