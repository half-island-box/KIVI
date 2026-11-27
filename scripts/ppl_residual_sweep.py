# coding=utf-8
"""Sweep KIVI's residual window and report how much PPL each size buys.

Why this exists
---------------
``residual_length`` is KIVI's own "keep the newest R tokens in FP" budget, and
it is exactly the lever M4 (decode-time delayed quantization) would tune.  Two
questions decide whether that direction has room:

1. How much of the 2-bit loss can *recency* recover at all -- i.e. how steep is
   the PPL-vs-R curve?
2. Where does the curve flatten?  If PPL only drops once R approaches the whole
   sequence, the damage is diffuse across decode tokens and a token-level
   dynamic scheme has nothing concentrated to target.

The endpoint R = prefill + decode keeps every token in FP, which must reproduce
the FP16 run -- a built-in correctness check for the whole measurement.

Usage::

    python scripts/ppl_residual_sweep.py \
        --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
        --device 7 --json_out results/sink/ppl/residual_sweep.json

Each R is a separate ``ppl_eval.py`` process, so a failure is isolated and the
GPU memory is released between runs.
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "sink", "ppl")


def _log(message):
    print(f"[sweep] {message}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep residual_length on the decode-based PPL measurement."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model_source", type=str, default="modelscope",
                        choices=("modelscope", "hf"))
    parser.add_argument("--k_bits", type=int, default=2, choices=(2, 4, 8))
    parser.add_argument("--v_bits", type=int, default=2, choices=(2, 4, 8))
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--prefill_length", type=int, default=512)
    parser.add_argument("--decode_length", type=int, default=2048)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--values", type=str, default="32,128,256,512,1024",
                        help="comma separated residual_length values; the all-FP "
                             "endpoint (prefill+decode) is appended automatically")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="re-run values whose JSON already exists")
    parser.add_argument("--name_prefix", type=str, default=None,
                        help="defaults to llama31_kivi{k}_R")
    parser.add_argument("--json_out", type=str, default=None,
                        help="where to write the combined summary")
    return parser


def parse_values(spec, span, group_size):
    values = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value % group_size:
            raise SystemExit(
                f"residual_length {value} is not a multiple of group_size {group_size}"
            )
        values.append(value)
    # The all-FP endpoint: every token stays in the FP window.
    if span not in values:
        values.append(span)
    return sorted(set(values))


def run_one(args, residual_length, out_path):
    command = [
        sys.executable, "ppl_eval.py",
        "--model_name_or_path", args.model_name_or_path,
        "--model_source", args.model_source,
        "--k_bits", str(args.k_bits),
        "--v_bits", str(args.v_bits),
        "--group_size", str(args.group_size),
        "--residual_length", str(residual_length),
        "--prefill_length", str(args.prefill_length),
        "--decode_length", str(args.decode_length),
        "--windows", str(args.windows),
        "--json_out", out_path,
    ]
    if args.cache_dir:
        command += ["--cache_dir", args.cache_dir]
    if args.device:
        command += ["--device", args.device]
    if args.data_file:
        command += ["--data_file", args.data_file]

    _log(f"R={residual_length}: {' '.join(command[1:])}")
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    span = args.prefill_length + args.decode_length
    values = parse_values(args.values, span, args.group_size)
    prefix = args.name_prefix or f"llama31_kivi{args.k_bits}_R"
    os.makedirs(args.out_dir, exist_ok=True)

    _log(f"sweeping residual_length over {values} (all-FP endpoint = {span})")
    for residual_length in values:
        out_path = os.path.join(args.out_dir, f"{prefix}{residual_length}.json")
        if os.path.isfile(out_path) and not args.force:
            _log(f"R={residual_length}: reusing {out_path}")
            continue
        code = run_one(args, residual_length, out_path)
        if code != 0:
            _log(f"R={residual_length}: FAILED (exit {code}); continuing")

    rows = []
    for residual_length in values:
        out_path = os.path.join(args.out_dir, f"{prefix}{residual_length}.json")
        record = None
        if os.path.isfile(out_path):
            with open(out_path, encoding="utf-8") as handle:
                record = json.load(handle)
        rows.append((residual_length, record))

    ceiling_row = next((r for r in rows if r[0] == span and r[1]), None)
    base_row = next((r for r in rows if r[0] == 128 and r[1]), None)
    ceiling_ppl = ceiling_row[1]["ppl"] if ceiling_row else None
    base_ppl = base_row[1]["ppl"] if base_row else None
    ceiling_gap = (base_ppl - ceiling_ppl) if (base_ppl and ceiling_ppl) else None

    print()
    print("residual | ppl      | vs R=128 | recovered of the recency gap | seconds")
    for residual_length, record in rows:
        if record is None:
            print(f"{residual_length:8d} | FAILED")
            continue
        ppl = record["ppl"]
        vs_base = f"{ppl - base_ppl:+.4f}" if base_ppl else "     -"
        recovered = "-"
        if ceiling_gap and base_ppl:
            recovered = f"{100 * (base_ppl - ppl) / ceiling_gap:+6.1f}%"
        print(f"{residual_length:8d} | {ppl:8.4f} | {vs_base:>8s} | {recovered:>28s} | "
              f"{record.get('seconds', 0):6.0f}")

    if args.json_out:
        out_path = os.path.abspath(os.path.expanduser(args.json_out))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        payload = {
            "model": args.model_name_or_path,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "group_size": args.group_size,
            "prefill_length": args.prefill_length,
            "decode_length": args.decode_length,
            "windows": args.windows,
            "base_residual_length": 128,
            "base_ppl": base_ppl,
            "all_fp_residual_length": span,
            "all_fp_ppl": ceiling_ppl,
            "recency_gap": ceiling_gap,
            "points": [
                {"residual_length": r, **(record or {"ppl": None})}
                for r, record in rows
            ],
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
