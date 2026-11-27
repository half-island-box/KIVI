# coding=utf-8
"""WikiText-2 perplexity for the KIVI KV cache, measured against FP16.

Why this exists
---------------
LongBench's F1 / ROUGE / accuracy are coarse: on the 4k-truncated 8-task set a
2-bit KV cache loses only ~0.6 points on average, which is the same order as
the scoring noise.  Perplexity is a continuous per-token loss and picks up
quantization error that discrete task scores dilute.  It is reported as a
**delta against the FP16 run**, never as an absolute number: an instruct model
continuing raw WikiText produces a meaningless absolute PPL, only the gap
between two quantization settings is informative.

Measurement contract
--------------------
- The corpus is tokenized once and cut into non-overlapping windows of
  ``--max_length`` (default 4096, matching ``model2maxlen`` for LongBench on
  this machine).
- Every window is a fresh prefill (``past_key_values=None``, ``use_cache=True``)
  on an empty cache, so the run exercises exactly the KIVI prefill path --
  flash attention + quantized segments + the FP residual window -- that M3 is
  meant to evaluate.  Decode-side quantization loss is out of scope here; that
  belongs to the generation tasks in M4.
- Only the first token of each window is excluded from the loss.  The
  convention is identical for FP16 and KIVI, so the comparison stays fair.
- Logits are consumed in ``--logit_chunk`` position slices and upcast to fp32
  per slice.  Materializing ``T x vocab`` logits in one go is exactly what
  forced LongBench down to 4k, so do not remove the chunking.

Data resolution order (mirrors ``utils/modelscope_hub.py``)
-----------------------------------------------------------
1. ``--data_file`` or ``$WIKITEXT_DATA_FILE`` -- a local ``.txt`` or ``.jsonl``
   with a ``text`` field.  Always works, no network.
2. ``<KIVI_DATA_DIR>/datasets/wikitext/<subset>_<split>.txt`` -- a cached plain
   text file from an earlier run.
3. A HuggingFace dataset script mirrored by ModelScope (``modelscope/wikitext``
   ships ``wikitext.py`` but no corpus): ``datasets.load_dataset`` is pointed at
   that local directory, which downloads the split from the URL the script
   hardcodes.  The result is cached to (2) so later runs stay offline.

Step 3 deliberately avoids ``modelscope.msdatasets``: its
``hf_datasets_util.py`` does ``from datasets import LargeList``, which fails on
``datasets==2.16.1`` (the version this repo pins) with
``ImportError: cannot import name 'LargeList'``.  ``datasets.load_dataset`` on a
local script directory has no such problem.

Note on provenance: the corpus itself comes from the upstream URL hardcoded in
that script (``s3.amazonaws.com/research.metamind.io``); ModelScope only mirrors
the loader.  That is the one unavoidable exception to the "datasets via
ModelScope only" rule, and should be recorded wherever the environment is.

Examples
--------
FP16 reference (run this first)::

    python ppl_eval.py \
        --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
        --device 0 --k_bits 16 --v_bits 16 \
        --json_out results/sink/ppl/llama31_fp16_4k.json

KIVI-2bit, reported as a delta against that reference::

    python ppl_eval.py \
        --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
        --device 0 --k_bits 2 --v_bits 2 \
        --reference results/sink/ppl/llama31_fp16_4k.json \
        --json_out results/sink/ppl/llama31_kivi2_4k.json

Wrapped so the timing and peak memory land in ``results/``::

    python scripts/m0_baseline.py run --name llama31_kivi2_ppl --repeat 1 -- \
        python ppl_eval.py --model_name_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
            --device 0 --k_bits 2 --v_bits 2
"""

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

DEFAULT_DATASET = "modelscope/wikitext"
DEFAULT_SUBSET = "wikitext-2-raw-v1"
DEFAULT_SPLIT = "test"
DATA_FILE_ENV = "WIKITEXT_DATA_FILE"


def _log(message):
    print(f"[ppl] {message}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="WikiText-2 perplexity for KIVI vs FP16 (prefill path only)."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    # Shared flags; both helpers live in utils/ and import torch transitively.
    from utils.modelscope_hub import add_model_source_argument
    from utils.device_utils import add_device_argument

    add_model_source_argument(parser)
    add_device_argument(parser)
    parser.add_argument("--k_bits", type=int, default=2, choices=(2, 4, 8, 16),
                        help="KV cache bits; 16 on both sides measures FP16")
    parser.add_argument("--v_bits", type=int, default=2, choices=(2, 4, 8, 16))
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--residual_length", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=4096,
                        help="window length in tokens (default 4096, matching LongBench)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="windows per forward; >1 is not validated for the KIVI path")
    parser.add_argument("--logit_chunk", type=int, default=512,
                        help="positions per logits slice; caps the fp32 upcast")
    parser.add_argument("--max_windows", type=int, default=0,
                        help="cap the number of windows (0 = all)")
    parser.add_argument("--data_file", type=str, default=None,
                        help="local .txt, or .jsonl with a 'text' field")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="ModelScope dataset repo id; used only in the download hint")
    parser.add_argument("--subset", type=str, default=DEFAULT_SUBSET)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--reference", type=str, default=None,
                        help="a previous ppl_eval JSON, normally the FP16 run")
    parser.add_argument("--json_out", type=str, default=None)
    return parser


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _wikitext_cache_dir(cache_dir=None):
    from utils.modelscope_hub import default_data_root

    root = os.path.expanduser(cache_dir) if cache_dir else default_data_root()
    return os.path.join(root, "datasets", "wikitext")


def _read_text(path):
    if path.endswith(".jsonl"):
        lines = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                lines.append(str(record.get("text", "")))
        return "\n".join(lines)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _find_dataset_script(cache_dir):
    """Return the mirrored ``wikitext.py`` loader path, if present.

    ``modelscope download --dataset modelscope/wikitext`` puts ``wikitext.py``,
    ``dataset_infos.json`` and ``README.md`` there -- but no corpus, which the
    script fetches itself at load time.
    """
    for candidate in (
        os.path.join(cache_dir, "wikitext.py"),
        os.path.join(cache_dir, "wikitext", "wikitext.py"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_via_dataset_script(script_path, subset, split):
    """Run ``datasets.load_dataset`` against a local dataset script."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise SystemExit(
            f"datasets is not importable ({type(exc).__name__}: {exc}).\n"
            "Pass --data_file <test.txt|test.jsonl> to use a local corpus."
        ) from exc

    _log(f"loading {subset}/{split} through {script_path}")
    try:
        dataset_obj = load_dataset(script_path, subset, split=split, trust_remote_code=True)
    except TypeError:
        # Older datasets releases do not accept trust_remote_code here.
        dataset_obj = load_dataset(script_path, subset, split=split)

    try:
        texts = dataset_obj["text"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(
            f"the dataset script produced no 'text' column ({type(exc).__name__}: {exc})"
        ) from exc
    return "\n".join(str(value) for value in texts)


def load_corpus(args):
    explicit = args.data_file or os.environ.get(DATA_FILE_ENV)
    if explicit:
        path = os.path.expanduser(explicit)
        if not os.path.isfile(path):
            raise SystemExit(f"--data_file {path!r} does not exist")
        _log(f"corpus: {path}")
        return _read_text(path)

    cache_dir = _wikitext_cache_dir(args.cache_dir)
    cached = os.path.join(cache_dir, f"{args.subset}_{args.split}.txt")
    if os.path.isfile(cached):
        _log(f"corpus: {cached} (cached)")
        return _read_text(cached)

    script_path = _find_dataset_script(cache_dir)
    if script_path is None:
        raise SystemExit(
            "no WikiText corpus found.\n"
            f"  expected a cached text file at {cached}\n"
            f"  expected a dataset script under {cache_dir}\n"
            "Fetch the ModelScope mirror first (the script does the rest):\n"
            f"    modelscope download --dataset {args.dataset} --local_dir {cache_dir}\n"
            "or download the raw split by hand and pass --data_file:\n"
            "    curl -L -O "
            "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip\n"
            "    unzip wikitext-2-raw-v1.zip   # -> wikitext-2-raw-v1/test.txt"
        )

    text = _load_via_dataset_script(script_path, args.subset, args.split)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cached, "w", encoding="utf-8") as handle:
        handle.write(text)
    _log(f"corpus: {cached} (loaded through the dataset script, cached for offline runs)")
    return text


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def load_model(args, model_path, device):
    import torch
    from transformers import LlamaConfig

    config = LlamaConfig.from_pretrained(model_path)
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    # The KIVI attention path asserts use_flash (models/llama_kivi.py).
    config.use_flash = True

    # Keep the whole model on one device: a PPL number measured across a
    # sharded model is not comparable with the LongBench runs.
    device_map = {"": str(device)}
    common = dict(
        pretrained_model_name_or_path=model_path,
        config=config,
        cache_dir=args.cache_dir or "./cached_models",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    if args.k_bits < 16 and args.v_bits < 16:
        from models.llama_kivi import LlamaForCausalLM_KIVI

        model = LlamaForCausalLM_KIVI.from_pretrained(**common)
    elif args.k_bits == 16 and args.v_bits == 16:
        from transformers import LlamaForCausalLM

        if device.type == "cuda":
            common["use_flash_attention_2"] = True
        model = LlamaForCausalLM.from_pretrained(**common)
    else:
        raise SystemExit(
            "k_bits and v_bits must both be < 16 (KIVI) or both be 16 (FP16); "
            f"got k_bits={args.k_bits}, v_bits={args.v_bits}"
        )

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# perplexity
# ---------------------------------------------------------------------------


def tokenize_corpus(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    return ids


def window_starts(num_tokens, window, max_windows=0):
    starts = list(range(0, num_tokens - window + 1, window))
    if max_windows:
        starts = starts[:max_windows]
    return starts


def window_nll(model, input_ids, logit_chunk):
    """Summed NLL over one batch of equal-length windows."""
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits
    length = input_ids.shape[1]
    total = 0.0
    for start in range(0, length - 1, logit_chunk):
        stop = min(start + logit_chunk, length - 1)
        chunk = logits[:, start:stop].float()
        targets = input_ids[:, start + 1:stop + 1]
        total += F.cross_entropy(chunk.transpose(1, 2), targets, reduction="sum").item()
    del logits, outputs
    return total


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.logit_chunk < 1:
        parser.error("--logit_chunk must be >= 1")

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

    text = load_corpus(args)
    ids = tokenize_corpus(tokenizer, text)
    starts = window_starts(len(ids), args.max_length, args.max_windows)
    if not starts:
        raise SystemExit(
            f"corpus has {len(ids)} tokens, fewer than one {args.max_length}-token window"
        )
    _log(
        f"corpus tokens: {len(ids)}, windows: {len(starts)} "
        f"of {args.max_length} tokens, k_bits={args.k_bits}, v_bits={args.v_bits}"
    )

    model = load_model(args, model_path, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for index in range(0, len(starts), args.batch_size):
            batch = starts[index:index + args.batch_size]
            input_ids = torch.tensor(
                [ids[start:start + args.max_length] for start in batch],
                dtype=torch.long,
                device=device,
            )
            total_nll += window_nll(model, input_ids, args.logit_chunk)
            total_tokens += input_ids.shape[0] * (input_ids.shape[1] - 1)
            done = index + len(batch)
            if done % max(len(starts) // 10, 1) == 0 or done == len(starts):
                _log(f"windows {done}/{len(starts)}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(mean_nll)
    peak_gb = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 3
        if device.type == "cuda"
        else None
    )

    print(f"windows: {len(starts)}, tokens: {total_tokens}")
    print(f"mean nll: {mean_nll:.6f}")
    print(f"used time: {elapsed_s * 1000:.1f} ms")
    if peak_gb is not None:
        print(f"peak mem: {peak_gb:.2f} GB")
    print(f"ppl: {ppl:.4f}", flush=True)

    reference = None
    if args.reference:
        with open(args.reference, encoding="utf-8") as handle:
            ref = json.load(handle)
        ref_ppl = float(ref["ppl"])
        reference = {"path": args.reference, "ppl": ref_ppl}
        delta = ppl - ref_ppl
        print(f"reference: {args.reference}")
        print(f"reference ppl: {ref_ppl:.4f}")
        print(f"delta: {delta:+.4f} ({100.0 * (ppl / ref_ppl - 1.0):+.2f}%)")

    record = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "data_file": args.data_file,
        "model": args.model_name_or_path,
        "model_source": args.model_source,
        "k_bits": args.k_bits,
        "v_bits": args.v_bits,
        "group_size": args.group_size,
        "residual_length": args.residual_length,
        "max_length": args.max_length,
        "windows": len(starts),
        "tokens": total_tokens,
        "nll_sum": total_nll,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "dtype": "float16",
        "device": str(device),
        "seconds": elapsed_s,
        "peak_mem_gb": peak_gb,
        "reference": reference,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if args.json_out:
        out_path = os.path.expanduser(args.json_out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
