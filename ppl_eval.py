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
The loss must be measured where the quantized cache is actually *read*.

KIVI's prefill attention runs on the freshly projected full-precision K/V
(``models/llama_kivi.py``: ``attn_output = self._flash_attention_forward(...)``
happens before the tensors are packed into the cache).  The packed cache is only
consumed on the following steps.  Therefore a plain "one prefill per window,
cache discarded" perplexity is bit-identical for 2-bit, 4-bit and FP16 -- it
never touches the quantized data.  Any prefill-only metric (logit deltas,
hidden-state deltas) has the same blind spot.

So each window is measured as:

1. one prefill over ``--prefill_length`` tokens (``past_key_values=None``),
   scoring the token that follows the prefix;
2. ``--decode_length - 1`` teacher-forced single-token steps: the true next
   token is fed back with the cache attached, so every step reads the quantized
   history and scores the token after it.

Windows are spread evenly over the corpus, ``--windows`` of them, and are
independent (one cache per window).  The reported number is therefore a
"continuation loss after a 512-token prefix", not the standard WikiText PPL --
it is comparable across quantization settings, not with published PPL tables.

Chunked prefill (feeding a whole chunk *with* a past cache) is not an option
here: the KIVI decode branch asserts ``value_full_length == residual_length + 1``
(``models/llama_kivi.py``), i.e. it only supports one new token per step.

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
        description=(
            "WikiText-2 continuation loss for KIVI vs FP16, measured on teacher-forced "
            "decode steps so the quantized cache is actually read."
        )
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
    parser.add_argument("--protect_prefix", type=int, default=0,
                        help="keep the first N positions in FP16 in every layer "
                             "(KIVI path only; 0 disables protection)")
    parser.add_argument("--prefill_length", type=int, default=512,
                        help="prefix tokens fed as one prefill (default 512)")
    parser.add_argument("--decode_length", type=int, default=2048,
                        help="teacher-forced steps scored per window (default 2048)")
    parser.add_argument("--windows", type=int, default=8,
                        help="number of independent windows spread over the corpus")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="windows decoded together; caps the prefill logits tensor")
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
    # Probe switch: keep the leading positions in FP16 (see models/llama_kivi.py).
    config.protect_prefix = args.protect_prefix
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


def window_starts(num_tokens, span, windows):
    """Evenly spaced start offsets for ``windows`` independent segments."""
    max_start = num_tokens - span
    if windows < 1 or max_start < 0:
        return []
    if windows == 1:
        return [0]
    step = max_start / (windows - 1)
    starts = sorted({int(round(index * step)) for index in range(windows)})
    return starts


def segment_nll(model, input_ids, prefill_length, decode_length, progress=None):
    """Summed NLL over the decode segment of a batch of equal-length windows.

    ``input_ids`` is ``[batch, prefill_length + decode_length]``.  The prefix is
    fed in one prefill; every following token is fed back one at a time with the
    cache attached, so each scored position reads the (quantized) history.
    """
    import torch.nn.functional as F

    outputs = model(input_ids=input_ids[:, :prefill_length], use_cache=True)
    cache = outputs.past_key_values
    total = F.cross_entropy(
        outputs.logits[:, -1].float(), input_ids[:, prefill_length], reduction="sum"
    ).item()
    del outputs

    for step in range(1, decode_length):
        token = input_ids[:, prefill_length + step - 1:prefill_length + step]
        outputs = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        total += F.cross_entropy(
            outputs.logits[:, -1].float(), input_ids[:, prefill_length + step], reduction="sum"
        ).item()
        del outputs
        if progress is not None:
            progress(step + 1, decode_length)

    del cache
    return total


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("prefill_length", "decode_length", "windows", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    if args.json_out and args.reference:
        same = os.path.abspath(os.path.expanduser(args.json_out)) == os.path.abspath(
            os.path.expanduser(args.reference)
        )
        if same:
            parser.error(
                "--reference and --json_out point at the same file; that silently "
                "produces a zero delta"
            )

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
    span = args.prefill_length + args.decode_length
    starts = window_starts(len(ids), span, args.windows)
    if not starts:
        raise SystemExit(
            f"corpus has {len(ids)} tokens, fewer than one "
            f"{span}-token window ({args.prefill_length} prefill + {args.decode_length} decode)"
        )
    _log(
        f"corpus tokens: {len(ids)}, windows: {len(starts)} "
        f"({args.prefill_length} prefill + {args.decode_length} decode each), "
        f"k_bits={args.k_bits}, v_bits={args.v_bits}"
    )

    model = load_model(args, model_path, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_nll = 0.0
    total_tokens = 0
    log_every = max(args.decode_length // 10, 1)

    def progress(step, total):
        if step % log_every == 0 or step == total:
            _log(f"  decode {step}/{total}")

    with torch.no_grad():
        for index in range(0, len(starts), args.batch_size):
            batch = starts[index:index + args.batch_size]
            input_ids = torch.tensor(
                [ids[start:start + span] for start in batch],
                dtype=torch.long,
                device=device,
            )
            _log(f"window batch {index // args.batch_size + 1} ({len(batch)} windows)")
            total_nll += segment_nll(
                model, input_ids, args.prefill_length, args.decode_length, progress
            )
            total_tokens += input_ids.shape[0] * args.decode_length
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
        if delta == 0.0:
            print(
                "warning: delta is exactly 0 -- --reference probably points at this "
                "run's own output, or both runs used the same k_bits"
            )

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
        "prefill_length": args.prefill_length,
        "decode_length": args.decode_length,
        "windows": len(starts),
        "tokens": total_tokens,
        "nll_sum": total_nll,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "metric": "continuation-loss (prefix prefill + teacher-forced decode)",
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
