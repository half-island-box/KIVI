# coding=utf-8
"""Model and dataset source resolution: local copy first, ModelScope by default.

Everything in this module is **local-first**: before anything is downloaded we
look for an existing copy on the machine (an explicit path, the project's
``cached_models/`` directory, the ModelScope cache, or a per-dataset extraction
cache) and reuse it. Only when no usable copy is found do we touch the network.

Models
    :func:`resolve_model_path` returns a directory that ``transformers`` can
    load. For ``model_source="hf"`` the id is returned untouched and the HF Hub
    resolves it; for ``modelscope`` (the default) the snapshot is downloaded
    with ``modelscope.snapshot_download`` and the local directory is returned.

Datasets
    :func:`load_long_bench_task` and :func:`load_gsm8k_split` return
    ``datasets.Dataset`` objects. For ``data_source="modelscope"`` the raw files
    are fetched from ModelScope and decoded locally, so ``huggingface.co`` is
    never contacted.
"""

import json
import os
import shutil
import tempfile
import zipfile

MODEL_SOURCE_CHOICES = ("modelscope", "hf")
DATA_SOURCE_CHOICES = ("modelscope", "hf")

LONGBENCH_MODELSCOPE_ID = "ZhipuAI/LongBench"
LONGBENCH_MODELSCOPE_REVISION = "master"
LONGBENCH_MODELSCOPE_FILE_URL = (
    "https://www.modelscope.cn/api/v1/datasets/{repo_id}/repo"
    "?Revision={revision}&FilePath={path}"
)
LONGBENCH_HF_ID = "THUDM/LongBench"
GSM8K_MODELSCOPE_ID = "modelscope/gsm8k"
GSM8K_HF_ID = "gsm8k"

# The ModelScope ``modelscope/gsm8k`` repository ships only a loading script; the
# script always pulls these two URLs (see its ``TRAIN_URL`` / ``TEST_URL``).
GSM8K_MODELSCOPE_SPLIT_URLS = {
    "train": "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/gsm8k/train.jsonl",
    "test": "https://sail-moe.oss-cn-hangzhou.aliyuncs.com/open_data/gsm8k/test.jsonl",
}

_WEIGHT_FILE_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".gguf",
    ".ckpt",
    ".safetensors.index.json",
)

_MODEL_SOURCE_ALIASES = {
    "modelscope": "modelscope",
    "model_scope": "modelscope",
    "ms": "modelscope",
    "hf": "hf",
    "huggingface": "hf",
    "huggingface_hub": "hf",
}


def _log(message):
    print(f"[source] {message}", flush=True)


def _expand(path):
    return os.path.abspath(os.path.expanduser(path))


# ---------------------------------------------------------------------------
# source names
# ---------------------------------------------------------------------------


def normalize_model_source(model_source):
    """Map user-provided aliases to ``modelscope`` or ``hf``."""
    source = (model_source or "modelscope").strip().lower()
    if source not in _MODEL_SOURCE_ALIASES:
        raise ValueError(
            f"Unknown model_source={model_source!r}; "
            f"expected one of {list(MODEL_SOURCE_CHOICES)}."
        )
    return _MODEL_SOURCE_ALIASES[source]


def normalize_data_source(data_source):
    """Map user-provided aliases to ``modelscope`` or ``hf``."""
    source = (data_source or "modelscope").strip().lower()
    if source not in _MODEL_SOURCE_ALIASES:
        raise ValueError(
            f"Unknown data_source={data_source!r}; "
            f"expected one of {list(DATA_SOURCE_CHOICES)}."
        )
    return _MODEL_SOURCE_ALIASES[source]


# ---------------------------------------------------------------------------
# local-copy discovery
# ---------------------------------------------------------------------------


def _cache_root(cache_dir=None):
    """Project-level cache root used for extracted dataset files."""
    if cache_dir:
        return _expand(cache_dir)
    return os.path.join(os.path.expanduser("~"), ".cache", "kivi")


def _hub_roots(cache_dir=None, extra_roots=()):
    """Directories that may already contain a ModelScope snapshot."""
    roots = []
    if cache_dir:
        roots.append(_expand(cache_dir))
    env_cache = os.environ.get("MODELSCOPE_CACHE")
    if env_cache:
        roots.append(_expand(env_cache))
        roots.append(os.path.join(_expand(env_cache), "hub"))
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub"))
    roots.append(os.path.join(os.path.expanduser("~"), ".cache", "modelscope"))
    for root in extra_roots:
        if root:
            roots.append(_expand(root))

    unique = []
    for root in roots:
        if root not in unique and os.path.isdir(root):
            unique.append(root)
    return unique


def _candidate_dirs(repo_id, roots):
    """Yield plausible local directories for ``repo_id`` under each search root."""
    namespace, _, name = repo_id.partition("/")
    if not name:
        namespace, name = "", namespace
    for root in roots:
        yield os.path.join(root, repo_id) if namespace else os.path.join(root, name)
        # People often keep a model under its short name (e.g. cached_models/<name>).
        yield os.path.join(root, name)
        yield os.path.join(root, "models", repo_id) if namespace else os.path.join(root, "models", name)
        yield os.path.join(root, "datasets", repo_id) if namespace else os.path.join(root, "datasets", name)
        if namespace:
            snapshots = os.path.join(root, f"models--{namespace}--{name}", "snapshots")
            if os.path.isdir(snapshots):
                for revision in sorted(os.listdir(snapshots), reverse=True):
                    yield os.path.join(snapshots, revision)
            # Newer ModelScope layout: <root>/{models,datasets}/{ns}--{name}/snapshots/<rev>
            for kind in ("models", "datasets"):
                snapshots = os.path.join(root, kind, f"{namespace}--{name}", "snapshots")
                if os.path.isdir(snapshots):
                    for revision in sorted(os.listdir(snapshots), reverse=True):
                        yield os.path.join(snapshots, revision)


def find_existing_dir(repo_id, roots, predicate=None):
    """Return the first candidate directory that exists and satisfies ``predicate``."""
    seen = set()
    for candidate in _candidate_dirs(repo_id, roots):
        if candidate in seen:
            continue
        seen.add(candidate)
        if not os.path.isdir(candidate):
            continue
        if predicate is not None and not predicate(candidate):
            continue
        return candidate
    return None


def _is_complete_model_dir(path):
    """A model directory must have a config and at least one weight file."""
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return any(entry.endswith(_WEIGHT_FILE_SUFFIXES) for entry in entries)


def _is_longbench_dir(path):
    return os.path.isfile(os.path.join(path, "data.zip")) or os.path.isdir(
        os.path.join(path, "data")
    )


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def resolve_model_path(
    model_name_or_path,
    model_source="modelscope",
    cache_dir=None,
    revision=None,
):
    """Resolve a model id to something transformers can load.

    - local directory: returned as-is
    - an already-downloaded local copy is reused (explicit ``cache_dir``,
      ``cached_models/``, the ModelScope cache, or an HF snapshot directory)
    - ``model_source="hf"``: id returned as-is (the HF Hub resolves the download)
    - ``model_source="modelscope"``: downloaded via ModelScope, local dir returned
    """
    source = normalize_model_source(model_source)
    local_path = _expand(model_name_or_path)
    if os.path.isdir(local_path):
        return local_path

    extra_roots = (
        os.path.join(os.getcwd(), "cached_models"),
        os.getcwd(),
    )
    existing = find_existing_dir(
        model_name_or_path,
        _hub_roots(cache_dir, extra_roots),
        predicate=_is_complete_model_dir,
    )
    if existing is not None:
        _log(f"using existing local copy of {model_name_or_path}: {existing}")
        return existing

    if source == "hf":
        return model_name_or_path

    try:
        from modelscope import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "model_source='modelscope' requires the `modelscope` package. "
            "Install it with `pip install modelscope`, or pass `--model_source hf` "
            "to load from Hugging Face instead."
        ) from exc

    _log(f"downloading {model_name_or_path} from ModelScope")
    return snapshot_download(
        model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
    )


# ---------------------------------------------------------------------------
# LongBench
# ---------------------------------------------------------------------------


def _longbench_task_file(root, task):
    for sub in ("data", ""):
        candidate = os.path.join(root, sub, f"{task}.jsonl") if sub else os.path.join(root, f"{task}.jsonl")
        if os.path.isfile(candidate):
            return candidate
    return None


def _longbench_zip(root):
    for candidate in (os.path.join(root, "data.zip"), os.path.join(root, "data", "data.zip")):
        if os.path.isfile(candidate):
            return candidate
    return None


def _safe_extract_member(archive, member, dest):
    if member.startswith("/") or ".." in member.split("/"):
        raise ValueError(f"Refusing to extract unsafe path {member!r}")
    archive.extract(member, dest)


def extract_longbench_task(zip_path, dest_root, task):
    """Extract ``data/<task>.jsonl`` from the LongBench archive, reusing prior runs."""
    existing = _longbench_task_file(dest_root, task)
    if existing is not None:
        return existing

    os.makedirs(dest_root, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        member = f"data/{task}.jsonl"
        if member not in names:
            matches = [name for name in names if name.endswith(f"/{task}.jsonl") or name == f"{task}.jsonl"]
            if not matches:
                raise FileNotFoundError(
                    f"{task}.jsonl not found inside {zip_path}; available tasks are "
                    f"{sorted(os.path.basename(n)[:-6] for n in names if n.endswith('.jsonl'))}"
                )
            member = matches[0]
        _log(f"extracting {member} -> {dest_root}")
        with tempfile.TemporaryDirectory(dir=dest_root, prefix=".longbench-") as tmp:
            _safe_extract_member(archive, member, tmp)
            src = os.path.join(tmp, member)
            dst_dir = os.path.dirname(os.path.join(dest_root, member))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, os.path.basename(member)))

    return os.path.join(dest_root, member)


def _resolve_longbench_source(data_dir=None, cache_dir=None):
    """Locate (or fetch) a directory containing LongBench jsonl files."""
    explicit = data_dir or os.environ.get("LONGBENCH_DATA_DIR")
    if explicit:
        root = _expand(explicit)
        if os.path.isfile(root):
            if not root.endswith(".zip"):
                raise ValueError(
                    f"data_dir={explicit!r} is a file; pass a directory holding "
                    "data/*.jsonl or data.zip"
                )
            return os.path.join(os.path.dirname(root), "longbench_data"), root
        if os.path.isfile(os.path.join(root, "data.zip")):
            return root, os.path.join(root, "data.zip")
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"data_dir={explicit!r} does not exist; download LongBench once or "
                "point --data_dir at a directory holding data/*.jsonl or data.zip"
            )
        zip_path = _longbench_zip(root)
        if zip_path is not None:
            return root, zip_path
        if os.path.isdir(os.path.join(root, "data")) or any(
            name.endswith(".jsonl") for name in os.listdir(root)
        ):
            return root, None
        _log(f"data_dir={root} is empty, downloading LongBench and extracting into it")
        return root, _download_longbench_zip(cache_dir)

    cache_root = os.path.join(_cache_root(cache_dir), "longbench")
    if os.path.isdir(os.path.join(cache_root, "data")):
        # Another task was already extracted; keep the archive for the rest.
        return cache_root, _longbench_zip(cache_root)

    snapshot = find_existing_dir(
        LONGBENCH_MODELSCOPE_ID,
        _hub_roots(cache_dir),
        predicate=_is_longbench_dir,
    )
    if snapshot is not None:
        _log(f"using existing LongBench snapshot: {snapshot}")
        zip_path = _longbench_zip(snapshot)
        return cache_root, zip_path

    zip_path = _download_longbench_zip(cache_dir)
    return cache_root, zip_path


def _download_longbench_zip(cache_dir=None):
    """Fetch LongBench's ``data.zip`` from ModelScope into the project cache.

    Uses the dataset file endpoint directly (rather than
    ``modelscope.snapshot_download``) so it does not depend on the installed
    ModelScope version treating the id as a dataset. The file is cached; later
    runs return the cached archive without touching the network.
    """
    dest_root = os.path.join(_cache_root(cache_dir), "longbench")
    zip_path = os.path.join(dest_root, "data.zip")
    if os.path.isfile(zip_path):
        return zip_path

    url = LONGBENCH_MODELSCOPE_FILE_URL.format(
        repo_id=LONGBENCH_MODELSCOPE_ID,
        revision=LONGBENCH_MODELSCOPE_REVISION,
        path="data.zip",
    )
    _log(f"downloading {LONGBENCH_MODELSCOPE_ID} data.zip (~114 MB) from ModelScope")
    _download_file(url, zip_path, validate=_require_zip)
    return zip_path


def _longbench_features():
    from datasets import Features, Sequence, Value

    return Features(
        {
            "input": Value("string"),
            "context": Value("string"),
            "answers": Sequence(Value("string")),
            "length": Value("int32"),
            "dataset": Value("string"),
            "language": Value("string"),
            "all_classes": Sequence(Value("string")),
            "_id": Value("string"),
        }
    )


def load_long_bench_task(
    task,
    split="test",
    data_source="modelscope",
    data_dir=None,
    cache_dir=None,
):
    """Load one LongBench config (e.g. ``qasper`` or ``qasper_e``).

    Local copies are reused before anything is downloaded; see the module
    docstring. With ``data_source="hf"`` the HF Hub is used directly. The
    local jsonl files hold the test split only, so ``split`` only matters for
    the Hugging Face path.
    """
    source = normalize_data_source(data_source)
    if source == "hf":
        from datasets import load_dataset

        _log(f"loading {LONGBENCH_HF_ID}/{task} from Hugging Face")
        return load_dataset(LONGBENCH_HF_ID, task, split=split)

    root, zip_path = _resolve_longbench_source(data_dir=data_dir, cache_dir=cache_dir)
    path = _longbench_task_file(root, task)
    if path is None:
        if zip_path is None:
            raise FileNotFoundError(
                f"{task}.jsonl not found under {root} and no data.zip is available; "
                "remove the directory or pass --data_dir"
            )
        path = extract_longbench_task(zip_path, root, task)

    from datasets import Dataset

    _log(f"loading {task} from {path}")
    return Dataset.from_json(path, features=_longbench_features())


# ---------------------------------------------------------------------------
# network helpers shared by the dataset loaders
# ---------------------------------------------------------------------------


def _download_file(url, destination, validate=None):
    """Download ``url`` to ``destination`` atomically (cached on later runs).

    ``validate`` is called with the temporary file before it is moved into
    place; if it raises, the partial download is removed so the next run retries
    instead of reusing a broken cache entry.
    """
    from urllib.request import urlopen

    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    partial = f"{destination}.part"
    _log(f"downloading {url}")
    with urlopen(url) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle)
    if validate is not None:
        try:
            validate(partial)
        except Exception:
            os.remove(partial)
            raise
    os.replace(partial, destination)
    return destination


def _require_zip(path):
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"{path} is not a zip archive; the ModelScope endpoint probably "
            "returned an error page instead of the data"
        )


def _require_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        first = handle.readline()
    if not first.strip():
        raise ValueError(f"{path} is empty; the download did not complete")
    json.loads(first)


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def _resolve_gsm8k_jsonl(split, data_dir=None, cache_dir=None):
    explicit = data_dir or os.environ.get("GSM8K_DATA_DIR")
    if explicit:
        root = _expand(explicit)
        for candidate in (
            os.path.join(root, f"{split}.jsonl"),
            os.path.join(root, "main", f"{split}.jsonl"),
        ):
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"data_dir={explicit!r} has no {split}.jsonl; expected "
            f"{root}/{split}.jsonl or {root}/main/{split}.jsonl"
        )

    cache_root = os.path.join(_cache_root(cache_dir), "gsm8k")
    cached = os.path.join(cache_root, f"{split}.jsonl")
    if os.path.isfile(cached):
        return cached

    if split not in GSM8K_MODELSCOPE_SPLIT_URLS:
        raise ValueError(
            f"split={split!r} is not available from {GSM8K_MODELSCOPE_ID}; "
            f"expected one of {sorted(GSM8K_MODELSCOPE_SPLIT_URLS)}"
        )
    return _download_file(
        GSM8K_MODELSCOPE_SPLIT_URLS[split], cached, validate=_require_jsonl
    )


def load_gsm8k_split(split="train", data_source="modelscope", data_dir=None, cache_dir=None):
    """Load one GSM8K split, reusing a local copy before downloading."""
    source = normalize_data_source(data_source)
    if source == "hf":
        from datasets import load_dataset

        _log(f"loading {GSM8K_HF_ID}/{split} from Hugging Face")
        return load_dataset(GSM8K_HF_ID, "main", split=split)

    path = _resolve_gsm8k_jsonl(split, data_dir=data_dir, cache_dir=cache_dir)
    from datasets import Dataset

    _log(f"loading gsm8k/{split} from {path}")
    return Dataset.from_json(path)


# ---------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------


def add_model_source_argument(parser, default="modelscope"):
    """Add the shared ``--model_source`` flag to an argparse parser."""
    parser.add_argument(
        "--model_source",
        "--model-source",
        dest="model_source",
        type=str,
        choices=MODEL_SOURCE_CHOICES,
        default=default,
        help="Where to fetch model config/tokenizer/weights: "
        "'modelscope' (default) or 'hf'.",
    )
    return parser


def add_data_source_argument(parser, default="modelscope"):
    """Add the shared ``--data_source`` flag to an argparse parser."""
    parser.add_argument(
        "--data_source",
        "--data-source",
        dest="data_source",
        type=str,
        choices=DATA_SOURCE_CHOICES,
        default=default,
        help="Where to fetch benchmark data: 'modelscope' (default) or 'hf'.",
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Local directory holding the benchmark data; reused instead of "
        "downloading when present.",
    )
    return parser
