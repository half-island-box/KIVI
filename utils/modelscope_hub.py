# coding=utf-8
"""Model source resolution: ModelScope by default, Hugging Face Hub on demand.

All entry points accept a ``model_source`` value (``modelscope`` or ``hf``) and
resolve ``model_name_or_path`` through :func:`resolve_model_path` before handing
it to ``transformers``. For ``modelscope`` the id is downloaded with
``modelscope.snapshot_download`` and the resulting local directory is returned,
so the rest of the code keeps using the stock transformers loaders.

Local directories are always returned unchanged, regardless of the source.
"""

import os

MODEL_SOURCE_CHOICES = ("modelscope", "hf")

_MODEL_SOURCE_ALIASES = {
    "modelscope": "modelscope",
    "model_scope": "modelscope",
    "ms": "modelscope",
    "hf": "hf",
    "huggingface": "hf",
    "huggingface_hub": "hf",
}


def normalize_model_source(model_source):
    """Map user-provided aliases to ``modelscope`` or ``hf``."""
    source = (model_source or "modelscope").strip().lower()
    if source not in _MODEL_SOURCE_ALIASES:
        raise ValueError(
            f"Unknown model_source={model_source!r}; "
            f"expected one of {list(MODEL_SOURCE_CHOICES)}."
        )
    return _MODEL_SOURCE_ALIASES[source]


def resolve_model_path(
    model_name_or_path,
    model_source="modelscope",
    cache_dir=None,
    revision=None,
):
    """Resolve a model id to something transformers can load.

    - local directory: returned as-is
    - ``model_source="hf"``: id returned as-is (HF Hub handles the download)
    - ``model_source="modelscope"``: downloaded via ModelScope, local dir returned
    """
    source = normalize_model_source(model_source)
    local_path = os.path.expanduser(model_name_or_path)
    if os.path.isdir(local_path):
        return local_path
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

    return snapshot_download(
        model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
    )


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
