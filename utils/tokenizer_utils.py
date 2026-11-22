# coding=utf-8
"""Tokenizer helpers shared by the KIVI entry points.

Llama-3.x ships a fast BPE tokenizer (``tokenizer.json``) and keeps the
SentencePiece model under ``original/tokenizer.model``; the repository root has
no ``tokenizer.model``. Forcing ``use_fast=False`` therefore instantiates the
slow ``LlamaTokenizer``, which calls ``sentencepiece.Load(None)`` and fails with
``TypeError: not a string``.

Llama-2 / LongChat keep using the slow tokenizer so the upstream baseline
behavior does not change.
"""


def is_llama3(*model_refs):
    """Return True if any reference looks like a Llama-3.x model."""
    return any("llama-3" in str(ref).lower() for ref in model_refs if ref)


def load_llama_tokenizer(model_path, *model_refs):
    """Load a Llama tokenizer with the appropriate fast/slow implementation.

    ``model_path`` is the resolved local directory or model id. The extra
    ``model_refs`` are the user-facing ids; they are used for fast/slow
    detection when a local directory name does not contain ``llama-3``.
    """
    from transformers import AutoTokenizer

    if is_llama3(model_path, *model_refs):
        return AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )
    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        tokenizer_type="llama",
    )
