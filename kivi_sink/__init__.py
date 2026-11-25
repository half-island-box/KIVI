# coding=utf-8
"""SinkBudget research package.

M1 scope: read-only attention statistics and token/layer sink scores.

Everything in this package is opt-in.  The KIVI inference path is not modified
and no module here is imported by ``models/`` or ``quant/``.  Calibration runs
in a separate process against a plain Hugging Face model; the functions in
:mod:`kivi_sink.attention_stats` and :mod:`kivi_sink.sink_score` are pure tensor
operations.

Importing this package must not change global state (RNG, default dtype,
``torch.backends.*``); calibration seeds are only used inside local generators.
"""

__all__ = []
