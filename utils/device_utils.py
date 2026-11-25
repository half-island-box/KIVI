"""Device selection helpers shared by the experiment entry points.

``--device`` follows PyTorch's visible-device numbering: with
``CUDA_VISIBLE_DEVICES`` unset, ``cuda:1`` is physical GPU 1; when the
environment masks GPUs, the index is relative to the visible set.
"""
from __future__ import annotations

import argparse
from typing import Optional, Union

import torch


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--device`` flag to an argparse parser."""
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Inference device, e.g. 'cuda:1', '1', or 'cpu'. "
            "Indexes follow PyTorch's visible-device numbering "
            "(CUDA_VISIBLE_DEVICES). Defaults to the current CUDA device "
            "(cuda:0 when visible), or CPU when CUDA is unavailable."
        ),
    )


def resolve_device(spec: Optional[str]) -> torch.device:
    """Convert a CLI device spec into a validated ``torch.device``.

    Accepts ``None``/empty (keep the old default), ``cpu``, ``cuda``,
    ``cuda:N``, and a bare integer such as ``1`` (shorthand for ``cuda:1``).
    """
    if spec is None or str(spec).strip() == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = str(spec).strip()
    if text.lstrip("-").isdigit():
        text = f"cuda:{text}"

    try:
        device = torch.device(text)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid --device {spec!r}; expected 'cuda:N', a device index, or 'cpu'."
        ) from exc

    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {spec!r} requests CUDA, but torch.cuda.is_available() is False."
        )

    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise ValueError(
            f"--device {spec!r} is out of range: {count} CUDA device(s) are visible."
        )
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def device_map_for(spec: Optional[str], device: torch.device) -> Union[str, dict]:
    """Return a ``device_map`` for ``from_pretrained``.

    When no ``--device`` was given, keep the original ``"auto"`` behaviour.
    When one was given, force the whole model onto that single device.
    """
    if spec is None or str(spec).strip() == "":
        return "auto"
    return {"": str(device)}
