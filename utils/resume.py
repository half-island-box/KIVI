"""Helpers for resumable JSONL prediction files."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def read_valid_jsonl_count(path: str, expected_total: Optional[int] = None) -> int:
    """Return the number of complete JSON lines, repairing a partial tail.

    A process killed mid-write can leave a partial last line. Only complete,
    parseable lines are kept; the file is truncated to the last complete line
    and a trailing newline is ensured before the caller appends more records.
    """
    if not os.path.exists(path):
        return 0

    valid = 0
    offset = 0
    with open(path, "rb") as handle:
        for line in handle:
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                break
            valid += 1
            offset += len(line)

    if offset != os.path.getsize(path):
        with open(path, "r+b") as handle:
            handle.truncate(offset)

    if offset > 0:
        with open(path, "r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")

    if expected_total is not None and valid > expected_total:
        raise ValueError(
            f"{path} has {valid} valid predictions but the dataset has only "
            f"{expected_total} samples; the dataset or output directory is inconsistent."
        )
    return valid


def load_or_write_meta(path: str, meta: Dict[str, Any], resume: bool) -> bool:
    """Validate resume metadata, or write it for a new run.

    Returns True when an existing metadata file was validated, False when a new
    one was written. Raises ValueError when the existing metadata does not match
    the current run.
    """
    if resume and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                previous = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(previous, dict):
            raise ValueError(f"{path} is not a JSON object.")

        mismatch = [
            key
            for key, value in meta.items()
            if key != "created_at" and previous.get(key) != value
        ]
        if mismatch:
            raise ValueError(
                f"{path} belongs to a different run; mismatched fields: "
                f"{', '.join(sorted(mismatch))}. Delete the output directory or "
                "rerun without --resume."
            )
        return True

    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return False
