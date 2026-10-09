"""Two stored adapters compared number by number, from their files alone: no model, no torch.

An adapter is saved as one safetensors file: eight bytes (the header's length, little-endian), a JSON header naming
each tensor's type, shape and place, then the tensors' bytes. `compared` reads two such files and counts the numbers
that are not the same, BIT FOR BIT, over the tensors both files hold under one name with one type and one shape. A
tensor in one file alone, or held with another type or shape, is not compared and counts for nothing: a number read
through a change of type could differ by its rounding alone, and a count of such numbers would say nothing.

It is what L4's check "the adapter differs from the start adapter's" reads (`domain/ladder_round/l4.py`,
`the_adapter_differs`): a training that saved the adapter it started from gives zero.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

ADAPTER_FILE = "adapter_model.safetensors"      # what an adapter's directory holds its weights in
HOW = "bit for bit, over the tensors both files hold under one name with one type and one shape"
_FLOATS = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}


def tensors_of(path: Path) -> tuple[dict, int]:
    """(by tensor name: its `dtype`, `shape` and `data_offsets`; where the tensors' bytes begin) of a safetensors file."""
    with open(path, "rb") as file:
        (length,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def _bits(path: Path, begins: int, entry: dict) -> np.ndarray | None:
    """One tensor's numbers as unsigned integers of their own width, read in place; None for a tensor with no number
    or with a width that is not 1, 2, 4 or 8 bytes."""
    begin, end = entry["data_offsets"]
    count = math.prod(entry["shape"])
    width = (end - begin) // count if count else 0
    if width not in (1, 2, 4, 8) or width * count != end - begin:
        return None
    return np.memmap(path, dtype=f"<u{width}", mode="r", offset=begins + begin, shape=(count,))


def _values(bits: np.ndarray, dtype: str) -> np.ndarray | None:
    """The numbers those bits are, for a floating type (bfloat16 is the upper half of a float32); None otherwise."""
    if dtype == "BF16":
        return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return np.asarray(bits).view(_FLOATS[dtype]).astype(np.float64) if dtype in _FLOATS else None


def compared(saved: Path, start: Path) -> dict:
    """The adapter in the file `saved` against the one in `start`: `tensors` compared, `tensors_not_compared` (in one
    file alone, or of another type or shape), `elements` (the numbers compared), `elements_changed` (those whose bits
    are not the same) and, for information, `largest_change` (the largest absolute difference among the floating
    numbers compared; None when there is none)."""
    of_saved, saved_begins = tensors_of(saved)
    of_start, start_begins = tensors_of(start)
    tensors = elements = changed = 0
    largest = None
    for name in sorted(set(of_saved) & set(of_start)):
        one, other = of_saved[name], of_start[name]
        if one["dtype"] != other["dtype"] or list(one["shape"]) != list(other["shape"]):
            continue
        bits, other_bits = _bits(saved, saved_begins, one), _bits(start, start_begins, other)
        if bits is None or other_bits is None or bits.dtype != other_bits.dtype:
            continue
        tensors += 1
        elements += int(bits.size)
        differ = np.asarray(bits != other_bits)
        changed += int(differ.sum())
        if differ.any() and _values(bits, one["dtype"]) is not None:
            moved = np.abs(_values(bits, one["dtype"])[differ] - _values(other_bits, one["dtype"])[differ])
            moved = moved[np.isfinite(moved)]
            if moved.size:
                largest = max(largest or 0.0, float(moved.max()))
    return {"how": HOW, "tensors": tensors, "tensors_not_compared": len(set(of_saved) | set(of_start)) - tensors, "elements": elements,
            "elements_changed": changed, "largest_change": largest}
