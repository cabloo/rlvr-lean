"""Two stored adapters compared number by number, from their files alone: no model, no torch.

An adapter is saved as one safetensors file: eight bytes (the header's length, little-endian), a JSON header naming
each tensor's type, shape and place, then the tensors' bytes. `compared` reads two such files and counts the numbers
that are not the same, BIT FOR BIT, over the tensors both files hold under one name with one type and one shape. A
tensor in one file alone, or held with another type or shape, is not compared and counts for nothing: a number read
through a change of type could differ by its rounding alone, and a count of such numbers would say nothing.

It is what L4's check "the adapter differs from the start adapter's" reads (`domain/ladder_round/l4.py`,
`the_adapter_differs`): a training that saved the adapter it started from gives zero.

THE MEAN OF SEVERAL ADAPTERS (spec "L5n: what is one training worth?": its two averaged models). `mean_of` writes the
file whose every number is the arithmetic mean of the numbers its adapters hold at that place, and `is_the_mean` reads
such a file back and holds it to them. The adapters must hold the same tensors, each under one name with one type and
one shape, of a floating type the mean can be written in (`MEAN_TYPES`); anything else is refused. A number's mean is
summed in float64 and rounded ONCE to the tensor's own type, so it is the nearest number of that type to the mean
(for float32: within half a unit in the last place, a relative 6e-8).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

ADAPTER_FILE = "adapter_model.safetensors"      # what an adapter's directory holds its weights in
HOW = "bit for bit, over the tensors both files hold under one name with one type and one shape"
_FLOATS = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
MEAN_TYPES = tuple(_FLOATS)                     # the types a mean is written in (bfloat16 and whole numbers are not: an adapter saved by this project's trainings holds float32)
MEAN_HOW = ("each number is the arithmetic mean of the adapters' numbers at its place, summed in float64 and rounded once to the tensor's own type; read back, the file's numbers "
            "are held to that bit for bit")


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


# ------------------------------------------------------------------------------- the mean of several adapters
def _alike(files: Sequence[Path]) -> tuple[list[tuple[dict, int]], list[str]]:
    """(each file's tensors and where their bytes begin; the tensors' names, sorted) of the adapters a mean is made of.
    Refused (ValueError): fewer than two files (the mean of one adapter is that adapter); a tensor that one file holds
    and another does not; a tensor held with another type or shape in two of them; a type the mean is not written in."""
    if len(files) < 2:
        raise ValueError(f"the mean of adapters is made of at least two, and {len(files)} was given")
    headers = [tensors_of(path) for path in files]
    first = headers[0][0]
    for path, (header, _) in zip(files[1:], headers[1:]):
        alone = sorted(set(header) ^ set(first))
        if alone:
            raise ValueError(f"{path} and {files[0]} do not hold the same tensors ({len(alone)} are in one alone; first: {alone[0]}): the mean is made tensor by tensor")
        other = sorted(name for name in first if header[name]["dtype"] != first[name]["dtype"] or list(header[name]["shape"]) != list(first[name]["shape"]))
        if other:
            raise ValueError(f"{path} holds {len(other)} tensors with another type or shape than {files[0]} (first: {other[0]}, {header[other[0]]['dtype']} "
                             f"{list(header[other[0]]['shape'])} against {first[other[0]]['dtype']} {list(first[other[0]]['shape'])}): the mean is made number by number")
    strange = sorted(name for name, entry in first.items() if entry["dtype"] not in _FLOATS)
    if strange:
        raise ValueError(f"{len(strange)} tensors of {files[0]} are of a type the mean is not written in (first: {strange[0]}, {first[strange[0]]['dtype']}): it is made of "
                         f"{', '.join(MEAN_TYPES)}")
    return headers, sorted(first)


def _mean(files: Sequence[Path], headers: Sequence[tuple[dict, int]], name: str) -> np.ndarray | None:
    """One tensor's mean over the files, in its own type: summed in float64, divided, rounded once. None for a tensor
    with no number. Refused (ValueError) when a file's bytes for it are not as many as its type and shape say."""
    entry = headers[0][0][name]
    kind = np.dtype(_FLOATS[entry["dtype"]])
    if math.prod(entry["shape"]) == 0:
        return None
    total = None
    for path, (header, begins) in zip(files, headers):
        bits = _bits(path, begins, header[name])
        if bits is None or bits.dtype.itemsize != kind.itemsize:
            raise ValueError(f"the tensor {name} of {path} does not hold the bytes its type ({entry['dtype']}) and shape ({list(entry['shape'])}) say")
        values = np.asarray(bits).view(kind).astype(np.float64)
        total = values if total is None else total + values
    return (total / len(files)).astype(kind)


def mean_of(files: Sequence[Path], target: Path) -> dict:
    """THE MEAN OF SEVERAL ADAPTERS, written to the file `target`: every tensor is the arithmetic mean of the tensors of
    that name in `files` (`_alike`: the same names, types and shapes, or refused). The file is the FIRST adapter's, byte
    for byte (its header: the names, types, shapes and places, as the library that saved it laid them out), with every
    tensor's numbers written over by their mean. Returns how many adapters, tensors and numbers it is made of."""
    headers, names = _alike(files)
    begins = headers[0][1]
    temporary = target.with_name(target.name + ".tmp")
    elements = 0
    try:
        shutil.copyfile(files[0], temporary)
        for name in names:
            mean = _mean(files, headers, name)
            if mean is None:
                continue
            own = np.memmap(temporary, dtype=mean.dtype, mode="r+", offset=begins + headers[0][0][name]["data_offsets"][0], shape=mean.shape)
            own[:] = mean
            own.flush()
            del own
            elements += int(mean.size)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)       # a mean that could not be made leaves nothing behind
    return {"how": MEAN_HOW, "adapters": len(files), "tensors": len(names), "elements": elements}


def is_the_mean(saved: Path, files: Sequence[Path]) -> dict:
    """The file `saved` READ BACK and held to the mean of `files`: `elements_not_the_mean` (the numbers whose bits are
    not the mean's, made again here from the files; a tensor `saved` lacks or holds with another type or shape counts
    whole), `tensors_of_the_file_alone` (tensors `saved` holds and the adapters do not) and, by adapter in the order
    given, `elements_not_those_of_each_adapter` (`compared`: how many of its numbers are not that adapter's). It IS the
    mean when no number is another and no tensor is its own alone, and it is NONE OF THEM when it differs from each."""
    headers, names = _alike(files)
    of_saved, saved_begins = tensors_of(saved)
    elements = wrong = 0
    for name in names:
        mean = _mean(files, headers, name)
        if mean is None:
            continue
        elements += int(mean.size)
        entry, own = headers[0][0][name], of_saved.get(name)
        bits = _bits(saved, saved_begins, own) if own is not None and own["dtype"] == entry["dtype"] and list(own["shape"]) == list(entry["shape"]) else None
        if bits is None or bits.dtype.itemsize != mean.dtype.itemsize:
            wrong += int(mean.size)
            continue
        wrong += int(np.asarray(np.asarray(bits) != mean.view(bits.dtype)).sum())
    alone = sorted(set(of_saved) - set(names))
    against = [compared(saved, path)["elements_changed"] for path in files]
    the_mean, none_of_them = bool(elements) and wrong == 0 and not alone, all(count > 0 for count in against)
    return {"how": MEAN_HOW, "adapters": len(files), "tensors": len(names), "tensors_of_the_file_alone": len(alone), "elements": elements, "elements_not_the_mean": wrong,
            "elements_not_those_of_each_adapter": against, "is_the_mean": the_mean, "is_none_of_them": none_of_them, "passes": the_mean and none_of_them}
