"""Two stored adapters compared from their files alone (`infrastructure/adapter_files.py`): what L4's check "the adapter
differs from the start adapter's" reads. The files here are written BY HAND, from the format's definition (eight bytes of
header length, a JSON header, the tensors' bytes), not by the reader under test. No model, no torch."""

import json
import struct
from pathlib import Path

import numpy as np

from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, compared, tensors_of

TYPES = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I64": "<i8"}


def _write(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]], metadata: dict | None = None, padding: int = 0) -> Path:
    """A safetensors file: `tensors` are (name, type, shape, bytes), laid out in the order given."""
    header, buffer = ({"__metadata__": metadata} if metadata is not None else {}), b""
    for name, kind, shape, data in tensors:
        header[name] = {"dtype": kind, "shape": list(shape), "data_offsets": [len(buffer), len(buffer) + len(data)]}
        buffer += data
    text = json.dumps(header).encode() + b" " * padding         # the format allows a header padded with spaces
    path.write_bytes(struct.pack("<Q", len(text)) + text + buffer)
    return path


def _of(kind: str, values) -> bytes:
    return np.asarray(values, dtype=TYPES[kind]).tobytes()


A = ("base_model.model.layers.0.q_proj.lora_A.weight", "F32", (2, 3), _of("F32", [[0.5, -1.25, 3.0], [0.0, 7.5, -2.0]]))
B = ("base_model.model.layers.0.q_proj.lora_B.weight", "F32", (3, 2), _of("F32", [[0.125, 0.25], [-0.5, 1.0], [2.0, -4.0]]))


def test_an_adapter_that_is_the_one_it_started_from_gives_zero_whatever_the_layout_of_its_file(tmp_path):
    start = _write(tmp_path / "start.safetensors", [A, B], metadata={"format": "pt"})
    # The same tensors in the other order, with no metadata and a padded header: other offsets, the same numbers.
    saved = _write(tmp_path / "saved.safetensors", [B, A], padding=5)
    header, begins = tensors_of(saved)
    assert sorted(header) == sorted([A[0], B[0]]) and header[A[0]]["data_offsets"] == [24, 48] and begins == saved.stat().st_size - 48
    assert "__metadata__" not in tensors_of(start)[0]
    result = compared(saved, start)
    assert (result["tensors"], result["tensors_not_compared"], result["elements"], result["elements_changed"], result["largest_change"]) == (2, 0, 12, 0, None)
    assert "bit for bit" in result["how"] and ADAPTER_FILE == "adapter_model.safetensors"


def test_one_number_moved_by_the_smallest_step_is_counted_and_nothing_is_rounded(tmp_path):
    start = _write(tmp_path / "start.safetensors", [A, B])
    values = np.frombuffer(B[3], dtype="<f4").copy()
    values[3] = np.nextafter(values[3], np.float32(2.0), dtype=np.float32)          # 1.0 to the next float32 above it: a difference of 2^-23
    values[5] = -3.0                                                                 # and -4.0 to -3.0
    saved = _write(tmp_path / "saved.safetensors", [A, (B[0], "F32", (3, 2), values.tobytes())])
    result = compared(saved, start)
    assert (result["tensors"], result["elements"], result["elements_changed"]) == (2, 12, 2) and result["largest_change"] == 1.0
    only_the_smallest = values.copy()
    only_the_smallest[5] = -4.0
    result = compared(_write(tmp_path / "smallest.safetensors", [A, (B[0], "F32", (3, 2), only_the_smallest.tobytes())]), start)
    assert result["elements_changed"] == 1 and result["largest_change"] == 2.0 ** -23       # far under any rounding a sum of the B matrices is stored with
    assert compared(start, saved)["elements_changed"] == 2                                   # the count does not depend on which file is which


def test_a_tensor_in_one_file_alone_or_of_another_type_or_shape_is_not_compared_and_counts_for_nothing(tmp_path):
    start = _write(tmp_path / "start.safetensors", [A, B, ("only_in_the_start", "F32", (2,), _of("F32", [1.0, 2.0]))])
    saved = _write(tmp_path / "saved.safetensors", [
        (A[0], "F64", (2, 3), _of("F64", [[0.5, -1.25, 3.0], [0.0, 7.5, -2.0]])),                # the same numbers held with another type
        (B[0], "F32", (2, 3), B[3]),                                                              # the same bytes under another shape
        ("only_in_the_saved", "F32", (2,), _of("F32", [9.0, 9.0]))])
    result = compared(saved, start)
    assert (result["tensors"], result["tensors_not_compared"], result["elements"], result["elements_changed"], result["largest_change"]) == (0, 4, 0, 0, None)
    # A type read through a change of type could differ by its rounding alone: float16's 0.1 is not float32's, and it is not counted.
    half = _write(tmp_path / "half.safetensors", [("w", "F16", (1,), _of("F16", [0.1]))])
    single = _write(tmp_path / "single.safetensors", [("w", "F32", (1,), _of("F32", [0.1]))])
    assert float(np.float16(0.1)) != float(np.float32(0.1)) and compared(half, single)["elements_changed"] == 0 and compared(half, single)["tensors_not_compared"] == 1
    # ... nor is one of another type with numbers of the same width: 1.0 is 0x3C00 as a float16 and 0x3F80 as a bfloat16.
    other_half = _write(tmp_path / "other_half.safetensors", [("w", "BF16", (1,), struct.pack("<H", 0x3F80))])
    one = _write(tmp_path / "one.safetensors", [("w", "F16", (1,), _of("F16", [1.0]))])
    assert _of("F16", [1.0]) == struct.pack("<H", 0x3C00) and compared(other_half, one)["elements_changed"] == 0 and compared(other_half, one)["tensors"] == 0


def test_bfloat16_and_whole_numbers_are_compared_by_their_bits(tmp_path):
    # bfloat16 is the upper half of a float32: 0x3F80 is 1.0, 0x4000 is 2.0, 0xBF00 is -0.5.
    start = _write(tmp_path / "start.safetensors", [("w", "BF16", (3,), struct.pack("<3H", 0x3F80, 0x4000, 0xBF00)), ("steps", "I64", (2,), _of("I64", [3, 4]))])
    saved = _write(tmp_path / "saved.safetensors", [("w", "BF16", (3,), struct.pack("<3H", 0x3F80, 0x3F80, 0xBF00)), ("steps", "I64", (2,), _of("I64", [3, 5]))])
    result = compared(saved, start)
    assert (result["tensors"], result["elements"], result["elements_changed"], result["largest_change"]) == (2, 5, 2, 1.0)        # 2.0 to 1.0; the whole number is counted, not measured
    assert compared(start, start)["elements_changed"] == 0
    # A tensor with no number, one whose numbers have a width no type has (three bytes), and one whose bytes are not a whole number of its elements
    # (nine bytes for two numbers) are not compared.
    odd = _write(tmp_path / "odd.safetensors", [("empty", "F32", (0, 4), b""), ("of_three_bytes", "F32", (2,), b"\x00" * 6), ("torn", "F32", (2,), b"\x00" * 9)])
    assert compared(odd, odd) == {"how": result["how"], "tensors": 0, "tensors_not_compared": 3, "elements": 0, "elements_changed": 0, "largest_change": None}
