"""Two stored adapters compared from their files alone (`infrastructure/adapter_files.py`): what L4's check "the adapter
differs from the start adapter's" reads; and THE MEAN OF SEVERAL ADAPTERS, written and read back (L5n's two averaged
models: spec "L5n: what is one training worth?"). The files here are written BY HAND, from the format's definition
(eight bytes of header length, a JSON header, the tensors' bytes), not by the reader under test. No model, no torch."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from rlvr_lean.infrastructure.adapter_files import ADAPTER_FILE, MEAN_TYPES, compared, is_the_mean, mean_of, tensors_of

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


# ------------------------------------------------------------------------------- the mean of several adapters
OF_A = ([[0.5, -1.25, 3.0], [0.0, 7.5, -2.0]], [[1.5, -1.0, 3.0], [1.0, 7.0, -2.0]], [[0.5, 0.25, 3.0], [2.0, 7.5, 6.0]], [[1.5, 1.0, 3.0], [1.0, 8.0, 2.0]])
OF_B = ([[0.125, 0.25], [-0.5, 1.0], [2.0, -4.0]], [[0.375, 0.25], [0.5, 1.0], [2.0, 4.0]], [[0.125, 0.75], [-0.5, 3.0], [2.5, -4.0]], [[0.375, 0.75], [0.5, 3.0], [2.5, 0.0]])
MEAN_OF_A, MEAN_OF_B = [1.0, -0.25, 3.0, 1.0, 7.5, 1.0], [0.25, 0.5, 0.0, 2.0, 2.25, -1.0]      # every number's mean over the four, by hand


def _numbers(path: Path, name: str, kind: str = "<f4") -> np.ndarray:
    """One tensor's numbers, read by hand from a file's header and bytes."""
    header, begins = tensors_of(path)
    begin, end = header[name]["data_offsets"]
    return np.frombuffer(path.read_bytes()[begins + begin:begins + end], dtype=kind)


def _four(directory: Path, kind: str = "F32") -> list[Path]:
    """Four made-up adapters of two tensors: the same names, type and shapes, other numbers."""
    directory.mkdir(parents=True, exist_ok=True)
    return [_write(directory / f"adapter_{index}.safetensors", [(A[0], kind, (2, 3), _of(kind, a)), (B[0], kind, (3, 2), _of(kind, b))], metadata={"format": "pt"})
            for index, (a, b) in enumerate(zip(OF_A, OF_B))]


def test_the_mean_of_four_adapters_is_every_numbers_mean_in_the_first_adapters_own_file_layout(tmp_path):
    four = _four(tmp_path)
    before = [path.read_bytes() for path in four]
    saved = tmp_path / "mean.safetensors"
    made = mean_of(four, saved)
    assert made == {"how": made["how"], "adapters": 4, "tensors": 2, "elements": 12} and "summed in float64 and rounded once to the tensor's own type" in made["how"]
    # EVERY NUMBER IS THE MEAN OF THE FOUR at its place: by hand.
    assert _numbers(saved, A[0]).tolist() == MEAN_OF_A and _numbers(saved, B[0]).tolist() == MEAN_OF_B
    # The file is the FIRST adapter's, byte for byte, but for the tensors' numbers: its header, its metadata, its places. Nothing is left beside it, and the four were only read.
    assert tensors_of(saved) == tensors_of(four[0]) and saved.stat().st_size == four[0].stat().st_size and saved.read_bytes()[:tensors_of(saved)[1]] == before[0][:tensors_of(four[0])[1]]
    assert sorted(path.name for path in tmp_path.iterdir()) == [*(f"adapter_{index}.safetensors" for index in range(4)), "mean.safetensors"] and [path.read_bytes() for path in four] == before
    # READ BACK and HELD: every number is that mean, bit for bit, and it is none of the four.
    held = is_the_mean(saved, four)
    assert held == {"how": made["how"], "adapters": 4, "tensors": 2, "tensors_of_the_file_alone": 0, "elements": 12, "elements_not_the_mean": 0,
                    "elements_not_those_of_each_adapter": [compared(saved, path)["elements_changed"] for path in four], "is_the_mean": True, "is_none_of_them": True, "passes": True}
    assert all(0 < count <= 12 for count in held["elements_not_those_of_each_adapter"])
    # WHICH adapters it is the mean of matters: of the first three it is another file; of the four in another order, the same numbers.
    of_three = tmp_path / "of_three.safetensors"
    assert mean_of(four[:3], of_three)["adapters"] == 3 and _numbers(of_three, A[0]).tolist()[:2] == [float(np.float32(2.5 / 3)), float(np.float32(-2.0 / 3))]
    assert is_the_mean(of_three, four)["is_the_mean"] is False and is_the_mean(saved, four[:3])["elements_not_the_mean"] > 0 and is_the_mean(of_three, four[:3])["passes"] is True
    mean_of(four[::-1], tmp_path / "reversed.safetensors")
    assert compared(tmp_path / "reversed.safetensors", saved)["elements_changed"] == 0 and is_the_mean(saved, four[::-1])["passes"] is True


def test_the_mean_is_summed_in_float64_and_rounded_once_to_the_tensors_own_type(tmp_path):
    # Numbers whose mean is no float32 (0.1, 0.2, 0.3 and 0.7 as float32 are not those decimals): the mean is rounded ONCE, from float64, to the nearest float32.
    values = [[0.1, 1e-8, 16777216.0], [0.2, 3e-8, 1.0], [0.3, 5e-8, 1.0], [0.7, 7e-8, 1.0]]
    four = [_write(tmp_path / f"adapter_{index}.safetensors", [("w", "F32", (3,), _of("F32", own))]) for index, own in enumerate(values)]
    mean_of(four, tmp_path / "mean.safetensors")
    exact = np.mean([np.asarray(own, dtype="<f4").astype(np.float64) for own in values], axis=0)
    saved = _numbers(tmp_path / "mean.safetensors", "w")
    assert saved.tolist() == exact.astype("<f4").tolist() and float((np.abs(saved.astype(np.float64) - exact) / np.abs(exact)).max()) < 6e-8       # within half a unit in the last place
    # A sum made in float32 would lose the three ones beside 2^24: the mean here keeps them.
    assert float(np.float32(16777216.0) + np.float32(1.0)) == 16777216.0 and saved[2] == np.float32(16777219.0 / 4) and is_the_mean(tmp_path / "mean.safetensors", four)["passes"] is True
    # Float16 and float64 adapters are averaged in their own types.
    for kind in ("F16", "F64"):
        own = _four(tmp_path / kind, kind)
        mean_of(own, tmp_path / kind / "mean.safetensors")
        assert _numbers(tmp_path / kind / "mean.safetensors", A[0], TYPES[kind]).tolist() == MEAN_OF_A and is_the_mean(tmp_path / kind / "mean.safetensors", own)["passes"] is True
        assert tensors_of(tmp_path / kind / "mean.safetensors")[0][A[0]]["dtype"] == kind
    assert MEAN_TYPES == ("F16", "F32", "F64")


def test_a_file_that_is_not_the_mean_or_is_one_of_its_adapters_is_held_to_be_neither(tmp_path):
    four = _four(tmp_path)
    saved = tmp_path / "mean.safetensors"
    mean_of(four, saved)
    of_a, of_b = _numbers(saved, A[0]).tobytes(), _numbers(saved, B[0]).tobytes()
    # ONE number off by the smallest step: counted, and the file is not the mean.
    numbers = _numbers(saved, B[0]).copy()
    numbers[4] = np.nextafter(numbers[4], np.float32(9.0), dtype=np.float32)
    off = _write(tmp_path / "off.safetensors", [(A[0], "F32", (2, 3), of_a), (B[0], "F32", (3, 2), numbers.tobytes())], metadata={"format": "pt"})
    held = is_the_mean(off, four)
    assert (held["elements_not_the_mean"], held["is_the_mean"], held["is_none_of_them"], held["passes"]) == (1, False, True, False)
    # ONE OF THE FOUR in the mean's place: it is one of them (no number of it is not that adapter's), and it is not the mean.
    one = is_the_mean(four[2], four)
    assert (one["elements_not_those_of_each_adapter"][2], one["is_none_of_them"], one["is_the_mean"], one["passes"]) == (0, False, False, False)
    # Adapters that are ONE adapter: their mean is each of them, and that is said.
    same = [four[0]] * 4
    mean_of(same, tmp_path / "same.safetensors")
    alike = is_the_mean(tmp_path / "same.safetensors", same)
    assert (alike["is_the_mean"], alike["elements_not_those_of_each_adapter"], alike["is_none_of_them"], alike["passes"]) == (True, [0, 0, 0, 0], False, False)
    # A file that lacks a tensor, or holds it with another shape, is not the mean: the tensor counts whole. Nor is one that holds a tensor of its own alone.
    lacking = _write(tmp_path / "lacking.safetensors", [(A[0], "F32", (2, 3), of_a)])
    assert (is_the_mean(lacking, four)["elements_not_the_mean"], is_the_mean(lacking, four)["passes"]) == (6, False)
    reshaped = _write(tmp_path / "reshaped.safetensors", [(A[0], "F32", (3, 2), of_a), (B[0], "F32", (3, 2), of_b)])
    assert is_the_mean(reshaped, four)["elements_not_the_mean"] == 6
    more = _write(tmp_path / "more.safetensors", [(A[0], "F32", (2, 3), of_a), (B[0], "F32", (3, 2), of_b), ("its_own", "F32", (1,), _of("F32", [1.0]))])
    found = is_the_mean(more, four)
    assert (found["elements_not_the_mean"], found["tensors_of_the_file_alone"], found["is_the_mean"], found["passes"]) == (0, 1, False, False)


def test_adapters_that_do_not_hold_the_same_tensors_are_refused_and_nothing_is_written(tmp_path):
    four = _four(tmp_path)

    def refused(files, match):
        with pytest.raises(ValueError, match=match):
            mean_of(files, tmp_path / "mean.safetensors")
        with pytest.raises(ValueError, match=match):
            is_the_mean(four[0], files)
        assert not (tmp_path / "mean.safetensors").exists() and not list(tmp_path.glob("*.tmp"))            # NOTHING WAS WRITTEN

    refused(four[:1], "the mean of adapters is made of at least two, and 1 was given")
    refused([], "the mean of adapters is made of at least two, and 0 was given")
    alone = _write(tmp_path / "alone.safetensors", [A, B, ("only_here", "F32", (1,), _of("F32", [1.0]))])
    refused([four[0], alone, four[2]], "alone.safetensors and .*adapter_0.safetensors do not hold the same tensors .1 are in one alone; first: only_here.")
    refused([alone, four[1]], "adapter_1.safetensors and .*alone.safetensors do not hold the same tensors")
    other_type = _write(tmp_path / "other_type.safetensors", [(A[0], "F64", (2, 3), _of("F64", [[0.0] * 3] * 2)), B])
    refused([four[0], four[1], other_type], "other_type.safetensors holds 1 tensors with another type or shape than .*adapter_0.safetensors .first: .*lora_A.weight, F64 .2, 3. against F32 .2, 3.")
    other_shape = _write(tmp_path / "other_shape.safetensors", [(A[0], "F32", (3, 2), A[3]), B])
    refused([four[0], other_shape], "other_shape.safetensors holds 1 tensors with another type or shape")
    # A type the mean is not written in: bfloat16, whole numbers.
    for kind, data in (("BF16", struct.pack("<2H", 0x3F80, 0x4000)), ("I64", _of("I64", [3, 4]))):
        of_that_type = [_write(tmp_path / f"{kind}_{index}.safetensors", [("w", kind, (2,), data)]) for index in range(2)]
        refused(of_that_type, f"1 tensors of .*{kind}_0.safetensors are of a type the mean is not written in .first: w, {kind}.: it is made of F16, F32, F64")
    # A tensor whose bytes are not what its type and shape say is refused; a tensor with no number is no number of the mean.
    torn = [_write(tmp_path / f"torn_{index}.safetensors", [("w", "F32", (2,), b"\x00" * 16)]) for index in range(2)]
    refused(torn, "the tensor w of .*torn_0.safetensors does not hold the bytes its type .F32. and shape")
    empty = [_write(tmp_path / f"empty_{index}.safetensors", [("none", "F32", (0, 4), b""), ("w", "F32", (1,), _of("F32", [float(index)]))]) for index in range(2)]
    assert mean_of(empty, tmp_path / "of_empty.safetensors")["elements"] == 1 and _numbers(tmp_path / "of_empty.safetensors", "w").tolist() == [0.5]
    assert is_the_mean(tmp_path / "of_empty.safetensors", empty)["passes"] is True
