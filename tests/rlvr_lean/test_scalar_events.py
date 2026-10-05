"""Spec fixture 13: what the stdlib `ScalarEventWriter` writes reads back through TensorBoard's own reader."""

import pytest

from rlvr_lean.runner.heartbeat import ScalarEventWriter

event_accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")


def test_scalars_round_trip_through_tensorboard(tmp_path):
    writer = ScalarEventWriter(tmp_path / "random_seed0")
    writer.scalars(0, {"heldout/loss_nf4": 0.941})
    for step, loss in enumerate([2.2324, 1.431, 0.0183], start=1):
        writer.scalars(step, {"train/loss": loss})
    writer.scalars(3, {"heldout/loss_nf4": 0.2205})
    accumulator = event_accumulator.EventAccumulator(str(tmp_path / "random_seed0"))
    accumulator.Reload()
    assert sorted(accumulator.Tags()["scalars"]) == ["heldout/loss_nf4", "train/loss"]
    train = accumulator.Scalars("train/loss")
    assert [event.step for event in train] == [1, 2, 3]
    assert [event.value for event in train] == pytest.approx([2.2324, 1.431, 0.0183], rel=1e-6)
    assert [(event.step, round(event.value, 4)) for event in accumulator.Scalars("heldout/loss_nf4")] == [(0, 0.941), (3, 0.2205)]
