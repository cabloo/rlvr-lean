"""Spec fixture 5: every method's training set has exactly N examples, one per conjecture, and a conjecture
chosen by several methods maps to the identical example in each."""

import pytest

from rlvr_lean.domain.selection import (
    DifficultyHeuristicSelection,
    LearningProgressSelection,
    PoolConjecture,
    RandomSelection,
    training_set_size,
)
from rlvr_lean.domain.training import TrainingExample, build_training_examples


def pool():
    members = []
    for index in range(600):
        verified_count = 1 + index % 12
        members.append(PoolConjecture(
            conjecture_id=f"conjecture-{index:04d}", sample_count=12, verified_count=verified_count,
            canonical_proof_id=f"attempt-{index:04d}-shortest", canonical_proof_tokens=30 + index % 50,
            learning_progress=((index * 7919) % 600) / 600))
    return members


def test_every_method_trains_on_exactly_n_examples_and_shared_conjectures_share_the_example():
    conjectures = pool()
    band_size = sum(member.pass_rate <= 0.25 for member in conjectures)
    size = training_set_size(len(conjectures), band_size).size
    assert size == 150

    training_sets = {
        method.name: build_training_examples(method.select(conjectures, size, seed=0))
        for method in (RandomSelection(), DifficultyHeuristicSelection(), LearningProgressSelection())
    }
    example_for_conjecture = {}
    for examples in training_sets.values():
        assert len(examples) == size
        assert len({example.conjecture_id for example in examples}) == size
        for example in examples:
            # The same conjecture always contributes the same (canonical) proof, whichever method chose it.
            assert example_for_conjecture.setdefault(example.conjecture_id, example) == example

    shared = ({example.conjecture_id for example in training_sets["random"]}
              & {example.conjecture_id for example in training_sets["difficulty_heuristic"]})
    assert shared, "the fixture should make the random and heuristic sets overlap"


def test_examples_follow_selection_order_and_use_the_canonical_proof():
    selected = pool()[:3]
    assert build_training_examples(selected) == [
        TrainingExample("conjecture-0000", "attempt-0000-shortest"),
        TrainingExample("conjecture-0001", "attempt-0001-shortest"),
        TrainingExample("conjecture-0002", "attempt-0002-shortest"),
    ]
    assert build_training_examples([]) == []


def test_a_conjecture_selected_twice_is_rejected():
    selected = pool()[:3]
    with pytest.raises(ValueError):
        build_training_examples(selected + selected[:1])
