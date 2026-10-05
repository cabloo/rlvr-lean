"""Spec fixture 3 and §5: the training-set size N, the three selection methods, the canonical proof."""

import math
import random

import pytest

from rlvr_lean.domain.selection import (
    DifficultyHeuristicSelection,
    HalfPassRateSelection,
    LearningProgressSelection,
    PoolConjecture,
    RandomSelection,
    TrainingSetSize,
    choose_canonical_proof,
    in_difficulty_band,
    selection_overlap,
    training_set_size,
)


def conjecture(identifier, verified_count, sample_count=12, learning_progress=None, learning_progress_cosine=None):
    return PoolConjecture(conjecture_id=identifier, sample_count=sample_count, verified_count=verified_count,
                          canonical_proof_id=f"proof-of-{identifier}", canonical_proof_tokens=40 + verified_count,
                          learning_progress=learning_progress, learning_progress_cosine=learning_progress_cosine)


def pool_with(band_count, above_band_count):
    """`band_count` conjectures at 1-3 of 12 verified, then `above_band_count` at 4-12 of 12."""
    band = [conjecture(f"band-{index:04d}", 1 + index % 3) for index in range(band_count)]
    above = [conjecture(f"above-{index:04d}", 4 + index % 9) for index in range(above_band_count)]
    return band + above


def ids(selection):
    return [member.conjecture_id for member in selection]


# --- N --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pool_size,band_size,expected", [
    (1500, 400, TrainingSetSize(400, False)),    # the band bounds N
    (3000, 900, TrainingSetSize(500, False)),    # the 500 cap
    (600, 300, TrainingSetSize(200, False)),     # a third of the pool
    (600, 150, TrainingSetSize(150, False)),     # exactly at the band minimum: the band still bounds N
    (600, 149, TrainingSetSize(200, True)),      # band too small: N ignores it and the heuristic fills
    (2000, 10, TrainingSetSize(500, True)),
    (2, 0, TrainingSetSize(0, True)),            # a tiny pool gives N = 0
    (0, 0, TrainingSetSize(0, True)),
])
def test_training_set_size(pool_size, band_size, expected):
    assert training_set_size(pool_size, band_size) == expected


def test_training_set_size_floors_an_exact_fraction():
    assert training_set_size(100, 100, pool_fraction=0.29, band_minimum=0).size == 29   # floats give 28
    assert training_set_size(1000, 1000, maximum=10_000, band_minimum=0).size == 333


@pytest.mark.parametrize("arguments", [
    {"pool_size": 10, "band_size": 11}, {"pool_size": -1, "band_size": 0},
    {"pool_size": 10, "band_size": 1, "pool_fraction": 0.0}, {"pool_size": 10, "band_size": 1, "maximum": -1},
])
def test_training_set_size_rejects_invalid_input(arguments):
    with pytest.raises(ValueError):
        training_set_size(**arguments)


# --- the pool -------------------------------------------------------------------------------------------

def test_pool_holds_solvable_conjectures_only():
    with pytest.raises(ValueError):
        conjecture("unsolved", 0)
    with pytest.raises(ValueError):
        conjecture("impossible", 13)
    with pytest.raises(ValueError):
        conjecture("bad-score", 1, learning_progress=math.nan)
    assert conjecture("ok", 3).pass_rate == 0.25


def test_band_is_zero_exclusive_to_a_quarter_inclusive():
    assert [in_difficulty_band(conjecture("x", verified)) for verified in (1, 2, 3, 4, 12)] == [True, True, True, False, False]
    assert in_difficulty_band(conjecture("x", 2, sample_count=8))          # exactly 0.25
    assert not in_difficulty_band(conjecture("x", 3, sample_count=8))


# --- difficulty heuristic -------------------------------------------------------------------------------

def test_heuristic_never_leaves_the_band_while_the_band_holds_at_least_n():
    pool = pool_with(band_count=300, above_band_count=300)
    size = training_set_size(len(pool), band_size=300)
    assert size == TrainingSetSize(200, False)
    selected = DifficultyHeuristicSelection().select(pool, size.size, seed=0)
    assert len(selected) == 200
    assert all(0 < member.pass_rate <= 0.25 for member in selected)
    # Ascending pass rate: all 100 at 1/12 first, then 100 of the 2/12 members.
    assert [member.verified_count for member in selected] == [1] * 100 + [2] * 100


def test_heuristic_breaks_ties_by_seed_and_ignores_input_order():
    pool = [conjecture(f"tied-{index:02d}", 2) for index in range(30)]
    heuristic = DifficultyHeuristicSelection()
    first = ids(heuristic.select(pool, 10, seed=0))
    assert first == ids(heuristic.select(pool, 10, seed=0))
    assert first == ids(heuristic.select(list(reversed(pool)), 10, seed=0))
    shuffled = pool[:]
    random.Random(3).shuffle(shuffled)
    assert first == ids(heuristic.select(shuffled, 10, seed=0))
    assert first != ids(heuristic.select(pool, 10, seed=1))


def test_heuristic_fills_upward_only_below_the_band_minimum():
    at_minimum = pool_with(band_count=150, above_band_count=300)
    size = training_set_size(len(at_minimum), band_size=150)
    assert size == TrainingSetSize(150, False)
    assert all(in_difficulty_band(member) for member in DifficultyHeuristicSelection().select(at_minimum, size.size, seed=0))

    short_band = pool_with(band_count=149, above_band_count=301)
    size = training_set_size(len(short_band), band_size=149)
    assert size == TrainingSetSize(150, True)
    selected = DifficultyHeuristicSelection().select(short_band, size.size, seed=0)
    fill_ins = [member for member in selected if not in_difficulty_band(member)]
    assert len(selected) == 150 and len(fill_ins) == 1
    assert fill_ins[0].verified_count == 4      # the lowest pass rate above the band
    assert ids(selected[:149]) == ids([member for member in selected if in_difficulty_band(member)])


def test_fill_ins_come_in_ascending_pass_rate():
    pool = [conjecture("band-a", 1), conjecture("high", 12), conjecture("mid", 6), conjecture("low", 4), conjecture("low-2", 4)]
    selected = DifficultyHeuristicSelection().select(pool, 4, seed=0)
    assert [member.verified_count for member in selected] == [1, 4, 4, 6]


def test_heuristic_adds_no_fill_ins_when_the_band_still_covers_n():
    # 140 in the band is under the minimum, so the fill rule is in force, but N = 100 fits in the band.
    pool = pool_with(band_count=140, above_band_count=160)
    size = training_set_size(len(pool), band_size=140)
    assert size == TrainingSetSize(100, True)
    assert all(in_difficulty_band(member) for member in DifficultyHeuristicSelection().select(pool, size.size, seed=0))


# --- random ---------------------------------------------------------------------------------------------

def test_random_is_a_seeded_permutation_independent_of_input_order():
    pool = pool_with(band_count=20, above_band_count=20)
    random_selection = RandomSelection()
    everything = random_selection.select(pool, len(pool), seed=4)
    assert sorted(ids(everything)) == sorted(ids(pool)) and ids(everything) != sorted(ids(pool))
    assert ids(everything) == ids(random_selection.select(list(reversed(pool)), len(pool), seed=4))
    assert ids(random_selection.select(pool, 10, seed=4)) == ids(everything)[:10]
    assert ids(random_selection.select(pool, 10, seed=5)) != ids(everything)[:10]


# --- learning progress ----------------------------------------------------------------------------------

def test_learning_progress_ranks_by_descending_score_with_id_tie_breaks():
    pool = [conjecture("c", 1, learning_progress=0.5, learning_progress_cosine=0.1),
            conjecture("a", 2, learning_progress=0.9, learning_progress_cosine=0.2),
            conjecture("b", 3, learning_progress=0.5, learning_progress_cosine=0.9),
            conjecture("d", 4, learning_progress=-1.0, learning_progress_cosine=0.0)]
    assert ids(LearningProgressSelection().select(pool, 3, seed=0)) == ["a", "b", "c"]
    assert ids(LearningProgressSelection(use_cosine=True).select(pool, 2, seed=0)) == ["b", "a"]
    assert LearningProgressSelection().name == "learning_progress"
    assert LearningProgressSelection(use_cosine=True).name == "learning_progress_cosine"


def test_learning_progress_requires_every_score():
    pool = [conjecture("a", 1, learning_progress=0.1), conjecture("b", 1)]
    with pytest.raises(ValueError):
        LearningProgressSelection().select(pool, 1, seed=0)
    with pytest.raises(ValueError):
        LearningProgressSelection(use_cosine=True).select(pool[:1], 1, seed=0)


# --- half pass rate (spec fixture 11, §13) --------------------------------------------------------------

def test_half_pass_rate_takes_one_half_first_then_the_next_closest_on_either_side():
    pool = [conjecture(f"c-{verified:02d}", verified) for verified in range(1, 13)]      # 1..12 of 12
    selected = HalfPassRateSelection().select(pool, 3, seed=0)
    assert ids(selected)[0] == "c-06"
    assert set(ids(selected)[1:]) == {"c-05", "c-07"}
    distances = [abs(2 * member.verified_count - member.sample_count) for member in HalfPassRateSelection().select(pool, 12, seed=0)]
    assert distances == sorted(distances)                 # exact: twice the distance from 0.5, in twelfths


def test_half_pass_rate_breaks_ties_by_seed_and_ignores_input_order():
    pool = [conjecture(f"low-{index:03d}", 5) for index in range(20)] + [conjecture(f"high-{index:03d}", 7) for index in range(20)]
    first = ids(HalfPassRateSelection().select(pool, 10, seed=1))
    assert first == ids(HalfPassRateSelection().select(list(reversed(pool)), 10, seed=1))
    assert first != ids(HalfPassRateSelection().select(pool, 10, seed=2))
    assert any(identifier.startswith("low") for identifier in first) and any(identifier.startswith("high") for identifier in first)


# --- every method ---------------------------------------------------------------------------------------

def scored_pool():
    pool = pool_with(band_count=40, above_band_count=60)
    return [PoolConjecture(member.conjecture_id, member.sample_count, member.verified_count, member.canonical_proof_id,
                           member.canonical_proof_tokens, learning_progress=(index * 37 % 101) / 101,
                           learning_progress_cosine=(index * 53 % 103) / 103)
            for index, member in enumerate(pool)]


ALL_METHODS = [RandomSelection(), DifficultyHeuristicSelection(), LearningProgressSelection(),
               LearningProgressSelection(use_cosine=True), HalfPassRateSelection()]


@pytest.mark.parametrize("method", ALL_METHODS, ids=lambda method: method.name)
@pytest.mark.parametrize("size", [0, 1, 33, 100, 250])
def test_every_method_returns_min_n_pool_distinct_members_deterministically(method, size):
    pool = scored_pool()
    selected = method.select(pool, size, seed=7)
    assert len(selected) == min(size, len(pool))
    assert len(set(ids(selected))) == len(selected)
    assert ids(selected) == ids(method.select(list(reversed(pool)), size, seed=7))


@pytest.mark.parametrize("method", ALL_METHODS, ids=lambda method: method.name)
def test_every_method_rejects_duplicate_ids_and_negative_sizes(method):
    pool = scored_pool()
    with pytest.raises(ValueError):
        method.select(pool + pool[:1], 5, seed=0)
    with pytest.raises(ValueError):
        method.select(pool, -1, seed=0)


def test_method_names():
    assert [method.name for method in ALL_METHODS] == [
        "random", "difficulty_heuristic", "learning_progress", "learning_progress_cosine", "half_pass_rate"]


def test_selection_overlap_is_jaccard_over_ids():
    pool = scored_pool()
    assert selection_overlap(pool[:10], pool[5:15]) == pytest.approx(5 / 15)
    assert selection_overlap(pool[:10], list(reversed(pool[:10]))) == 1.0
    assert selection_overlap(pool[:10], pool[10:20]) == 0.0
    assert selection_overlap([], []) == 1.0


# --- canonical proof ------------------------------------------------------------------------------------

def test_canonical_proof_is_the_shortest_with_smallest_id_on_ties():
    assert choose_canonical_proof([("attempt-c", 90), ("attempt-b", 40), ("attempt-a", 55)]) == "attempt-b"
    assert choose_canonical_proof([("attempt-c", 40), ("attempt-b", 40), ("attempt-d", 12_000)]) == "attempt-b"
    assert choose_canonical_proof([("only", 7)]) == "only"
    with pytest.raises(ValueError):
        choose_canonical_proof([])
    with pytest.raises(ValueError):
        choose_canonical_proof([("empty-proof", 0)])
