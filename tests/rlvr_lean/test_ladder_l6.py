"""L6, a second round: search 2's two parts and the partition they are held to, too few round-2 solves, search 2's table,
the bins of a proof's lines and the ordinary solves matched in length (the shortfall filled from the next shorter bin), the
four sets of 2N rows and what they are held to (the published rows the first of the one hash order, nested), the three reads
on the held-aside set with their names and their edges at zero, reach's two conditions, the spread L5's seeds measured, what
follows by the scout's table, and the report on hand-made rows with a known answer (the loss rise printed and deciding
nothing, the adapter half deciding). Spec: docs/spec/ladder-loop.spec.md, "L6: a second round. Does the cycle
compound?". Pure: no model, no Lean. The stage end to end is `test_ladder_l6_stage.py`."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import CHANGED  # noqa: E402
from test_ladder_l4_dose import MORE, PRETRAIN_HALF  # noqa: E402
from test_ladder_l4_rows import FALLING, SAMPLINGS  # noqa: E402
from test_ladder_l4_rows2 import PRETRAINING, _draw_order, _solved  # noqa: E402 - 40 rows the start model was pretrained on, without text
from test_ladder_l5 import AGAIN, ASIDE, BASE, HA_BASE, HA_LESS, HA_MORE, LESS, _aside, _model  # noqa: E402 - L5's report's hand-made world

from rlvr_lean.domain.ladder_round import l4_rows, l5  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import SIGN_TEST_LEVEL, THE_MINIMUM, THE_RULE, rehearsal_rows  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import NOT_SEPARATED, TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l6 import (  # noqa: E402
    COMPOUNDING,
    CYCLE_ADDS,
    DOES_NOT_CLEAR,
    FOLLOWS,
    HARD_TO_FIND,
    IN_BETWEEN,
    IT_COSTS,
    LENGTH,
    LONG_DO_BETTER,
    LONG_LABEL,
    LONG_ORDINARY,
    MODELS,
    NOT_FROM_LENGTH,
    NOT_SHOWN,
    ONE_ROUND,
    ORDINARY,
    ORIGINS,
    OWN,
    PRIMARY,
    PUBLISHED2,
    REACH,
    REACH_SHOWN,
    ROUND_1,
    ROUND_2,
    SEARCH_LABEL,
    SECOND_ROUND_ADDS,
    SPREAD_NOT_GIVEN,
    TWO_FURTHER_SEEDS,
    TWO_ROUNDS,
    bin_of,
    compounding,
    enough_round_two_solves,
    hard_or_long,
    hold_the_sets,
    line_bins,
    matched_in_length,
    no_barred_row,
    primary_branch,
    reach,
    second_search,
    second_search_table,
    spread_of_l5,
    the_sets,
    what_follows,
)
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import attempts_on_g, per_attempt, solved  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l5 import held_aside_key, pair_name  # noqa: E402
from rlvr_lean.reporting.ladder_l6 import CANNOT_SAY, IS, LONG, PAIRS, READS, THE_ADAPTER, THE_HELD_ASIDE, THE_SEARCH, build_l6_report  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS, OWN_SETTINGS = CONFIG["ladder_loop"]["l4"], CONFIG["ladder_loop"]["l6"]
SEED = 0
EDGES = OWN_SETTINGS["long_ordinary"]["line_bins"]
NOT_MEASURED = {"problems": 0, "mean": None, "low": None, "high": None}


def test_the_names_and_the_settings_are_the_specs():
    assert MODELS == (PUBLISHED2, ONE_ROUND, TWO_ROUNDS, LONG_ORDINARY) == ("published2", "one_round", "two_rounds", "long_ordinary")      # trained and measured in this order
    assert OWN == (ROUND_1, ROUND_2, ORDINARY) and ORIGINS == {PUBLISHED2: (), ONE_ROUND: (ROUND_1,), TWO_ROUNDS: (ROUND_1, ROUND_2), LONG_ORDINARY: (ORDINARY,)}
    assert PRIMARY == (CYCLE_ADDS, NOT_SEPARATED, IT_COSTS, INCONCLUSIVE, NOT_READ) == ("THE CYCLE ADDS WHAT THE SAME NUMBER OF PUBLISHED PROOFS DOES NOT", "NOT SEPARATED AT THIS SIZE", "IT COSTS",
                                                                                         INCONCLUSIVE, NOT_READ)
    assert COMPOUNDING == (SECOND_ROUND_ADDS, NOT_SHOWN, IT_COSTS, INCONCLUSIVE, NOT_READ) and (SECOND_ROUND_ADDS, NOT_SHOWN) == ("THE SECOND ROUND ADDS", "NOT SHOWN")
    assert LENGTH == (HARD_TO_FIND, NOT_FROM_LENGTH, LONG_DO_BETTER, INCONCLUSIVE, NOT_READ) == ("HARD TO FIND MATTERS BEYOND LENGTH", "NOT SEPARATED FROM LENGTH", "LONG ORDINARY SOLVES DO BETTER",
                                                                                               INCONCLUSIVE, NOT_READ)
    assert REACH == (REACH_SHOWN, NOT_SHOWN, INCONCLUSIVE, NOT_READ) and REACH_SHOWN == "REACH SHOWN AT THIS SEED"
    assert FOLLOWS == (TWO_FURTHER_SEEDS, DOES_NOT_CLEAR, IN_BETWEEN, SPREAD_NOT_GIVEN, NOT_READ) and "TWO FURTHER SEEDS" in TWO_FURTHER_SEEDS and "DOES NOT CLEAR THE BAR" in DOES_NOT_CLEAR
    assert len({SEARCH_LABEL, LONG_LABEL, l5.SEARCH_LABEL, l5.CAP_LABEL, l5.SHALLOW_LABEL, l5.HELD_ASIDE_LABEL, l4_rows.REHEARSAL_LABEL, l4_rows.KEEP_LABEL}) == 8
    # The section's numbers, in the config and nowhere in the code: 64 attempts, at least 150 round-2 solves, the bins 1, 2 to 3, 4 to 7, 8 to 15, 16 or more; an ordinary solve
    # here is a map problem solved in AT LEAST one map attempt. ONE search 2, by the task of seed 0.
    assert OWN_SETTINGS == {"start": "pre_r64", "models": list(MODELS), "search_seed": 0, "search": {"attempts": 64, "sampling_place": 52, "minimum": 150},
                            "long_ordinary": {"solved_at_least": 1, "line_bins": [1, 2, 4, 8, 16]}}
    of_l5 = CONFIG["ladder_loop"]["l5"]
    assert OWN_SETTINGS["start"] == of_l5["start"] and OWN_SETTINGS["search_seed"] == of_l5["search_seed"] and OWN_SETTINGS["search"]["attempts"] == CONFIG["ladder_loop"]["base_map"]["episodes"] + of_l5["deep"]["attempts"]
    # SEARCH 2'S SAMPLING SEED is no stored sampling's: its place is none of the places a measurement has, nor L5's two, and at no seed is it another's.
    from rlvr_lean.gpu import ladder_l2, ladder_l3a, ladder_l3a2, ladder_l3c
    places = {0, 1, 2, 3, *(ladder_l2.ROUND_SEED_PLACE + number for number in range(1, 7)), ladder_l2.CONTROL_SEED_PLACE, ladder_l3a.SEED_PLACE, ladder_l3a2.SEED_PLACE,
              ladder_l3c.SEED_PLACE, ladder_l3c.PILOT_SEED_PLACE, of_l5["deep"]["sampling_place"], of_l5["deep"]["held_aside_sampling_place"]}
    place, first = OWN_SETTINGS["search"]["sampling_place"], CONFIG["ladder_loop"]["round"]["sampling_seed"]
    others = {CONFIG["ladder_loop"]["goal"]["sampling_seed"], CONFIG["ladder_loop"]["base_map"]["sampling_seed"], *SETTINGS["rows"]["sampling_seeds_hot"]}
    assert place == 52 and place not in places and not {first + 100 * seed + place for seed in range(10)} & {*others, *(first + 100 * seed + taken for seed in range(10) for taken in places)}
    assert set(IS) == set(MODELS) and len(CANNOT_SAY) == 6 and "One seed" in CANNOT_SAY[0] and "One M(1) made search 2" in CANNOT_SAY[1] and "(a) and (b)" in CANNOT_SAY[2]
    assert "second pass over rows the start model has seen" in CANNOT_SAY[3] and "0.007 per attempt" in CANNOT_SAY[4]
    assert READS == (("primary", TWO_ROUNDS, PUBLISHED2), ("compounding", TWO_ROUNDS, ONE_ROUND), ("hard_to_find_or_long", TWO_ROUNDS, LONG_ORDINARY)) and len(PAIRS) == 6 == len(set(PAIRS))
    assert PAIRS[:3] == tuple((after, before) for _, after, before in READS) and {frozenset(pair) for pair in PAIRS} == {frozenset((a, b)) for a in MODELS for b in MODELS if a != b} and LONG == "8+"


# ------------------------------------------------------------------------------------------------ search 2
# A hand-made map of 24 base-map problems, 8 attempts each: ten failed every time (the candidates), the others solved once or more.
MAP_K = [0] * 10 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8]
MAP_ROWS = [{"problem_id": f"b{index}", "resolved": k, "episodes": 8, "sides": 1, "attempts_without_an_answer": 0} for index, k in enumerate(MAP_K)]
_taken = lambda ids, why=THE_RULE: [{"problem_id": key, "map_k": MAP_K[int(key[1:])], "map_n": 8, "taken_by": why} for key in ids]      # noqa: E731
DRAWN, HELD = _taken(["b3", "b0", "b7", "b5", "b1"]), _taken(["b9", "b2"])     # search 1 attempted five candidates, in ITS draw's order; two are held aside; b4, b6, b8 are in neither
SOLVED = ["b0", "b5"]                                                           # ... and solved two


def test_search_2_is_what_search_1_left_unsolved_and_the_candidates_neither_draw_took_and_never_a_held_aside_problem():
    found = second_search(MAP_ROWS, DRAWN, HELD, SOLVED, SEED)
    problems = found["problems"]
    # (a): the problems search 1 attempted and did NOT solve, in its draw's order. (b): the candidates in neither draw, in the map's order. Each failed 8 of 8, each by the rule.
    assert [(row["problem_id"], row["part"]) for row in problems] == [("b3", "a"), ("b7", "a"), ("b1", "a"), ("b4", "b"), ("b6", "b"), ("b8", "b")]
    assert all(row == {"problem_id": row["problem_id"], "part": row["part"], "map_k": 0, "map_n": 8, "taken_by": THE_RULE} for row in problems)
    ids = {row["problem_id"] for row in problems}
    assert not ids & {"b9", "b2"} and not ids & set(SOLVED) and not ids & {f"b{index}" for index in range(10, 24)}          # NO held-aside problem, NO round-1 solve, no problem the map's model solved
    # THE PARTITION, recorded: round 1's solves, (a), (b) and the held-aside set are the candidates, each once.
    assert found["partition"] == {"candidates": 10, "attempted_by_search_1": 5, "solved_by_search_1": 2, "a_left_unsolved_by_search_1": 3, "b_in_neither_draw": 3, "held_aside": 2,
                                  "taken_by_the_runs_minimum": 0}
    assert 2 + 3 + 3 + 2 == 10 and ids | set(SOLVED) | {"b9", "b2"} == {f"b{index}" for index in range(10)}
    # Search 1 solved NOTHING: all it attempted is (a). It solved everything and took, with the held-aside set, every candidate: search 2 has no problem.
    assert [row["problem_id"] for row in second_search(MAP_ROWS, DRAWN, HELD, [], SEED)["problems"]][:5] == [row["problem_id"] for row in DRAWN]
    every = _taken([f"b{index}" for index in range(8)])
    assert second_search(MAP_ROWS, every, HELD[:1] + _taken(["b8"]), [row["problem_id"] for row in every], SEED)["problems"] == []
    # HELD, each a refusal: a solve of a problem search 1 did not attempt; a problem both searched and held aside; one that stands twice; one a draw took by the rule that is no candidate.
    with pytest.raises(ValueError, match="round 1 does not agree with the map: 0 problems a draw took by the rule are no candidates .*1 problems search 1 solved are none it attempted, 0 are both "
                                         "searched and held aside, 0 stand twice .first: b4.. Refused"):
        second_search(MAP_ROWS, DRAWN, HELD, [*SOLVED, "b4"], SEED)
    with pytest.raises(ValueError, match="0 problems search 1 solved are none it attempted, 1 are both searched and held aside, 1 stand twice .first: b3."):
        second_search(MAP_ROWS, DRAWN, [*HELD, DRAWN[0]], SOLVED, SEED)
    with pytest.raises(ValueError, match="0 are both searched and held aside, 1 stand twice .first: b9."):
        second_search(MAP_ROWS, DRAWN, [*HELD, HELD[0]], SOLVED, SEED)
    with pytest.raises(ValueError, match="1 problems a draw took by the rule are no candidates .the map's model did not fail them in every attempt.*first: b12"):
        second_search(MAP_ROWS, [*DRAWN, *_taken(["b12"])], HELD, SOLVED, SEED)
    with pytest.raises(ValueError, match="1 problems a draw took by the rule are no candidates"):
        second_search(MAP_ROWS, DRAWN, [*HELD, *_taken(["b20"])], SOLVED, SEED)


def test_a_smoke_runs_search_2_takes_the_maps_other_problems_when_the_rule_gives_too_few_and_marks_them():
    # A fixture's map: every problem solved. Search 1 and the held-aside set each took one problem BY THE RUN'S MINIMUM; search 1 solved its own. The rule gives search 2 nothing.
    solved_all = [row for row in MAP_ROWS if row["resolved"] > 0]
    drawn, held = _taken(["b10"], THE_MINIMUM), _taken(["b11"], THE_MINIMUM)
    assert second_search(solved_all, drawn, held, ["b10"], SEED)["problems"] == []
    loosely = second_search(solved_all, drawn, held, ["b10"], SEED, at_least=2)
    left = sorted((row["problem_id"] for row in solved_all if row["problem_id"] not in ("b10", "b11")), key=lambda key: rank(SEED, SEARCH_LABEL, key))
    assert [row["problem_id"] for row in loosely["problems"]] == left[:2] and {(row["part"], row["taken_by"]) for row in loosely["problems"]} == {("b", THE_MINIMUM)}
    assert all(row["map_k"] > 0 for row in loosely["problems"]) and loosely["partition"]["candidates"] == 0 and loosely["partition"]["taken_by_the_runs_minimum"] == 2
    assert [row["problem_id"] for row in second_search(solved_all, drawn, held, ["b10"], 1, at_least=2)["problems"]] != left[:2]      # by the seed's own hash order
    # NEVER A HELD-ASIDE PROBLEM, were it the first of the hash order: it is passed over, and the next two join.
    passed_over = second_search(solved_all, drawn, _taken([left[0]], THE_MINIMUM), ["b10"], SEED, at_least=2)["problems"]
    in_order = sorted((row["problem_id"] for row in solved_all if row["problem_id"] != "b10"), key=lambda key: rank(SEED, SEARCH_LABEL, key))
    assert in_order[0] == left[0] and [row["problem_id"] for row in passed_over] == in_order[1:3] and left[0] not in {row["problem_id"] for row in passed_over}
    # Search 1 did NOT solve its problem: it is (a), carried as the minimum's, and one more joins. The rule gives enough: nothing joins.
    unsolved = second_search(solved_all, drawn, held, [], SEED, at_least=2)["problems"]
    assert [(row["problem_id"], row["part"], row["taken_by"]) for row in unsolved] == [("b10", "a", THE_MINIMUM), (left[0], "b", THE_MINIMUM)]
    assert second_search(MAP_ROWS, DRAWN, HELD, SOLVED, SEED, at_least=2) == second_search(MAP_ROWS, DRAWN, HELD, SOLVED, SEED)
    assert THE_MINIMUM == "the run's minimum" != THE_RULE


def test_fewer_round_two_solves_than_the_minimum_and_nothing_is_trained():
    assert enough_round_two_solves(150, 150) == {"what": "search 2 found at least 150 round-2 solves (fewer: nothing is trained, and the search is then the finding)", "round_two_solves": 150,
                                                 "minimum": 150, "passes": True}
    assert enough_round_two_solves(149, 150)["passes"] is False and enough_round_two_solves(0, 1)["passes"] is False and enough_round_two_solves(1, 1)["passes"] is True


def test_search_2s_table_counts_each_parts_solves_by_the_attempt_of_the_first_success_among_its_own():
    problems = second_search(MAP_ROWS, DRAWN, HELD, SOLVED, SEED)["problems"]
    results = [{"problem_id": row["problem_id"], "resolved": resolved, "episodes": 64, "sides": 2 if row["problem_id"] == "b3" else 1, "attempts_without_an_answer": 1 if row["problem_id"] == "b3" else 0}
               for row, resolved in zip(problems, (3, 0, 1, 2, 0, 1))]
    solves = [{"problem_id": key, "first_success": first} for key, first in (("b3", 1), ("b1", 8), ("b4", 9), ("b8", 64))]
    table = second_search_table(problems, results, solves, 64, 8)
    # The bands: 1 to the map's 8 attempts, then each ends at twice the one before. Each part by itself.
    assert table["a"] == {"problems": 3, "solved": 2, "share_solved": round(2 / 3, 5), "solved_by_first_success": {"1_to_8": 2, "9_to_16": 0, "17_to_32": 0, "33_to_64": 0}}
    assert table["b"] == {"problems": 3, "solved": 2, "share_solved": round(2 / 3, 5), "solved_by_first_success": {"1_to_8": 0, "9_to_16": 1, "17_to_32": 0, "33_to_64": 1}}
    assert (table["problems"], table["solved"], table["share_solved"], table["taken_by_the_rule"], table["taken_by_the_runs_minimum"]) == (6, 4, round(4 / 6, 5), 6, 0)
    assert (table["episodes_each"], table["episodes"], table["attempts"], table["successes"], table["per_1000"], table["without_an_answer"]) == (64, 384, 448, 7, round(7000 / 384, 2), 1)
    assert list(second_search_table(problems, results, solves, 20, 8)["a"]["solved_by_first_success"]) == ["1_to_8", "9_to_16", "17_to_20"]
    assert second_search_table([], [], [], 64, 8)["share_solved"] is None and second_search_table([], [], [], 64, 8)["per_1000"] is None


# ------------------------------------------------------------------------------- ordinary solves matched in length
def _own(problem_id, lines, origin=ROUND_1, **more):
    return {"id": f"{problem_id}#search#statement#{lines}", "attempt_id": f"{problem_id}#search#statement#{lines}", "problem_id": problem_id, "side": "statement", "theorem": f"theorem {problem_id} : P := by\n",
            "completion": "  step\n" * (lines - 1) + "  done\n", "kind": "lean_workbook", "published_side": "true", "origin": origin, "k": 1, "n": 64, "lines": lines, "first_success": 9,
            "map_k": 0, "taken_by": THE_RULE, **more}


def _offered(lines_by_problem):
    return [_own(key, lines, ORDINARY, id=f"{key}#map#statement#0", map_k=2) for key, lines in lines_by_problem.items()]


def test_the_bins_of_a_proofs_lines_are_the_settings_edges():
    assert line_bins(EDGES) == [("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, None)]
    assert [bin_of(lines, EDGES) for lines in (1, 2, 3, 4, 7, 8, 15, 16, 400)] == ["1", "2-3", "2-3", "4-7", "4-7", "8-15", "8-15", "16+", "16+"] and bin_of(0, EDGES) == "1"
    assert line_bins([1, 5]) == [("1-4", 1, 4), ("5+", 5, None)] and line_bins([1]) == [("1+", 1, None)] and bin_of(4, [1, 5]) == "1-4"
    for wrong in ([], [2, 4], [1, 4, 4], [1, 8, 4], [1, 2.5], [True, 2]):
        with pytest.raises(ValueError, match="the line bins' lower edges are .*: whole numbers that begin at 1 and rise"):
            line_bins(wrong)


def test_the_ordinary_solves_are_drawn_to_have_the_deep_rows_line_counts_bin_by_bin():
    own = [_own(f"d{index}", lines) for index, lines in enumerate((1, 2, 3, 5, 9, 12, 20))]                                          # by bin: 1, 2, 1, 2, 1
    offered = _offered({**{f"s{index}": 1 for index in range(3)}, **{f"t{index}": 2 + index % 2 for index in range(4)}, **{f"u{index}": 4 + index for index in range(4)},
                        **{f"v{index}": 8 + index for index in range(5)}, **{f"w{index}": 16 + 10 * index for index in range(3)}})
    taken, table = matched_in_length(own, offered, EDGES, SEED)
    assert [entry["lines"] for entry in table] == ["1", "2-3", "4-7", "8-15", "16+"]                                                 # the table, from the shortest bin
    assert [(entry["wanted"], entry["offered"], entry["asked_by_longer_bins"], entry["taken"], entry["shortfall"]) for entry in table] == [(1, 3, 0, 1, 0), (2, 4, 0, 2, 0), (1, 4, 0, 1, 0),
                                                                                                                                       (2, 5, 0, 2, 0), (1, 3, 0, 1, 0)]
    assert len(taken) == len(own) == 7 and sorted(bin_of(row["lines"], EDGES) for row in taken) == sorted(bin_of(row["lines"], EDGES) for row in own)        # THE SAME LINE COUNTS, bin by bin
    assert all(row["bin"] == bin_of(row["lines"], EDGES) and row["taken_by"] == THE_RULE and row["origin"] == ORDINARY for row in taken) and len({row["problem_id"] for row in taken}) == 7
    # WITHIN A BIN a seeded draw: the rows of the smallest content hash of the task's seed and the problem's id, under the draw's own label; another seed is another draw.
    by_hash = lambda prefix, count, seed=SEED: sorted((row["problem_id"] for row in offered if row["problem_id"].startswith(prefix)), key=lambda key: rank(seed, LONG_LABEL, key))[:count]      # noqa: E731
    assert {row["problem_id"] for row in taken} == {*by_hash("s", 1), *by_hash("t", 2), *by_hash("u", 1), *by_hash("v", 2), *by_hash("w", 1)}
    other = {row["problem_id"] for row in matched_in_length(own, offered, EDGES, 1)[0]}
    assert other == {*by_hash("s", 1, 1), *by_hash("t", 2, 1), *by_hash("u", 1, 1), *by_hash("v", 2, 1), *by_hash("w", 1, 1)} != {row["problem_id"] for row in taken}
    assert matched_in_length(own, list(reversed(offered)), EDGES, SEED)[1] == table and {row["problem_id"] for row in matched_in_length(own, list(reversed(offered)), EDGES, SEED)[0]} == {
        row["problem_id"] for row in taken}                                                                                           # the order the map offers them in changes nothing


def test_a_bin_the_map_cannot_fill_is_filled_from_the_next_shorter_one_and_the_shortfall_is_recorded():
    own = [_own(f"d{index}", lines) for index, lines in enumerate((2, 5, 9, 10, 11, 20, 30, 40))]                                    # by bin: 0, 1, 1, 3, 3
    offered = _offered({**{f"s{index}": 1 for index in range(2)}, **{f"t{index}": 3 for index in range(4)}, **{f"u{index}": 6 for index in range(3)}, "v0": 8, "v1": 15, "w0": 16})
    taken, table = matched_in_length(own, offered, EDGES, SEED)
    # 16+: three wanted, one offered: TWO short, asked of 8-15, which is asked for five and has two: THREE short, asked of 4-7, which is asked for four and has three: ONE short,
    # asked of 2-3, which gives its own one and that one. Nothing reaches the shortest bin.
    assert [(entry["lines"], entry["wanted"], entry["offered"], entry["asked_by_longer_bins"], entry["taken"], entry["shortfall"]) for entry in table] == [
        ("1", 0, 2, 0, 0, 0), ("2-3", 1, 4, 1, 2, 0), ("4-7", 1, 3, 3, 3, 1), ("8-15", 3, 2, 2, 2, 3), ("16+", 3, 1, 0, 1, 2)]
    assert len(taken) == 8 and sorted(row["bin"] for row in taken) == ["16+", "2-3", "2-3", "4-7", "4-7", "4-7", "8-15", "8-15"] and all(row["taken_by"] == THE_RULE for row in taken)
    assert sum(entry["taken"] for entry in table) == sum(entry["wanted"] for entry in table) == 8 and all(entry["taken"] <= entry["offered"] for entry in table)
    assert not [row for row in taken if row["bin"] == "1"]                                                                           # NEVER from a longer bin, and no further down than is needed
    # THE SHORTEST BIN CANNOT GIVE WHAT IS ASKED OF IT: refused; the section names no fill from a longer bin, though the map offers enough rows in all.
    short = [_own(f"d{index}", 1) for index in range(3)]
    with pytest.raises(ValueError, match="the map offers 12 ordinary solves, 2 of them in the shortest bin .`1`., and 1 of the 3 rows asked for are still missing there: a bin the map cannot fill "
                                         "is filled from the next SHORTER one, and no bin is shorter. Refused"):
        matched_in_length(short, offered, EDGES, SEED)
    with pytest.raises(ValueError, match="and 7 of the 8 rows asked for are still missing there"):
        matched_in_length(own, offered[-1:], EDGES, SEED)
    # A SMOKE RUN ALONE: the rows still missing are taken among what is left, in the same hash order, each marked as the minimum's; fewer rows than asked in all is refused all the same.
    loosely, table = matched_in_length(short, offered, EDGES, SEED, loosely=True)
    left = sorted((row["problem_id"] for row in offered if row["lines"] > 1), key=lambda key: rank(SEED, LONG_LABEL, key))
    assert [(row["problem_id"], row["taken_by"]) for row in loosely] == [*((row["problem_id"], THE_RULE) for row in loosely[:2]), (left[0], THE_MINIMUM)] and loosely[2]["bin"] == bin_of(loosely[2]["lines"], EDGES)
    assert {row["problem_id"] for row in loosely[:2]} == {"s0", "s1"} and table[0] == {"lines": "1", "wanted": 3, "offered": 2, "asked_by_longer_bins": 0, "taken": 2, "shortfall": 1}
    with pytest.raises(ValueError, match="the map offers 1 ordinary solves, 0 of them in the shortest bin .`1`., and 3 of the 3 rows asked for are still missing there"):
        matched_in_length(short, offered[-1:], EDGES, SEED, loosely=True)
    assert matched_in_length(own, offered, EDGES, SEED, loosely=True) == (taken, [entry for entry in matched_in_length(own, offered, EDGES, SEED)[1]])      # the rule gives enough: nothing is taken loosely


# ------------------------------------------------------------------------------------------------- the sets
ROUND_1_ROWS = [_own(f"b{index}", 2 + index) for index in range(6)]
ROUND_2_ROWS = [_own(f"b{index}", 9 + index, ROUND_2, part="a") for index in range(6, 10)]
ORDINARY_ROWS = [_own(f"b{index}", 3 + index % 7, ORDINARY, id=f"b{index}#map#statement#1") for index in range(10, 20)]
_placed = lambda row: (row["id"], row["problem_id"], row["row_of_the_file"])      # noqa: E731
_of_the_file = lambda rows: [row for row in rows if row["origin"] == "pretraining"]      # noqa: E731


def test_the_four_sets_hold_2n_rows_each_and_the_published_rows_are_the_first_of_the_one_order():
    sets = the_sets(ROUND_1_ROWS, ROUND_2_ROWS, ORDINARY_ROWS, PRETRAINING, SEED)
    assert list(sets) == list(MODELS) and [len(sets[model]) for model in MODELS] == [20, 20, 20, 20] and hold_the_sets(sets, PRETRAINING, SEED) == 10      # N = 6 + 4: no cap
    counts = {model: {origin: sum(row["origin"] == origin for row in sets[model]) for origin in (*OWN, "pretraining")} for model in MODELS}
    assert counts == {PUBLISHED2: {ROUND_1: 0, ROUND_2: 0, ORDINARY: 0, "pretraining": 20}, ONE_ROUND: {ROUND_1: 6, ROUND_2: 0, ORDINARY: 0, "pretraining": 14},
                      TWO_ROUNDS: {ROUND_1: 6, ROUND_2: 4, ORDINARY: 0, "pretraining": 10}, LONG_ORDINARY: {ROUND_1: 0, ROUND_2: 0, ORDINARY: 10, "pretraining": 10}}
    # ROUND 1's solves are in `one_round` AND in `two_rounds`, the same rows; round 2's in `two_rounds` alone; the ordinary solves in `long_ordinary` alone.
    ids = lambda model, origin: sorted(row["id"] for row in sets[model] if row["origin"] == origin)      # noqa: E731
    assert ids(ONE_ROUND, ROUND_1) == ids(TWO_ROUNDS, ROUND_1) == sorted(row["id"] for row in ROUND_1_ROWS) and ids(TWO_ROUNDS, ROUND_2) == sorted(row["id"] for row in ROUND_2_ROWS)
    assert ids(LONG_ORDINARY, ORDINARY) == sorted(row["id"] for row in ORDINARY_ROWS)
    # THE PUBLISHED ROWS: the first 2N, 2N - 6, N and N of the ONE hash order L4t's rehearsal draw uses, by id and place in the file: each smaller draw the first rows of each larger.
    order = rehearsal_rows(PRETRAINING, 40, SEED)
    assert [row["id"] for row in order] == [row["id"] for row in _draw_order(order)] and len(order) == 40
    for model, count in ((PUBLISHED2, 20), (ONE_ROUND, 14), (TWO_ROUNDS, 10), (LONG_ORDINARY, 10)):
        assert list(map(_placed, _draw_order(_of_the_file(sets[model])))) == list(map(_placed, order[:count]))
        assert [row["id"] for row in sets[model]] == [row["id"] for row in training_order(sets[model], SEED)]                        # each in the arm's content-hash order
        assert not any(field in row for row in _of_the_file(sets[model]) for field in TEXT_FIELDS)                                   # no published row with text
    assert list(map(_placed, _of_the_file(sets[TWO_ROUNDS]))) == list(map(_placed, _of_the_file(sets[LONG_ORDINARY])))               # THE SAME N, in the same order
    # By the task's seed: another seed is another draw of the published rows and another order.
    other = the_sets(ROUND_1_ROWS, ROUND_2_ROWS, ORDINARY_ROWS, PRETRAINING, 1)
    assert {row["id"] for row in _of_the_file(other[TWO_ROUNDS])} != {row["id"] for row in _of_the_file(sets[TWO_ROUNDS])} and [row["id"] for row in other[PUBLISHED2]] != [row["id"] for row in sets[PUBLISHED2]]
    # At least one solve of each round, and as many ordinary solves as both rounds have deep rows.
    for round_1, round_2, ordinary in (([], ROUND_2_ROWS, ORDINARY_ROWS[:4]), (ROUND_1_ROWS, [], ORDINARY_ROWS[:6]), (ROUND_1_ROWS, ROUND_2_ROWS, ORDINARY_ROWS[:9])):
        with pytest.raises(ValueError, match="at least one of each round, and as many ordinary solves as both rounds have deep rows. Refused"):
            the_sets(round_1, round_2, ordinary, PRETRAINING, SEED)
    with pytest.raises(ValueError, match="`pre` was trained on 40 rows and 42 are asked"):
        the_sets([*ROUND_1_ROWS, *(_own(f"c{index}", 3) for index in range(11))], ROUND_2_ROWS, [*ORDINARY_ROWS, *(_own(f"e{index}", 3, ORDINARY) for index in range(11))], PRETRAINING, SEED)


def test_the_four_sets_are_held_from_the_outside_and_each_departure_is_refused():
    sets = the_sets(ROUND_1_ROWS, ROUND_2_ROWS, ORDINARY_ROWS, PRETRAINING, SEED)
    order = rehearsal_rows(PRETRAINING, 40, SEED)
    in_order = lambda rows: training_order(rows, SEED)      # noqa: E731
    changed = lambda model, rows: {**sets, model: in_order(rows)}      # noqa: E731
    own = lambda model: [row for row in sets[model] if row["origin"] != "pretraining"]      # noqa: E731

    def refused(given, match, **more):
        with pytest.raises(ValueError, match=match):
            hold_the_sets(given, PRETRAINING, SEED, **more)

    sizes = "the four sets hold .rows of the model's own, published rows."
    refused(changed(PUBLISHED2, sets[PUBLISHED2][:-1]), sizes)                                                                       # a set that is not 2N rows
    refused(changed(ONE_ROUND, [*own(ONE_ROUND), *order[:13]]), sizes)
    refused(changed(ONE_ROUND, [*own(ONE_ROUND)[:-1], *order[:15]]), sizes)                                                          # `one_round` without one of round 1's solves
    refused(changed(LONG_ORDINARY, [*own(LONG_ORDINARY)[:-1], *order[:11]]), sizes)
    refused(changed(TWO_ROUNDS, [*ROUND_1_ROWS, *order[:10]]), sizes)                                                                # no solve of round 2: `two_rounds` would be `one_round`
    refused(changed(TWO_ROUNDS, [*ROUND_2_ROWS, *order[:10]]), sizes)
    # A WHOLE FAMILY WITHOUT A SOLVE OF ROUND 2, each set of the right size for N = 6: `two_rounds` would be `one_round`, and it is refused all the same.
    none = {PUBLISHED2: in_order(order[:12]), ONE_ROUND: in_order([*ROUND_1_ROWS, *order[:6]]), TWO_ROUNDS: in_order([*ROUND_1_ROWS, *order[:6]]), LONG_ORDINARY: in_order([*ORDINARY_ROWS[:6], *order[:6]])}
    refused(none, sizes)
    # A row of another origin than the set's own; an id twice; a problem in two rows of a set's own.
    refused(changed(ONE_ROUND, [*own(ONE_ROUND)[:-1], {**own(ONE_ROUND)[-1], "origin": ROUND_2}, *order[:14]]), "`one_round` holds 1 rows of another origin than its own")
    refused(changed(LONG_ORDINARY, [*own(LONG_ORDINARY)[:-1], {**own(LONG_ORDINARY)[-1], "origin": ROUND_1}, *order[:10]]), "`long_ordinary` holds 1 rows of another origin than its own")
    refused(changed(PUBLISHED2, [*order[:19], order[0]]), "`published2` holds 0 rows of another origin than its own .*, 1 ids twice")
    refused(changed(LONG_ORDINARY, [*own(LONG_ORDINARY)[:-1], {**own(LONG_ORDINARY)[-1], "problem_id": own(LONG_ORDINARY)[0]["problem_id"]}, *order[:10]]), "0 ids twice and 1 problems in two rows of its own")
    # A set that does not stand in the content-hash order.
    refused({**sets, TWO_ROUNDS: list(reversed(sets[TWO_ROUNDS]))}, "`two_rounds` does not stand in the content-hash order of this seed")
    # `one_round`'s round-1 rows are not `two_rounds`': another verified attempt of one problem.
    other = {**own(ONE_ROUND)[0], "id": "another#attempt"}
    refused(changed(ONE_ROUND, [other, *own(ONE_ROUND)[1:], *order[:14]]), "`one_round` does not hold the rows of round 1 that `two_rounds` holds: round 1's solves are in both, the same rows")
    # Published rows that are not the first of the order: by id, and by place in the file.
    refused(changed(TWO_ROUNDS, [*own(TWO_ROUNDS), *order[1:11]]), "the 10 published rows of `two_rounds` are not the first 10 of the hash order among the 40 rows")
    refused(changed(LONG_ORDINARY, [*own(LONG_ORDINARY), *order[:9], order[12]]), "the 10 published rows of `long_ordinary` are not the first 10 of the hash order")
    refused(changed(ONE_ROUND, [*own(ONE_ROUND), *order[:13], order[20]]), "the 14 published rows of `one_round` are not the first 14 of the hash order")
    refused(changed(PUBLISHED2, [*order[:19], order[25]]), "the 20 published rows of `published2` are not the first 20 of the hash order")
    refused(changed(PUBLISHED2, [*order[:19], {**order[19], "row_of_the_file": 99}]), "the 20 published rows of `published2` are not the first 20")
    # A problem both a deep solve and an ordinary one (a smoke run's looser draws alone may take one twice); an own row of a problem the pretraining file holds.
    both = [*own(LONG_ORDINARY)[:-1], {**own(LONG_ORDINARY)[-1], "problem_id": ROUND_2_ROWS[0]["problem_id"]}, *order[:10]]
    refused(changed(LONG_ORDINARY, both), "1 problems are both a deep solve of `two_rounds` and an ordinary solve of `long_ordinary`")
    assert hold_the_sets(changed(LONG_ORDINARY, both), PRETRAINING, SEED, loosely=True) == 10
    seen = [*own(LONG_ORDINARY)[:-1], {**own(LONG_ORDINARY)[-1], "problem_id": PRETRAINING[3]["problem_id"]}, *order[:10]]
    refused(changed(LONG_ORDINARY, seen), "0 problems are both .*, and 1 problems of the models' own rows are rows of the pretraining file", loosely=True)


def test_the_published_rows_are_held_to_the_one_order_were_the_draw_no_longer_nested(monkeypatch):
    sets = the_sets(ROUND_1_ROWS, ROUND_2_ROWS, ORDINARY_ROWS, PRETRAINING, SEED)
    from rlvr_lean.domain.ladder_round import l6
    # A draw whose smaller counts are NOT the first rows of its larger ones (the last rows, here): the sets built on it would not share their published rows, and are refused.
    draw = l6.rehearsal_rows
    monkeypatch.setattr(l6, "rehearsal_rows", lambda pretraining, count, seed: draw(pretraining, len(pretraining), seed)[-count:] if count < 20 else draw(pretraining, count, seed))
    with pytest.raises(ValueError, match="the 14 rows drawn for `one_round` are not the first 14 of the hash order `published2`'s 20 stand in. Refused"):
        hold_the_sets(sets, PRETRAINING, SEED)
    with pytest.raises(ValueError, match="are not the first .* of the hash order `published2`'s 20 stand in"):
        the_sets(ROUND_1_ROWS, ROUND_2_ROWS, ORDINARY_ROWS, PRETRAINING, SEED)


def test_two_rounds_and_long_ordinary_hold_their_published_rows_in_one_order_were_the_training_order_no_longer_held(monkeypatch):
    sets = the_sets(ROUND_1_ROWS, ROUND_2_ROWS, ORDINARY_ROWS, PRETRAINING, SEED)
    from rlvr_lean.domain.ladder_round import l6
    monkeypatch.setattr(l6, "training_order", lambda rows, seed: list(rows))    # were each set's own order not held: the two sets' published rows must still stand in ONE order
    published = _of_the_file(sets[LONG_ORDINARY])
    turned = {**sets, LONG_ORDINARY: [*(row for row in sets[LONG_ORDINARY] if row["origin"] != "pretraining"), *reversed(published)]}
    with pytest.raises(ValueError, match="`two_rounds` and `long_ordinary` do not hold the same published rows in the same order. Refused"):
        hold_the_sets(turned, PRETRAINING, SEED)


# --------------------------------------------------------------------------------------------- the branches
def _interval(mean, low, high):
    return {"problems": 600, "mean": mean, "low": low, "high": high}


ABOVE_ZERO, THROUGH, BELOW_ZERO = _interval(0.00734, 0.00205, 0.01358), _interval(0.00538, -0.001, 0.01272), _interval(-0.01014, -0.01903, -0.00117)
ENDS_AT_ZERO_BELOW, ENDS_AT_ZERO_ABOVE = _interval(0.004, 0.0, 0.008), _interval(-0.004, -0.008, 0.0)


@pytest.mark.parametrize("branch, names, pair", [(primary_branch, (CYCLE_ADDS, NOT_SEPARATED, IT_COSTS), (TWO_ROUNDS, PUBLISHED2)), (compounding, (SECOND_ROUND_ADDS, NOT_SHOWN, IT_COSTS), (TWO_ROUNDS, ONE_ROUND)),
                                                 (hard_or_long, (HARD_TO_FIND, NOT_FROM_LENGTH, LONG_DO_BETTER), (TWO_ROUNDS, LONG_ORDINARY))])
def test_each_read_has_three_names_and_an_interval_that_ends_at_zero_holds_zero(branch, names, pair):
    above, holds, below = names
    said = f"on the held-aside problems `{pair[0]}` minus `{pair[1]}`"
    assert branch([], ABOVE_ZERO)["name"] == above and branch([], THROUGH)["name"] == holds and branch([], BELOW_ZERO)["name"] == below
    assert f"{said} is above zero (+0.00734 [+0.00205, +0.01358])" in branch([], ABOVE_ZERO)["reason"] and f"{said} is below zero" in branch([], BELOW_ZERO)["reason"]
    # AN INTERVAL THAT ENDS AT ZERO HOLDS ZERO, at either end; and what such a difference resolves is half the width of its interval.
    assert branch([], ENDS_AT_ZERO_BELOW)["name"] == holds == branch([], ENDS_AT_ZERO_ABOVE)["name"] and branch([], _interval(0.0, 0.0, 0.0))["name"] == holds
    assert branch([], THROUGH)["resolves"] == 0.00686 and "This difference resolves 0.00686 per attempt" in branch([], THROUGH)["reason"] and f"{said} holds zero" in branch([], THROUGH)["reason"]
    assert "resolves" not in branch([], ABOVE_ZERO) and "resolves" not in branch([], BELOW_ZERO) and len({above, holds, below}) == 3
    # A failed check: INCONCLUSIVE, whatever was measured, and nothing else is said. Nothing measured on both sides: NOT READ.
    failed = branch(["`two_rounds`: no_barred_row"], ABOVE_ZERO)
    assert (failed["name"], failed["failed_checks"]) == (INCONCLUSIVE, ["`two_rounds`: no_barred_row"]) and "the run could not have been read (`two_rounds`: no_barred_row)" in failed["reason"]
    assert branch([], NOT_MEASURED)["name"] == NOT_READ and branch(["x"], NOT_MEASURED)["name"] == INCONCLUSIVE


def test_reach_is_shown_when_more_long_problems_are_gained_than_lost_and_the_sign_test_is_under_the_level():
    shown = reach([], _solved(14, 3, problems=74, before=25))
    assert shown["name"] == REACH_SHOWN and (shown["gained"], shown["lost"], shown["more_gained_than_lost"], shown["sign_test_under_the_level"]) == (14, 3, True, True) and shown["sign_test_p"] < 0.05
    assert "of the 74 long goal problems `two_rounds` solves 36 and `published2` 25: gained 14, lost 3" in shown["reason"] and SIGN_TEST_LEVEL == 0.05
    # MORE GAINED, the sign test NOT under the level: not shown. UNDER the level, more LOST: not shown. As many gained as lost: not shown.
    assert sign_test(9, 3) > 0.05 > sign_test(10, 2) and sign_test(3, 14) < 0.05
    assert reach([], _solved(9, 3, problems=74))["name"] == NOT_SHOWN and (reach([], _solved(9, 3, problems=74))["more_gained_than_lost"], reach([], _solved(9, 3, problems=74))["sign_test_under_the_level"]) == (True, False)
    assert reach([], _solved(3, 14, problems=74))["name"] == NOT_SHOWN and (reach([], _solved(3, 14, problems=74))["more_gained_than_lost"], reach([], _solved(3, 14, problems=74))["sign_test_under_the_level"]) == (False, True)
    assert reach([], _solved(10, 2, problems=74))["name"] == REACH_SHOWN and reach([], _solved(5, 5, problems=74))["name"] == NOT_SHOWN and reach([], _solved(0, 0, problems=74))["name"] == NOT_SHOWN
    assert reach([], _solved(10, 2, problems=74), level=0.01)["name"] == NOT_SHOWN and "(no problem changed)" in reach([], _solved(0, 0, problems=74))["reason"]
    assert reach([], _solved(5, 5, problems=74))["more_gained_than_lost"] is False and reach([], _solved(0, 0, problems=74))["more_gained_than_lost"] is False      # as many is not more
    assert reach(["x"], _solved(14, 3, problems=74))["name"] == INCONCLUSIVE and reach([], {})["name"] == NOT_READ and reach([], _solved(0, 0, problems=0))["name"] == NOT_READ


def _spread(published=1.2, shallow=2.5, deep=1.9, seeds=(0, 1, 2), **more):
    of = lambda value: {"held_aside_per_1000": {"by_seed": {}, "mean": 50.0, "standard_deviation": value}, "all_of_g_per_1000": {"standard_deviation": 9.9}}      # noqa: E731
    return {"seeds_read": list(seeds), "pooled": {"the_spread": {"published": of(published), "shallow": of(shallow), "deep": of(deep)}}, **more}


def test_the_spread_is_the_largest_of_the_kinds_standard_deviations_on_the_held_aside_set_over_three_seeds_of_l5():
    given = spread_of_l5(_spread())
    # Per 1,000 attempts in the read over L5's seeds; per attempt here. THE LARGEST of the three kinds': a kill then asks the most. The held-aside set's figure, and no other's.
    assert (given["given"], given["of"], given["per_attempt"], given["per_1000_by_kind"], given["seeds_read"]) == (True, "shallow", 0.0025, {"published": 1.2, "shallow": 2.5, "deep": 1.9}, [0, 1, 2])
    assert spread_of_l5(_spread(published=3.1))["of"] == "published" and spread_of_l5(_spread(published=3.1))["per_attempt"] == 0.0031
    # NOT GIVEN: no file; a read of fewer seeds than three, or of more; one that pooled nothing; a kind without a standard deviation.
    assert spread_of_l5(None) == {"given": False, "why": "no file of the read over L5's seeds is in the run directory"}
    for read in (_spread(seeds=(0, 1)), _spread(seeds=(0, 1, 2, 3)), {"seeds_read": [0, 1, 2], "pooled": None}, _spread(deep=None), {"seeds_read": [0, 1, 2]}):
        found = spread_of_l5(read)
        assert found["given"] is False and "the spread is the one 3 seeds measured" in found["why"] and "per_attempt" not in found


def test_what_follows_is_the_scouts_table_in_its_order():
    spread, none = spread_of_l5(_spread()), spread_of_l5(None)                  # twice the spread: 0.005 per attempt
    primary = lambda difference: primary_branch([], difference)      # noqa: E731
    shown, not_shown = reach([], _solved(14, 3, problems=74)), reach([], _solved(3, 3, problems=74))
    adds, no = compounding([], ABOVE_ZERO), compounding([], THROUGH)
    of = lambda difference, of_compounding=no, of_reach=not_shown, given=spread: what_follows(primary(difference), of_compounding, of_reach, difference, given)      # noqa: E731
    # THE PRIMARY'S POINT AT OR ABOVE ZERO (whatever its interval), or REACH SHOWN, or THE SECOND ROUND ADDS: two further seeds. With or without the spread.
    for found in (of(ABOVE_ZERO), of(THROUGH), of(_interval(0.0, -0.01, 0.01)), of(BELOW_ZERO, of_reach=shown), of(BELOW_ZERO, of_compounding=adds), of(THROUGH, given=none),
                  of(BELOW_ZERO, of_reach=shown, given=none)):
        assert found["name"] == TWO_FURTHER_SEEDS and "two further seeds of the four models on the same two searches" in found["reason"]
    # THE POINT BELOW ZERO BY MORE THAN TWICE THE SPREAD, reach not shown and no second round's gain: the cycle at this size does not clear the bar.
    kill = of(BELOW_ZERO)
    assert kill["name"] == DOES_NOT_CLEAR and (kill["point"], kill["twice_the_spread"]) == (-0.01014, 0.005) and "the question of a larger model needs a decision" in kill["reason"]
    # IN BETWEEN: below zero by no more than twice the spread (exactly twice is no more), whatever its interval says.
    between = of(_interval(-0.003, -0.012, 0.006))
    assert between["name"] == IN_BETWEEN and "stated as it is, for a decision" in between["reason"] and of(_interval(-0.005, -0.006, -0.004))["name"] == IN_BETWEEN
    assert of(_interval(-0.00501, -0.006, -0.004))["name"] == DOES_NOT_CLEAR and of(_interval(-0.00001, -0.02, 0.02))["name"] == IN_BETWEEN
    # THE SPREAD IS NOT GIVEN and the point is below zero: the point is given, and NO KILL IS NAMED.
    unknown = of(BELOW_ZERO, given=none)
    assert unknown["name"] == SPREAD_NOT_GIVEN and unknown["point"] == -0.01014 and "the primary's point is -0.01014" in unknown["reason"] and "is NOT GIVEN (no file of the read over L5's seeds" in unknown["reason"]
    assert "No kill is named" in unknown["reason"] and "twice_the_spread" not in unknown
    # Nothing is named when the primary is INCONCLUSIVE or was not read.
    assert what_follows(primary_branch(["x"], ABOVE_ZERO), adds, shown, ABOVE_ZERO, spread)["name"] == NOT_READ and what_follows(primary_branch([], NOT_MEASURED), adds, shown, NOT_MEASURED, spread)["name"] == NOT_READ


# ------------------------------------------------------------------------------- the report, on hand-made rows
_published = lambda count: [{"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:count])]      # noqa: E731
_of_a_search = lambda names, origin: [{"id": f"{name}#search", "problem_id": name, "origin": origin, "k": 1 + index % 5, "lines": 5 + index % 9} for index, name in enumerate(names)]      # noqa: E731
TRAINED = {PUBLISHED2: _published(40), ONE_ROUND: [*_of_a_search(BASE[:12], ROUND_1), *_published(28)], TWO_ROUNDS: [*_of_a_search(BASE[:12], ROUND_1), *_of_a_search(BASE[12:20], ROUND_2), *_published(20)],
           LONG_ORDINARY: [*({"id": f"{name}#map", "problem_id": name, "origin": ORDINARY, "k": 1 + index % 8, "lines": 5 + index % 9} for index, name in enumerate(BASE[20:40])), *_published(20)]}
TRAINS = {name: {"model": name, "rows": len(own), "steps": -(-len(own) // 8), "stand_in_engine": False, "trained_from": "the stored adapter pre_r64",
                 "against_the_start_adapter": {name: CHANGED}} for name, own in TRAINED.items()}
LOSSES = {name: FALLING(len(own)) for name, own in TRAINED.items()}
RISING = list(reversed(FALLING(40)))
STATS = {"proofs": 20, "median": 5.0, "mean": 5.5, "share_with_8_lines_or_more": 0.25, "longest": 13, "share_with_4_lines_or_more": 0.75}
_part = lambda problems, solved_: {"problems": problems, "solved": solved_, "share_solved": round(solved_ / problems, 5), "solved_by_first_success": {"1_to_8": solved_ - 1, "9_to_16": 1, "17_to_32": 0, "33_to_64": 0}}      # noqa: E731
FOUND = {"problems": 60, "taken_by_the_rule": 60, "taken_by_the_runs_minimum": 0, "solved": 8, "share_solved": 0.13333, "a": _part(40, 5), "b": _part(20, 3), "episodes_each": 64, "episodes": 3840,
         "attempts": 3904, "successes": 21, "per_1000": 5.47, "without_an_answer": 2, "lines_of_the_deep_rows": {"round_1": STATS, "round_2": {**STATS, "proofs": 8}}}
BINS = [{"lines": name, "wanted": wanted, "offered": 30, "asked_by_longer_bins": 0, "taken": wanted, "shortfall": 0} for name, wanted in (("1", 0), ("2-3", 0), ("4-7", 9), ("8-15", 11), ("16+", 0))]
PREPARE = {"stage": "l4", "label": LABEL, "check": "L6", "seed": 0, "start": "pre_r64", "start_adapter": "runs/ladder_l4_pretrain_r64_seed0/adapters/pre_r64", "stand_in_engine": False,
           "l4t2_run": "runs/ladder_l4_rows2_pre_r64_seed0", "map_run": "runs/ladder_l4_map_pre_r64_seed0", "models": list(MODELS), "start_recipe": {"rank": 64, "alpha": 128},
           "enough_round_two_solves": enough_round_two_solves(8, 5), "round_one_solves": 12, "round_two_solves": 8, "deep_solves": 20, "rows_a_set": 40, "the_search_found": FOUND,
           "trainings": {"run": True},
           "round_one": {"run": "runs/ladder_l5_deep_pre_r64_seed0", "seed": 0, "model": "deep", "adapter": "runs/ladder_l5_deep_pre_r64_seed0/adapters/deep", "attempted": 100, "solved": 12,
                         "inconclusive": False, "failed_checks": []},
           "the_search": {"problems": 60, "by_part": {"a": 40, "b": 20}, "partition": {"candidates": 172}, "episodes_each": 64, "sampling_seed": 1052, "taken_by_the_runs_minimum": 0},
           "the_held_aside": {"problems": 12, "candidates_not_searched": 72, "episodes_each": 16, "sampling_seed": 1051, "taken_by_the_rule": 12, "taken_by_the_runs_minimum": 0},
           "the_stored_model": {"model": "second_full", "there": True, "run": "runs/ladder_l4_dose_pre_r64_seed0", "adapter": "runs/ladder_l4_dose_pre_r64_seed0/adapters/second_full"},
           "matched_in_length": {"what": "bin by bin", "line_bins": EDGES, "ordinary_solves_the_map_offers": 150, "bins": BINS, "bins_short": [], "rows_short": 0, "rows_of_another_bin_than_wanted": 0},
           "lines_of_the_rows": {"what": "the lines", "round_1_rows": STATS, "round_2_rows": {**STATS, "proofs": 8}, "ordinary_rows": STATS, "the_published_rows_beside_them": STATS,
                                 "published2_rows": {**STATS, "proofs": 40}},
           "rows_not_taken_by_the_rule": {"what": "a smoke run alone", TWO_ROUNDS: 0, LONG_ORDINARY: 0}, "smoke": None, "training_sets": {name: {"rows": 40} for name in MODELS},
           "goal_samplings": SAMPLINGS, "attempts_a_goal_problem": 93, "rung_episodes": 8, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020}, "contradicted_side_setting": "audit",
           "stored_models": {"pre_r64": {"run": "runs/ladder_l4_pretrain_r64_seed0"}, "deep": {"run": "runs/ladder_l5_deep_pre_r64_seed0"}, "second_full": {"run": "runs/ladder_l4_dose_pre_r64_seed0"}}}
TOO_FEW = {**PREPARE, "enough_round_two_solves": enough_round_two_solves(4, 5), "round_two_solves": 4, "deep_solves": 0, "rows_a_set": 0, "training_sets": {}, "lines_of_the_rows": None, "matched_in_length": None,
           "rows_not_taken_by_the_rule": None, "the_search_found": {**FOUND, "solved": 4},
           "trainings": {"run": False, "why": "search 2 found 4 round-2 solves and 5 are asked: nothing is trained. No set is built and no model is trained or measured: the search is the finding"}}
HA_MOST = [4] * 12                                                              # one more success than `HA_MORE` on every held-aside problem
NO_LONG = ([6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0], [9, 9, 9, 6, 6, 6, 5, 4, 3, 0, 0, 0])               # the start model without a success on the three long goal problems
NOTHING = ([0] * 12, [0] * 12)
ALL_LONG = {key: {"length_group": "8+"} for key in GOAL_IDS}                    # another world's lengths: every goal problem is a long one


def _report(models=None, aside=None, prepare=PREPARE, trains=TRAINS, losses=LOSSES, trained=TRAINED, never=None, again=AGAIN, measured=MODELS, stored=True, beside=None, spread=None,
            lengths=LENGTHS):
    """The report with the start model's own rows standing in for every model that is not given, on G and on the
    held-aside set (`aside`: by model, its successes on each held-aside problem, or its rows). `stored`: whether
    `second_full` was read and measured on the held-aside set. `beside`: by stored model, its rows on G."""
    models = {name: (models or {}).get(name) or _model() for name in measured}
    given = lambda name: (aside or {}).get(name, HA_BASE)      # noqa: E731
    names = ("pre_r64", *measured, *(("second_full",) if stored else ())) if measured else ()
    on_the_set = {name: given(name) if given(name) and isinstance(given(name)[0], dict) else _aside(given(name)) for name in names}
    on_g = {"deep": _model()["goal"], **({"second_full": _model()["goal"]} if stored else {})} if beside is None else beside
    return build_l6_report(prepare, {name: trains[name] for name in measured}, {name: losses[name] for name in measured}, {name: trained[name] for name in measured}, GROUPS, lengths,
                           {"pre_r64": _model()}, models, again, SETTINGS, EVALUATION, BASE, on_the_set, on_g, never, spread)


def _on_ha(after, before):
    return per_attempt(_aside(after), _aside(before), ASIDE, EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"])


def _on_g(entry):
    return attempts_on_g(entry["goal"])


def test_with_the_start_models_rows_standing_in_every_difference_is_zero_and_the_report_says_so_in_the_specs_order():
    report = _report()
    assert report["branches"] == {"the_primary": NOT_SEPARATED, "compounding": NOT_SHOWN, "hard_to_find_or_long": NOT_FROM_LENGTH, "reach": NOT_SHOWN, "what_follows": TWO_FURTHER_SEEDS}
    assert (report["ok"], report["inconclusive"], report["failed_checks"], report["measured"], report["rank"], report["deep_solves"], report["rows_a_set"]) == (True, False, [], list(MODELS), 64, 20, 40)
    assert list(report["can_this_run_be_read"]) == ["what", "search_2", *MODELS, "the_held_aside_sets"] and report["can_this_run_be_read"]["search_2"]["passes"] is True
    for model in MODELS:        # FOUR checks a model, and the LOSS is none of them
        assert list(report["can_this_run_be_read"][model]) == [THE_ADAPTER, "it_still_writes_proofs", "lean_answered", "no_barred_row"] and all(entry["passes"] for entry in report["can_this_run_be_read"][model].values())
    answered = report["can_this_run_be_read"]["the_held_aside_sets"]
    assert answered["passes"] is True and answered["share_without_an_answer"] == {held_aside_key(name): 0.0 for name in ("pre_r64", *MODELS, "second_full")}
    for key, _, _ in READS:
        assert report[key]["difference"]["mean"] == 0.0 and report[key]["difference"]["attempts_each"] == 16 * 12 and report[key]["solved"]["problems"] == 12 and report[key]["resolves"] == 0.0
    assert (report["reach"]["long_goal_problems"], report["reach"]["solved"]["problems"], report["reach"]["gained"], report["reach"]["lost"]) == (3, 3, 0, 0)
    assert report["what_follows"]["point"] == 0.0 and report["the_spread_of_l5"] == {"given": False, "why": "no file of the read over L5's seeds is in the run directory"}
    beside = report["beside"]
    assert list(beside) == ["what", "each_model", "pairs", "two_rounds_against_the_stored_models", "overfitting", "timed_out_in_lean"] and list(beside["each_model"]) == ["pre_r64", *MODELS]
    assert list(beside["pairs"]) == [pair_name(*pair) for pair in PAIRS] and all(list(pair) == ["on_all_of_g", "on_the_goal_problems_not_in_the_goal_set_again", "on_the_goal_set_again", "goal_problems_solved"]
                                                                                 for pair in beside["pairs"].values())
    assert list(beside["overfitting"]) == list(MODELS) and all(own["narrower"] is False and own["against"] == "pre_r64" for own in beside["overfitting"].values())
    assert report["what_it_cannot_say"] == CANNOT_SAY and report["label"] == LABEL and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    # THE LINES, IN THE SPEC'S ORDER: the checks (the loss printed after them, deciding nothing); the primary; compounding; hard to find, or long; reach; what follows; beside; the search.
    marks = ("CHECK, search 2 found at least 5 round-2 solves: 8: PASS", "`published2` (", "`one_round` (", "`two_rounds` (", "`long_ordinary` (", "CHECK, Lean answered on each held-aside set",
             "PRINTED, DECIDING NOTHING: each training's loss", "PRIMARY, the bet at equal steps. THE HELD-ASIDE SET (12 problems", "PRIMARY: NOT SEPARATED AT THIS SIZE. ",
             "COMPOUNDING (round 2's solves against the same number of published rows", "COMPOUNDING: NOT SHOWN. ", "HARD TO FIND, OR LONG (ordinary solves matched in length",
             "HARD TO FIND, OR LONG: NOT SEPARATED FROM LENGTH. ", "REACH, on the goal set (3 goal problems", "WHAT FOLLOWS: TWO FURTHER SEEDS", "BESIDE, naming nothing: on the held-aside set (12 problems)",
             "BESIDE, naming nothing: on all of G (12 problems, 93 attempts each)", "BESIDE, naming nothing: `two_rounds` minus `published2`", "BESIDE, naming nothing: `two_rounds` minus `one_round`",
             "BESIDE, naming nothing: `two_rounds` minus `long_ordinary`", "BESIDE, naming nothing: `one_round` minus `published2`", "BESIDE, naming nothing: `long_ordinary` minus `published2`",
             "BESIDE, naming nothing: `long_ordinary` minus `one_round`", "BESIDE, naming nothing: `two_rounds` against the stored `deep` (M(1)", "BESIDE, naming nothing: `two_rounds` against the stored `second_full`",
             "BESIDE, naming nothing: `two_rounds` against `second_full` ON THE HELD-ASIDE SET", "BESIDE, naming nothing: OVERFITTING, `published2` against `pre_r64`",
             "BESIDE, naming nothing: OVERFITTING, `long_ordinary` against `pre_r64`", "BESIDE, naming nothing: TIMEOUTS", "SEARCH 2: 60 problems attempted (64 attempts each), 8 solved",
             "THE LINES OF THE ROWS (the proofs trained on): round 1 rows: 20 proofs", "THE ORDINARY SOLVES MATCHED IN LENGTH")
    places = [next(index for index, line in enumerate(report["lines"]) if f"{SAY}: {mark}" in line) for mark in marks]
    assert places == sorted(places) and len(set(places)) == len(marks)
    first = report["lines"][0]
    assert "ROUND 1 IS L5's, only read (runs/ladder_l5_deep_pre_r64_seed0): its deep search solved 12 of 100 problems, and M(1) is its model `deep`" in first
    assert "SEARCH 2: M(1), served at rank 64, made 64 attempts at 60 problems, sampling seed 1052: (a) the 40 search 1 left unsolved, (b) the 20 of the 172 candidates neither draw took" in first
    assert "each trained FROM `pre_r64` AT RANK 64, one pass, the arm's recipe, on 40 rows (N = 20: every deep solve of both rounds" in first and "ONE sampling seed, 1051" in first
    assert "(a) the problems search 1 left unsolved: 5 of 40 solved (a share of 0.125); by the attempt of the first success: 1 to 8: 4; 9 to 16: 1; 17 to 32: 0; 33 to 64: 0" in report["lines"][places[-3]]
    assert "(b) the candidates neither draw took: 3 of 20 solved" in report["lines"][places[-3]] and "3,904 attempts, 21 of 3,840 episodes resolved their problem (5.47 per 1,000)" in report["lines"][places[-3]]
    assert report["lines"][places[-1]].endswith("4-7 lines: 9 wanted, the map offers 30, 9 taken; 8-15 lines: 11 wanted, the map offers 30, 11 taken; 16+ lines: 0 wanted, the map offers 30, 0 taken. Every bin was filled")
    assert report["headline"].startswith(f"L6 (a second round: does the cycle compound; {LABEL}) seed 0, from `pre_r64`: NOT SEPARATED AT THIS SIZE. Search 2 solved 8 of 60 problems; N = 20. "
                                         "Primary (the held-aside set, 12 problems, `two_rounds` minus `published2`): +0.00000")
    assert not any("A SMOKE RUN" in line or "ONE sampling of G" in line or "NOT TO BE READ" in line or "NOT THERE" in line or "IT ROSE BY MORE" in line for line in report["lines"])
    # The fourth check is L5's own (the deep and ordinary rows are base-map problems, and are not held to a half) and, beside it, NO ROW OF A HELD-ASIDE PROBLEM.
    barred = report["can_this_run_be_read"][TWO_ROUNDS]["no_barred_row"]
    assert (barred["rows_of_the_models_own"], barred["published_rows_of_the_pretrain_half"], barred["own_rows_not_of_the_base_map"], barred["held_aside_problems"]) == (20, 20, 0, 0)
    assert any("`two_rounds`, CHECK 4, no barred row (no held-out problem; no held-aside problem; its published rows of the `pretrain` half; the rows of the model's own are base-map problems): "
               "0 held-out problems, 0 held-aside problems, 0 of its 20 published rows not of the `pretrain` half, 0 of its 20 own rows not base-map problems: PASS" in line for line in report["lines"])
    assert any("`one_round` (" in line and "40 rows, 5 optimizer steps, from `pre_r64`), CHECK 1, the adapter it saved is not the start adapter's: 1,000 of the 1,000 numbers of its adapter are not the "
               "start adapter's: PASS" in line for line in report["lines"])


def test_each_read_is_two_rounds_minus_its_own_other_side_on_the_held_aside_set_and_the_goal_set_stands_beside_and_names_nothing():
    # ON THE HELD-ASIDE SET every model has successes of its own: `two_rounds` 3 a problem, `published2` 2, `one_round` 1, `long_ordinary` 4. ON G IT IS THE OTHER WAY ROUND for the
    # primary's pair: `two_rounds` is below `published2` there. The held-aside set decides, and G is said beside.
    aside = {TWO_ROUNDS: HA_MORE, PUBLISHED2: HA_BASE, ONE_ROUND: HA_LESS, LONG_ORDINARY: HA_MOST}
    two_rounds, published2 = _model(on_g=LESS), _model(on_g=MORE)
    report = _report({TWO_ROUNDS: two_rounds, PUBLISHED2: published2}, aside)
    assert report["primary"]["difference"] == _on_ha(HA_MORE, HA_BASE) and report["compounding"]["difference"] == _on_ha(HA_MORE, HA_LESS) and report["hard_to_find_or_long"]["difference"] == _on_ha(HA_MORE, HA_MOST)
    assert report["primary"]["difference"]["low"] > 0 and report["compounding"]["difference"]["low"] > report["primary"]["difference"]["high"] and report["hard_to_find_or_long"]["difference"]["high"] < 0
    assert report["branches"] == {"the_primary": CYCLE_ADDS, "compounding": SECOND_ROUND_ADDS, "hard_to_find_or_long": LONG_DO_BETTER, "reach": NOT_SHOWN, "what_follows": TWO_FURTHER_SEEDS}
    for key, before in (("primary", HA_BASE), ("compounding", HA_LESS), ("hard_to_find_or_long", HA_MOST)):
        assert report[key]["solved"] == solved(_aside(HA_MORE), _aside(before), ASIDE) and report[key]["difference"]["attempts_each"] == 16 * 12
    pair = report["beside"]["pairs"]["two_rounds_minus_published2"]
    resamples, seed = EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"]
    assert pair["on_all_of_g"] == per_attempt(_on_g(two_rounds), _on_g(published2), GOAL_IDS, resamples, seed) and pair["on_all_of_g"]["high"] < 0                  # BELOW zero on G: no branch moves
    # THE SPLIT of G: the goal problems NOT in G' (the start model solves them in its first sampling) and G', each over ALL the attempts.
    not_again = [key for key in GOAL_IDS if key not in AGAIN]
    assert pair["on_the_goal_problems_not_in_the_goal_set_again"] == per_attempt(_on_g(two_rounds), _on_g(published2), not_again, resamples, seed) and len(not_again) == 8 and len(AGAIN) == 4
    assert pair["on_the_goal_set_again"] == per_attempt(_on_g(two_rounds), _on_g(published2), AGAIN, resamples, seed) and pair["on_the_goal_set_again"]["attempts_each"] == 93 * 4
    assert pair["goal_problems_solved"] == solved(_on_g(two_rounds), _on_g(published2), GOAL_IDS) and "naming nothing" in report["beside"]["what"]
    other = report["beside"]["pairs"]["one_round_minus_published2"]
    assert other["on_all_of_g"] == per_attempt(_on_g(_model()), _on_g(published2), GOAL_IDS, resamples, seed) and other["on_all_of_g"]["mean"] < 0 < report["beside"]["pairs"]["long_ordinary_minus_one_round"]["on_all_of_g"]["mean"] + 1
    each = report["beside"]["each_model"]
    assert {name: own["held_aside"]["successes"] for name, own in each.items()} == {"pre_r64": 24, PUBLISHED2: 24, ONE_ROUND: 12, TWO_ROUNDS: 36, LONG_ORDINARY: 48}
    assert each[TWO_ROUNDS]["all_of_g"]["successes"] == sum(row["resolved"] for row in _on_g(two_rounds)) and each[PUBLISHED2]["all_of_g"]["solved_at_least_once"] == 12
    text = "\n".join(report["lines"])
    assert f"{SAY}: PRIMARY: {CYCLE_ADDS}. on the held-aside problems `two_rounds` minus `published2` is above zero" in text
    assert f"{SAY}: COMPOUNDING: {SECOND_ROUND_ADDS}. on the held-aside problems `two_rounds` minus `one_round` is above zero" in text
    assert f"{SAY}: HARD TO FIND, OR LONG: {LONG_DO_BETTER}. on the held-aside problems `two_rounds` minus `long_ordinary` is below zero" in text
    assert "`two_rounds` minus `published2`, paired by problem, 95% bootstrap over problems: +0.06250" in text and "`two_rounds` minus `long_ordinary`, paired by problem, 95% bootstrap over problems: -0.06250" in text
    # EACH READ BY ITSELF: far above every other model ON G and level with it on the held-aside set names nothing there.
    on_g_alone = _report({TWO_ROUNDS: _model(on_g=MORE)})
    assert on_g_alone["branches"] == {"the_primary": NOT_SEPARATED, "compounding": NOT_SHOWN, "hard_to_find_or_long": NOT_FROM_LENGTH, "reach": NOT_SHOWN, "what_follows": TWO_FURTHER_SEEDS}
    assert on_g_alone["beside"]["pairs"]["two_rounds_minus_published2"]["on_all_of_g"]["low"] > 0
    # ... and each read moves with ITS OWN other side alone.
    assert _report(aside={PUBLISHED2: HA_LESS})["branches"] == {"the_primary": CYCLE_ADDS, "compounding": NOT_SHOWN, "hard_to_find_or_long": NOT_FROM_LENGTH, "reach": NOT_SHOWN, "what_follows": TWO_FURTHER_SEEDS}
    assert _report(aside={ONE_ROUND: HA_MORE})["branches"]["compounding"] == IT_COSTS and _report(aside={ONE_ROUND: HA_MORE})["branches"]["the_primary"] == NOT_SEPARATED
    assert _report(aside={LONG_ORDINARY: HA_LESS})["branches"]["hard_to_find_or_long"] == HARD_TO_FIND and _report(aside={LONG_ORDINARY: HA_LESS})["branches"]["compounding"] == NOT_SHOWN
    assert _report(aside={TWO_ROUNDS: HA_LESS})["branches"]["the_primary"] == IT_COSTS


def test_reach_is_read_on_the_long_goal_problems_two_rounds_against_published2_and_feeds_what_follows():
    # The three long goal problems: `two_rounds` solves them, `published2` none: gained 3, lost 0; three problems cannot be under 0.05: NOT SHOWN.
    report = _report({PUBLISHED2: _model(on_g=NO_LONG)})
    long_ids = GOAL_IDS[-3:]
    assert report["reach"]["solved"] == solved(_on_g(_model()), _on_g(_model(on_g=NO_LONG)), long_ids) and (report["reach"]["gained"], report["reach"]["lost"], report["reach"]["name"]) == (3, 0, NOT_SHOWN)
    assert [LENGTHS[key]["length_group"] for key in long_ids] == ["8+"] * 3 and report["beside"]["each_model"][PUBLISHED2]["long_goal_problems"]["solved_at_least_once"] == 0
    # THE OTHER SIDE is `published2`, not `one_round` or the start model: with `one_round` unable on them and `published2` able, nothing is gained.
    assert _report({ONE_ROUND: _model(on_g=NO_LONG)})["reach"]["gained"] == 0 and _report({TWO_ROUNDS: _model(on_g=NO_LONG)})["reach"]["lost"] == 3
    # IN A WORLD WHERE EVERY GOAL PROBLEM IS LONG: `two_rounds` solves twelve that `published2` does not: REACH SHOWN, and it is on ALL the attempts (not the first sampling alone).
    shown = _report({PUBLISHED2: _model(on_g=NOTHING)}, lengths=ALL_LONG)
    assert (shown["reach"]["name"], shown["reach"]["gained"], shown["reach"]["lost"], shown["reach"]["long_goal_problems"]) == (REACH_SHOWN, 12, 0, 12)
    late = _report({PUBLISHED2: _model(on_g=NOTHING), TWO_ROUNDS: _model(on_g=([0] * 12, [1] * 12))}, lengths=ALL_LONG)
    assert (late["reach"]["name"], late["reach"]["gained"]) == (REACH_SHOWN, 12)
    # REACH SHOWN sends the run to two further seeds though the primary's point is below zero and far beyond twice the spread.
    below = {"aside": {TWO_ROUNDS: HA_LESS}, "spread": _spread()}
    assert _report(**below)["branches"]["what_follows"] == DOES_NOT_CLEAR
    assert _report({PUBLISHED2: _model(on_g=NOTHING)}, lengths=ALL_LONG, **below)["branches"] == {"the_primary": IT_COSTS, "compounding": IT_COSTS, "hard_to_find_or_long": LONG_DO_BETTER, "reach": REACH_SHOWN,
                                                                                               "what_follows": TWO_FURTHER_SEEDS}
    # ... and so does THE SECOND ROUND ADDS (`two_rounds` above `one_round` there, below `published2`).
    assert _report(aside={TWO_ROUNDS: HA_LESS, ONE_ROUND: [0] * 12}, spread=_spread())["branches"] == {"the_primary": IT_COSTS, "compounding": SECOND_ROUND_ADDS, "hard_to_find_or_long": LONG_DO_BETTER,
                                                                                                       "reach": NOT_SHOWN, "what_follows": TWO_FURTHER_SEEDS}


def test_what_follows_reads_the_spread_of_l5s_three_seeds_when_the_read_over_them_is_given():
    # `two_rounds` one success below `published2` on every held-aside problem: the point is -0.0625 per attempt.
    cost = {"aside": {TWO_ROUNDS: HA_LESS}}
    unknown = _report(**cost)
    assert unknown["branches"]["what_follows"] == SPREAD_NOT_GIVEN and unknown["the_spread_of_l5"]["given"] is False and unknown["what_follows"]["point"] == -0.0625
    assert any(line.startswith(f"{SAY}: WHAT FOLLOWS: {SPREAD_NOT_GIVEN}. the primary's point is -0.06250 (IT COSTS)") and "No kill is named" in line for line in unknown["lines"])
    kill = _report(spread=_spread(), **cost)                                    # the largest of the kinds' spreads: 2.5 per 1,000; twice it, 0.005 per attempt
    assert kill["branches"]["what_follows"] == DOES_NOT_CLEAR and kill["the_spread_of_l5"]["per_attempt"] == 0.0025 and kill["what_follows"]["twice_the_spread"] == 0.005
    between = _report(spread=_spread(shallow=40.0), **cost)                     # a spread of 0.04 per attempt: twice it is 0.08, and the point is within
    assert between["branches"]["what_follows"] == IN_BETWEEN and between["the_spread_of_l5"]["of"] == "shallow" and between["what_follows"]["twice_the_spread"] == 0.08
    assert _report(spread=_spread(seeds=(0, 1)), **cost)["branches"]["what_follows"] == SPREAD_NOT_GIVEN
    assert _report(spread=_spread())["branches"]["what_follows"] == TWO_FURTHER_SEEDS      # the point at zero: two further seeds, whatever the spread
    # THE POINT IS THE PRIMARY'S: `two_rounds` below `published2` and LEVEL with `one_round` (compounding's own point is zero, and it is not shown) is a kill, not further seeds.
    level = _report(aside={TWO_ROUNDS: HA_LESS, ONE_ROUND: HA_LESS}, spread=_spread())
    assert level["compounding"]["difference"]["mean"] == 0.0 and level["branches"]["compounding"] == NOT_SHOWN and level["branches"]["what_follows"] == DOES_NOT_CLEAR
    assert level["what_follows"]["point"] == level["primary"]["difference"]["mean"] == -0.0625


def test_a_failed_check_makes_the_run_inconclusive_and_a_rise_of_the_loss_is_printed_and_decides_nothing():
    passing = _report(aside={TWO_ROUNDS: HA_MORE})
    assert passing["branches"]["the_primary"] == CYCLE_ADDS
    # A RISE OF A TRAINING'S LOSS, its adapter not the start adapter's: PRINTED on a line of its own, NAMED, and the run is read all the same.
    rose = _report(aside={TWO_ROUNDS: HA_MORE}, losses={**LOSSES, ONE_ROUND: RISING})
    assert rose["inconclusive"] is False and rose["failed_checks"] == [] and rose["branches"] == passing["branches"] and rose["primary"]["difference"] == passing["primary"]["difference"]
    of_the_loss = rose["the_loss_of_each_training"]
    assert of_the_loss["rose_by_more"] == [ONE_ROUND] and of_the_loss[ONE_ROUND]["passes"] is False and of_the_loss[ONE_ROUND]["rise"] > of_the_loss[ONE_ROUND]["allowed"] > 0 > of_the_loss[TWO_ROUNDS]["rise"]
    assert list(rose["can_this_run_be_read"][ONE_ROUND]) == [THE_ADAPTER, "it_still_writes_proofs", "lean_answered", "no_barred_row"] and passing["the_loss_of_each_training"]["rose_by_more"] == []
    (printed,) = [line for line in rose["lines"] if "PRINTED, DECIDING NOTHING" in line]
    assert "`one_round`: 1.9625 over the last 4 rows against 1.0625 over the first, a rise of 0.9 where 0.04564 would be allowed: IT ROSE BY MORE" in printed
    assert printed.endswith("THE LOSS ROSE BY MORE THAN THAT IN `one_round`: named, and the run is read all the same")
    assert any(f"{SAY}: PRIMARY: {CYCLE_ADDS}" in line for line in rose["lines"]) and not any(INCONCLUSIVE in line for line in rose["lines"])
    # THE ADAPTER HALF STILL DECIDES: an adapter that is the start adapter's (no number changed), or one that was never compared: INCONCLUSIVE, one verdict, and nothing else is said.
    for against, said in (({**CHANGED, "elements_changed": 0}, "0 of the 1,000 numbers of its adapter are not the start adapter's: FAIL"), (None, "its adapter was not compared with the start adapter's: FAIL")):
        same = _report(aside={TWO_ROUNDS: HA_MORE}, trains={**TRAINS, TWO_ROUNDS: {**TRAINS[TWO_ROUNDS], "against_the_start_adapter": {TWO_ROUNDS: against} if against else {}}})
        assert same["inconclusive"] is True and same["failed_checks"] == [f"`two_rounds`: {THE_ADAPTER}"] and any(f"`two_rounds` (" in line and said in line for line in same["lines"])
        assert same["branches"] == {"the_primary": INCONCLUSIVE, "compounding": INCONCLUSIVE, "hard_to_find_or_long": INCONCLUSIVE, "reach": INCONCLUSIVE, "what_follows": NOT_READ}
        assert same["primary"]["difference"] == {"mean": None} and same["primary"]["measured_and_not_read"]["difference"] == passing["primary"]["difference"]
        assert list(same["beside"]) == ["what", "measured_and_not_read"] and same["reach"]["measured_and_not_read"] == passing["reach"]["solved"] and "solved" not in same["reach"]
        assert not any(line.startswith(f"{SAY}: {mark}") for line in same["lines"] for mark in ("PRIMARY", "COMPOUNDING", "HARD TO FIND", "REACH", "WHAT FOLLOWS", "BESIDE"))
        assert any(line.startswith(f"{SAY}: {INCONCLUSIVE}. the run could not have been read (`two_rounds`: {THE_ADAPTER})") for line in same["lines"]) and f": {INCONCLUSIVE}. Search 2 solved" in same["headline"]
    # ANY model's check, and each kind: a held-out problem among the rows `long_ordinary` was trained on; an own row that is no base-map problem; a model that writes no proofs.
    held_out = [*TRAINED[LONG_ORDINARY][:-1], {**TRAINED[LONG_ORDINARY][-1], "id": f"{GOAL_IDS[0]}#pretraining", "problem_id": GOAL_IDS[0]}]
    assert _report(trained={**TRAINED, LONG_ORDINARY: held_out})["failed_checks"] == ["`long_ordinary`: no_barred_row"]
    stray = [{**TRAINED[TWO_ROUNDS][0], "problem_id": "not_of_the_base_map"}, *TRAINED[TWO_ROUNDS][1:]]
    assert _report(trained={**TRAINED, TWO_ROUNDS: stray})["failed_checks"] == ["`two_rounds`: no_barred_row"]
    # A ROW OF A HELD-ASIDE PROBLEM, of the model's own or published: the held-aside set is the test, and the run is INCONCLUSIVE.
    for place in (0, -1):
        on_the_set = [*TRAINED[ONE_ROUND]]
        on_the_set[place] = {**on_the_set[place], "problem_id": ASIDE[3]}
        tested = _report(trained={**TRAINED, ONE_ROUND: on_the_set})
        assert tested["failed_checks"] == ["`one_round`: no_barred_row"] and tested["can_this_run_be_read"][ONE_ROUND]["no_barred_row"]["held_aside_problems"] == 1
        assert any("`one_round`, CHECK 4" in line and "0 held-out problems, 1 held-aside problems" in line and line.endswith("FAIL") for line in tested["lines"])
    assert no_barred_row(TRAINED[TWO_ROUNDS], set(GOAL_IDS), 0, BASE, ASIDE)["passes"] is True and no_barred_row(TRAINED[TWO_ROUNDS], set(GOAL_IDS), 0, BASE, BASE[:1])["first"] == [BASE[0]]
    assert no_barred_row([], set(), 0, BASE, ASIDE)["passes"] is False
    silent = _model()
    silent = {**silent, "rungs": [{**row, "attempts_without_an_answer": row["episodes"]} for row in silent["rungs"]]}
    assert "`published2`: it_still_writes_proofs" in _report({PUBLISHED2: silent})["failed_checks"]
    # LEAN DID NOT ANSWER a held-aside set (the stored model's is one of them): the check fails, and the report is NOT TO BE READ.
    for name in (TWO_ROUNDS, "second_full", "pre_r64"):
        unanswered = _report(aside={name: _aside(HA_BASE, attempts_without_an_answer=3)})
        assert unanswered["failed_checks"] == [THE_HELD_ASIDE] and unanswered["ok"] is False and unanswered["not_to_be_read"] == [held_aside_key(name)]
        assert unanswered["can_this_run_be_read"]["the_held_aside_sets"]["sets_not_answered"] == [held_aside_key(name)] and any("NOT TO BE READ" in line for line in unanswered["lines"])


def test_too_few_round_two_solves_nothing_was_trained_and_the_search_is_the_finding():
    report = _report(prepare=TOO_FEW, measured=())
    assert report["inconclusive"] is True and report["failed_checks"] == [THE_SEARCH] and report["measured"] == [] and report["ok"] is True and (report["deep_solves"], report["rows_a_set"]) == (0, 0)
    assert report["branches"] == {"the_primary": INCONCLUSIVE, "compounding": INCONCLUSIVE, "hard_to_find_or_long": INCONCLUSIVE, "reach": INCONCLUSIVE, "what_follows": NOT_READ}
    assert list(report["can_this_run_be_read"]) == ["what", "search_2"] and report["search_2"]["the_finding"] is True and report["models"] == {} and report["primary"]["difference"] == {"mean": None}
    assert report["lines"][0].endswith("NO MODEL WAS TRAINED: search 2 found too few round-2 solves")
    assert any(line == f"{SAY}: CHECK, search 2 found at least 5 round-2 solves: 4: FAIL" for line in report["lines"])
    assert any(line.startswith(f"{SAY}: SEARCH 2, WHICH IS THE FINDING: 60 problems attempted") for line in report["lines"])
    assert any(line.startswith(f"{SAY}: THE LINES OF THE ROWS (the deep solves of both rounds; nothing was trained): round 1 rows: 20 proofs") for line in report["lines"])
    assert any(line.startswith(f"{SAY}: {INCONCLUSIVE}. the run could not have been read ({THE_SEARCH})") and line.endswith("the search is the finding") for line in report["lines"])
    # The four models, or none; and never a model's rows without enough round-2 solves.
    with pytest.raises(ValueError, match="the report is given the trained models .'published2', 'one_round'.: it reads all four"):
        _report(measured=(PUBLISHED2, ONE_ROUND))
    with pytest.raises(ValueError, match="it reads all four .*, or none when search 2 found too few round-2 solves"):
        _report(prepare=TOO_FEW)
    with pytest.raises(ValueError, match="the report is given held-aside rows of .*each on the same problems"):
        _report(aside={TWO_ROUNDS: _aside(HA_BASE, problems=BASE[:12])})


def test_two_rounds_is_read_against_the_stored_models_beside_and_a_stored_model_that_is_not_there_is_said_so():
    # M(1) (round 1's `deep`) and `second_full`, each from its stored rows on G; `second_full` on the held-aside set too, where it was measured.
    resamples, seed = EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"]
    report = _report({TWO_ROUNDS: _model(on_g=MORE)}, {TWO_ROUNDS: HA_MORE, "second_full": HA_MOST}, beside={"deep": _model(on_g=LESS)["goal"], "second_full": _model()["goal"]})
    against = report["beside"]["two_rounds_against_the_stored_models"]
    assert list(against) == ["deep", "second_full"] and list(against["deep"]) == ["on_g", "all_of_g", "long_goal_problems"]
    assert against["deep"]["on_g"]["on_all_of_g"] == per_attempt(_on_g(_model(on_g=MORE)), _on_g(_model(on_g=LESS)), GOAL_IDS, resamples, seed)
    assert against["second_full"]["on_g"]["on_all_of_g"] == per_attempt(_on_g(_model(on_g=MORE)), _on_g(_model()), GOAL_IDS, resamples, seed)
    assert against["deep"]["on_g"]["on_all_of_g"]["mean"] > against["second_full"]["on_g"]["on_all_of_g"]["mean"] > 0
    assert against["second_full"]["on_the_held_aside_set"]["difference"] == _on_ha(HA_MORE, HA_MOST) and against["second_full"]["on_the_held_aside_set"]["difference"]["mean"] < 0
    assert against["second_full"]["on_the_held_aside_set"]["held_aside"]["successes"] == 48 and "on_the_held_aside_set" not in against["deep"]
    assert report["branches"]["the_primary"] == CYCLE_ADDS                     # ... and no branch reads a stored model: `second_full` above `two_rounds` there moves nothing
    text = "\n".join(report["lines"])
    assert "`two_rounds` against the stored `deep` (M(1), the model round 1 trained, which made search 2) on all of G: +" in text
    assert "`two_rounds` against `second_full` ON THE HELD-ASIDE SET: -0.06250" in text and "NOT THERE" not in text
    # `second_full` NOT THERE: neither its rows on G nor a measurement on the held-aside set: said, nothing refused, and the reads are what they were.
    without = _report({TWO_ROUNDS: _model(on_g=MORE)}, {TWO_ROUNDS: HA_MORE}, stored=False, prepare={**PREPARE, "the_stored_model": {"model": "second_full", "there": False, "why": "the run is not on the box"}})
    assert list(without["beside"]["two_rounds_against_the_stored_models"]) == ["deep"] and without["branches"] == report["branches"] and without["primary"] == report["primary"]
    assert list(without["can_this_run_be_read"]["the_held_aside_sets"]["share_without_an_answer"]) == [held_aside_key(name) for name in ("pre_r64", *MODELS)]
    text = "\n".join(without["lines"])
    assert "`second_full` (L4p's whole second pass) on the held-aside set: NOT THERE (the run is not on the box); nothing is refused for it" in text
    assert "`two_rounds` against the stored `second_full` (L4p's whole second pass, the best pretraining-only model stored) on G: NOT THERE" in text and "ON THE HELD-ASIDE SET: NOT THERE (it was not measured there)" in text
    # ITS ROWS ON G THERE, its adapter not: read on G, NOT THERE on the held-aside set.
    half = _report(stored=False, beside={"deep": _model()["goal"], "second_full": _model()["goal"]})
    assert list(half["beside"]["two_rounds_against_the_stored_models"]["second_full"]) == ["on_g", "all_of_g", "long_goal_problems"]
    assert any("`two_rounds` against `second_full` ON THE HELD-ASIDE SET: NOT THERE" in line for line in half["lines"])


def test_the_overfitting_reads_the_timeouts_the_never_solved_and_a_smoke_runs_line_stand_beside():
    # OVERFITTING, each model against the start model: NARROWER when it lost more goal problems than it gained under 0.05 (`l4_dose.overfitting`); it names no branch.
    narrow = ([6, 6, 6, 4, 4, 4, 0, 0, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 0, 0, 0, 0, 0, 0])            # no problem of 4 lines or more solved: lost 6 (p = 0.03)
    report = _report({TWO_ROUNDS: _model(on_g=narrow)}, never=[GOAL_IDS[0], GOAL_IDS[-1], "not_a_goal_problem"])
    overfit = report["beside"]["overfitting"]
    assert (overfit[TWO_ROUNDS]["narrower"], overfit[TWO_ROUNDS]["gained"], overfit[TWO_ROUNDS]["lost"]) == (True, 0, 6) and overfit[ONE_ROUND]["narrower"] is False
    assert any(f"OVERFITTING, `two_rounds` against `pre_r64`: NARROWER (gained 0, lost 6" in line for line in report["lines"]) and report["branches"]["the_primary"] == NOT_SEPARATED
    assert list(overfit[TWO_ROUNDS]["the_three_rungs"]) == ["below", "in", "above"] and "share_of_distinct_attempts_on_g" in overfit[TWO_ROUNDS]
    # ... each AGAINST THE START MODEL, never against another trained model: two models that lost the same six problems are both NARROWER.
    both = _report({TWO_ROUNDS: _model(on_g=narrow), PUBLISHED2: _model(on_g=narrow)})["beside"]["overfitting"]
    assert (both[TWO_ROUNDS]["narrower"], both[PUBLISHED2]["narrower"], both[ONE_ROUND]["narrower"]) == (True, True, False)
    # THE NEVER-SOLVED, when their ids are given: how many each model solves; `two_rounds` lost the long one.
    each = report["beside"]["each_model"]
    assert (each[PUBLISHED2]["never_solved_before"]["solved"], each[TWO_ROUNDS]["never_solved_before"]["solved"], each[TWO_ROUNDS]["never_solved_before"]["goal_problems"]) == (2, 1, 2)
    assert any("the goal problems nothing stored had solved: of the 2 listed, `published2` 2, `one_round` 2, `two_rounds` 1, `long_ordinary` 2" in line for line in report["lines"])
    assert any("the goal problems nothing stored had solved: not given" in line for line in _report()["lines"])
    assert (each[PUBLISHED2]["solved_reliably"], each[TWO_ROUNDS]["all_of_g"]["solved_at_least_once"], each[PUBLISHED2]["all_of_g"]["solved_at_least_once"]) == (_report()["beside"]["each_model"]["pre_r64"]["solved_reliably"], 6, 12)
    # TIMEOUTS: each measured set's share, and where two models' shares differ by more than a point on one kind of set.
    slow = _report(aside={LONG_ORDINARY: _aside(HA_BASE, attempts_timed_out=2)})
    timeouts = slow["beside"]["timed_out_in_lean"]
    assert timeouts["share"][held_aside_key(LONG_ORDINARY)] == 0.125 and timeouts["share"][held_aside_key("second_full")] == 0.0 and len(timeouts["apart"]) == 5
    assert {tuple(entry["models"]) for entry in timeouts["apart"]} == {(name, LONG_ORDINARY) for name in ("pre_r64", PUBLISHED2, ONE_ROUND, TWO_ROUNDS)} | {(LONG_ORDINARY, "second_full")}
    assert any("TIMEOUTS" in line and "held_aside: `pre_r64` 0.0 against `long_ordinary` 0.125" in line for line in slow["lines"]) and any("TIMEOUTS" in line and line.endswith("nowhere") for line in report["lines"])
    # A SMOKE RUN: what the first step went on from, and the rows that are not the rule's, are said; one sampling of G is said.
    smoke = {"what": "a smoke run alone", "problems_at_least": 2, "round_one_read_though_inconclusive": True, "round_ones_failed_checks": ["`shallow`: its_training_ran"]}
    prepare = {**PREPARE, "smoke": smoke, "rows_not_taken_by_the_rule": {"what": "a smoke run alone", TWO_ROUNDS: 2, LONG_ORDINARY: 1}, "the_search": {**PREPARE["the_search"], "taken_by_the_runs_minimum": 2}}
    said = _report(prepare=prepare)
    assert any("A SMOKE RUN: round 1's run reads INCONCLUSIVE (`shallow`: its_training_ran) and was read all the same; 2 of search 2's problems and 2 of `two_rounds`'s own rows, 1 of `long_ordinary`'s own "
               "rows are of problems the rule did not give" in line for line in said["lines"]) and said["smoke"] == smoke
    # A SHORTFALL of the length matching is printed with its bins.
    short = [{**BINS[0]}, {**BINS[1]}, {**BINS[2], "asked_by_longer_bins": 4, "taken": 13}, {**BINS[3], "offered": 7, "taken": 7, "shortfall": 4}, {**BINS[4]}]
    printed = _report(prepare={**PREPARE, "matched_in_length": {**PREPARE["matched_in_length"], "bins": short, "bins_short": ["8-15"], "rows_short": 4}})["lines"][-1]
    assert "8-15 lines: 11 wanted, the map offers 7, 7 taken, 4 SHORT and filled from the next shorter bin" in printed and printed.endswith("SHORTFALL: 4 rows, in 8-15")
