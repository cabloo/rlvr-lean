"""L5, do deep solves teach what ordinary solves do not: the deep search's candidates and its seeded draw, the held-aside
set (a draw of its own among the candidates the search did not take, disjoint from everything trained on), a deep solve's
row (one verified attempt, drawn as a round draws one, with the place of its first success), D and the cap, the ordinary
solves, the three sets of 2D rows and what they are held to (the same published rows, the first of the one hash order),
what is barred, the primary's three outcomes ON THE HELD-ASIDE SET and its edge at zero, the mechanism, the replication,
the branch for what follows, the deep search's table, and the report on hand-made rows with a known answer. Spec:
docs/spec/ladder-loop.spec.md, "L5: do DEEP solves teach what ordinary solves do not? One step-matched check". Pure:
no model, no Lean. The stage end to end is `test_ladder_l5_stage.py`."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3d1 import EVALUATION, GOAL_IDS, GROUPS, LENGTHS  # noqa: E402 - Step 1's hand-made world: 12 goal problems, 3 a length group
from test_ladder_l4 import CHANGED, _made  # noqa: E402 - L4's hand-made models
from test_ladder_l4_dose import MORE, PRETRAIN_HALF, REHEARSE_ONLY, REWARD_REHEARSE, START  # noqa: E402 - L4p's hand-made start model and one above it, and its two stored sets
from test_ladder_l4_rows import FALLING, SAMPLINGS, _kept  # noqa: E402
from test_ladder_l4_rows2 import PRETRAINING, _draw_order, _solved  # noqa: E402 - 40 rows the start model was pretrained on, without text

from rlvr_lean.domain.ladder_round import l4_rows, l5  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, PRETRAIN, goal_set_again, half_of  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_dose import MATCHED, THEY_COST, dose_sets  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import SIGN_TEST_LEVEL, THE_MINIMUM, THE_RULE, rehearsal_rows  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import NOT_SEPARATED, TEXT_FIELDS, refuse_text  # noqa: E402
from rlvr_lean.domain.ladder_round.l4b import breadth_beside  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import (  # noqa: E402
    ABOVE,
    BELOW,
    CAP_LABEL,
    DEEP,
    DEEP_ADDS,
    DEPTH_MATTERS,
    FOLLOWS,
    HELD_ASIDE_LABEL,
    HOLDS_ZERO,
    MECHANISM,
    MODELS,
    MORE_SEEDS,
    NOT_SHOWN,
    OWN,
    PRIMARY,
    PUBLISHED,
    RAISE_THE_DEPTH,
    REPLICATION,
    SEARCH_LABEL,
    SHALLOW,
    SHALLOW_LABEL,
    NEEDS_A_DECISION,
    bands,
    capped,
    enough_deep_solves,
    failed_every_attempt,
    held_aside,
    hold_the_sets,
    mechanism,
    no_barred_row,
    ordinary_draw,
    primary_branch,
    refuse_barred,
    refuse_overlap,
    replication,
    search_problems,
    search_table,
    solve_rows,
    the_sets,
    what_follows,
)
from rlvr_lean.domain.ladder_round.rounds import sign_test  # noqa: E402
from rlvr_lean.domain.ladder_round.training_set import training_examples  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.reporting.ladder_ceiling import per_attempt, solved  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import LABEL, SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l5 import CANNOT_SAY, FAVOURED, IS, PAIRS, THE_HELD_ASIDE, THE_SEARCH, build_l5_report, held_aside_key, pair_name  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS, OWN_SETTINGS = CONFIG["ladder_loop"]["l4"], CONFIG["ladder_loop"]["l5"]
SEED = 0
NOT_MEASURED = {"problems": 0, "mean": None, "low": None, "high": None}


def test_the_names_and_the_settings_are_the_specs():
    assert MODELS == (PUBLISHED, SHALLOW, DEEP) == ("published", "shallow", "deep") and OWN == (DEEP, SHALLOW)      # trained and measured in this order; a row of the model's own is of one of two origins
    assert (DEEP_ADDS, NOT_SEPARATED, THEY_COST) == ("DEEP SOLVES ADD WHAT THE SAME NUMBER OF PUBLISHED PROOFS DO NOT", "NOT SEPARATED AT THIS SIZE", "THEY COST")
    assert PRIMARY == (DEEP_ADDS, NOT_SEPARATED, THEY_COST, INCONCLUSIVE, NOT_READ) and MECHANISM == (DEPTH_MATTERS, NOT_SHOWN, INCONCLUSIVE, NOT_READ)
    assert (DEPTH_MATTERS, NOT_SHOWN) == ("DEPTH MATTERS", "NOT SHOWN") and REPLICATION == (ABOVE, HOLDS_ZERO, BELOW, INCONCLUSIVE, NOT_READ) == (
        "ABOVE ZERO", "HOLDS ZERO", "BELOW ZERO", INCONCLUSIVE, NOT_READ)
    assert FOLLOWS == (MORE_SEEDS, NEEDS_A_DECISION, RAISE_THE_DEPTH, NOT_READ) and len({SEARCH_LABEL, CAP_LABEL, SHALLOW_LABEL, HELD_ASIDE_LABEL, l4_rows.REHEARSAL_LABEL, l4_rows.KEEP_LABEL}) == 6
    # The section's numbers, in the config and nowhere in the code: 1,000 problems, 56 more attempts (64 with the map's 8), at most 400, at least 150, 1 to 3 of the map's 8;
    # and its amendment's: 600 held-aside problems, 16 fresh attempts at each by each of four models (38,400 in all). ONE search for every seed: seed 0's (the further seeds).
    assert OWN_SETTINGS == {"start": "pre_r64", "models": list(MODELS), "search_seed": 0,
                            "deep": {"problems": 1000, "attempts": 56, "sampling_place": 50, "held_aside": 600, "held_aside_attempts": 16, "held_aside_sampling_place": 51, "cap": 400,
                                     "minimum": 150}, "shallow": {"solved_at_least": 1, "solved_at_most": 3}}
    assert OWN_SETTINGS["deep"]["held_aside"] * OWN_SETTINGS["deep"]["held_aside_attempts"] * (1 + len(MODELS)) == 38400
    assert OWN_SETTINGS["start"] == SETTINGS["rows2"]["start"] == SETTINGS["dose"]["start"] and CONFIG["ladder_loop"]["base_map"]["episodes"] + OWN_SETTINGS["deep"]["attempts"] == 64
    # THE SEARCH'S SAMPLING SEED is no stored sampling's: its place is none of the places a measurement has (the kinds 0 to 3, the rounds' 11 to 16, the control's 20, L3's 30 to
    # 33), and at no seed is it the base's placing seed, the map's, or one of L4t's hot ones.
    from rlvr_lean.gpu import ladder_l2, ladder_l3a, ladder_l3a2, ladder_l3c
    places = {0, 1, 2, 3, *(ladder_l2.ROUND_SEED_PLACE + number for number in range(1, 7)), ladder_l2.CONTROL_SEED_PLACE, ladder_l3a.SEED_PLACE, ladder_l3a2.SEED_PLACE,
              ladder_l3c.SEED_PLACE, ladder_l3c.PILOT_SEED_PLACE}
    place, aside_place, first = OWN_SETTINGS["deep"]["sampling_place"], OWN_SETTINGS["deep"]["held_aside_sampling_place"], CONFIG["ladder_loop"]["round"]["sampling_seed"]
    others = {CONFIG["ladder_loop"]["goal"]["sampling_seed"], CONFIG["ladder_loop"]["base_map"]["sampling_seed"], *SETTINGS["rows"]["sampling_seeds_hot"]}
    for own in (place, aside_place):         # the search's, and the held-aside set's ONE seed for its four models: each a seed of its own, and not the other's
        assert own not in places and max(places) < 40 and not {first + 100 * seed + own for seed in range(10)} & {*others, *(first + 100 * seed + taken for seed in range(10) for taken in places)}
    assert (place, aside_place) == (50, 51)
    assert set(IS) == set(MODELS) and len(CANNOT_SAY) == 4 and "COMPOUND" in CANNOT_SAY[0] and "One seed" in CANNOT_SAY[1] and "A depth of 64 is shallow" in CANNOT_SAY[2]
    assert "favoured by the selection" in CANNOT_SAY[3] and "that selection favours a trained model against it, as on G'" in FAVOURED
    assert PAIRS == ((DEEP, PUBLISHED), (DEEP, SHALLOW), (SHALLOW, PUBLISHED)) and pair_name(DEEP, PUBLISHED) == "deep_minus_published"


# ------------------------------------------------------------------------------------------ the deep search
# A hand-made map of 24 base-map problems, 8 attempts each: eight failed every time, six solved once, twice or three times, the others more often.
MAP_K = [0] * 8 + [1, 1, 2, 2, 3, 3] + [4, 5, 6, 7, 8, 8, 8, 8, 4, 5]
MAP_ROWS = [{"problem_id": f"b{index}", "resolved": k, "episodes": 8, "sides": 1, "attempts_without_an_answer": 0} for index, k in enumerate(MAP_K)]
BASE_MAP = [row["problem_id"] for row in MAP_ROWS]
in_search_order = lambda ids, seed=SEED: sorted(ids, key=lambda key: rank(seed, SEARCH_LABEL, key))      # noqa: E731


def test_the_deep_search_draws_among_the_problems_failed_in_every_map_attempt_and_among_no_other():
    candidates = failed_every_attempt(MAP_ROWS)
    assert [row["problem_id"] for row in candidates] == [f"b{index}" for index in range(8)] and all(row["resolved"] == 0 for row in candidates)
    drawn = search_problems(MAP_ROWS, 5, SEED)
    # A seeded draw: the five of the smallest content hash of the seed and the problem's id, in that order; each failed 8 of 8, each taken by the rule.
    assert [row["problem_id"] for row in drawn] == in_search_order(row["problem_id"] for row in candidates)[:5] != [f"b{index}" for index in range(5)]
    assert all(row == {"problem_id": row["problem_id"], "map_k": 0, "map_n": 8, "taken_by": THE_RULE} for row in drawn)
    assert [row["problem_id"] for row in search_problems(MAP_ROWS, 8, SEED)][:5] == [row["problem_id"] for row in drawn]             # a larger draw begins with the smaller
    assert {row["problem_id"] for row in search_problems(MAP_ROWS, 8, SEED)} == {f"b{index}" for index in range(8)}
    assert [row["problem_id"] for row in search_problems(MAP_ROWS, 5, 1)] == in_search_order((row["problem_id"] for row in candidates), 1)[:5] != [row["problem_id"] for row in drawn]
    # A problem solved ONCE in the map is no candidate; and fewer candidates than the draw is a refusal, never a smaller or looser draw.
    assert "b8" not in {row["problem_id"] for row in search_problems(MAP_ROWS, 8, SEED)}
    with pytest.raises(ValueError, match="the map holds 8 problems its model failed in every attempt and the deep search draws 9 of them: refused"):
        search_problems(MAP_ROWS, 9, SEED)
    # A SMOKE RUN ALONE: what the rule gives, and the map's other problems in the same hash order up to the least number, each marked as NOT the rule's.
    solved_all = [row for row in MAP_ROWS if row["resolved"] > 0]
    loosely = search_problems(solved_all, 1000, SEED, at_least=2)
    assert [row["problem_id"] for row in loosely] == in_search_order(row["problem_id"] for row in solved_all)[:2] and {row["taken_by"] for row in loosely} == {THE_MINIMUM}
    assert all(row["map_k"] > 0 for row in loosely) and THE_MINIMUM == "the run's minimum" != THE_RULE
    mixed = search_problems(MAP_ROWS[6:], 1000, SEED, at_least=4)                    # two candidates, and two more by the minimum
    assert [row["taken_by"] for row in mixed] == [THE_RULE, THE_RULE, THE_MINIMUM, THE_MINIMUM] and {row["problem_id"] for row in mixed[:2]} == {"b6", "b7"}
    assert [row["taken_by"] for row in search_problems(MAP_ROWS, 5, SEED, at_least=2)] == [THE_RULE] * 5          # the rule gives enough: nothing is taken loosely
    assert search_problems(MAP_ROWS, 5, SEED, at_least=2) == drawn


def test_the_held_aside_set_is_a_draw_of_its_own_among_the_candidates_the_search_did_not_take():
    searched = [row["problem_id"] for row in search_problems(MAP_ROWS, 5, SEED)]
    left = [f"b{index}" for index in range(8) if f"b{index}" not in searched]
    in_its_order = lambda ids, seed=SEED: sorted(ids, key=lambda key: rank(seed, HELD_ASIDE_LABEL, key))      # noqa: E731
    aside = held_aside(MAP_ROWS, searched, 2, SEED)
    # Candidates (failed 8 of 8) that the search's draw did NOT take: the two of the smallest hash under ITS OWN label, in that order; each taken by the rule.
    assert len(left) == 3 and [row["problem_id"] for row in aside] == in_its_order(left)[:2] and not {row["problem_id"] for row in aside} & set(searched)
    assert all(row == {"problem_id": row["problem_id"], "map_k": 0, "map_n": 8, "taken_by": THE_RULE} for row in aside)
    assert [row["problem_id"] for row in held_aside(MAP_ROWS, searched, 3, SEED)] == in_its_order(left) and [row["problem_id"] for row in held_aside(MAP_ROWS, searched, 3, SEED)][:2] == [
        row["problem_id"] for row in aside]
    # A DRAW OF ITS OWN: with nothing searched it is not the search's own order of the same candidates, and another seed is another draw.
    every = [f"b{index}" for index in range(8)]
    assert [row["problem_id"] for row in held_aside(MAP_ROWS, [], 8, SEED)] == in_its_order(every) != in_search_order(every)
    assert [row["problem_id"] for row in held_aside(MAP_ROWS, [], 8, 1)] == in_its_order(every, 1) != in_its_order(every)
    # A problem solved once in the map is no candidate, and a searched problem is never held aside; fewer left than the draw is a refusal.
    assert {row["problem_id"] for row in held_aside(MAP_ROWS, [], 8, SEED)} == set(every) and {row["problem_id"] for row in held_aside(MAP_ROWS, every[:4], 4, SEED)} == set(every[4:])
    with pytest.raises(ValueError, match="the deep search's draw left 3 of the problems the map's model failed in every attempt and the held-aside set draws 4 of them: refused"):
        held_aside(MAP_ROWS, searched, 4, SEED)
    # A SMOKE RUN ALONE: what the rule gives, then the map's other problems the search did not take, in the same hash order, each marked as NOT the rule's.
    solved_all = [row for row in MAP_ROWS if row["resolved"] > 0]
    loosely = held_aside(solved_all, ["b8"], 600, SEED, at_least=2)
    assert [row["problem_id"] for row in loosely] == in_its_order(row["problem_id"] for row in solved_all if row["problem_id"] != "b8")[:2] and {row["taken_by"] for row in loosely} == {THE_MINIMUM}
    assert [row["taken_by"] for row in held_aside(MAP_ROWS, searched, 600, SEED, at_least=5)] == [THE_RULE] * 3 + [THE_MINIMUM] * 2
    assert held_aside(MAP_ROWS, searched, 2, SEED, at_least=1) == aside                                                              # the rule gives enough: nothing is taken loosely


def test_no_held_aside_problem_is_searched_or_is_a_row_of_a_training_set():
    sets = the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED)
    aside = ["b4", "b5", "b6"]
    assert refuse_overlap(aside, ["b0", "b1", "b2", "b3"], sets) is None and refuse_overlap(aside, [], {}) is None and refuse_overlap([], ["b0"], sets) is None
    with pytest.raises(ValueError, match="1 problems of the held-aside set are problems the deep search attempted, 0 are rows of `published`, 0 are rows of `shallow`, 0 are rows of "
                                         "`deep` .first: b4.: no model here trains on a held-aside problem, and none of them was searched. Refused"):
        refuse_overlap(aside, ["b0", "b4"], sets)
    with pytest.raises(ValueError, match="1 problems of the held-aside set are problems the deep search attempted .first: b5."):
        refuse_overlap(aside, ["b5"], {})
    for problem, said in (("b1", "0 are rows of `published`, 0 are rows of `shallow`, 1 are rows of `deep` .first: b1."),               # a deep row's problem
                          ("b9", "0 are rows of `published`, 1 are rows of `shallow`, 0 are rows of `deep` .first: b9."),               # an ordinary solve's
                          (sets[PUBLISHED][-1]["problem_id"], "1 are rows of `published`")):                                         # a published row's
        with pytest.raises(ValueError, match=f"0 problems of the held-aside set are problems the deep search attempted, {said}"):
            refuse_overlap([*aside, problem], ["b0"], sets)


def _problem(problem_id, negation=True, side="true"):
    return {"problem_id": problem_id, "kind": "stp", "side": side, "statement": f"theorem {problem_id} : P := by\n",
            "negation": f"theorem negation_of_{problem_id} : ¬ P := by\n" if negation else None, "one_side_reason": None}


def _attempt(problem_id, episode, status, completion="  done\n", side="statement", set_name="l5_deep_search"):
    return {"attempt_id": f"{problem_id}#{set_name}#{side}#{episode}", "problem_id": problem_id, "side": side, "episode": episode, "completion": completion, "status": status}


def _result(problem_id, resolved, episodes=56, **more):
    return {"problem_id": problem_id, "resolved": resolved, "episodes": episodes, "sides": 1, "attempts_without_an_answer": 0, **more}


SEARCHED = [_problem(f"b{index}") for index in range(5)]
# b0: verified at its attempts 3, 20 and 40 (three proofs of 1, 2 and 3 lines); b1: never; b2: its last attempt only; b3: its first; b4: the negation, at attempt 10.
ATTEMPTS = [_attempt("b0", 3, "verified", "  a\n"), _attempt("b0", 20, "verified", "  a\n  b\n"), _attempt("b0", 40, "verified", "  a\n  b\n  c\n"), _attempt("b0", 0, "lean_error"),
            _attempt("b1", 0, "lean_error"), _attempt("b1", 55, "timeout"), _attempt("b2", 55, "verified"), _attempt("b2", 54, "lean_error"), _attempt("b3", 0, "verified"),
            _attempt("b4", 10, "verified", "  intro h\n", side="negation"), _attempt("b4", 10, "contradicted_side_not_checked")]
RESULTS = [_result("b0", 3), _result("b1", 0), _result("b2", 1), _result("b3", 1), _result("b4", 1)]


def test_a_deep_solves_row_is_one_verified_attempt_drawn_as_a_round_draws_one_with_the_place_of_its_first_success():
    rows = solve_rows(SEARCHED, ATTEMPTS, RESULTS, SEED, DEEP, attempts_before=8)
    assert [row["problem_id"] for row in rows] == ["b0", "b2", "b3", "b4"]                                   # one row a problem solved, in the problems' order; b1 has none
    # THE ATTEMPT IS THE ONE A ROUND DRAWS: `training_set.training_examples`, the same function, by the same seed.
    examples = training_examples(SEARCHED, ATTEMPTS, SEED)
    assert [{key: row[key] for key in example} for row, example in zip(rows, examples)] == examples and [row["id"] for row in rows] == [example["attempt_id"] for example in examples]
    chosen = min((attempt for attempt in ATTEMPTS if attempt["problem_id"] == "b0" and attempt["status"] == "verified"), key=lambda attempt: rank(SEED, "training_target", attempt["attempt_id"]))
    assert rows[0]["id"] == chosen["attempt_id"] and rows[0]["completion"] == chosen["completion"] and rows[0]["verified_attempts"] == 3
    assert {solve_rows(SEARCHED, ATTEMPTS, RESULTS, seed, DEEP, 8)[0]["id"] for seed in range(12)} == {attempt["attempt_id"] for attempt in ATTEMPTS[:3]}       # the draw is the seed's
    # THE PLACE OF THE FIRST SUCCESS counts the map's 8 attempts: the first of a search of 56 is the 9th, the last the 64th. It is the problem's FIRST verified attempt, whichever is drawn.
    assert {row["problem_id"]: row["first_success"] for row in rows} == {"b0": 12, "b2": 64, "b3": 9, "b4": 19}
    assert all(row["first_success"] == 12 for seed in range(12) for row in solve_rows(SEARCHED, ATTEMPTS, RESULTS, seed, DEEP, 8)[:1])
    assert [row["first_success"] for row in solve_rows(SEARCHED, ATTEMPTS, RESULTS, SEED, DEEP)] == [4, 56, 1, 11]              # counted from the sampling's own first attempt
    # Its k of n in that sampling, the lines of its proof, its origin; on whichever side was proved.
    assert all((row["origin"], row["n"]) == (DEEP, 56) for row in rows) and [row["k"] for row in rows] == [3, 1, 1, 1] and rows[0]["lines"] == rows[0]["completion"].count("\n")
    assert (rows[3]["side"], rows[3]["theorem"]) == ("negation", "theorem negation_of_b4 : ¬ P := by\n") and rows[1]["theorem"] == "theorem b2 : P := by\n"
    assert set(rows[0]) == {"id", "problem_id", "side", "attempt_id", "theorem", "completion", "verified_attempts", "origin", "k", "n", "lines", "first_success"}
    # Only the problems asked for are read: the attempts and results of another problem change nothing.
    assert solve_rows(SEARCHED[:2], ATTEMPTS, RESULTS, SEED, DEEP, 8) == rows[:1]
    assert [row["origin"] for row in solve_rows(SEARCHED, ATTEMPTS, RESULTS, SEED, SHALLOW)] == [SHALLOW] * 4
    # REFUSED: a stored sampling that does not agree with itself, and an origin that is not the model's own.
    with pytest.raises(ValueError, match="the stored sampling does not agree with itself: 5 of its 5 problems are counted as resolved, 4 of the 5 asked for have a verified attempt .first that differs: b1."):
        solve_rows(SEARCHED, ATTEMPTS, [*RESULTS[:1], _result("b1", 1), *RESULTS[2:]], SEED, DEEP)
    with pytest.raises(ValueError, match="3 of its 5 problems are counted as resolved, 4 of the 5 asked for have a verified attempt .first that differs: b3."):
        solve_rows(SEARCHED, ATTEMPTS, [*RESULTS[:3], _result("b3", 0), RESULTS[4]], SEED, DEEP)
    with pytest.raises(ValueError, match="first that differs: b4"):
        solve_rows(SEARCHED, ATTEMPTS, RESULTS[:4], SEED, DEEP)
    with pytest.raises(ValueError, match="4 of its 4 problems are counted as resolved, 4 of the 5 asked for have a verified attempt .first that differs: b1."):
        solve_rows(SEARCHED, ATTEMPTS, [row for row in RESULTS if row["problem_id"] != "b1"], SEED, DEEP)       # a problem never solved, and without its result
    with pytest.raises(ValueError, match="'pretraining' is not where a row of the model's own is from: it is one of .'deep', 'shallow'."):
        solve_rows(SEARCHED, ATTEMPTS, RESULTS, SEED, "pretraining")


def _deep_row(problem_id, lines=2, first_success=9, **more):
    return {"id": f"{problem_id}#l5_deep_search#statement#0", "problem_id": problem_id, "side": "statement", "attempt_id": f"{problem_id}#l5_deep_search#statement#0",
            "theorem": f"theorem {problem_id} : P := by\n", "completion": "  step\n" * lines, "verified_attempts": 1, "origin": DEEP, "k": 1, "n": 56, "lines": lines,
            "first_success": first_success, **more}


def _shallow_row(problem_id, lines=3, **more):
    return {**_deep_row(problem_id, lines, 1), "id": f"{problem_id}#l4_map_pre_r64#statement#0", "attempt_id": f"{problem_id}#l4_map_pre_r64#statement#0", "origin": SHALLOW, "n": 8, **more}


def test_d_is_the_deep_solves_under_the_cap_and_more_are_a_seeded_draw():
    rows = [_deep_row(f"b{index}") for index in range(8)]
    assert capped(rows, 8, SEED) == rows and capped(rows, 400, SEED) == rows and capped([], 400, SEED) == []                       # at most the cap: every one
    kept = capped(rows, 5, SEED)
    wanted = set(sorted((row["problem_id"] for row in rows), key=lambda key: rank(SEED, CAP_LABEL, key))[:5])
    assert len(kept) == 5 and {row["problem_id"] for row in kept} == wanted != {f"b{index}" for index in range(5)}                 # the five of the smallest hash, not the first five
    assert kept == [row for row in rows if row["problem_id"] in wanted]                                                             # none moved among the others
    assert {row["problem_id"] for row in capped(rows, 5, 1)} != wanted and len(capped(rows, 7, SEED)) == 7 and len(capped(rows, 1, SEED)) == 1


def test_an_ordinary_solve_is_a_map_problem_solved_in_one_to_three_of_the_maps_attempts():
    of_the_rule = [row["problem_id"] for row in MAP_ROWS if 1 <= row["resolved"] <= 3]
    assert of_the_rule == [f"b{index}" for index in range(8, 14)]
    in_order = lambda ids: sorted(ids, key=lambda key: rank(SEED, SHALLOW_LABEL, key))      # noqa: E731
    drawn = ordinary_draw(MAP_ROWS, 1, 3, 4, SEED)
    assert [row["problem_id"] for row in drawn] == in_order(of_the_rule)[:4] and all(row["taken_by"] == THE_RULE and 1 <= row["map_k"] <= 3 and row["map_n"] == 8 for row in drawn)
    assert [row["problem_id"] for row in ordinary_draw(MAP_ROWS, 1, 3, 6, SEED)] == in_order(of_the_rule)                          # every one, in the hash order
    assert {row["problem_id"] for row in ordinary_draw(MAP_ROWS, 2, 2, 2, SEED)} == {"b10", "b11"}                                  # the bounds are the settings'
    assert in_order(of_the_rule) != sorted(of_the_rule, key=lambda key: rank(SEED, SEARCH_LABEL, key))                              # a draw of its own, not the search's order
    # Neither a problem failed every time (k = 0) nor one solved four times is an ordinary solve; fewer than asked is a refusal.
    with pytest.raises(ValueError, match="the map holds 6 problems its model solved in 1 to 3 of its attempts and `shallow` takes 7 of them .as many as `deep` has deep rows.: refused"):
        ordinary_draw(MAP_ROWS, 1, 3, 7, SEED)
    with pytest.raises(ValueError, match="the map holds 5 problems its model solved in 1 to 3 of its attempts and `shallow` takes 6"):
        ordinary_draw(MAP_ROWS, 1, 3, 6, SEED, not_among={"b8"})                                                                     # a problem the search attempted is not drawn
    # A SMOKE RUN ALONE: the map's other SOLVED problems join in the same hash order, each marked as NOT the rule's; a problem never solved has no proof and never joins.
    loosely = ordinary_draw(MAP_ROWS, 1, 3, 9, SEED, loosely=True)
    others = in_order(row["problem_id"] for row in MAP_ROWS if row["resolved"] > 3)
    assert [row["problem_id"] for row in loosely] == [*in_order(of_the_rule), *others[:3]] and [row["taken_by"] for row in loosely] == [THE_RULE] * 6 + [THE_MINIMUM] * 3
    assert ordinary_draw(MAP_ROWS, 1, 3, 4, SEED, loosely=True) == drawn                                                             # the rule gives enough: nothing is taken loosely
    with pytest.raises(ValueError, match="`shallow` takes 17 of them"):
        ordinary_draw(MAP_ROWS, 1, 3, 17, SEED, loosely=True)                                                                         # sixteen are solved at all


# ------------------------------------------------------------------------------------------------- the sets
DEEP_ROWS = [_deep_row(f"b{index}", first_success=9 + 5 * index) for index in range(4)]
SHALLOW_ROWS = [_shallow_row(f"b{index}") for index in range(8, 12)]
placed = lambda row: (row["id"], row["problem_id"], row["row_of_the_file"])      # noqa: E731
of_the_file = lambda rows: [row for row in rows if row["origin"] == "pretraining"]      # noqa: E731


def test_the_three_sets_hold_2d_rows_each_and_deep_and_shallow_the_same_d_published_rows_the_first_of_the_one_order():
    sets = the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED)
    order = rehearsal_rows(PRETRAINING, 40, SEED)                                    # the hash order L4t's rehearsal draw uses: the smallest hash first
    assert list(sets) == list(MODELS) and [len(sets[model]) for model in MODELS] == [8, 8, 8]                                       # 2D rows each: the same optimizer steps
    assert {model: sorted({row["origin"] for row in sets[model]}) for model in MODELS} == {PUBLISHED: ["pretraining"], SHALLOW: ["pretraining", "shallow"], DEEP: ["deep", "pretraining"]}
    # `deep`: the D deep rows and the FIRST D rows of the order; `shallow`: D ordinary solves and THE SAME D rows, by id and by place; `published`: the first 2D.
    assert [row for row in sets[DEEP] if row["origin"] == DEEP] == training_order(DEEP_ROWS, SEED) and [row for row in sets[SHALLOW] if row["origin"] == SHALLOW] == training_order(SHALLOW_ROWS, SEED)
    assert list(map(placed, _draw_order(of_the_file(sets[DEEP])))) == list(map(placed, order[:4])) == list(map(placed, _draw_order(of_the_file(sets[SHALLOW]))))
    assert list(map(placed, of_the_file(sets[DEEP]))) == list(map(placed, of_the_file(sets[SHALLOW])))                              # ... and in the same order
    assert list(map(placed, _draw_order(sets[PUBLISHED]))) == list(map(placed, order[:8])) and list(map(placed, _draw_order(sets[PUBLISHED])[:4])) == list(map(placed, order[:4]))
    assert {row["id"] for row in of_the_file(sets[DEEP])} < {row["id"] for row in sets[PUBLISHED]} and {row["id"] for row in of_the_file(sets[DEEP])} != {row["id"] for row in order[4:8]}
    # L4p's CONTROL IS THE SAME PREFIX: `matched_1262`, the first rows of the order as many as a set has published rows, is these D rows when that set has D.
    of_l4p = dose_sets(REHEARSE_ONLY, REWARD_REHEARSE, PRETRAINING, SEED)[MATCHED]
    assert len(of_l4p) == 4 and list(map(placed, of_l4p)) == list(map(placed, of_the_file(sets[DEEP])))
    # Each set stands in the arm's content-hash order: a set with rows left out is trained in the same order with those rows left out.
    for model in MODELS:
        assert [row["id"] for row in sets[model]] == [row["id"] for row in training_order(sets[model], SEED)]
    assert [row["id"] for row in of_the_file(sets[DEEP])] == [row["id"] for row in sets[PUBLISHED] if row["id"] in {given["id"] for given in of_the_file(sets[DEEP])}]
    # A published row is named by its id and its place, NEVER with its text; a row of the model's own carries its own proof.
    assert all(set(row) == {"problem_id", "kind", "proof_lines", "lines", "id", "side", "origin", "k", "row_of_the_file"} and row["k"] is None for model in MODELS for row in of_the_file(sets[model]))
    assert not any(field in row for model in MODELS for row in of_the_file(sets[model]) for field in TEXT_FIELDS) and all(refuse_text(sets[model], model) is None for model in MODELS)
    assert all(row["completion"] and row["theorem"] for model in OWN for row in sets[model] if row["origin"] == model)
    assert hold_the_sets(sets, PRETRAINING, SEED) == 4
    # D is the deep rows': one deep row gives three sets of two rows; the two kinds must be as many, and at least one.
    assert [len(own) for own in the_sets(DEEP_ROWS[:1], SHALLOW_ROWS[:1], PRETRAINING, SEED).values()] == [2, 2, 2]
    for deep, shallow, said in ((DEEP_ROWS, SHALLOW_ROWS[:3], "`deep` has 4 deep rows and `shallow` 3 ordinary solves"), ([], [], "`deep` has 0 deep rows and `shallow` 0 ordinary solves")):
        with pytest.raises(ValueError, match=f"{said}: the two hold the same number, at least 1. Refused"):
            the_sets(deep, shallow, PRETRAINING, SEED)
    with pytest.raises(ValueError, match="`pre` was trained on 40 rows and 42 are asked for the rehearsal"):                         # 2D rows of the file must be there
        the_sets([_deep_row(f"b{index}") for index in range(21)], [_shallow_row(f"c{index}") for index in range(21)], PRETRAINING, SEED)


def test_the_three_sets_are_held_from_the_outside_and_each_departure_is_refused():
    sets = the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED)
    order = rehearsal_rows(PRETRAINING, 40, SEED)
    changed = lambda model, rows: {**sets, model: training_order(rows, SEED)}      # noqa: E731
    without = lambda model, row_id: [row for row in sets[model] if row["id"] != row_id]      # noqa: E731
    first, ninth = of_the_file(sets[DEEP])[0], order[8]
    # ---- the sizes: 2D each, D deep rows, D ordinary solves, and nothing else
    for broken in (changed(DEEP, sets[DEEP][1:]), changed(PUBLISHED, sets[PUBLISHED][1:]), changed(SHALLOW, [*sets[SHALLOW], ninth]), changed(PUBLISHED, [*sets[PUBLISHED], DEEP_ROWS[0]]),
                   {**sets, DEEP: sets[PUBLISHED]}):
        with pytest.raises(ValueError, match="the three sets hold .rows of the model's own, published rows. .*`published` 2D published rows, D at least 1. Refused"):
            hold_the_sets(broken, PRETRAINING, SEED)
    # ---- a row of another origin than the set's own, an id twice, another order than the content hash's
    with pytest.raises(ValueError, match="`shallow` holds 1 rows of another origin than its own and 0 ids twice"):
        hold_the_sets(changed(SHALLOW, [*without(SHALLOW, SHALLOW_ROWS[0]["id"]), {**SHALLOW_ROWS[0], "origin": DEEP}]), PRETRAINING, SEED)
    with pytest.raises(ValueError, match="`deep` holds 0 rows of another origin than its own and 1 ids twice"):
        hold_the_sets(changed(DEEP, [*without(DEEP, DEEP_ROWS[0]["id"]), {**DEEP_ROWS[1]}]), PRETRAINING, SEED)
    with pytest.raises(ValueError, match="`published` does not stand in the content-hash order of this seed"):
        hold_the_sets({**sets, PUBLISHED: list(reversed(sets[PUBLISHED]))}, PRETRAINING, SEED)
    with pytest.raises(ValueError, match="`published` does not stand in the content-hash order of this seed"):
        hold_the_sets(sets, PRETRAINING, 1)                                                                                         # another seed's order is another order
    # ---- the published rows: the first D of the order for `deep` and `shallow`, the first 2D for `published`; by id AND by place in the file
    for model in (DEEP, SHALLOW):
        with pytest.raises(ValueError, match=f"the 4 published rows of `{model}` are not the first 4 of the hash order among the 40 rows the start model was pretrained on"):
            hold_the_sets(changed(model, [*without(model, first["id"]), ninth]), PRETRAINING, SEED)
        with pytest.raises(ValueError, match=f"the 4 published rows of `{model}` are not the first 4 of the hash order"):
            hold_the_sets(changed(model, [*without(model, first["id"]), {**first, "row_of_the_file": 39 - first["row_of_the_file"]}]), PRETRAINING, SEED)
    with pytest.raises(ValueError, match="the 8 published rows of `published` are not the first 8 of the hash order"):
        hold_the_sets(changed(PUBLISHED, [*without(PUBLISHED, first["id"]), order[9]]), PRETRAINING, SEED)
    with pytest.raises(ValueError, match="are not the first 4 of the hash order among the 39 rows"):
        hold_the_sets(sets, [row for row in PRETRAINING if row["problem_id"] != first["problem_id"]], SEED)                         # another file is another draw
    # ---- a deep solve is no ordinary solve; and neither is a problem whose published proof a model has seen
    with pytest.raises(ValueError, match="1 problems are in both `deep` and `shallow`, and 0 problems of the model's own rows are rows of the pretraining file .first: b0."):
        the_sets(DEEP_ROWS, [*SHALLOW_ROWS[:3], _shallow_row("b0")], PRETRAINING, SEED)
    with pytest.raises(ValueError, match="0 problems are in both `deep` and `shallow`, and 1 problems of the model's own rows are rows of the pretraining file .first: q7."):
        the_sets([*DEEP_ROWS[:3], _deep_row("q7")], SHALLOW_ROWS, PRETRAINING, SEED)


def test_deep_and_shallow_hold_their_published_rows_in_one_order_were_the_training_order_no_longer_held(monkeypatch):
    sets = the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED)
    monkeypatch.setattr(l5, "training_order", lambda rows, seed: list(rows))            # an order that holds nothing: the check of each set's order passes whatever it is
    turned = {**sets, SHALLOW: [*(row for row in sets[SHALLOW] if row["origin"] == SHALLOW), *reversed(of_the_file(sets[SHALLOW]))]}
    with pytest.raises(ValueError, match="`deep` and `shallow` do not hold the same published rows in the same order. Refused"):
        hold_the_sets(turned, PRETRAINING, SEED)
    assert hold_the_sets(sets, PRETRAINING, SEED) == 4


def test_the_published_rows_are_held_to_the_one_order_were_the_draw_no_longer_nested(monkeypatch):
    drawn = l4_rows.rehearsal_rows
    # A draw whose smaller count is no longer the first rows of its larger one: the sets are refused, not built on rows that differ between the models.
    monkeypatch.setattr(l5, "rehearsal_rows", lambda pretraining, count, seed: drawn(pretraining, count, seed) if count != 4 else drawn(pretraining, 40, seed)[1:5])
    with pytest.raises(ValueError, match="the 4 rows drawn for `deep` and `shallow` are not the first 4 of the hash order `published`'s 8 stand in. Refused"):
        the_sets(DEEP_ROWS, SHALLOW_ROWS, PRETRAINING, SEED)


def test_what_is_barred_is_a_held_out_problem_a_published_row_of_the_other_half_and_an_own_row_that_is_no_base_map_problem():
    pretrain_half = PRETRAIN_HALF[:8]
    other_half = next(f"p{index}" for index in range(400) if half_of(f"p{index}", 0) != PRETRAIN)
    published = [{"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4} for name in pretrain_half]
    base_map = [f"m{index}" for index in range(40)]
    own = [{"id": f"{name}#a", "problem_id": name, "origin": DEEP, "k": 1, "lines": 2} for name in base_map[:8]]
    rows = [*own, *published]
    found = refuse_barred(rows, {"g1", "g2"}, 0, base_map, "the training set of `deep`")
    assert found == {"rows": 16, "rows_of_the_models_own": 8, "published_rows": 8, "held_out": [], "published_rows_not_of_the_pretrain_half": [], "own_rows_not_of_the_base_map": []}
    # A BASE-MAP PROBLEM IS IN NEITHER HALF'S FILES: about half of them hash to the `pretrain` half, which L4t's own rule would bar for a round's row. Not here.
    assert {half_of(name, 0) for name in base_map[:8]} == {"pretrain", "loop"} and l4_rows.barred(rows, set(), 0)["of_the_pretrain_half_among_the_rounds_rows"]
    check = no_barred_row(rows, {"g1"}, 0, base_map)
    assert check["passes"] is True and (check["rows"], check["rows_of_the_models_own"], check["published_rows_of_the_pretrain_half"]) == (16, 8, 8) and check["first"] == []
    for changed, key, said in (([*rows, {**own[0], "id": "g1#a", "problem_id": "g1"}], "held_out_problems", "1 problems of the set are held-out problems .first: g1."),
                               ([*rows, {**published[0], "id": f"{other_half}#pretraining", "problem_id": other_half}], "published_rows_not_of_the_pretrain_half",
                                f"1 problems of the set are published rows that are not of the `pretrain` half .first: {other_half}."),
                               ([*rows, {**own[0], "id": "x9#a", "problem_id": "x9"}], "own_rows_not_of_the_base_map",
                                "1 problems of the set are rows of the model's own that are not base-map problems .first: x9.")):
        with pytest.raises(ValueError, match=said + ": refused, nothing was written"):
            refuse_barred(changed, {"g1"}, 0, [*base_map, "g1"], "the set")
        failed = no_barred_row(changed, {"g1"}, 0, [*base_map, "g1"])
        assert failed["passes"] is False and failed[key] == 1 and len(failed["first"]) == 1
    # A published row is never held to the base map, and an own row never to a half; a model trained on nothing does not pass.
    assert no_barred_row(published, set(), 0, [])["passes"] is True and no_barred_row([], set(), 0, base_map)["passes"] is False
    assert no_barred_row(own, set(), 0, base_map)["passes"] is True and no_barred_row(own, set(), 0, base_map[1:])["own_rows_not_of_the_base_map"] == 1


# ------------------------------------------------------------------------------------- the deep search's table
def test_the_deep_searchs_table_counts_the_solved_by_the_attempt_of_the_first_success():
    assert bands(8, 64) == [(9, 16), (17, 32), (33, 64)] and bands(8, 40) == [(9, 16), (17, 32), (33, 40)] and bands(8, 16) == [(9, 16)] and bands(8, 8) == []
    assert bands(4, 64) == [(5, 8), (9, 16), (17, 32), (33, 64)] and bands(8, 12) == [(9, 12)]
    drawn = [{"problem_id": f"b{index}", "map_k": 0, "map_n": 8, "taken_by": THE_RULE if index else THE_MINIMUM} for index in range(6)]
    results = [_result(f"b{index}", resolved, sides=1 + (index == 5), attempts_without_an_answer=index % 2) for index, resolved in enumerate((3, 0, 1, 1, 1, 0))]
    solves = [_deep_row("b0", first_success=9), _deep_row("b2", first_success=16), _deep_row("b3", first_success=17), _deep_row("b4", first_success=64)]
    table = search_table(drawn, results, solves, 8, 56)
    assert {key: value for key, value in table.items() if key != "what"} == {
        "problems": 6, "taken_by_the_rule": 5, "taken_by_the_runs_minimum": 1, "solved": 4, "share_solved": 0.66667, "solved_by_first_success": {"9_to_16": 2, "17_to_32": 1, "33_to_64": 1},
        "episodes_each": 56, "attempts_before_each": 8, "episodes": 336, "attempts": 392, "successes": 6, "per_1000": 17.86, "without_an_answer": 3}
    assert sum(table["solved_by_first_success"].values()) == table["solved"] and "64 with the map's 8" in table["what"]
    assert search_table([], [], [], 8, 56)["share_solved"] is None and search_table([], [], [], 8, 56)["per_1000"] is None
    # THE FIRST CHECK: at least the minimum; exactly the minimum is enough.
    assert [enough_deep_solves(found, 150)["passes"] for found in (0, 149, 150, 151, 400)] == [False, False, True, True, True]
    assert enough_deep_solves(149, 150) == {"what": enough_deep_solves(1, 150)["what"], "deep_solves": 149, "minimum": 150, "passes": False}
    assert "the search is then the finding" in enough_deep_solves(1, 150)["what"]


# --------------------------------------------------------------------------------------------- the branches
ABOVE_ZERO, THROUGH, BELOW_ZERO = _kept(0.00734, 0.00205, 0.01358), _kept(0.00538, -0.001, 0.01272), _kept(-0.01014, -0.01903, -0.00117)
EVEN, GAINED, LOST = _solved(30, 30), _solved(40, 20), _solved(22, 39)              # p = 1; gained more, p = 0.013; lost more, p = 0.04


def test_the_primary_has_three_outcomes_an_interval_that_ends_at_zero_holds_zero_and_the_breadth_stands_beside_every_one():
    assert sign_test(40, 20) < 0.05 > sign_test(22, 39) and sign_test(30, 30) == 1.0
    for difference, name in ((ABOVE_ZERO, DEEP_ADDS), (THROUGH, NOT_SEPARATED), (BELOW_ZERO, THEY_COST)):
        for breadth, narrower in ((EVEN, False), (GAINED, False), (LOST, True)):          # NARROWER beside the branch, WHATEVER the branch
            branch = primary_branch([], difference, breadth)
            assert branch["name"] == name and branch["narrower"] is narrower and branch["failed_checks"] == [] and branch["name"] in PRIMARY
            assert branch["narrower"] == breadth_beside(breadth, None, None, PUBLISHED)["narrower"] and branch["breadth"]["counts"] in branch["reason"]
            assert ("NARROWER against `published`" in branch["reason"]) is narrower and ("not narrower against `published`" in branch["reason"]) is not narrower
            assert (branch["breadth"]["gained"], branch["breadth"]["lost"], branch["breadth"]["sign_test_p"]) == (breadth["gained"], breadth["lost"], breadth["sign_test_p"])
    assert "on the held-aside problems `deep` minus `published` is above zero (+0.00734 [+0.00205, +0.01358])" in primary_branch([], ABOVE_ZERO, EVEN)["reason"]
    assert "held-aside problems solved at least once in their attempts, `deep` against `published`" in primary_branch([], ABOVE_ZERO, EVEN)["breadth"]["what"]
    assert "is below zero (-0.01014 [-0.01903, -0.00117]): in the place of the same number of published proofs the deep solves cost" in primary_branch([], BELOW_ZERO, EVEN)["reason"]
    held = primary_branch([], THROUGH, EVEN)
    assert held["resolves"] == 0.00686 and "holds zero (+0.00538 [-0.00100, +0.01272]): not separated at this size. This difference resolves 0.00686 per attempt" in held["reason"]
    assert "resolves" not in primary_branch([], ABOVE_ZERO, EVEN) and "resolves" not in primary_branch([], BELOW_ZERO, EVEN)
    # AN INTERVAL THAT ENDS AT ZERO HOLDS ZERO, at either end; one a hair off it does not.
    assert primary_branch([], _kept(0.005, 0.0, 0.011), EVEN)["name"] == NOT_SEPARATED == primary_branch([], _kept(-0.005, -0.011, 0.0), EVEN)["name"]
    assert primary_branch([], _kept(0.005, 0.00001, 0.011), EVEN)["name"] == DEEP_ADDS and primary_branch([], _kept(-0.005, -0.011, -0.00001), EVEN)["name"] == THEY_COST
    # NARROWER is LOST more than gained, under 0.05: the mirror split and a split at the level are not.
    assert sign_test(23, 39) > 0.05 > sign_test(23, 40)
    assert [primary_branch([], THROUGH, breadth)["narrower"] for breadth in (_solved(23, 39), _solved(23, 40), _solved(40, 23), _solved(0, 0))] == [False, True, False, False]
    # A failed check: INCONCLUSIVE, and nothing else is said. Nothing measured on both sides: NOT READ.
    failed = primary_branch(["`deep`: its_training_ran"], ABOVE_ZERO, LOST)
    assert failed["name"] == INCONCLUSIVE and failed["failed_checks"] == ["`deep`: its_training_ran"] and failed["narrower"] is None and "breadth" not in failed
    assert "nothing is said about what deep solves teach" in failed["reason"] and "+0.00734" not in failed["reason"]
    unread = primary_branch([], NOT_MEASURED, EVEN)
    assert unread["name"] == NOT_READ and unread["narrower"] is None and "the primary has nothing to be read on" in unread["reason"]
    assert primary_branch(["the_deep_search_found_enough"], NOT_MEASURED, {})["name"] == INCONCLUSIVE                               # the failed check comes first


def test_the_mechanism_is_depth_matters_when_the_interval_is_above_zero_or_more_were_gained_than_lost():
    for difference, breadth, name in ((ABOVE_ZERO, EVEN, DEPTH_MATTERS), (ABOVE_ZERO, LOST, DEPTH_MATTERS), (THROUGH, GAINED, DEPTH_MATTERS), (BELOW_ZERO, GAINED, DEPTH_MATTERS),
                                      (ABOVE_ZERO, GAINED, DEPTH_MATTERS), (THROUGH, EVEN, NOT_SHOWN), (THROUGH, LOST, NOT_SHOWN), (BELOW_ZERO, EVEN, NOT_SHOWN), (BELOW_ZERO, LOST, NOT_SHOWN)):
        read = mechanism([], difference, breadth)
        assert read["name"] == name and read["name"] in MECHANISM and read["above_zero_on_the_held_aside_set"] is (difference is ABOVE_ZERO) and read["broader"] is (breadth is GAINED)
        assert read["narrower"] is (breadth is LOST) and "`deep` minus `shallow`" in read["reason"] and "`deep` against `shallow`" in read["reason"]
    assert mechanism([], ABOVE_ZERO, EVEN)["reason"].endswith("the interval is above zero: a solve the model could not sample extends it where one it could does not")
    assert mechanism([], ABOVE_ZERO, EVEN)["reason"].startswith("on the held-aside problems `deep` minus `shallow` is +0.00734 [+0.00205, +0.01358]; held-aside problems solved, ")
    assert "more were gained than lost with the sign test under 0.05: a solve the model could not sample" in mechanism([], THROUGH, GAINED)["reason"]
    assert "the interval is above zero and more were gained than lost" in mechanism([], ABOVE_ZERO, GAINED)["reason"]
    assert mechanism([], THROUGH, LOST)["reason"].endswith("not shown; `deep` is NARROWER against `shallow`") and mechanism([], THROUGH, EVEN)["reason"].endswith(": not shown")
    # The edges: an interval that ENDS at zero is not above it; gained more than lost must be UNDER 0.05, and GAINED more (the mirror split is a narrowing, not a gain).
    assert mechanism([], _kept(0.005, 0.0, 0.011), EVEN)["name"] == NOT_SHOWN and mechanism([], _kept(0.005, 0.00001, 0.011), EVEN)["name"] == DEPTH_MATTERS
    assert mechanism([], THROUGH, _solved(39, 23))["name"] == NOT_SHOWN and mechanism([], THROUGH, _solved(40, 23))["name"] == DEPTH_MATTERS
    assert mechanism([], THROUGH, _solved(23, 40))["name"] == NOT_SHOWN and mechanism([], THROUGH, _solved(0, 0))["name"] == NOT_SHOWN
    assert mechanism(["`shallow`: lean_answered"], ABOVE_ZERO, GAINED)["name"] == INCONCLUSIVE and mechanism([], NOT_MEASURED, GAINED)["name"] == NOT_READ


def test_the_replication_is_named_plainly_above_holding_or_below_zero_and_narrower_or_not():
    for difference, name in ((ABOVE_ZERO, ABOVE), (THROUGH, HOLDS_ZERO), (BELOW_ZERO, BELOW), (_kept(0.005, 0.0, 0.011), HOLDS_ZERO), (_kept(-0.005, -0.011, 0.0), HOLDS_ZERO),
                             (_kept(0.005, 0.00001, 0.011), ABOVE), (_kept(-0.005, -0.011, -0.00001), BELOW)):
        for breadth, narrower in ((EVEN, False), (GAINED, False), (LOST, True)):
            read = replication([], difference, breadth)
            assert read["name"] == name and read["name"] in REPLICATION and read["narrower"] is narrower and ("resolves" in read) is (name == HOLDS_ZERO)
            assert f": {name}; " + ("NARROWER" if narrower else "not narrower") + " against `published`" in read["reason"] and "`shallow` minus `published`" in read["reason"]
    assert replication([], THROUGH, EVEN)["resolves"] == 0.00686
    assert replication(["x"], ABOVE_ZERO, LOST)["name"] == INCONCLUSIVE and replication(["x"], ABOVE_ZERO, LOST)["narrower"] is None and replication([], NOT_MEASURED, LOST)["name"] == NOT_READ


def test_what_follows_is_the_specs_branch_in_its_order():
    adds, held, cost = primary_branch([], ABOVE_ZERO, EVEN), primary_branch([], THROUGH, EVEN), primary_branch([], BELOW_ZERO, EVEN)
    adds_narrower, held_narrower, cost_narrower = primary_branch([], ABOVE_ZERO, LOST), primary_branch([], THROUGH, LOST), primary_branch([], BELOW_ZERO, LOST)
    matters, not_shown = mechanism([], ABOVE_ZERO, EVEN), mechanism([], THROUGH, EVEN)
    # DEEP SOLVES ADD (whatever the mechanism and the breadth say), or DEPTH MATTERS with the primary NOT BELOW ZERO: two more seeds.
    for primary, depth in ((adds, matters), (adds, not_shown), (adds_narrower, not_shown), (held, matters), (held_narrower, matters)):
        assert what_follows(primary, depth)["name"] == MORE_SEEDS == "TWO MORE SEEDS, THEN A LOOP WHOSE ROUNDS ARE DEEP SEARCHES"
    # THEY COST and NARROWER: needs a decision, with the three results together, EVEN WHEN DEPTH MATTERS (the primary is below zero).
    for depth in (matters, not_shown):
        follows = what_follows(cost_narrower, depth)
        assert follows["name"] == NEEDS_A_DECISION and "needs a decision, with the three results together" in follows["reason"]
    # OTHERWISE: not separated, and the depth is the thing to raise. THEY COST without NARROWER is this branch, with either mechanism.
    for primary, depth in ((held, not_shown), (held_narrower, not_shown), (cost, not_shown), (cost, matters)):
        follows = what_follows(primary, depth)
        assert follows["name"] == RAISE_THE_DEPTH and follows["reason"].endswith("not separated, and the depth is the thing to raise before the seeds")
    assert "the primary reads THEY COST (not narrower) and the mechanism" in what_follows(cost, matters)["reason"]
    # Nothing is named when the primary is INCONCLUSIVE or was not read.
    for primary in (primary_branch(["x"], ABOVE_ZERO, EVEN), primary_branch([], NOT_MEASURED, EVEN)):
        assert what_follows(primary, mechanism(["x"], ABOVE_ZERO, EVEN))["name"] == NOT_READ
    assert all(what_follows(primary, depth)["name"] in FOLLOWS for primary in (adds, held, cost, cost_narrower) for depth in (matters, not_shown))


# ------------------------------------------------------------------------------- the report, on hand-made rows
BASE = [f"m{index}" for index in range(60)]                                         # the base map's problems
ASIDE = [f"m{index}" for index in range(48, 60)]                                    # the held-aside problems: twelve base-map problems no row is of
TRAINED = {PUBLISHED: [{"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:40])],
           SHALLOW: [*({"id": f"{name}#map", "problem_id": name, "origin": SHALLOW, "k": 1 + index % 3, "lines": 3 + index % 9} for index, name in enumerate(BASE[20:40])),
                     *({"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:20]))],
           DEEP: [*({"id": f"{name}#search", "problem_id": name, "origin": DEEP, "k": 1 + index % 5, "lines": 5 + index % 9} for index, name in enumerate(BASE[:20])),
                  *({"id": f"{name}#pretraining", "problem_id": name, "origin": "pretraining", "k": None, "lines": 4 + index % 9} for index, name in enumerate(PRETRAIN_HALF[:20]))]}
TRAINS = {name: {"model": name, "rows": len(own), "steps": -(-len(own) // 8), "stand_in_engine": False, "trained_from": "the stored adapter pre_r64",
                 "against_the_start_adapter": {name: CHANGED}} for name, own in TRAINED.items()}
LOSSES = {name: FALLING(len(own)) for name, own in TRAINED.items()}
STATS = {"proofs": 20, "median": 5.0, "mean": 5.5, "share_with_8_lines_or_more": 0.25, "longest": 13, "share_with_4_lines_or_more": 0.75}
FOUND = {"problems": 100, "taken_by_the_rule": 100, "taken_by_the_runs_minimum": 0, "solved": 20, "share_solved": 0.2, "solved_by_first_success": {"9_to_16": 9, "17_to_32": 6, "33_to_64": 5},
         "episodes_each": 56, "attempts_before_each": 8, "episodes": 5600, "attempts": 5712, "successes": 31, "per_1000": 5.54, "without_an_answer": 2, "lines_of_the_deep_rows": STATS}
PREPARE = {"stage": "l4", "label": LABEL, "check": "L5", "seed": 0, "start": "pre_r64", "start_adapter": "runs/ladder_l4_pretrain_r64_seed0/adapters/pre_r64", "stand_in_engine": False,
           "l4t2_run": "runs/ladder_l4_rows2_pre_r64_seed0", "map_run": "runs/ladder_l4_map_pre_r64_seed0", "models": list(MODELS), "start_recipe": {"rank": 64, "alpha": 128},
           "enough_deep_solves": enough_deep_solves(20, 15), "deep_solves": 20, "rows_of_each_kind": 20, "the_search_found": FOUND, "trainings": {"run": True},
           "the_search": {"problems": 100, "candidates": 198, "episodes_each": 56, "attempts_with_the_maps": 64, "sampling_seed": 1050, "taken_by_the_runs_minimum": 0},
           "the_held_aside": {"problems": 12, "candidates_not_searched": 98, "episodes_each": 16, "sampling_seed": 1051, "taken_by_the_rule": 12, "taken_by_the_runs_minimum": 0},
           "lines_of_the_rows": {"what": "the lines", "deep_rows": STATS, "shallow_rows": STATS, "the_published_rows_beside_them": STATS, "published_rows": {**STATS, "proofs": 40}},
           "rows_not_taken_by_the_rule": {"what": "a smoke run alone", DEEP: 0, SHALLOW: 0},
           "training_sets": {PUBLISHED: {"rows": 40}, SHALLOW: {"rows": 40}, DEEP: {"rows": 40}},
           "goal_samplings": SAMPLINGS, "attempts_a_goal_problem": 93, "rung_episodes": 8, "sampling_seeds": {"rungs": 1002, "reach": 1001, "more": 1020},
           "contradicted_side_setting": "audit", "stored_models": {"pre_r64": {"run": "runs/ladder_l4_pretrain_r64_seed0"}}}
TOO_FEW = {**PREPARE, "enough_deep_solves": enough_deep_solves(9, 15), "deep_solves": 9, "rows_of_each_kind": 0, "training_sets": {}, "lines_of_the_rows": None,
           "rows_not_taken_by_the_rule": None, "the_search_found": {**FOUND, "solved": 9},
           "trainings": {"run": False, "why": "the deep search found 9 deep solves and 15 are asked: the three sets would be too small to read. No set is built and no model is "
                                              "trained or measured: the search is the finding"}}
AGAIN = goal_set_again(_made("pre", on_g=START)["goal"][0], GOAL_IDS)
LESS = ([6, 6, 6, 4, 4, 4, 2, 2, 0, 0, 0, 0], [9, 9, 9, 6, 6, 6, 5, 4, 1, 0, 0, 0])               # the start model with fewer successes on the last four: below it on G'
# Successes of 16 fresh attempts on each of the twelve held-aside problems: the same for every model unless said; one more and one fewer on EVERY problem; and none on eight of them.
HA_BASE, HA_MORE, HA_LESS, HA_NARROW = [2] * 12, [3] * 12, [1] * 12, [0] * 8 + [2] * 4


def _model(on_g=START, rungs=None, **changed):
    return {**_made("pre", on_g=on_g, rungs=rungs), "stand_in_engine": False, **changed}


def _aside(successes=HA_BASE, problems=ASIDE, **changed):
    """One model's per-problem rows on the held-aside set."""
    return [{"problem_id": key, "resolved": resolved, "episodes": 16, "sides": 1, "attempts_capped": 0, "attempts_timed_out": 0, "attempts_without_an_answer": 0, **changed}
            for key, resolved in zip(problems, successes)]


def _report(models=None, aside=None, prepare=PREPARE, trains=TRAINS, losses=LOSSES, trained=TRAINED, never=None, again=AGAIN, measured=MODELS, base_map=BASE):
    """The report with the start model's own rows standing in for every model that is not given, on G and on the
    held-aside set (`aside`: by model, its successes on each held-aside problem, or its rows)."""
    models = {name: (models or {}).get(name) or _model() for name in measured}
    given = lambda name: (aside or {}).get(name, HA_BASE)      # noqa: E731
    on_the_set = {name: given(name) if given(name) and isinstance(given(name)[0], dict) else _aside(given(name)) for name in (("pre_r64", *measured) if measured else ())}
    return build_l5_report(prepare, {name: trains[name] for name in measured}, {name: losses[name] for name in measured}, {name: trained[name] for name in measured}, GROUPS, LENGTHS,
                           {"pre_r64": _model()}, models, again, SETTINGS, EVALUATION, base_map, on_the_set, never)


def _on_g(entry):
    return [{**first, "resolved": first["resolved"] + more["resolved"], "episodes": first["episodes"] + more["episodes"]} for first, more in zip(*entry["goal"])]


def _on_ha(after, before):
    return per_attempt(_aside(after), _aside(before), ASIDE, EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"])


def test_with_the_start_models_rows_standing_in_every_difference_is_zero_and_nothing_is_separated():
    report = _report()
    assert report["branches"] == {"the_primary": NOT_SEPARATED, "the_mechanism": NOT_SHOWN, "the_replication": HOLDS_ZERO, "what_follows": RAISE_THE_DEPTH}
    assert (report["ok"], report["inconclusive"], report["failed_checks"], report["measured"], report["rank"], report["rows_of_each_kind"]) == (True, False, [], list(MODELS), 64, 20)
    assert list(report["can_this_run_be_read"]) == ["what", "the_deep_search", *MODELS, "the_held_aside_sets"] and report["can_this_run_be_read"]["the_deep_search"]["passes"] is True
    for model in MODELS:
        assert list(report["can_this_run_be_read"][model]) == ["its_training_ran", "it_still_writes_proofs", "lean_answered", "no_barred_row"]
        assert all(entry["passes"] for entry in report["can_this_run_be_read"][model].values())
    answered = report["can_this_run_be_read"]["the_held_aside_sets"]
    assert answered["passes"] is True and answered["share_without_an_answer"] == {held_aside_key(name): 0.0 for name in ("pre_r64", *MODELS)} and answered["sets_not_answered"] == []
    assert report["primary"]["difference"]["mean"] == 0.0 and report["primary"]["resolves"] == 0.0 and report["primary"]["narrower"] is False
    assert all(report[key]["difference"]["mean"] == 0.0 and report[key]["solved"]["gained"] == 0 and report[key]["solved"]["problems"] == 12 for key in ("primary", "the_mechanism", "the_replication"))
    assert all(report[key]["difference"]["attempts_each"] == 16 * 12 for key in ("primary", "the_mechanism", "the_replication"))    # THE HELD-ASIDE SET: 16 attempts on each of its 12 problems
    assert list(report["pairs"]) == ["what", *(pair_name(*pair) for pair in PAIRS)] and all(report["pairs"][pair_name(*pair)]["on_all_of_g"]["mean"] == 0.0 for pair in PAIRS)
    assert report["what_it_cannot_say"] == CANNOT_SAY and report["label"] == LABEL and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    # THE LINES, IN THE SPEC'S ORDER: the checks; the primary ON THE HELD-ASIDE SET and its branch, G' beside; the mechanism; the replication; what follows; the secondary
    # reads (each trained model against the start model on the held-aside set first); the deep search's table.
    marks = ("CHECK, the deep search found at least 15 deep solves: 20: PASS", "`published` (", "`shallow` (", "`deep` (", "CHECK, Lean answered on each held-aside set",
             "PRIMARY. THE HELD-ASIDE SET (12 problems", "BRANCH: NOT SEPARATED AT THIS SIZE. ", "PRIMARY, BESIDE IT, the held-out goal set's word, naming no branch: G' (4 goal problems",
             "THE MECHANISM (the claim under test", "THE MECHANISM: NOT SHOWN. ", "THE MECHANISM, BESIDE IT", "THE REPLICATION L4c owes", "THE REPLICATION: HOLDS ZERO. ",
             "THE REPLICATION, BESIDE IT", "WHAT FOLLOWS: NOT SEPARATED: THE DEPTH", "SECONDARY, each trained model against `pre_r64` ON THE HELD-ASIDE SET",
             "SECONDARY, each trained model against the stored `pre_r64` on ALL OF G (12 problems, 93 attempts each",
             "SECONDARY, `deep` minus `published`", "SECONDARY, `deep` minus `shallow`", "SECONDARY, `shallow` minus `published`", "`published`, SECONDARY", "`shallow`, SECONDARY",
             "`deep`, SECONDARY", "THE DEEP SEARCH: 100 problems attempted", "THE LINES OF THE ROWS (the proofs trained on): deep rows: 20 proofs")
    places = [next(index for index, line in enumerate(report["lines"]) if f"{SAY}: {mark}" in line) for mark in marks]
    assert places == sorted(places) and len(set(places)) == len(marks)
    assert "by the attempt of the first success, counting the map's: 9 to 16: 9; 17 to 32: 6; 33 to 64: 5; 5,712 attempts, 31 of 5,600 episodes resolved their problem (5.54 per 1,000)" in report["lines"][places[-2]]
    assert "each trained FROM `pre_r64` AT RANK 64, one pass, the arm's recipe, on 40 rows (D = 20: the same number of rows and of optimizer steps)" in report["lines"][0]
    assert "at 100 of the 198 problems of its own map that it failed in every map attempt" in report["lines"][0] and "sampling seed 1050" in report["lines"][0]
    assert ("THE HELD-ASIDE SET: 12 of the 98 candidates the search's draw did NOT take, a draw of its own; `pre_r64` and the three trained models each made 16 fresh attempts at "
            "every one, with ONE sampling seed (1051). THE PRIMARY is ON THE HELD-ASIDE SET, `deep` minus `published`") in report["lines"][0]
    assert report["headline"].startswith(f"L5 (do deep solves teach what ordinary solves do not; {LABEL}) seed 0, from `pre_r64`: NOT SEPARATED AT THIS SIZE. The deep search solved 20 of 100 problems; D = 20. "
                                         "Primary (the held-aside set, 12 problems, `deep` minus `published`): +0.00000")
    assert not any("A SMOKE RUN" in line or "ONE sampling of G" in line or "NOT TO BE READ" in line for line in report["lines"])
    # The fourth check is this stage's own: the deep and ordinary rows are base-map problems, and are not held to a half.
    barred = report["can_this_run_be_read"][DEEP]["no_barred_row"]
    assert (barred["rows_of_the_models_own"], barred["published_rows_of_the_pretrain_half"], barred["own_rows_not_of_the_base_map"]) == (20, 20, 0)
    assert any("`deep`, CHECK 4, no barred row (no held-out problem; its published rows of the `pretrain` half; the rows of the model's own are base-map problems): 0 held-out problems, 0 of "
               "its 20 published rows not of the `pretrain` half, 0 of its 20 own rows not base-map problems: PASS" in line for line in report["lines"])


def test_the_primary_is_deep_minus_published_on_the_held_aside_set_and_g_again_stands_beside_and_names_no_branch():
    # ON THE HELD-ASIDE SET `deep` is above `published` and `shallow` below it. ON G' IT IS THE OTHER WAY ROUND: the held-aside set decides, and G' is said beside.
    deep, published, shallow = _model(on_g=LESS), _model(on_g=MORE), _model(on_g=MORE)
    report = _report({DEEP: deep, PUBLISHED: published, SHALLOW: shallow}, {DEEP: HA_MORE, SHALLOW: HA_LESS})
    assert report["primary"]["difference"] == _on_ha(HA_MORE, HA_BASE) and report["the_mechanism"]["difference"] == _on_ha(HA_MORE, HA_LESS) and report["the_replication"]["difference"] == _on_ha(HA_LESS, HA_BASE)
    assert report["primary"]["difference"]["low"] > 0 > report["the_replication"]["difference"]["high"] and report["primary"]["difference"]["attempts_each"] == 16 * 12
    assert report["branches"] == {"the_primary": DEEP_ADDS, "the_mechanism": DEPTH_MATTERS, "the_replication": BELOW, "what_follows": MORE_SEEDS}
    beside = report["primary"]["beside_on_the_goal_set"]
    on_again = per_attempt(deep["goal"][1], published["goal"][1], AGAIN, EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"])
    assert beside["on_the_goal_set_again"] == on_again and on_again["high"] < 0 and on_again["attempts_each"] == 61 * len(AGAIN)       # G': the FRESH attempts alone; BELOW zero, and no branch moves
    assert beside["goal_problems_solved"] == solved(_on_g(deep), _on_g(published), GOAL_IDS) and "naming no branch" in beside["what"]
    for key, (after, before) in zip(("primary", "the_mechanism", "the_replication"), ((HA_MORE, HA_BASE), (HA_MORE, HA_LESS), (HA_LESS, HA_BASE))):
        assert report[key]["solved"] == solved(_aside(after), _aside(before), ASIDE)                                                # the breadth: the held-aside problems solved
    assert report["pairs"]["deep_minus_published"]["on_all_of_g"] == per_attempt(_on_g(deep), _on_g(published), GOAL_IDS, EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"])
    assert list(report["pairs"]["deep_minus_published"]["by_length_group"]) == ["1", "2-3", "4-7", "8+", "4_or_more"]
    text = "\\n".join(report["lines"])
    assert f"{SAY}: BRANCH: {DEEP_ADDS}. on the held-aside problems `deep` minus `published` is above zero" in text
    assert f"{SAY}: WHAT FOLLOWS: {MORE_SEEDS}. the primary reads {DEEP_ADDS} and the mechanism {DEPTH_MATTERS}" in text
    assert f"{SAY}: PRIMARY, BESIDE IT, the held-out goal set's word, naming no branch: G' (4 goal problems `pre_r64` does not solve in its first sampling), successes per attempt over the 61 fresh attempts, `deep` minus `published`: -" in text
    # DEPTH MATTERS IS DECIDED ON THE HELD-ASIDE SET: far above `shallow` on G' and level with it there is NOT SHOWN; level on G' and above it there is DEPTH MATTERS.
    on_g_alone = _report({DEEP: _model(on_g=MORE), SHALLOW: _model(on_g=LESS)})
    assert on_g_alone["branches"]["the_mechanism"] == NOT_SHOWN and on_g_alone["the_mechanism"]["beside_on_the_goal_set"]["on_the_goal_set_again"]["low"] > 0
    assert _report(aside={DEEP: HA_MORE})["branches"]["the_mechanism"] == DEPTH_MATTERS and _report(aside={DEEP: HA_MORE})["the_mechanism"]["beside_on_the_goal_set"]["on_the_goal_set_again"]["mean"] == 0.0
    assert on_g_alone["branches"]["the_primary"] == NOT_SEPARATED and on_g_alone["primary"]["beside_on_the_goal_set"]["on_the_goal_set_again"]["low"] > 0       # ... and so is the primary
    # ... and by the BREADTH of the held-aside set alone: `deep` solves eleven of its problems once, `shallow` the twelfth eleven times: the interval holds zero; gained 11, lost 1.
    broad = _report(aside={DEEP: [1] * 11 + [0], SHALLOW: [0] * 11 + [11]})
    assert broad["branches"]["the_mechanism"] == DEPTH_MATTERS and (broad["the_mechanism"]["above_zero_on_the_held_aside_set"], broad["the_mechanism"]["broader"]) == (False, True)
    assert (broad["the_mechanism"]["solved"]["gained"], broad["the_mechanism"]["solved"]["lost"]) == (11, 1) and broad["the_mechanism"]["difference"]["low"] < 0 < broad["the_mechanism"]["difference"]["high"]
    assert broad["the_mechanism"]["beside_on_the_goal_set"]["goal_problems_solved"]["gained"] == 0 and sign_test(11, 1) < SIGN_TEST_LEVEL
    # `deep` below `published` on the held-aside set and narrower than it there: THEY COST, NARROWER, and it needs a decision.
    cost = _report(aside={DEEP: HA_NARROW})
    assert cost["branches"] == {"the_primary": THEY_COST, "the_mechanism": NOT_SHOWN, "the_replication": HOLDS_ZERO, "what_follows": NEEDS_A_DECISION}
    assert cost["primary"]["narrower"] is True and (cost["primary"]["breadth"]["gained"], cost["primary"]["breadth"]["lost"]) == (0, 8) and cost["primary"]["breadth"]["sign_test_p"] < SIGN_TEST_LEVEL
    assert any(line.startswith(f"{SAY}: BRANCH: THEY COST, NARROWER. ") for line in cost["lines"]) and "seed 0, from `pre_r64`: THEY COST, NARROWER. " in cost["headline"]
    assert any(line.startswith(f"{SAY}: PRIMARY. THE HELD-ASIDE SET") and line.endswith(": NARROWER") for line in cost["lines"])
    # THE REPLICATION names NARROWER too: `shallow` solves four of the held-aside problems, `published` every one.
    narrow = _report(aside={SHALLOW: HA_NARROW})
    assert (narrow["branches"]["the_replication"], narrow["the_replication"]["narrower"]) == (BELOW, True) and any(line.startswith(f"{SAY}: THE REPLICATION: BELOW ZERO, NARROWER. ") for line in narrow["lines"])
    assert narrow["branches"]["the_primary"] == NOT_SEPARATED and narrow["branches"]["the_mechanism"] == DEPTH_MATTERS and narrow["branches"]["what_follows"] == MORE_SEEDS


def test_any_failed_check_makes_the_run_inconclusive_and_nothing_else_is_said():
    passing = _report(aside={DEEP: HA_MORE})
    assert passing["branches"]["the_primary"] == DEEP_ADDS
    held_out = [*TRAINED[SHALLOW][:-1], {**TRAINED[SHALLOW][-1], "id": f"{GOAL_IDS[0]}#pretraining", "problem_id": GOAL_IDS[0]}]
    not_of_the_map = [{**TRAINED[DEEP][0], "id": "x1#search", "problem_id": "x1"}, *TRAINED[DEEP][1:]]
    cases = (({"trains": {**TRAINS, PUBLISHED: {**TRAINS[PUBLISHED], "against_the_start_adapter": {PUBLISHED: {**CHANGED, "elements_changed": 0}}}}}, "`published`: its_training_ran"),
             ({"losses": {**LOSSES, DEEP: list(reversed(LOSSES[DEEP]))}}, "`deep`: its_training_ran"),
             ({"trained": {**TRAINED, SHALLOW: held_out}}, "`shallow`: no_barred_row"),
             ({"trained": {**TRAINED, DEEP: not_of_the_map}}, "`deep`: no_barred_row"))
    for changed, failed in cases:
        report = _report(aside={DEEP: HA_MORE}, **changed)
        assert report["failed_checks"] == [failed] and report["inconclusive"] is True and report["ok"] is True
        assert report["branches"] == {"the_primary": INCONCLUSIVE, "the_mechanism": INCONCLUSIVE, "the_replication": INCONCLUSIVE, "what_follows": NOT_READ}
        # NOTHING ELSE IS SAID: no figure of a read is printed or stored as read; what was measured is kept apart.
        for key in ("primary", "the_mechanism", "the_replication"):
            assert report[key]["difference"] == {"mean": None} and "solved" not in report[key] and "beside_on_the_goal_set" not in report[key]
            assert report[key]["measured_and_not_read"]["difference"]["mean"] is not None and "beside_on_the_goal_set" in report[key]["measured_and_not_read"]
        assert list(report["pairs"]) == ["what", "measured_and_not_read"] and all(entry["secondary"] is None and "measured_and_not_read" in entry for entry in report["models"].values())
        assert list(report["the_held_aside_set"]["against_the_start"]) == ["measured_and_not_read"] and list(report["against_the_start_on_all_of_g"]) == ["what", "measured_and_not_read"]
        text = "\\n".join(report["lines"][1:])                                         # (the first line says what WOULD be read)
        assert f"{SAY}: INCONCLUSIVE. the run could not have been read ({failed}): nothing is said about what deep solves teach" in text
        assert not any(mark in text for mark in ("PRIMARY", "BRANCH:", "THE MECHANISM", "THE REPLICATION", "WHAT FOLLOWS", "SECONDARY", "BESIDE IT")) and "THE DEEP SEARCH: 100 problems attempted" in text
        assert [line.split(": ", 1)[1].split(",")[0].split(".")[0].split(":")[0] for line in report["lines"][-3:]] == ["INCONCLUSIVE", "THE DEEP SEARCH", "THE LINES OF THE ROWS (the proofs trained on)"]
        assert f"from `pre_r64`: INCONCLUSIVE. The deep search solved 20 of 100 problems; D = 20. the run could not have been read ({failed})" in report["headline"]
    # A base-map problem of the `pretrain` half by its hash is NOT a barred row here (L4t's own fourth check would fail on it).
    assert {half_of(name, 0) for name in BASE[:20]} == {"pretrain", "loop"}
    # A set of G that Lean did not answer: its model's check fails, and the report is NOT TO BE READ (the step fails, and the task queued again measures that set again).
    silent = _model(on_g=MORE)
    silent["goal"][1] = [{**row, "attempts_without_an_answer": 3} for row in silent["goal"][1]]
    report = _report({DEEP: silent})
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_deep_2"] and report["failed_checks"] == ["`deep`: lean_answered"] and "NOT TO BE READ" in report["lines"][-1]
    # A HELD-ASIDE SET that Lean did not answer (more than 2% of its attempts), of a trained model or of the start model: the last check fails, and the report is not to be read.
    for name in (DEEP, "pre_r64"):
        report = _report(aside={name: _aside(HA_MORE, attempts_without_an_answer=1)})
        assert report["failed_checks"] == [THE_HELD_ASIDE] == ["lean_answered_on_the_held_aside_sets"] and report["ok"] is False and report["not_to_be_read"] == [held_aside_key(name)]
        answered = report["can_this_run_be_read"]["the_held_aside_sets"]
        assert answered["passes"] is False and answered["sets_not_answered"] == [held_aside_key(name)] and answered["share_without_an_answer"][held_aside_key(name)] == 0.0625
        assert report["branches"]["the_primary"] == INCONCLUSIVE and any(line.endswith(f"{held_aside_key(name)} 0.0625" + ("" if name == DEEP else "; held_aside_published 0.0; held_aside_shallow 0.0; held_aside_deep 0.0")
                                                                                       + ": FAIL") for line in report["lines"])
    # ... and at most 2% without an answer passes (one attempt of the set's 192).
    one = [{**row, "attempts_without_an_answer": int(index == 0)} for index, row in enumerate(_aside(HA_MORE))]
    assert _report(aside={DEEP: one})["failed_checks"] == [] and _report(aside={DEEP: one})["can_this_run_be_read"]["the_held_aside_sets"]["share_without_an_answer"]["held_aside_deep"] == 0.00521
    # All three models, or none: a report on two of them is refused; so are held-aside rows of other models, or of other problems for one model.
    with pytest.raises(ValueError, match="the report is given the trained models .'published', 'shallow'.: it reads all three"):
        _report(measured=(PUBLISHED, SHALLOW))
    rows = {name: _aside() for name in ("pre_r64", *MODELS)}
    for given in ({name: rows[name] for name in MODELS}, {**rows, DEEP: _aside(problems=[*ASIDE[:-1], "m0"])}, {**rows, DEEP: _aside()[:-1]}, {name: [] for name in rows}, {**rows, "another": _aside()}):
        with pytest.raises(ValueError, match="the report is given held-aside rows of .*: the start model `pre_r64` and the three trained models are measured there, each on the same problems"):
            build_l5_report(PREPARE, TRAINS, LOSSES, TRAINED, GROUPS, LENGTHS, {"pre_r64": _model()}, {name: _model() for name in MODELS}, AGAIN, SETTINGS, EVALUATION, BASE, given, None)


def test_too_few_deep_solves_nothing_was_trained_and_the_search_is_the_finding():
    report = _report(prepare=TOO_FEW, measured=())
    assert report["failed_checks"] == [THE_SEARCH] == ["the_deep_search_found_enough"] and report["inconclusive"] is True and report["measured"] == [] and report["ok"] is True
    assert report["branches"] == {"the_primary": INCONCLUSIVE, "the_mechanism": INCONCLUSIVE, "the_replication": INCONCLUSIVE, "what_follows": NOT_READ}
    assert report["can_this_run_be_read"]["the_deep_search"] == {**enough_deep_solves(9, 15)} and list(report["can_this_run_be_read"]) == ["what", "the_deep_search"]
    assert report["models"] == {} and report["pairs"] == {"what": report["pairs"]["what"], "measured_and_not_read": {}} and report["trainings"]["run"] is False
    assert report["the_deep_search"]["the_finding"] is True and report["the_deep_search"]["solved"] == 9 and _report()["the_deep_search"]["the_finding"] is False
    assert "against_the_start" not in report["the_held_aside_set"] and report["the_held_aside_set"]["problems_measured"] == 0 and report["primary"]["difference"] == {"mean": None}
    lines = report["lines"]
    assert "NO MODEL WAS TRAINED: the search found too few deep solves for the three sets" in lines[0] and "THREE MODELS" not in lines[0]
    assert lines[1] == f"{SAY}: CHECK, the deep search found at least 15 deep solves: 9: FAIL"
    assert lines[2].startswith(f"{SAY}: INCONCLUSIVE. the run could not have been read (the_deep_search_found_enough)") and lines[2].endswith("No set is built and no model is trained or measured: the search is the finding")
    assert lines[3].startswith(f"{SAY}: THE DEEP SEARCH, WHICH IS THE FINDING: 100 problems attempted (56 more attempts each, after the map's 8), 9 solved")
    assert lines[4].startswith(f"{SAY}: THE LINES OF THE ROWS (the deep solves found; nothing was trained): deep rows: 20 proofs") and len(lines) == 5
    assert "INCONCLUSIVE. The deep search solved 9 of 100 problems. the run could not have been read" in report["headline"] and "D =" not in report["headline"]
    # Trained models with a search that found too few is not a report this stage can make; nor is a held-aside row when nothing was trained.
    with pytest.raises(ValueError, match="or none when the deep search found too few deep solves and nothing was trained"):
        _report(prepare=TOO_FEW)
    with pytest.raises(ValueError, match="or no model when nothing was trained"):
        build_l5_report(TOO_FEW, {}, {}, {}, GROUPS, LENGTHS, {"pre_r64": _model()}, {}, AGAIN, SETTINGS, EVALUATION, BASE, {"pre_r64": _aside()}, None)


def test_a_run_with_one_sampling_of_g_still_reads_the_held_aside_set_and_a_smoke_runs_rows_are_said_not_to_be_the_rules():
    one = {**PREPARE, "goal_samplings": SAMPLINGS[:1], "attempts_a_goal_problem": 32, "rows_not_taken_by_the_rule": {"what": "a smoke run alone", DEEP: 2, SHALLOW: 1},
           "the_search": {**PREPARE["the_search"], "taken_by_the_runs_minimum": 2}, "the_held_aside": {**PREPARE["the_held_aside"], "taken_by_the_runs_minimum": 12}}
    models = {name: {**_model(), "goal": _model()["goal"][:1]} for name in MODELS}
    on_the_set = {"pre_r64": _aside(), PUBLISHED: _aside(), SHALLOW: _aside(), DEEP: _aside(HA_MORE)}
    report = build_l5_report(one, TRAINS, LOSSES, TRAINED, GROUPS, LENGTHS, {"pre_r64": {**_model(), "goal": _model()["goal"][:1]}}, models, AGAIN, SETTINGS, EVALUATION, BASE, on_the_set, None)
    # THE PRIMARY IS READ: it is on the held-aside set, which every run measures. What stands beside it on G' has no fresh attempts and is not measured.
    assert report["branches"] == {"the_primary": DEEP_ADDS, "the_mechanism": DEPTH_MATTERS, "the_replication": HOLDS_ZERO, "what_follows": MORE_SEEDS} and report["inconclusive"] is False
    assert report["primary"]["difference"] == _on_ha(HA_MORE, HA_BASE) and report["primary"]["beside_on_the_goal_set"]["on_the_goal_set_again"]["mean"] is None
    assert report["primary"]["beside_on_the_goal_set"]["goal_problems_solved"]["problems"] == 12
    text = "\\n".join(report["lines"])
    assert "the models have ONE sampling of G here (a smoke run): G was attempted 32 times a problem, and G' has no fresh attempts" in text
    assert "successes per attempt over fresh attempts, which this run does not have, `deep` minus `published`: not measured" in text
    assert (f"{SAY}: A SMOKE RUN: rows of problems the rule did not give were taken, so that every step has rows to run on: 2 of `deep`'s own rows, 1 of `shallow`'s own rows; 2 of the "
            "search's problems; 12 of the held-aside problems. Nothing here is a read of the rule") in text


def test_the_secondary_reads_are_each_model_against_the_start_model_on_the_held_aside_set_and_l4ts_by_itself():
    never = [GOAL_IDS[-1], GOAL_IDS[-2], "not_a_goal_problem"]
    report = _report({DEEP: _model(on_g=MORE)}, {DEEP: HA_MORE, SHALLOW: HA_NARROW}, never=never)
    # EACH TRAINED MODEL AGAINST THE START MODEL ON THE HELD-ASIDE SET: successes per attempt and the problems solved; said to be favoured by the selection.
    against = report["the_held_aside_set"]["against_the_start"]
    assert list(against) == list(MODELS) and report["the_held_aside_set"]["note"] == FAVOURED and report["the_held_aside_set"]["problems_measured"] == 12
    for name, own in ((PUBLISHED, HA_BASE), (SHALLOW, HA_NARROW), (DEEP, HA_MORE)):
        assert against[name] == {"per_attempt": _on_ha(own, HA_BASE), "solved": solved(_aside(own), _aside(HA_BASE), ASIDE)}
    line = next(line for line in report["lines"] if "SECONDARY, each trained model against `pre_r64` ON THE HELD-ASIDE SET (12 problems, 16 attempts each" in line)
    assert "`shallow`: " in line and "solved 4 to 12, gained 0, lost 8" in line and line.endswith(f"NOTE: {FAVOURED}")
    # ... AND ON ALL OF G, against its STORED rows: successes per attempt over all the attempts, and the goal problems solved.
    on_all_of_g = report["against_the_start_on_all_of_g"]
    assert list(on_all_of_g) == ["what", *MODELS] and on_all_of_g[PUBLISHED]["per_attempt"]["mean"] == 0.0 and on_all_of_g[PUBLISHED]["solved"]["gained"] == 0
    assert on_all_of_g[DEEP] == {"per_attempt": per_attempt(_on_g(_model(on_g=MORE)), _on_g(_model()), GOAL_IDS, EVALUATION["bootstrap_resamples"], EVALUATION["bootstrap_seed"]),
                                 "solved": solved(_on_g(_model(on_g=MORE)), _on_g(_model()), GOAL_IDS)}
    assert on_all_of_g[DEEP]["per_attempt"]["low"] > 0 and on_all_of_g[DEEP]["per_attempt"]["attempts_each"] == 93 * 12
    # Against THE START MODEL's stored rows and no trained model's (`published` below the start here); gained is solved by the trained model and not by the start model.
    lower = _report({DEEP: _model(on_g=MORE), PUBLISHED: _model(on_g=LESS)})["against_the_start_on_all_of_g"]
    assert lower[DEEP] == on_all_of_g[DEEP] and lower[PUBLISHED]["per_attempt"]["mean"] < 0
    assert [lower[PUBLISHED]["solved"][key] for key in ("resolved_after", "resolved_before", "gained", "lost")] == [9, 12, 0, 3]
    line = next(line for line in _report({PUBLISHED: _model(on_g=LESS)})["lines"] if "SECONDARY, each trained model against the stored `pre_r64` on ALL OF G" in line)
    assert "`published`: " in line and "solved 9 to 12, gained 0, lost 3" in line
    for model in MODELS:
        secondary = report["models"][model]["secondary"]
        assert list(secondary) == ["the_goal_set_again", "by_length_group", "goal_problems_solved_by_attempts_alone", "the_three_rungs", "distinct_attempts", "verified_proof_lines",
                                   "never_solved_before", "trained_on"]
        others = [other for other in MODELS if other != model]
        assert list(secondary["distinct_attempts"])[1:] == ["pre_r64", *others, model] and secondary["never_solved_before"]["given"] is True
        assert "minus_pre_r64" in secondary["the_three_rungs"] and secondary["trained_on"]["rows"] == 40
        assert all(f"solved_by_{other}" in secondary["never_solved_before"] for other in others)
    assert report["models"][DEEP]["secondary"]["trained_on"]["rows_by_origin"] == {"deep": 20, "pretraining": 20}
    assert report["models"][PUBLISHED]["secondary"]["trained_on"]["rows_by_origin"] == {"pretraining": 40}
    assert report["models"][DEEP]["secondary"]["the_goal_set_again"]["low"] > 0 and report["models"][PUBLISHED]["secondary"]["the_goal_set_again"]["mean"] == 0.0
    assert _report()["models"][DEEP]["secondary"]["never_solved_before"]["given"] is False


def test_no_rule_is_written_a_second_time():
    from rlvr_lean.gpu import ladder_l4_rows, ladder_l5
    from rlvr_lean.reporting import ladder_l4_rows as l4t_report
    from rlvr_lean.reporting import ladder_l5 as report

    assert l5.rehearsal_rows is l4_rows.rehearsal_rows and l5.training_order is training_order and l5.training_examples is training_examples      # the draw, the order, the attempt a round draws
    assert l5.breadth_beside is breadth_beside and l5.barred is l4_rows.barred and l5.THE_RULE is THE_RULE and l5.THEY_COST is THEY_COST and l5.NOT_SEPARATED is NOT_SEPARATED
    assert (report.world_of, report.the_four_checks, report.check_lines, report.secondary_reads, report.secondary_lines) == (
        l4t_report.world_of, l4t_report.the_four_checks, l4t_report.check_lines, l4t_report.secondary_reads, l4t_report.secondary_lines)
    assert report.per_attempt is per_attempt and report.solved is solved
    assert (ladder_l5.train_a_model, ladder_l5.measure_a_model, ladder_l5.read_the_trained) == (ladder_l4_rows.train_a_model, ladder_l4_rows.measure_a_model, ladder_l4_rows.read_the_trained)
    assert ladder_l5.refuse_text is refuse_text and ladder_l5.THE_STAGE.prepare == "ladder_l5_prepare" and isinstance(ladder_l5.THE_STAGE, ladder_l4_rows.Stage)
