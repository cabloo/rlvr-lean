"""L6b, the stage end to end on what L5's FINISHED search, L4p's run and the FINISHED RUNS OF L5d, L5f AND L5c OF THE
SEED stored: the refusals of its first step (the seed, L5's run, `second_full`, B in L5f's run, the bar in L5c's run, the
old own rows in L5d's, the next published rows); THE SEARCH served from the averaged adapter another run keeps, on L6's
search-2 problems made by L6's own functions; the two sets (K new solves or K old own rows beside THE SAME published
rows, never with text); eight trainings FROM B, two averaged adapters, two measurements of G's first sampling alone;
the report's checks failing when they should; the reads at the branches a one-problem world reaches; too few new
solves; the smoke stage IN THE BOX'S OWN SMOKE WORLD; the registration; and every run it reads left file for
file as it was. Spec: docs/spec/ladder-loop.spec.md, "L6b: a second round, from the best model". The engine and
Lean are scripted; nothing touches a GPU or the network. The rules on hand-made rows are `test_ladder_l6b.py`."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adapter_files import _numbers  # noqa: E402
from test_ladder_l2_stage import ARM  # noqa: E402
from test_ladder_l4_dose_stage import _stage_p  # noqa: E402 - L4p's own stage, as its tests run it
from test_ladder_l4_rows_r64 import _trainings as _trainings_of_the_chain  # noqa: E402 - the smoke runs before L5d's: a training records no learning rate
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l4_stage import FIXTURE, _on_the_gpu  # noqa: E402
from test_ladder_l5_cont_stage import OF_L5F, SUCCESSES, _rewritten, _stage5c  # noqa: E402 - L5c's own stage: its `again` is the bar
from test_ladder_l5_cont_stage import OWN_FILES as L5CS_FILES  # noqa: E402 - what a seed of L5c leaves, frozen there
from test_ladder_l5_cont_stage import cont  # noqa: E402, F401 - L5's, L4p's, L5d's and L5f's runs of the seed, each run to its end by its own stage
from test_ladder_l5_draws_stage import _l5ns_report, _stage5d  # noqa: E402
from test_ladder_l5_full_stage import OWN_FILES as L5FS_FILES  # noqa: E402 - what a seed of L5f leaves, frozen there
from test_ladder_l5_full_stage import _as_text, _changed, _file_rows, _stage5f, _swapped  # noqa: E402
from test_ladder_l5_measure_stage import _a_smoke_task  # noqa: E402 - one smoke task as the box runs it
from test_ladder_l5_noise_stage import GOAL, NoiseSolver, _adapter_of, _trainings  # noqa: E402
from test_ladder_l5_stage import (  # noqa: E402, F401
    CONFIG,
    PACKAGE,
    SEEDS,
    DeepSolver,
    _stage5,
    _write_the_map,
    arm,
    deep,
    dose,
    l4,
    loop,
    r64,
    rows,
    rows2,
    stage,
)
from test_ladder_l6_stage import ATTEMPTS, NEITHER, SEARCHED_2, SOLVED_1, UNSOLVED_1, CycleSolver  # noqa: E402 - L6's search-2 world: the same four problems
from test_ladder_round import _run_stage  # noqa: E402 - L1's own stage, as its tests run it

from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_cont import OF_THEIRS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR  # noqa: E402
from rlvr_lean.domain.ladder_round.l6 import PART_A, PART_B  # noqa: E402
from rlvr_lean.domain.ladder_round.l6b import (  # noqa: E402
    ADD,
    COST,
    MODELS,
    NEW,
    NEW_OWN,
    NOT_SHOWN_AT_ONE,
    OLD_OWN,
    SAID_AS_IT_IS,
    TEACH_LESS,
    TEACH_MORE,
    THE_CYCLE_TURNS,
    THE_SEARCH_IS_THE_FINDING,
    TO_A_DECISION,
    TRAINED,
    training_order,
)
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import (  # noqa: E402
    ladder_ceiling,
    ladder_l2,
    ladder_l3d2,
    ladder_l4,
    ladder_l4_dose,
    ladder_l4_rank,
    ladder_l4_rows,
    ladder_l4_rows2,
    ladder_l4_start,
    ladder_l5,
    ladder_l5_cont,
    ladder_l5_draws,
    ladder_l5_full,
    ladder_l5_measure,
    ladder_l5_noise,
    ladder_l5_stock,
    ladder_l6,
    ladder_l6b,
    ladder_round,
    milestone2,
    pipeline,
)
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.adapter_files import is_the_mean  # noqa: E402

RUN, SMOKE = ladder_l6b.RUN_VARIABLE, ladder_l6b.SMOKE_VARIABLE
L4P_RUN, L5F_RUN, L5C_RUN, L6_RUN = ladder_l6.L4P_RUN_VARIABLE, ladder_l5_full.RUN_VARIABLE, ladder_l5_cont.RUN_VARIABLE, ladder_l6.RUN_VARIABLE
SEARCH_PREPARE, SEARCH, PREPARE, REPORT = "ladder_l6b_search_prepare", "ladder_l6b_search", "ladder_l6b_prepare", "ladder_l6b_report"
STEPS = [SEARCH_PREPARE, SEARCH, PREPARE,
         *(step for model in MODELS for step in (*(f"ladder_l6b_train_{model}_{order}" for order in (1, 2, 3, 4)), f"ladder_l6b_average_{model}", f"ladder_l6b_measure_{model}")),
         REPORT]
STORED = [f"l4_stored_{who}_reach.jsonl" for who in ("pre_r64", "second_full", "l5f_published", "l5c_again")]
SEARCH_FILES = ["l6b_search_problems.jsonl", "l6b_old_solves.jsonl", "l6b_published_next.jsonl", "l6b_new_solves.jsonl"]
OWN_FILES = sorted([
    *(f"adapters/{name}/{file}" for name in (*MODELS, *TRAINED) for file in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l6b_loss_{name}.json" for name in TRAINED), *(f"l6b_training_{name}.jsonl" for name in (*MODELS, *TRAINED)),
    *(f"{step}.done.json" for step in STEPS), *STORED, *SEARCH_FILES, "problems.jsonl", "report_ladder_l6b.json", "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl",
    "l5_held_aside_problems.jsonl"])
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "first_start", "second_start", "rank", "models", "recipe",
               "trainings", "branches", "inconclusive", "failed_checks", "can_this_run_be_read", "of_this_seed", "the_search", "figures", "the_bar", "the_primary", "whose_proofs",
               "beside", "timed_out_in_lean", "what_follows", "the_seeds", "the_start_model", "the_best", "l5cs_run", "l5ds_run", "round_one", "the_dose", "the_sets", "the_orders",
               "the_sampling", "training_sets", "training_recipe", "models_made", "smoke", "what_it_cannot_say", "attempts", "not_to_be_read", "not_built", "not_measured",
               "adapters"]
B_IS_L5FS, L5CS_RUN, THE_SEARCHED, THE_SETS, THE_BARS_SAMPLING, SEARCH_ANSWERED = (
    "b_is_l5fs_averaged_published_and_everything_here_is_from_it", "l5cs_run_of_the_seed_holds_the_bar", "the_searched_set_is_l6s_search_2", "each_set_is_its_rule_made_again",
    "the_stored_first_sampling_of_the_bar_is_the_settings", "lean_answered_the_search")
# The checks of the run, in the report's order; the last (the sets) is made when the trainings ran.
THE_RUN_CHECKS = [B_IS_L5FS, L5CS_RUN, THE_SEARCHED, "the_stored_first_sampling_of_the_start_model_is_the_settings", THE_BARS_SAMPLING, SEARCH_ANSWERED, THE_SETS]
# The smoke runs this stage's smoke task READS on the box, and everything the box's smoke world holds before it (the runs
# those were made from beside them; the real run of L5n that L5d's smoke stage reads for its recipe).
SMOKE_RUNS_NEEDED = ["ladder_l4_dose_r64_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows2_r64_smoke", "ladder_l4b_map_smoke", "ladder_l5_cont_smoke", "ladder_l5_deep_smoke",
                     "ladder_l5_draws_smoke_mean_constant", "ladder_l5_full_smoke"]
# What B writes in the search: by problem, the attempts (from 0, of 64) whose proof closes the goal, each with the lines of that proof. ONE problem is solved, one of (a).
SCRIPT = {UNSOLVED_1[0]: {20: 9, 40: 9}}
OF_THESE = (12, 8)                              # of the 32 attempts at the goal problem: `new_own`, then `old_own` (B's stored: 8; L5c's `again`, THE BAR: 8)
B, BAR = OF_L5F[0], SUCCESSES[2]
A_LAYER = "base_model.model.layers.0.q_proj.lora_A.weight"
NOTHING_READ = {"primary": INCONCLUSIVE, "whose_proofs": INCONCLUSIVE, "what_follows": NOT_READ}


def _stage6b(config, monkeypatch, until=None, stage_name="ladder_l6b", prepare_on_the_gpu=False, only=None):
    """The stage's steps in its order; the trainings and the averagings by their GPU path (the stand-in makes no adapter)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l6b.STEPS or (only is not None and step not in only):
            continue
        run = ladder_l6b.STEPS[step]
        on_the_gpu = "_train_" in step or "_average_" in step or (prepare_on_the_gpu and step == SEARCH_PREPARE)
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if on_the_gpu else run(config)
        if step == until:
            break
    return summaries


def _but_what(record):
    return {key: value for key, value in record.items() if key != "what"}


def _nothing_written(runs):
    return not [path for path in runs.iterdir() if path.name.startswith("ladder_l6b") and any(path.iterdir())]


def _engines(monkeypatch, solver, kits):
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"]}),
        (solver, stand_in_parameters))[1]))


@pytest.fixture
def second(cont, monkeypatch):  # noqa: F811
    """L5c's world (a base map of twelve problems; L5's finished search, which solved two of its four problems; L4p's run
    with `second_full`; L5d's and L5f's OWN STAGES run to their end at seed 3) WITH L5c's OWN STAGE RUN TO ITS END at
    seed 3 (its `again` closes the goal in 8 of 32: THE BAR), and this stage's task of SEED 3, which carries NO variable.
    The search's four problems are L6's search 2 of that world: B solves ONE, which search 1 left unsolved (`SCRIPT`), so
    N = 1, K = 1 and a set is one own row and three published rows (with K = 2 the four rows of `old_own` are the world's
    own and two of the four order seeds give them one order, which the stage rightly refuses outside a smoke run). One
    new solve is enough (L6's setting, made small). `calls()`: this stage's own trainings, the k-th of which saves an
    adapter that holds 24 + k (L5f's twelve hold 1 to 12, L5c's 13 to 24)."""
    for name in (RUN, SMOKE, L6_RUN, L5C_RUN):
        monkeypatch.delenv(name, raising=False)
    config = cont.config
    theirs = _stage5c(config, monkeypatch)
    l5c = cont.runs / "ladder_l5_cont_second_full_seed3"
    assert theirs["ladder_l5_cont_report"]["inconclusive"] is False and len(cont.made) == 24
    assert sorted(name for name in _all(l5c) if not name.startswith("episodes_")) == L5CS_FILES
    config["ladder_loop"]["l6"]["search"]["minimum"] = 1
    kits, searching = [], CycleSolver(cont.deep.solver, SCRIPT)
    solver = NoiseSolver(searching, OF_THESE)
    _engines(monkeypatch, solver, kits)
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l6b"])        # THE STAGE names no variable
    assert config["ladder_loop"]["l6b"] == {"start": "published", "bar": "again", "seeds": [3]}
    return SimpleNamespace(config=config, cont=cont, deep=cont.deep, kits=kits, solver=solver, searching=searching, runs=cont.runs, l5=cont.l5, l5f=cont.l5f, l5d=cont.l5d,
                           l5c=l5c, l4p=cont.l4p, best=cont.l5f / "adapters" / "published", calls=lambda: cont.made[24:], theirs=theirs,
                           store=lambda: ladder_l6b._store(config, ladder_l6b.the_settings(config)))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_searches_from_b_trains_new_own_and_old_own_from_it_averages_and_reads_each_against_l5cs_stored_again(second, monkeypatch):
    config, kits, solver, deep_ = second.config, second.kits, second.solver, second.deep
    read_only = {path: _all(path) for path in (second.l5, second.l5f, second.l5d, second.l5c, second.l4p, deep_.map, deep_.l4t2, deep_.rank)}
    summaries = _stage6b(config, monkeypatch)
    store, calls = second.store(), second.calls()
    # THE RUN DIRECTORY IS ITS OWN, beside the runs it reads, which were ONLY READ: file for file. L6's own run is not read, and none is made.
    assert store.root.name == "ladder_l6b_published_seed3" and store.root.parent == second.l5.parent and list(summaries) == STEPS == list(ladder_l6b.STEPS)
    assert {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert not [path.name for path in second.runs.iterdir() if path.name.startswith("ladder_l6_")]

    # ---- the first step: the seed, B in L5f's run, the bar in L5c's, the old own rows in L5d's, round 1 in L5's, the search's problems by L6's own functions
    read = summaries[SEARCH_PREPARE]
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"], read["first_start"], read["second_start"], read["models"]) == (
        "l4", "pretrained on published proofs", "L6b", 3, "published", "pre_r64", "second_full", list(MODELS))
    assert (read["start_adapter"], read["start_run"], read["second_start_adapter"]) == (str(second.best), str(second.l5f), str(second.l4p / "adapters" / "second_full"))
    assert read["smoke"] is None and (read["start_recipe"]["rank"], read["start_recipe"]["alpha"]) == (64, 128) and read["the_seeds"]["seeds"] == [3]
    assert read["the_start_model"] == second.theirs["ladder_l5_cont_prepare"]["the_start_model"]              # `second_full`, read as L5f and L5c read it
    of_b, of_the_bar, of_the_old = read["the_best"], read["l5cs_run"], read["l5ds_run"]
    assert (of_b["run"], of_b["adapter"], of_b["rows_of_its_set"]) == (str(second.l5f), str(second.best), 4) and "`ladder_l5_full` for seed 3" in of_b["task"]
    assert (of_b["held"]["passes"], of_b["held"]["recorded"], of_b["held"]["read_now"], list(of_b["held"]["trainings"])) == (True, True, None, list(OF_THEIRS))      # no GPU box here
    assert of_b["held"]["the_averaged"]["is_the_mean"] is True and of_b["at_the_first_step"]["inconclusive"] is False
    assert (of_the_bar["run"], of_the_bar["trained_from"], of_the_bar["the_next_rows"]["rows"], of_the_bar["read_though_inconclusive"]) == (str(second.l5c), str(second.best), 4, False)
    assert "`ladder_l5_cont` for seed 3" in of_the_bar["task"] and of_the_bar["at_the_first_step"]["seed"] == 3 and of_the_bar["at_the_first_step"]["start"] == "second_full"
    assert (of_the_old["run"], of_the_old["old_own_rows"], of_the_old["file"]) == (str(second.l5d), 2, "l6b_old_solves.jsonl")
    first = read["round_one"]
    assert (first["run"], first["seed"], first["attempted"], first["solved"], first["sampling_seed"], first["inconclusive"]) == (str(second.l5), 0, 4, 2, 1050, False)
    search = read["the_search"]
    assert (search["set"], search["file"], search["model"], search["adapter"], search["problems"], search["by_part"], search["episodes_each"], search["sampling_seed"],
            search["sampling_place"]) == ("l6b_search", "l6b_search_problems.jsonl", "published", str(second.best), 4, {PART_A: 2, PART_B: 2}, ATTEMPTS, 1052, 52)
    assert (search["taken_by_the_rule"], search["taken_by_the_runs_minimum"], search["problems_at_least"], search["seed_of_the_search"], search["minimum_new_solves"]) == (4, 0, 0, 0, 1)
    assert search["partition"] == {"candidates": 7, "attempted_by_search_1": 4, "solved_by_search_1": 2, "a_left_unsolved_by_search_1": 2, "b_in_neither_draw": 2, "held_aside": 1,
                                   "taken_by_the_runs_minimum": 0} and 1052 not in search["seeds_of_the_stored_samplings_it_is_none_of"].values()
    searched = store.read_rows("l6b_search_problems.jsonl")
    assert [row["problem_id"] for row in searched] == SEARCHED_2 and [row["part"] for row in searched] == [PART_A, PART_A, PART_B, PART_B]       # L6's search 2 of this world
    assert not {row["problem_id"] for row in searched} & ({*SOLVED_1} | {row["problem_id"] for row in store.read_rows("l5_held_aside_problems.jsonl")})
    sampling = read["the_sampling"]
    assert (sampling["name"], sampling["episodes"], sampling["sampling_seed"], sampling["sets"]) == ("reach", 32, 1001, {model: f"l4_reach_{model}" for model in MODELS})
    assert [sampling[key] for key in ("stored_first_start", "stored_second_start", "stored_best", "stored_bar")] == STORED
    assert read["recipe"]["name"] == "mean_constant" and read["recipe"]["adapter_seed"] == 3 and read["recipe"]["order_seeds"] == [1, 2, 3, 4]
    # THE FOUR STORED FIRST SAMPLINGS are copied: `pre_r64`'s, `second_full`'s from L4p's run, B's from L5f's run, and THE BAR's from L5c's run.
    first_rows, start_rows, best_rows, bar_rows = (store.read_rows(name) for name in STORED)
    assert [(row["problem_id"], row["episodes"]) for own in (first_rows, start_rows, best_rows, bar_rows) for row in own] == [(GOAL, 32)] * 4
    assert (best_rows[0]["resolved"], bar_rows[0]["resolved"]) == (B, BAR) and start_rows == _file_rows(second.l4p / "episodes_l4_reach_second_full_problems.jsonl")
    assert best_rows == _file_rows(second.l5f / "episodes_l4_reach_published_problems.jsonl") and bar_rows == _file_rows(second.l5c / "episodes_l4_reach_again_problems.jsonl")
    # THE OLD OWN ROWS are the own rows of the `deep` set L5d's run of the seed stores, in its order; THE NEXT ROWS are L5c's stored `again` set, by id and place, never with text.
    old = store.read_rows("l6b_old_solves.jsonl")
    assert old == [row for row in _file_rows(second.l5d / "l5_draws_training_deep.jsonl") if row["origin"] != "pretraining"] and [row["origin"] for row in old] == ["deep"] * 2
    assert sorted(row["problem_id"] for row in old) == sorted(SOLVED_1)
    next_rows, of_l5c = store.read_rows("l6b_published_next.jsonl"), _file_rows(second.l5c / "l5_cont_training_again.jsonl")
    placed = lambda row: (row["id"], row["problem_id"], row.get("row_of_the_file"))      # noqa: E731
    assert sorted(map(placed, next_rows)) == sorted(map(placed, of_l5c)) and len(next_rows) == 4 and [placed(row) for row in training_order(next_rows, 3)] == [placed(row) for row in of_l5c]
    assert not any(field in row for row in next_rows for field in TEXT_FIELDS) and all(row["origin"] == "pretraining" for row in next_rows)
    assert not {row["id"] for row in next_rows} & {row["id"] for row in _file_rows(second.l5f / "l5_full_training_published.jsonl")}

    # ---- THE SEARCH: B makes 64 attempts at each of the four problems, L6's sampling seed; one problem is solved
    assert second.searching.searched == [(key, ATTEMPTS, 1052, 1024) for key in SEARCHED_2]
    sampled = summaries[SEARCH]
    assert (sampled["model"], sampled["adapter"], sampled["set"], sampled["problems"], sampled["episodes_each"], sampled["sampling_seed"], sampled["resolved_at_least_once"],
            sampled["attempts"]) == ("published", str(second.best), "l6b_search", 4, ATTEMPTS, 1052, 1, 4 * ATTEMPTS)

    # ---- THE TWO SETS: K = 1 (the smaller of half a set, the old own rows and the new solves); the SAME three published rows in both; never with text
    made = summaries[PREPARE]
    assert (made["new_solves"], made["old_own_rows"], made["K"], made["rows_a_set"], made["published_rows_a_set"], made["trained"], made["averaged"], made["optimizer_steps"]) == (
        1, 2, 1, 4, 3, list(TRAINED), list(MODELS), 4)
    assert _but_what(made["enough_new_solves"]) == {"round_two_solves": 1, "minimum": 1, "passes": True} and made["trainings"] == {"run": True}
    assert made["the_sets_held"]["passes"] is True
    assert {key: value for key, value in made["the_dose"].items() if key != "what"} == {"half_a_set": 2, "old_own_rows": 2, "new_solves": 1, "K": 1, "new_solves_drawn_by_the_seed": False}
    found = made["the_search_found"]
    assert (found["new_solves"], found["problems"], found[PART_A]["solved"], found[PART_A]["problems"], found[PART_B]["solved"], found[PART_B]["problems"],
            found["solved_in_under_half_of_its_attempts"], found["per_1000"], found["enough"]) == (1, 4, 1, 2, 0, 2, 1, 7.81, True)
    assert (found["lines"]["new_solves"]["median"], found["lines"]["old_own_rows"]["median"]) == (9, 2.0)
    new = store.read_rows("l6b_new_solves.jsonl")
    assert [(row["problem_id"], row["origin"], row["part"], row["k"], row["n"], row["first_success"], row["lines"], row["in_the_set"]) for row in new] == [
        (UNSOLVED_1[0], NEW, PART_A, 2, ATTEMPTS, 21, 9, True)] and new[0]["attempt_id"].rsplit("#", 1)[1] in ("20", "40")       # ONE verified attempt, drawn by the seed
    sets = {model: store.read_rows(f"l6b_training_{model}.jsonl") for model in MODELS}
    own_of = {model: [row for row in sets[model] if row["origin"] != "pretraining"] for model in MODELS}
    published_of = {model: sorted(placed(row) for row in sets[model] if row["origin"] == "pretraining") for model in MODELS}
    assert [len(sets[model]) for model in MODELS] == [4, 4] and published_of[NEW_OWN] == published_of[OLD_OWN] == sorted(placed(row) for row in next_rows[:3])
    assert [(row["id"], row["origin"]) for row in own_of[NEW_OWN]] == [(new[0]["id"], NEW)] and [(row["id"], row["origin"]) for row in own_of[OLD_OWN]] == [(old[0]["id"], "deep")]
    assert not any(field in row for own in sets.values() for row in own if row["origin"] == "pretraining" for field in TEXT_FIELDS)
    for model in MODELS:
        assert [placed(row) for row in sets[model]] == [placed(row) for row in training_order(sets[model], 3)] and [row["row"] for row in sets[model]] == [0, 1, 2, 3]
        assert made["the_sets"][model]["rows"] == 4 and made["the_sets"][model]["rows_by_origin"] == {("new" if model == NEW_OWN else "deep"): 1, "pretraining": 3}
        assert made["the_orders"][model] == {"the_order_of_the_tasks_seed": [3], "alike": []}
        for order_seed in (1, 2, 3, 4):
            own, entry_ = store.read_rows(f"l6b_training_{model}_{order_seed}.jsonl"), made["training_sets"][f"{model}_{order_seed}"]
            assert [placed(row) for row in own] == [placed(row) for row in training_order(sets[model], order_seed)] and [row["row"] for row in own] == [0, 1, 2, 3]
            assert (entry_["order_seed"], entry_["model"], entry_["rows"], entry_["optimizer_steps"]) == (order_seed, model, 4, 4)

    # ---- EIGHT TRAININGS, each FROM B (L5f's averaged `published`, in L5f's run) at rank 64, by L4t's own training body
    assert [call["more"] for call in calls] == [{"start": second.best, "learning_rates": True}] * 8 and [call["seed"] for call in calls] == [3] * 8
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l6b_published_seed3_{name}" for name in TRAINED]
    assert all((call["lora"]["rank"], call["lora"]["alpha"]) == (64, 128) for call in calls)
    for call, name in zip(calls, TRAINED):
        train, loss = summaries[f"ladder_l6b_train_{name}"], json.loads(store.path(f"l6b_loss_{name}.json").read_text())
        own = store.read_rows(f"l6b_training_{name}.jsonl")
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in own] and loss["rows_trained"] == [row["id"] for row in own]
        assert train["adapter"] == str(store.root / "adapters" / name) and train["learning_rate"]["decay"] is False and f"the ORDER SEED {name[-1]}" in train["order"]
        assert train["trained_from"] == f"the stored adapter {second.best}" and (train["adapter_saved"]["rank"], train["adapter_saved"]["alpha"], len(loss["training_steps"])) == (64, 128, 4)
    # TWO AVERAGED ADAPTERS, trained by nothing, by L5n's own function: each the mean of its four (the k-th training of this stage holds 24 + k).
    number = lambda directory: set(_numbers(directory / "adapter_model.safetensors", A_LAYER).tolist())      # noqa: E731
    for model, mean, held in zip(MODELS, (26.5, 30.5), ((25.0, 26.0, 27.0, 28.0), (29.0, 30.0, 31.0, 32.0))):
        averaged, directory = summaries[f"ladder_l6b_average_{model}"], store.root / "adapters" / model
        members = [store.root / "adapters" / f"{model}_{order}" for order in (1, 2, 3, 4)]
        assert [number(path) for path in members] == [{value} for value in held] and number(directory) == {mean}
        assert (averaged["model"], averaged["recipe"], averaged["adapter"], averaged["averaged"], averaged["check"], averaged["of"]) == (
            model, "mean_constant", str(directory), True, "L6b", [path.name for path in members])
        assert averaged["read_back"] == is_the_mean(directory / "adapter_model.safetensors", [path / "adapter_model.safetensors" for path in members]) and averaged["read_back"]["passes"] is True
    # TWO MEASUREMENTS OF THE GOAL SET'S FIRST SAMPLING ALONE, each served at rank 64: no rung, no second sampling, no held-aside set.
    assert solver.measured == [(GOAL, 32, 1001, 1024)] * 2 and [kit["max_lora_rank"] for kit in kits] == [64] * 3 == [kit["rank"] for kit in kits]      # the search, then the two models
    for model, closes in zip(MODELS, OF_THESE):
        measured = summaries[f"ladder_l6b_measure_{model}"]
        assert (measured["model"], measured["adapter"], list(measured["goal"]), measured["goal"]["reach"]["set"]) == (model, str(store.root / "adapters" / model), ["reach"], f"l4_reach_{model}")
        assert [(row["problem_id"], row["episodes"], row["resolved"]) for row in _file_rows(store.root / f"episodes_l4_reach_{model}_problems.jsonl")] == [(GOAL, 32, closes)]
    assert sorted({row["set"] for row in store.read_rows("problems.jsonl")}) == ["l4_reach_new_own", "l4_reach_old_own", "l6b_search"]

    # ---- THE REPORT: the checks first, each PASS; then the search, the primary, whose proofs, what follows
    report = summaries[REPORT]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["failed_checks"] == [] and report["not_to_be_read"] == []
    assert json.loads(store.path("report_ladder_l6b.json").read_text()) == json.loads(json.dumps(report)) and store.done_summary(REPORT)["branches"] == report["branches"]
    checks = report["can_this_run_be_read"]
    assert list(checks) == ["what", "the_run", *TRAINED, *MODELS] and list(checks["the_run"]) == THE_RUN_CHECKS and all(
        entry_["passes"] is True for name, of in checks.items() if name != "what" for entry_ in of.values())
    of_b, of_theirs, of_the_searched, of_the_sets = (checks["the_run"][key] for key in (B_IS_L5FS, L5CS_RUN, THE_SEARCHED, THE_SETS))
    assert (of_b["adapter"], of_b["the_search_was_served_from"], of_b["trainings_not_from_it"], of_b["its_report_now"]) == (str(second.best), str(second.best), [], True)
    assert (of_theirs["its_report_now"], of_theirs["the_next_rows_are_its_again"], of_theirs["rows"], of_theirs["trained_from"]) == (True, True, 4, str(second.best))
    assert (of_the_searched["problems"], of_the_searched["problems_of_the_rule"], of_the_searched["is_the_rule_made_again"], of_the_searched["sampled_as_prepared"],
            of_the_searched["new_solves"], of_the_searched["new_solves_made_again"]) == (4, 4, True, True, 1, 1)
    assert (of_the_searched["held_out"], of_the_searched["of_the_goal_set"], of_the_searched["of_the_held_aside_set"], of_the_searched["solved_by_search_one"]) == (0, 0, 0, 0)
    assert (of_the_sets["K"], of_the_sets["new_solves"], of_the_sets["old_own_rows"], of_the_sets["rows"], of_the_sets["sets_that_are_not_the_rule"]) == (1, 1, 2, {NEW_OWN: 4, OLD_OWN: 4}, [])
    assert report["branches"] == {"primary": ADD, "whose_proofs": TEACH_MORE, "what_follows": THE_CYCLE_TURNS}
    primary, whose = report["the_primary"], report["whose_proofs"]
    assert (primary["name"], primary["difference"], primary["line"], primary[NEW_OWN], primary["L5c's stored again"]) == (
        ADD, 0.125, 0.0085, {"successes": 12, "attempts": 32}, {"successes": BAR, "attempts": 32})
    assert (whose["name"], whose["difference"], whose[NEW_OWN], whose[OLD_OWN]) == (TEACH_MORE, 0.125, {"successes": 12, "attempts": 32}, {"successes": 8, "attempts": 32})
    of_this_seed = report["of_this_seed"]
    assert (of_this_seed["seed"], of_this_seed["start"], of_this_seed["recipe"], of_this_seed["trainings_run"], of_this_seed["new_solves"], of_this_seed["the_reads"]) == (
        3, "published", "mean_constant", True, 1, {"primary": 0.125, "whose_proofs": 0.125})
    assert report["the_search"]["found"] == found and report["the_search"]["sampled"]["adapter"] == str(second.best) and report["beside"]["old_own_minus_the_bar"] == 0.0
    figures = report["figures"]
    assert list(figures["the_stored_models"]) == ["l5f_published", "l5c_again", "second_full", "pre_r64"] and [figures[model]["all_of_g"]["successes"] for model in MODELS] == list(OF_THESE)
    assert (figures["the_stored_models"]["l5f_published"]["successes"], figures["the_stored_models"]["l5c_again"]["successes"], report["the_bar"]["run"]) == (B, BAR, str(second.l5c))
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == sorted([*MODELS, *TRAINED]) and report["rank"] == 64 and report["smoke"] is None
    lines = report["lines"]
    assert all(line.startswith("l4 (pretrained on published proofs): ") for line in lines) and lines[0].startswith(
        "l4 (pretrained on published proofs): L6b, A SECOND ROUND, FROM THE BEST MODEL. Seed 3. B is L5f's stored averaged `published` of this seed")
    assert "made 64 attempts at each of 4 problems of L6's search 2 (2 that round 1's search attempted and left unsolved, 2 candidates neither draw took), sampling seed 1052" in lines[0]
    assert "4 rows each, K = 1: `new_own` (K of B's new solves and the first 3 of the next published rows of the seed's order)" in lines[0]
    said = lambda start: next(line for line in lines if line.startswith(f"l4 (pretrained on published proofs): {start}"))      # noqa: E731
    assert said("THE SEARCH, a finding of its own: B solved 1 of the 4 problems (NEW SOLVES; 1 are asked for a training): 1 of the 2 that round 1's search left unsolved")
    assert said("THE PRIMARY (a scout's read): ").endswith(f"{ADD}. `new_own` minus L5c's stored `again`, per 1,000 attempts on all of G: +125.00 (12 of 32 against 8 of 32); the lines "
                                                           "are +8.5 and -8.5")
    assert said("WHOSE PROOFS (a scout's read): ").endswith(f"{TEACH_MORE}. `new_own` minus `old_own`, per 1,000 attempts on all of G: +125.00 (12 of 32 against 8 of 32); the lines are "
                                                            "+8.5 and -8.5")
    assert said("WHAT FOLLOWS: ").startswith(f"l4 (pretrained on published proofs): WHAT FOLLOWS: {THE_CYCLE_TURNS}. The primary reads") and said("BESIDE, NAMING NOTHING") and said("TIMEOUTS")
    assert sum(line.startswith("l4 (pretrained on published proofs): CHECK, ") for line in lines) == 11 and all(line.endswith(": PASS") for line in lines if ": CHECK, " in line)
    assert ADD in report["headline"] and "THE SEARCH: 1 new solves of 4 problems" in report["headline"] and "seed 3, from L5f's averaged `published`" in report["headline"]

    # ---- QUEUED AGAIN, every step returns what is stored: nothing is searched, trained, averaged or measured again; the report is built again, the same
    before = (len(calls), len(kits), len(second.searching.searched), len(solver.measured))
    again = _stage6b(config, monkeypatch)
    assert (len(second.calls()), len(kits), len(second.searching.searched), len(solver.measured)) == before and json.loads(json.dumps(again[REPORT])) == json.loads(json.dumps(report))
    assert {path: _all(path) for path in read_only} == read_only


@pytest.mark.parametrize("new, old, primary, whose, follows", [
    (8, 8, NOT_SHOWN_AT_ONE, NOT_SHOWN_AT_ONE, SAID_AS_IT_IS),       # each as the bar: between the lines
    (7, 9, COST, TEACH_LESS, TO_A_DECISION),                           # `new_own` below the bar and below `old_own`
    (9, 12, ADD, TEACH_LESS, THE_CYCLE_TURNS)])                        # above the bar, and the weaker model's proofs taught more: the two reads are two questions
def test_the_two_reads_at_the_branches_a_one_problem_world_reaches(second, monkeypatch, new, old, primary, whose, follows):
    second.solver.successes[:] = [new, old]
    report = _stage6b(second.config, monkeypatch)[REPORT]
    assert report["inconclusive"] is False and report["branches"] == {"primary": primary, "whose_proofs": whose, "what_follows": follows}
    assert (report["the_primary"]["difference"], report["whose_proofs"]["difference"]) == (round((new - BAR) / 32, 6), round((new - old) / 32, 6))
    assert report["beside"]["old_own_minus_the_bar"] == round((old - BAR) / 32, 6) and report["what_follows"]["whose_proofs"] == whose
    assert any(line.startswith(f"l4 (pretrained on published proofs): WHAT FOLLOWS: {follows}. The primary reads {primary}: ") for line in report["lines"])


def test_a_failed_check_makes_the_run_inconclusive_and_nothing_else_is_said(second, monkeypatch):
    config = second.config
    first = _stage6b(config, monkeypatch)[REPORT]
    store = second.store()
    assert first["inconclusive"] is False
    rebuilt = lambda: ladder_l6b.ladder_l6b_report(config)      # noqa: E731 - the report step is built again each time it is run
    of_the_seed = lambda path, **changed: _changed(path, lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], **changed}}))      # noqa: E731

    def nothing_is_said(report, failed):
        assert report["inconclusive"] is True and report["failed_checks"] == [f"the run: {check}" for check in failed] and report["branches"] == NOTHING_READ
        assert (report["the_primary"]["name"], report["whose_proofs"]["name"], report["what_follows"]["name"]) == (INCONCLUSIVE, INCONCLUSIVE, NOT_READ)
        assert report["of_this_seed"]["figures"] is None and report["of_this_seed"]["the_reads"] is None and report["of_this_seed"]["new_solves"] is None
        assert "measured_and_not_read" in report["figures"] and "measured_and_not_read" in report["beside"] and "measured_and_not_read" in report["the_search"]
        assert not any(": THE PRIMARY" in line or ": WHOSE PROOFS" in line or ": THE SEARCH, " in line or ": THE FIGURES" in line or ": WHAT FOLLOWS" in line for line in report["lines"])
        assert any(line.endswith("Nothing else is said of it") and f"THIS RUN IS {INCONCLUSIVE}: the run: {failed[0]}" in line for line in report["lines"])
        assert f"THIS RUN IS {INCONCLUSIVE}" in report["headline"] and report["ok"] is True
        return report["can_this_run_be_read"]["the_run"]

    # ---- B: L5f's report reads INCONCLUSIVE now; a training of this run that was not started from B.
    kept = of_the_seed(second.l5f / "report_ladder_l5_full.json", inconclusive=True, failed_checks=["`deep`: lean_answered"])
    assert nothing_is_said(rebuilt(), [B_IS_L5FS])[B_IS_L5FS]["its_report_now"] is False
    (second.l5f / "report_ladder_l5_full.json").write_text(kept)
    marker = store.path("ladder_l6b_train_new_own_2.done.json")
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "trained_from": "the stored adapter /another/adapter"}))
    assert nothing_is_said(rebuilt(), [B_IS_L5FS])[B_IS_L5FS]["trainings_not_from_it"] == ["new_own_2"]
    marker.write_text(kept)
    # ... and a search that was served from another adapter than B's.
    marker = store.path(f"{SEARCH}.done.json")
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "adapter": "/another/adapter"}))
    assert nothing_is_said(rebuilt(), [B_IS_L5FS])[B_IS_L5FS]["the_search_was_served_from"] == "/another/adapter"
    marker.write_text(kept)
    # ---- THE BAR: L5c's report reads INCONCLUSIVE now; its stored `again` set is no longer the next rows of the seed's order (the sets' check reads it too).
    kept = of_the_seed(second.l5c / "report_ladder_l5_cont.json", inconclusive=True, failed_checks=["`more_2`: no_barred_row"])
    of_theirs = nothing_is_said(rebuilt(), [L5CS_RUN])[L5CS_RUN]
    assert of_theirs["its_report_now"] is False and "it reads INCONCLUSIVE" in of_theirs["why_not"] and of_theirs["the_next_rows_are_its_again"] is True
    (second.l5c / "report_ladder_l5_cont.json").write_text(kept)
    kept = _changed(second.l5c / "l5_cont_training_again.jsonl", _swapped)
    checks = nothing_is_said(rebuilt(), [L5CS_RUN, THE_SETS])
    assert checks[L5CS_RUN]["the_next_rows_are_its_again"] is False and "first that differs at place 0" in checks[L5CS_RUN]["refused"] and checks[THE_SETS]["the_next_rows_are_l5cs_again"] is False
    (second.l5c / "l5_cont_training_again.jsonl").write_text(kept)
    # ... and its stored first sampling, as this run copied it, is not the setting's (16 attempts a problem).
    kept = _rewritten(store.path("l4_stored_l5c_again_reach.jsonl"), episodes=16)
    assert nothing_is_said(rebuilt(), [THE_BARS_SAMPLING])[THE_BARS_SAMPLING]["attempts_a_problem_that_are_not_the_settings"] == [16]
    store.path("l4_stored_l5c_again_reach.jsonl").write_text(kept)
    # ---- THE SEARCHED SET: a stored problem that L6's rule, made again, does not give (one of round 1's solves in the place of one of (b)).
    problems = store.path("l6b_search_problems.jsonl")
    kept = _changed(problems, lambda text: _as_text([{**row, "problem_id": SOLVED_1[0]} if row["problem_id"] == NEITHER[0] else row for row in map(json.loads, text.splitlines())]))
    of_the_searched = nothing_is_said(rebuilt(), [THE_SEARCHED])[THE_SEARCHED]
    assert (of_the_searched["is_the_rule_made_again"], of_the_searched["solved_by_search_one"]) == (False, 1)
    problems.write_text(kept)
    # ---- THE SETS: `old_own` with another own row than the FIRST of L5d's stored ones; a published row that carries text.
    stored, old = store.path("l6b_training_old_own.jsonl"), store.read_rows("l6b_old_solves.jsonl")
    kept = _changed(stored, lambda text: _as_text([{**old[1], "row": row["row"]} if row["origin"] == "deep" else row for row in map(json.loads, text.splitlines())]))
    of_the_sets = nothing_is_said(rebuilt(), [THE_SETS])[THE_SETS]
    assert OLD_OWN in of_the_sets["sets_that_are_not_the_rule"] and of_the_sets["refused"] is not None and of_the_sets["the_next_rows_are_l5cs_again"] is False
    stored.write_text(kept)
    stored = store.path("l6b_training_new_own.jsonl")
    kept = _changed(stored, lambda text: _as_text([{**row, "completion": "  done\n"} if row["origin"] == "pretraining" else row for row in map(json.loads, text.splitlines())]))
    assert nothing_is_said(rebuilt(), [THE_SETS])[THE_SETS]["published_rows_with_text"] == 3
    stored.write_text(kept)
    # WITH EVERYTHING AS IT WAS the report is the one it was.
    assert json.loads(json.dumps(rebuilt())) == json.loads(json.dumps(first))


def test_as_on_the_gpu_the_first_step_reads_bs_five_adapters_the_search_is_served_from_b_and_every_measurement_from_its_own_models_adapter(second, monkeypatch):
    config, kits, runs = second.config, second.kits, second.runs
    second_full, theirs = second.l4p / "adapters" / "second_full", second.l5f / "adapters"
    prepare = ladder_l6b.STEPS[SEARCH_PREPARE]
    _adapter_of(second_full, 64, 128, 9.0)          # the scripted trainings of this world leave adapters alike: as on the box, `second_full`'s numbers are its own
    # ONE OF L5f's ADAPTERS IS NOT THERE (its run recorded it, the files are gone): refused by L5c's own reading of that run, in this stage's words, before anything is written.
    (theirs / "published_2").rename(theirs / "kept")
    with pytest.raises(RuntimeError, match="L6b reads L5f's run of seed 3 as L5c does .its stored averaged `published` is B: it makes the search and both models are trained from it., "
                                           "and that reading refuses it: .*adapters/published_2 is not there: .*nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    (theirs / "kept").rename(theirs / "published_2")
    assert _nothing_written(runs)
    # WITH THE FIVE AS L5f's RUN LEFT THEM the first step reads the files; the search and every measurement ask the model server for an adapter.
    with monkeypatch.context() as patched:
        for module in (ladder_l5, ladder_ceiling):                              # each names the test of a GPU box as its own: the search's body, the measuring body
            patched.setattr(module, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        summaries = _stage6b(config, patched, prepare_on_the_gpu=True)
    store = second.store()
    now = summaries[SEARCH_PREPARE]["the_best"]["held"]["read_now"]
    assert list(now) == [*OF_THEIRS, "published"] and all(own["passes"] is True and (own["read_back"]["rank"], own["read_back"]["alpha"]) == (64, 128) for own in now.values())
    report = summaries[REPORT]
    assert report["inconclusive"] is False and any("CHECK, B is L5f's averaged `published` of this seed" in line and "; the files themselves were read; " in line and line.endswith(": PASS")
                                                   for line in report["lines"])
    # THE SEARCH IS SERVED FROM B's ADAPTER, which L5F's RUN keeps; each measurement from the averaged adapter THIS run keeps. All at rank 64.
    kept_here = store.root / "adapters"
    assert [kit["adapter"][2] for kit in kits] == [str(second.best), *(str(kept_here / model) for model in MODELS)] and [kit["max_lora_rank"] for kit in kits] == [64] * 3
    assert summaries[SEARCH]["stand_in_engine"] is False and summaries["ladder_l6b_measure_new_own"]["stand_in_engine"] is False and not (kept_here / "published").exists()


# ----------------------------------------------------------------------------------------- too few new solves
def test_with_fewer_new_solves_than_are_asked_no_set_is_built_nothing_is_trained_and_the_search_is_the_finding(second, monkeypatch):
    config = second.config
    config["ladder_loop"]["l6"]["search"]["minimum"] = 2            # B's search solves ONE problem here
    read_only = {path: _all(path) for path in (second.l5, second.l5f, second.l5d, second.l5c, second.l4p)}
    summaries = _stage6b(config, monkeypatch)
    store = second.store()
    made = summaries[PREPARE]
    assert list(summaries) == STEPS and _but_what(made["enough_new_solves"]) == {"round_two_solves": 1, "minimum": 2, "passes": False} and made["trainings"]["run"] is False
    assert (made["new_solves"], made["K"], made["trained"], made["averaged"], made["training_sets"]) == (1, 0, [], [], {}) and "nothing is trained" in made["trainings"]["why"]
    assert second.calls() == [] and second.solver.measured == [] and len(second.kits) == 1                    # the search alone asked an engine
    assert all(summaries[step]["run"] is False for step in STEPS[3:-1])
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == sorted([
        *(f"{step}.done.json" for step in (SEARCH_PREPARE, SEARCH, PREPARE, REPORT)), *STORED, *SEARCH_FILES, "problems.jsonl", "report_ladder_l6b.json", "l4_goal_set_again.jsonl",
        "l4_heldout_groups.jsonl", "l5_held_aside_problems.jsonl"])
    assert [(row["problem_id"], row["in_the_set"]) for row in store.read_rows("l6b_new_solves.jsonl")] == [(UNSOLVED_1[0], False)]
    report = summaries[REPORT]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["failed_checks"] == [] and report["models"] == []
    assert report["branches"] == {"primary": NOT_READ, "whose_proofs": NOT_READ, "what_follows": THE_SEARCH_IS_THE_FINDING} and report["trainings"]["run"] is False
    assert list(report["can_this_run_be_read"]) == ["what", "the_run"] and list(report["can_this_run_be_read"]["the_run"]) == THE_RUN_CHECKS[:-1]
    assert report["the_search"]["found"]["new_solves"] == 1 and report["of_this_seed"]["trainings_run"] is False and report["of_this_seed"]["figures"] is None
    assert report["adapters"]["there"] == [] and "NO MODEL WAS TRAINED" in report["lines"][0] and "NO MODEL WAS TRAINED" in report["headline"]
    assert any(line.startswith("l4 (pretrained on published proofs): THE SEARCH, a finding of its own: B solved 1 of the 4 problems (NEW SOLVES; 2 are asked for a training)")
               for line in report["lines"])
    assert any(line.startswith(f"l4 (pretrained on published proofs): WHAT FOLLOWS: {THE_SEARCH_IS_THE_FINDING}. ") for line in report["lines"])
    assert not any(": THE PRIMARY" in line or ": WHOSE PROOFS" in line or ": THE FIGURES" in line for line in report["lines"])
    assert {path: _all(path) for path in read_only} == read_only


# ----------------------------------------------------------------------------------------------- the refusals
def test_the_first_step_refuses_before_anything_is_written(second, monkeypatch):
    config, runs = second.config, second.runs
    prepare = ladder_l6b.ladder_l6b_search_prepare
    of_the_seed = lambda path, **changed: _changed(path, lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], **changed}}))      # noqa: E731
    # A SEED THE SETTING DOES NOT NAME: no run holds B or the bar at it.
    for seed in ("0", "4"):
        monkeypatch.setenv(SEEDS, seed)
        with pytest.raises(RuntimeError, match=rf"this task's seed is {seed} and ladder_loop.l6b.seeds is \[3\]: a task of another seed is not run .it needs L5f's `published` and L5c's "
                                               r"`again` of that seed, which no run holds. \(ladder_loop.l6b\). Nothing was written"):
            prepare(config)
    monkeypatch.setenv(SEEDS, "3")
    # THE SETTING ITSELF, by its one reader: another start, another bar, a seed that is none of L5c's, a key that is none of its own.
    settings = config["ladder_loop"]["l6b"]
    for changed, said in (({"start": "deep"}, "ladder_loop.l6b.start is 'deep': this stage is specified from L5f's stored averaged `published` of the seed alone"),
                          ({"bar": "wide"}, "ladder_loop.l6b.bar is 'wide': the bar is L5c's stored `again` of the seed alone"),
                          ({"seeds": [4]}, r"ladder_loop.l6b.seeds is \[4\]: distinct whole numbers of at least 0, at least one, each a seed L5c's setting fixes \(\[3\]\)"),
                          ({"seeds": []}, r"ladder_loop.l6b.seeds is \[\]"), ({"more_seeds": [4, 5]}, r"ladder_loop.l6b holds the keys \['bar', 'more_seeds', 'seeds', 'start'\]")):
        kept = dict(settings)
        settings.update(changed)
        with pytest.raises(ValueError, match=said):
            prepare(config)
        settings.clear()
        settings.update(kept)
    # THE SMOKE-ONLY VARIABLE without a run directory of the task's naming.
    monkeypatch.setenv(SMOKE, "2")
    with pytest.raises(RuntimeError, match=r"RLVR_LEAN_LADDER_L6B_SMOKE_PROBLEMS is set \(2\): it is for a SMOKE run alone, in a run directory the task names "
                                           r".RLVR_LEAN_LADDER_L6B_RUN. and on the fixtures' base map. A run in the seed's own directory reads a run of L5 whose checks passed and takes "
                                           "its problems by the rule and by nothing else. Nothing was written"):
        prepare(config)
    monkeypatch.delenv(SMOKE)
    # L5's RUN not to be read: refused by L5's own reading, in this stage's words.
    report_file = second.l5 / "report_ladder_l5.json"
    kept = _changed(report_file, lambda text: json.dumps({**json.loads(text), "inconclusive": True, "failed_checks": ["`shallow`: the_training_ran"]}))
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .*: no second round stands on such a run"):
        prepare(config)
    report_file.write_text(kept)
    # L4p's RUN, where `second_full` lives, not there: refused as L5f's own first step refuses it.
    second.l4p.rename(second.l4p.with_name("kept_l4p"))
    with pytest.raises(RuntimeError, match="`second_full` is NOT THERE .*Run the task of stage `ladder_l4_dose_r64` for seed 0 .* to its end first; nothing was written"):
        prepare(config)
    second.l4p.with_name("kept_l4p").rename(second.l4p)
    # ---- B: L5f's RUN OF THE SEED, by L5c's own reading of it, in this stage's words.
    report_file = second.l5f / "report_ladder_l5_full.json"
    kept = of_the_seed(report_file, inconclusive=True, failed_checks=["`deep`: lean_answered"])
    with pytest.raises(RuntimeError, match="L6b reads L5f's run of seed 3 as L5c does .*and that reading refuses it: L5f's run of seed 3 cannot carry L5c: .*it reads INCONCLUSIVE.*"
                                           "Run the task of stage `ladder_l5_full` for seed 3 .* to its end first; nothing was written"):
        prepare(config)
    report_file.write_text(kept)
    marker = second.l5f / "ladder_l5_full_average_published.done.json"
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "read_back": {**json.loads(text)["read_back"], "is_the_mean": False, "passes": False}}))
    with pytest.raises(RuntimeError, match="L6b reads L5f's run of seed 3 as L5c does .*the adapters of `published` in L5f's run ladder_l5_draws_second_full_seed3 are not held"):
        prepare(config)
    marker.write_text(kept)
    # ---- THE BAR: L5c's RUN OF THE SEED. Not there; INCONCLUSIVE; of another recipe, start or seed; not to be read; measured otherwise. Each refused, naming its task.
    second.l5c.rename(second.l5c.with_name("kept_l5c"))
    with pytest.raises(RuntimeError, match=r"L5c's run of seed 3 cannot carry L6b: ladder_l5_cont_second_full_seed3 does not hold \['ladder_l5_cont_report.done.json', "
                                           r"'report_ladder_l5_cont.json'\]: L5c's run of this seed is not on the box, or not finished. Its stored `again` .from the same model, the next "
                                           r"published rows of the same order, the same recipe. is the bar of the primary read. Run the task of stage `ladder_l5_cont` for seed 3 "
                                           r"\(`python -m rlvr_lean.runner.entry --stage ladder_l5_cont --seeds 3`; for the smoke run, stage `ladder_l5_cont_smoke`\) to its end first; nothing was written"):
        prepare(config)
    second.l5c.with_name("kept_l5c").rename(second.l5c)
    report_file = second.l5c / "report_ladder_l5_cont.json"
    for changed, why in (({"inconclusive": True, "failed_checks": ["`again_3`: no_barred_row"]}, "it reads INCONCLUSIVE .`again_3`: no_barred_row."),
                         ({"recipe": "constant"}, "its recipe is `constant` and this run's `mean_constant`"), ({"ok": False}, "its report is not to be read"),
                         ({"start": "pre_r64"}, "it is a run from `pre_r64` and L5c's models are from `second_full`"), ({"seed": 4}, "it is a report of seed 4"),
                         ({"the_measurement": {"sampling": "reach", "episodes": 16}}, "it was measured otherwise")):
        kept = of_the_seed(report_file, **changed)
        with pytest.raises(RuntimeError, match=f"L5c's run of seed 3 cannot carry L6b: .*{why}.*Run the task of stage `ladder_l5_cont` for seed 3 .* to its end first; nothing was written"):
            prepare(config)
        report_file.write_text(kept)
    # ... without the marker of one of `again`'s trainings; with an `again` that was trained from another adapter than B's; with rows on the first sampling that would not pair.
    marker = second.l5c / "ladder_l5_cont_train_again_2.done.json"
    marker.rename(marker.with_name("kept.json"))
    with pytest.raises(RuntimeError, match=r"ladder_l5_cont_second_full_seed3 does not hold \['ladder_l5_cont_train_again_2.done.json'\]: the trainings of `again` L5c's run of seed 3 "
                                           "made, its averaged adapter's record, its stored set or its rows on `reach`. Run the task of stage `ladder_l5_cont` for seed 3 .* to its end "
                                           "first; nothing was written"):
        prepare(config)
    marker.with_name("kept.json").rename(marker)
    marker = second.l5c / "ladder_l5_cont_prepare.done.json"
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "the_models": {**json.loads(text)["the_models"], "again": {"from": "/another/adapter"}}}))
    with pytest.raises(RuntimeError, match="L5c's run ladder_l5_cont_second_full_seed3 trained `again` from /another/adapter and this stage starts from .*adapters/published: the bar is "
                                           "a model trained from the SAME model as `new_own` and `old_own`, or the primary reads two starts. Nothing was written"):
        prepare(config)
    marker.write_text(kept)
    marker = second.l5c / "episodes_l4_reach_again.done.json"
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "episodes_each": 16}))
    with pytest.raises(RuntimeError, match=r"ladder_l5_cont_second_full_seed3 measured `again` \(reach\) with sampling seed 1001 and 16 episodes; this config gives 1001 and 32: .* "
                                           r"\(L5c's run of seed 3: its stored `again` is the bar, paired by problem with this run's models\). Nothing was written"):
        prepare(config)
    marker.write_text(kept)
    # ... and THE NEXT ROWS of the seed's order are not the set it stored as `again` (two of its rows changed places).
    kept = _changed(second.l5c / "l5_cont_training_again.jsonl", _swapped)
    with pytest.raises(RuntimeError, match=r"the next rows of the seed's order are not the set L5c's run of this seed stored as `again`, id for id \(4 rows here and 4 there; first that "
                                           r"differs at place 0: .*Refused \(seed 3; L5f's run ladder_l5_draws_second_full_seed3; L5c's run ladder_l5_cont_second_full_seed3\). Nothing "
                                           "was written"):
        prepare(config)
    (second.l5c / "l5_cont_training_again.jsonl").write_text(kept)
    # ---- THE OLD OWN ROWS: L5d's RUN OF THE SEED, by L5f's own reading of it; and a stored `deep` set that is not L5d's rule for the seed.
    report_file = second.l5d / "report_ladder_l5_draws.json"
    kept = of_the_seed(report_file, inconclusive=True, failed_checks=["`deep`: lean_answered"])
    with pytest.raises(RuntimeError, match="L6b reads L5d's run of seed 3 as L5f does .its stored `deep` set holds the old own rows of `old_own`., and that reading refuses it: "):
        prepare(config)
    report_file.write_text(kept)
    kept = _changed(second.l5d / "l5_draws_training_deep.jsonl", _swapped)
    with pytest.raises(RuntimeError, match="L5d's rule for seed 3 gives a `deep` set of 4 rows and its run ladder_l5_draws_pre_r64_seed3 stores 4, or not the same ids in the same order: "
                                           "the old own rows of `old_own` are the first of the ones that run trained. Nothing was written"):
        prepare(config)
    (second.l5d / "l5_draws_training_deep.jsonl").write_text(kept)
    assert _nothing_written(runs)
    # WITH EVERYTHING AS IT WAS the first step passes; a later step of a run that was prepared for another B is refused.
    read = prepare(config)
    assert read["the_best"]["held"]["passes"] is True and second.store().is_done(SEARCH_PREPARE)
    marker = second.store().path(f"{SEARCH_PREPARE}.done.json")
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "the_search": {**json.loads(text)["the_search"], "adapter": "/another/adapter"}}))
    with pytest.raises(RuntimeError, match="this run was prepared to search with /another/adapter and B is now .*adapters/published: a search is made as it was prepared. Nothing was sampled"):
        ladder_l6b.ladder_l6b_search(config)
    marker.write_text(kept)
    # A STEP BEFORE ITS TURN is refused, naming what it needs.
    with pytest.raises(RuntimeError, match="ladder_l6b_search"):
        ladder_l6b.ladder_l6b_prepare(config)


def test_the_steps_run_through_the_gpu_entry_point(second, monkeypatch, tmp_path):
    monkeypatch.setattr(milestone2, "load_config", lambda path: second.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    # In the stand-in's world no adapter is trained and none is averaged, so the report says INCONCLUSIVE and makes no read.
    assert json.loads((tmp_path / "ladder_l6b_average_new_own.json").read_text())["averaged"] is False
    report = json.loads((tmp_path / "ladder_l6b_report.json").read_text())
    assert second.store().is_done(REPORT) and report["inconclusive"] is True and report["stand_in_engine"] is True and report["branches"] == NOTHING_READ


# ------------------------------------------------------------------------- the smoke stage, in the box's own world
SMOKE_SCRIPT = {f"fixture_p{index}": {7: 3} for index in range(1, 5)}          # B's search in the smoke world: every problem it attempts is solved at its eighth attempt


def test_in_the_boxs_own_smoke_world_the_smoke_stage_runs_every_step_on_what_the_smoke_runs_left_and_says_so(arm, monkeypatch):  # noqa: F811
    """THE SMOKE WORLD AS THE BOX HAS IT, every run made by its own smoke stage in its own environment and in the box's
    order: L1's; the pretraining's, the arm's, the rank check's, L4t's and L4t2's; L4p's; L5's; L5d's; L5f's; L5c's. Then
    this stage's smoke task as its job queues it: all sixteen steps."""
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RUN, SMOKE, L4P_RUN, L5F_RUN, L5C_RUN, L6_RUN, SEEDS, ladder_l5.RUN_VARIABLE, ladder_l5.L4T2_RUN_VARIABLE, ladder_l5.SMOKE_VARIABLE, ladder_l4_start.MAP_RUN_VARIABLE,
                 ladder_l4_rank.RUN_VARIABLE, ladder_l4_rank.CHECK_VARIABLE, ladder_l4_rows.RUN_VARIABLE, ladder_l4_rows.ARM_RUN_VARIABLE, ladder_l4_rows.START_VARIABLE,
                 ladder_l4_rows.MINIMUM_VARIABLE, ladder_l4_rows.START_CHECKS_VARIABLE, ladder_l4_rows2.RUN_VARIABLE, ladder_l4_rows2.L4T_RUN_VARIABLE, ladder_l4_dose.RUN_VARIABLE,
                 ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE, ladder_l5_full.SMOKE_VARIABLE, ladder_l5_cont.SMOKE_VARIABLE, ladder_l6.SMOKE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings_of_the_chain(monkeypatch, calls)
    searching = DeepSolver(arm.solver, {f"fixture_p{index}": (4,) for index in range(1, 5)})       # in L5's search every problem is solved at its fifth attempt
    engines = lambda solver: lambda self, adapter=None: (lambda given: (kits.append(given["vllm"]["max_lora_rank"]), (solver, stand_in_parameters))[1])      # noqa: E731
    # ---- the five smoke tasks before L4p's (as L5's own smoke test runs them): the pretraining's, the arm's, the check's, L4t's from `pre_r64`, then L4t2's
    steps = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS, **ladder_l4_rank.STEPS, **ladder_l4_rows.STEPS, **ladder_l4_rows2.STEPS}

    def of_the_chain(stage_name):
        def run(patched):
            if stage_name == "ladder_l4_smoke":
                arm.with_lean()
                patched.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
            if stage_name in ("ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
                patched.setattr(ladder_round.Engines, "kit", engines(arm.solver))
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                if environment == "gpu" and step in steps:
                    _on_the_gpu(patched, steps[step], config) if "_train" in step else steps[step](config)
        return run

    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
        _a_smoke_task(monkeypatch, stage_name, of_the_chain(stage_name))
    runs = pipeline.STORE / "runs-v4.27"
    arm.with_lean()
    monkeypatch.setattr(ladder_round.Engines, "kit", engines(arm.solver))
    # ---- L4p's smoke task (`second_full` of the smoke world), then the smoke `pre_r64`'s map as the box has it (every problem solved), then L5's
    _a_smoke_task(monkeypatch, "ladder_l4_dose_r64_smoke", lambda patched: _stage_p(config, patched, stage_name="ladder_l4_dose_r64_smoke"))
    _write_the_map(config, {"fixture_p1": (0, 1, 2), "fixture_p2": tuple(range(8)), "fixture_p3": tuple(range(8)), "fixture_p4": tuple(range(8))}, directory=runs / "ladder_l4b_map_smoke")
    monkeypatch.setattr(ladder_round.Engines, "kit", engines(searching))
    of_l5 = _a_smoke_task(monkeypatch, "ladder_l5_deep_smoke", lambda patched: _stage5(config, patched, stage_name="ladder_l5_deep_smoke"))
    assert of_l5["ladder_l5_report"]["inconclusive"] is True                    # a real run is refused for such a round 1; a smoke run reads it all the same
    # ---- L5d's, L5f's and L5c's smoke tasks, seed 3 (their jobs' `--seeds`). AS ON THE BOX one optimizer step on the same rows is ONE training: the adapters of a set are alike.
    del calls[:]
    _trainings(monkeypatch, calls, value=lambda number: 1.0)
    _l5ns_report(runs, AVERAGE_OF_FOUR)                                        # L5n's REAL run, which L5d's smoke stage reads for the recipe (it is on the box)
    solver = CycleSolver(searching, SMOKE_SCRIPT)
    monkeypatch.setattr(ladder_round.Engines, "kit", engines(solver))
    of_l5d = _a_smoke_task(monkeypatch, "ladder_l5_draws_smoke", lambda patched: _stage5d(config, patched, stage_name="ladder_l5_draws_smoke"), seed=3)
    of_l5f = _a_smoke_task(monkeypatch, "ladder_l5_full_smoke", lambda patched: _stage5f(config, patched, stage_name="ladder_l5_full_smoke"), seed=3)
    of_l5c = _a_smoke_task(monkeypatch, "ladder_l5_cont_smoke", lambda patched: _stage5c(config, patched, stage_name="ladder_l5_cont_smoke"), seed=3)
    assert [own[key]["inconclusive"] for own, key in ((of_l5d, "ladder_l5_draws_report"), (of_l5f, "ladder_l5_full_report"), (of_l5c, "ladder_l5_cont_report"))] == [False] * 3
    l5, l4p, l5d, l5f, l5c = (runs / name for name in ("ladder_l5_deep_smoke", "ladder_l4_dose_r64_smoke", "ladder_l5_draws_smoke_mean_constant", "ladder_l5_full_smoke",
                                                         "ladder_l5_cont_smoke"))
    assert sorted(name for name in _all(l5c) if not name.startswith("episodes_")) == L5CS_FILES and sorted(name for name in _all(l5f) if not name.startswith("episodes_")) == L5FS_FILES
    on_the_box = sorted(path.name for path in runs.iterdir() if path.is_dir())
    assert set(SMOKE_RUNS_NEEDED) <= set(on_the_box) and not [name for name in on_the_box if name.startswith("ladder_l6")]      # L6's own smoke run is not there, and not needed

    # ---- THIS STAGE's SMOKE TASK, as its job queues it: EVERY STEP RUNS; every run it reads is left file for file as it was. WHAT IT NEEDS ON THE BOX IS `SMOKE_RUNS_NEEDED`
    # AND NOTHING ELSE: every other run of the smoke world is moved away while it runs, and moved back after.
    aside = runs.with_name("moved_aside_for_the_smoke_task")
    aside.mkdir()
    for name in on_the_box:
        if name not in SMOKE_RUNS_NEEDED:
            (runs / name).rename(aside / name)
    assert sorted(path.name for path in runs.iterdir() if path.is_dir()) == SMOKE_RUNS_NEEDED
    read_only = {path: _all(path) for path in runs.iterdir() if path.is_dir()}
    before, engines_before = len(calls), len(kits)
    summaries = _a_smoke_task(monkeypatch, "ladder_l6b_smoke", lambda patched: _stage6b(config, patched, stage_name="ladder_l6b_smoke"), seed=3)
    own = runs / "ladder_l6b_smoke"
    assert list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only
    for path in sorted(aside.iterdir()):
        path.rename(runs / path.name)
    aside.rmdir()
    assert sorted(path.name for path in runs.iterdir() if path.name.startswith("ladder_l6")) == ["ladder_l6b_smoke"]
    assert sorted(name for name in _all(own) if not name.startswith("episodes_")) == OWN_FILES
    read = summaries[SEARCH_PREPARE]
    smoke, search = read["smoke"], read["the_search"]
    assert (smoke["problems_at_least"], smoke["l5_read_though_inconclusive"], smoke["l5s_failed_checks"], smoke["l5c_read_though_inconclusive"]) == (
        2, True, of_l5["ladder_l5_report"]["failed_checks"], False) and read["seed"] == 3
    assert (read["start_adapter"], read["start_run"], read["l5cs_run"]["run"], read["l5ds_run"]["run"], read["round_one"]["run"], read["second_start_adapter"]) == (
        str(l5f / "adapters" / "published"), str(l5f), str(l5c), str(l5d), str(l5), str(l4p / "adapters" / "second_full"))
    # THE FIXTURE'S MAP LEAVES SEARCH 2 NO PROBLEM: the search attempts the two map problems no search took, each marked as NOT the rule's. One new solve is enough.
    assert (search["problems"], search["taken_by_the_rule"], search["taken_by_the_runs_minimum"], search["problems_at_least"], search["minimum_new_solves"]) == (2, 0, 2, 2, 1)
    taken = {row["problem_id"] for name in ("l5_search_problems.jsonl", "l5_held_aside_problems.jsonl") for row in _file_rows(l5 / name)}
    searched = _file_rows(own / "l6b_search_problems.jsonl")
    assert sorted(row["problem_id"] for row in searched) == sorted({f"fixture_p{index}" for index in range(1, 5)} - taken) and len(taken) == 2
    assert [name.removeprefix("negation_of_") for name, *_ in solver.searched] == [row["problem_id"] for row in searched]      # a problem is attempted by either side
    assert {(n, seed) for _, n, seed, _ in solver.searched} == {(ATTEMPTS, 1052)}
    # B solves both; K is 1 (half of L5f's smoke set of two rows; L5d's smoke `deep` set holds one own row): the new solve of each set is DRAWN BY THE SEED among the two.
    made = summaries[PREPARE]
    assert (made["new_solves"], made["old_own_rows"], made["K"], made["rows_a_set"], made["published_rows_a_set"], made["optimizer_steps"]) == (2, 1, 1, 2, 1, 2 // config["training"]["effective_batch"] or 1)
    assert made["the_dose"]["new_solves_drawn_by_the_seed"] is True and made["trainings"] == {"run": True}
    assert [row["in_the_set"] for row in _file_rows(own / "l6b_new_solves.jsonl")].count(True) == 1 and len(_file_rows(own / "l6b_new_solves.jsonl")) == 2
    sets = {model: _file_rows(own / f"l6b_training_{model}.jsonl") for model in MODELS}
    assert {model: sorted(row["origin"] for row in sets[model]) for model in MODELS} == {NEW_OWN: ["new", "pretraining"], OLD_OWN: ["deep", "pretraining"]}
    published = [[row["id"] for row in sets[model] if row["origin"] == "pretraining"] for model in MODELS]
    assert published[0] == published[1] and published[0][0] in [row["id"] for row in _file_rows(l5c / "l5_cont_training_again.jsonl")]
    assert not any(field in row for own_ in sets.values() for row in own_ if row["origin"] == "pretraining" for field in TEXT_FIELDS)
    assert made["the_orders"][NEW_OWN]["alike"] != []                                                         # two rows have two orders: four order seeds hold pairs alike, trained all the same
    # EIGHT trainings FROM L5f's smoke averaged `published`; two averaged adapters that ARE each of their members (kept, served and SAID); the search and two measurements at rank 64.
    own_calls = calls[before:]
    assert len(own_calls) == 8 and [call["seed"] for call in own_calls] == [3] * 8 and [call["more"]["start"] for call in own_calls] == [l5f / "adapters" / "published"] * 8
    assert kits[engines_before:] == [64] * 3
    for model in MODELS:
        averaged = summaries[f"ladder_l6b_average_{model}"]
        assert (averaged["read_back"]["is_the_mean"], averaged["read_back"]["is_none_of_them"], averaged["one_of_its_models_in_a_smoke_run"]) == (True, False, True)
        assert summaries[f"ladder_l6b_measure_{model}"]["adapter"] == str(own / "adapters" / model)
    # THE REPORT says A SMOKE RUN; the checks pass; its lines are no reads of the second round.
    report = summaries[REPORT]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["failed_checks"] == [] and report["smoke"] == smoke
    assert any("A SMOKE RUN: L5's smoke run reads INCONCLUSIVE" in line and "the search attempts at least 2 problems (2 NOT by the rule), one new solve is enough" in line
               and line.endswith("Nothing here is a read of the second round") for line in report["lines"])
    assert report["branches"]["primary"] in (ADD, NOT_SHOWN_AT_ONE, COST) and report["branches"]["whose_proofs"] in (TEACH_MORE, NOT_SHOWN_AT_ONE, TEACH_LESS)
    assert report["can_this_run_be_read"]["the_run"][THE_SEARCHED]["problems_of_the_rule"] == 2 and report["the_bar"]["run"] == str(l5c)
    # ---- A REAL RUN IS REFUSED FOR SUCH A ROUND 1 (its report is INCONCLUSIVE): the smoke-only variable is what lets a smoke run go on from it.
    options = entry.step_fields(entry.STAGES["ladder_l6b_smoke"][2])[2]["environment"]
    with monkeypatch.context() as patched:
        for name, value in options.items():
            patched.setenv(name, value)
        patched.setenv(SEEDS, "3")
        patched.setenv(RUN, "ladder_l6b_smoke_again")
        patched.delenv(SMOKE)
        with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no second round stands on such a run"):
            ladder_l6b.ladder_l6b_search_prepare(config)
    assert not (runs / "ladder_l6b_smoke_again").exists() or not any((runs / "ladder_l6b_smoke_again").iterdir())
    assert {path: _all(path) for path in read_only} == read_only


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone(monkeypatch):
    for name in (RUN, SMOKE, L4P_RUN, L5F_RUN, L5C_RUN, L6_RUN, ladder_l5.RUN_VARIABLE, ladder_l5_draws.RUN_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(SEEDS, "3")                                              # the task's seed, as its job gives it
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l6b"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l6b.STEPS) == STEPS and ladder_l6b.ENTRY_STAGE == "ladder_l6b" and len(STEPS) == 16
    # THE SECTION'S ORDER: the first step; the search; the two sets; `new_own`'s four trainings, its averaged adapter and its measurement; `old_own`'s; the report.
    assert STEPS[:3] == ["ladder_l6b_search_prepare", "ladder_l6b_search", "ladder_l6b_prepare"]
    assert STEPS[3:9] == [*(f"ladder_l6b_train_new_own_{order}" for order in (1, 2, 3, 4)), "ladder_l6b_average_new_own", "ladder_l6b_measure_new_own"]
    assert STEPS[9:] == [*(f"ladder_l6b_train_old_own_{order}" for order in (1, 2, 3, 4)), "ladder_l6b_average_old_own", "ladder_l6b_measure_old_own", "ladder_l6b_report"]
    # THE SMOKE STAGE: the same steps, under the runs L5f's smoke stage names WITHOUT its smoke-only variable, L5c's run, a run of its own, and ITS variable.
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l6b_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    of_l5f = entry.step_fields(entry.STAGES["ladder_l5_full_smoke"][2])[2]["environment"]
    own_options = smoke[0][2]["environment"]
    assert own_options == {**{name: value for name, value in of_l5f.items() if name != ladder_l5_full.SMOKE_VARIABLE}, L5C_RUN: "ladder_l5_cont_smoke", RUN: "ladder_l6b_smoke",
                           SMOKE: "2"}
    assert (own_options[ladder_l5.RUN_VARIABLE], own_options[L4P_RUN], own_options[L5F_RUN], own_options[ladder_l5_draws.RUN_VARIABLE]) == (
        "ladder_l5_deep_smoke", "ladder_l4_dose_r64_smoke", "ladder_l5_full_smoke", "ladder_l5_draws_smoke")
    assert not {ladder_l5.SMOKE_VARIABLE, ladder_l6.SMOKE_VARIABLE, L6_RUN, ladder_l5_stock.SMOKE_VARIABLE, ladder_l5_full.SMOKE_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE,
                    ladder_l5_cont.SMOKE_VARIABLE, ladder_l5_measure.SMOKE_VARIABLE} & set(own_options)
    # THE SETTING, as the package's own config has it: B, the bar, the one seed; and what it stands on, each by its own reader.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    assert whole["ladder_loop"]["l6b"] == {"start": "published", "bar": "again", "seeds": [3]} == CONFIG["ladder_loop"]["l6b"]
    own = ladder_l6b.the_settings(whole)
    of_l5c, of_l6 = ladder_l5_cont.the_settings(whole), ladder_l6.the_settings(whole)
    assert (own.start, own.bar, own.seeds, own.cont, own.full, own.draws, own.deep, own.cycle) == ("published", "again", (3,), of_l5c, of_l5c.full, of_l5c.draws, of_l5c.deep, of_l6)
    assert set(own.seeds) <= set(of_l5c.seeds) and (of_l6.attempts, of_l6.minimum, of_l6.sampling_place, of_l6.seed_of_the_search) == (64, 150, 52, 0)
    assert (own.draws.sampling, own.draws.attempts, own.draws.recipe) == ("reach", 32, "mean_constant")
    # THE RUN DIRECTORY IS ITS OWN; B and the bar live where L5c's own functions find them: no path is stated twice.
    assert ladder_l6b.run_name(own, 3) == "ladder_l6b_published_seed3" and ladder_l6b.the_best(whole, own) == ladder_l5_cont.the_bar(whole, of_l5c)
    best = ladder_l6b.the_best(whole, own)
    assert (best.name, best.check.rank, best.check.alpha, best.adapter.name, best.directory.name) == ("published", 64, 128, "published", "ladder_l5_draws_second_full_seed3")
    assert ladder_l6b.l5c_directory(whole, own).name == "ladder_l5_cont_second_full_seed3" and ladder_l6.search_seed(whole, of_l6) == 1052
    assert (ladder_l6b.average_marker("old_own"), ladder_l6b.REPORT_FILE, ladder_l6b.THE_BEST, ladder_l6b.THE_BAR, ladder_l6b.SEARCH_SET) == (
        "ladder_l6b_average_old_own", "report_ladder_l6b.json", "l5f_published", "l5c_again", "l6b_search")
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l6b"]' in path.read_text()] == ["gpu/ladder_l6b.py"]
    others = (set(ladder_l5.STEPS) | set(ladder_l6.STEPS) | set(ladder_l4_rows.STEPS) | set(ladder_l5_noise.STEPS) | set(ladder_l5_draws.STEPS) | set(ladder_l5_stock.STEPS)
              | set(ladder_l5_full.STEPS) | set(ladder_l5_cont.STEPS) | set(ladder_l5_measure.STEPS))
    assert not set(ladder_l6b.STEPS) & others and all(name.startswith("ladder_l6b_") for name in ladder_l6b.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l6b.STEPS for step in entry.STAGES[name])) == ["ladder_l6b", "ladder_l6b_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(SMOKE) == ["ladder_l6b_smoke"] == names(RUN) and SMOKE == "RLVR_LEAN_LADDER_L6B_SMOKE_PROBLEMS" and RUN == "RLVR_LEAN_LADDER_L6B_RUN"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l6b.py", "runner/entry.py"]
    # ONE BODY EACH, CALLED AND NOT COPIED: the search's problems by L6's own two functions and its body by L5's; L5's run by L5d's first-step reading; `second_full` by L5f's;
    # B by L5c's reading of L5f's run; L5d's run by L5f's; the trainings by L4t's body (handed B); the averaged adapters by L5n's; the measurements by the ceiling's, with no rung.
    assert ladder_l6b.train_a_model is ladder_l4_rows.train_a_model and ladder_l6b.measure_model is ladder_ceiling.measure_model and ladder_l6b.the_set is ladder_l5_draws.the_set
    own_stage = ladder_l6b.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == (
        "ladder_l6b", "ladder_l6b", "l6b", "L6b", "ladder_l6b_prepare", "ladder_loop.l6b.start")
    source = (PACKAGE / "gpu" / "ladder_l6b.py").read_text()
    for called in ("ladder_l6.solved_by_search_one(", "ladder_l6.the_second_search(", "ladder_l6.search_seed(", "ladder_l5.search_deep(", "ladder_l5_draws.drawn_by_the_seed(",
                   "ladder_l5_full.read_second_full(", "ladder_l5_cont.read_l5fs_run(", "ladder_l5_cont.the_bar(", "ladder_l5_full.read_l5ds_run(", "ladder_l5_noise.the_mean_adapter(",
                   "solve_rows(", "start_from=best", "rungs=False"):
        assert called in source, called
    assert "_episodes(" not in source and "LoRARequest" not in source and "mean_of_adapters" not in source and "read_round_one(" not in source and "ladder_l6_search_prepare(" not in source
    assert "ladder_l5_draws_second_full" not in source and "ladder_l5_cont_second_full" not in source and '"second_full"' not in source and '["ladder_loop"]["l6"]' not in source
    report_source = (PACKAGE / "reporting" / "ladder_l6b.py").read_text()
    assert "checks_of_the_models(" in report_source and "lines_of_the_checks(" in report_source and "against_the_start(" in report_source and "l5fs_run(" in report_source
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l6b.STEPS" in main and "ladder_l6b," in main
    # WHAT WAS CUT OUT OF L6 FOR IT IS CALLED BY L6's OWN FIRST STEP TOO: the problems search 1 solved, and search 2's problems with their sampling seed held.
    of_l6s = (PACKAGE / "gpu" / "ladder_l6.py").read_text()
    assert of_l6s.count("def solved_by_search_one(") == 1 and of_l6s.count("def the_second_search(") == 1
    assert of_l6s.count("solved = solved_by_search_one(found)") == 1 and of_l6s.count("second = the_second_search(config, cycle, found, read, of_round_one[\"solved\"], at_least)") == 1
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if "the_second_search(" in path.read_text()] == ["gpu/ladder_l6.py", "gpu/ladder_l6b.py"]
