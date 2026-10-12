"""L5f, the stage end to end on what L5's FINISHED run, L4p's run and L5d's OWN FINISHED RUN OF THE SEED stored: the
refusals of its first step (the seed, the escalation, the start, L5d's recipe, L4p's run and its `second_full`, L5d's run
of the seed, a set that is not that run's); the three sets, HELD to be L5d's stored ones id for id and order for order;
each set trained in four orders FROM `second_full`'s ADAPTER, its averaged adapter and its measurement on the goal set's
first sampling ALONE; the report's checks failing when they should; THE STACK at its edges; the smoke stage; the
registration; and L5d left writing what it wrote. Spec: docs/spec/ladder-loop.spec.md, "L5f: the three
kinds again, from the twice-pretrained model". The engine and Lean are scripted; nothing touches a GPU or the network.
The rules on hand-made rows are `test_ladder_l5_full.py`."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adapter_files import _numbers  # noqa: E402 - an adapter's weights file read by hand
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l4_stage import _on_the_gpu  # noqa: E402
from test_ladder_l5_draws_stage import _l5ns_report, _stage5d  # noqa: E402 - L5d's own stage, whose run of the seed this one stands on
from test_ladder_l5_draws_stage import _own_files as l5d_files  # noqa: E402 - what a seed of L5d leaves, frozen there
from test_ladder_l5_noise_stage import GOAL, NoiseSolver, _adapter_of, _trainings  # noqa: E402 - the goal set's first sampling scripted model by model; the trainings with the model taken out
from test_ladder_l5_stage import (  # noqa: E402, F401
    CONFIG,
    PACKAGE,
    SEEDS,
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
from test_ladder_l6_stage import DRAWN, _round_one, _smoke_world  # noqa: E402 - L4p's run and L5's stage run to its end on a base map of twelve problems; the smoke runs' world

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_draws import MEAN_CONSTANT  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_full import COSTS, ESCALATE, FURTHER_SEEDS, NO_FURTHER_SEED  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR, ORDERS  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_l4_rows, ladder_l5, ladder_l5_draws, ladder_l5_full, ladder_l5_noise, ladder_l5_stock, ladder_l6, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.adapter_files import compared, is_the_mean  # noqa: E402

RUN, SMOKE, L4P_RUN = ladder_l5_full.RUN_VARIABLE, ladder_l5_full.SMOKE_VARIABLE, ladder_l6.L4P_RUN_VARIABLE
VARIABLES = (RUN, SMOKE, L4P_RUN, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE)
TRAINED = [f"{model}_{order}" for model in MODELS for order in ORDERS]
STEPS = ["ladder_l5_full_prepare",
         *(step for model in MODELS for step in (*(f"ladder_l5_full_train_{model}_{order}" for order in ORDERS), f"ladder_l5_full_average_{model}", f"ladder_l5_full_measure_{model}")),
         "ladder_l5_full_report"]
OWN_FILES = sorted([
    *(f"adapters/{name}/{file}" for name in (*MODELS, *TRAINED) for file in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l5_full_loss_{name}.json" for name in TRAINED), *(f"l5_full_training_{name}.jsonl" for name in (*MODELS, *TRAINED)),
    *(f"{step}.done.json" for step in STEPS), "problems.jsonl", "report_ladder_l5_full.json",
    "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", "l4_stored_pre_r64_reach.jsonl", "l4_stored_second_full_reach.jsonl", "l5_held_aside_problems.jsonl"])
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "first_start", "rank", "models", "recipe", "branches", "inconclusive",
               "failed_checks", "can_this_run_be_read", "l4ps_own_checks_of_the_start_model", "of_this_seed", "figures", "counts", "the_stack", "the_primarys_difference_at_this_seed",
               "the_level", "l5d_at_this_seed", "against_the_start_models", "overfitting", "timed_out_in_lean", "what_follows", "the_seeds", "the_start_model", "the_search", "l5ds_run",
               "the_sets", "the_orders", "the_sampling", "training_sets", "training_recipe", "models_made", "smoke", "what_it_cannot_say", "attempts", "not_to_be_read", "not_built",
               "not_measured", "adapters"]
THEIRS = (4, 8, 12)                                                         # of the 32 attempts at the goal problem: L5d's `published`, `shallow` and `deep` at seed 3, from `pre_r64`
SUCCESSES = (8, 6, 10)                                                      # ... and this stage's three, from `second_full`: the mean of `shallow` and `deep` is exactly `published`
THE_RUN_CHECKS = ["each_set_is_l5s_rule_for_its_seed_made_again_and_held", "the_sets_are_the_ones_l5ds_run_of_the_seed_trained",
                  "the_start_adapter_is_there_at_its_rank_and_is_not_the_first_starts", "the_stored_first_sampling_of_the_start_model_is_the_settings"]


def _stage5f(config, monkeypatch, until=None, stage_name="ladder_l5_full", prepare_on_the_gpu=False):
    """The stage's steps in its order; the trainings and the averagings by their GPU path (the stand-in makes no adapter)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l5_full.STEPS:
            continue
        run = ladder_l5_full.STEPS[step]
        on_the_gpu = "_train_" in step or "_average_" in step or (prepare_on_the_gpu and step == STEPS[0])
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if on_the_gpu else run(config)
        if step == until:
            break
    return summaries


def _nothing_written(runs):
    return not [path for path in runs.iterdir() if (path.name.startswith("ladder_l5_draws_second_full") or path.name.startswith("ladder_l5_full")) and any(path.iterdir())]


@pytest.fixture
def full(deep, monkeypatch, tmp_path):  # noqa: F811
    """`deep`'s world with what this stage reads on the box: L4p's run (its `second_full` kept), a base map of twelve
    problems with a hand-made map of `pre_r64`, L5's own stage RUN TO ITS END there at its search's seed (four deep
    solves, so a set is EIGHT rows; the batch is ONE row), and L5d's OWN STAGE RUN TO ITS END at seed 3 under the
    averaged recipe (its three models close the goal in `THEIRS` of 32). Then this stage's task of SEED 3, which carries
    NO variable. `solver.measured`: every prompt of a measurement of the goal set's first sampling made from the fixture
    on; this stage's three close the goal as `SUCCESSES` says. `calls`, `kits`: this stage's own trainings and engines."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    config["training"]["effective_batch"] = 1
    first = _round_one(deep, monkeypatch, tmp_path, solved=tuple(DRAWN), l4p=True, settings={"problems": 4, "held_aside": 1, "minimum": 2})
    root, l4p = deep.store().root, deep.dose.store().root
    assert root.name == "ladder_l5_deep_pre_r64_seed0" and first["ladder_l5_report"]["inconclusive"] is False and first["ladder_l5_report"]["ok"] is True
    assert l4p.name == "ladder_l4_dose_pre_r64_seed0" and first["ladder_l5_prepare"]["rows_of_each_kind"] == 4
    calls, kits = [], []
    _trainings(monkeypatch, calls)
    solver = NoiseSolver(deep.solver, (*THEIRS, *SUCCESSES))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"]}),
        (solver, stand_in_parameters))[1]))
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l5_full"])        # THE STAGE names no variable
    monkeypatch.setenv(SEEDS, "3")
    runs = ladder_l5._runs(config)
    _l5ns_report(runs, AVERAGE_OF_FOUR)
    theirs = _stage5d(config, monkeypatch)
    l5d = runs / "ladder_l5_draws_pre_r64_seed3"
    assert theirs["ladder_l5_draws_report"]["inconclusive"] is False and sorted(name for name in _all(l5d) if not name.startswith("episodes_")) == l5d_files(MEAN_CONSTANT)
    del calls[:], kits[:]
    return SimpleNamespace(config=config, deep=deep, first=first, l5=root, l4p=l4p, l5d=l5d, theirs=theirs, calls=calls, kits=kits, solver=solver, runs=runs,
                           store=lambda: ladder_l5_full._store(config, ladder_l5_full.the_settings(config)))


def _file_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _as_text(own):
    return "".join(json.dumps(row) + "\n" for row in own)


def _changed(path, changed):
    """A file with `changed` of its text in its place; returns the text it held."""
    kept = Path(path).read_text()
    Path(path).write_text(changed(kept))
    return kept


def _swapped(text):
    """A stored set with its first two rows in each other's place."""
    own = [json.loads(line) for line in text.splitlines()]
    return _as_text([{**own[1], "row": 0}, {**own[0], "row": 1}, *own[2:]])


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_holds_its_sets_to_l5ds_run_trains_each_in_four_orders_from_second_full_measures_the_mean_on_g_alone_and_reads_the_stack(full, monkeypatch):  # noqa: F811
    config, calls, kits, solver, deep_ = full.config, full.calls, full.kits, full.solver, full.deep
    read_only = {path: _all(path) for path in (full.l5, full.l5d, full.l4p, deep_.map, deep_.l4t2, deep_.rank)}
    summaries = _stage5f(config, monkeypatch)
    store = full.store()
    # THE RUN DIRECTORY IS L5d's WITH THE START's NAME IN IT, beside the runs it reads, which were ONLY READ: file for file. L5d's run is what L5d's own test holds it to.
    assert store.root.name == "ladder_l5_draws_second_full_seed3" and store.root.parent == full.l5.parent and list(summaries) == STEPS == list(ladder_l5_full.STEPS)
    assert {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert sorted(name for name in _all(full.l5d) if not name.startswith("episodes_")) == l5d_files(MEAN_CONSTANT)

    # ---- the first step: the seed, THE ONE CHANGE (the start adapter, where L4p's run keeps it), L5d's run of the seed, the one sampling
    read = summaries["ladder_l5_full_prepare"]
    second_full, pre_r64 = full.l4p / "adapters" / "second_full", deep_.rank / "adapters" / "pre_r64"
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"], read["first_start"], read["models"], read["trained"], read["averaged"]) == (
        "l4", "pretrained on published proofs", "L5f", 3, "second_full", "pre_r64", list(MODELS), TRAINED, list(MODELS))
    assert (read["start_adapter"], read["start_run"], read["first_start_adapter"], read["first_start_run"]) == (str(second_full), str(full.l4p), str(pre_r64), str(deep_.rank))
    assert read["smoke"] is None and read["start_recipe"]["rank"] == 64 and read["start_recipe"]["alpha"] == 128
    assert read["the_seeds"] == {"what": read["the_seeds"]["what"], "seeds": [3], "more_seeds": [4, 5], "escalated": False, "fixed": [3]}
    of_the_start = read["the_start_model"]
    assert (of_the_start["model"], of_the_start["run"], of_the_start["adapter"], of_the_start["trained_from"], of_the_start["rank"], of_the_start["alpha"]) == (
        "second_full", str(full.l4p), str(second_full), "pre_r64", 64, 128)
    recorded = of_the_start["recorded_by_l4p"]
    assert (recorded["rank"], recorded["alpha"], recorded["adapter_saved"]["rank"], recorded["adapter_saved"]["alpha"], recorded["against_the_first_start"]["passes"]) == (64, 128, 64, 128, True)
    assert of_the_start["read_now"] is None and of_the_start["passes"] is True          # no GPU box here: it passes on what L4p's run recorded when it saved the adapter
    assert of_the_start["first_sampling"] == {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 1} and "ladder_l4_dose_r64" in of_the_start["task"]
    # L4p's OWN CHECKS OF THE MODEL are recorded and decide nothing.
    own_checks = of_the_start["l4ps_own_checks"]
    assert (own_checks["report_there"], own_checks["failed"], own_checks["inconclusive_by_its_own_report"]) == (True, [], False) and "DECIDING NOTHING" in own_checks["what"]
    of_theirs = read["l5ds_run"]
    assert (of_theirs["run"], of_theirs["held"]["passes"], of_theirs["held"]["sets"], of_theirs["held"]["trainings_of_theirs"]) == (str(full.l5d), True, 15, 12)
    assert of_theirs["at_the_first_step"]["read"] is True and "`ladder_l5_draws` for seed 3" in of_theirs["task"]
    assert (read["the_search"]["run"], read["the_search"]["seed"], read["the_search"]["deep_solves"], read["the_search"]["rows_of_each_kind"]) == (str(full.l5), 0, 4, 4)
    sampling = read["the_sampling"]
    assert (sampling["name"], sampling["episodes"], sampling["sampling_seed"], sampling["sets"]) == ("reach", 32, 1001, {model: f"l4_reach_{model}" for model in MODELS})
    assert (sampling["stored_start"], sampling["stored_first_start"]) == ("l4_stored_second_full_reach.jsonl", "l4_stored_pre_r64_reach.jsonl")
    assert read["recipe"]["name"] == "mean_constant" and read["recipe"]["adapter_seed"] == 3 and read["recipe"]["optimizer_steps"] == 8

    # ---- THE THREE SETS, AND EACH IN ITS FOUR ORDERS, ARE L5d's STORED ONES: row for row. A published row by id and place, NEVER with text.
    for name in (*MODELS, *TRAINED):
        own, theirs = store.read_rows(f"l5_full_training_{name}.jsonl"), _file_rows(full.l5d / f"l5_draws_training_{name}.jsonl")
        assert own == theirs and len(own) == 8, name
        assert not any(field in row for row in own if row["origin"] == "pretraining" for field in TEXT_FIELDS)
    sets = {model: store.read_rows(f"l5_full_training_{model}.jsonl") for model in MODELS}
    placed = lambda row: (row["id"], row["problem_id"], row.get("row_of_the_file"))      # noqa: E731
    for model in MODELS:
        for order in ORDERS:
            own, entry_ = store.read_rows(f"l5_full_training_{model}_{order}.jsonl"), read["training_sets"][f"{model}_{order}"]
            assert [placed(row) for row in own] == [placed(row) for row in training_order(sets[model], order)] and [row["row"] for row in own] == list(range(8))
            assert (entry_["order_seed"], entry_["model"], entry_["rows"], entry_["optimizer_steps"]) == (order, model, 8, 8)
        assert read["the_orders"][model] == {"the_order_of_the_tasks_seed": [3], "alike": []}
    assert {model: read["the_sets"][model]["rows_by_origin"] for model in MODELS} == {PUBLISHED: {"pretraining": 8}, SHALLOW: {"shallow": 4, "pretraining": 4}, DEEP: {"deep": 4, "pretraining": 4}}
    # BOTH START MODELS' FIRST SAMPLINGS are copied: `second_full`'s from L4p's run, `pre_r64`'s as L5d copies it.
    start_rows, first_rows = store.read_rows("l4_stored_second_full_reach.jsonl"), store.read_rows("l4_stored_pre_r64_reach.jsonl")
    assert [(row["problem_id"], row["episodes"]) for row in start_rows] == [(GOAL, 32)] == [(row["problem_id"], row["episodes"]) for row in first_rows]
    assert start_rows == _file_rows(full.l4p / "episodes_l4_reach_second_full_problems.jsonl") and first_rows == _file_rows(full.l5d / "l4_stored_pre_r64_reach.jsonl")

    # ---- TWELVE TRAININGS, model by model and order by order, each FROM `second_full`'s ADAPTER at rank 64, the recipe as it is with its rates recorded, with the task's seed
    assert [call["more"] for call in calls] == [{"start": second_full, "learning_rates": True}] * 12 and [call["seed"] for call in calls] == [3] * 12
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_draws_second_full_seed3_{name}" for name in TRAINED]
    assert all((call["lora"]["rank"], call["lora"]["alpha"]) == (64, 128) for call in calls)
    for call, name in zip(calls, TRAINED):
        train, loss = summaries[f"ladder_l5_full_train_{name}"], json.loads(store.path(f"l5_full_loss_{name}.json").read_text())
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in store.read_rows(f"l5_full_training_{name}.jsonl")]
        assert loss["rows_trained"] == [row["id"] for row in store.read_rows(f"l5_full_training_{name}.jsonl")] and train["adapter"] == str(store.root / "adapters" / name)
        assert train["learning_rate"]["decay"] is False and f"the ORDER SEED {name[-1]}" in train["order"] and len(loss["training_steps"]) == 8
        assert (train["adapter_saved"]["rank"], train["adapter_saved"]["alpha"]) == (64, 128)
    # THREE AVERAGED ADAPTERS, trained by nothing, by L5n's own function: every number the mean of its four's at that place; read back; none of them; at rank 64; and what
    # of each is not the start adapter's is counted against `second_full`'s FILE.
    for index, model in enumerate(MODELS):
        made, numbers = summaries[f"ladder_l5_full_average_{model}"], [4 * index + order for order in ORDERS]                    # the k-th training's adapter holds k and k / 2
        directory, of = store.root / "adapters" / model, [store.root / "adapters" / f"{model}_{order}" for order in ORDERS]
        assert (made["model"], made["recipe"], made["of"], made["adapter"], made["averaged"], made["check"]) == (model, "mean_constant", [f"{model}_{order}" for order in ORDERS], str(directory), True, "L5f")
        assert set(_numbers(directory / "adapter_model.safetensors", "base_model.model.layers.0.q_proj.lora_A.weight").tolist()) == {sum(numbers) / 4}
        assert made["read_back"] == is_the_mean(directory / "adapter_model.safetensors", [path / "adapter_model.safetensors" for path in of]) and made["read_back"]["passes"] is True
        assert made["adapter_saved"]["rank"] == 64 and made["one_of_its_models_in_a_smoke_run"] is False
        assert made["against_the_start_adapter"] == compared(directory / "adapter_model.safetensors", second_full / "adapter_model.safetensors")
    # THREE MEASUREMENTS, of the averaged adapters and of nothing else, each right after its adapter is made, the model server started for rank 64.
    assert len(solver.measured) == 6 and solver.measured[3:] == [(GOAL, 32, 1001, 1024)] * 3 and [kit["max_lora_rank"] for kit in kits] == [64] * 3 == [kit["rank"] for kit in kits]
    order_of = list(summaries)
    assert all(order_of.index(f"ladder_l5_full_measure_{model}") == order_of.index(f"ladder_l5_full_average_{model}") + 1 for model in MODELS)
    assert sorted(path.name for path in store.root.glob("episodes_*.done.json") if "_block_" not in path.name) == sorted(f"episodes_l4_reach_{model}.done.json" for model in MODELS)

    # ---- THE REPORT: the checks first, all passing; then the reads in the section's order
    report = summaries["ladder_l5_full_report"]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["failed_checks"] == [] and report["not_to_be_read"] == []
    assert json.loads(store.path("report_ladder_l5_full.json").read_text()) == report and store.done_summary("ladder_l5_full_report")["headline"] == report["headline"]
    assert (report["start"], report["first_start"], report["rank"], report["recipe"], report["check"]) == ("second_full", "pre_r64", 64, "mean_constant", "L5f")
    checks = report["can_this_run_be_read"]
    assert list(checks) == ["what", "the_run", *TRAINED, *MODELS] and list(checks["the_run"]) == THE_RUN_CHECKS
    assert all(entry_["passes"] is True for name in ("the_run", *TRAINED, *MODELS) for entry_ in checks[name].values())
    assert all(list(checks[model]) == ["its_numbers_are_the_mean_of_its_models_and_it_is_served", "lean_answered"] for model in MODELS)
    theirs_check = checks["the_run"]["the_sets_are_the_ones_l5ds_run_of_the_seed_trained"]
    assert (theirs_check["sets"], theirs_check["trainings_of_theirs"], theirs_check["sets_that_are_not_the_stored_ones"], theirs_check["sets_that_are_not_what_was_trained"]) == (15, 12, [], [])
    assert report["l4ps_own_checks_of_the_start_model"] == own_checks and report["the_start_model"] == of_the_start
    figures = report["figures"]
    assert [(figures[model]["all_of_g"]["successes"], figures[model]["all_of_g"]["attempts"]) for model in MODELS] == [(count, 32) for count in SUCCESSES]
    of_start, of_first = start_rows[0]["resolved"] / 32, first_rows[0]["resolved"] / 32
    assert (figures["the_start_models"]["second_full"]["per_attempt"], figures["the_start_models"]["pre_r64"]["per_attempt"]) == (round(of_start, 6), round(of_first, 6))
    # THE STACK, at its upper edge: the mean of `shallow` and `deep`, minus `published`, is exactly zero, which is AT OR ABOVE ZERO.
    stack = report["the_stack"]
    assert (stack["name"], stack["difference"], stack[PUBLISHED], stack[SHALLOW], stack[DEEP]) == (
        ESCALATE, 0.0, {"successes": 8, "attempts": 32}, {"successes": 6, "attempts": 32}, {"successes": 10, "attempts": 32})
    assert report["branches"] == {"the_stack": ESCALATE, "what_follows": FURTHER_SEEDS} and report["what_follows"]["seeds"] == [4, 5]
    assert report["of_this_seed"]["the_stack"] == 0.0 and report["of_this_seed"]["start"] == "second_full" and report["of_this_seed"]["inconclusive"] is False
    # THE LEVEL: each minus `second_full`'s stored first sampling, and minus the same kind of L5d's run of this seed; the highest beside the start models' and L5d's three.
    level, l5d = report["the_level"], report["l5d_at_this_seed"]
    assert level["minus_the_start"] == {model: round(count / 32 - of_start, 6) for model, count in zip(MODELS, SUCCESSES)}
    assert level["minus_the_same_kind_of_l5d"] == {model: round((count - theirs) / 32, 6) for model, count, theirs in zip(MODELS, SUCCESSES, THEIRS)} and level["l5d_read"] is True
    assert level["highest"] == {"model": DEEP, "per_attempt": round(10 / 32, 6)}
    assert level["beside"] == {"pre_r64": round(of_first, 6), "second_full": round(of_start, 6), "l5d": {model: round(theirs / 32, 6) for model, theirs in zip(MODELS, THEIRS)}}
    assert (l5d["read"], l5d["run"], l5d["the_primarys_difference"]) == (True, str(full.l5d), round((8 + 12) / 64 - 4 / 32, 6))
    # THE SOLVE RATE AGAINST BOTH START MODELS, paired by problem; overfitting against `second_full`; the timeouts with `second_full`'s stored set beside.
    for model, count in zip(MODELS, SUCCESSES):
        against = report["against_the_start_models"][model]
        assert list(against) == ["second_full", "pre_r64"]
        assert against["second_full"]["per_attempt"]["all_of_g"] == {"problems": 1, "model": round(count / 32, 6), "start": round(of_start, 6), "difference": round(count / 32 - of_start, 6)}
        assert against["pre_r64"]["per_attempt"]["all_of_g"] == {"problems": 1, "model": round(count / 32, 6), "start": round(of_first, 6), "difference": round(count / 32 - of_first, 6)}
        assert set(report["overfitting"][model]) == {"lost", "gained", "share_of_distinct_attempts", "long", "long_solved_by_the_start"}
        assert (report["overfitting"][model]["lost"], report["overfitting"][model]["gained"]) == (against["second_full"]["lost"], against["second_full"]["gained"])
    assert report["overfitting"]["the_start_model"] == {"share_of_distinct_attempts": of_the_start["distinct_attempts"]["goal"]["mean_share_distinct"]}
    assert report["timed_out_in_lean"]["checked_again"] is False and set(report["timed_out_in_lean"]["share"]) == {*MODELS, "second_full"}
    # THE READS STAND IN THE SECTION'S ORDER, after the checks and L4p's own checks (printed, deciding nothing): one line each, the last of them what follows.
    in_order = ("THE FIGURES, ", "THE STACK (the scout's read): ", "THE LEVEL, ", "THE SOLVE RATE AGAINST THE TWO START MODELS, ", "OVERFITTING, ", "TIMEOUTS, ", "WHAT FOLLOWS: ")
    said = [line.split(": ", 1)[1] for line in report["lines"]]
    assert [own.startswith(start) for own, start in zip(said[-len(in_order):], in_order)] == [True] * len(in_order)
    assert said[-len(in_order) - 1].startswith("L4p's OWN CHECKS OF `second_full`, PRINTED AND DECIDING NOTHING") and "its report failed none of them" in said[-len(in_order) - 1]
    assert all(own.startswith("CHECK, ") for own in said[1:-len(in_order) - 1]) and len(said) == 1 + 8 + 1 + len(in_order)
    assert said[2].startswith("CHECK, the sets are the ones L5d's run of this seed trained") and said[2].endswith(": PASS")
    assert said[3].startswith("CHECK, the adapter of `second_full` is there, is at its rank (64, alpha 128) and is not `pre_r64`'s: L4p's run saved it at rank 64 and alpha 128") and said[3].endswith(": PASS")
    assert all(line.startswith("l4 (pretrained on published proofs): ") for line in report["lines"]) and said[-1].startswith(f"WHAT FOLLOWS: {FURTHER_SEEDS}")
    assert said[0].startswith("L5f, THE THREE KINDS AGAIN, FROM THE TWICE-PRETRAINED MODEL. Seed 3: L5d's STAGE WITH ONE CHANGE, THE START ADAPTER")
    assert report["headline"].startswith("L5f (the stack; pretrained on published proofs) seed 3, from `second_full`: per 1,000 attempts on all of G `published` 250.00, `shallow` 187.50, "
                                         "`deep` 312.50 (`second_full` 375.00, `pre_r64` 375.00); THE STACK: THE MODEL'S OWN PROOFS DO NOT FALL BEHIND PUBLISHED ONES AFTER A SECOND PASS: "
                                         "ESCALATE (+0.00). What follows: ESCALATE")
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == sorted([*MODELS, *TRAINED]) and len(report["not_built"]) == 3

    # ---- THE STACK'S OTHER BRANCH A ONE-PROBLEM WORLD REACHES, the report built again (no step is made again): `deep` one success lower is 15.6 per 1,000 below zero: THEY COST
    rows_of_deep = store.path("episodes_l4_reach_deep_problems.jsonl")
    kept = _changed(rows_of_deep, lambda text: _as_text([{**row, "resolved": 9, "resolved_by_statement": 9} for row in map(json.loads, text.splitlines())]))
    costs = ladder_l5_full.ladder_l5_full_report(config)
    assert (costs["the_stack"]["name"], costs["the_stack"]["difference"], costs["branches"]) == (COSTS, round(-0.5 / 32, 6), {"the_stack": COSTS, "what_follows": NO_FURTHER_SEED})
    assert "seeds" not in costs["what_follows"] and "THE STACK: THEY COST AFTER A SECOND PASS AT THIS SEED (-15.62)" in costs["headline"] and costs["inconclusive"] is False
    rows_of_deep.write_text(kept)
    assert len(solver.measured) == 6 and len(calls) == 12
    # L5d's REPORT OF THE SEED GONE SINCE THE FIRST STEP: the level's second difference is NOT READ, and said; the stack is this run's own and is read; nothing is refused.
    for name in ("ladder_l5_draws_report.done.json", "report_ladder_l5_draws.json"):
        (full.l5d / name).rename(full.l5d / f"kept_{name}")
    alone = ladder_l5_full.ladder_l5_full_report(config)
    assert alone["branches"] == report["branches"] and alone["inconclusive"] is False and alone["the_level"]["minus_the_start"] == level["minus_the_start"]
    assert (alone["the_level"]["l5d_read"], alone["the_level"]["minus_the_same_kind_of_l5d"], alone["the_level"]["beside"]["l5d"]) == (False, {model: None for model in MODELS}, None)
    assert alone["l5d_at_this_seed"]["read"] is False and "L5d's run of this seed is not on the box, or not finished" in alone["l5d_at_this_seed"]["why"]
    assert any("Each minus the same kind of L5d's run of this seed" in line and "NOT READ (" in line for line in alone["lines"])
    for name in ("ladder_l5_draws_report.done.json", "report_ladder_l5_draws.json"):
        (full.l5d / f"kept_{name}").rename(full.l5d / name)
    assert ladder_l5_full.ladder_l5_full_report(config)["the_level"] == level


def test_a_failed_check_makes_the_run_inconclusive_and_nothing_else_is_said(full, monkeypatch):  # noqa: F811
    config = full.config
    report = _stage5f(config, monkeypatch)["ladder_l5_full_report"]
    store = full.store()
    assert report["inconclusive"] is False

    def failing(path, changed):
        kept = _changed(path, changed)
        failed = ladder_l5_full.ladder_l5_full_report(config)
        Path(path).write_text(kept)
        return failed

    # A SET OF L5d's RUN CHANGED SINCE THE FIRST STEP (two of its stored rows changed places): this run's set is no longer the stored one. THE RUN IS INCONCLUSIVE, no read is made.
    failed = failing(full.l5d / "l5_draws_training_deep_2.jsonl", _swapped)
    check = failed["can_this_run_be_read"]["the_run"]["the_sets_are_the_ones_l5ds_run_of_the_seed_trained"]
    assert failed["failed_checks"] == ["the run: the_sets_are_the_ones_l5ds_run_of_the_seed_trained"] and failed["inconclusive"] is True
    assert (check["sets_that_are_not_the_stored_ones"], check["first_that_differs"]["set"], check["first_that_differs"]["place"], check["sets_that_are_not_what_was_trained"]) == (["deep_2"], "deep_2", 0, [])
    assert failed["branches"] == {"the_stack": INCONCLUSIVE, "what_follows": NOT_READ} and "difference" not in failed["the_stack"]
    assert list(failed["figures"]) == ["what", "problems", "the_start_models", "measured_and_not_read"] and list(failed["counts"]) == ["what", "long_problems", "measured_and_not_read"]
    assert list(failed["the_level"]) == list(failed["against_the_start_models"]) == list(failed["overfitting"]) == list(failed["the_primarys_difference_at_this_seed"]) == ["measured_and_not_read"]
    assert failed["of_this_seed"]["figures"] is None and failed["of_this_seed"]["the_stack"] is None and failed["of_this_seed"]["inconclusive"] is True
    assert failed["lines"][-1].endswith("THIS RUN IS INCONCLUSIVE: the run: the_sets_are_the_ones_l5ds_run_of_the_seed_trained. Nothing else is said of it")
    assert not any("THE STACK (the scout's read)" in line or "WHAT FOLLOWS" in line or "THE LEVEL" in line for line in failed["lines"]) and "THIS RUN IS INCONCLUSIVE" in failed["headline"]
    # A TRAINING OF L5d's THAT WAS NOT OF THESE ROWS, each once (its loop's own record holds a row twice): the same check fails, and names the training.
    failed = failing(full.l5d / "l5_draws_loss_shallow_3.json", lambda text: json.dumps({**json.loads(text), "rows_trained": [*json.loads(text)["rows_trained"], json.loads(text)["rows_trained"][0]]}))
    assert failed["can_this_run_be_read"]["the_run"]["the_sets_are_the_ones_l5ds_run_of_the_seed_trained"]["sets_that_are_not_what_was_trained"] == ["shallow_3"] and failed["inconclusive"] is True
    # A SET OF THIS RUN THAT IS NOT IN ITS ORDER SEED'S ORDER fails L5d's own first check and this stage's: it is neither the rule's nor the stored one.
    failed = failing(store.path("l5_full_training_shallow_2.jsonl"), _swapped)
    assert failed["failed_checks"] == ["the run: each_set_is_l5s_rule_for_its_seed_made_again_and_held", "the run: the_sets_are_the_ones_l5ds_run_of_the_seed_trained"]
    assert failed["can_this_run_be_read"]["the_run"]["each_set_is_l5s_rule_for_its_seed_made_again_and_held"]["sets_not_in_their_order_seeds_order"] == ["shallow_2"]
    # THE START ADAPTER, as the first step recorded it: not held then, the check fails now.
    marker = store.path("ladder_l5_full_prepare.done.json")
    failed = failing(marker, lambda text: json.dumps({**json.loads(text), "the_start_model": {**json.loads(text)["the_start_model"], "passes": False}}))
    assert failed["failed_checks"] == ["the run: the_start_adapter_is_there_at_its_rank_and_is_not_the_first_starts"] and failed["branches"]["the_stack"] == INCONCLUSIVE
    # `second_full`'s STORED FIRST SAMPLING, as this run copied it, IS NOT THE SETTING'S (half the attempts a problem).
    failed = failing(store.path("l4_stored_second_full_reach.jsonl"), lambda text: _as_text([{**row, "episodes": 16} for row in map(json.loads, text.splitlines())]))
    assert failed["failed_checks"] == ["the run: the_stored_first_sampling_of_the_start_model_is_the_settings"] and failed["inconclusive"] is True
    # AN AVERAGED ADAPTER THAT IS ONE OF ITS MODELS (its step's stored read-back is made to say so) fails its check in a real run.
    marker = store.path("ladder_l5_full_average_published.done.json")
    failed = failing(marker, lambda text: json.dumps({**json.loads(text), "read_back": {**json.loads(text)["read_back"], "is_none_of_them": False, "passes": False}}))
    assert failed["failed_checks"] == ["`published`: its_numbers_are_the_mean_of_its_models_and_it_is_served"] and failed["branches"]["the_stack"] == INCONCLUSIVE
    # ... and with everything as it was the run reads again.
    assert ladder_l5_full.ladder_l5_full_report(config)["branches"] == report["branches"] == {"the_stack": ESCALATE, "what_follows": FURTHER_SEEDS}
    # A STEP NEEDS THE ONES BEFORE IT: with `published`'s fourth training, its averaged adapter and its measurement not done.
    for step in ("ladder_l5_full_train_published_4", "ladder_l5_full_average_published", "ladder_l5_full_measure_published"):
        store.path(f"{step}.done.json").unlink()
    with pytest.raises(RuntimeError, match="the averaged adapter `published` needs the step ladder_l5_full_train_published_4 of this run, which is not done"):
        _on_the_gpu(monkeypatch, ladder_l5_full.STEPS["ladder_l5_full_average_published"], config)
    with pytest.raises(RuntimeError, match="the measurement of L5f's model `published` needs the step ladder_l5_full_average_published of this run, which is not done"):
        ladder_l5_full.STEPS["ladder_l5_full_measure_published"](config)
    with pytest.raises(RuntimeError, match="L5f's report needs the step ladder_l5_full_train_published_4 of this run, which is not done"):
        ladder_l5_full.ladder_l5_full_report(config)


def test_as_on_the_gpu_the_first_step_reads_second_fulls_files_and_every_measurement_serves_its_own_models_adapter(full, monkeypatch):  # noqa: F811
    config, kits, runs = full.config, full.kits, full.runs
    second_full, pre_r64 = full.l4p / "adapters" / "second_full", full.deep.rank / "adapters" / "pre_r64"
    prepare = ladder_l5_full.STEPS["ladder_l5_full_prepare"]
    # THE ADAPTER IS NOT THERE (L4p's run recorded it, the files are gone): refused by the first step, naming L4p's task, before anything is written.
    second_full.rename(second_full.with_name("kept"))
    with pytest.raises(RuntimeError, match="adapters/second_full is not there: the adapter `second_full` is what every model here is trained from, and L4p's run keeps it. Queue the task of "
                                           "stage `ladder_l4_dose_r64` for seed 0 .* again .*nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    # IT IS `pre_r64`'s ADAPTER, number for number (whatever L4p's run recorded of it): not a second pass. Refused.
    shutil.copytree(pre_r64, second_full)
    with pytest.raises(RuntimeError, match="the adapter `second_full` at .*adapters/second_full, read back, has rank 64 and alpha 128 .*; 0 of its .* numbers are not `pre_r64`'s.*"
                                           "or nothing is trained from it. Nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    # AT ANOTHER RANK THAN `pre_r64`'s: refused too.
    _adapter_of(second_full, 32, 64, 9.0)
    with pytest.raises(RuntimeError, match="the adapter `second_full` at .*adapters/second_full, read back, has rank 32 and alpha 64 .*`pre_r64`: 64 and 128.*Nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    shutil.rmtree(second_full)
    shutil.rmtree(second_full.with_name("kept"))
    assert _nothing_written(runs)
    _adapter_of(second_full, 64, 128, 9.0)          # the scripted trainings of this world leave adapters alike: as on the box, `second_full`'s numbers are its own
    # WITH AN ADAPTER THAT IS A SECOND PASS'S the first step reads the files: there, read back at the rank, and numbers that are not `pre_r64`'s. The report's line says so.
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        summaries = _stage5f(config, patched, prepare_on_the_gpu=True)
    store = full.store()
    now = summaries["ladder_l5_full_prepare"]["the_start_model"]["read_now"]
    assert now["there"] is True and (now["read_back"]["rank"], now["read_back"]["alpha"]) == (64, 128) and summaries["ladder_l5_full_prepare"]["the_start_model"]["passes"] is True
    assert now["against_the_first_start"] == compared(second_full / "adapter_model.safetensors", pre_r64 / "adapter_model.safetensors") and now["against_the_first_start"]["elements_changed"] > 0
    report = summaries["ladder_l5_full_report"]
    assert report["inconclusive"] is False and any("CHECK, the adapter of `second_full` is there" in line and "; read now at rank 64 and alpha 128, " in line and line.endswith(": PASS")
                                                   for line in report["lines"])
    # AS ON THE GPU every measurement asks the model server for ITS model's adapter, the one this run keeps, with rank 64 as its largest.
    kept_here = store.root / "adapters"
    assert [kit["adapter"][2] for kit in kits] == [str(kept_here / model) for model in MODELS] and [kit["max_lora_rank"] for kit in kits] == [64] * 3
    assert summaries["ladder_l5_full_measure_deep"]["stand_in_engine"] is False


# ----------------------------------------------------------------------------------------------- the refusals
def test_the_first_step_refuses_before_anything_is_written(full, monkeypatch):  # noqa: F811
    config, runs = full.config, full.runs
    prepare = ladder_l5_full.ladder_l5_full_prepare
    # A SEED THE SETTING DOES NOT LIST; and a further seed while the escalation is not recorded.
    monkeypatch.setenv(SEEDS, "0")
    with pytest.raises(RuntimeError, match=r"seed 0 is none of this stage's seeds \[3\] \(nor of \[4, 5\], which wait for the escalation\).*\(ladder_loop.l5_full\). Nothing was written"):
        prepare(config)
    monkeypatch.setenv(SEEDS, "4")
    with pytest.raises(RuntimeError, match=r"seed 4 is one of the further seeds \[4, 5\], which run once the scout's read at \[3\] names the escalation and the setting records it "
                                           r"\(escalated is false\).*Nothing was written"):
        prepare(config)
    # ... and ONCE IT IS RECORDED a further seed waits for L5d's OWN RUN of that seed, which is not there: refused, naming the task to run.
    config["ladder_loop"]["l5_full"]["escalated"] = True
    with pytest.raises(RuntimeError, match="L5d's run of seed 4 cannot carry L5f: .*L5d's run of this seed is not on the box, or not finished. The three sets of this stage are HELD to be the "
                                           "ones that run trained, each in its four orders. Run the task of stage `ladder_l5_draws` for seed 4 .* to its end first; nothing was written"):
        prepare(config)
    config["ladder_loop"]["l5_full"]["escalated"] = False
    monkeypatch.setenv(SEEDS, "3")
    # THE SETTING ITSELF, by its one reader: another start, seeds that are not L5d's, an escalation that is no true or false, and a recipe of L5d's that is not the averaged one.
    settings, of_l5d = config["ladder_loop"]["l5_full"], config["ladder_loop"]["l5_draws"]
    for changed, said in (({"start": "pre_r64"}, "ladder_loop.l5_full.start is 'pre_r64': this stage is specified from `second_full` alone"),
                          ({"seeds": [9]}, r"ladder_loop.l5_full.seeds is \[9\] and more_seeds \[4, 5\]: distinct whole numbers of at least 0, at least one in seeds, each a seed of L5d's"),
                          ({"more_seeds": [3, 4]}, r"ladder_loop.l5_full.seeds is \[3\] and more_seeds \[3, 4\]"), ({"seeds": []}, r"ladder_loop.l5_full.seeds is \[\]"),
                          ({"escalated": "yes"}, "ladder_loop.l5_full.escalated is 'yes': true once the scout's read names the escalation, false until then")):
        kept = dict(settings)
        settings.update(changed)
        with pytest.raises(ValueError, match=said):
            prepare(config)
        settings.clear()
        settings.update(kept)
    for recipe in ("constant", "decay"):
        of_l5d["recipe"] = recipe
        with pytest.raises(ValueError, match=f"ladder_loop.l5_draws.recipe is `{recipe}`: this stage trains each set in four orders and measures the averaged adapter .`mean_constant`."):
            prepare(config)
    of_l5d["recipe"] = "mean_constant"
    # THE SMOKE-ONLY VARIABLE without a run directory of the task's naming.
    monkeypatch.setenv(SMOKE, "1")
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_FULL_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_FULL_RUN. and on the "
                                           "fixtures' base map. A run in the seed's own directory reads a run of L5 whose checks passed. Nothing was written"):
        prepare(config)
    monkeypatch.delenv(SMOKE)
    # L5's RUN not to be read: refused by L5's own reading, in this stage's words.
    report_file = full.l5 / "report_ladder_l5.json"
    kept = _changed(report_file, lambda text: json.dumps({**json.loads(text), "inconclusive": True, "failed_checks": ["`shallow`: the_training_ran"]}))
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .*: no model here stands beside such a run"):
        prepare(config)
    report_file.write_text(kept)
    # ---- L4p's RUN, where `second_full` lives: not there; its models recorded at another rank; an adapter it recorded as no different from `pre_r64`'s.
    full.l4p.rename(full.l4p.with_name("kept_l4p"))
    with pytest.raises(RuntimeError, match="`second_full` is NOT THERE .*: it is the model every model of this stage is trained FROM, kept by L4p's run .*ladder_l4_dose_pre_r64_seed0.. Run "
                                           "the task of stage `ladder_l4_dose_r64` for seed 0 .* to its end first; nothing was written"):
        prepare(config)
    full.l4p.with_name("kept_l4p").rename(full.l4p)
    marker = full.l4p / "ladder_l4_dose_prepare.done.json"
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "start_recipe": {**json.loads(text)["start_recipe"], "rank": 32, "alpha": 64}}))
    with pytest.raises(RuntimeError, match="L4p's run ladder_l4_dose_pre_r64_seed0 recorded rank 32 and alpha 64 for its models.*`pre_r64` is rank 64 and alpha 128, and every model here is "
                                           "trained and served at that rank. Nothing was written"):
        prepare(config)
    marker.write_text(kept)
    marker = full.l4p / "ladder_l4_dose_train_second_full.done.json"
    same = lambda text: json.dumps({**json.loads(text), "against_the_start_adapter": {name: {**own, "elements_changed": 0} for name, own in json.loads(text)["against_the_start_adapter"].items()}})      # noqa: E731
    kept = _changed(marker, same)
    with pytest.raises(RuntimeError, match="L4p's run ladder_l4_dose_pre_r64_seed0 recorded that no number of the adapter `second_full` differs from `pre_r64`'s: it is not a second pass"):
        prepare(config)
    marker.write_text(kept)
    # ---- L5d's RUN OF THE SEED: INCONCLUSIVE; of another recipe; without a stored set; and A SET THAT IS NOT THIS RUN's (two of its rows changed places): each refused.
    report_file = full.l5d / "report_ladder_l5_draws.json"
    for changed, why in (({"inconclusive": True, "failed_checks": ["`deep`: lean_answered"]}, "it reads INCONCLUSIVE"), ({"recipe": "constant"}, "its recipe is `constant` and this run's `mean_constant`"),
                         ({"ok": False}, "its report is not to be read")):
        kept = _changed(report_file, lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], **changed}}))
        with pytest.raises(RuntimeError, match=f"L5d's run of seed 3 cannot carry L5f: .*{why}.*Run the task of stage `ladder_l5_draws` for seed 3 .* to its end first; nothing was written"):
            prepare(config)
        report_file.write_text(kept)
    stored = full.l5d / "l5_draws_training_shallow_4.jsonl"
    stored.rename(stored.with_name("kept.jsonl"))
    with pytest.raises(RuntimeError, match=r"ladder_l5_draws_pre_r64_seed3 does not hold \['l5_draws_training_shallow_4.jsonl'\]: the sets L5d's run of seed 3 stored"):
        prepare(config)
    stored.with_name("kept.jsonl").rename(stored)
    kept = _changed(full.l5d / "l5_draws_training_deep_2.jsonl", _swapped)
    with pytest.raises(RuntimeError, match=r"the sets \['deep_2'\] are not the ones L5d's run of this seed stored .`deep_2`: 8 rows here and 8 there; first that differs at place 0: .*"
                                           r"Refused \(seed 3; L5's run ladder_l5_deep_pre_r64_seed0; L5d's run ladder_l5_draws_pre_r64_seed3\). Nothing was written"):
        prepare(config)
    (full.l5d / "l5_draws_training_deep_2.jsonl").write_text(kept)
    assert _nothing_written(runs)
    # WITH EVERYTHING AS IT WAS the first step passes. L4p's OWN CHECK OF `second_full` FAILING (its report reads the model INCONCLUSIVE) IS RECORDED AND REFUSES NOTHING.
    report_file = full.l4p / "report_ladder_l4_dose.json"
    whole = json.loads(report_file.read_text())
    its_checks = whole["models"]["second_full"]["can_this_model_be_read"]
    first_check = next(name for name, own in its_checks.items() if isinstance(own, dict) and "passes" in own)
    report_file.write_text(json.dumps({**whole, "models": {**whole["models"], "second_full": {**whole["models"]["second_full"], "can_this_model_be_read": {
        **its_checks, first_check: {**its_checks[first_check], "passes": False}}}}}))
    read = prepare(config)
    assert read["the_start_model"]["l4ps_own_checks"]["failed"] == [first_check] and read["the_start_model"]["l4ps_own_checks"]["inconclusive_by_its_own_report"] is True
    assert read["the_start_model"]["passes"] is True and full.store().is_done("ladder_l5_full_prepare")


def test_the_steps_run_through_the_gpu_entry_point(full, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(milestone2, "load_config", lambda path: full.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    # In the stand-in's world no adapter is trained and none is averaged, so the report says INCONCLUSIVE and reads no stack.
    assert json.loads((tmp_path / "ladder_l5_full_average_published.json").read_text())["averaged"] is False
    report = json.loads((tmp_path / "ladder_l5_full_report.json").read_text())
    assert full.store().is_done(ladder_l5_full.REPORT) and report["inconclusive"] is True and report["stand_in_engine"] is True
    assert report["branches"] == {"the_stack": INCONCLUSIVE, "what_follows": NOT_READ}


# ---------------------------------------------------------------------------------------- the smoke-only path
def test_the_smoke_stage_goes_on_from_the_smoke_runs_of_l5_l4p_and_l5d_trains_three_sets_of_two_rows_from_the_smoke_second_full_and_says_so(deep, monkeypatch):  # noqa: F811
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    first = _smoke_world(deep, monkeypatch)
    config, runs = deep.config, ladder_l5._runs(deep.config)
    of_l5, failed = runs / "ladder_l5_deep_smoke", first["ladder_l5_report"]["failed_checks"]
    calls, measured = [], []
    _trainings(monkeypatch, calls, value=lambda number: 1.0)          # AS ON THE BOX: one optimizer step on the same rows, whatever their order, is ONE training: adapters alike
    solver = NoiseSolver(deep.solver, (*THEIRS, *SUCCESSES))          # L5d's smoke run's three models, then this stage's three
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (measured.append(given["vllm"]["max_lora_rank"]), (solver, stand_in_parameters))[1]))
    options = entry.step_fields(entry.STAGES["ladder_l5_full_smoke"][2])[2]["environment"]
    # ---- L4p's SMOKE RUN, under the name the smoke stage gives it by L6's variable: this world's run of L4p, moved there (it is only read).
    of_l4p = runs / options[L4P_RUN]
    deep.dose.store().root.rename(of_l4p)
    monkeypatch.setenv(L4P_RUN, options[L4P_RUN])
    monkeypatch.setenv(SEEDS, "3")
    # ---- A REAL RUN IS REFUSED FOR SUCH A RUN OF L5 (its report is INCONCLUSIVE), in the seed's own directory and in one the task names.
    assert failed != []
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no model here stands beside such a run"):
        ladder_l5_full.ladder_l5_full_prepare(config)
    monkeypatch.setenv(RUN, options[RUN])
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE"):
        ladder_l5_full.ladder_l5_full_prepare(config)
    # THE SMOKE-ONLY VARIABLE on a map that is no fixture's: refused, and nothing is written.
    monkeypatch.setenv(SMOKE, options[SMOKE])
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_FULL_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_FULL_RUN. and on the "
                                           "fixtures' base map. A run on the pool's problems reads a run of L5 whose checks passed. Nothing was written"):
        ladder_l5_full.ladder_l5_full_prepare(config)
    marker.write_text(json.dumps(recorded))
    # WITHOUT L5d's SMOKE RUN of this recipe the first step REFUSES, naming the task: the sets are held to that run's.
    _l5ns_report(runs, AVERAGE_OF_FOUR)
    monkeypatch.setenv(ladder_l5_draws.RUN_VARIABLE, options[ladder_l5_draws.RUN_VARIABLE])
    with pytest.raises(RuntimeError, match="L5d's run of seed 3 cannot carry L5f: .*ladder_l5_draws_smoke_mean_constant does not hold .*Run the task of stage `ladder_l5_draws` for seed 3 "
                                           ".*stage `ladder_l5_draws_smoke`. to its end first; nothing was written"):
        ladder_l5_full.ladder_l5_full_prepare(config)
    assert _nothing_written(runs)
    # ---- L5d's SMOKE RUN under the averaged recipe, by L5d's own smoke stage: the run whose sets this stage's are held to.
    monkeypatch.delenv(RUN)
    monkeypatch.delenv(SMOKE)
    monkeypatch.setenv(ladder_l5_draws.SMOKE_VARIABLE, "1")
    of_l5d = _stage5d(config, monkeypatch, stage_name="ladder_l5_draws_smoke")
    theirs = runs / "ladder_l5_draws_smoke_mean_constant"
    assert of_l5d["ladder_l5_draws_report"]["inconclusive"] is False and sorted(name for name in _all(theirs) if not name.startswith("episodes_")) == l5d_files(MEAN_CONSTANT)
    monkeypatch.delenv(ladder_l5_draws.SMOKE_VARIABLE)                  # THIS stage's smoke task names L5d's RUN and L4p's, and neither L5d's smoke-only variable nor L5's
    assert ladder_l5_draws.SMOKE_VARIABLE not in options and ladder_l5.SMOKE_VARIABLE not in options and ladder_l6.SMOKE_VARIABLE not in options
    monkeypatch.setenv(RUN, options[RUN])
    monkeypatch.setenv(SMOKE, options[SMOKE])
    # ---- THE SMOKE RUN: its own directory, the fixtures' map, the variable. L5's run is read all the same, and that is RECORDED.
    read_only = {path: _all(path) for path in (of_l5, theirs, of_l4p, deep.map, deep.l4t2, deep.rank)}
    del calls[:]
    summaries = _stage5f(config, monkeypatch, stage_name="ladder_l5_full_smoke")
    store = ladder_l5_full._store(config, ladder_l5_full.the_settings(config))
    assert store.root.name == "ladder_l5_full_smoke" and list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    read = summaries["ladder_l5_full_prepare"]
    smoke = read["smoke"]
    assert (smoke["problems_at_least"], smoke["l5_read_though_inconclusive"], smoke["l5s_failed_checks"]) == (1, True, failed) and read["seed"] == 3
    assert (read["the_search"]["run"], read["the_search"]["deep_solves"], read["the_search"]["inconclusive"]) == (str(of_l5), 1, True)
    # THE START ADAPTER IS THE SMOKE `second_full`, in the run L6's variable names; the sets are L5d's SMOKE run's: D = 1, three sets of TWO rows.
    assert (read["start_adapter"], read["start_run"], read["l5ds_run"]["run"]) == (str(of_l4p / "adapters" / "second_full"), str(of_l4p), str(theirs))
    assert {model: read["the_sets"][model]["rows"] for model in MODELS} == {model: 2 for model in MODELS} and read["l5ds_run"]["held"]["passes"] is True
    for name in (*MODELS, *TRAINED):
        assert store.read_rows(f"l5_full_training_{name}.jsonl") == _file_rows(theirs / f"l5_draws_training_{name}.jsonl")
    # TWELVE trainings of two optimizer steps from the smoke `second_full`, three averaged adapters that ARE each of their four (kept, served and SAID), three measurements at rank 64.
    assert len(calls) == 12 and [call["seed"] for call in calls] == [3] * 12 and all(call["more"]["start"] == of_l4p / "adapters" / "second_full" for call in calls)
    assert solver.measured[-3:] == [(GOAL, 32, 1001, 1024)] * 3 and measured[-3:] == [64] * 3
    for model in MODELS:
        made = summaries[f"ladder_l5_full_average_{model}"]
        assert (made["read_back"]["is_the_mean"], made["read_back"]["is_none_of_them"], made["one_of_its_models_in_a_smoke_run"]) == (True, False, True)
    # THE REPORT says A SMOKE RUN; the checks pass; its line THE STACK is no read of the stack.
    report = summaries["ladder_l5_full_report"]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["smoke"] == smoke
    assert any(f"A SMOKE RUN: L5's smoke run reads INCONCLUSIVE ({', '.join(failed)}) and was read all the same; one deep solve is enough" in line and
               line.endswith("Nothing here is a read of the stack") for line in report["lines"])
    assert report["l5d_at_this_seed"]["read"] is True and report["l5d_at_this_seed"]["run"] == str(theirs) and report["the_stack"]["name"] == ESCALATE
    # WITHOUT THE VARIABLE nothing is taken loosely.
    monkeypatch.delenv(SMOKE)
    assert ladder_l5_full.smoke_run() is False


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_full"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l5_full.STEPS) == STEPS and ladder_l5_full.ENTRY_STAGE == "ladder_l5_full" and len(STEPS) == 20
    # L5d's ORDER: model by model its four trainings, its averaged adapter and its measurement; then the report.
    assert STEPS[1:7] == [*(f"ladder_l5_full_train_published_{order}" for order in (1, 2, 3, 4)), "ladder_l5_full_average_published", "ladder_l5_full_measure_published"]
    used_by_l5d = [step.replace("ladder_l5_draws_", "ladder_l5_full_") for _, step, _ in map(entry.step_fields, entry.STAGES["ladder_l5_draws"])
                   if step in ladder_l5_draws.STEPS and step.rpartition("_train_")[2] not in MODELS]
    assert used_by_l5d == STEPS                                      # the steps L5d's averaged recipe uses, in its order, under this stage's names
    # THE SMOKE STAGE: the same steps, under the runs L5d's smoke stage names (WITHOUT its smoke-only variable, and without L5's), L4p's smoke run by L6's variable, a run of
    # its own, and ITS smoke-only variable.
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_full_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    of_l5d = entry.step_fields(entry.STAGES["ladder_l5_draws_smoke"][2])[2]["environment"]
    of_l6 = entry.step_fields(entry.STAGES["ladder_l6_cycle_smoke"][2])[2]["environment"]
    assert smoke[0][2]["environment"] == {**{name: value for name, value in of_l5d.items() if name != ladder_l5_draws.SMOKE_VARIABLE}, L4P_RUN: of_l6[L4P_RUN], RUN: "ladder_l5_full_smoke",
                                          SMOKE: "1"}
    assert smoke[0][2]["environment"][ladder_l5.RUN_VARIABLE] == "ladder_l5_deep_smoke" and smoke[0][2]["environment"][L4P_RUN] == "ladder_l4_dose_r64_smoke"
    assert not {ladder_l5.SMOKE_VARIABLE, ladder_l6.SMOKE_VARIABLE, ladder_l5_stock.SMOKE_VARIABLE} & set(smoke[0][2]["environment"])
    # THE SETTING, as the package's own config has it: the start, the scout's seed, two further ones that wait for the escalation; every seed one of L5d's.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    assert whole["ladder_loop"]["l5_full"] == {"start": "second_full", "seeds": [3], "more_seeds": [4, 5], "escalated": False} == CONFIG["ladder_loop"]["l5_full"]
    own = ladder_l5_full.the_settings(whole)
    assert (own.start, own.seeds, own.more, own.escalated, own.draws, own.deep) == ("second_full", (3,), (4, 5), False, ladder_l5_draws.the_settings(whole), ladder_l5.the_settings(whole))
    assert set((*own.seeds, *own.more)) <= set(own.draws.seeds) and own.draws.recipe == "mean_constant" and (own.draws.sampling, own.draws.attempts, own.draws.start) == ("reach", 32, "pre_r64")
    # THE RUN DIRECTORY IS L5d's WITH THE START's NAME IN IT; and `second_full` lives where L6's own function finds L4p's run: no path is stated twice.
    assert [ladder_l5_full.run_name(own, seed) for seed in (3, 5)] == ["ladder_l5_draws_second_full_seed3", "ladder_l5_draws_second_full_seed5"]
    assert ladder_l5_draws.run_name(own.draws, 3) != ladder_l5_full.run_name(own, 3) and ladder_l5_full.l4p_directory(whole) == ladder_l6.l4p_directory(whole, ladder_l6.the_settings(whole))
    start = ladder_l5_full.the_start(whole, own)
    assert (start.name, start.directory, start.adapter, start.check) == ("second_full", ladder_l5_full.l4p_directory(whole), ladder_l5_full.l4p_directory(whole) / "adapters" / "second_full",
                                                                         ladder_l5_full.the_first_start(whole, own).check)
    assert (start.check.rank, start.check.alpha, start.directory.name) == (64, 128, "ladder_l4_dose_pre_r64_seed0")
    assert (ladder_l5_full.average_marker("deep"), ladder_l5_full.REPORT_FILE) == ("ladder_l5_full_average_deep", "report_ladder_l5_full.json")
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l5_full"]' in path.read_text()] == ["gpu/ladder_l5_full.py"]
    others = set(ladder_l5.STEPS) | set(ladder_l6.STEPS) | set(ladder_l4_rows.STEPS) | set(ladder_l5_noise.STEPS) | set(ladder_l5_draws.STEPS) | set(ladder_l5_stock.STEPS)
    assert not set(ladder_l5_full.STEPS) & others and all(name.startswith("ladder_l5_full_") for name in ladder_l5_full.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l5_full.STEPS for step in entry.STAGES[name])) == ["ladder_l5_full", "ladder_l5_full_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(SMOKE) == ["ladder_l5_full_smoke"] and names(RUN) == ["ladder_l5_cont_smoke", "ladder_l5_full_smoke", "ladder_l5_measure_smoke", "ladder_l6b_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L5_FULL_SMOKE" and RUN == "RLVR_LEAN_LADDER_L5_FULL_RUN"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l5_full.py", "runner/entry.py"]
    # ONE BODY EACH: its trainings are L4t's own training body (handed the start model), its measurements the ceiling's own measuring body with no rung, its averaged adapters
    # L5n's own function; L5's run is read and the three sets drawn by L5d's own first-step reading; `second_full` by L6's own reading; L5d's run by L5s's.
    assert ladder_l5_full.train_a_model is ladder_l4_rows.train_a_model and ladder_l5_full.measure_model is ladder_ceiling.measure_model and ladder_l5_full.the_set is ladder_l5_draws.the_set
    own_stage = ladder_l5_full.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == (
        "ladder_l5_full", "ladder_l5_full", "l5_full", "L5f", "ladder_l5_full_prepare", "ladder_loop.l5_full.start")
    source = (PACKAGE / "gpu" / "ladder_l5_full.py").read_text()
    for called in ("ladder_l5_draws.drawn_by_the_seed(", "ladder_l6.read_the_stored_model(", "ladder_l6.l4p_directory(", "ladder_l5_noise.the_mean_adapter(", "ladder_l5_stock.read_l5d(",
                   "start_from=the_start(config, full)", "rungs=False"):
        assert called in source, called
    assert "_episodes(" not in source and "LoRARequest" not in source and "mean_of_adapters" not in source and "read_the_search(" not in source and "solve_rows(" not in source
    assert "ladder_l4_dose_pre_r64" not in source and "adapters/second_full" not in source and '"second_full"' not in source
    report_source = (PACKAGE / "reporting" / "ladder_l5_full.py").read_text()
    assert "ladder_l5_draws.the_checks(" in report_source and "lines_of_the_checks(" in report_source and "against_the_start(" in report_source
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l5_full.STEPS" in main and "ladder_l5_full," in main
    # L4t's TRAINING BODY takes the start model it is handed, and finds it itself otherwise: every other caller hands it none.
    callers = sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if "start_from=" in path.read_text())
    # ... but L5c's, which hands it L5f's own start or L5f's averaged `published`, and L6b's, which hands it that averaged `published`
    assert callers == ["gpu/ladder_l5_cont.py", "gpu/ladder_l5_full.py", "gpu/ladder_l6b.py"]
    body = (PACKAGE / "gpu" / "ladder_l4_rows.py").read_text()
    assert "start_from: Start | None = None" in body and "of_the_start = start_from or " in body
