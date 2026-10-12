"""L5d, the stage end to end on what L5's FINISHED run and the runs that task read stored: the refusals of its first step
(the seed, the recipe held to L5n's report, L5's run); the three sets, WHICH ARE WHAT A FURTHER SEED OF L5 BUILDS FOR THAT
SEED; each recipe's steps (the recipe as it is, the decay, four orders and their averaged adapter), the steps of another
recipe making nothing; every model measured on the goal set's first sampling ALONE; a seed's report and the read over the
fresh seeds from the runs that are there; a run that keeps its recipe; the smoke stage and its smoke-only path; the
registration; and L5, L5n and L6 left writing what they wrote. Spec: docs/spec/ladder-loop.spec.md, "L5d:
the three kinds again, over fresh draws, measured on the goal set's first sampling". The engine and Lean are scripted;
nothing touches a GPU or the network. The rules and the read over the seeds on hand-made rows are `test_ladder_l5_draws.py`."""

import json
import shutil
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adapter_files import _numbers, _write  # noqa: E402 - an adapter's weights file written and read by hand
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l4_stage import CHANGED, _on_the_gpu  # noqa: E402
from test_ladder_l5_noise_stage import OWN_FILES as L5N_FILES  # noqa: E402 - what L5n leaves, frozen there
from test_ladder_l5_noise_stage import REPORT_KEYS as L5N_REPORT_KEYS  # noqa: E402
from test_ladder_l5_noise_stage import STEPS as L5N_STEPS  # noqa: E402
from test_ladder_l5_noise_stage import GOAL, NoiseSolver, _stage5n, _trainings  # noqa: E402 - the goal set's first sampling scripted model by model; the trainings with the model taken out
from test_ladder_l5_seeds import FURTHER_FILE, NOT_MADE_AGAIN  # noqa: E402 - what a further seed of L5 leaves, frozen there
from test_ladder_l5_stage import OWN_FILES as L5_FILES  # noqa: E402 - what L5 leaves at the search's own seed, frozen there
from test_ladder_l5_stage import STEPS as L5_STEPS  # noqa: E402
from test_ladder_l5_stage import (  # noqa: E402, F401
    CONFIG,
    PACKAGE,
    SEEDS,
    _file_rows,
    _stage5,
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
from test_ladder_l6_stage import DRAWN, HELD, _round_one, _smoke_world, _stage6, cycle  # noqa: E402, F401 - L5's stage run to its end on a base map of twelve problems; the smoke runs' world; L6's own
from test_ladder_l6_stage import OWN_FILES as L6_FILES  # noqa: E402 - what L6 leaves, frozen there
from test_ladder_l6_stage import STEPS as L6_STEPS  # noqa: E402

from rlvr_lean.domain.ladder_round import l5_draws  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.dose import learning_rates  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import DEEP, MODELS, PUBLISHED, SHALLOW, hold_the_sets  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_draws import (  # noqa: E402
    BEAT,
    CONSTANT,
    DECAY,
    L6_RUNS,
    MEAN_CONSTANT,
    THE_PRIMARY,
    its_trainings,
    primary,
    trained_models,
)
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR, ORDERS, THE_RECIPES_OWN, WITH_THE_DECAY  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_l3c, ladder_l4_rows, ladder_l4_start, ladder_l5, ladder_l5_draws, ladder_l5_noise, ladder_l6, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.adapter_files import compared, is_the_mean  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY  # noqa: E402
from rlvr_lean.reporting.ladder_l5_draws import lines_over_the_seeds  # noqa: E402
from rlvr_lean.tools import ladder_l5_draws as tool  # noqa: E402

RUN, SMOKE = ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE
STEPS = ["ladder_l5_draws_prepare",
         *(step for model in MODELS for step in (f"ladder_l5_draws_train_{model}", *(f"ladder_l5_draws_train_{model}_{order}" for order in ORDERS),
                                                 f"ladder_l5_draws_average_{model}", f"ladder_l5_draws_measure_{model}")), "ladder_l5_draws_report"]
TRAININGS = tuple(step for step in STEPS if "_train_" in step)
AVERAGES = tuple(step for step in STEPS if "_average_" in step)
RATE, WARMUP = CONFIG["training"]["learning_rate"], CONFIG["training"]["warmup_steps"]
# What each model writes on the goal problem, in the order the models are measured, seed by seed: it closes the goal in its first so many of the 32 attempts. The primary's
# difference (the mean of `shallow` and `deep`, minus `published`) is 6, 5 and 6 of 32 at the seeds 3, 4 and 5: above zero over the three.
SUCCESSES = {3: (4, 8, 12), 4: (6, 10, 12), 5: (8, 12, 16)}
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "rank", "models", "recipe", "the_recipe", "branches", "inconclusive",
               "failed_checks", "can_this_run_be_read", "of_this_seed", "figures", "counts", "the_primarys_difference_at_this_seed", "over_the_fresh_seeds", "the_seeds_of_l5",
               "timed_out_in_lean", "the_seeds", "the_search", "the_sets", "the_orders", "the_sampling", "training_sets", "training_recipe", "models_made", "smoke",
               "what_it_cannot_say", "attempts", "not_to_be_read", "not_measured", "adapters"]


def _used(recipe):
    """The steps of the stage that make something under a recipe, in the stage's order: every other step makes nothing."""
    trained = {f"ladder_l5_draws_train_{name}" for name in trained_models(recipe)}
    return [step for step in STEPS if (step in trained if "_train_" in step else recipe == MEAN_CONSTANT if "_average_" in step else True)]


def _own_files(recipe):
    """What a seed's run leaves under a recipe, the measured sets' own files aside."""
    trained = trained_models(recipe)
    made = list(dict.fromkeys((*MODELS, *trained)))
    return sorted([
        *(f"adapters/{name}/{file}" for name in made for file in ("adapter_config.json", "adapter_model.safetensors")),
        *(f"l5_draws_loss_{name}.json" for name in trained), *(f"l5_draws_training_{name}.jsonl" for name in made),
        *(f"{step}.done.json" for step in _used(recipe)), "problems.jsonl", "report_ladder_l5_draws.json",
        "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", "l4_stored_pre_r64_reach.jsonl", "l5_held_aside_problems.jsonl"])


def _stage5d(config, monkeypatch, until=None, stage_name="ladder_l5_draws", only=None):
    """The stage's steps in its order; the trainings and the averagings by their GPU path (the stand-in makes no adapter)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l5_draws.STEPS or (only and step not in only):
            continue
        run = ladder_l5_draws.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in (*TRAININGS, *AVERAGES) else run(config)
        if step == until:
            break
    return summaries


def _l5ns_report(runs, name=WITH_THE_DECAY, **changed):
    """L5n's FINISHED run, hand-made (it is only read): its report's "what follows" names `name`. Returns its directory."""
    directory = runs / "ladder_l5_noise_pre_r64_seed0"
    directory.mkdir(exist_ok=True)
    (directory / "ladder_l5_noise_report.done.json").write_text(json.dumps({"ok": True}))
    (directory / "report_ladder_l5_noise.json").write_text(json.dumps({"ok": True, "inconclusive": False, "headline": "L5n", "branches": {"what_follows": name}, **changed}))
    return directory


def _another_seed_of_l5(world, seed, successes):
    """A FINISHED run of L5 at another of its seeds, hand-made (it is only read): the run of the search's seed, copied,
    with its three models closing the goal problem in `successes` of the first sampling's attempts. Returns its directory."""
    directory = world.l5.with_name(world.l5.name.replace("seed0", f"seed{seed}"))
    shutil.copytree(world.l5, directory)
    for model, count in zip(MODELS, successes):
        path = directory / f"episodes_l4_reach_{model}_problems.jsonl"
        path.write_text("".join(json.dumps({**row, "resolved": count, "resolved_by_statement": count}) + "\n" for row in _file_rows(path)))
    return directory


@pytest.fixture
def draws(deep, monkeypatch, tmp_path):  # noqa: F811
    """`deep`'s world with what this stage reads on the box: a base map of twelve problems with a hand-made map of
    `pre_r64`, and L5's own stage RUN TO ITS END there at its search's seed (four deep solves, TWO asked for at least, so
    a set is EIGHT rows). The batch is ONE row, in L5's run and here, so that a training of eight rows is eight optimizer steps: three of them
    after the five warm-up steps. Then this stage's task of SEED 3, which carries NO variable. L5n's run is NOT THERE
    (`_l5ns_report` makes one). `solver.measured`: every prompt of a measurement of the goal set's first sampling made
    from here on; the k-th closes the goal as `SUCCESSES` says, seed by seed in the order 3, 4, 5."""
    for name in (RUN, SMOKE):
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    config["training"]["effective_batch"] = 1
    config["ladder_loop"]["l5_draws"]["recipe"] = CONSTANT          # A TEST STATES ITS RECIPE (`_recipe`): the package's own setting is the one L5n's real report named
    first = _round_one(deep, monkeypatch, tmp_path, solved=tuple(DRAWN), l4p=False, settings={"problems": 4, "held_aside": 1, "minimum": 2})
    root = deep.store().root
    assert root.name == "ladder_l5_deep_pre_r64_seed0" and first["ladder_l5_report"]["inconclusive"] is False and first["ladder_l5_report"]["ok"] is True
    assert (first["ladder_l5_prepare"]["rows_of_each_kind"], first["ladder_l5_train_published"]["rows"], first["ladder_l5_train_published"]["steps"]) == (4, 8, 8)
    calls, kits = [], []
    _trainings(monkeypatch, calls)
    solver = NoiseSolver(deep.solver, [count for seed in (3, 4, 5) for count in SUCCESSES[seed]])
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"]}),
        (solver, stand_in_parameters))[1]))
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l5_draws"])        # THE STAGE names no variable
    monkeypatch.setenv(SEEDS, "3")
    return SimpleNamespace(config=config, deep=deep, first=first, l5=root, calls=calls, kits=kits, solver=solver, asked=len(deep.solver.asked), runs=ladder_l5._runs(config),
                           store=lambda: ladder_l5_draws._store(config, ladder_l5_draws.the_settings(config)))


def _recipe(world, recipe, monkeypatch=None):
    """The setting names `recipe`, and L5n's finished report (hand-made) names the case that asks for it."""
    world.config["ladder_loop"]["l5_draws"]["recipe"] = recipe
    _l5ns_report(world.runs, {CONSTANT: THE_RECIPES_OWN, DECAY: WITH_THE_DECAY, MEAN_CONSTANT: AVERAGE_OF_FOUR}[recipe])


# ------------------------------------------------------------------------------------------------ the stage
def test_a_seed_builds_l5s_three_sets_for_its_seed_trains_each_as_l5_does_measures_g_alone_and_reports(draws, monkeypatch):  # noqa: F811
    config, calls, kits, solver, deep_ = draws.config, draws.calls, draws.kits, draws.solver, draws.deep
    read_only = {path: _all(path) for path in (draws.l5, deep_.map, deep_.l4t2, deep_.rank, deep_.l4t, deep_.arm, deep_.pre)}
    summaries = _stage5d(config, monkeypatch)
    store = draws.store()
    assert store.root.name == "ladder_l5_draws_pre_r64_seed3" and store.root.parent == draws.l5.parent and list(summaries) == STEPS == list(ladder_l5_draws.STEPS)
    assert {path: _all(path) for path in read_only} == read_only               # L5's RUN AND EVERY RUN IT READ WERE ONLY READ: file for file
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == _own_files(CONSTANT)
    assert sorted(path.name for path in draws.runs.iterdir() if path.name.startswith(("ladder_l5", "ladder_l6"))) == ["ladder_l5_deep_pre_r64_seed0", "ladder_l5_draws_pre_r64_seed3"]
    # THE STEPS OF ANOTHER RECIPE MAKE NOTHING, say why, and mark nothing.
    for step in STEPS:
        if step not in _used(CONSTANT):
            own = summaries[step]
            assert own == {"stage": "l4", "label": "pretrained on published proofs", "check": "L5d", "run": False, "why": own["why"]} and not store.is_done(step)
            assert own["why"].startswith("the recipe of this run is `constant`: ") and ("no adapter is averaged" in own["why"] if "_average_" in step else "never in the order of an order seed" in own["why"])

    # ---- the first step: the seed, the recipe (L5n's run is NOT THERE: the recipe as it is, and that is said), L5's search, the one sampling
    read = summaries["ladder_l5_draws_prepare"]
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"], read["models"], read["trained"], read["averaged"]) == (
        "l4", "pretrained on published proofs", "L5d", 3, "pre_r64", list(MODELS), list(MODELS), [])
    assert (read["start_adapter"], read["start_run"]) == (str(deep_.rank / "adapters" / "pre_r64"), str(deep_.rank))
    of_the_recipe = read["the_recipe"]
    assert (of_the_recipe["recipe"], of_the_recipe["asked_for_by_l5n"], of_the_recipe["l5n_named_it"]) == ("constant", "constant", False)
    assert of_the_recipe["of_l5n"] == {"there": False, "run": str(draws.runs / "ladder_l5_noise_pre_r64_seed0"),
                                       "why": "ladder_l5_noise_pre_r64_seed0 does not hold ['ladder_l5_noise_report.done.json', 'report_ladder_l5_noise.json']"}
    assert of_the_recipe["said_beside_every_read"] == ("L5n's run is not there or not finished (ladder_l5_noise_pre_r64_seed0 does not hold ['ladder_l5_noise_report.done.json', "
                                                       "'report_ladder_l5_noise.json']): the recipe as it is, one training a model, is trained with, and L5n did not name it")
    assert read["the_seeds"] == {"what": read["the_seeds"]["what"], "seeds": [3, 4, 5], "more_seeds": [6, 7, 8], "more_seeds_approved": False, "fixed": [3, 4, 5], "of_l5": [0, 1, 2]}
    of_the_search = read["the_search"]
    assert (of_the_search["run"], of_the_search["seed"], of_the_search["problems"], of_the_search["deep_solves"], of_the_search["rows_of_each_kind"], of_the_search["cap"]) == (
        str(draws.l5), 0, 4, 4, 4, 400)
    assert (of_the_search["ordinary_solves_drawn"], of_the_search["held_aside_problems"], of_the_search["inconclusive"], of_the_search["enough_deep_solves"]["passes"]) == (4, 1, False, True)
    assert of_the_search["enough_deep_solves"]["minimum"] == 2 == draws.first["ladder_l5_prepare"]["enough_deep_solves"]["minimum"]      # A REAL RUN asks the setting's minimum, as L5's own task did
    sampling = read["the_sampling"]
    assert (sampling["name"], sampling["episodes"], sampling["sampling_seed"]) == ("reach", 32, 1001) == tuple(draws.first["ladder_l5_search_prepare"]["goal_samplings"][0].values())
    assert sampling["sets"] == {model: f"l4_reach_{model}" for model in MODELS} and sampling["stored_start"] == "l4_stored_pre_r64_reach.jsonl"
    assert read["goal_samplings"] == [{"name": "reach", "episodes": 32, "sampling_seed": 1001}] and (read["goal_set"], read["attempts_a_goal_problem"]) == (1, 32)
    problems = store.read_rows("problems.jsonl")
    assert [row["set"] for row in problems] == [f"l4_reach_{model}" for model in MODELS] and {row["problem_id"] for row in problems} == {GOAL}       # NO SET OF A RUNG, OF A SECOND SAMPLING
    theirs = draws.first["ladder_l5_search_prepare"]
    assert {key: read["recipe"][key] for key in ("learning_rate", "warmup_steps", "effective_batch", "lora")} == {key: theirs["recipe"][key] for key in ("learning_rate", "warmup_steps", "effective_batch", "lora")}
    assert (read["recipe"]["adapter_seed"], read["recipe"]["optimizer_steps"]) == (3, 8) and "learning_rate_factor" not in read["recipe"] and read["start_recipe"] == theirs["start_recipe"]
    assert (read["smoke"], read["the_orders"], read["the_averaged"]) == (None, None, {})
    # WHAT IT COPIED of L5's run and of the runs that task read: the groups, G', the held-aside problems, the start model's first sampling.
    for name in ("l4_heldout_groups.jsonl", "l4_goal_set_again.jsonl", "l5_held_aside_problems.jsonl", "l4_stored_pre_r64_reach.jsonl"):
        assert store.read_rows(name) == _file_rows(draws.l5 / name)
    assert [row["problem_id"] for row in store.read_rows("l5_held_aside_problems.jsonl")] == [HELD]

    # ---- THE THREE SETS ARE WHAT A FURTHER SEED OF L5 BUILDS FOR THIS SEED: L5's own task of seed 3, run to its three sets in the same store, stores the same rows in the same order
    sets = {model: store.read_rows(f"l5_draws_training_{model}.jsonl") for model in MODELS}
    asked, trained = len(deep_.solver.asked), len(calls)
    further = _stage5(config, monkeypatch, until="ladder_l5_prepare")
    of_l5 = ladder_l5._store(config, ladder_l5.the_settings(config)).root
    assert of_l5.name == "ladder_l5_deep_pre_r64_seed3" and further["ladder_l5_prepare"]["seed"] == 3 and (len(deep_.solver.asked), len(calls)) == (asked, trained)
    for model in MODELS:
        assert sets[model] == _file_rows(of_l5 / f"l5_training_{model}.jsonl") and len(sets[model]) == 8
        assert [row["id"] for row in sets[model]] == [row["id"] for row in training_order(sets[model], 3)] and [row["row"] for row in sets[model]] == list(range(8))
        assert not any(field in row for row in sets[model] if row["origin"] == "pretraining" for field in TEXT_FIELDS)              # no published row with its text
        assert not {GOAL, HELD} & {row["problem_id"] for row in sets[model]}
        entry_, theirs_ = read["the_sets"][model], further["ladder_l5_prepare"]["training_sets"][model]
        assert (entry_["file"], entry_["rows"], entry_["rows_by_origin"], entry_["optimizer_steps"]) == (f"l5_draws_training_{model}.jsonl", 8, theirs_["rows_by_origin"], 8)
        assert entry_["problems_of_g_or_of_the_held_aside_set"] == {"problems_of_the_goal_set": 0, "problems_of_the_held_aside_set": 0}
        assert read["training_sets"][model]["file"] == entry_["file"] and read["training_sets"][model]["model"] == model and "order_seed" not in read["training_sets"][model]
    assert {model: read["the_sets"][model]["rows_by_origin"] for model in MODELS} == {PUBLISHED: {"pretraining": 8}, SHALLOW: {"shallow": 4, "pretraining": 4}, DEEP: {"deep": 4, "pretraining": 4}}
    of_seed_0 = {model: _file_rows(draws.l5 / f"l5_training_{model}.jsonl") for model in MODELS}
    assert all({row["id"] for row in sets[model]} != {row["id"] for row in of_seed_0[model]} for model in MODELS)                  # ... and they are NOT seed 0's: a fresh draw
    pretraining = ladder_l4_start.pretrained_on(deep_.rank, read["pretraining_file_sha256"], "pre_r64")[0]
    assert hold_the_sets(sets, pretraining, 3) == 4
    shutil.rmtree(of_l5)

    # ---- the three trainings: EXACTLY THE CALL L5 TRAINS A SEED'S MODEL BY (no decay, no learning rate recorded), from `pre_r64` at rank 64, with the task's seed
    start = deep_.rank / "adapters" / "pre_r64"
    assert [(call["seed"], call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(3, 64, 128, {"start": start})] * 3
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_draws_pre_r64_seed3_{model}" for model in MODELS]
    for call, model in zip(calls, MODELS):
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in sets[model]] and all(example["theorem"] and example["completion"] for example in call["examples"])
        assert call["orders"] == [list(range(8))] and call["schedule"]["checkpoint_steps"] == {model: 8}
        train, loss = summaries[f"ladder_l5_draws_train_{model}"], json.loads(store.path(f"l5_draws_loss_{model}.json").read_text())
        assert (train["model"], train["rows"], train["steps"], train["adapter"], train["passes"]) == (model, 8, 8, str(store.root / "adapters" / model), 1)
        assert train["adapter_saved"]["rank"] == 64 and train["against_the_start_adapter"] == {model: CHANGED} and loss["rows_trained"] == [row["id"] for row in sets[model]]
        assert "learning_rate" not in train and "learning_rates" not in train and all("learning_rate" not in row for row in loss["training_steps"])
        assert train["order"] == "a content hash of the task's seed and each row's id (`assembly.training_order`)" == draws.first[f"ladder_l5_train_{model}"]["order"]
        assert sorted(train) == sorted(draws.first[f"ladder_l5_train_{model}"])                                                    # the summary L5's own training writes, key for key

    # ---- the three measurements: the goal set's FIRST SAMPLING ALONE, 32 attempts with the sampling seed `pre_r64`'s run recorded; no rung, no second sampling, no held-aside set
    assert solver.measured == [(GOAL, 32, 1001, 1024)] * 3 and len(deep_.solver.asked) == draws.asked                # NOTHING else was sampled: the world's own solver was asked nothing
    assert [kit["max_lora_rank"] for kit in kits] == [64] * 3 and {kit["temperature"] for kit in kits} == {1.0}
    for model, count in zip(MODELS, SUCCESSES[3]):
        measured = summaries[f"ladder_l5_draws_measure_{model}"]
        assert (measured["model"], measured["check"], measured["adapter"]) == (model, "L5d", str(store.root / "adapters" / model)) and "rungs" not in measured and list(measured["goal"]) == ["reach"]
        assert (measured["goal"]["reach"]["set"], measured["goal"]["reach"]["episodes_each"], measured["goal"]["reach"]["sampling_seed"]) == (f"l4_reach_{model}", 32, 1001)
        assert [(row["problem_id"], row["episodes"], row["resolved"]) for row in store.read_rows(f"episodes_l4_reach_{model}_problems.jsonl")] == [(GOAL, 32, count)]
    assert not list(store.root.glob("episodes_l4_rungs_*")) and not list(store.root.glob("episodes_l4_more_*")) and not list(store.root.glob("episodes_l5_held_aside_*"))
    assert sorted(path.name for path in store.root.glob("episodes_*.done.json") if "_block_" not in path.name) == sorted(f"episodes_l4_reach_{model}.done.json" for model in MODELS)

    # ---- the report: this seed's checks pass and its figures are read; THE READ OVER THE SEEDS IS NOT READ, naming the seeds that are not there
    report = summaries["ladder_l5_draws_report"]
    assert json.loads(store.path("report_ladder_l5_draws.json").read_text()) == json.loads(json.dumps(report, default=str)) and report["ok"] is True and list(report) == REPORT_KEYS
    assert report["inconclusive"] is False and report["failed_checks"] == [] and all(line.startswith(f"{SAY}: ") for line in report["lines"]) and report["recipe"] == "constant"
    checks = report["can_this_run_be_read"]
    assert list(checks) == ["what", "the_run", *MODELS] and list(checks["the_run"]) == ["each_set_is_l5s_rule_for_its_seed_made_again_and_held", "the_stored_first_sampling_of_the_start_model_is_the_settings"]
    assert all(list(checks[model]) == ["its_adapter_is_not_the_start_adapters_and_is_read_back_at_its_rank", "it_was_trained_on_its_set_and_on_nothing_else", "no_barred_row",
                                       "no_problem_of_g_or_of_the_held_aside_set", "lean_answered"] for model in MODELS)
    assert checks["the_run"]["each_set_is_l5s_rule_for_its_seed_made_again_and_held"]["rows_of_each_kind"] == 4
    of_this_seed = report["of_this_seed"]
    assert (of_this_seed["seed"], of_this_seed["recipe"], of_this_seed["ok"], of_this_seed["inconclusive"], of_this_seed["failed_checks"]) == (3, "constant", True, False, [])
    assert of_this_seed["the_measurement"] == {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 1}
    assert {model: (of_this_seed["figures"][model]["all_of_g"]["successes"], of_this_seed["figures"][model]["all_of_g"]["attempts"]) for model in MODELS} == {
        model: (count, 32) for model, count in zip(MODELS, SUCCESSES[3])}
    assert report["the_primarys_difference_at_this_seed"][THE_PRIMARY] == 6 / 32 and report["figures"]["problems"]["all_of_g"] == 1
    assert {model: of_this_seed["counts"][model]["solved"] for model in MODELS} == {model: 1 for model in MODELS}
    over = report["over_the_fresh_seeds"]
    assert (over["read"], over["fixed_seeds"], over["seeds_there"], over["seeds_missing"]) == (False, [3, 4, 5], [3], [4, 5])
    assert report["branches"] == {"primary": NOT_READ, "where_it_sits": NOT_READ, "what_follows": NOT_READ}
    assert over["primary"]["reason"] == "the reads are fixed over the seeds [3, 4, 5] and the runs of [4, 5] are not there: nothing is read over the seeds"
    assert any("THE PRIMARY, over the fresh seeds [3, 4, 5] alone" in line and f": {NOT_READ}. the reads are fixed over the seeds [3, 4, 5] and the runs of [4, 5] are not there" in line
               and line.endswith(f" NOTE: {of_the_recipe['said_beside_every_read']}") for line in report["lines"])
    assert any(f"THE RECIPE is `constant`: {of_the_recipe['found']}. SAID BESIDE EVERY READ: " in line for line in report["lines"]) and of_the_recipe["said_beside_every_read"] in report["headline"]
    # L5's OWN SEEDS on the same sampling: seed 0's run is there, seeds 1 and 2 are not (no such run in this world); they are read by nothing while the read is not made.
    of_l5s = report["the_seeds_of_l5"]
    assert list(of_l5s) == ["0", "1", "2"] and (of_l5s["0"]["there"], of_l5s["0"]["run"]) == (True, str(draws.l5)) and of_l5s["1"]["there"] is False and of_l5s["2"]["there"] is False
    assert {model: of_l5s["0"]["figures"][model]["all_of_g"]["attempts"] for model in MODELS} == {model: 32 for model in MODELS}
    assert report["timed_out_in_lean"]["checked_again"] is False and report["adapters"]["there"] == sorted(MODELS) and report["adapters"]["kept"] is True and report["smoke"] is None
    assert store.done_summary("ladder_l5_draws_report") == {"stage": "l4", "label": "pretrained on published proofs", "check": "L5d", "seed": 3, "recipe": "constant",
                                                           "headline": report["headline"], "branches": report["branches"], "measured": list(MODELS), "ok": True}
    # ---- queued again: every step returns what is stored; nothing is trained or sampled; nothing that was read moved
    own_files = _all(store.root)
    again = _stage5d(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}
    assert (len(calls), len(solver.measured)) == (3, 3) and again["ladder_l5_draws_report"]["branches"] == report["branches"] and {path: _all(path) for path in read_only} == read_only
    assert {name: content for name, content in _all(store.root).items() if "report" not in name} == {name: content for name, content in own_files.items() if "report" not in name}
    # ---- THE LONG PROBLEMS are the goal problems whose shortest published proof has 8 lines or more, as the lengths file (which the report alone reads) groups them: the
    # fixture's goal problem is in none of its groups; put in the longest it is the one long problem, and in the shortest it is not.
    assert report["counts"]["long_problems"] == 0 and {model: report["counts"][model]["long"]["problems"] for model in MODELS} == {model: 0 for model in MODELS}
    monkeypatch.setattr(ladder_l3c, "proof_lengths", lambda: {GOAL: {"problem_id": GOAL, "length_group": "8+"}})
    of_the_long = ladder_l5_draws.ladder_l5_draws_report(config)["counts"]
    assert of_the_long["long_problems"] == 1 and {model: of_the_long[model]["long"] for model in MODELS} == {model: {"problems": 1, "solved": 1, "gained": 0, "lost": 0} for model in MODELS}
    monkeypatch.setattr(ladder_l3c, "proof_lengths", lambda: {GOAL: {"problem_id": GOAL, "length_group": "1"}})
    assert ladder_l5_draws.ladder_l5_draws_report(config)["counts"]["long_problems"] == 0


def test_g_again_is_copied_as_l5s_reading_hands_it_on_and_a_stored_sampling_it_was_not_made_from_gives_no_figure_on_a_part(draws, monkeypatch):  # noqa: F811
    config = draws.config
    # In this world the start model solves the one goal problem, so G' holds no problem. Here L5's reading is made to hand on a G' that holds it: the first step copies WHAT IT
    # IS HANDED, and the report, whose parts of G are the stored first sampling's, finds that G' is not that sampling's: it gives no figure on a part.
    by_l5 = ladder_l5.read_for_the_search
    monkeypatch.setattr(ladder_l5, "read_for_the_search", lambda *arguments, **more: {**by_l5(*arguments, **more), "again": [GOAL]})
    summaries = _stage5d(config, monkeypatch)
    store = draws.store()
    assert store.read_rows("l4_goal_set_again.jsonl") == [{"problem_id": GOAL}] and summaries["ladder_l5_draws_prepare"]["goal_set_again"] == 1
    report = summaries["ladder_l5_draws_report"]
    assert report["inconclusive"] is False and report["figures"]["problems"] == {"all_of_g": 1, "solved_by_the_start": 0, "goal_set_again": 0}
    assert {model: [report["figures"][model][part]["attempts"] for part in l5_draws.PARTS] for model in MODELS} == {model: [32, 0, 0] for model in MODELS}
    assert report["the_primarys_difference_at_this_seed"][THE_PRIMARY] == 6 / 32 and any("on G' (0) " in line for line in report["lines"])


def test_the_other_fresh_seeds_are_read_from_their_runs_when_those_are_there_and_finished_and_by_no_run_the_task_names(draws, monkeypatch):  # noqa: F811
    config, runs = draws.config, draws.runs
    settings = ladder_l5_draws.the_settings(config)

    def a_run(seed, entry, marked=True):
        """A run of another seed, hand-made (it is only read): its report with what it says of its own seed, and its report step's marker."""
        directory = runs / f"ladder_l5_draws_pre_r64_seed{seed}"
        directory.mkdir()
        (directory / "report_ladder_l5_draws.json").write_text(json.dumps({"check": "L5d", "of_this_seed": entry}))
        if marked:
            (directory / "ladder_l5_draws_report.done.json").write_text(json.dumps({"ok": True}))
        return directory

    there, unfinished = a_run(4, {"seed": 4, "recipe": "constant"}), a_run(5, {"seed": 5, "recipe": "constant"}, marked=False)
    of_another, further = a_run(6, {"seed": 4, "recipe": "constant"}), a_run(8, {"seed": 8, "recipe": "decay"})
    found = ladder_l5_draws.read_the_seeds(config, settings, 3)
    # EVERY SEED OF THE SETTING BUT THE TASK'S OWN, the further ones too, whatever the approval: a run that is there is handed to the read, which names it.
    assert list(found) == [4, 5, 6, 7, 8] and found[4] == {"seed": 4, "recipe": "constant", "there": True, "run": str(there)}
    assert found[8] == {"seed": 8, "recipe": "decay", "there": True, "run": str(further)}
    # A RUN WHOSE REPORT STEP IS NOT MARKED DONE, one whose report is of another seed, and one that is not there are NOT THERE, each with why; nothing is refused for them.
    assert found[5] == {"there": False, "run": str(unfinished), "why": "ladder_l5_draws_pre_r64_seed5 does not hold ['ladder_l5_draws_report.done.json']"}
    assert found[6] == {"there": False, "run": str(of_another), "why": "the report of ladder_l5_draws_pre_r64_seed6 is not a report of L5d at seed 6"}
    assert found[7] == {"there": False, "run": str(runs / "ladder_l5_draws_pre_r64_seed7"),
                        "why": "ladder_l5_draws_pre_r64_seed7 does not hold ['ladder_l5_draws_report.done.json', 'report_ladder_l5_draws.json']"}
    assert list(ladder_l5_draws.read_the_seeds(config, settings, 4)) == [3, 5, 6, 7, 8]
    # A RUN IN A DIRECTORY THE TASK NAMES (a smoke run) reads no other seed's run, though they are there.
    monkeypatch.setenv(RUN, "ladder_l5_draws_smoke")
    named = ladder_l5_draws.read_the_seeds(config, settings, 3)
    assert named == {seed: {"there": False, "why": "this run is in a directory the task names (RLVR_LEAN_LADDER_L5_DRAWS_RUN: a smoke run), and no other seed's run is read from it"}
                     for seed in (4, 5, 6, 7, 8)}


# ------------------------------------------------------------------------------------- the read over the seeds
def test_the_read_over_the_fresh_seeds_is_made_from_the_runs_that_are_there_and_l5s_own_seeds_stand_beside_as_exploratory(draws, monkeypatch, capsys):  # noqa: F811
    config, runs = draws.config, draws.runs
    reports, sets = {}, {}
    for seed in (3, 4, 5):
        monkeypatch.setenv(SEEDS, str(seed))
        reports[seed] = _stage5d(config, monkeypatch)["ladder_l5_draws_report"]
        sets[seed] = {model: [row["id"] for row in draws.store().read_rows(f"l5_draws_training_{model}.jsonl")] for model in MODELS}
    directories = [runs / f"ladder_l5_draws_pre_r64_seed{seed}" for seed in (3, 4, 5)]
    assert all(path.is_dir() for path in directories) and [call["seed"] for call in draws.calls] == [3] * 3 + [4] * 3 + [5] * 3
    assert all(sets[seed][model] != sets[other][model] for seed, other in ((3, 4), (3, 5), (4, 5)) for model in MODELS)             # EACH SEED IS A FRESH DRAW, and its own order
    # A SEED'S REPORT READS THE SEEDS THAT ARE THERE WHEN IT IS WRITTEN: the first names the two that are not, the second the one, the third reads the three.
    assert [reports[seed]["over_the_fresh_seeds"]["seeds_missing"] for seed in (3, 4, 5)] == [[4, 5], [5], []]
    assert [reports[seed]["branches"]["primary"] for seed in (3, 4, 5)] == [NOT_READ, NOT_READ, BEAT]
    last, at_each = reports[5], {3: 6 / 32, 4: 5 / 32, 5: 6 / 32}
    over = last["over_the_fresh_seeds"]
    assert (over["read"], over["seeds_there"], over["recipe"], over["primary"]) == (True, [3, 4, 5], "constant", primary(at_each)) and over["primary"]["low"] > 0
    assert {seed: {model: over["figures"][seed][model]["all_of_g"]["successes"] for model in MODELS} for seed in ("3", "4", "5")} == {
        str(seed): dict(zip(MODELS, SUCCESSES[seed])) for seed in (3, 4, 5)}
    assert over["beside"]["deep_minus_published"]["at_each_seed"] == {"3": 0.25, "4": 0.1875, "5": 0.25} and over["beside"]["deep_minus_shallow"]["at_each_seed"] == {"3": 0.125, "4": 0.0625, "5": 0.125}
    assert over["the_spread"][PUBLISHED]["by_seed"] == {"3": 0.125, "4": 0.1875, "5": 0.25}
    # WHERE IT SITS is not read in this world: its one goal problem is solved by the start model's own first sampling, so G' holds no problem and has no difference.
    assert last["figures"]["problems"] == {"all_of_g": 1, "solved_by_the_start": 1, "goal_set_again": 0} and over["where_it_sits"]["name"] == NOT_READ and "a part of G holds no problem" in over["where_it_sits"]["reason"]
    assert last["branches"] == {"primary": BEAT, "where_it_sits": NOT_READ, "what_follows": L6_RUNS} and f"Over the fresh seeds [3, 4, 5]: {BEAT}. What follows: {L6_RUNS}" in last["headline"]
    # L5's OWN SEEDS BESIDE: seed 0's run is there, seeds 1 and 2 are not: NOT READ, with why.
    assert over["exploratory"]["read"] is False and over["exploratory"]["why"].startswith("L5's seeds are not there: seed 1: ladder_l5_deep_pre_r64_seed1 does not hold ")
    assert "seed 2: ladder_l5_deep_pre_r64_seed2 does not hold " in over["exploratory"]["why"] and "seed 0" not in over["exploratory"]["why"]
    # THE FIRST SEED'S TASK QUEUED AGAIN builds its report again with the seeds that are there: the same read, and nothing trained or sampled.
    trained, measured = len(draws.calls), len(draws.solver.measured)
    monkeypatch.setenv(SEEDS, "3")
    again = _stage5d(config, monkeypatch)["ladder_l5_draws_report"]
    assert again["over_the_fresh_seeds"] == over and again["branches"] == last["branches"] and (len(draws.calls), len(draws.solver.measured)) == (trained, measured)
    assert again["of_this_seed"] == reports[3]["of_this_seed"]
    # ---- L5's SEEDS 1 AND 2 THERE AND FINISHED: they stand beside the fresh ones, EXPLORATORY, and the primary is the fresh seeds' alone
    of_seed_0 = {model: _file_rows(draws.l5 / f"episodes_l4_reach_{model}_problems.jsonl")[0]["resolved"] for model in MODELS}
    seed_1, seed_2 = _another_seed_of_l5(draws, 1, (4, 8, 8)), _another_seed_of_l5(draws, 2, (8, 10, 14))
    beside = ladder_l5_draws.ladder_l5_draws_report(config)
    exploratory = beside["over_the_fresh_seeds"]["exploratory"]
    theirs = {0: ((of_seed_0[SHALLOW] + of_seed_0[DEEP]) / 2 - of_seed_0[PUBLISHED]) / 32, 1: 4 / 32, 2: 4 / 32}
    assert (exploratory["read"], exploratory["exploratory"], exploratory["seeds_that_found_the_sign"]) == (True, True, [0, 1, 2]) and exploratory["of_l5"] == {str(seed): round(value, 6) for seed, value in theirs.items()}
    assert exploratory["pooled"]["seeds"] == 6 and list(exploratory["pooled"]["at_each_seed"]) == ["0", "1", "2", "3", "4", "5"] and beside["over_the_fresh_seeds"]["primary"] == over["primary"]
    assert {seed: (own["there"], own["run"]) for seed, own in beside["the_seeds_of_l5"].items()} == {"0": (True, str(draws.l5)), "1": (True, str(seed_1)), "2": (True, str(seed_2))}
    assert any("EXPLORATORY, naming nothing: L5's own seeds [0, 1, 2] FOUND THE SIGN" in line and "pooled with the fresh seeds, 6 seeds" in line for line in beside["lines"])
    # A SEED OF L5 THAT IS NOT FINISHED, NOT TO BE READ, OR WOULD NOT PAIR is NOT THERE, with why; nothing is refused for it, and the report is written.

    def not_there(change, why):
        kept = dict(_all(seed_2))
        change()
        found = ladder_l5_draws.ladder_l5_draws_report(config)
        assert found["ok"] is True and found["the_seeds_of_l5"]["2"]["there"] is False and why in found["the_seeds_of_l5"]["2"]["why"], found["the_seeds_of_l5"]["2"]
        assert found["over_the_fresh_seeds"]["exploratory"]["read"] is False and found["branches"]["primary"] == BEAT
        for name, content in kept.items():
            (seed_2 / name).write_bytes(content)

    not_there(lambda: (seed_2 / "ladder_l5_report.done.json").unlink(), "ladder_l5_deep_pre_r64_seed2 does not hold ['ladder_l5_report.done.json']")
    not_there(lambda: (seed_2 / "ladder_l5_measure_shallow.done.json").unlink(), "does not hold ['ladder_l5_measure_shallow.done.json']")
    not_there(lambda: (seed_2 / "episodes_l4_reach_deep_problems.jsonl").unlink(), "does not hold ['episodes_l4_reach_deep_problems.jsonl']")
    not_there(lambda: (seed_2 / "report_ladder_l5.json").write_text(json.dumps({"ok": False})), "the report of ladder_l5_deep_pre_r64_seed2 is not to be read")
    not_there(lambda: (seed_2 / "episodes_l4_reach_deep.done.json").write_text(json.dumps({**json.loads((seed_2 / "episodes_l4_reach_deep.done.json").read_text()), "sampling_seed": 9})),
              "measured `deep` of L5's seed 2 with sampling seed 9 and 32 episodes")
    # THE RUNS OF L5 WERE ONLY READ.
    assert sorted(path.name for path in runs.iterdir() if path.name.startswith("ladder_l5")) == [
        *(f"ladder_l5_deep_pre_r64_seed{seed}" for seed in (0, 1, 2)), *(f"ladder_l5_draws_pre_r64_seed{seed}" for seed in (3, 4, 5))]
    # ---- THE COMMAND makes the same read again from the seeds' directories (here the runs themselves: a pulled task directory holds the same report), and prints its lines.
    # L5's seeds are those of the first report that holds them all; here no report does (seed 3's, the last one written, says seed 2 would not pair; the two others were
    # written before L5's seeds 1 and 2 were there): NOT READ, with why, as the last directory given says it.
    said = beside["the_recipe"]["said_beside_every_read"]
    capsys.readouterr()
    assert tool.main([str(path) for path in directories]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 7 and "L5's seeds beside the fresh ones are NOT READ (L5's seeds are not there: seed 1: ladder_l5_deep_pre_r64_seed1 does not hold " in printed[4]
    assert ladder_l5_draws.ladder_l5_draws_report(config)["over_the_fresh_seeds"] == beside["over_the_fresh_seeds"]              # seed 3's report, built again with L5's three seeds there
    capsys.readouterr()
    assert tool.main([str(path) for path in directories]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed == lines_over_the_seeds(beside["over_the_fresh_seeds"], said) and all(line in beside["lines"] for line in printed) and len(printed) == 7
    assert tool.main([str(directories[0]), str(directories[2])]) == 0 and "the runs of [4] are not there" in capsys.readouterr().out


def test_a_seed_whose_check_failed_is_inconclusive_says_nothing_else_and_makes_the_read_over_the_seeds_inconclusive(draws, monkeypatch):  # noqa: F811
    config = draws.config
    for seed in (3, 4, 5):
        monkeypatch.setenv(SEEDS, str(seed))
        _stage5d(config, monkeypatch)
    monkeypatch.setenv(SEEDS, "4")
    store = draws.store()
    report = lambda: ladder_l5_draws.ladder_l5_draws_report(config)      # noqa: E731

    def failed(path, change, *checks):
        """The report with one stored file changed: THE checks that fail, and no other. Returns it; the file is as it was afterwards."""
        kept = path.read_text()
        path.write_text(change(kept))
        found, said = report(), "; ".join(checks)
        assert found["inconclusive"] is True and found["failed_checks"] == list(checks) and found["ok"] is True and found["of_this_seed"]["figures"] is None and found["of_this_seed"]["counts"] is None
        assert found["branches"] == {"primary": INCONCLUSIVE, "where_it_sits": NOT_READ, "what_follows": NOT_READ} and list(found["figures"]) == ["what", "problems", "measured_and_not_read"]
        assert any(f"THIS SEED IS {INCONCLUSIVE}: {said}. Nothing else is said of it" in line for line in found["lines"]) and not any("THIS SEED'S FIGURES" in line for line in found["lines"])
        assert found["over_the_fresh_seeds"]["seeds_whose_checks_failed"] == {"4": list(checks)} and f"THIS SEED IS {INCONCLUSIVE} ({said})" in found["headline"]
        # ANOTHER SEED's report, built again while this one stands so: the read over the seeds is INCONCLUSIVE there too, until this seed is run again.
        monkeypatch.setenv(SEEDS, "5")
        other = report()
        assert other["inconclusive"] is False and other["branches"]["primary"] == INCONCLUSIVE and "a check failed at seed 4" in other["over_the_fresh_seeds"]["primary"]["reason"]
        monkeypatch.setenv(SEEDS, "4")
        path.write_text(kept)
        assert report()["branches"]["primary"] == BEAT
        return found

    as_rows = lambda change: (lambda text: "".join(json.dumps(row) + "\n" for row in change([json.loads(line) for line in text.splitlines()])))      # noqa: E731
    as_json = lambda change: (lambda text: json.dumps(change(json.loads(text))))      # noqa: E731
    # A STORED SET THAT IS NOT L5's RULE FOR THIS SEED: two rows in another order; a published row with its text.
    failed(store.path("l5_draws_training_deep.jsonl"), as_rows(lambda own: [{**own[1], "row": 0}, {**own[0], "row": 1}, *own[2:]]), "the run: each_set_is_l5s_rule_for_its_seed_made_again_and_held")
    published = next(index for index, row in enumerate(store.read_rows("l5_draws_training_shallow.jsonl")) if row["origin"] == "pretraining")
    failed(store.path("l5_draws_training_shallow.jsonl"), as_rows(lambda own: [{**row, "completion": "by simp"} if index == published else row for index, row in enumerate(own)]),
           "the run: each_set_is_l5s_rule_for_its_seed_made_again_and_held")
    # THE START MODEL'S STORED FIRST SAMPLING is not the setting's: another number of attempts at a problem.
    failed(store.path("l4_stored_pre_r64_reach.jsonl"), as_rows(lambda own: [{**own[0], "episodes": 61}, *own[1:]]), "the run: the_stored_first_sampling_of_the_start_model_is_the_settings")
    # A TRAINING WHOSE ADAPTER IS THE START ADAPTER'S; one that recorded a row twice.
    failed(store.path("ladder_l5_draws_train_shallow.done.json"), as_json(lambda own: {**own, "against_the_start_adapter": {"shallow": {**CHANGED, "elements_changed": 0}}}),
           "`shallow`: its_adapter_is_not_the_start_adapters_and_is_read_back_at_its_rank")
    failed(store.path("l5_draws_loss_published.json"), as_json(lambda own: {**own, "rows_trained": [*own["rows_trained"][:-1], own["rows_trained"][0]]}),
           "`published`: it_was_trained_on_its_set_and_on_nothing_else")
    # ... one whose adapter, read back, is at another rank than the start adapter's.
    failed(store.path("ladder_l5_draws_train_shallow.done.json"), as_json(lambda own: {**own, "adapter_saved": {**own["adapter_saved"], "rank": 16}}),
           "`shallow`: its_adapter_is_not_the_start_adapters_and_is_read_back_at_its_rank")
    # A ROW OF A GOAL PROBLEM (a held-out problem, and a problem of G): barred, and of G. A ROW OF A HELD-ASIDE PROBLEM (a base-map problem, so not barred): of the held-aside set.
    own_row = next(index for index, row in enumerate(store.read_rows("l5_draws_training_deep.jsonl")) if row["origin"] == "deep")
    of_problem = lambda problem: as_rows(lambda own: [{**row, "problem_id": problem} if index == own_row else row for index, row in enumerate(own)])      # noqa: E731
    of_g = failed(store.path("l5_draws_training_deep.jsonl"), of_problem(GOAL), "`deep`: no_barred_row", "`deep`: no_problem_of_g_or_of_the_held_aside_set")["can_this_run_be_read"][DEEP]
    assert (of_g["no_barred_row"]["held_out_problems"], of_g["no_barred_row"]["first"][0]) == (1, GOAL)
    assert (of_g["no_problem_of_g_or_of_the_held_aside_set"]["problems_of_the_goal_set"], of_g["no_problem_of_g_or_of_the_held_aside_set"]["problems_of_the_held_aside_set"]) == (1, 0)
    of_the_set = failed(store.path("l5_draws_training_deep.jsonl"), of_problem(HELD), "`deep`: no_problem_of_g_or_of_the_held_aside_set")["can_this_run_be_read"][DEEP]
    assert (of_the_set["no_problem_of_g_or_of_the_held_aside_set"]["problems_of_the_goal_set"], of_the_set["no_problem_of_g_or_of_the_held_aside_set"]["problems_of_the_held_aside_set"]) == (0, 1)
    assert of_the_set["no_barred_row"]["held_out_problems"] == 0
    # THOSE CHECKS READ WHAT THE TRAINING LOOP RECORDED, not the stored set: a stored row of a goal problem that the loop did not record is a row not trained on, and no row of G.
    set_path, loss_path = store.path("l5_draws_training_deep.jsonl"), store.path("l5_draws_loss_deep.json")
    kept_set, kept_loss = set_path.read_text(), loss_path.read_text()
    row_id = store.read_rows("l5_draws_training_deep.jsonl")[own_row]["id"]
    set_path.write_text(of_problem(GOAL)(kept_set))
    loss_path.write_text(as_json(lambda own: {**own, "rows_trained": [key for key in own["rows_trained"] if key != row_id]})(kept_loss))
    found = report()
    assert found["failed_checks"] == ["`deep`: it_was_trained_on_its_set_and_on_nothing_else"]
    assert (found["can_this_run_be_read"][DEEP]["no_problem_of_g_or_of_the_held_aside_set"]["problems_of_the_goal_set"], found["can_this_run_be_read"][DEEP]["no_barred_row"]["held_out_problems"]) == (0, 0)
    set_path.write_text(kept_set)
    loss_path.write_text(kept_loss)
    assert report()["inconclusive"] is False
    # THE SETTING NOW NAMES ANOTHER NUMBER OF ATTEMPTS than the stored first sampling has (L5n's with it: the settings' reader holds the two alike): the start model's stored
    # sampling is held to the setting AS IT IS NOW, and not to what the run recorded of it.
    of_draws, of_noise = config["ladder_loop"]["l5_draws"], config["ladder_loop"]["l5_noise"]
    of_draws["attempts"] = of_noise["attempts"] = 48
    found = report()
    assert found["failed_checks"] == ["the run: the_stored_first_sampling_of_the_start_model_is_the_settings"] and found["inconclusive"] is True
    assert found["can_this_run_be_read"]["the_run"]["the_stored_first_sampling_of_the_start_model_is_the_settings"]["attempts_a_problem_that_are_not_the_settings"] == [32]
    of_draws["attempts"] = of_noise["attempts"] = 32
    assert report()["inconclusive"] is False


def test_a_set_lean_did_not_answer_fails_the_report_and_is_sampled_again_from_its_adapter(draws, monkeypatch):  # noqa: F811
    config = draws.config
    _stage5d(config, monkeypatch, until="ladder_l5_draws_measure_deep")
    store = draws.store()
    path = store.path("episodes_l4_reach_shallow_problems.jsonl")
    kept = path.read_text()
    path.write_text("".join(json.dumps({**row, "attempts_without_an_answer": 4}) + "\n" for row in _file_rows(path)))       # 4 of 32: more than 0.02
    report = ladder_l5_draws.ladder_l5_draws_report(config)
    assert report["ok"] is False and report["not_to_be_read"] == ["shallow"] and report["inconclusive"] is True and report["failed_checks"] == ["`shallow`: lean_answered"]
    assert report["of_this_seed"]["ok"] is False and any("NOT TO BE READ: too many attempts without a verdict from Lean in the sets of `shallow`" in line for line in report["lines"])
    assert store.done_summary("ladder_l5_draws_report")["ok"] is False and "NOT TO BE READ" in report["headline"]
    # QUEUED AGAIN: that set, and no other, is sampled again from the kept adapter; no model is trained again.
    asked, trained = len(draws.solver.measured), len(draws.calls)
    again = _stage5d(config, monkeypatch)
    assert len(draws.solver.measured) == asked + 1 and len(draws.calls) == trained and again["ladder_l5_draws_measure_shallow"]["measured_again"] == ["l4_reach_shallow"]
    assert again["ladder_l5_draws_report"]["ok"] is True and again["ladder_l5_draws_report"]["inconclusive"] is False and path.read_text() != kept


# ------------------------------------------------------------------------------------------ the other recipes
def test_with_the_decay_each_model_is_trained_once_with_l5ns_decay_and_its_recorded_rates_are_held_to_it(draws, monkeypatch):  # noqa: F811
    config, calls = draws.config, draws.calls
    _recipe(draws, DECAY)
    summaries = _stage5d(config, monkeypatch)
    store = draws.store()
    assert store.root.name == "ladder_l5_draws_pre_r64_seed3" and sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == _own_files(DECAY)
    assert [step for step in STEPS if summaries[step].get("run") is not False] == _used(DECAY) == _used(CONSTANT)
    read = summaries["ladder_l5_draws_prepare"]
    of_the_recipe = read["the_recipe"]
    assert (of_the_recipe["recipe"], of_the_recipe["asked_for_by_l5n"], of_the_recipe["l5n_named_it"], of_the_recipe["said_beside_every_read"]) == ("decay", "decay", True, None)
    assert of_the_recipe["found"] == f"L5n's report names {WITH_THE_DECAY}" and of_the_recipe["of_l5n"]["what_follows"] == WITH_THE_DECAY and of_the_recipe["of_l5n"]["there"] is True
    assert read["recipe"]["learning_rate_factor"] == {"what": read["recipe"]["learning_rate_factor"]["what"], "steps": 8, "warmup_steps": 5,
                                                      "decay": {"1": 0.2, "4": 0.8, "5": 1.0, "6": 0.75, "7": 0.5, "8": 0.25}}
    assert (read["trained"], read["averaged"]) == (list(MODELS), [])
    # ONE CHANGE: the decay is asked for, each step's rate is recorded beside its row, and the summary says what it was held to. The sets are the recipe-as-it-is's: L5's rule.
    start = draws.deep.rank / "adapters" / "pre_r64"
    assert [call["more"] for call in calls] == [{"start": start, "learning_rates": True, "decay": True}] * 3 and [call["seed"] for call in calls] == [3] * 3
    rates = learning_rates(RATE, WARMUP, 8, True)
    for model in MODELS:
        train, loss = summaries[f"ladder_l5_draws_train_{model}"], json.loads(store.path(f"l5_draws_loss_{model}.json").read_text())
        assert [row["learning_rate"] for row in loss["training_steps"]] == rates and rates[5:] == [RATE * 0.75, RATE * 0.5, RATE * 0.25]
        held = train["learning_rate"]
        assert (held["decay"], held["steps"], held["of_the_first_step"], held["of_the_last_step"]) == (True, 8, RATE / 5, RATE * 0.25) and "the task's seed" in train["order"]
        assert [row["id"] for row in store.read_rows(f"l5_draws_training_{model}.jsonl")] == loss["rows_trained"]
    report = summaries["ladder_l5_draws_report"]
    assert report["inconclusive"] is False and report["recipe"] == "decay" and report["of_this_seed"]["recipe"] == "decay" and not any("NOTE:" in line for line in report["lines"])
    assert any(f"THE RECIPE is `decay`: L5n's report names {WITH_THE_DECAY}" in line and "SAID BESIDE" not in line for line in report["lines"]) and "one training a model, with L5n's decay" in report["lines"][0]
    # UNDER ANOTHER RECIPE THAN L5's, L5's own seeds do not stand beside these: none is read.
    assert report["the_seeds_of_l5"] == {}
    # A TRAINING ASKED FOR THE DECAY THAT RECORDED THE RECIPE'S OWN RATES is refused, and its step is not marked (L4t's body, as L5n's trainings).
    monkeypatch.setenv(SEEDS, "4")
    ladder_l5_draws.ladder_l5_draws_prepare(config)
    _trainings(monkeypatch, calls, rates=lambda more, recorded: learning_rates(RATE, WARMUP, len(recorded), False))
    with pytest.raises(RuntimeError, match="`published`: the learning rate recorded at step 6 is 0.0001 and the schedule with the decay gives 7.5\\d*e-05"):
        _on_the_gpu(monkeypatch, ladder_l5_draws.STEPS["ladder_l5_draws_train_published"], config)
    assert not draws.store().is_done("ladder_l5_draws_train_published")
    with pytest.raises(RuntimeError, match="the measurement of L5d's model `published` needs the step ladder_l5_draws_train_published of this run, which is not done"):
        ladder_l5_draws.STEPS["ladder_l5_draws_measure_published"](config)


def test_with_the_average_of_four_each_set_is_trained_in_four_orders_and_the_model_measured_is_the_mean_of_the_four(draws, monkeypatch):  # noqa: F811
    config, calls, kits, solver = draws.config, draws.calls, draws.kits, draws.solver
    _recipe(draws, MEAN_CONSTANT)
    summaries = _stage5d(config, monkeypatch)
    store = draws.store()
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == _own_files(MEAN_CONSTANT)
    assert [step for step in STEPS if summaries[step].get("run") is not False] == _used(MEAN_CONSTANT) and len(_used(MEAN_CONSTANT)) == 20
    for model in MODELS:                                                        # THE ONE TRAINING IN THE SEED'S OWN ORDER is no step of this recipe
        own = summaries[f"ladder_l5_draws_train_{model}"]
        assert own["run"] is False and "a model is trained in four orders, and never once in the seed's own" in own["why"] and not store.is_done(f"ladder_l5_draws_train_{model}")
    read = summaries["ladder_l5_draws_prepare"]
    trained = [f"{model}_{order}" for model in MODELS for order in ORDERS]
    assert (read["trained"], read["averaged"], read["the_recipe"]["recipe"], read["the_recipe"]["l5n_named_it"]) == (trained, list(MODELS), "mean_constant", True)
    assert read["the_averaged"] == {model: {"what": read["the_averaged"][model]["what"], "of": [f"{model}_{order}" for order in ORDERS]} for model in MODELS}
    assert read["recipe"]["learning_rate_factor"]["constant"] == {"1": 0.2, "4": 0.8, "5": 1.0, "6": 1.0, "7": 1.0, "8": 1.0} and "decay" not in read["recipe"]["learning_rate_factor"]
    # EACH SET, by L5's rule for the seed, IN THE ORDER OF EACH ORDER SEED: the same eight rows; the order seed that is the task's own seed (3) gives the seed's own order, and is NAMED.
    sets = {model: store.read_rows(f"l5_draws_training_{model}.jsonl") for model in MODELS}
    placed = lambda row: (row["id"], row["problem_id"], row.get("row_of_the_file"))      # noqa: E731
    for model in MODELS:
        for order in ORDERS:
            own, entry_ = store.read_rows(f"l5_draws_training_{model}_{order}.jsonl"), read["training_sets"][f"{model}_{order}"]
            assert [placed(row) for row in own] == [placed(row) for row in training_order(sets[model], order)] and [row["row"] for row in own] == list(range(8))
            assert ([placed(row) for row in own] == [placed(row) for row in sets[model]]) is (order == 3) and (entry_["order_seed"], entry_["model"], entry_["rows"], entry_["optimizer_steps"]) == (order, model, 8, 8)
            assert not any(field in row for row in own if row["origin"] == "pretraining" for field in TEXT_FIELDS)
        assert read["the_orders"][model] == {"the_order_of_the_tasks_seed": [3], "alike": []}
    assert read["the_orders"]["order_seeds"] == [1, 2, 3, 4]
    # TWELVE TRAININGS, model by model and order by order, each the recipe as it is with its rates recorded (as L5n's `constant_k`), with the task's seed for the adapter.
    start = draws.deep.rank / "adapters" / "pre_r64"
    assert [call["more"] for call in calls] == [{"start": start, "learning_rates": True}] * 12 and [call["seed"] for call in calls] == [3] * 12
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_draws_pre_r64_seed3_{name}" for name in trained]
    constant = learning_rates(RATE, WARMUP, 8, False)
    for call, name in zip(calls, trained):
        train, loss = summaries[f"ladder_l5_draws_train_{name}"], json.loads(store.path(f"l5_draws_loss_{name}.json").read_text())
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in store.read_rows(f"l5_draws_training_{name}.jsonl")]
        assert [row["learning_rate"] for row in loss["training_steps"]] == constant and train["learning_rate"]["decay"] is False and f"the ORDER SEED {name[-1]}" in train["order"]
        assert loss["rows_trained"] == [row["id"] for row in store.read_rows(f"l5_draws_training_{name}.jsonl")] and train["adapter"] == str(store.root / "adapters" / name)
    # THREE AVERAGED ADAPTERS, trained by nothing, by L5n's own function: every number the mean of its four's at that place; read back; none of them; at rank 64.
    for index, model in enumerate(MODELS):
        made, numbers = summaries[f"ladder_l5_draws_average_{model}"], [4 * index + order for order in ORDERS]                    # the k-th training's adapter holds k and k / 2
        directory, of = store.root / "adapters" / model, [store.root / "adapters" / f"{model}_{order}" for order in ORDERS]
        assert (made["model"], made["recipe"], made["of"], made["adapter"], made["averaged"], made["check"]) == (model, "mean_constant", [f"{model}_{order}" for order in ORDERS], str(directory), True, "L5d")
        a = "base_model.model.layers.0.q_proj.lora_A.weight"
        assert set(_numbers(directory / "adapter_model.safetensors", a).tolist()) == {sum(numbers) / 4} and made["read_back"] == is_the_mean(directory / "adapter_model.safetensors", [path / "adapter_model.safetensors" for path in of])
        assert made["read_back"]["passes"] is True and made["adapter_saved"]["rank"] == 64 and made["one_of_its_models_in_a_smoke_run"] is False
        assert made["against_the_start_adapter"] == compared(directory / "adapter_model.safetensors", start / "adapter_model.safetensors")
    assert len(calls) == 12
    # THREE MEASUREMENTS, of the averaged adapters and of nothing else, each right after its adapter is made.
    assert solver.measured == [(GOAL, 32, 1001, 1024)] * 3 and [kit["max_lora_rank"] for kit in kits] == [64] * 3
    order_of = list(summaries)
    assert all(order_of.index(f"ladder_l5_draws_measure_{model}") == order_of.index(f"ladder_l5_draws_average_{model}") + 1 for model in MODELS)
    assert sorted(path.name for path in store.root.glob("episodes_*.done.json") if "_block_" not in path.name) == sorted(f"episodes_l4_reach_{model}.done.json" for model in MODELS)
    report = summaries["ladder_l5_draws_report"]
    assert report["inconclusive"] is False and report["ok"] is True and report["recipe"] == "mean_constant" and report["the_orders"] == read["the_orders"]
    checks = report["can_this_run_be_read"]
    assert list(checks) == ["what", "the_run", *trained, *MODELS] and all(list(checks[model]) == ["its_numbers_are_the_mean_of_its_models_and_it_is_served", "lean_answered"] for model in MODELS)
    assert all(checks[model]["its_numbers_are_the_mean_of_its_models_and_it_is_served"]["passes"] is True and checks[model]["its_numbers_are_the_mean_of_its_models_and_it_is_served"]["served"] is True for model in MODELS)
    assert checks["the_run"]["each_set_is_l5s_rule_for_its_seed_made_again_and_held"]["sets_not_in_their_order_seeds_order"] == []
    assert any("CHECK, each averaged adapter's numbers ARE the mean of its four's" in line and line.endswith(": PASS") for line in report["lines"])
    assert report["adapters"]["there"] == sorted([*MODELS, *trained]) and list(report["models_made"][DEEP]["trainings"]) == list(its_trainings(MEAN_CONSTANT, DEEP)) and "averaging" in report["models_made"][DEEP]
    # A SET THAT IS NOT IN ITS ORDER SEED'S ORDER (two of its stored rows changed places) fails the first check: the seed is INCONCLUSIVE.
    path = store.path("l5_draws_training_shallow_2.jsonl")
    kept = path.read_text()
    own = [json.loads(line) for line in kept.splitlines()]
    path.write_text("".join(json.dumps(row) + "\n" for row in [{**own[1], "row": 0}, {**own[0], "row": 1}, *own[2:]]))
    failed = ladder_l5_draws.ladder_l5_draws_report(config)
    assert failed["failed_checks"] == ["the run: each_set_is_l5s_rule_for_its_seed_made_again_and_held"] and failed["can_this_run_be_read"]["the_run"]["each_set_is_l5s_rule_for_its_seed_made_again_and_held"][
        "sets_not_in_their_order_seeds_order"] == ["shallow_2"]
    path.write_text(kept)
    # AN AVERAGED ADAPTER THAT IS ONE OF ITS MODELS (its step's stored read-back is made to say so) fails its check in a real run: a smoke run alone keeps one.
    path = store.path("ladder_l5_draws_average_shallow.done.json")
    kept = path.read_text()
    own = json.loads(kept)
    path.write_text(json.dumps({**own, "read_back": {**own["read_back"], "is_none_of_them": False, "passes": False}}))
    failed = ladder_l5_draws.ladder_l5_draws_report(config)
    check = failed["can_this_run_be_read"][SHALLOW]["its_numbers_are_the_mean_of_its_models_and_it_is_served"]
    assert failed["failed_checks"] == ["`shallow`: its_numbers_are_the_mean_of_its_models_and_it_is_served"] and (check["is_the_mean"], check["is_none_of_them"], check["one_of_its_models_in_a_smoke_run"]) == (True, False, False)
    path.write_text(kept)
    # AN AVERAGED MODEL IS MEASURED ONLY ONCE ITS OWN STEP HAS MADE ITS ADAPTER, and that step needs its four trainings.
    monkeypatch.setenv(SEEDS, "4")
    _stage5d(config, monkeypatch, until="ladder_l5_draws_train_published_3")
    with pytest.raises(RuntimeError, match="the averaged adapter `published` needs the step ladder_l5_draws_train_published_4 of this run, which is not done"):
        _on_the_gpu(monkeypatch, ladder_l5_draws.STEPS["ladder_l5_draws_average_published"], config)
    with pytest.raises(RuntimeError, match="the measurement of L5d's model `published` needs the step ladder_l5_draws_average_published of this run, which is not done"):
        ladder_l5_draws.STEPS["ladder_l5_draws_measure_published"](config)
    with pytest.raises(RuntimeError, match="L5d's report needs the step ladder_l5_draws_train_published_4 of this run, which is not done"):
        ladder_l5_draws.ladder_l5_draws_report(config)
    assert not draws.store().is_done("ladder_l5_draws_average_published") and len(solver.measured) == 3
    # FOUR TRAININGS THAT LEFT ONE ADAPTER (here they are made alike): their mean is each of them, an average of nothing. A REAL run refuses it, and its step is not marked done.
    monkeypatch.setenv(SEEDS, "5")
    _trainings(monkeypatch, calls, value=lambda number: 1.0)
    _stage5d(config, monkeypatch, until="ladder_l5_draws_train_published_4")
    with pytest.raises(RuntimeError, match="the adapter `published` saved at .*adapters/published, read back, has 0 of .* numbers that are not the mean of its models', differs from each of them in .*"
                                           "it is the mean of its models, none of them, and at the start adapter's rank, or nothing of it is measured. The step is not marked done"):
        _on_the_gpu(monkeypatch, ladder_l5_draws.STEPS["ladder_l5_draws_average_published"], config)
    assert not draws.store().is_done("ladder_l5_draws_average_published") and len(solver.measured) == 3


def test_as_on_the_gpu_every_measurement_serves_the_adapter_of_its_own_model_the_averaged_one_too_and_the_start_adapters_type_is_held(draws, monkeypatch):  # noqa: F811
    config, kits = draws.config, draws.kits
    _recipe(draws, MEAN_CONSTANT)
    # The stand-in engine takes no adapter, so in the scripted world none is asked for. As on the GPU every measurement asks the model server for ITS model's adapter.
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        start = draws.deep.rank / "adapters" / "pre_r64" / "adapter_model.safetensors"
        own_root = draws.runs / "ladder_l5_draws_pre_r64_seed3"
        # THE START ADAPTER's numbers are of a type the averaged adapters cannot be written in: refused by the FIRST step, before anything is written or trained.
        kept = start.read_bytes()
        _write(start, [("base_model.model.layers.0.q_proj.lora_A.weight", "BF16", (2,), struct.pack("<2H", 0x3F80, 0x4000))])
        with pytest.raises(RuntimeError, match=r"adapters/pre_r64 holds tensors of type \['BF16'\]: the averaged adapters are written in \['F16', 'F32', 'F64'\].*Nothing was written"):
            ladder_l5_draws.ladder_l5_draws_prepare(config)
        assert not own_root.exists() or not any(own_root.iterdir())
        # ... and under a recipe that averages nothing its type is nothing to the run: the same start adapter, the recipe as it is, another seed's task, prepared.
        _recipe(draws, CONSTANT)
        patched.setenv(SEEDS, "4")
        assert ladder_l5_draws.ladder_l5_draws_prepare(config)["averaged"] == []
        patched.setenv(SEEDS, "3")
        _recipe(draws, MEAN_CONSTANT)
        start.write_bytes(kept)
        summaries = _stage5d(config, patched)
        store = draws.store()
        kept_here = store.root / "adapters"
        assert [kit["adapter"] for kit in kits] == [(f"ladder_l5_draws_{model}", 1, str(kept_here / model)) for model in MODELS] and [kit["max_lora_rank"] for kit in kits] == [64] * 3
        assert summaries["ladder_l5_draws_measure_deep"]["stand_in_engine"] is False and summaries["ladder_l5_draws_report"]["inconclusive"] is False
        # An adapter that is no longer there: the measurement is refused, not made from another model's.
        shutil.rmtree(kept_here / "shallow")
        store.path("ladder_l5_draws_measure_shallow.done.json").unlink()
        with pytest.raises(RuntimeError, match="adapters/shallow is not there: the adapter `shallow` is kept by this stage"):
            ladder_l5_draws.STEPS["ladder_l5_draws_measure_shallow"](config)
        assert not store.is_done("ladder_l5_draws_measure_shallow")
        # AN AVERAGED ADAPTER IS MADE AGAIN when its step is queued again (its four are kept): the same numbers; but not when one of its four is gone.
        store.path("ladder_l5_draws_average_shallow.done.json").unlink()
        assert ladder_l5_draws.STEPS["ladder_l5_draws_average_shallow"](config)["read_back"]["passes"] is True
        shutil.rmtree(kept_here / "deep_2")
        store.path("ladder_l5_draws_average_deep.done.json").unlink()
        with pytest.raises(RuntimeError, match="adapters/deep_2.. are not there: the adapters `deep` is the mean of are kept by this stage. Nothing was averaged"):
            ladder_l5_draws.STEPS["ladder_l5_draws_average_deep"](config)
        assert not store.is_done("ladder_l5_draws_average_deep")


# ----------------------------------------------------------------------------------------------- the refusals
def test_the_first_step_refuses_before_anything_is_written(draws, monkeypatch):  # noqa: F811
    config, root, runs = draws.config, draws.l5, draws.runs
    settings = config["ladder_loop"]["l5_draws"]
    task = "the task of stage `ladder_l5_deep` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l5_deep --seeds 0`; for the smoke run, stage `ladder_l5_deep_smoke`."
    why = "L5d runs no search: a seed of it is a further seed of L5, and draws its three sets from the finished search of seed 0"

    def nothing_written():
        return not [path for path in runs.iterdir() if path.name.startswith("ladder_l5_draws") and any(path.iterdir())]

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l5_draws.ladder_l5_draws_prepare(config)
        assert nothing_written()                                                # NOTHING WAS WRITTEN, in this seed's run or in any other

    def with_changed(path, change, match):
        kept = path.read_text()
        path.write_text(change(kept))
        refused(match)
        path.write_text(kept)

    def without(path, match):
        path.rename(path.with_name(path.name + ".away"))
        refused(match)
        path.with_name(path.name + ".away").rename(path)

    def with_setting(key, value, match, error=RuntimeError, of=None):
        of = settings if of is None else of
        kept, of[key] = of[key], value
        refused(match, error)
        of[key] = kept

    as_json = lambda change: (lambda text: json.dumps(change(json.loads(text))))      # noqa: E731
    # ---- THE SEED: one of L5's own (they found the sign); one of the further ones without explicit approval; one that is in neither list
    for seed in (0, 1, 2):
        monkeypatch.setenv(SEEDS, str(seed))
        refused(rf"seed {seed} is one of L5's own .\[0, 1, 2\].: those seeds found the sign, and this stage reads it on fresh seeds alone .\[3, 4, 5\]; \[6, 7, 8\] with explicit approval. "
                r".ladder_loop.l5_draws.. Nothing was written")
    for seed in (6, 7, 8):
        monkeypatch.setenv(SEEDS, str(seed))
        refused(rf"seed {seed} is one of the further seeds \[6, 7, 8\], which run with explicit approval, and the approval is not recorded .more_seeds_approved is false.: more than three seeds "
                "needs it in advance .ladder_loop.l5_draws.. Nothing was written")
    monkeypatch.setenv(SEEDS, "9")
    refused(r"seed 9 is none of the fresh seeds \[3, 4, 5\] .nor of \[6, 7, 8\], which need explicit approval.: the number of seeds is fixed before the read")
    settings["more_seeds_approved"] = True
    refused(r"seed 9 is none of the fresh seeds \[3, 4, 5\] and \[6, 7, 8\]: the number of seeds is fixed before the read, and a seed outside it is no seed of this stage")
    settings["more_seeds_approved"] = False
    monkeypatch.setenv(SEEDS, "3")
    # ---- THE RECIPE, HELD TO L5n's REPORT. L5n's run is not there: the recipe as it is, and any other is refused, naming both.
    said = r".ladder_loop.l5_draws.recipe; L5n's run is written by the task of stage `ladder_l5_noise` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l5_noise --seeds 0`... Nothing was written"
    for recipe in (DECAY, MEAN_CONSTANT):
        with_setting("recipe", recipe, f"the setting names the recipe `{recipe}` and L5n's run is not there or not finished .ladder_l5_noise_pre_r64_seed0 does not hold .*which asks for "
                                       "`constant`: the recipe is the one L5n names, taken as named. Change the setting, or finish L5n's run " + said)
    # ... its report names a case: the setting is the recipe that case asks for, and any other is refused, naming both.
    of_l5n = runs / "ladder_l5_noise_pre_r64_seed0"
    for name, asked in ((WITH_THE_DECAY, DECAY), (AVERAGE_OF_FOUR, MEAN_CONSTANT), (THE_RECIPES_OWN, CONSTANT)):
        _l5ns_report(runs, name)
        for recipe in (CONSTANT, DECAY, MEAN_CONSTANT):
            if recipe != asked:
                with_setting("recipe", recipe, f"the setting names the recipe `{recipe}` and L5n's report names {name} .{of_l5n}., which asks for `{asked}`.*" + said)
    # ... its report is INCONCLUSIVE, not to be read, or names no case; or its report step is not marked done: the recipe as it is, and no other.
    for changed in ({"inconclusive": True}, {"ok": False}, {"branches": {"what_follows": NOT_READ}}):
        _l5ns_report(runs, WITH_THE_DECAY, **changed)
        with_setting("recipe", DECAY, "the setting names the recipe `decay` and L5n's report (reads INCONCLUSIVE|is not to be read|names no case for what follows .NOT READ.) .*which asks for `constant`")
    _l5ns_report(runs, WITH_THE_DECAY)
    (of_l5n / "ladder_l5_noise_report.done.json").unlink()
    with_setting("recipe", DECAY, r"L5n's run is not there or not finished .ladder_l5_noise_pre_r64_seed0 does not hold \['ladder_l5_noise_report.done.json'\].*which asks for `constant`")
    shutil.rmtree(of_l5n)
    # ---- L5's RUN MISSING; UNFINISHED: a step's marker, its report or a draw is not there (L5's own reading, in this stage's words)
    root.rename(root.with_name("elsewhere"))
    refused(f"ladder_l5_deep_pre_r64_seed0 does not hold .*the FINISHED run of the deep search. {why}. Run {task} to its end first; nothing was written")
    root.with_name("elsewhere").rename(root)
    for name in ("ladder_l5_prepare.done.json", "ladder_l5_report.done.json", "report_ladder_l5.json", "l5_search_problems.jsonl"):
        without(root / name, f"ladder_l5_deep_pre_r64_seed0 does not hold .*{name}.*the FINISHED run of the deep search. L5d runs no search")
    # ---- A REPORT NOT TO BE READ, or INCONCLUSIVE: no model here stands beside such a run
    with_changed(root / "report_ladder_l5.json", as_json(lambda report: {**report, "ok": False}), "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is not to be read .*: no model here stands beside such a run")
    with_changed(root / "report_ladder_l5.json", as_json(lambda report: {**report, "inconclusive": True, "failed_checks": ["`shallow`: its_training_ran"]}),
                 "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .`shallow`: its_training_ran.: no model here stands beside such a run")
    # ---- MADE OTHERWISE (L5's own holds): another rank recorded for the start adapter
    with_changed(root / "ladder_l5_search_prepare.done.json", as_json(lambda own: {**own, "start_recipe": {**own["start_recipe"], "rank": 16}}),
                 "was made with another start model, rank, search or held-aside set than this task would make: the start adapter's rank and alpha .*L5d's models are trained on rows drawn from that run's search")
    # ---- THE RECIPE is the one that trained L5's models: another learning rate, warm-up or batch now is refused
    for key, value in (("learning_rate", 5e-5), ("warmup_steps", 3), ("effective_batch", 8)):
        with_setting(key, value, f"L5's run ladder_l5_deep_pre_r64_seed0 trained its models with .* and this task would train with .*'{key}': {value!r}.*a seed here is a further seed of L5, and "
                                 "`constant` is the recipe that run's models were trained with. Nothing was written", of=config["training"])
    # ---- THE SAMPLING is the start model's FIRST, with its recorded attempts, as L5n measures: a setting that is not L5n's is refused by the settings' reader; one that is L5n's
    # and is not what the start model's run recorded, by the step
    with_setting("sampling", "more", "ladder_loop.l5_draws names the sampling `more` of 32 attempts a problem and ladder_loop.l5_noise `reach` of 32: a model here is measured as L5n measures", ValueError)
    with_setting("attempts", 48, "ladder_loop.l5_draws names the sampling `reach` of 48 attempts a problem and ladder_loop.l5_noise `reach` of 32", ValueError)
    of_noise = config["ladder_loop"]["l5_noise"]
    kept, of_noise["attempts"], settings["attempts"] = of_noise["attempts"], 48, 48
    refused("ladder_loop.l5_draws names the sampling `reach` of 48 attempts a problem, and the run of `pre_r64` .ladder_l4_pretrain_r64_seed0. recorded its first sampling of the goal set as `reach` of 32.*Nothing was written")
    of_noise["attempts"], settings["attempts"] = kept, kept
    # ---- A SET WITH A PROBLEM OF G, OR OF THE HELD-ASIDE SET (L5's rule gives none: here its sets are made to hold one)
    by_the_rule = ladder_l5.sets_by_the_seed
    for problem, said_ in ((GOAL, "1 problems of the training set of `deep` are goal problems and 0 are held-aside problems .first: fixture_h3."),
                           (HELD, f"0 problems of the training set of `deep` are goal problems and 1 are held-aside problems .first: {HELD}.")):
        def with_one(*arguments, problem=problem):
            made = by_the_rule(*arguments)
            made["sets"][DEEP][0] = {**made["sets"][DEEP][0], "problem_id": problem}
            return made
        monkeypatch.setattr(ladder_l5, "sets_by_the_seed", with_one)
        refused(said_ + ": refused, nothing was written .seed 3; L5's run ladder_l5_deep_pre_r64_seed0.. Nothing was written")
    monkeypatch.setattr(ladder_l5, "sets_by_the_seed", by_the_rule)
    # ---- L5's OWN REFUSALS, in the two functions both stages draw by. THE SEARCH FOUND AGAIN IN THE STORED ROWS IS NOT THE ONE ITS OWN TASK COUNTED (that task's three sets step
    # is made to have recorded one deep solve more)
    with_changed(root / "ladder_l5_prepare.done.json", as_json(lambda own: {**own, "deep_solves": own["deep_solves"] + 1}),
                 "the deep search read from ladder_l5_deep_pre_r64_seed0 is not the one its own task built its sets on: this task counts 4 deep solves and D = 4, that run 5 and 4")
    # ... THE MAP IS NOT THE ONE THE SEARCH WAS PREPARED ON; A SET WITH A HELD-OUT PROBLEM (L5's rule gives none: here what L5's reading hands on is made to say so)
    by_l5 = ladder_l5.read_for_the_search
    handed = lambda change: monkeypatch.setattr(ladder_l5, "read_for_the_search", lambda *arguments, **more: change(by_l5(*arguments, **more)))      # noqa: E731
    handed(lambda made: {**made, "by_k": {**made["by_k"], min(made["by_k"]): made["by_k"][min(made["by_k"])] + 1}})
    refused(f"the map {draws.deep.map.name} is not the map this run's search was prepared on .its problems by k are no longer .*.. Nothing was written")
    handed(lambda made: {**made, "groups": [*made["groups"], {**made["groups"][0], "problem_id": DRAWN[0]}]})
    refused(f"1 problems of the training set of `deep` are held-out problems .first: {DRAWN[0]}.: refused, nothing was written .the map {draws.deep.map.name}; `pre_r64`'s .*. Nothing was written")
    monkeypatch.setattr(ladder_l5, "read_for_the_search", by_l5)
    # ---- FEWER DEEP SOLVES THAN ARE ASKED (L5's own report is INCONCLUSIVE then, and refused above: here the count alone is made short)
    by_the_seed = ladder_l5.deep_solves_by_the_seed
    monkeypatch.setattr(ladder_l5, "deep_solves_by_the_seed", lambda *arguments: {**by_the_seed(*arguments), "enough": {"passes": False, "minimum": 150}})
    refused("the deep search of ladder_l5_deep_pre_r64_seed0 found 4 deep solves and 150 are asked: no set is built on it, and L5d has nothing to train. Nothing was written")
    monkeypatch.setattr(ladder_l5, "deep_solves_by_the_seed", by_the_seed)
    # ---- UNDER THE AVERAGED RECIPE, two order seeds that give one order of a set: the averaged adapter would be the mean of fewer than four trainings
    _recipe(draws, MEAN_CONSTANT)
    in_its_order = l5_draws.the_orders
    monkeypatch.setattr(l5_draws, "the_orders", lambda rows_, orders, loosely=False: (lambda made: (made[0], {**made[1], "alike": [[1, 2]]}))(in_its_order(rows_, orders, loosely=loosely)))
    refused(r"of the order seeds \[1, 2, 3, 4\], the pairs \[\[1, 2\]\] give one order of these 8 rows: the averaged adapter is the mean of four trainings that differ by their order. Refused "
            ".seed 3; L5's run ladder_l5_deep_pre_r64_seed0.. Nothing was written")
    monkeypatch.setattr(l5_draws, "the_orders", in_its_order)
    shutil.rmtree(of_l5n)
    settings["recipe"] = CONSTANT
    # ---- THE SETTINGS: a recipe that is none of the three; seeds that are not distinct whole numbers or hold one of L5's own; an approval that is no yes or no; a number of seeds
    # the interval over seeds is not given for
    for key, value, match in (("recipe", "cosine", r"ladder_loop.l5_draws.recipe is 'cosine': one of \['constant', 'decay', 'mean_constant'\] .the one L5n's report names."),
                              ("seeds", [3, 4, 4], r"ladder_loop.l5_draws.seeds is \[3, 4, 4\] and more_seeds \[6, 7, 8\]: distinct whole numbers of at least 0, none of them a seed of L5's own "
                                                   r".\[0, 1, 2\]: ladder_loop.l5_noise.published_seeds., which found the sign"),
                              ("seeds", [2, 3, 4], r"ladder_loop.l5_draws.seeds is \[2, 3, 4\] and more_seeds \[6, 7, 8\]: distinct whole numbers"),
                              ("seeds", [3, 4, 6], r"ladder_loop.l5_draws.seeds is \[3, 4, 6\] and more_seeds \[6, 7, 8\]: distinct whole numbers"),
                              ("more_seeds", [6, -7, 8], r"ladder_loop.l5_draws.seeds is \[3, 4, 5\] and more_seeds \[6, -7, 8\]: distinct whole numbers"),
                              ("more_seeds", [6, True, 8], r"more_seeds \[6, True, 8\]: distinct whole numbers"),
                              ("more_seeds_approved", "yes", "ladder_loop.l5_draws.more_seeds_approved is 'yes': true once the further seeds are approved, false until then"),
                              ("seeds", [3], r"ladder_loop.l5_draws.seeds holds 1 seeds and, with more_seeds, 4: the interval over seeds is given for \[2, 3, 4, 5, 6, 7, 8, 9\] seeds"),
                              ("more_seeds", [6, 7, 8, 9, 10, 11, 12], r"ladder_loop.l5_draws.seeds holds 3 seeds and, with more_seeds, 10: the interval over seeds is given for")):
        with_setting(key, value, match, ValueError)
    # ---- THE SMOKE-ONLY VARIABLE in a run of the seed's own directory: refused (the smoke stage's test holds the rest of its guard)
    monkeypatch.setenv(SMOKE, "1")
    refused("RLVR_LEAN_LADDER_L5_DRAWS_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_DRAWS_RUN. and on the fixtures' base map. A run in "
            "the seed's own directory reads a run of L5 whose checks passed. Nothing was written")
    monkeypatch.delenv(SMOKE)
    # ---- and with nothing changed it prepares: every refusal above left the runs as they were
    assert ladder_l5_draws.ladder_l5_draws_prepare(config)["the_search"]["rows_of_each_kind"] == 4


def test_a_further_seed_runs_once_explicit_approval_is_recorded_and_the_reads_are_then_over_the_six(draws, monkeypatch):  # noqa: F811
    config = draws.config
    monkeypatch.setenv(SEEDS, "6")
    with pytest.raises(RuntimeError, match="seed 6 is one of the further seeds"):
        ladder_l5_draws.ladder_l5_draws_prepare(config)
    config["ladder_loop"]["l5_draws"]["more_seeds_approved"] = True
    report = _stage5d(config, monkeypatch)["ladder_l5_draws_report"]
    store = draws.store()
    assert store.root.name == "ladder_l5_draws_pre_r64_seed6" and store.done_summary("ladder_l5_draws_prepare")["the_seeds"]["fixed"] == [3, 4, 5, 6, 7, 8] and [call["seed"] for call in draws.calls] == [6] * 3
    over = report["over_the_fresh_seeds"]
    assert (over["fixed_seeds"], over["seeds_there"], over["seeds_missing"], report["branches"]["primary"]) == ([3, 4, 5, 6, 7, 8], [6], [3, 4, 5, 7, 8], NOT_READ)
    assert hold_the_sets({model: store.read_rows(f"l5_draws_training_{model}.jsonl") for model in MODELS},
                         ladder_l4_start.pretrained_on(draws.deep.rank, store.done_summary("ladder_l5_draws_prepare")["pretraining_file_sha256"], "pre_r64")[0], 6) == 4
    # WITH THE APPROVAL TAKEN BACK the report is over the first three again, and this seed, which is there, is no seed of that number: named, and left out.
    config["ladder_loop"]["l5_draws"]["more_seeds_approved"] = False
    again = ladder_l5_draws.ladder_l5_draws_report(config)["over_the_fresh_seeds"]
    assert (again["fixed_seeds"], again["seeds_missing"], again["seeds_not_of_the_fixed_number"]) == ([3, 4, 5], [3, 4, 5], [6])


def test_a_run_keeps_its_recipe_and_seeds_of_two_recipes_are_not_read_together(draws, monkeypatch):  # noqa: F811
    config, runs = draws.config, draws.runs
    _stage5d(config, monkeypatch)                                               # seed 3, with the recipe as it is (L5n's run was not there)
    store = draws.store()
    own = _all(store.root)
    # L5n's REPORT COMES, AND NAMES THE DECAY: the setting is changed to it. SEED 3's RUN KEEPS ITS RECIPE: its first step returns what is stored, and every later step refuses.
    _recipe(draws, DECAY)
    assert ladder_l5_draws.ladder_l5_draws_prepare(config)["the_recipe"]["recipe"] == "constant"
    kept = "this run .ladder_l5_draws_pre_r64_seed3. was prepared with the recipe `constant` and ladder_loop.l5_draws.recipe now names `decay`: a run keeps its recipe, and seeds of two recipes are not read together. " \
           "Name the recipe it was prepared with, or move the run away. {} is not made"
    for step, what in (("ladder_l5_draws_measure_deep", "the measurement of `deep`"), ("ladder_l5_draws_report", "L5d's report"), ("ladder_l5_draws_average_deep", "the averaged adapter `deep`")):
        with pytest.raises(RuntimeError, match=kept.format(what)):
            ladder_l5_draws.STEPS[step](config)
    store.path("ladder_l5_draws_train_deep.done.json").rename(store.path("away.json"))
    with pytest.raises(RuntimeError, match=kept.format("the training of `deep`")):
        ladder_l5_draws.STEPS["ladder_l5_draws_train_deep"](config)
    store.path("away.json").rename(store.path("ladder_l5_draws_train_deep.done.json"))
    assert _all(store.root) == own
    # SEED 4 IS PREPARED WITH THE DECAY: its report reads seed 3's, of another recipe, and POOLS NOTHING: the read is NOT READ, and says why.
    monkeypatch.setenv(SEEDS, "4")
    report = _stage5d(config, monkeypatch)["ladder_l5_draws_report"]
    over = report["over_the_fresh_seeds"]
    assert report["recipe"] == "decay" and report["inconclusive"] is False and report["branches"] == {"primary": NOT_READ, "where_it_sits": NOT_READ, "what_follows": NOT_READ}
    assert (over["read"], over["seeds_there"]) == (False, [3, 4]) and over["refused"] == "the seeds were trained with two recipes (seed 3: `constant`, seed 4: `decay`): seeds of two recipes are not pooled"
    assert any("the seeds were trained with two recipes (seed 3: `constant`, seed 4: `decay`): seeds of two recipes are not pooled: nothing is read over the seeds" in line for line in report["lines"])
    # The first seed's run moved away, its task runs again under the recipe L5n names; the two then read together (seed 5 is still not there).
    (runs / "ladder_l5_draws_pre_r64_seed3").rename(runs / "the_run_with_the_recipe_as_it_is")
    monkeypatch.setenv(SEEDS, "3")
    again = _stage5d(config, monkeypatch)["ladder_l5_draws_report"]["over_the_fresh_seeds"]
    assert (again["seeds_there"], again["seeds_missing"], again["recipe"]) == ([3, 4], [5], "decay") and "refused" not in again


def test_the_steps_run_through_the_gpu_entry_point(draws, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(milestone2, "load_config", lambda path: draws.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    # A STEP THE RECIPE DOES NOT USE answers through the entry point too, with nothing made; in the stand-in's world no adapter is trained, so the report says INCONCLUSIVE.
    assert json.loads((tmp_path / "ladder_l5_draws_train_deep_2.json").read_text())["run"] is False and not draws.store().is_done("ladder_l5_draws_train_deep_2")
    report = json.loads((tmp_path / "ladder_l5_draws_report.json").read_text())
    assert draws.store().is_done(ladder_l5_draws.REPORT) and report["inconclusive"] is True and report["stand_in_engine"] is True and len(report["failed_checks"]) == 3


# ---------------------------------------------------------------------------------------- the smoke-only path
def _smoke(deep, monkeypatch):  # noqa: F811
    """The smoke runs' world (`_smoke_world`: L5's smoke run, INCONCLUSIVE, in a run the task names) and a task of seed 3
    that carries NEITHER of this stage's variables yet."""
    for name in (RUN, SMOKE):
        monkeypatch.delenv(name, raising=False)
    first = _smoke_world(deep, monkeypatch)
    deep.config["ladder_loop"]["l5_draws"]["recipe"] = CONSTANT     # the test states its recipe, as `draws` does: L5n's real run is not there in this world
    calls, measured = [], []
    _trainings(monkeypatch, calls, value=lambda number: 1.0)          # AS ON THE BOX: one optimizer step on the same two rows, whatever their order, is ONE training: adapters alike
    solver = NoiseSolver(deep.solver, (4, 8, 12))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (measured.append(given["vllm"]["max_lora_rank"]), (solver, stand_in_parameters))[1]))
    monkeypatch.setenv(SEEDS, "3")
    return SimpleNamespace(config=deep.config, first=first, calls=calls, measured=measured, solver=solver, runs=ladder_l5._runs(deep.config), of_l5=ladder_l5._runs(deep.config) / "ladder_l5_deep_smoke")


def test_the_smoke_stage_goes_on_from_l5s_inconclusive_smoke_run_builds_its_three_sets_of_two_rows_and_says_so(deep, monkeypatch):  # noqa: F811
    world = _smoke(deep, monkeypatch)
    config, runs, of_l5, calls, solver = world.config, world.runs, world.of_l5, world.calls, world.solver
    failed = world.first["ladder_l5_report"]["failed_checks"]
    assert failed != [] and len(_file_rows(of_l5 / "l5_training_published.jsonl")) == 2
    nothing_written = lambda: not [path for path in runs.iterdir() if path.name.startswith("ladder_l5_draws") and any(path.iterdir())]      # noqa: E731
    # ---- A REAL RUN IS REFUSED FOR SUCH A RUN OF L5 (its report is INCONCLUSIVE), in the seed's own directory and in one the task names.
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no model here stands beside such a run"):
        ladder_l5_draws.ladder_l5_draws_prepare(config)
    monkeypatch.setenv(RUN, "ladder_l5_draws_smoke")
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE"):
        ladder_l5_draws.ladder_l5_draws_prepare(config)
    monkeypatch.delenv(RUN)
    # THE SMOKE-ONLY VARIABLE without a run directory of the task's naming; and with one, on a map that is no fixture's: refused, and nothing is written.
    said = "RLVR_LEAN_LADDER_L5_DRAWS_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_DRAWS_RUN. and on the fixtures' base map"
    monkeypatch.setenv(SMOKE, "1")
    with pytest.raises(RuntimeError, match=said + ". A run in the seed's own directory reads a run of L5 whose checks passed. Nothing was written"):
        ladder_l5_draws.ladder_l5_draws_prepare(config)
    monkeypatch.setenv(RUN, "ladder_l5_draws_smoke")
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match=said + ". A run on the pool's problems reads a run of L5 whose checks passed. Nothing was written"):
        ladder_l5_draws.ladder_l5_draws_prepare(config)
    marker.write_text(json.dumps(recorded))
    # A report NOT TO BE READ is refused for a smoke run too; so is a seed that is none of the setting's.
    report_file = of_l5 / "report_ladder_l5.json"
    kept = report_file.read_text()
    report_file.write_text(json.dumps({**json.loads(kept), "ok": False}))
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is not to be read"):
        ladder_l5_draws.ladder_l5_draws_prepare(config)
    report_file.write_text(kept)
    monkeypatch.setenv(SEEDS, "0")
    with pytest.raises(RuntimeError, match="seed 0 is one of L5's own"):
        ladder_l5_draws.ladder_l5_draws_prepare(config)
    monkeypatch.setenv(SEEDS, "3")
    assert nothing_written()
    # ---- THE SMOKE RUN: its own directory, the fixtures' map, the variable. L5's run is read all the same, and that is RECORDED; one deep solve is enough.
    read_only = {path: _all(path) for path in (of_l5, deep.map, deep.l4t2, deep.rank)}
    summaries = _stage5d(config, monkeypatch, stage_name="ladder_l5_draws_smoke")
    store = ladder_l5_draws._store(config, ladder_l5_draws.the_settings(config))
    assert store.root.name == "ladder_l5_draws_smoke" and list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == _own_files(CONSTANT)
    read = summaries["ladder_l5_draws_prepare"]
    smoke = read["smoke"]
    assert (smoke["problems_at_least"], smoke["l5_read_though_inconclusive"], smoke["l5s_failed_checks"]) == (1, True, failed) and read["seed"] == 3
    of_the_search = read["the_search"]
    assert (of_the_search["run"], of_the_search["deep_solves"], of_the_search["rows_of_each_kind"], of_the_search["inconclusive"]) == (str(of_l5), 1, 1, True)
    assert of_the_search["enough_deep_solves"]["passes"] is True and of_the_search["enough_deep_solves"]["minimum"] == 1
    # THE RECIPE IS HELD TO L5n's REAL RUN HERE TOO (a smoke run names no other): it is not there, so the recipe as it is, and that is said.
    assert read["the_recipe"]["recipe"] == "constant" and read["the_recipe"]["of_l5n"]["run"] == str(runs / "ladder_l5_noise_pre_r64_seed0") and read["the_recipe"]["l5n_named_it"] is False
    # THE THREE SETS, of TWO rows each, by L5's own rule for seed 3 on the smoke search: held as a real seed's are.
    sets = {model: store.read_rows(f"l5_draws_training_{model}.jsonl") for model in MODELS}
    pretraining = ladder_l4_start.pretrained_on(deep.rank, read["pretraining_file_sha256"], "pre_r64")[0]
    assert hold_the_sets(sets, pretraining, 3) == 1 and all(len(own) == 2 for own in sets.values())
    assert {model: read["the_sets"][model]["rows_by_origin"] for model in MODELS} == {PUBLISHED: {"pretraining": 2}, SHALLOW: {"shallow": 1, "pretraining": 1}, DEEP: {"deep": 1, "pretraining": 1}}
    # THREE TRAININGS of one optimizer step, exactly L5's call; THREE measurements of the one goal problem's first sampling, each at rank 64.
    assert [set(call["more"]) for call in calls] == [{"start"}] * 3 and [call["schedule"]["checkpoint_steps"] for call in calls] == [{model: 1} for model in MODELS] and [call["seed"] for call in calls] == [3] * 3
    assert solver.measured == [(GOAL, 32, 1001, 1024)] * 3 and world.measured == [64] * 3 and sorted(path.name for path in (store.root / "adapters").iterdir()) == sorted(MODELS)
    # THE REPORT says A SMOKE RUN; this seed's checks pass; no other seed's run is read from a run the task names, and L5's other seeds are not there.
    report = summaries["ladder_l5_draws_report"]
    assert report["ok"] is True and report["inconclusive"] is False and report["smoke"] == smoke and report["branches"] == {"primary": NOT_READ, "where_it_sits": NOT_READ, "what_follows": NOT_READ}
    assert any(f"A SMOKE RUN: L5's smoke run reads INCONCLUSIVE ({', '.join(failed)}) and was read all the same; one deep solve is enough" in line for line in report["lines"])
    over = report["over_the_fresh_seeds"]
    assert (over["seeds_there"], over["seeds_missing"]) == ([3], [4, 5]) and {seed: (own["there"], own["inconclusive"]) if own["there"] else False for seed, own in report["the_seeds_of_l5"].items()} == {
        "0": (True, True), "1": False, "2": False}
    assert "ladder_l5_deep_smoke_seed1 does not hold" in report["the_seeds_of_l5"]["1"]["why"]
    assert sorted(path.name for path in runs.iterdir() if path.name.startswith("ladder_l5_draws") and any(path.iterdir())) == ["ladder_l5_draws_smoke"]      # the refused real run left its directory empty
    # ---- A SMOKE RUN OF ANOTHER RECIPE has a run directory of its own (the recipe in its name): the smoke run of one recipe does not stand in another's way.
    _l5ns_report(runs, AVERAGE_OF_FOUR)
    config["ladder_loop"]["l5_draws"]["recipe"] = MEAN_CONSTANT
    del calls[:]
    averaged = _stage5d(config, monkeypatch, stage_name="ladder_l5_draws_smoke")
    other = ladder_l5_draws._store(config, ladder_l5_draws.the_settings(config))
    assert other.root.name == "ladder_l5_draws_smoke_mean_constant" and sorted(name for name in _all(other.root) if not name.startswith("episodes_")) == _own_files(MEAN_CONSTANT)
    assert _all(store.root) == {name: content for name, content in _all(store.root).items()} and len(calls) == 12 and len(solver.measured) == 6
    # TWO ROWS HAVE TWO ORDERS: order seeds that give one order are trained all the same, and RECORDED; and four trainings of one optimizer step on the same rows are ONE adapter,
    # whose mean is each of them: a real run refuses it (an average of nothing), a smoke run keeps it, serves it, and SAYS so.
    of_the_orders = averaged["ladder_l5_draws_prepare"]["the_orders"]
    assert all(len(of_the_orders[model]["alike"]) >= 2 for model in MODELS) and averaged["ladder_l5_draws_prepare"]["the_recipe"]["l5n_named_it"] is True
    for model in MODELS:
        made = averaged[f"ladder_l5_draws_average_{model}"]
        assert (made["read_back"]["is_the_mean"], made["read_back"]["is_none_of_them"], made["one_of_its_models_in_a_smoke_run"]) == (True, False, True)
        check = averaged["ladder_l5_draws_report"]["can_this_run_be_read"][model]["its_numbers_are_the_mean_of_its_models_and_it_is_served"]
        assert (check["one_of_its_models_in_a_smoke_run"], check["passes"]) == (True, True)
    assert averaged["ladder_l5_draws_report"]["inconclusive"] is False and averaged["ladder_l5_draws_report"]["recipe"] == "mean_constant"
    # WITHOUT THE VARIABLE nothing is taken loosely.
    monkeypatch.delenv(SMOKE)
    assert ladder_l5_draws.smoke_run() is False


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_draws"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l5_draws.STEPS) == STEPS and ladder_l5_draws.ENTRY_STAGE == "ladder_l5_draws" and len(STEPS) == 23
    # THE STEPS ARE THOSE OF THE LARGEST RECIPE: eight make something under the recipe as it is and under the decay, twenty under the average of four; none is of no recipe.
    assert (len(_used(CONSTANT)), len(_used(DECAY)), len(_used(MEAN_CONSTANT))) == (8, 8, 20) and set(_used(CONSTANT)) | set(_used(MEAN_CONSTANT)) == set(STEPS)
    # THE SMOKE STAGE: the same steps, under the runs L5's smoke stage names (WITHOUT its smoke-only variable), a run of its own, and ITS smoke-only variable.
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_draws_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    of_l5 = entry.step_fields(entry.STAGES["ladder_l5_deep_smoke"][2])[2]["environment"]
    assert smoke[0][2]["environment"] == {**{name: value for name, value in of_l5.items() if name != ladder_l5.SMOKE_VARIABLE}, RUN: "ladder_l5_draws_smoke", SMOKE: "1"}
    assert smoke[0][2]["environment"][ladder_l5.RUN_VARIABLE] == "ladder_l5_deep_smoke" and ladder_l5.SMOKE_VARIABLE in of_l5 and ladder_l5_noise.RUN_VARIABLE not in smoke[0][2]["environment"]
    # THE SETTING, as the package's own config has it: three fresh seeds, three more that wait for explicit approval, L5n's sampling, and THE RECIPE L5n's REAL REPORT
    # NAMED on 2026-10-11 (the average of four: `mean_constant`).
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings = whole["ladder_loop"]["l5_draws"]
    assert settings == {"seeds": [3, 4, 5], "more_seeds": [6, 7, 8], "more_seeds_approved": False, "sampling": "reach", "attempts": 32, "recipe": "mean_constant"} == CONFIG["ladder_loop"]["l5_draws"]
    own = ladder_l5_draws.the_settings(whole)
    assert (own.start, own.seeds, own.more, own.approved, own.sampling, own.attempts, own.recipe, own.of_l5) == ("pre_r64", (3, 4, 5), (6, 7, 8), False, "reach", 32, "mean_constant", (0, 1, 2))
    assert own.noise == ladder_l5_noise.the_settings(whole) and own.deep == ladder_l5.the_settings(whole)
    assert [ladder_l5_draws.run_name(own, seed) for seed in (3, 8)] == ["ladder_l5_draws_pre_r64_seed3", "ladder_l5_draws_pre_r64_seed8"] and ladder_l5.search_directory(whole, own.deep).name == "ladder_l5_deep_pre_r64_seed0"
    assert (ladder_l5_draws.the_set("deep", "reach"), ladder_l5_draws.average_marker("deep"), ladder_l5_draws.REPORT_FILE) == ("l4_reach_deep", "ladder_l5_draws_average_deep", "report_ladder_l5_draws.json")
    assert ladder_l5_draws.the_set("published", "reach") == ladder_l5_noise.the_set("published", "reach")                         # the sets are named as L5's and L5n's are: they pair by problem
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l5_draws"]' in path.read_text()] == ["gpu/ladder_l5_draws.py"]
    others = set(ladder_l5.STEPS) | set(ladder_l6.STEPS) | set(ladder_l4_rows.STEPS) | set(ladder_l5_noise.STEPS)
    assert not set(ladder_l5_draws.STEPS) & others and all(name.startswith("ladder_l5_draws_") for name in ladder_l5_draws.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l5_draws.STEPS for step in entry.STAGES[name])) == ["ladder_l5_draws", "ladder_l5_draws_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    # (L5s's smoke stage names this stage's RUN too, and not its smoke-only variable: its dose is read against this stage's smoke run, by `own_run`.)
    # (L5f's smoke stage names it too: its sets are held to this stage's smoke run of the seed.)
    assert names(RUN) == ["ladder_l5_draws_smoke", "ladder_l5_full_smoke", "ladder_l5_stock_smoke", "ladder_l6b_smoke"] and names(SMOKE) == ["ladder_l5_draws_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L5_DRAWS_SMOKE" and RUN == "RLVR_LEAN_LADDER_L5_DRAWS_RUN"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l5_draws.py", "runner/entry.py"]
    # ONE BODY EACH: its trainings are L4t's own training body, its measurements the ceiling's own measuring body with no rung, its averaged adapters L5n's own function, and
    # L5's run is read, and the three sets are drawn, by L5's own functions.
    assert ladder_l5_draws.train_a_model is ladder_l4_rows.train_a_model and ladder_l5_draws.measure_model is ladder_ceiling.measure_model
    own_stage = ladder_l5_draws.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == (
        "ladder_l5_draws", "ladder_l5_draws", "l5_draws", "L5d", "ladder_l5_draws_prepare", "ladder_loop.l5.start")
    source = (PACKAGE / "gpu" / "ladder_l5_draws.py").read_text()
    for called in ("ladder_l5.read_the_search(", "ladder_l5.read_for_the_search(", "ladder_l5.deep_solves_by_the_seed(", "ladder_l5.sets_by_the_seed(", "ladder_l5_noise.the_mean_adapter(", "rungs=False"):
        assert called in source, called
    assert "_episodes(" not in source and "LoRARequest" not in source and "mean_of_adapters" not in source and "the_sets(" not in source
    of_l5s = (PACKAGE / "gpu" / "ladder_l5.py").read_text()
    assert of_l5s.count("deep_solves_by_the_seed(") == 2 and of_l5s.count("sets_by_the_seed(") == 2                                # L5's own step calls the two functions it defines
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l5_draws.STEPS" in main and "ladder_l5_draws," in main


# ------------------------------------------------------------------- the stages that existed before this one
def test_l5_and_l5n_write_what_they_wrote_with_this_stage_run_beside_them(draws, monkeypatch):  # noqa: F811
    config, deep_, first = draws.config, draws.deep, draws.first
    as_it_wrote = _all(draws.l5)
    # L5's run is L5's OWN PATH at the search's seed, BY THE TWO FUNCTIONS THIS STAGE DRAWS ITS SETS BY TOO: its files and steps are what `test_ladder_l5_stage.py` holds them
    # to; nothing of it names this stage; each of its trainings hands the pass the start adapter and nothing more.
    assert sorted(name for name in as_it_wrote if not name.startswith("episodes_")) == L5_FILES and list(first) == L5_STEPS
    assert not [name for name in as_it_wrote if "l5_draws" in name] and "l5_draws" not in json.dumps(first, default=str) and not [line for line in first["ladder_l5_report"]["lines"] if "L5d" in line]
    assert all(set(call["more"]) == {"start"} for call in deep_.calls[-3:]) and len(deep_.calls) >= 3
    _stage5d(config, monkeypatch)
    root = draws.store().root
    own = _all(root)
    assert _all(draws.l5) == as_it_wrote
    # L5's own task, queued again AFTER this stage ran in the same store: every step returns what is stored, nothing is trained or sampled, no file of either run moves.
    monkeypatch.delenv(SEEDS)
    trained, asked, measured = len(draws.calls), len(deep_.solver.asked), len(draws.solver.measured)
    again = _stage5(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in first.items() if "report" not in name}
    assert again["ladder_l5_report"]["lines"] == first["ladder_l5_report"]["lines"] and again["ladder_l5_report"]["branches"] == first["ladder_l5_report"]["branches"]
    assert (len(draws.calls), len(deep_.solver.asked), len(draws.solver.measured)) == (trained, asked, measured) and _all(draws.l5) == as_it_wrote and _all(root) == own
    # A FURTHER SEED OF L5, run AFTER this stage: it reads the search's run as it always did and writes its own files, and no other: this stage's run is nothing to it.
    monkeypatch.setenv(SEEDS, "1")
    further = _stage5(config, monkeypatch)
    of_the_seed = ladder_l5._store(config, ladder_l5.the_settings(config)).root
    assert of_the_seed.name == "ladder_l5_deep_pre_r64_seed1" and list(further) == L5_STEPS and all(further[step]["run"] is False for step in NOT_MADE_AGAIN)
    assert sorted(name for name in _all(of_the_seed) if not name.startswith("episodes_")) == sorted({*L5_FILES, FURTHER_FILE} - {f"{step}.done.json" for step in NOT_MADE_AGAIN})
    assert all(set(call["more"]) == {"start"} for call in draws.calls[trained:]) and len(draws.calls) == trained + 3 and "l5_draws" not in json.dumps(further, default=str)
    assert all("learning_rate" not in further[f"ladder_l5_train_{model}"] for model in MODELS) and _all(draws.l5) == as_it_wrote and _all(root) == own
    # L5n, run AFTER this stage (its task is of the search's seed): its twenty-two steps, its own files and no other, its report's keys; ITS AVERAGED ADAPTERS ARE MADE AND HELD BY
    # THE BODY THIS STAGE'S AVERAGING STEP CALLS, under L5n's own names; this stage's run is nothing to it.
    monkeypatch.delenv(SEEDS)
    trained = len(draws.calls)
    of_l5n = _stage5n(config, monkeypatch)
    its = ladder_l5_noise._store(config, ladder_l5_noise.the_settings(config)).root
    assert its.name == "ladder_l5_noise_pre_r64_seed0" and list(of_l5n) == L5N_STEPS
    assert sorted(name for name in _all(its) if not name.startswith("episodes_")) == sorted([*L5N_FILES, "l5_noise_stored_published_seed1.jsonl"])     # L5's seed 1 is there by now: read beside, as L5n reads it
    assert list(of_l5n["ladder_l5_noise_report"]) == L5N_REPORT_KEYS and "l5_draws" not in json.dumps(of_l5n, default=str) and not [name for name in _all(its) if "l5_draws" in name]
    assert len(draws.calls) == trained + 8 and [set(call["more"]) for call in draws.calls[trained:]] == [{"start", "learning_rates"}, {"start", "decay", "learning_rates"}] * 4
    for recipe in ("constant", "decay"):
        made = of_l5n[f"ladder_l5_noise_average_{recipe}"]
        assert (made["check"], made["model"], made["recipe"], made["of"]) == ("L5n", f"mean_{recipe}", recipe, [f"{recipe}_{order}" for order in ORDERS])
        assert (made["read_back"]["passes"], made["one_of_its_models_in_a_smoke_run"], made["adapter"]) == (True, False, str(its / "adapters" / f"mean_{recipe}"))
        check = of_l5n["ladder_l5_noise_report"]["can_this_run_be_read"][f"mean_{recipe}"]["its_numbers_are_the_mean_of_its_models_and_it_is_served"]
        assert (check["is_the_mean"], check["is_none_of_them"], check["served"], check["passes"]) == (True, True, True, True)
    assert _all(draws.l5) == as_it_wrote and _all(root) == own
    # L5's names are its own as they were, and so are L5n's.
    of_l5, of_noise = ladder_l5.THE_STAGE, ladder_l5_noise.THE_STAGE
    assert (of_l5.name, of_l5.steps, of_l5.files, of_l5.reader, of_l5.prepare, of_l5.start_said_by) == ("ladder_l5_deep", "ladder_l5", "l5", "L5", "ladder_l5_prepare", "ladder_loop.l5.start")
    assert (of_noise.name, of_noise.steps, of_noise.files, of_noise.reader, of_noise.prepare) == ("ladder_l5_noise", "ladder_l5_noise", "l5_noise", "L5n", "ladder_l5_noise_prepare")
    # THIS STAGE's task queued again once L5n's report is there: the run was prepared before it and KEEPS what it was prepared with (the recipe as it is, and what was said of it).
    monkeypatch.setenv(SEEDS, "3")
    prepared = ladder_l5_draws.ladder_l5_draws_prepare(config)
    assert prepared["the_recipe"]["of_l5n"]["there"] is False and prepared["the_recipe"]["l5n_named_it"] is False and len(draws.calls) == trained + 8


def test_l6_writes_what_it_wrote_with_this_stage_in_the_code(cycle, monkeypatch):  # noqa: F811
    # L6 IN ITS OWN WORLD: its seventeen steps, its own files and no other; round 1 is read by L5's own reading, as it was; each of its four trainings hands the pass the start
    # adapter and NOTHING MORE.
    summaries = _stage6(cycle.config, monkeypatch)
    root = cycle.store().root
    start = cycle.deep.rank / "adapters" / "pre_r64"
    assert list(summaries) == L6_STEPS and sorted(name for name in _all(root) if not name.startswith("episodes_")) == L6_FILES and not [name for name in _all(root) if "l5_draws" in name]
    assert [call["more"] for call in cycle.calls] == [{"start": start}] * 4 and "l5_draws" not in json.dumps(summaries, default=str)
    assert not set(ladder_l6.STEPS) & set(ladder_l5_draws.STEPS) and "ladder_l5_draws" not in (PACKAGE / "gpu" / "ladder_l6.py").read_text()
