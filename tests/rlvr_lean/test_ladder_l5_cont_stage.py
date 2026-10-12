"""L5c, the stage end to end on what L5's FINISHED run, L4p's run and L5f's OWN FINISHED RUN OF THE SEED stored: the
refusals of its first step (the seed, the escalation, the start, L5's run, `second_full`, L5f's run of the seed, its
adapters, a set that is not that run's); `single` measured from the adapter L5f's run keeps; `wide`, `again` and `more`,
each trained in four orders from ITS start, averaged (`wide` with L5f's four: the mean of eight) and measured on the
goal set's first sampling ALONE; the report's checks failing when they should; WHAT MADE IT and the three reads at the
branches a one-problem world reaches; the smoke stage; the registration; and L5f's run left file for file as
it was. Spec: docs/spec/ladder-loop.spec.md, "L5c: the continuation". The engine and Lean are scripted; nothing
touches a GPU or the network. The rules on hand-made rows are `test_ladder_l5_cont.py`."""

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
from test_ladder_l5_draws_stage import _l5ns_report, _stage5d  # noqa: E402 - L5d's own stage, whose run of the seed L5f's stands on
from test_ladder_l5_full_stage import OWN_FILES as L5FS_FILES  # noqa: E402 - what a seed of L5f leaves, frozen there
from test_ladder_l5_full_stage import _as_text, _changed, _file_rows, _stage5f, _swapped  # noqa: E402 - L5f's own stage, whose run of the seed this one stands on
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
from test_ladder_l6_stage import SOLVED_1, _round_one, _smoke_world  # noqa: E402 - L4p's run and L5's stage run to its end on a base map of twelve problems; the smoke runs' world

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import rehearsal_rows  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_cont import (  # noqa: E402
    ADDS,
    AGAIN,
    AVERAGED,
    COSTS,
    FURTHER_SEEDS,
    MODELS,
    MORE,
    NOT_SHOWN_AT_ONE,
    OF_THEIRS,
    ONE_CONTINUATION,
    ORDERS_OF,
    SINGLE,
    THE_AVERAGE_IS_PART,
    THE_RECIPE_STANDS,
    WIDE,
)
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import (  # noqa: E402
    ladder_ceiling,
    ladder_l4_rows,
    ladder_l5,
    ladder_l5_cont,
    ladder_l5_draws,
    ladder_l5_full,
    ladder_l5_noise,
    ladder_l5_stock,
    ladder_l6,
    ladder_round,
    milestone2,
    pipeline,
)
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.adapter_files import compared, is_the_mean  # noqa: E402

RUN, SMOKE, L4P_RUN, L5F_RUN = ladder_l5_cont.RUN_VARIABLE, ladder_l5_cont.SMOKE_VARIABLE, ladder_l6.L4P_RUN_VARIABLE, ladder_l5_full.RUN_VARIABLE
VARIABLES = (RUN, SMOKE, L4P_RUN, L5F_RUN, ladder_l5_full.SMOKE_VARIABLE, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE)
TRAINED = [f"{model}_{order}" for model in AVERAGED for order in ORDERS_OF[model]]
STEPS = ["ladder_l5_cont_prepare", "ladder_l5_cont_measure_single",
         *(step for model in AVERAGED for step in (*(f"ladder_l5_cont_train_{model}_{order}" for order in ORDERS_OF[model]), f"ladder_l5_cont_average_{model}",
                                                   f"ladder_l5_cont_measure_{model}")),
         "ladder_l5_cont_report"]
OWN_FILES = sorted([
    *(f"adapters/{name}/{file}" for name in (*AVERAGED, *TRAINED) for file in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l5_cont_loss_{name}.json" for name in TRAINED), *(f"l5_cont_training_{name}.jsonl" for name in (*AVERAGED, *TRAINED)),
    *(f"{step}.done.json" for step in STEPS), "problems.jsonl", "report_ladder_l5_cont.json",
    "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", "l4_stored_pre_r64_reach.jsonl", "l4_stored_second_full_reach.jsonl", "l4_stored_l5f_published_reach.jsonl",
    "l5_held_aside_problems.jsonl"])
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "first_start", "rank", "models", "recipe", "branches", "inconclusive",
               "failed_checks", "can_this_run_be_read", "l4ps_own_checks_of_the_start_model", "of_this_seed", "figures", "counts", "the_bar", "what_made_it", "the_reads",
               "against_the_stored_models", "overfitting", "timed_out_in_lean", "what_follows", "the_seeds", "the_start_model", "the_search", "l5fs_run", "the_models", "the_sets",
               "the_more", "the_orders", "the_sampling", "training_sets", "training_recipe", "models_made", "smoke", "what_it_cannot_say", "attempts", "not_to_be_read", "not_built",
               "not_measured", "adapters"]
OF_L5D = (4, 8, 12)                                                         # of the 32 attempts at the goal problem: L5d's three models at seed 3, from `pre_r64`
OF_L5F = (8, 6, 10)                                                         # ... L5f's `published` (THE BAR: 8 of 32), `shallow` and `deep`, from `second_full`
SUCCESSES = (8, 9, 8, 7)                                                    # ... and this stage's `single` (the bar's own figure), `wide` (one more), `again` (the same), `more` (one fewer)
BAR = OF_L5F[0]
THE_RUN_CHECKS = ["each_set_is_its_rule_made_again_and_wide_and_more_are_l5fs_and_l5ss", "l5fs_run_of_the_seed_carries_this_stage",
                  "the_start_adapter_is_there_at_its_rank_and_is_not_the_first_starts", "single_was_sampled_from_l5fs_stored_training",
                  "the_stored_first_sampling_of_the_start_model_is_the_settings", "the_stored_first_sampling_of_the_bar_is_the_settings"]
THE_SETS, CARRIES = THE_RUN_CHECKS[0], THE_RUN_CHECKS[1]
A_LAYER = "base_model.model.layers.0.q_proj.lora_A.weight"
NOTHING_READ = {"what_made_it": INCONCLUSIVE, WIDE: INCONCLUSIVE, AGAIN: INCONCLUSIVE, MORE: INCONCLUSIVE, "what_follows": NOT_READ}


def _stage5c(config, monkeypatch, until=None, stage_name="ladder_l5_cont", prepare_on_the_gpu=False):
    """The stage's steps in its order; the trainings and the averagings by their GPU path (the stand-in makes no adapter)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l5_cont.STEPS:
            continue
        run = ladder_l5_cont.STEPS[step]
        on_the_gpu = "_train_" in step or "_average_" in step or (prepare_on_the_gpu and step == STEPS[0])
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if on_the_gpu else run(config)
        if step == until:
            break
    return summaries


def _nothing_written(runs):
    return not [path for path in runs.iterdir() if path.name.startswith("ladder_l5_cont") and any(path.iterdir())]


@pytest.fixture
def cont(deep, monkeypatch, tmp_path):  # noqa: F811
    """`deep`'s world with what this stage reads on the box: L4p's run (its `second_full` kept), a base map of twelve
    problems with a hand-made map of `pre_r64` (FOUR ordinary solves by the rule), L5's own stage RUN TO ITS END there
    at its search's seed (TWO deep solves, so L5d's and L5f's sets are FOUR rows; the batch is ONE row), L5d's OWN STAGE
    run to its end at seed 3 under the averaged recipe, and L5f's OWN STAGE RUN TO ITS END at seed 3 (its `published`
    closes the goal in 8 of 32: THE BAR). So here `wide` is those four rows, `again` the next four of the seed's order,
    and `more` twice the six own solves in stock: the fixture's twelve published rows, all of them. Then this stage's
    task of SEED 3, which carries NO variable. `solver.measured`: every prompt of a measurement of the goal set's first
    sampling made from the fixture on; this stage's four close the goal as `SUCCESSES` says. `calls`, `kits`: this
    stage's own trainings and engines; the k-th training's adapter holds 12 + k (L5f's twelve hold 1 to 12)."""
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    config["training"]["effective_batch"] = 1
    first = _round_one(deep, monkeypatch, tmp_path, solved=SOLVED_1, l4p=True, settings={"problems": 4, "held_aside": 1, "minimum": 2})
    root, l4p = deep.store().root, deep.dose.store().root
    assert root.name == "ladder_l5_deep_pre_r64_seed0" and first["ladder_l5_report"]["inconclusive"] is False and first["ladder_l5_report"]["ok"] is True
    assert l4p.name == "ladder_l4_dose_pre_r64_seed0" and first["ladder_l5_prepare"]["rows_of_each_kind"] == 2
    made, calls, kits = [], [], []
    _trainings(monkeypatch, made)
    solver = NoiseSolver(deep.solver, (*OF_L5D, *OF_L5F, *SUCCESSES))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"]}),
        (solver, stand_in_parameters))[1]))
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l5_cont"])        # THE STAGE names no variable
    monkeypatch.setenv(SEEDS, "3")
    runs = ladder_l5._runs(config)
    _l5ns_report(runs, AVERAGE_OF_FOUR)
    assert _stage5d(config, monkeypatch)["ladder_l5_draws_report"]["inconclusive"] is False
    del made[:]
    theirs = _stage5f(config, monkeypatch)
    l5f = runs / "ladder_l5_draws_second_full_seed3"
    assert theirs["ladder_l5_full_report"]["inconclusive"] is False and theirs["ladder_l5_full_report"]["ok"] is True and len(made) == 12
    assert sorted(name for name in _all(l5f) if not name.startswith("episodes_")) == L5FS_FILES and len(_file_rows(l5f / "l5_full_training_published.jsonl")) == 4
    del kits[:]
    # THIS STAGE's trainings are the calls after L5f's twelve: `calls` is a view of them that the tests read.
    return SimpleNamespace(config=config, deep=deep, first=first, l5=root, l4p=l4p, l5f=l5f, l5d=runs / "ladder_l5_draws_pre_r64_seed3", theirs=theirs, made=made,
                           calls=lambda: made[12:], kits=kits, solver=solver, runs=runs, store=lambda: ladder_l5_cont._store(config, ladder_l5_cont.the_settings(config)))


def _rewritten(path, **changed):
    """A per-problem rows file with `changed` in every row; returns the text it held."""
    return _changed(path, lambda text: _as_text([{**row, **changed} for row in map(json.loads, text.splitlines())]))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_measures_single_trains_wide_again_and_more_from_their_starts_averages_and_reads_each_against_l5fs_stored_published(cont, monkeypatch):  # noqa: F811
    config, kits, solver, deep_ = cont.config, cont.kits, cont.solver, cont.deep
    read_only = {path: _all(path) for path in (cont.l5, cont.l5f, cont.l5d, cont.l4p, deep_.map, deep_.l4t2, deep_.rank)}
    summaries = _stage5c(config, monkeypatch)
    store, calls = cont.store(), cont.calls()
    # THE RUN DIRECTORY IS ITS OWN, beside the runs it reads, which were ONLY READ: file for file. L5f's run is what L5f's own test holds it to.
    assert store.root.name == "ladder_l5_cont_second_full_seed3" and store.root.parent == cont.l5.parent and list(summaries) == STEPS == list(ladder_l5_cont.STEPS)
    assert {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert sorted(name for name in _all(cont.l5f) if not name.startswith("episodes_")) == L5FS_FILES

    # ---- the first step: the seed, the two start models, L5f's run of the seed and what of it was held, the one sampling
    read = summaries["ladder_l5_cont_prepare"]
    second_full, pre_r64, theirs = cont.l4p / "adapters" / "second_full", deep_.rank / "adapters" / "pre_r64", cont.l5f / "adapters"
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"], read["first_start"], read["models"], read["trained"], read["averaged"]) == (
        "l4", "pretrained on published proofs", "L5c", 3, "second_full", "pre_r64", list(MODELS), TRAINED, list(AVERAGED))
    assert (read["start_adapter"], read["start_run"], read["first_start_adapter"], read["first_start_run"]) == (str(second_full), str(cont.l4p), str(pre_r64), str(deep_.rank))
    assert read["smoke"] is None and read["start_recipe"]["rank"] == 64 and read["start_recipe"]["alpha"] == 128
    assert read["the_seeds"] == {"what": read["the_seeds"]["what"], "seeds": [3], "more_seeds": [4, 5], "escalated": False, "fixed": [3]}
    assert read["the_start_model"] == cont.theirs["ladder_l5_full_prepare"]["the_start_model"]              # `second_full`, read as L5f read it
    of_theirs = read["l5fs_run"]
    assert (of_theirs["run"], of_theirs["adapters"]) == (str(cont.l5f), {name: str(theirs / name) for name in (*OF_THEIRS, "published")}) and "`ladder_l5_full` for seed 3" in of_theirs["task"]
    held = of_theirs["held"]
    assert (held["passes"], held["recorded"], held["read_now"], held["rank"], held["alpha"], list(held["trainings"])) == (True, True, None, 64, 128, list(OF_THEIRS))     # no GPU box here
    assert held["the_averaged"]["of"] == list(OF_THEIRS) and held["the_averaged"]["is_the_mean"] is True and held["the_averaged"]["is_none_of_them"] is True
    assert of_theirs["the_sets"]["rows"] == {WIDE: 4, AGAIN: 4, MORE: 12} and of_theirs["the_members"] == {"what": of_theirs["the_members"]["what"], "trainings": list(OF_THEIRS), "rows": 4, "passes": True}
    assert of_theirs["at_the_first_step"]["seed"] == 3 and of_theirs["at_the_first_step"]["start"] == "second_full" and of_theirs["at_the_first_step"]["inconclusive"] is False
    assert read["the_more"] == {"what": read["the_more"]["what"], "deep_solves": 2, "ordinary_solves": 4, "ordinary_solves_of_the_rule": 4, "ordinary_solves_taken_by_the_runs_minimum": 0,
                                "own_solves": 6, "rows": 12}
    sampling = read["the_sampling"]
    assert (sampling["name"], sampling["episodes"], sampling["sampling_seed"], sampling["sets"]) == ("reach", 32, 1001, {model: f"l4_reach_{model}" for model in MODELS})
    assert (sampling["stored_start"], sampling["stored_first_start"], sampling["stored_bar"]) == (
        "l4_stored_second_full_reach.jsonl", "l4_stored_pre_r64_reach.jsonl", "l4_stored_l5f_published_reach.jsonl")
    assert read["recipe"]["name"] == "mean_constant" and read["recipe"]["adapter_seed"] == 3 and read["recipe"]["optimizer_steps"] == {WIDE: 4, AGAIN: 4, MORE: 12}
    assert read["the_averaged"][WIDE] == {"what": read["the_averaged"][WIDE]["what"], "kept_by_l5f": list(OF_THEIRS), "of": [f"wide_{order}" for order in (5, 6, 7, 8)]}
    assert read["the_averaged"][AGAIN]["kept_by_l5f"] == [] == read["the_averaged"][MORE]["kept_by_l5f"]

    # ---- THE THREE SETS: `wide` IS L5f's STORED `published`, row for row; `again` the NEXT rows of the seed's order; `more` every row of the fixture. NEVER with text.
    sets = {model: store.read_rows(f"l5_cont_training_{model}.jsonl") for model in AVERAGED}
    assert sets[WIDE] == _file_rows(cont.l5f / "l5_full_training_published.jsonl") and [len(sets[model]) for model in AVERAGED] == [4, 4, 12]
    assert not any(field in row for own in sets.values() for row in own for field in TEXT_FIELDS) and all(row["origin"] == "pretraining" for own in sets.values() for row in own)
    pretraining = [{"problem_id": row["problem_id"]} for row in sorted(sets[MORE], key=lambda row: row["row_of_the_file"])]       # the fixture's twelve rows, in the file's order
    order = [row["id"] for row in rehearsal_rows(pretraining, 12, 3)]
    assert (sorted(row["id"] for row in sets[WIDE]), sorted(row["id"] for row in sets[AGAIN])) == (sorted(order[:4]), sorted(order[4:8]))
    placed = lambda row: (row["id"], row["problem_id"], row.get("row_of_the_file"))      # noqa: E731
    for model in AVERAGED:
        for order_seed in ORDERS_OF[model]:
            own, entry_ = store.read_rows(f"l5_cont_training_{model}_{order_seed}.jsonl"), read["training_sets"][f"{model}_{order_seed}"]
            assert [placed(row) for row in own] == [placed(row) for row in training_order(sets[model], order_seed)] and [row["row"] for row in own] == list(range(len(sets[model])))
            assert (entry_["order_seed"], entry_["model"], entry_["rows"], entry_["optimizer_steps"]) == (order_seed, model, len(sets[model]), len(sets[model]))
        assert read["the_orders"][model] == {"the_order_of_the_tasks_seed": [3], "alike": []}                # for `wide`, order seed 3 is one of L5f's four
        assert read["the_sets"][model]["rows_by_origin"] == {"pretraining": len(sets[model])}
    # ... and L5f's four stored trainings ARE four orders of `wide`'s rows: with its own four, the eight members of the mean.
    for order_seed, name in enumerate(OF_THEIRS, start=1):
        assert [placed(row) for row in _file_rows(cont.l5f / f"l5_full_training_{name}.jsonl")] == [placed(row) for row in training_order(sets[WIDE], order_seed)]
    # THE THREE STORED FIRST SAMPLINGS are copied: `second_full`'s from L4p's run, `pre_r64`'s, and THE BAR's from L5f's run.
    start_rows, first_rows, bar_rows = (store.read_rows(sampling[key]) for key in ("stored_start", "stored_first_start", "stored_bar"))
    assert [(row["problem_id"], row["episodes"]) for own in (start_rows, first_rows, bar_rows) for row in own] == [(GOAL, 32)] * 3 and bar_rows[0]["resolved"] == BAR
    assert start_rows == _file_rows(cont.l4p / "episodes_l4_reach_second_full_problems.jsonl") and bar_rows == _file_rows(cont.l5f / "episodes_l4_reach_published_problems.jsonl")

    # ---- TWELVE TRAININGS, each FROM ITS MODEL'S START at rank 64: `wide` and `more` from `second_full`, `again` FROM L5f's AVERAGED `published`
    starts = {WIDE: second_full, AGAIN: theirs / "published", MORE: second_full}
    assert [call["more"] for call in calls] == [{"start": starts[name.rpartition("_")[0]], "learning_rates": True} for name in TRAINED] and [call["seed"] for call in calls] == [3] * 12
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_cont_second_full_seed3_{name}" for name in TRAINED]
    assert all((call["lora"]["rank"], call["lora"]["alpha"]) == (64, 128) for call in calls)
    for call, name in zip(calls, TRAINED):
        train, loss = summaries[f"ladder_l5_cont_train_{name}"], json.loads(store.path(f"l5_cont_loss_{name}.json").read_text())
        own = store.read_rows(f"l5_cont_training_{name}.jsonl")
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in own] and loss["rows_trained"] == [row["id"] for row in own]
        assert train["adapter"] == str(store.root / "adapters" / name) and train["learning_rate"]["decay"] is False and f"the ORDER SEED {name[-1]}" in train["order"]
        assert (train["adapter_saved"]["rank"], train["adapter_saved"]["alpha"], len(loss["training_steps"])) == (64, 128, len(own))
    # THREE AVERAGED ADAPTERS, trained by nothing, by L5n's own function. `wide` IS THE MEAN OF EIGHT: L5f's four stored trainings (only read) and its own four.
    number = lambda directory: set(_numbers(directory / "adapter_model.safetensors", A_LAYER).tolist())      # noqa: E731
    members = {WIDE: [*(theirs / name for name in OF_THEIRS), *(store.root / "adapters" / f"wide_{order}" for order in (5, 6, 7, 8))],
               AGAIN: [store.root / "adapters" / f"again_{order}" for order in (1, 2, 3, 4)], MORE: [store.root / "adapters" / f"more_{order}" for order in (1, 2, 3, 4)]}
    assert [number(path) for path in members[WIDE]] == [{1.0}, {2.0}, {3.0}, {4.0}, {13.0}, {14.0}, {15.0}, {16.0}]      # the k-th training's adapter holds k: L5f's first four, this stage's first four
    for model, mean in zip(AVERAGED, (8.5, 18.5, 22.5)):
        made, directory = summaries[f"ladder_l5_cont_average_{model}"], store.root / "adapters" / model
        assert (made["model"], made["recipe"], made["adapter"], made["averaged"], made["check"]) == (model, "mean_constant", str(directory), True, "L5c")
        assert made["of"] == [path.name for path in members[model]] and made["kept_by_l5f"] == (list(OF_THEIRS) if model == WIDE else [])
        assert number(directory) == {mean} and made["made"]["adapters"] == len(members[model])
        assert made["read_back"] == is_the_mean(directory / "adapter_model.safetensors", [path / "adapter_model.safetensors" for path in members[model]]) and made["read_back"]["passes"] is True
        assert made["adapter_saved"]["rank"] == 64 and made["one_of_its_models_in_a_smoke_run"] is False
        assert made["against_the_start_adapter"] == compared(directory / "adapter_model.safetensors", starts[model] / "adapter_model.safetensors")
    # FOUR MEASUREMENTS: `single` right after the first step (no training), each averaged model right after its adapter is made; the model server started for rank 64.
    assert len(solver.measured) == 10 and solver.measured[6:] == [(GOAL, 32, 1001, 1024)] * 4 and [kit["max_lora_rank"] for kit in kits] == [64] * 4 == [kit["rank"] for kit in kits]
    order_of = list(summaries)
    assert order_of[:2] == STEPS[:2] and all(order_of.index(f"ladder_l5_cont_measure_{model}") == order_of.index(f"ladder_l5_cont_average_{model}") + 1 for model in AVERAGED)
    assert summaries["ladder_l5_cont_measure_single"]["adapter"] == str(theirs / "published_1") and not (store.root / "adapters" / SINGLE).exists()
    assert sorted(path.name for path in store.root.glob("episodes_*.done.json") if "_block_" not in path.name) == sorted(f"episodes_l4_reach_{model}.done.json" for model in MODELS)

    # ---- THE REPORT: the checks first, all passing; then the reads in the section's order, each against L5f's STORED `published`
    report = summaries["ladder_l5_cont_report"]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["failed_checks"] == [] and report["not_to_be_read"] == []
    assert json.loads(store.path("report_ladder_l5_cont.json").read_text()) == report and store.done_summary("ladder_l5_cont_report")["headline"] == report["headline"]
    assert (report["start"], report["first_start"], report["rank"], report["recipe"], report["check"], report["models"]) == ("second_full", "pre_r64", 64, "mean_constant", "L5c", list(MODELS))
    checks = report["can_this_run_be_read"]
    assert list(checks) == ["what", "the_run", *TRAINED, *MODELS] and list(checks["the_run"]) == THE_RUN_CHECKS
    assert all(entry_["passes"] is True for name in ("the_run", *TRAINED, *MODELS) for entry_ in checks[name].values())
    assert list(checks[SINGLE]) == ["lean_answered"] and all(list(checks[model]) == ["its_numbers_are_the_mean_of_its_models_and_it_is_served", "lean_answered"] for model in AVERAGED)
    of_the_sets = checks["the_run"][THE_SETS]
    assert (of_the_sets["rows"], of_the_sets["wide_is_l5fs_stored_set"], of_the_sets["rows_of_more_by_l5ss_rule"], of_the_sets["more_is_l5ss_rule"], of_the_sets["sets_that_are_not_the_rule"],
            of_the_sets["l5fs_trainings_that_are_not_orders_of_wide"], of_the_sets["l5fs_trainings_not_of_these_rows"]) == ({WIDE: 4, AGAIN: 4, MORE: 12}, True, 12, True, [], [], [])
    assert checks["the_run"][CARRIES]["its_report_now"] is True and checks["the_run"]["single_was_sampled_from_l5fs_stored_training"]["served_from"] == str(theirs / "published_1")
    figures = report["figures"]
    assert [(figures[model]["all_of_g"]["successes"], figures[model]["all_of_g"]["attempts"]) for model in MODELS] == [(count, 32) for count in SUCCESSES]
    of_start, of_first = start_rows[0]["resolved"] / 32, first_rows[0]["resolved"] / 32
    stored = figures["the_stored_models"]
    assert (stored["l5f_published"]["successes"], stored["l5f_published"]["attempts"], stored["second_full"]["per_attempt"], stored["pre_r64"]["per_attempt"]) == (
        BAR, 32, round(of_start, 6), round(of_first, 6))
    assert report["the_bar"]["run"] == str(cont.l5f) and report["the_bar"]["figures"]["all_of_g"]["successes"] == BAR
    # WHAT MADE IT: `single` reads exactly the bar: ONE CONTINUATION GIVES IT. Then the three reads, one a branch each: one more of 32 is +31.25 per 1,000, one fewer -31.25.
    made, reads = report["what_made_it"], report["the_reads"]
    assert (made["name"], made["difference"], made[SINGLE], made["published"]) == (ONE_CONTINUATION, 0.0, {"successes": 8, "attempts": 32}, {"successes": BAR, "attempts": 32})
    assert made["single_minus_the_start"] == round(8 / 32 - of_start, 6)
    assert [(reads[model]["name"], reads[model]["difference"]) for model in AVERAGED] == [(ADDS, round(1 / 32, 6)), (NOT_SHOWN_AT_ONE, 0.0), (COSTS, round(-1 / 32, 6))]
    assert report["branches"] == {"what_made_it": ONE_CONTINUATION, WIDE: ADDS, AGAIN: NOT_SHOWN_AT_ONE, MORE: COSTS, "what_follows": FURTHER_SEEDS}
    follows = report["what_follows"]
    assert (follows["adds"], follows["seeds"], follows["candidate"]) == ([WIDE], [4, 5], {"model": WIDE, "of": "this stage", "per_attempt": round(9 / 32, 6)})
    of_this_seed = report["of_this_seed"]
    assert (of_this_seed["seed"], of_this_seed["start"], of_this_seed["inconclusive"], of_this_seed["what_made_it"], of_this_seed["the_reads"]) == (
        3, "second_full", False, 0.0, {WIDE: round(1 / 32, 6), AGAIN: 0.0, MORE: round(-1 / 32, 6)})
    # BESIDE: each model against the bar, `second_full` and `pre_r64`, paired by problem; overfitting against `second_full`; the timeouts with the two stored sets beside.
    for model, count in zip(MODELS, SUCCESSES):
        against = report["against_the_stored_models"][model]
        assert list(against) == ["l5f_published", "second_full", "pre_r64"]
        assert against["l5f_published"]["per_attempt"]["all_of_g"] == {"problems": 1, "model": round(count / 32, 6), "start": round(BAR / 32, 6), "difference": round((count - BAR) / 32, 6)}
        assert against["second_full"]["per_attempt"]["all_of_g"] == {"problems": 1, "model": round(count / 32, 6), "start": round(of_start, 6), "difference": round(count / 32 - of_start, 6)}
        assert against["pre_r64"]["per_attempt"]["all_of_g"]["difference"] == round(count / 32 - of_first, 6)
        assert set(report["overfitting"][model]) == {"lost", "gained", "share_of_distinct_attempts", "long", "long_solved_by_the_start"}
        assert (report["overfitting"][model]["lost"], report["overfitting"][model]["gained"]) == (against["second_full"]["lost"], against["second_full"]["gained"])
    assert set(report["counts"]) == {"what", "long_problems", "l5f_published", *MODELS} and report["counts"][WIDE]["solved"] == 1
    assert report["timed_out_in_lean"]["checked_again"] is False and set(report["timed_out_in_lean"]["share"]) == {*MODELS, "l5f_published", "second_full"}
    assert list(report["models_made"]) == list(MODELS) and report["models_made"][SINGLE]["adapter"] == str(theirs / "published_1")
    assert report["models_made"][WIDE]["kept_by_l5f"] == list(OF_THEIRS) and list(report["models_made"][WIDE]["trainings"]) == [f"wide_{order}" for order in (5, 6, 7, 8)]
    # THE READS STAND IN THE SECTION'S ORDER, after the checks and L4p's own checks (printed, deciding nothing): one line each, the last of them what follows.
    in_order = ("THE FIGURES, ", "WHAT MADE IT: ", "A WIDER AVERAGE (a scout's read): ", "A SECOND STEP (a scout's read): ", "MORE ROWS (a scout's read): ", "BESIDE, NAMING NOTHING, ",
                "OVERFITTING, ", "TIMEOUTS, ", "WHAT FOLLOWS: ")
    said = [line.split(": ", 1)[1] for line in report["lines"]]
    assert [own.startswith(start) for own, start in zip(said[-len(in_order):], in_order)] == [True] * len(in_order)
    assert said[-len(in_order) - 1].startswith("L4p's OWN CHECKS OF `second_full`, PRINTED AND DECIDING NOTHING")
    assert all(own.startswith("CHECK, ") and own.endswith(": PASS") for own in said[1:-len(in_order) - 1]) and len(said) == 1 + 10 + 1 + len(in_order)
    assert said[1].startswith("CHECK, each set is its rule made again (`wide` 4 rows, `again` the next 4, `more` 12; L5s's rule counted again: 12), `wide` is the set L5f's run stores as `published`")
    assert said[2].startswith("CHECK, L5f's run of this seed carries this stage") and "its report now: read, of this seed, start and recipe, measured alike" in said[2]
    assert said[4].startswith(f"CHECK, `single`'s set was sampled from L5f's stored `published_1` ({theirs / 'published_1'})")
    assert any("each averaged adapter's numbers ARE the mean of its members' (eight for `wide`, four for the two others)" in own for own in said)
    assert all(line.startswith("l4 (pretrained on published proofs): ") for line in report["lines"]) and said[-1].startswith(f"WHAT FOLLOWS: {FURTHER_SEEDS}")
    assert said[0].startswith("L5c, THE CONTINUATION. Seed 3: what made L5f's `published`") and "NO ROW HERE IS ONE OF THE MODEL'S OWN PROOFS" in said[0]
    assert said[-8].startswith(f"WHAT MADE IT: {ONE_CONTINUATION}. `single` minus L5f's stored `published`, per 1,000 attempts on all of G: +0.00 (8 of 32 against 8 of 32)")
    assert said[-7].startswith(f"A WIDER AVERAGE (a scout's read): {ADDS}. `wide` minus L5f's stored `published`, per 1,000 attempts on all of G: +31.25 (9 of 32 against 8 of 32)")
    assert report["headline"].startswith("L5c (the continuation; pretrained on published proofs) seed 3, against L5f's stored `published` from `second_full`: per 1,000 attempts on all "
                                         "of G `single` 250.00, `wide` 281.25, `again` 250.00, `more` 218.75 (the bar 250.00, `second_full` 375.00, `pre_r64` 375.00); WHAT MADE IT: "
                                         "ONE CONTINUATION GIVES IT (+0.00); `wide`: ADDS: ESCALATE (+31.25); `again`: NOT SHOWN AT ONE SEED (+0.00); `more`: COSTS (-31.25). What "
                                         "follows: ESCALATE: TWO FURTHER SEEDS")
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == sorted([*AVERAGED, *TRAINED]) and len(report["not_built"]) == 3

    # ---- THE OTHER BRANCHES A ONE-PROBLEM WORLD REACHES, the report built again (no step is made again). `single` one success lower is 31.25 per 1,000 below the bar,
    # more than 13: THE AVERAGE IS PART OF THE GAIN.
    rows_of_single = store.path("episodes_l4_reach_single_problems.jsonl")
    kept = _rewritten(rows_of_single, resolved=7, resolved_by_statement=7)
    part = ladder_l5_cont.ladder_l5_cont_report(config)
    assert (part["what_made_it"]["name"], part["what_made_it"]["difference"], part["branches"]["what_made_it"]) == (THE_AVERAGE_IS_PART, round(-1 / 32, 6), THE_AVERAGE_IS_PART)
    assert part["branches"][WIDE] == ADDS and "WHAT MADE IT: THE AVERAGE IS PART OF THE GAIN (-31.25)" in part["headline"] and part["inconclusive"] is False
    rows_of_single.write_text(kept)
    # `wide` AT THE BAR: nothing adds, L5f's `published` recipe stands, and with no model above the bar THE BAR is the candidate for a measurement in full.
    rows_of_wide = store.path("episodes_l4_reach_wide_problems.jsonl")
    kept = _rewritten(rows_of_wide, resolved=8, resolved_by_statement=8)
    stands = ladder_l5_cont.ladder_l5_cont_report(config)
    assert stands["branches"] == {"what_made_it": ONE_CONTINUATION, WIDE: NOT_SHOWN_AT_ONE, AGAIN: NOT_SHOWN_AT_ONE, MORE: COSTS, "what_follows": THE_RECIPE_STANDS}
    assert stands["what_follows"]["candidate"] == {"model": "published", "of": "L5f's stored run of the seed", "per_attempt": 0.25} and "seeds" not in stands["what_follows"]
    rows_of_wide.write_text(kept)
    assert len(solver.measured) == 10 and len(calls) == 12 and ladder_l5_cont.ladder_l5_cont_report(config)["branches"] == report["branches"]


def test_a_failed_check_makes_the_run_inconclusive_and_nothing_else_is_said(cont, monkeypatch):  # noqa: F811
    config = cont.config
    report = _stage5c(config, monkeypatch)["ladder_l5_cont_report"]
    store = cont.store()
    assert report["inconclusive"] is False

    def failing(path, changed):
        kept = _changed(path, changed)
        failed = ladder_l5_cont.ladder_l5_cont_report(config)
        Path(path).write_text(kept)
        return failed

    marked = lambda **changed: (lambda text: json.dumps({**json.loads(text), **changed}))      # noqa: E731
    # THE SET L5f's RUN STORES AS `published` CHANGED SINCE THE FIRST STEP (two of its rows changed places): `wide` is no longer that set. THE RUN IS INCONCLUSIVE, no read is made.
    failed = failing(cont.l5f / "l5_full_training_published.jsonl", _swapped)
    check = failed["can_this_run_be_read"]["the_run"][THE_SETS]
    assert failed["failed_checks"] == [f"the run: {THE_SETS}"] and failed["inconclusive"] is True
    assert (check["wide_is_l5fs_stored_set"], check["first_that_differs_from_l5fs"]["place"], check["sets_that_are_not_the_rule"], check["l5fs_trainings_not_of_these_rows"]) == (False, 0, [], [])
    assert failed["branches"] == NOTHING_READ and "difference" not in failed["what_made_it"] and all("difference" not in failed["the_reads"][model] for model in AVERAGED)
    assert list(failed["figures"]) == ["what", "problems", "the_stored_models", "measured_and_not_read"] and list(failed["counts"]) == ["what", "long_problems", "measured_and_not_read"]
    assert list(failed["against_the_stored_models"]) == list(failed["overfitting"]) == ["measured_and_not_read"] and list(failed["the_bar"]) == ["what", "run", "file", "measured_and_not_read"]
    assert failed["of_this_seed"]["figures"] is None and failed["of_this_seed"]["what_made_it"] is None and failed["of_this_seed"]["inconclusive"] is True
    assert failed["lines"][-1].endswith(f"THIS RUN IS INCONCLUSIVE: the run: {THE_SETS}. Nothing else is said of it")
    assert not any("WHAT MADE IT" in line or "WHAT FOLLOWS" in line or "A WIDER AVERAGE" in line for line in failed["lines"]) and "THIS RUN IS INCONCLUSIVE" in failed["headline"]
    # ONE OF L5f's FOUR TRAININGS WAS NOT OF THESE ROWS, each once (its loop's own record holds a row twice); one of its stored sets is not an order of `wide`.
    failed = failing(cont.l5f / "l5_full_loss_published_3.json", lambda text: json.dumps({**json.loads(text), "rows_trained": [*json.loads(text)["rows_trained"], json.loads(text)["rows_trained"][0]]}))
    assert failed["can_this_run_be_read"]["the_run"][THE_SETS]["l5fs_trainings_not_of_these_rows"] == ["published_3"] and failed["failed_checks"] == [f"the run: {THE_SETS}"]
    failed = failing(cont.l5f / "l5_full_training_published_2.jsonl", _swapped)
    assert failed["can_this_run_be_read"]["the_run"][THE_SETS]["l5fs_trainings_that_are_not_orders_of_wide"] == ["published_2"] and failed["inconclusive"] is True
    # A SET OF THIS RUN THAT IS NOT IN ITS ORDER SEED'S ORDER; and `more` NOT THE ROWS L5s's RULE GIVES, counted again (the first step's record of the run's minimum changed).
    failed = failing(store.path("l5_cont_training_more_2.jsonl"), _swapped)
    assert failed["failed_checks"] == [f"the run: {THE_SETS}"] and failed["can_this_run_be_read"]["the_run"][THE_SETS]["sets_not_in_their_order_seeds_order"] == ["more_2"]
    marker = store.path("ladder_l5_cont_prepare.done.json")
    failed = failing(marker, lambda text: json.dumps({**json.loads(text), "the_more": {**json.loads(text)["the_more"], "ordinary_solves_taken_by_the_runs_minimum": 1}}))
    check = failed["can_this_run_be_read"]["the_run"][THE_SETS]
    assert (failed["failed_checks"], check["rows_of_more_by_l5ss_rule"], check["more_is_l5ss_rule"]) == ([f"the run: {THE_SETS}"], 14, False)
    # L5f's REPORT OF THE SEED, AS IT IS NOW: gone; INCONCLUSIVE. The bar is that run's: nothing is read against a run that cannot carry the stage.
    for name in ("ladder_l5_full_report.done.json", "report_ladder_l5_full.json"):
        (cont.l5f / name).rename(cont.l5f / f"kept_{name}")
    gone = ladder_l5_cont.ladder_l5_cont_report(config)
    for name in ("ladder_l5_full_report.done.json", "report_ladder_l5_full.json"):
        (cont.l5f / f"kept_{name}").rename(cont.l5f / name)
    check = gone["can_this_run_be_read"]["the_run"][CARRIES]
    assert gone["failed_checks"] == [f"the run: {CARRIES}"] and check["its_report_now"] is False and "L5f's run of this seed is not on the box, or not finished" in check["why_not"]
    assert any("CHECK, L5f's run of this seed carries this stage" in line and "its report now: NOT (" in line and line.endswith(": FAIL") for line in gone["lines"])
    failed = failing(cont.l5f / "report_ladder_l5_full.json", lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], "inconclusive": True,
                                                                                                                         "failed_checks": ["`deep`: lean_answered"]}}))
    assert failed["failed_checks"] == [f"the run: {CARRIES}"] and failed["can_this_run_be_read"]["the_run"][CARRIES]["why_not"] == "it reads INCONCLUSIVE (`deep`: lean_answered)"
    # ... and its adapters, as the first step held them: not held then, the check fails now.
    failed = failing(marker, lambda text: json.dumps({**json.loads(text), "l5fs_run": {**json.loads(text)["l5fs_run"], "held": {**json.loads(text)["l5fs_run"]["held"], "passes": False}}}))
    assert failed["failed_checks"] == [f"the run: {CARRIES}"] and failed["can_this_run_be_read"]["the_run"][CARRIES]["adapters_held_by_the_first_step"] is False
    # THE START ADAPTER, as the first step recorded it: not held then, the check fails now.
    failed = failing(marker, lambda text: json.dumps({**json.loads(text), "the_start_model": {**json.loads(text)["the_start_model"], "passes": False}}))
    assert failed["failed_checks"] == ["the run: the_start_adapter_is_there_at_its_rank_and_is_not_the_first_starts"] and failed["branches"] == NOTHING_READ
    # `single` SAMPLED FROM ANOTHER ADAPTER than L5f's stored `published_1`.
    failed = failing(store.path("ladder_l5_cont_measure_single.done.json"), marked(adapter=str(cont.l5f / "adapters" / "published_2")))
    assert failed["failed_checks"] == ["the run: single_was_sampled_from_l5fs_stored_training"] and failed["inconclusive"] is True
    # A STORED FIRST SAMPLING, as this run copied it, THAT IS NOT THE SETTING'S (half the attempts a problem): `second_full`'s; the bar's.
    halved = lambda text: _as_text([{**row, "episodes": 16} for row in map(json.loads, text.splitlines())])      # noqa: E731
    assert failing(store.path("l4_stored_second_full_reach.jsonl"), halved)["failed_checks"] == ["the run: the_stored_first_sampling_of_the_start_model_is_the_settings"]
    assert failing(store.path("l4_stored_l5f_published_reach.jsonl"), halved)["failed_checks"] == ["the run: the_stored_first_sampling_of_the_bar_is_the_settings"]
    # AN AVERAGED ADAPTER THAT IS ONE OF ITS MODELS (its step's stored read-back is made to say so) fails its check in a real run.
    marker = store.path("ladder_l5_cont_average_wide.done.json")
    failed = failing(marker, lambda text: json.dumps({**json.loads(text), "read_back": {**json.loads(text)["read_back"], "is_none_of_them": False, "passes": False}}))
    assert failed["failed_checks"] == ["`wide`: its_numbers_are_the_mean_of_its_models_and_it_is_served"] and failed["branches"] == NOTHING_READ
    # ... and with everything as it was the run reads again.
    assert ladder_l5_cont.ladder_l5_cont_report(config)["branches"] == report["branches"]
    # A STEP NEEDS THE ONES BEFORE IT: with `again`'s fourth training, its averaged adapter and its measurement not done.
    for step in ("ladder_l5_cont_train_again_4", "ladder_l5_cont_average_again", "ladder_l5_cont_measure_again"):
        store.path(f"{step}.done.json").unlink()
    with pytest.raises(RuntimeError, match="the averaged adapter `again` needs the step ladder_l5_cont_train_again_4 of this run, which is not done"):
        _on_the_gpu(monkeypatch, ladder_l5_cont.STEPS["ladder_l5_cont_average_again"], config)
    with pytest.raises(RuntimeError, match="the measurement of L5c's model `again` needs the step ladder_l5_cont_average_again of this run, which is not done"):
        ladder_l5_cont.STEPS["ladder_l5_cont_measure_again"](config)
    with pytest.raises(RuntimeError, match="L5c's report needs the step ladder_l5_cont_train_again_4 of this run, which is not done"):
        ladder_l5_cont.ladder_l5_cont_report(config)


def test_as_on_the_gpu_the_first_step_reads_l5fs_five_adapters_and_every_measurement_serves_its_own_models_adapter(cont, monkeypatch):  # noqa: F811
    config, kits, runs = cont.config, cont.kits, cont.runs
    second_full, theirs = cont.l4p / "adapters" / "second_full", cont.l5f / "adapters"
    prepare = ladder_l5_cont.STEPS["ladder_l5_cont_prepare"]
    _adapter_of(second_full, 64, 128, 9.0)          # the scripted trainings of this world leave adapters alike: as on the box, `second_full`'s numbers are its own
    # ONE OF L5f's ADAPTERS IS NOT THERE (its run recorded it, the files are gone): refused by the first step, naming L5f's task, before anything is written.
    (theirs / "published_2").rename(theirs / "kept")
    with pytest.raises(RuntimeError, match="adapters/published_2 is not there: the adapter `published_2` of L5f's run of seed 3 is read by this stage, and that run keeps it. Queue the "
                                           "task of stage `ladder_l5_full` for seed 3 .* again .*nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    # IT IS `second_full`'s ADAPTER, number for number (whatever L5f's run recorded of it): no training of the rows. Refused.
    shutil.copytree(second_full, theirs / "published_2")
    with pytest.raises(RuntimeError, match=r"the adapters of `published` in L5f's run ladder_l5_draws_second_full_seed3 are not held: .*; read now, \['published_2'\] are not at rank "
                                           r"64 and alpha 128, are the start adapter's number for number, or are of a type the mean is not written in .*Queue the task of stage "
                                           r"`ladder_l5_full` for seed 3 .* again; nothing was written"):
        _on_the_gpu(monkeypatch, prepare, config)
    # AT ANOTHER RANK than `pre_r64`'s: refused too.
    _adapter_of(theirs / "published_2", 32, 64, 2.0)
    with pytest.raises(RuntimeError, match=r"read now, \['published_2'\] are not at rank 64 and alpha 128"):
        _on_the_gpu(monkeypatch, prepare, config)
    shutil.rmtree(theirs / "published_2")
    (theirs / "kept").rename(theirs / "published_2")
    assert _nothing_written(runs)
    # WITH THE FIVE AS L5f's RUN LEFT THEM the first step reads the files: there, read back at the rank, numbers that are not `second_full`'s. The report's line says so.
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        summaries = _stage5c(config, patched, prepare_on_the_gpu=True)
    store = cont.store()
    now = summaries["ladder_l5_cont_prepare"]["l5fs_run"]["held"]["read_now"]
    assert list(now) == [*OF_THEIRS, "published"] and all(own["passes"] is True and (own["read_back"]["rank"], own["read_back"]["alpha"]) == (64, 128) and own["other_types"] == []
                                                          for own in now.values())
    assert now["published_3"]["against_the_start_adapter"] == compared(theirs / "published_3" / "adapter_model.safetensors", second_full / "adapter_model.safetensors")
    report = summaries["ladder_l5_cont_report"]
    assert report["inconclusive"] is False and any("CHECK, L5f's run of this seed carries this stage" in line and "; the files themselves were read; " in line and line.endswith(": PASS")
                                                   for line in report["lines"])
    # AS ON THE GPU every measurement asks the model server for ITS model's adapter: `single` for the one L5F's RUN keeps, the three others for the ones this run keeps.
    kept_here = store.root / "adapters"
    assert [kit["adapter"][2] for kit in kits] == [str(theirs / "published_1"), *(str(kept_here / model) for model in AVERAGED)] and [kit["max_lora_rank"] for kit in kits] == [64] * 4
    assert summaries["ladder_l5_cont_measure_single"]["stand_in_engine"] is False and not (kept_here / SINGLE).exists()


# ----------------------------------------------------------------------------------------------- the refusals
def test_the_first_step_refuses_before_anything_is_written(cont, monkeypatch):  # noqa: F811
    config, runs = cont.config, cont.runs
    prepare = ladder_l5_cont.ladder_l5_cont_prepare
    # A SEED THE SETTING DOES NOT LIST; and a further seed while the escalation is not recorded.
    monkeypatch.setenv(SEEDS, "0")
    with pytest.raises(RuntimeError, match=r"seed 0 is none of this stage's seeds \[3\] \(nor of \[4, 5\], which wait for the escalation\).*\(ladder_loop.l5_cont\). Nothing was written"):
        prepare(config)
    monkeypatch.setenv(SEEDS, "4")
    with pytest.raises(RuntimeError, match=r"seed 4 is one of the further seeds \[4, 5\], which run once the scout's read at \[3\] names the escalation and the setting records it "
                                           r"\(escalated is false\).*Nothing was written"):
        prepare(config)
    # ... and ONCE IT IS RECORDED a further seed waits for L5f's OWN RUN of that seed, which is not there (no run holds the bar at it): refused, naming the task to run.
    config["ladder_loop"]["l5_cont"]["escalated"] = True
    with pytest.raises(RuntimeError, match="L5f's run of seed 4 cannot carry L5c: .*L5f's run of this seed is not on the box, or not finished. Its stored `published` is the bar of every "
                                           "read here, the start of `again`, and with its four trainings half of `wide`. Run the task of stage `ladder_l5_full` for seed 4 .* to its end "
                                           "first; nothing was written"):
        prepare(config)
    config["ladder_loop"]["l5_cont"]["escalated"] = False
    monkeypatch.setenv(SEEDS, "3")
    # THE SETTING ITSELF, by its one reader: another start, seeds that are not L5f's, an escalation that is no true or false.
    settings = config["ladder_loop"]["l5_cont"]
    for changed, said in (({"start": "pre_r64"}, "ladder_loop.l5_cont.start is 'pre_r64': this stage is specified from `second_full` alone"),
                          ({"seeds": [9]}, r"ladder_loop.l5_cont.seeds is \[9\] and more_seeds \[4, 5\]: distinct whole numbers of at least 0, at least one in seeds, each a seed of L5f's"),
                          ({"more_seeds": [3, 4]}, r"ladder_loop.l5_cont.seeds is \[3\] and more_seeds \[3, 4\]"), ({"seeds": []}, r"ladder_loop.l5_cont.seeds is \[\]"),
                          ({"escalated": "yes"}, "ladder_loop.l5_cont.escalated is 'yes': true once the scout's read names the escalation, false until then")):
        kept = dict(settings)
        settings.update(changed)
        with pytest.raises(ValueError, match=said):
            prepare(config)
        settings.clear()
        settings.update(kept)
    # THE SMOKE-ONLY VARIABLE without a run directory of the task's naming.
    monkeypatch.setenv(SMOKE, "1")
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_CONT_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_CONT_RUN. and on the "
                                           "fixtures' base map. A run in the seed's own directory reads a run of L5 whose checks passed. Nothing was written"):
        prepare(config)
    monkeypatch.delenv(SMOKE)
    # L5's RUN not to be read: refused by L5's own reading, in this stage's words.
    report_file = cont.l5 / "report_ladder_l5.json"
    kept = _changed(report_file, lambda text: json.dumps({**json.loads(text), "inconclusive": True, "failed_checks": ["`shallow`: the_training_ran"]}))
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .*: no model here stands beside such a run"):
        prepare(config)
    report_file.write_text(kept)
    # L4p's RUN, where `second_full` lives, not there: refused as L5f's own first step refuses it.
    cont.l4p.rename(cont.l4p.with_name("kept_l4p"))
    with pytest.raises(RuntimeError, match="`second_full` is NOT THERE .*Run the task of stage `ladder_l4_dose_r64` for seed 0 .* to its end first; nothing was written"):
        prepare(config)
    cont.l4p.with_name("kept_l4p").rename(cont.l4p)
    # ---- L5f's RUN OF THE SEED: INCONCLUSIVE; of another recipe, start or seed; not to be read; measured otherwise. Each refused, naming its task.
    report_file = cont.l5f / "report_ladder_l5_full.json"
    for changed, why in (({"inconclusive": True, "failed_checks": ["`deep`: lean_answered"]}, "it reads INCONCLUSIVE"), ({"recipe": "constant"}, "its recipe is `constant` and this run's `mean_constant`"),
                         ({"ok": False}, "its report is not to be read"), ({"start": "pre_r64"}, "it is a run from `pre_r64` and this stage's bar is from `second_full`"),
                         ({"seed": 4}, "it is a report of seed 4"), ({"the_measurement": {"sampling": "reach", "episodes": 16}}, "it was measured otherwise")):
        kept = _changed(report_file, lambda text: json.dumps({**json.loads(text), "of_this_seed": {**json.loads(text)["of_this_seed"], **changed}}))
        with pytest.raises(RuntimeError, match=f"L5f's run of seed 3 cannot carry L5c: .*{why}.*Run the task of stage `ladder_l5_full` for seed 3 .* to its end first; nothing was written"):
            prepare(config)
        report_file.write_text(kept)
    # ... without one of its stored sets; with a training its own step recorded at another rank; with an averaged adapter its step did not record as the mean.
    stored = cont.l5f / "l5_full_training_published_4.jsonl"
    stored.rename(stored.with_name("kept.jsonl"))
    with pytest.raises(RuntimeError, match=r"ladder_l5_draws_second_full_seed3 does not hold \['l5_full_training_published_4.jsonl'\]: the trainings of `published` L5f's run of seed 3 made"):
        prepare(config)
    stored.with_name("kept.jsonl").rename(stored)
    marker = cont.l5f / "ladder_l5_full_train_published_2.done.json"
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "adapter_saved": {**json.loads(text)["adapter_saved"], "rank": 32, "alpha": 64}}))
    with pytest.raises(RuntimeError, match="the adapters of `published` in L5f's run ladder_l5_draws_second_full_seed3 are not held: its own steps recorded .*Each of the four is read back "
                                           "at `pre_r64`'s rank and is not `second_full`'s, and the averaged one is their mean, or nothing here stands on them. Queue the task of stage "
                                           "`ladder_l5_full` for seed 3 .* again; nothing was written"):
        prepare(config)
    marker.write_text(kept)
    marker = cont.l5f / "ladder_l5_full_average_published.done.json"
    kept = _changed(marker, lambda text: json.dumps({**json.loads(text), "read_back": {**json.loads(text)["read_back"], "is_the_mean": False, "passes": False}}))
    with pytest.raises(RuntimeError, match="the adapters of `published` in L5f's run ladder_l5_draws_second_full_seed3 are not held"):
        prepare(config)
    marker.write_text(kept)
    # ---- THE SETS: L5f's stored `published` is not this seed's first rows (two of its rows changed places); one of its four trainings is not an order of `wide`.
    kept = _changed(cont.l5f / "l5_full_training_published.jsonl", _swapped)
    with pytest.raises(RuntimeError, match=r"`wide` is not the set L5f's run of this seed stored as `published`, id for id \(4 rows here and 4 there; first that differs at place 0: .*"
                                           r"Refused \(seed 3; L5's run ladder_l5_deep_pre_r64_seed0; L5f's run ladder_l5_draws_second_full_seed3\). Nothing was written"):
        prepare(config)
    (cont.l5f / "l5_full_training_published.jsonl").write_text(kept)
    kept = _changed(cont.l5f / "l5_full_training_published_2.jsonl", _swapped)
    with pytest.raises(RuntimeError, match=r"of L5f's four trainings of `published`, the stored sets of \['published_2'\] are not `wide`'s 4 rows in their order seed's order.*"
                                           r"`wide` is the mean of eight trainings of ONE set. Refused \(seed 3; .*Nothing was written"):
        prepare(config)
    (cont.l5f / "l5_full_training_published_2.jsonl").write_text(kept)
    assert _nothing_written(runs)
    # WITH EVERYTHING AS IT WAS the first step passes, and L5f's run is what it was.
    read = prepare(config)
    assert read["l5fs_run"]["held"]["passes"] is True and cont.store().is_done("ladder_l5_cont_prepare")
    assert sorted(name for name in _all(cont.l5f) if not name.startswith("episodes_")) == L5FS_FILES


def test_the_steps_run_through_the_gpu_entry_point(cont, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(milestone2, "load_config", lambda path: cont.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    # In the stand-in's world no adapter is trained and none is averaged, so the report says INCONCLUSIVE and makes no read.
    assert json.loads((tmp_path / "ladder_l5_cont_average_wide.json").read_text())["averaged"] is False
    report = json.loads((tmp_path / "ladder_l5_cont_report.json").read_text())
    assert cont.store().is_done(ladder_l5_cont.REPORT) and report["inconclusive"] is True and report["stand_in_engine"] is True and report["branches"] == NOTHING_READ


# ---------------------------------------------------------------------------------------- the smoke-only path
def test_the_smoke_stage_goes_on_from_the_smoke_runs_of_l5_l4p_and_l5f_makes_its_four_models_of_a_handful_of_rows_and_says_so(deep, monkeypatch):  # noqa: F811
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    first = _smoke_world(deep, monkeypatch)
    config, runs = deep.config, ladder_l5._runs(deep.config)
    of_l5, failed = runs / "ladder_l5_deep_smoke", first["ladder_l5_report"]["failed_checks"]
    calls, measured = [], []
    _trainings(monkeypatch, calls, value=lambda number: 1.0)          # AS ON THE BOX: one optimizer step on the same rows, whatever their order, is ONE training: adapters alike
    solver = NoiseSolver(deep.solver, (*OF_L5D, *OF_L5F, *SUCCESSES))  # L5d's smoke run's three models, L5f's three, then this stage's four
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (measured.append(given["vllm"]["max_lora_rank"]), (solver, stand_in_parameters))[1]))
    options = entry.step_fields(entry.STAGES["ladder_l5_cont_smoke"][2])[2]["environment"]
    of_l5fs = entry.step_fields(entry.STAGES["ladder_l5_full_smoke"][2])[2]["environment"]
    # ---- THE SMOKE RUNS THIS ONE READS, each by its own smoke stage: L4p's (this world's run of L4p, under the name L6's variable gives it), L5d's, then L5f's.
    of_l4p = runs / options[L4P_RUN]
    deep.dose.store().root.rename(of_l4p)
    monkeypatch.setenv(L4P_RUN, options[L4P_RUN])
    monkeypatch.setenv(SEEDS, "3")
    monkeypatch.setenv(L5F_RUN, options[L5F_RUN])
    monkeypatch.setenv(RUN, options[RUN])
    monkeypatch.setenv(SMOKE, options[SMOKE])
    # WITHOUT L5f's SMOKE RUN the first step REFUSES, naming its job: its `published` is the bar here.
    with pytest.raises(RuntimeError, match="L5f's run of seed 3 cannot carry L5c: ladder_l5_full_smoke does not hold .*Run the task of stage `ladder_l5_full` for seed 3 "
                                           ".*stage `ladder_l5_full_smoke`. to its end first; nothing was written"):
        ladder_l5_cont.ladder_l5_cont_prepare(config)
    assert _nothing_written(runs)
    monkeypatch.delenv(RUN)
    monkeypatch.delenv(SMOKE)
    _l5ns_report(runs, AVERAGE_OF_FOUR)
    monkeypatch.setenv(ladder_l5_draws.RUN_VARIABLE, of_l5fs[ladder_l5_draws.RUN_VARIABLE])
    monkeypatch.setenv(ladder_l5_draws.SMOKE_VARIABLE, "1")
    assert _stage5d(config, monkeypatch, stage_name="ladder_l5_draws_smoke")["ladder_l5_draws_report"]["inconclusive"] is False
    monkeypatch.delenv(ladder_l5_draws.SMOKE_VARIABLE)
    monkeypatch.setenv(ladder_l5_full.SMOKE_VARIABLE, of_l5fs[ladder_l5_full.SMOKE_VARIABLE])
    of_l5f = _stage5f(config, monkeypatch, stage_name="ladder_l5_full_smoke")
    theirs = runs / "ladder_l5_full_smoke"
    assert of_l5f["ladder_l5_full_report"]["inconclusive"] is False and sorted(name for name in _all(theirs) if not name.startswith("episodes_")) == L5FS_FILES
    # THIS stage's smoke task names L5's, L4p's and L5f's RUNS, and neither L5f's smoke-only variable, nor L5d's run or smoke-only variable, nor L5's or L6's smoke-only one.
    monkeypatch.delenv(ladder_l5_full.SMOKE_VARIABLE)
    monkeypatch.delenv(ladder_l5_draws.RUN_VARIABLE)
    assert not {ladder_l5_full.SMOKE_VARIABLE, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE, ladder_l5.SMOKE_VARIABLE, ladder_l6.SMOKE_VARIABLE} & set(options)
    # ---- A REAL RUN IS REFUSED FOR SUCH A RUN OF L5 (its report is INCONCLUSIVE), in the seed's own directory and in one the task names.
    assert failed != []
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no model here stands beside such a run"):
        ladder_l5_cont.ladder_l5_cont_prepare(config)
    monkeypatch.setenv(RUN, options[RUN])
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE"):
        ladder_l5_cont.ladder_l5_cont_prepare(config)
    # THE SMOKE-ONLY VARIABLE on a map that is no fixture's: refused, and nothing is written.
    monkeypatch.setenv(SMOKE, options[SMOKE])
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_CONT_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_CONT_RUN. and on the "
                                           "fixtures' base map. A run on the pool's problems reads a run of L5 whose checks passed. Nothing was written"):
        ladder_l5_cont.ladder_l5_cont_prepare(config)
    marker.write_text(json.dumps(recorded))
    assert _nothing_written(runs)
    # ---- THE SMOKE RUN: its own directory, the fixtures' map, the variable. L5's run is read all the same, and that is RECORDED.
    read_only = {path: _all(path) for path in (of_l5, theirs, of_l4p, deep.map, deep.l4t2, deep.rank)}
    before = len(calls)
    summaries = _stage5c(config, monkeypatch, stage_name="ladder_l5_cont_smoke")
    store = ladder_l5_cont._store(config, ladder_l5_cont.the_settings(config))
    assert store.root.name == "ladder_l5_cont_smoke" and list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    read = summaries["ladder_l5_cont_prepare"]
    smoke = read["smoke"]
    assert (smoke["problems_at_least"], smoke["l5_read_though_inconclusive"], smoke["l5s_failed_checks"]) == (1, True, failed) and read["seed"] == 3
    assert (read["the_search"]["run"], read["the_search"]["deep_solves"], read["the_search"]["inconclusive"]) == (str(of_l5), 1, True)
    # THE BAR IS L5f's SMOKE RUN's; the start the smoke `second_full`. One deep solve and one ordinary solve in stock: `wide` is TWO rows, `again` the next two, `more` FOUR.
    assert (read["start_adapter"], read["start_run"], read["l5fs_run"]["run"]) == (str(of_l4p / "adapters" / "second_full"), str(of_l4p), str(theirs))
    assert {model: read["the_sets"][model]["rows"] for model in AVERAGED} == {WIDE: 2, AGAIN: 2, MORE: 4} and read["l5fs_run"]["held"]["passes"] is True
    of_the_more = read["the_more"]
    assert (of_the_more["deep_solves"], of_the_more["ordinary_solves"], of_the_more["own_solves"], of_the_more["rows"]) == (1, 1, 2, 4)
    assert of_the_more["ordinary_solves_of_the_rule"] + of_the_more["ordinary_solves_taken_by_the_runs_minimum"] == 1
    assert store.read_rows("l5_cont_training_wide.jsonl") == _file_rows(theirs / "l5_full_training_published.jsonl")
    assert read["l5fs_run"]["held"]["the_averaged"]["one_of_its_models_in_a_smoke_run"] is True               # L5f's smoke `published` IS each of its four, and its run said so
    assert read["the_orders"][WIDE]["alike"] != []                                                            # two rows have two orders: eight order seeds hold pairs alike, trained all the same
    # TWELVE trainings, `again`'s four FROM L5f's smoke averaged `published`; three averaged adapters that ARE each of their members (kept, served and SAID); four measurements.
    own_calls = calls[before:]
    assert len(own_calls) == 12 and [call["seed"] for call in own_calls] == [3] * 12
    assert [call["more"]["start"] for call in own_calls] == [*[of_l4p / "adapters" / "second_full"] * 4, *[theirs / "adapters" / "published"] * 4, *[of_l4p / "adapters" / "second_full"] * 4]
    assert solver.measured[-4:] == [(GOAL, 32, 1001, 1024)] * 4 and measured[-4:] == [64] * 4
    for model in AVERAGED:
        made = summaries[f"ladder_l5_cont_average_{model}"]
        assert (made["read_back"]["is_the_mean"], made["read_back"]["is_none_of_them"], made["one_of_its_models_in_a_smoke_run"]) == (True, False, True)
    assert summaries["ladder_l5_cont_average_wide"]["made"]["adapters"] == 8 and summaries["ladder_l5_cont_measure_single"]["adapter"] == str(theirs / "adapters" / "published_1")
    # THE REPORT says A SMOKE RUN; the checks pass; its lines are no reads of the continuation.
    report = summaries["ladder_l5_cont_report"]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["smoke"] == smoke
    assert any(f"A SMOKE RUN: L5's smoke run reads INCONCLUSIVE ({', '.join(failed)}) and was read all the same; one deep solve is enough" in line and
               line.endswith("Nothing here is a read of the continuation") for line in report["lines"])
    assert report["can_this_run_be_read"]["the_run"][THE_SETS]["rows_of_more_by_l5ss_rule"] == 4 and report["the_bar"]["run"] == str(theirs)
    assert report["branches"] == {"what_made_it": ONE_CONTINUATION, WIDE: ADDS, AGAIN: NOT_SHOWN_AT_ONE, MORE: COSTS, "what_follows": FURTHER_SEEDS}
    # WITHOUT THE VARIABLE nothing is taken loosely.
    monkeypatch.delenv(SMOKE)
    assert ladder_l5_cont.smoke_run() is False


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_cont"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l5_cont.STEPS) == STEPS and ladder_l5_cont.ENTRY_STAGE == "ladder_l5_cont" and len(STEPS) == 21
    # THE SECTION'S ORDER: the first step; `single` measured; `wide`'s four trainings (order seeds 5 to 8), its averaged adapter and its measurement; `again`'s; `more`'s; the report.
    assert STEPS[1:8] == ["ladder_l5_cont_measure_single", *(f"ladder_l5_cont_train_wide_{order}" for order in (5, 6, 7, 8)), "ladder_l5_cont_average_wide", "ladder_l5_cont_measure_wide"]
    assert STEPS[8:14] == [*(f"ladder_l5_cont_train_again_{order}" for order in (1, 2, 3, 4)), "ladder_l5_cont_average_again", "ladder_l5_cont_measure_again"]
    assert STEPS[14:] == [*(f"ladder_l5_cont_train_more_{order}" for order in (1, 2, 3, 4)), "ladder_l5_cont_average_more", "ladder_l5_cont_measure_more", "ladder_l5_cont_report"]
    # THE SMOKE STAGE: the same steps, under the runs L5f's smoke stage names WITHOUT its smoke-only variable and without L5d's run variable, a run of its own, and ITS variable.
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_cont_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    of_l5f = entry.step_fields(entry.STAGES["ladder_l5_full_smoke"][2])[2]["environment"]
    assert smoke[0][2]["environment"] == {**{name: value for name, value in of_l5f.items() if name not in (ladder_l5_full.SMOKE_VARIABLE, ladder_l5_draws.RUN_VARIABLE)},
                                          RUN: "ladder_l5_cont_smoke", SMOKE: "1"}
    own_options = smoke[0][2]["environment"]
    assert (own_options[ladder_l5.RUN_VARIABLE], own_options[L4P_RUN], own_options[L5F_RUN]) == ("ladder_l5_deep_smoke", "ladder_l4_dose_r64_smoke", "ladder_l5_full_smoke")
    assert not {ladder_l5.SMOKE_VARIABLE, ladder_l6.SMOKE_VARIABLE, ladder_l5_stock.SMOKE_VARIABLE, ladder_l5_full.SMOKE_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE} & set(own_options)
    # THE SETTING, as the package's own config has it: the start, the scout's seed, two further ones that wait for the escalation; every seed one of L5f's.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    assert whole["ladder_loop"]["l5_cont"] == {"start": "second_full", "seeds": [3], "more_seeds": [4, 5], "escalated": False} == CONFIG["ladder_loop"]["l5_cont"]
    own = ladder_l5_cont.the_settings(whole)
    of_theirs = ladder_l5_full.the_settings(whole)
    assert (own.start, own.seeds, own.more, own.escalated, own.full, own.draws, own.deep) == ("second_full", (3,), (4, 5), False, of_theirs, of_theirs.draws, of_theirs.deep)
    assert set((*own.seeds, *own.more)) <= set((*of_theirs.seeds, *of_theirs.more)) and own.draws.recipe == "mean_constant" and (own.draws.sampling, own.draws.attempts) == ("reach", 32)
    # THE RUN DIRECTORY IS ITS OWN; L5f's run of the seed, `second_full` and the bar live where L5f's own functions find them: no path is stated twice.
    assert [ladder_l5_cont.run_name(own, seed) for seed in (3, 5)] == ["ladder_l5_cont_second_full_seed3", "ladder_l5_cont_second_full_seed5"]
    start = ladder_l5_cont.the_start(whole, own)
    assert start == ladder_l5_full.the_start(whole, of_theirs) and (start.name, start.check.rank, start.check.alpha) == ("second_full", 64, 128)
    assert (ladder_l5_cont.average_marker("more"), ladder_l5_cont.REPORT_FILE, ladder_l5_cont.THE_BAR) == ("ladder_l5_cont_average_more", "report_ladder_l5_cont.json", "l5f_published")
    assert list(ladder_l5_cont.l5fs_adapters(Path("/run"))) == [*OF_THEIRS, "published"] and ladder_l5_cont.l5fs_adapters(Path("/run"))["published_1"] == Path("/run/adapters/published_1")
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l5_cont"]' in path.read_text()] == ["gpu/ladder_l5_cont.py"]
    others = (set(ladder_l5.STEPS) | set(ladder_l6.STEPS) | set(ladder_l4_rows.STEPS) | set(ladder_l5_noise.STEPS) | set(ladder_l5_draws.STEPS) | set(ladder_l5_stock.STEPS)
              | set(ladder_l5_full.STEPS))
    assert not set(ladder_l5_cont.STEPS) & others and all(name.startswith("ladder_l5_cont_") for name in ladder_l5_cont.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l5_cont.STEPS for step in entry.STAGES[name])) == ["ladder_l5_cont", "ladder_l5_cont_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(SMOKE) == ["ladder_l5_cont_smoke"] and names(RUN) == ["ladder_l5_cont_smoke", "ladder_l5_measure_smoke", "ladder_l6b_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L5_CONT_SMOKE" and RUN == "RLVR_LEAN_LADDER_L5_CONT_RUN"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l5_cont.py", "runner/entry.py"]
    # ONE BODY EACH: its trainings are L4t's own training body (handed the start model), its measurements the ceiling's own measuring body with no rung, its averaged adapters
    # L5n's own function (handed the adapters another run keeps); L5's run is read by L5d's own first-step reading; `second_full` by L5f's; `more` by L5s's own rule.
    assert ladder_l5_cont.train_a_model is ladder_l4_rows.train_a_model and ladder_l5_cont.measure_model is ladder_ceiling.measure_model and ladder_l5_cont.the_set is ladder_l5_draws.the_set
    own_stage = ladder_l5_cont.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == (
        "ladder_l5_cont", "ladder_l5_cont", "l5_cont", "L5c", "ladder_l5_cont_prepare", "ladder_loop.l5_cont.start")
    source = (PACKAGE / "gpu" / "ladder_l5_cont.py").read_text()
    for called in ("ladder_l5_draws.drawn_by_the_seed(", "ladder_l5_full.read_second_full(", "ladder_l5_full.the_start(", "ladder_l5_full.own_run(", "ladder_l5_noise.the_mean_adapter(",
                   "ladder_l5_stock.stock_by_the_seed(", "ladder_l5_stock.the_rule_again(", "start_from=start", "rungs=False"):
        assert called in source, called
    assert "_episodes(" not in source and "LoRARequest" not in source and "mean_of_adapters" not in source and "read_the_search(" not in source and "solve_rows(" not in source
    assert "ladder_l4_dose_pre_r64" not in source and "ladder_l5_draws_second_full" not in source and '"second_full"' not in source and '["ladder_loop"]["l5_full"]' not in source
    report_source = (PACKAGE / "reporting" / "ladder_l5_cont.py").read_text()
    assert "checks_of_the_models(" in report_source and "lines_of_the_checks(" in report_source and "against_the_start(" in report_source and "the_start_adapter(" in report_source
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l5_cont.STEPS" in main and "ladder_l5_cont," in main
    # WHAT WAS CUT OUT FOR IT IS CALLED BY THE STAGE IT WAS CUT OUT OF TOO: L5s's first step builds its sets by the function this stage calls; L5n's averaging body takes adapters another
    # run keeps and, handed none, averages its own run's alone; L5d's check lines say "its four's" unless told otherwise.
    of_l5s = (PACKAGE / "gpu" / "ladder_l5_stock.py").read_text()
    assert of_l5s.count("stock_by_the_seed(config, deep, seed, smoke, drawn)") == 1 and "def stock_by_the_seed(" in of_l5s
    assert "kept: Mapping[str, Path] | None = None" in (PACKAGE / "gpu" / "ladder_l5_noise.py").read_text()
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if "kept={" in path.read_text()] == ["gpu/ladder_l5_cont.py"]
    assert 'mean_of: str = "its four\'s"' in (PACKAGE / "reporting" / "ladder_l5_draws.py").read_text()
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if "mean_of=" in path.read_text()] == ["reporting/ladder_l5_cont.py"]
