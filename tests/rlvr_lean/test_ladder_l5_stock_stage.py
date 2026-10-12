"""L5s, the stage end to end on what L5's FINISHED run and the runs that task read stored: the refusals of its first step
(the seed, the escalation, L5d's recipe, L5's run, a pretraining file too small for the bar); the two sets, WHICH HOLD
EVERY deep solve and EVERY ordinary solve and THAT SEED's ROWS OF L5d's THREE SETS; each set trained in four orders, its
averaged adapter and its measurement on the goal set's first sampling ALONE; the report's checks failing when they
should; THE DOSE against L5d's run of the seed at its edges, and NOT READ when that run is not there; the smoke stage and
its smoke-only path; the registration; and L5 and L5d left writing what they wrote. Spec:
docs/spec/ladder-loop.spec.md, "L5s: the dose. Every own solve in stock that the model finds less than half the
time, against the same number of published proofs". The engine and Lean are scripted; nothing touches a GPU or the
network. The rules on hand-made rows are `test_ladder_l5_stock.py`."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adapter_files import _numbers  # noqa: E402 - an adapter's weights file read by hand
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l4_stage import _on_the_gpu  # noqa: E402
from test_ladder_l5_draws_stage import _l5ns_report, _stage5d  # noqa: E402 - L5d's own stage, run beside this one
from test_ladder_l5_draws_stage import _own_files as l5d_files  # noqa: E402 - what a seed of L5d leaves, frozen there
from test_ladder_l5_noise_stage import GOAL, NoiseSolver, _trainings  # noqa: E402 - the goal set's first sampling scripted model by model; the trainings with the model taken out
from test_ladder_l5_stage import OWN_FILES as L5_FILES  # noqa: E402 - what L5 leaves at the search's own seed, frozen there
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
from test_ladder_l6_stage import DRAWN, SOLVED_1, _round_one, _smoke_world  # noqa: E402 - L5's stage run to its end on a base map of twelve problems; the smoke runs' world

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import DEEP, PUBLISHED, SHALLOW  # noqa: E402
from rlvr_lean.domain.ladder_round.l5 import MODELS as OF_L5D  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_draws import MEAN_CONSTANT  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_noise import AVERAGE_OF_FOUR, ORDERS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_stock import COSTS, ESCALATE, FURTHER_SEEDS, MODELS, NO_FURTHER_SEED, PUBLISHED_STOCK, STOCK, hold_the_sets  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_l4_rows, ladder_l4_start, ladder_l5, ladder_l5_draws, ladder_l5_noise, ladder_l5_stock, ladder_l6, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.adapter_files import is_the_mean  # noqa: E402

RUN, SMOKE = ladder_l5_stock.RUN_VARIABLE, ladder_l5_stock.SMOKE_VARIABLE
TRAINED = [f"{model}_{order}" for model in MODELS for order in ORDERS]
STEPS = ["ladder_l5_stock_prepare",
         *(step for model in MODELS for step in (*(f"ladder_l5_stock_train_{model}_{order}" for order in ORDERS), f"ladder_l5_stock_average_{model}", f"ladder_l5_stock_measure_{model}")),
         "ladder_l5_stock_report"]
OWN_FILES = sorted([
    *(f"adapters/{name}/{file}" for name in (*MODELS, *TRAINED) for file in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l5_stock_loss_{name}.json" for name in TRAINED), *(f"l5_stock_training_{name}.jsonl" for name in (*MODELS, *TRAINED)),
    *(f"{step}.done.json" for step in STEPS), "problems.jsonl", "report_ladder_l5_stock.json", "l5_stock_ordinary_solves.jsonl",
    "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", "l4_stored_pre_r64_reach.jsonl", "l5_held_aside_problems.jsonl"])
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "rank", "models", "recipe", "branches", "inconclusive", "failed_checks",
               "can_this_run_be_read", "of_this_seed", "figures", "the_dose", "l5d_at_this_seed", "beside_the_dose", "against_the_start_model", "overfitting", "timed_out_in_lean",
               "what_follows", "the_seeds", "the_search", "the_stock", "the_sets", "the_orders", "the_sampling", "training_sets", "training_recipe", "models_made", "smoke",
               "what_it_cannot_say", "attempts", "not_to_be_read", "not_built", "not_measured", "adapters"]
ORDINARY = ["fixture_p8", "fixture_p9", "fixture_p10", "fixture_p12"]       # the map problems `pre_r64` solved in 1 to 3 of its 8 map attempts (fixture_p11: in 4, no ordinary solve)
SUCCESSES = (12, 6)                                                         # of the 32 attempts at the goal problem: `stock`, then `published_stock`


def _stage5s(config, monkeypatch, until=None, stage_name="ladder_l5_stock"):
    """The stage's steps in its order; the trainings and the averagings by their GPU path (the stand-in makes no adapter)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l5_stock.STEPS:
            continue
        run = ladder_l5_stock.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if "_train_" in step or "_average_" in step else run(config)
        if step == until:
            break
    return summaries


def _l5ds_report(directory, successes=(4, 8, 12), seed=3, **changed):
    """L5d's FINISHED run of a seed, hand-made (it is only read): what its report says of its own seed, its three models
    closing the goal problem in `successes` of the first sampling's 32 attempts. Returns its directory."""
    directory.mkdir(exist_ok=True)
    figure = lambda count: {"problems": 1, "successes": count, "attempts": 32, "per_attempt": round(count / 32, 6), "per_1000": round(1000 * count / 32, 2), "sampling_noise": 0.05}      # noqa: E731
    of_this_seed = {"seed": seed, "recipe": "mean_constant", "ok": True, "inconclusive": False, "failed_checks": [],
                    "the_measurement": {"sampling": "reach", "episodes": 32, "sampling_seed": 1001, "problems": 1},
                    "figures": {model: {"all_of_g": figure(count)} for model, count in zip(OF_L5D, successes)}, **changed}
    (directory / "ladder_l5_draws_report.done.json").write_text(json.dumps({"ok": True}))
    (directory / "report_ladder_l5_draws.json").write_text(json.dumps({"ok": True, "of_this_seed": of_this_seed}))
    return directory


@pytest.fixture
def stock(deep, monkeypatch, tmp_path):  # noqa: F811
    """`deep`'s world with what this stage reads on the box: a base map of twelve problems with a hand-made map of
    `pre_r64` (FOUR ordinary solves by the rule), and L5's own stage RUN TO ITS END there at its search's seed (TWO deep
    solves, so L5d's sets are FOUR rows and this stage's N is six: `stock` is twelve rows and `published_stock` the
    fixture's twelve published rows, all of them). The batch is ONE row. Then this stage's task of SEED 3, which carries
    NO variable. L5d's run of seed 3 is NOT THERE (`_l5ds_report` makes one). `solver.measured`: every prompt of a
    measurement of the goal set's first sampling made from here on; the k-th closes the goal as `SUCCESSES` says."""
    for name in (RUN, SMOKE, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    config["training"]["effective_batch"] = 1
    first = _round_one(deep, monkeypatch, tmp_path, solved=SOLVED_1, l4p=False, settings={"problems": 4, "held_aside": 1, "minimum": 2})
    root = deep.store().root
    assert root.name == "ladder_l5_deep_pre_r64_seed0" and first["ladder_l5_report"]["inconclusive"] is False and first["ladder_l5_report"]["ok"] is True
    assert first["ladder_l5_prepare"]["rows_of_each_kind"] == 2
    calls, kits = [], []
    _trainings(monkeypatch, calls)
    solver = NoiseSolver(deep.solver, SUCCESSES)
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"]}),
        (solver, stand_in_parameters))[1]))
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l5_stock"])        # THE STAGE names no variable
    monkeypatch.setenv(SEEDS, "3")
    runs = ladder_l5._runs(config)
    return SimpleNamespace(config=config, deep=deep, first=first, l5=root, calls=calls, kits=kits, solver=solver, runs=runs, l5d=runs / "ladder_l5_draws_pre_r64_seed3",
                           store=lambda: ladder_l5_stock._store(config, ladder_l5_stock.the_settings(config)))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_builds_the_two_sets_trains_each_in_four_orders_measures_the_mean_on_g_alone_and_reads_the_dose(stock, monkeypatch):  # noqa: F811
    config, calls, kits, solver, deep_ = stock.config, stock.calls, stock.kits, stock.solver, stock.deep
    _l5ds_report(stock.l5d)                                                         # L5d's run of seed 3 is THERE: `deep` closes the goal in 12 of 32, as `stock` does
    read_only = {path: _all(path) for path in (stock.l5, stock.l5d, deep_.map, deep_.l4t2, deep_.rank, deep_.l4t, deep_.arm, deep_.pre)}
    summaries = _stage5s(config, monkeypatch)
    store = stock.store()
    assert store.root.name == "ladder_l5_stock_pre_r64_seed3" and store.root.parent == stock.l5.parent and list(summaries) == STEPS == list(ladder_l5_stock.STEPS)
    assert {path: _all(path) for path in read_only} == read_only               # L5's RUN, L5d's AND EVERY RUN L5 READ WERE ONLY READ: file for file
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert sorted(name for name in _all(stock.l5) if not name.startswith("episodes_")) == L5_FILES      # ... and L5's run is what L5's own test holds it to

    # ---- the first step: the seed, L5's search, the stock, the one sampling
    read = summaries["ladder_l5_stock_prepare"]
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"], read["models"], read["trained"], read["averaged"]) == (
        "l4", "pretrained on published proofs", "L5s", 3, "pre_r64", list(MODELS), TRAINED, list(MODELS))
    assert (read["start_adapter"], read["start_run"]) == (str(deep_.rank / "adapters" / "pre_r64"), str(deep_.rank)) and read["smoke"] is None
    assert read["the_seeds"] == {"what": read["the_seeds"]["what"], "seeds": [3], "more_seeds": [4, 5], "escalated": False, "fixed": [3]}
    of_the_search, of_the_stock = read["the_search"], read["the_stock"]
    assert (of_the_search["run"], of_the_search["seed"], of_the_search["deep_solves"], of_the_search["rows_of_each_kind"], of_the_search["inconclusive"]) == (str(stock.l5), 0, 2, 2, False)
    # EVERY deep solve and EVERY ordinary solve the rule gives (four: twice the two L5d's `shallow` holds), and as many published rows; the bar is twice as many published rows.
    assert {key: of_the_stock[key] for key in ("deep_solves", "ordinary_solves", "ordinary_solves_of_the_rule", "ordinary_solves_taken_by_the_runs_minimum", "own_rows", "rows",
                                                "solved_at_least", "solved_at_most", "ordinary_file")} == {
        "deep_solves": 2, "ordinary_solves": 4, "ordinary_solves_of_the_rule": 4, "ordinary_solves_taken_by_the_runs_minimum": 0, "own_rows": 6,
        "rows": {"stock": 12, "published_stock": 12}, "solved_at_least": 1, "solved_at_most": 3, "ordinary_file": "l5_stock_ordinary_solves.jsonl"}
    assert (read["the_sampling"]["name"], read["the_sampling"]["episodes"], read["the_sampling"]["sampling_seed"], read["the_sampling"]["sets"]) == (
        "reach", 32, 1001, {"stock": "l4_reach_stock", "published_stock": "l4_reach_published_stock"})
    assert read["recipe"]["name"] == "mean_constant" and read["recipe"]["adapter_seed"] == 3 and read["recipe"]["optimizer_steps"] == 12

    # ---- THE TWO SETS, as stored: held by the rule's own function; a published row by id and place, NEVER with text
    sets = {model: store.read_rows(f"l5_stock_training_{model}.jsonl") for model in MODELS}
    pretraining = ladder_l4_start.pretrained_on(deep_.rank, read["pretraining_file_sha256"], "pre_r64")[0]
    assert hold_the_sets(sets, pretraining, 3, list(SOLVED_1), ORDINARY) == {"own_rows": 6, "deep_solves": 2, "ordinary_solves": 4, "rows": {"stock": 12, "published_stock": 12}}
    own = [row for row in sets[STOCK] if row["origin"] != "pretraining"]
    assert sorted(row["problem_id"] for row in own if row["origin"] == "deep") == sorted(SOLVED_1) and sorted(row["problem_id"] for row in own if row["origin"] == "shallow") == sorted(ORDINARY)
    assert {model: read["the_sets"][model]["rows_by_origin"] for model in MODELS} == {STOCK: {"deep": 2, "shallow": 4, "pretraining": 6}, PUBLISHED_STOCK: {"pretraining": 12}}
    published = [row for own_ in sets.values() for row in own_ if row["origin"] == "pretraining"]
    assert len(published) == 18 and not any(field in row for row in published for field in TEXT_FIELDS) and all(isinstance(row["row_of_the_file"], int) for row in published)
    assert {row["id"] for row in sets[STOCK] if row["origin"] == "pretraining"} < {row["id"] for row in sets[PUBLISHED_STOCK]} and len({row["id"] for row in sets[PUBLISHED_STOCK]}) == len(pretraining) == 12
    drawn = store.read_rows("l5_stock_ordinary_solves.jsonl")
    assert sorted(row["problem_id"] for row in drawn) == sorted(ORDINARY) and all(row["taken_by"] == "the rule" for row in drawn)
    # THEY HOLD THAT SEED's ROWS: L5d's three sets at seed 3, built here by L5d's own first step, are rows of these two, by id.
    monkeypatch.setenv(ladder_l5_draws.RUN_VARIABLE, "l5d_beside")
    _l5ns_report(stock.runs, AVERAGE_OF_FOUR)
    theirs = ladder_l5_draws.drawn_by_the_seed(config, ladder_l5_draws.the_settings(config), 3, False, ladder_l5_draws.the_reader(ladder_l5_draws.the_settings(config), False), True)["sets"]
    monkeypatch.delenv(ladder_l5_draws.RUN_VARIABLE)
    ids = {model: {row["id"] for row in sets[model]} for model in MODELS}
    assert all(len(theirs[model]) == 4 for model in OF_L5D) and {row["id"] for row in theirs[DEEP]} <= ids[STOCK] and {row["id"] for row in theirs[SHALLOW]} <= ids[STOCK]
    assert {row["id"] for row in theirs[PUBLISHED]} <= ids[PUBLISHED_STOCK] and read["holds_the_seeds_rows"]["rows"] == {"published": 4, "shallow": 4, "deep": 4}
    # EACH SET IN THE ORDER OF EACH ORDER SEED: the same twelve rows; the order seed that is the task's own seed (3) gives the seed's own order, and is NAMED.
    placed = lambda row: (row["id"], row["problem_id"], row.get("row_of_the_file"))      # noqa: E731
    for model in MODELS:
        for order in ORDERS:
            own_, entry_ = store.read_rows(f"l5_stock_training_{model}_{order}.jsonl"), read["training_sets"][f"{model}_{order}"]
            assert [placed(row) for row in own_] == [placed(row) for row in training_order(sets[model], order)] and [row["row"] for row in own_] == list(range(12))
            assert ([placed(row) for row in own_] == [placed(row) for row in sets[model]]) is (order == 3) and (entry_["order_seed"], entry_["model"], entry_["rows"], entry_["optimizer_steps"]) == (order, model, 12, 12)
        assert read["the_orders"][model] == {"the_order_of_the_tasks_seed": [3], "alike": []}

    # ---- EIGHT TRAININGS, `stock`'s four and then `published_stock`'s, each the recipe as it is with its rates recorded, from the start adapter, with the task's seed
    start = deep_.rank / "adapters" / "pre_r64"
    assert [call["more"] for call in calls] == [{"start": start, "learning_rates": True}] * 8 and [call["seed"] for call in calls] == [3] * 8
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_stock_pre_r64_seed3_{name}" for name in TRAINED] and all(call["lora"]["rank"] == 64 for call in calls)
    for call, name in zip(calls, TRAINED):
        train, loss = summaries[f"ladder_l5_stock_train_{name}"], json.loads(store.path(f"l5_stock_loss_{name}.json").read_text())
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in store.read_rows(f"l5_stock_training_{name}.jsonl")]
        assert loss["rows_trained"] == [row["id"] for row in store.read_rows(f"l5_stock_training_{name}.jsonl")] and train["adapter"] == str(store.root / "adapters" / name)
        assert train["learning_rate"]["decay"] is False and f"the ORDER SEED {name[-1]}" in train["order"] and len(loss["training_steps"]) == 12
    # TWO AVERAGED ADAPTERS, trained by nothing, by L5n's own function: every number the mean of its four's at that place; read back; none of them; at rank 64.
    for index, model in enumerate(MODELS):
        made, numbers = summaries[f"ladder_l5_stock_average_{model}"], [4 * index + order for order in ORDERS]                    # the k-th training's adapter holds k and k / 2
        directory, of = store.root / "adapters" / model, [store.root / "adapters" / f"{model}_{order}" for order in ORDERS]
        assert (made["model"], made["recipe"], made["of"], made["adapter"], made["averaged"], made["check"]) == (model, "mean_constant", [f"{model}_{order}" for order in ORDERS], str(directory), True, "L5s")
        assert set(_numbers(directory / "adapter_model.safetensors", "base_model.model.layers.0.q_proj.lora_A.weight").tolist()) == {sum(numbers) / 4}
        assert made["read_back"] == is_the_mean(directory / "adapter_model.safetensors", [path / "adapter_model.safetensors" for path in of]) and made["read_back"]["passes"] is True
        assert made["adapter_saved"]["rank"] == 64 and made["one_of_its_models_in_a_smoke_run"] is False
    # TWO MEASUREMENTS, of the averaged adapters and of nothing else, each right after its adapter is made: `stock` is trained, averaged and measured before `published_stock`.
    assert solver.measured == [(GOAL, 32, 1001, 1024)] * 2 and [kit["max_lora_rank"] for kit in kits] == [64] * 2
    assert sorted(path.name for path in store.root.glob("episodes_*.done.json") if "_block_" not in path.name) == sorted(f"episodes_l4_reach_{model}.done.json" for model in MODELS)

    # ---- THE REPORT: the checks first, all passing; then the reads in the section's order
    report = summaries["ladder_l5_stock_report"]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["failed_checks"] == [] and report["not_to_be_read"] == []
    assert json.loads(store.path("report_ladder_l5_stock.json").read_text()) == report and store.done_summary("ladder_l5_stock_report")["headline"] == report["headline"]
    checks = report["can_this_run_be_read"]
    assert list(checks) == ["what", "the_run", *TRAINED, *MODELS] and list(checks["the_run"]) == [
        "each_set_is_its_rule_made_again_and_held", "the_stored_first_sampling_of_the_start_model_is_the_settings"]
    assert checks["the_run"]["each_set_is_its_rule_made_again_and_held"]["counted"] == {"own_rows": 6, "deep_solves": 2, "ordinary_solves": 4, "rows": {"stock": 12, "published_stock": 12}}
    assert all(entry_["passes"] is True for name in ("the_run", *TRAINED, *MODELS) for entry_ in checks[name].values())
    assert all(list(checks[model]) == ["its_numbers_are_the_mean_of_its_models_and_it_is_served", "lean_answered"] for model in MODELS)
    figures = report["figures"]
    assert (figures[STOCK]["all_of_g"]["successes"], figures[STOCK]["all_of_g"]["attempts"], figures[PUBLISHED_STOCK]["all_of_g"]["successes"]) == (12, 32, 6)
    # THE DOSE, at its upper edge: `stock` minus `deep` of L5d's run of this seed is exactly zero, which is AT OR ABOVE ZERO.
    assert report["l5d_at_this_seed"]["read"] is True and report["l5d_at_this_seed"]["run"] == str(stock.l5d) and report["l5d_at_this_seed"]["the_primarys_difference"] == round((8 + 12) / 64 - 4 / 32, 6)
    assert (report["the_dose"]["name"], report["the_dose"]["difference"], report["the_dose"]["stock"], report["the_dose"]["deep"]) == (
        ESCALATE, 0.0, {"successes": 12, "attempts": 32}, {"successes": 12, "attempts": 32})
    assert report["branches"] == {"the_dose": ESCALATE, "what_follows": FURTHER_SEEDS} and report["what_follows"]["seeds"] == [4, 5]
    beside = report["beside_the_dose"]
    assert (beside["own_proofs_at_this_dose"]["difference"], beside["own_proofs_at_this_dose"]["l5ds_primary_at_this_seed"]) == (round(6 / 32, 6), round((8 + 12) / 64 - 4 / 32, 6))
    assert beside["the_steps_without_its_own_proofs"] == {"what": beside["the_steps_without_its_own_proofs"]["what"], "read": True, "why": None, "difference": round(2 / 32, 6)}
    start_rows = store.read_rows("l4_stored_pre_r64_reach.jsonl")
    of_the_start = start_rows[0]["resolved"] / start_rows[0]["episodes"]
    for model, count in zip(MODELS, SUCCESSES):
        against = report["against_the_start_model"][model]
        assert against["per_attempt"]["all_of_g"] == {"problems": 1, "model": round(count / 32, 6), "start": round(of_the_start, 6), "difference": round(count / 32 - of_the_start, 6)}
        assert (against["solved"], against["solved_by_the_start"], against["gained"], against["lost"]) == (1, int(of_the_start > 0), int(of_the_start == 0), 0)
        assert report["overfitting"][model]["lost"] == 0 and set(report["overfitting"][model]) == {"lost", "gained", "share_of_distinct_attempts", "long", "long_solved_by_the_start"}
    assert report["timed_out_in_lean"]["checked_again"] is False and report["of_this_seed"]["the_dose"] == 0.0 and report["of_this_seed"]["stock_minus_published_stock"] == round(6 / 32, 6)
    # THE READS STAND IN THE SECTION'S ORDER, after the checks: one line each, the last of them what follows.
    in_order = ("THE FIGURES, ", "THE DOSE (the scout's read): ", "ITS OWN PROOFS AT THIS DOSE (6 own rows against the 2 of L5d's sets)", "THE STEPS WITHOUT ITS OWN PROOFS, ",
                "THE SOLVE RATE AGAINST THE START MODEL, ", "OVERFITTING, ", "TIMEOUTS, ", "WHAT FOLLOWS: ")
    assert [line.split(": ", 1)[1].startswith(said) for line, said in zip(report["lines"][-len(in_order):], in_order)] == [True] * len(in_order)
    assert all(" CHECK, " in line for line in report["lines"][1:-len(in_order)]) and len(report["lines"]) > len(in_order) + 3
    assert all(line.startswith("l4 (pretrained on published proofs): ") for line in report["lines"]) and report["lines"][-1].split(": ", 1)[1].startswith(f"WHAT FOLLOWS: {FURTHER_SEEDS}")
    assert report["headline"].startswith("L5s (the dose; pretrained on published proofs) seed 3, from `pre_r64`: per 1,000 attempts on all of G `stock` 375.00, `published_stock` 187.50 "
                                         "(L5d's `deep` at this seed 375.00); THE DOSE: THE SOLVE RATE DOES NOT FALL WITH THE DOSE: ESCALATE (+0.00). What follows: ESCALATE")
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == sorted([*MODELS, *TRAINED]) and len(report["not_built"]) == 2

    # ---- THE DOSE'S OTHER BRANCHES, the report built again each time (no step is made again): L5d's `deep` one success above `stock` is 31 per 1,000 below zero: IT COSTS
    _l5ds_report(stock.l5d, successes=(4, 8, 13))
    costs = ladder_l5_stock.ladder_l5_stock_report(config)
    assert (costs["the_dose"]["name"], costs["the_dose"]["difference"], costs["branches"]) == (COSTS, round(-1 / 32, 6), {"the_dose": COSTS, "what_follows": NO_FURTHER_SEED})
    assert len(solver.measured) == 2 and len(calls) == 8
    # L5d's RUN OF THE SEED NOT THERE, of another recipe, INCONCLUSIVE, or measured otherwise: THE DOSE IS NOT READ, and said; nothing is refused, and every other read is made.
    for changed, why in (({"recipe": "constant"}, "its recipe is `constant` and this run's `mean_constant`"), ({"inconclusive": True, "failed_checks": ["`deep`: lean_answered"]}, "it reads INCONCLUSIVE"),
                         ({"seed": 4}, "it is a report of seed 4"), ({"the_measurement": {"sampling": "reach", "episodes": 16, "sampling_seed": 1001, "problems": 1}}, "it was measured otherwise")):
        _l5ds_report(stock.l5d, **changed)
        not_read = ladder_l5_stock.ladder_l5_stock_report(config)
        assert not_read["branches"] == {"the_dose": NOT_READ, "what_follows": NOT_READ} and why in not_read["l5d_at_this_seed"]["why"] and not_read["inconclusive"] is False
    for name in ("ladder_l5_draws_report.done.json", "report_ladder_l5_draws.json"):
        (stock.l5d / name).unlink()
    not_there = ladder_l5_stock.ladder_l5_stock_report(config)
    assert not_there["l5d_at_this_seed"] == {"read": False, "run": str(stock.l5d), "why": "ladder_l5_draws_pre_r64_seed3 does not hold ['ladder_l5_draws_report.done.json', 'report_ladder_l5_draws.json']: "
                                                                                              "L5d's run of this seed is not on the box, or not finished"}
    assert not_there["the_dose"]["name"] == NOT_READ and "NOT READ" in not_there["the_dose"]["reason"] and not_there["beside_the_dose"]["the_steps_without_its_own_proofs"]["read"] is False
    assert not_there["beside_the_dose"]["own_proofs_at_this_dose"] == {"what": beside["own_proofs_at_this_dose"]["what"], "difference": round(6 / 32, 6), "l5ds_primary_at_this_seed": None}
    assert any("THE DOSE (the scout's read): NOT READ" in line for line in not_there["lines"]) and not_there["ok"] is True and "THE DOSE: NOT READ" in not_there["headline"]


def test_a_failed_check_makes_the_run_inconclusive_and_nothing_else_is_said(stock, monkeypatch):  # noqa: F811
    config = stock.config
    _l5ds_report(stock.l5d)
    report = _stage5s(config, monkeypatch)["ladder_l5_stock_report"]
    store = stock.store()
    assert report["inconclusive"] is False

    def failing(path, changed):
        kept = path.read_text()
        path.write_text(changed(kept))
        failed = ladder_l5_stock.ladder_l5_stock_report(config)
        path.write_text(kept)
        return failed

    as_rows = lambda text: [json.loads(line) for line in text.splitlines()]                                             # noqa: E731
    as_text = lambda own: "".join(json.dumps(row) + "\n" for row in own)                                               # noqa: E731
    # A SET THAT IS NOT IN ITS ORDER SEED'S ORDER (two of its stored rows changed places): the first check fails, THE RUN IS INCONCLUSIVE, and no read is made.
    failed = failing(store.path("l5_stock_training_stock_2.jsonl"), lambda text: as_text([{**as_rows(text)[1], "row": 0}, {**as_rows(text)[0], "row": 1}, *as_rows(text)[2:]]))
    check = failed["can_this_run_be_read"]["the_run"]["each_set_is_its_rule_made_again_and_held"]
    assert failed["failed_checks"] == ["the run: each_set_is_its_rule_made_again_and_held"] and check["sets_not_in_their_order_seeds_order"] == ["stock_2"] and failed["inconclusive"] is True
    assert failed["branches"] == {"the_dose": INCONCLUSIVE, "what_follows": NOT_READ} and "difference" not in failed["the_dose"] and list(failed["figures"]) == ["what", "problems", "measured_and_not_read"]
    assert list(failed["beside_the_dose"]) == list(failed["against_the_start_model"]) == list(failed["overfitting"]) == ["measured_and_not_read"]
    assert failed["of_this_seed"]["figures"] is None and failed["of_this_seed"]["the_dose"] is None and failed["of_this_seed"]["inconclusive"] is True
    assert failed["lines"][-1].endswith("THIS RUN IS INCONCLUSIVE: the run: each_set_is_its_rule_made_again_and_held. Nothing else is said of it")
    assert not any("THE DOSE (the scout's read)" in line or "WHAT FOLLOWS" in line for line in failed["lines"]) and "THIS RUN IS INCONCLUSIVE" in failed["headline"]
    # `stock` WITHOUT ONE OF THE ORDINARY SOLVES THE RULE GIVES (the map read again by the report now counts a fifth): the stored set is not the rule made again.
    of_the_map = store.done_summary("ladder_l5_stock_prepare")["the_map"]
    failed = failing(Path(of_the_map["run"]) / f"episodes_{of_the_map['set']}_problems.jsonl", lambda text: as_text([{**row, "resolved": 2} if row["problem_id"] == "fixture_p11" else row for row in as_rows(text)]))
    check = failed["can_this_run_be_read"]["the_run"]["each_set_is_its_rule_made_again_and_held"]
    assert failed["inconclusive"] is True and check["counted"] is None and "`stock` holds 4 rows of origin `shallow` and the rule gives 5" in check["refused"] and "fixture_p11" in check["refused"]
    # A PUBLISHED ROW WITH ITS TEXT in a stored set fails it too.
    failed = failing(store.path("l5_stock_training_published_stock.jsonl"), lambda text: as_text([{**as_rows(text)[0], "proof": "  done"}, *as_rows(text)[1:]]))
    assert failed["can_this_run_be_read"]["the_run"]["each_set_is_its_rule_made_again_and_held"]["published_rows_with_text"] == 1 and failed["inconclusive"] is True
    # AN AVERAGED ADAPTER THAT IS ONE OF ITS MODELS (its step's stored read-back is made to say so) fails its check in a real run.
    marker = store.path("ladder_l5_stock_average_stock.done.json")
    failed = failing(marker, lambda text: json.dumps({**json.loads(text), "read_back": {**json.loads(text)["read_back"], "is_none_of_them": False, "passes": False}}))
    assert failed["failed_checks"] == ["`stock`: its_numbers_are_the_mean_of_its_models_and_it_is_served"] and failed["branches"]["the_dose"] == INCONCLUSIVE
    # ... and with everything as it was the run reads again.
    assert ladder_l5_stock.ladder_l5_stock_report(config)["branches"] == report["branches"] == {"the_dose": ESCALATE, "what_follows": FURTHER_SEEDS}
    # A STEP NEEDS THE ONES BEFORE IT: at a further seed (the escalation recorded), with three of `stock`'s four trainings made.
    config["ladder_loop"]["l5_stock"]["escalated"] = True
    monkeypatch.setenv(SEEDS, "4")
    _stage5s(config, monkeypatch, until="ladder_l5_stock_train_stock_3")
    with pytest.raises(RuntimeError, match="the averaged adapter `stock` needs the step ladder_l5_stock_train_stock_4 of this run, which is not done"):
        _on_the_gpu(monkeypatch, ladder_l5_stock.STEPS["ladder_l5_stock_average_stock"], config)
    with pytest.raises(RuntimeError, match="the measurement of L5s's model `stock` needs the step ladder_l5_stock_average_stock of this run, which is not done"):
        ladder_l5_stock.STEPS["ladder_l5_stock_measure_stock"](config)
    with pytest.raises(RuntimeError, match="L5s's report needs the step ladder_l5_stock_train_stock_4 of this run, which is not done"):
        ladder_l5_stock.ladder_l5_stock_report(config)
    assert stock.store().root.name == "ladder_l5_stock_pre_r64_seed4" and stock.store().done_summary("ladder_l5_stock_prepare")["the_seeds"]["fixed"] == [3, 4, 5]


# ----------------------------------------------------------------------------------------------- the refusals
def test_the_first_step_refuses_before_anything_is_written(stock, monkeypatch):  # noqa: F811
    config, runs = stock.config, stock.runs
    nothing_written = lambda: not [path for path in runs.iterdir() if path.name.startswith("ladder_l5_stock") and any(path.iterdir())]      # noqa: E731
    # A SEED THE SETTING DOES NOT LIST; and a further seed while the escalation is not recorded.
    monkeypatch.setenv(SEEDS, "0")
    with pytest.raises(RuntimeError, match=r"seed 0 is none of this stage's seeds \[3\] \(nor of \[4, 5\], which wait for the escalation\).*\(ladder_loop.l5_stock\). Nothing was written"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    monkeypatch.setenv(SEEDS, "4")
    with pytest.raises(RuntimeError, match=r"seed 4 is one of the further seeds \[4, 5\], which run once the scout's read at \[3\] names the escalation and the setting records it "
                                           r"\(escalated is false\).*Nothing was written"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    monkeypatch.setenv(SEEDS, "3")
    # THE SETTING ITSELF, by its one reader: seeds that are not L5d's, an escalation that is no true or false, and a recipe of L5d's that is not the averaged one.
    settings, of_l5d = config["ladder_loop"]["l5_stock"], config["ladder_loop"]["l5_draws"]
    for changed, said in (({"seeds": [9]}, r"ladder_loop.l5_stock.seeds is \[9\] and more_seeds \[4, 5\]: distinct whole numbers of at least 0, at least one in seeds, each a seed of L5d's"),
                          ({"more_seeds": [3, 4]}, r"ladder_loop.l5_stock.seeds is \[3\] and more_seeds \[3, 4\]"), ({"seeds": []}, r"ladder_loop.l5_stock.seeds is \[\]"),
                          ({"escalated": "yes"}, "ladder_loop.l5_stock.escalated is 'yes': true once the scout's read names the escalation, false until then")):
        kept = dict(settings)
        settings.update(changed)
        with pytest.raises(ValueError, match=said):
            ladder_l5_stock.ladder_l5_stock_prepare(config)
        settings.clear()
        settings.update(kept)
    for recipe in ("constant", "decay"):
        of_l5d["recipe"] = recipe
        with pytest.raises(ValueError, match=f"ladder_loop.l5_draws.recipe is `{recipe}`: this stage trains each set in four orders and measures the averaged adapter .`mean_constant`."):
            ladder_l5_stock.ladder_l5_stock_prepare(config)
    of_l5d["recipe"] = "mean_constant"
    # THE SMOKE-ONLY VARIABLE without a run directory of the task's naming; and with one, on a map that is no fixture's (this world's is hand-made): refused by L5d's own reading.
    monkeypatch.setenv(SMOKE, "1")
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_STOCK_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_STOCK_RUN. and on the "
                                           "fixtures' base map. A run in the seed's own directory reads a run of L5 whose checks passed. Nothing was written"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    monkeypatch.delenv(SMOKE)
    # L5's RUN: not finished (its report's marker is gone), and not to be read: refused by L5's own reading, in this stage's words.
    marker = stock.l5 / "ladder_l5_report.done.json"
    kept = marker.read_text()
    marker.unlink()
    with pytest.raises(RuntimeError, match=r"ladder_l5_deep_pre_r64_seed0 does not hold \['ladder_l5_report.done.json'\]: the FINISHED run of the deep search. L5s runs no search: its sets "
                                           "are every own solve in stock, from the finished search of seed 0 and the start model's map.*nothing was written"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    marker.write_text(kept)
    report_file = stock.l5 / "report_ladder_l5.json"
    kept = report_file.read_text()
    report_file.write_text(json.dumps({**json.loads(kept), "inconclusive": True, "failed_checks": ["`shallow`: the_training_ran"]}))
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .*: no model here stands beside such a run"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    report_file.write_text(kept)
    assert nothing_written()


def test_a_run_whose_bar_asks_for_more_published_rows_than_the_file_holds_is_refused(deep, monkeypatch, tmp_path):  # noqa: F811
    # FOUR deep solves and the rule's four ordinary solves: N is eight, and `published_stock` would be sixteen rows of the fixture's twelve.
    for name in (RUN, SMOKE, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    config["training"]["effective_batch"] = 1
    _round_one(deep, monkeypatch, tmp_path, solved=tuple(DRAWN), l4p=False, settings={"problems": 4, "held_aside": 1, "minimum": 2})
    monkeypatch.setenv(SEEDS, "3")
    with pytest.raises(RuntimeError, match=r"\(seed 3; L5's run ladder_l5_deep_pre_r64_seed0\). Nothing was written"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    runs = ladder_l5._runs(config)
    assert not [path for path in runs.iterdir() if path.name.startswith("ladder_l5_stock") and any(path.iterdir())]


def test_the_steps_run_through_the_gpu_entry_point(stock, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(milestone2, "load_config", lambda path: stock.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    # In the stand-in's world no adapter is trained and none is averaged, so the report says INCONCLUSIVE and reads no dose (L5d's run of the seed is not there either).
    assert json.loads((tmp_path / "ladder_l5_stock_average_stock.json").read_text())["averaged"] is False
    report = json.loads((tmp_path / "ladder_l5_stock_report.json").read_text())
    assert stock.store().is_done(ladder_l5_stock.REPORT) and report["inconclusive"] is True and report["stand_in_engine"] is True and report["branches"] == {"the_dose": INCONCLUSIVE, "what_follows": NOT_READ}


# ---------------------------------------------------------------------------------------- the smoke-only path
def test_the_smoke_stage_goes_on_from_l5s_inconclusive_smoke_run_takes_its_ordinary_solves_by_the_runs_minimum_and_says_so(deep, monkeypatch):  # noqa: F811
    for name in (RUN, SMOKE, ladder_l5_draws.RUN_VARIABLE, ladder_l5_draws.SMOKE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    first = _smoke_world(deep, monkeypatch)
    config, runs = deep.config, ladder_l5._runs(deep.config)
    of_l5, failed = runs / "ladder_l5_deep_smoke", first["ladder_l5_report"]["failed_checks"]
    calls, measured = [], []
    _trainings(monkeypatch, calls, value=lambda number: 1.0)          # AS ON THE BOX: one optimizer step on the same rows, whatever their order, is ONE training: adapters alike
    solver = NoiseSolver(deep.solver, (4, 8, 12, *SUCCESSES))         # L5d's smoke run's three models, then this stage's two
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (measured.append(given["vllm"]["max_lora_rank"]), (solver, stand_in_parameters))[1]))
    monkeypatch.setenv(SEEDS, "3")
    nothing_written = lambda: not [path for path in runs.iterdir() if path.name.startswith("ladder_l5_stock") and any(path.iterdir())]      # noqa: E731
    # ---- A REAL RUN IS REFUSED FOR SUCH A RUN OF L5 (its report is INCONCLUSIVE), in the seed's own directory and in one the task names.
    assert failed != []
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no model here stands beside such a run"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    monkeypatch.setenv(RUN, "ladder_l5_stock_smoke")
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    # THE SMOKE-ONLY VARIABLE on a map that is no fixture's: refused, and nothing is written.
    monkeypatch.setenv(SMOKE, "1")
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match="RLVR_LEAN_LADDER_L5_STOCK_SMOKE is set: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_STOCK_RUN. and on the "
                                           "fixtures' base map. A run on the pool's problems reads a run of L5 whose checks passed. Nothing was written"):
        ladder_l5_stock.ladder_l5_stock_prepare(config)
    marker.write_text(json.dumps(recorded))
    assert nothing_written()
    monkeypatch.delenv(RUN)
    monkeypatch.delenv(SMOKE)
    # ---- L5d's SMOKE RUN under the averaged recipe, by L5d's own smoke stage: the run this stage's dose is read against.
    _l5ns_report(runs, AVERAGE_OF_FOUR)
    monkeypatch.setenv(ladder_l5_draws.RUN_VARIABLE, "ladder_l5_draws_smoke")
    monkeypatch.setenv(ladder_l5_draws.SMOKE_VARIABLE, "1")
    of_l5d = _stage5d(config, monkeypatch, stage_name="ladder_l5_draws_smoke")
    theirs = runs / "ladder_l5_draws_smoke_mean_constant"
    assert of_l5d["ladder_l5_draws_report"]["inconclusive"] is False and sorted(name for name in _all(theirs) if not name.startswith("episodes_")) == l5d_files(MEAN_CONSTANT)
    monkeypatch.delenv(ladder_l5_draws.SMOKE_VARIABLE)                  # THIS stage's smoke task names L5d's RUN, and neither L5d's smoke-only variable nor L5's
    options = entry.step_fields(entry.STAGES["ladder_l5_stock_smoke"][2])[2]["environment"]
    assert options[ladder_l5_draws.RUN_VARIABLE] == "ladder_l5_draws_smoke" and ladder_l5_draws.SMOKE_VARIABLE not in options and ladder_l5.SMOKE_VARIABLE not in options
    monkeypatch.setenv(RUN, options[RUN])
    monkeypatch.setenv(SMOKE, options[SMOKE])
    # ---- THE SMOKE RUN: its own directory, the fixtures' map, the variable. L5's run is read all the same, and that is RECORDED.
    read_only = {path: _all(path) for path in (of_l5, theirs, deep.map, deep.l4t2, deep.rank)}
    del calls[:]
    summaries = _stage5s(config, monkeypatch, stage_name="ladder_l5_stock_smoke")
    store = ladder_l5_stock._store(config, ladder_l5_stock.the_settings(config))
    assert store.root.name == "ladder_l5_stock_smoke" and list(summaries) == STEPS and {path: _all(path) for path in read_only} == read_only
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    read = summaries["ladder_l5_stock_prepare"]
    smoke, of_the_stock = read["smoke"], read["the_stock"]
    assert (smoke["problems_at_least"], smoke["l5_read_though_inconclusive"], smoke["l5s_failed_checks"]) == (1, True, failed) and read["seed"] == 3
    assert (read["the_search"]["run"], read["the_search"]["deep_solves"], read["the_search"]["inconclusive"]) == (str(of_l5), 1, True)
    # ONE deep solve, and ONE ordinary solve: the rule's own if the map left it one, by the run's minimum otherwise, and MARKED either way. N = 2: two sets of FOUR rows.
    ordinary = store.read_rows("l5_stock_ordinary_solves.jsonl")
    by_the_minimum = sum(row["taken_by"] != "the rule" for row in ordinary)
    assert (of_the_stock["deep_solves"], of_the_stock["ordinary_solves"], of_the_stock["own_rows"], of_the_stock["rows"]) == (1, 1, 2, {"stock": 4, "published_stock": 4}) and len(ordinary) == 1
    assert of_the_stock["ordinary_solves_of_the_rule"] + by_the_minimum == 1 and smoke["ordinary_solves_taken_by_the_runs_minimum"] == of_the_stock["ordinary_solves_taken_by_the_runs_minimum"] == by_the_minimum
    assert {model: read["the_sets"][model]["rows_by_origin"] for model in MODELS} == {STOCK: {"deep": 1, "shallow": 1, "pretraining": 2}, PUBLISHED_STOCK: {"pretraining": 4}}
    # EIGHT trainings of four optimizer steps (the batch is one row here), two averaged adapters that ARE each of their four (kept, served and SAID), two measurements at rank 64.
    assert len(calls) == 8 and [call["seed"] for call in calls] == [3] * 8 and solver.measured[-2:] == [(GOAL, 32, 1001, 1024)] * 2 and measured[-2:] == [64] * 2
    for model in MODELS:
        made = summaries[f"ladder_l5_stock_average_{model}"]
        assert (made["read_back"]["is_the_mean"], made["read_back"]["is_none_of_them"], made["one_of_its_models_in_a_smoke_run"]) == (True, False, True)
    # THE REPORT says A SMOKE RUN; the checks pass (the rule made again takes the ordinary solve the run's minimum took as the first step recorded it); and THE DOSE IS READ
    # against L5d's SMOKE run, which L5d's own function finds by L5d's run variable.
    report = summaries["ladder_l5_stock_report"]
    assert list(report) == REPORT_KEYS and report["ok"] is True and report["inconclusive"] is False and report["smoke"] == smoke
    assert any(f"A SMOKE RUN: L5's smoke run reads INCONCLUSIVE ({', '.join(failed)}) and was read all the same; one deep solve is enough; {by_the_minimum} ordinary solves were taken by the "
               "run's minimum" in line for line in report["lines"])
    assert report["l5d_at_this_seed"]["read"] is True and report["l5d_at_this_seed"]["run"] == str(theirs) and report["the_dose"]["name"] == ESCALATE and report["the_dose"]["difference"] == 0.0
    assert report["l5d_at_this_seed"]["figures"]["deep"]["successes"] == 12 == report["figures"][STOCK]["all_of_g"]["successes"]
    # WITHOUT L5d's SMOKE RUN the dose is NOT READ, and nothing is refused.
    (theirs / "ladder_l5_draws_report.done.json").rename(theirs / "kept.done.json")
    assert ladder_l5_stock.ladder_l5_stock_report(config)["branches"] == {"the_dose": NOT_READ, "what_follows": NOT_READ}
    (theirs / "kept.done.json").rename(theirs / "ladder_l5_draws_report.done.json")
    # WITHOUT THE VARIABLE nothing is taken loosely.
    monkeypatch.delenv(SMOKE)
    assert ladder_l5_stock.smoke_run() is False


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_stock"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l5_stock.STEPS) == STEPS and ladder_l5_stock.ENTRY_STAGE == "ladder_l5_stock" and len(STEPS) == 14
    # `stock` is trained four times, averaged and measured BEFORE `published_stock`; then the report.
    assert STEPS[1:7] == [*(f"ladder_l5_stock_train_stock_{order}" for order in (1, 2, 3, 4)), "ladder_l5_stock_average_stock", "ladder_l5_stock_measure_stock"]
    # THE SMOKE STAGE: the same steps, under the runs L5d's smoke stage names (WITHOUT its smoke-only variable, and without L5's), a run of its own, and ITS smoke-only variable.
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_stock_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    of_l5d = entry.step_fields(entry.STAGES["ladder_l5_draws_smoke"][2])[2]["environment"]
    assert smoke[0][2]["environment"] == {**{name: value for name, value in of_l5d.items() if name != ladder_l5_draws.SMOKE_VARIABLE}, RUN: "ladder_l5_stock_smoke", SMOKE: "1"}
    assert smoke[0][2]["environment"][ladder_l5.RUN_VARIABLE] == "ladder_l5_deep_smoke" and ladder_l5.SMOKE_VARIABLE not in smoke[0][2]["environment"]
    # THE SETTING, as the package's own config has it: the scout's seed, two further ones that wait for the escalation; every seed one of L5d's; L5d's recipe the averaged one.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    assert whole["ladder_loop"]["l5_stock"] == {"seeds": [3], "more_seeds": [4, 5], "escalated": False} == CONFIG["ladder_loop"]["l5_stock"]
    own = ladder_l5_stock.the_settings(whole)
    assert (own.start, own.seeds, own.more, own.escalated, own.draws, own.deep) == ("pre_r64", (3,), (4, 5), False, ladder_l5_draws.the_settings(whole), ladder_l5.the_settings(whole))
    assert set((*own.seeds, *own.more)) <= set(own.draws.seeds) and own.draws.recipe == "mean_constant" and (own.draws.sampling, own.draws.attempts) == ("reach", 32)
    assert [ladder_l5_stock.run_name(own, seed) for seed in (3, 5)] == ["ladder_l5_stock_pre_r64_seed3", "ladder_l5_stock_pre_r64_seed5"]
    assert (ladder_l5_stock.average_marker("stock"), ladder_l5_stock.REPORT_FILE, ladder_l5_stock.ORDINARY_FILE) == ("ladder_l5_stock_average_stock", "report_ladder_l5_stock.json", "l5_stock_ordinary_solves.jsonl")
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l5_stock"]' in path.read_text()] == ["gpu/ladder_l5_stock.py"]
    others = set(ladder_l5.STEPS) | set(ladder_l6.STEPS) | set(ladder_l4_rows.STEPS) | set(ladder_l5_noise.STEPS) | set(ladder_l5_draws.STEPS)
    assert not set(ladder_l5_stock.STEPS) & others and all(name.startswith("ladder_l5_stock_") for name in ladder_l5_stock.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l5_stock.STEPS for step in entry.STAGES[name])) == ["ladder_l5_stock", "ladder_l5_stock_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(RUN) == names(SMOKE) == ["ladder_l5_stock_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L5_STOCK_SMOKE" and RUN == "RLVR_LEAN_LADDER_L5_STOCK_RUN"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l5_stock.py", "runner/entry.py"]
    # ONE BODY EACH: its trainings are L4t's own training body, its measurements the ceiling's own measuring body with no rung, its averaged adapters L5n's own function, and
    # L5's run is read, the deep solves and that seed's three sets drawn, and the ordinary solves drawn, by L5d's and L5's own functions.
    assert ladder_l5_stock.train_a_model is ladder_l4_rows.train_a_model and ladder_l5_stock.measure_model is ladder_ceiling.measure_model and ladder_l5_stock.the_set is ladder_l5_draws.the_set
    own_stage = ladder_l5_stock.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == (
        "ladder_l5_stock", "ladder_l5_stock", "l5_stock", "L5s", "ladder_l5_stock_prepare", "ladder_loop.l5.start")
    source = (PACKAGE / "gpu" / "ladder_l5_stock.py").read_text()
    for called in ("ladder_l5_draws.drawn_by_the_seed(", "ladder_l5.ordinary_by_the_seed(", "ladder_l5_noise.the_mean_adapter(", "ladder_l5_draws.own_run(", "rungs=False"):
        assert called in source, called
    assert "_episodes(" not in source and "LoRARequest" not in source and "mean_of_adapters" not in source and "read_the_search(" not in source and "solve_rows(" not in source
    report_source = (PACKAGE / "reporting" / "ladder_l5_stock.py").read_text()
    assert "checks_of_the_models(" in report_source and "lines_of_the_checks(" in report_source
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l5_stock.STEPS" in main and "ladder_l5_stock," in main
