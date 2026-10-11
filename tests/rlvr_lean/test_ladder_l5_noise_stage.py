"""L5n, the stage end to end on what L5's FINISHED run and the runs that task read stored: the refusals of its first
step; the fixed rows and each order seed's order of them; each order trained with the recipe as it is and with the decay,
by L4t's own training body, its learning rate recorded and held to its schedule; the two averaged adapters, written, read
back and held to be the mean of their models; every model measured on the goal set's first sampling ALONE; the report;
L5's `published` at its other seeds, read when those runs are there; the smoke stage and its smoke-only path, which no
real run can take; the registration; and L5 and L6 left writing what they wrote. Spec:
docs/spec/ladder-loop.spec.md, "L5n: what is one training worth? The spread between trainings, and a recipe that
lowers it". The engine and Lean are scripted; nothing touches a GPU or the network. The rules and the report on hand-made
rows are `test_ladder_l5_noise.py`; the mean of adapters on made-up tensors is `test_adapter_files.py`."""

import json
import re
import shutil
import statistics
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adapter_files import _numbers, _of, _write  # noqa: E402 - an adapter's weights file written and read by hand
from test_ladder_l4_rank_stage import MODULES  # noqa: E402
from test_ladder_l4_rows_stage import _all  # noqa: E402
from test_ladder_l4_stage import CHANGED, _on_the_gpu  # noqa: E402
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
from test_ladder_l6_stage import DRAWN, _round_one, _smoke_world, _stage6, cycle  # noqa: E402, F401 - L5's stage run to its end on a base map of twelve problems; the smoke runs' world; L6's own
from test_ladder_l6_stage import MODELS as L6_MODELS  # noqa: E402
from test_ladder_l6_stage import OWN_FILES as L6_FILES  # noqa: E402 - what L6 leaves, frozen there
from test_ladder_l6_stage import STEPS as L6_STEPS  # noqa: E402

from rlvr_lean.domain.ladder_round import l5_noise  # noqa: E402
from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.dose import learning_rates, run_dose  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.ladder_round.l5_noise import (  # noqa: E402
    AT_LEAST_THE_MEAN,
    AVERAGED,
    DECAY_LOWERS,
    MODELS,
    NAMED_NOTHING,
    ORDER_ALONE,
    ORDERS,
    RECIPES,
    ROWS_DRAWN,
    TRAINED,
    WITH_THE_DECAY,
    figure,
    mean_of,
    model_of,
    models_of,
)
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_dose, ladder_l4_rank, ladder_l4_rows, ladder_l5, ladder_l5_noise, ladder_l6, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import StandInOutput, StandInSample, stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure import adapter_files  # noqa: E402
from rlvr_lean.infrastructure.adapter_files import compared, is_the_mean  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY  # noqa: E402

RUN, SMOKE = ladder_l5_noise.RUN_VARIABLE, ladder_l5_noise.SMOKE_VARIABLE
STEPS = ["ladder_l5_noise_prepare", *(f"ladder_l5_noise_{step}_{recipe}_{order}" for order in ORDERS for recipe in RECIPES for step in ("train", "measure")),
         *(step for recipe in RECIPES for step in (f"ladder_l5_noise_average_{recipe}", f"ladder_l5_noise_measure_mean_{recipe}")), "ladder_l5_noise_report"]
SMOKE_STEPS = ["ladder_l5_noise_prepare", *(f"ladder_l5_noise_{step}_{recipe}_{order}" for order in (1, 2) for recipe in RECIPES for step in ("train", "measure")), *STEPS[-5:]]
TRAININGS = tuple(f"ladder_l5_noise_train_{model}" for model in TRAINED)
AVERAGES = tuple(f"ladder_l5_noise_average_{recipe}" for recipe in RECIPES)
ATTEMPTS = CONFIG["ladder_loop"]["l5_noise"]["attempts"]                    # 32: the goal set's first sampling; in this world no other set is sampled with as many once L5 has run
RATE, WARMUP = CONFIG["training"]["learning_rate"], CONFIG["training"]["warmup_steps"]
GOAL = "fixture_h3"                                                         # the scripted world's one goal problem
# What each model writes on the goal problem, in the order the models are measured: it closes the goal in its first so many of the 32 attempts. The recipe as it is: 4, 8, 12, 16;
# with the decay half as far apart and level with it: 8, 10, 12, 14; each averaged model near its models' mean.
SUCCESSES = dict(zip(MODELS, (4, 8, 8, 10, 12, 12, 16, 14, 10, 11)))
OF_L5 = 12                                                                  # L5's own `published` of seed 0 there (the world's scripted solver)
OWN_FILES = sorted([
    *(f"adapters/{model}/{name}" for model in MODELS for name in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l5_noise_loss_{model}.json" for model in TRAINED), *(f"l5_noise_training_{model}.jsonl" for model in TRAINED), "l5_noise_rows.jsonl", "l5_noise_stored_published_seed0.jsonl",
    *(f"ladder_l5_noise_train_{model}.done.json" for model in TRAINED), *(f"ladder_l5_noise_measure_{model}.done.json" for model in MODELS),
    *(f"ladder_l5_noise_average_{recipe}.done.json" for recipe in RECIPES), "ladder_l5_noise_prepare.done.json", "ladder_l5_noise_report.done.json", "problems.jsonl",
    "report_ladder_l5_noise.json"])
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "rank", "models", "orders", "recipes", "branches", "inconclusive",
               "failed_checks", "can_this_run_be_read", "figures", "the_spread_of_the_recipe_as_it_is", "does_decay_lower_it", "and_its_level", "averaging", "what_follows", "beside",
               "the_rows", "the_orders", "the_sampling", "training_sets", "recipe", "models_made", "smoke", "what_it_cannot_say", "attempts", "not_to_be_read", "not_measured", "adapters"]


class NoiseSolver:
    """`other` (the world's scripted solver) for every sampling but the goal set's first (32 samples a prompt); for that
    one, the k-th measurement closes the goal in its first `successes[k]` attempts, so that the models differ.
    `measured`: every prompt of such a measurement, with what the engine was asked for."""

    def __init__(self, other, successes):
        self.other, self.successes, self.measured = other, list(successes), []

    def generate(self, prompts, parameters, lora_request=None):
        if parameters.n != ATTEMPTS:
            return self.other.generate(prompts, parameters, lora_request)
        outputs = []
        for prompt in prompts:
            closes = self.successes[len(self.measured) % len(self.successes)]
            self.measured.append((re.findall(r"theorem (\S+)", prompt)[-1], parameters.n, parameters.seed, parameters.max_tokens))
            outputs.append(StandInOutput([StandInSample("  done\n" if index < closes else "  bad_step\n  done\n", [0] * 4, "stop") for index in range(parameters.n)]))
        return outputs


def _adapter_of(directory, rank, alpha, value, modules_first="q_proj"):
    """An adapter's directory as the library saves one, every number of an A matrix `value` and of a B matrix half of it;
    its config names the target modules in one order or the other (a set is saved in no fixed order)."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors = []
    for name, inputs, outputs in MODULES:
        tensors += [(f"base_model.model.layers.0.{name}.lora_A.weight", "F32", (rank, inputs), _of("F32", [[value] * inputs] * rank)),
                    (f"base_model.model.layers.0.{name}.lora_B.weight", "F32", (outputs, rank), _of("F32", [[value / 2] * rank] * outputs))]
    _write(directory / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    modules = sorted((name for name, _, _ in MODULES), key=lambda name: name != modules_first)
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": alpha, "peft_type": "LORA", "target_modules": modules}))


def _trainings(monkeypatch, calls, rates=None, value=None):
    """The training's GPU path with the model taken out: `_train_with_readings` is the schedule alone (stand-in losses).
    AS THE REAL ONE DOES, it attaches the adapter the config it is HANDED asks for, saves an adapter of that rank (the
    k-th training's numbers are k and k / 2: no two alike), records what of it is not the start adapter's and, ASKED
    FOR ITS LEARNING RATES, each step's: the schedule's with the decay when the decay is asked for, the recipe's own
    otherwise. `rates`: given what it was asked, the rates to record in their place (a training that was not made with
    its schedule). `value`: given the call's number, the number its adapter holds in the place of that number."""
    def train_with_readings(config, examples, sample, pairs, schedule, seed, directory, tensorboard_run, orders=None, per_example=False, positions=False, **more):
        calls.append({"lora": dict(config["lora"]), "max_lora_rank": config["vllm"]["max_lora_rank"], "examples": examples, "schedule": schedule, "seed": seed, "directory": directory,
                      "tensorboard_run": tensorboard_run, "orders": orders, "more": more,
                      "training": {key: config["training"][key] for key in ("learning_rate", "effective_batch", "max_sequence_tokens", "warmup_steps")}})
        train_step, read, _ = ladder_dose._stand_in_calls(len(examples), [], [])
        number = len(calls)
        holds = float(number) if value is None else value(number)
        result = run_dose(len(examples), config["training"]["effective_batch"], schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"], train_step, read,
                          lambda name, step_number: _adapter_of(directory / name, config["lora"]["rank"], config["lora"]["alpha"], holds, "q_proj" if (number // 2) % 2 else "down_proj"),
                          orders=orders, per_example=per_example, positions=positions)
        recorded = None
        if more.get("learning_rates"):
            recorded = learning_rates(config["training"]["learning_rate"], config["training"]["warmup_steps"], result["steps"], bool(more.get("decay")))
            recorded = recorded if rates is None else rates(more, recorded)
        return {**result, "target_format": "native", "seconds": 1.0, "peak_allocated_gb": 12.3,
                **({"against_the_start_adapter": {name: dict(CHANGED) for name in schedule["checkpoint_steps"]}} if "start" in more else {}),
                **({"learning_rates": recorded} if recorded is not None else {})}

    monkeypatch.setattr(ladder_dose, "_train_with_readings", train_with_readings)
    monkeypatch.setattr(pipeline, "_cap_torch_memory", lambda config: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0, max_memory_reserved=lambda: 12_640_000_000)))


def _stage5n(config, monkeypatch, until=None, stage_name="ladder_l5_noise", only=None):
    """The stage's steps in its order; the trainings and the averagings by their GPU path (the stand-in makes no adapter)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l5_noise.STEPS or (only and step not in only):
            continue
        run = ladder_l5_noise.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in (*TRAININGS, *AVERAGES) else run(config)
        if step == until:
            break
    return summaries


@pytest.fixture
def noise(deep, monkeypatch, tmp_path):  # noqa: F811
    """`deep`'s world with what this stage reads on the box: a base map of twelve problems with a hand-made map of
    `pre_r64`, and L5's own stage RUN TO ITS END there at its search's seed (four deep solves: `published` is trained on
    EIGHT rows of the fixture's twelve). The batch is ONE row, in L5's run and here, so that a training of eight rows
    is eight optimizer steps: three of them after the five warm-up steps. Then this stage's task, which carries NO
    variable. `solver.measured`: every prompt of a measurement of the goal set's first sampling made from here on."""
    for name in (RUN, SMOKE):
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    config["training"]["effective_batch"] = 1
    first = _round_one(deep, monkeypatch, tmp_path, solved=tuple(DRAWN), l4p=False)
    root = deep.store().root
    assert root.name == "ladder_l5_deep_pre_r64_seed0" and first["ladder_l5_report"]["inconclusive"] is False and first["ladder_l5_report"]["ok"] is True
    assert (first["ladder_l5_prepare"]["rows_of_each_kind"], first["ladder_l5_train_published"]["rows"], first["ladder_l5_train_published"]["steps"]) == (4, 8, 8)
    calls, kits = [], []
    _trainings(monkeypatch, calls)
    solver = NoiseSolver(deep.solver, SUCCESSES.values())
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"]}),
        (solver, stand_in_parameters))[1]))
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l5_noise"])        # THE STAGE names no variable
    return SimpleNamespace(config=config, deep=deep, first=first, l5=root, calls=calls, kits=kits, solver=solver, asked=len(deep.solver.asked), runs=ladder_l5._runs(config),
                           store=lambda: ladder_l5_noise._store(config, ladder_l5_noise.the_settings(config)))


def figure_sd(successes):
    """The standard deviation (n - 1) of so many successes of 32 on the one goal problem, per attempt."""
    return statistics.stdev([count / 32 for count in successes])


def _another_seed_of_l5(noise, seed, successes):  # noqa: F811
    """A FINISHED run of L5 at another seed, hand-made (it is only read): the run of the search's seed, copied, with its
    `published` closing the goal problem in `successes` of the first sampling's attempts. Returns its directory."""
    directory = noise.l5.with_name(noise.l5.name.replace("seed0", f"seed{seed}"))
    shutil.copytree(noise.l5, directory)
    path = directory / "episodes_l4_reach_published_problems.jsonl"
    path.write_text("".join(json.dumps({**row, "resolved": successes, "resolved_by_statement": successes}) + "\n" for row in _file_rows(path)))
    return directory


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_trains_the_fixed_rows_in_four_orders_with_each_recipe_averages_each_recipes_adapters_measures_g_alone_and_reports(noise, monkeypatch):  # noqa: F811
    config, calls, kits, solver, deep_ = noise.config, noise.calls, noise.kits, noise.solver, noise.deep
    read_only = {path: _all(path) for path in (noise.l5, deep_.map, deep_.l4t2, deep_.rank, deep_.l4t, deep_.arm, deep_.pre)}
    summaries = _stage5n(config, monkeypatch)
    store = noise.store()
    assert store.root.name == "ladder_l5_noise_pre_r64_seed0" and store.root.parent == noise.l5.parent and list(summaries) == STEPS == list(ladder_l5_noise.STEPS)
    assert {path: _all(path) for path in read_only} == read_only               # L5's RUN AND EVERY RUN IT READ WERE ONLY READ: file for file
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert sorted(path.name for path in noise.runs.iterdir() if path.name.startswith(("ladder_l5", "ladder_l6"))) == ["ladder_l5_deep_pre_r64_seed0", "ladder_l5_noise_pre_r64_seed0"]

    # ---- the first step: THE FIXED ROWS are L5's stored set of `published`, by id and place, never with text; each order seed's order of them; the one sampling
    read, of_l5 = summaries["ladder_l5_noise_prepare"], _file_rows(noise.l5 / "l5_training_published.jsonl")
    assert (read["stage"], read["label"], read["check"], read["seed"], read["start"], read["models"], read["orders"], read["recipes"]) == (
        "l4", "pretrained on published proofs", "L5n", 0, "pre_r64", list(MODELS), [1, 2, 3, 4], ["constant", "decay"])
    assert (read["trained"], read["averaged"], read["start_adapter"], read["start_run"]) == (list(TRAINED), list(AVERAGED), str(deep_.rank / "adapters" / "pre_r64"), str(deep_.rank))
    fixed = store.read_rows("l5_noise_rows.jsonl")
    placed = lambda row: (row["id"], row["problem_id"], row["row_of_the_file"])      # noqa: E731
    assert fixed == of_l5 and len(fixed) == 8 and not any(field in row for row in fixed for field in TEXT_FIELDS) and {row["origin"] for row in fixed} == {"pretraining"}
    of_the_rows = read["the_rows"]
    assert (of_the_rows["model"], of_the_rows["run"], of_the_rows["seed"], of_the_rows["rows"], of_the_rows["rows_by_origin"], of_the_rows["optimizer_steps"]) == (
        "published", str(noise.l5), 0, 8, {"pretraining": 8}, 8)
    assert of_the_rows["its_training"]["rows"] == 8 and of_the_rows["barred"]["held_out"] == [] and of_the_rows["barred"]["published_rows_not_of_the_pretrain_half"] == []
    assert read["the_orders"] == {"what": read["the_orders"]["what"], "order_seeds": [1, 2, 3, 4], "the_order_trained_before": [], "alike": []} and read["smoke"] is None
    sets = {model: store.read_rows(f"l5_noise_training_{model}.jsonl") for model in TRAINED}
    for order in ORDERS:
        own = sets[f"constant_{order}"]
        # EACH ORDER IS ITS ORDER SEED's: the content hash of THAT seed and each row's id; the same eight rows by id and place; not the order L5 trained them in.
        assert [placed(row) for row in own] == [placed(row) for row in training_order(fixed, order)] != [placed(row) for row in fixed] and [row["row"] for row in own] == list(range(8))
        assert sets[f"decay_{order}"] == own and sorted(map(placed, own)) == sorted(map(placed, fixed)) and not any(field in row for row in own for field in TEXT_FIELDS)
        for recipe in RECIPES:
            entry_ = read["training_sets"][f"{recipe}_{order}"]
            assert (entry_["order_seed"], entry_["recipe"], entry_["rows"], entry_["optimizer_steps"], entry_["rows_by_origin"]) == (order, recipe, 8, 8, {"pretraining": 8})
    assert len({tuple(row["id"] for row in sets[f"constant_{order}"]) for order in ORDERS}) == 4
    assert read["the_averaged"] == {f"mean_{recipe}": {"what": read["the_averaged"][f"mean_{recipe}"]["what"], "of": [f"{recipe}_{order}" for order in ORDERS]} for recipe in RECIPES}
    sampling = read["the_sampling"]
    assert (sampling["name"], sampling["episodes"], sampling["sampling_seed"]) == ("reach", 32, 1001) == tuple(noise.first["ladder_l5_search_prepare"]["goal_samplings"][0].values())
    assert read["goal_samplings"] == [{"name": "reach", "episodes": 32, "sampling_seed": 1001}] and (read["goal_set"], read["attempts_a_goal_problem"]) == (1, 32)
    problems = store.read_rows("problems.jsonl")
    assert [row["set"] for row in problems] == [f"l4_reach_{model}" for model in MODELS] and {row["problem_id"] for row in problems} == {GOAL}       # NO SET OF A RUNG, OF A SECOND SAMPLING
    theirs = noise.first["ladder_l5_search_prepare"]
    assert {key: read["recipe"][key] for key in ("learning_rate", "warmup_steps", "effective_batch", "lora")} == {key: theirs["recipe"][key] for key in ("learning_rate", "warmup_steps", "effective_batch", "lora")}
    assert read["recipe"]["learning_rate_factor"]["constant"] == {"1": 0.2, "4": 0.8, "5": 1.0, "6": 1.0, "7": 1.0, "8": 1.0}
    assert read["recipe"]["learning_rate_factor"]["decay"] == {"1": 0.2, "4": 0.8, "5": 1.0, "6": 0.75, "7": 0.5, "8": 0.25} and read["start_recipe"] == theirs["start_recipe"]
    # L5's own `published` on that sampling, copied as stored: one more draw of `constant`.
    assert store.read_rows("l5_noise_stored_published_seed0.jsonl") == _file_rows(noise.l5 / "episodes_l4_reach_published_problems.jsonl")
    of_published = read["the_published_of_l5"]
    assert (of_published["seed"], of_published["run"], of_published["set"], of_published["file"], of_published["inconclusive"], of_published["seeds_read_by_the_report"]) == (
        0, str(noise.l5), "l4_reach_published", "l5_noise_stored_published_seed0.jsonl", False, [0, 1, 2])

    # ---- the eight trainings, order by order: `constant_k` THE RECIPE AS IT IS (nothing of the decay is asked), then `decay_k`; each from `pre_r64` at rank 64 on ITS order
    start = deep_.rank / "adapters" / "pre_r64"
    as_it_is, with_decay = {"start": start, "learning_rates": True}, {"start": start, "learning_rates": True, "decay": True}       # beside the start adapter L4t's body always hands over
    assert [call["more"] for call in calls] == [as_it_is, with_decay] * 4 and [(call["lora"]["rank"], call["lora"]["alpha"]) for call in calls] == [(64, 128)] * 8
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l5_noise_pre_r64_seed0_{model}" for model in TRAINED]
    assert all(call["training"] == {"learning_rate": RATE, "effective_batch": 1, "max_sequence_tokens": CONFIG["training"]["max_sequence_tokens"], "warmup_steps": WARMUP} for call in calls)
    for call, model in zip(calls, TRAINED):
        assert [example["problem_id"] for example in call["examples"]] == [row["problem_id"] for row in sets[model]] and all(example["theorem"] and example["completion"] for example in call["examples"])
        assert call["orders"] == [list(range(8))] and call["schedule"]["checkpoint_steps"] == {model: 8}
    constant, decay = learning_rates(RATE, WARMUP, 8, False), learning_rates(RATE, WARMUP, 8, True)
    assert decay[:5] == constant[:5] and decay[5:] == [RATE * 0.75, RATE * 0.5, RATE * 0.25] and constant[5:] == [RATE] * 3
    for order in ORDERS:
        for recipe, rates in (("constant", constant), ("decay", decay)):
            model = f"{recipe}_{order}"
            train, loss = summaries[f"ladder_l5_noise_train_{model}"], json.loads(store.path(f"l5_noise_loss_{model}.json").read_text())
            assert (train["model"], train["rows"], train["steps"], train["adapter"], train["passes"]) == (model, 8, 8, str(store.root / "adapters" / model), 1)
            assert train["adapter_saved"]["rank"] == 64 and train["against_the_start_adapter"] == {model: CHANGED} and f"the ORDER SEED {order}" in train["order"]
            # EVERY STEP'S LEARNING RATE is recorded beside its row, and the summary says what it was held to.
            assert [row["learning_rate"] for row in loss["training_steps"]] == rates and [row["step"] for row in loss["training_steps"]] == list(range(1, 9))
            held = train["learning_rate"]
            assert (held["decay"], held["steps"], held["of_the_first_step"], held["of_the_last_step"], held["factor_of_the_last_step"]) == (recipe == "decay", 8, RATE / 5, rates[-1], rates[-1] / RATE)
            assert loss["rows_trained"] == [row["id"] for row in sets[model]] and "learning_rates" not in train

    # ---- the two averaged adapters, trained by nothing: every number the mean of its four models' at that place; read back; none of them; at rank 64
    for recipe, numbers in (("constant", (1, 3, 5, 7)), ("decay", (2, 4, 6, 8))):           # the k-th training's adapter holds k and k / 2
        made, name = summaries[f"ladder_l5_noise_average_{recipe}"], f"mean_{recipe}"
        directory, of = store.root / "adapters" / name, [store.root / "adapters" / f"{recipe}_{order}" for order in ORDERS]
        assert (made["model"], made["recipe"], made["of"], made["adapter"], made["averaged"], made["check"]) == (name, recipe, [f"{recipe}_{order}" for order in ORDERS], str(directory), True, "L5n")
        a, b = "base_model.model.layers.0.q_proj.lora_A.weight", "base_model.model.layers.0.down_proj.lora_B.weight"
        assert set(_numbers(directory / "adapter_model.safetensors", a).tolist()) == {sum(numbers) / 4} and set(_numbers(directory / "adapter_model.safetensors", b).tolist()) == {sum(numbers) / 8}
        assert [set(_numbers(path / "adapter_model.safetensors", a).tolist()) for path in of] == [{float(number)} for number in numbers]
        held = made["read_back"]
        assert held == is_the_mean(directory / "adapter_model.safetensors", [path / "adapter_model.safetensors" for path in of]) and held["passes"] is True
        assert (held["adapters"], held["elements"], held["elements_not_the_mean"], held["is_none_of_them"], made["made"]["elements"]) == (4, 1600, 0, True, 1600)
        assert all(count == 1600 for count in held["elements_not_those_of_each_adapter"]) and made["adapter_saved"]["rank"] == 64 and made["adapter_saved"]["ranks"] == [64]
        assert made["against_the_start_adapter"] == compared(directory / "adapter_model.safetensors", start / "adapter_model.safetensors") and made["against_the_start_adapter"]["elements_changed"] == 1600
        # SAVED WITH THE ADAPTER CONFIG ITS MODELS SHARE (their module names were saved in two orders: one config all the same).
        assert (directory / "adapter_config.json").read_text() == (of[0] / "adapter_config.json").read_text() != (of[1] / "adapter_config.json").read_text()
    assert len(calls) == 8                                                                   # NO TRAINING made either

    # ---- the ten measurements: the goal set's FIRST SAMPLING ALONE, 32 attempts with the sampling seed `pre_r64`'s run recorded; no rung, no second sampling
    assert solver.measured == [(GOAL, 32, 1001, 1024)] * 10 and len(deep_.solver.asked) == noise.asked                # NOTHING else was sampled: the world's own solver was asked nothing
    assert [kit["max_lora_rank"] for kit in kits] == [64] * 10 and {kit["temperature"] for kit in kits} == {1.0}
    figures = {}
    for model in MODELS:
        measured = summaries[f"ladder_l5_noise_measure_{model}"]
        assert (measured["model"], measured["check"], measured["adapter"]) == (model, "L5n", str(store.root / "adapters" / model)) and "rungs" not in measured and list(measured["goal"]) == ["reach"]
        assert (measured["goal"]["reach"]["set"], measured["goal"]["reach"]["episodes_each"], measured["goal"]["reach"]["sampling_seed"]) == (f"l4_reach_{model}", 32, 1001)
        own = store.read_rows(f"episodes_l4_reach_{model}_problems.jsonl")
        assert [(row["problem_id"], row["episodes"], row["resolved"]) for row in own] == [(GOAL, 32, SUCCESSES[model])]
        figures[model] = figure(own)
    assert not list(store.root.glob("episodes_l4_rungs_*")) and not list(store.root.glob("episodes_l4_more_*"))
    assert sorted(path.name for path in store.root.glob("episodes_*.done.json") if "_block_" not in path.name) == sorted(f"episodes_l4_reach_{model}.done.json" for model in MODELS)
    order_of = list(summaries)
    assert all(order_of.index(f"ladder_l5_noise_measure_{model}") == order_of.index(f"ladder_l5_noise_train_{model}") + 1 for model in TRAINED)
    assert all(order_of.index(f"ladder_l5_noise_measure_mean_{recipe}") == order_of.index(f"ladder_l5_noise_average_{recipe}") + 1 for recipe in RECIPES)

    # ---- the report: the checks pass; each read is on the rows the steps stored
    report = summaries["ladder_l5_noise_report"]
    assert json.loads(store.path("report_ladder_l5_noise.json").read_text()) == json.loads(json.dumps(report, default=str)) and report["ok"] is True and list(report) == REPORT_KEYS
    assert report["inconclusive"] is False and report["failed_checks"] == [] and all(line.startswith(f"{SAY}: ") for line in report["lines"])
    assert {name: {key: value for key, value in own.items() if key != "share_of_distinct_attempts"} for name, own in report["figures"].items() if name != "what"} == figures
    assert {name: report["figures"][name]["successes"] for name in MODELS} == SUCCESSES
    as_it_is_, lowers, level = report["the_spread_of_the_recipe_as_it_is"], report["does_decay_lower_it"], report["and_its_level"]
    assert (as_it_is_["trainings"], lowers["of_decay_over_of_constant"], level["differences"]) == (4, 0.5, [0.125, 0.0625, 0.0, -0.0625])
    # L5's `published` at seeds 1 and 2 is NOT THERE (no such run in this world): the spread is given and its name is NOT READ. Seed 0's stands beside: one more draw.
    assert report["branches"] == {"the_spread_of_the_recipe_as_it_is": NOT_READ, "does_decay_lower_it": DECAY_LOWERS, "and_its_level": NAMED_NOTHING,
                                  "averaging": {"mean_constant": AT_LEAST_THE_MEAN, "mean_decay": AT_LEAST_THE_MEAN}, "what_follows": WITH_THE_DECAY}
    of_seeds = report["beside"]["the_published_of_l5"]
    assert list(of_seeds) == ["0", "1", "2"] and (of_seeds["0"]["there"], of_seeds["0"]["successes"], of_seeds["0"]["run"]) == (True, OF_L5, str(noise.l5))
    assert of_seeds["1"] == {"there": False, "why": "ladder_l5_deep_pre_r64_seed1 does not hold " + str([
        "ladder_l5_report.done.json", "report_ladder_l5.json", "ladder_l5_train_published.done.json", "ladder_l5_measure_published.done.json", "episodes_l4_reach_published_problems.jsonl",
        "episodes_l4_reach_published.done.json"])} and of_seeds["2"]["there"] is False
    assert (report["averaging"]["mean_constant"]["per_attempt"], report["averaging"]["mean_constant"]["mean_of_its_models"], report["averaging"]["mean_decay"]["per_attempt"]) == (0.3125, 0.3125, 0.34375)
    assert report["adapters"]["there"] == sorted(MODELS) and report["adapters"]["kept"] is True and report["smoke"] is None
    assert store.done_summary("ladder_l5_noise_report") == {"stage": "l4", "label": "pretrained on published proofs", "check": "L5n", "headline": report["headline"],
                                                           "branches": report["branches"], "measured": list(MODELS), "ok": True}
    # ---- queued again: every step returns what is stored; nothing is trained, averaged or sampled; nothing that was read moved
    own_files = _all(store.root)
    again = _stage5n(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}
    assert (len(calls), len(solver.measured)) == (8, 10) and again["ladder_l5_noise_report"]["branches"] == report["branches"] and {path: _all(path) for path in read_only} == read_only
    assert {name: content for name, content in _all(store.root).items() if "report" not in name} == {name: content for name, content in own_files.items() if "report" not in name}


def test_as_on_the_gpu_every_measurement_serves_the_adapter_of_its_own_model_the_averaged_ones_too_and_the_start_adapters_type_is_held(noise, monkeypatch):  # noqa: F811
    config, kits = noise.config, noise.kits
    # The stand-in engine takes no adapter, so in the scripted world none is asked for. As on the GPU every measurement asks the model server for ITS model's adapter.
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        start = noise.deep.rank / "adapters" / "pre_r64" / "adapter_model.safetensors"
        own_root = noise.runs / "ladder_l5_noise_pre_r64_seed0"
        # THE START ADAPTER's numbers are of a type the averaged adapters cannot be written in: refused by the FIRST step, before anything is written or trained.
        kept = start.read_bytes()
        _write(start, [("base_model.model.layers.0.q_proj.lora_A.weight", "BF16", (2,), struct.pack("<2H", 0x3F80, 0x4000))])
        with pytest.raises(RuntimeError, match=r"adapters/pre_r64 holds tensors of type \['BF16'\]: the averaged adapters are written in \['F16', 'F32', 'F64'\].*Nothing was written"):
            ladder_l5_noise.ladder_l5_noise_prepare(config)
        assert not any(own_root.iterdir())
        start.write_bytes(kept)
        summaries = _stage5n(config, patched)
        store = noise.store()
        kept_here = store.root / "adapters"
        assert [kit["adapter"] for kit in kits] == [(f"ladder_l5_noise_{model}", 1, str(kept_here / model)) for model in MODELS] and [kit["max_lora_rank"] for kit in kits] == [64] * 10
        assert len(summaries) == len(STEPS) and summaries["ladder_l5_noise_measure_mean_decay"]["stand_in_engine"] is False and summaries["ladder_l5_noise_report"]["inconclusive"] is False
        assert summaries["ladder_l5_noise_report"]["can_this_run_be_read"]["mean_decay"]["its_numbers_are_the_mean_of_its_models_and_it_is_served"]["served"] is True
        # An adapter that is no longer there: the measurement is refused, not made from another model's.
        for model in ("decay_3", "mean_constant"):
            shutil.rmtree(kept_here / model)
            store.path(f"ladder_l5_noise_measure_{model}.done.json").unlink()
            with pytest.raises(RuntimeError, match=f"adapters/{model} is not there: the adapter `{model}` is kept by this stage"):
                ladder_l5_noise.STEPS[f"ladder_l5_noise_measure_{model}"](config)
            assert not store.is_done(f"ladder_l5_noise_measure_{model}")
        # AN AVERAGED ADAPTER IS MADE AGAIN when its step is queued again (its models are kept): the same numbers.
        store.path("ladder_l5_noise_average_constant.done.json").unlink()
        made = ladder_l5_noise.STEPS["ladder_l5_noise_average_constant"](config)
        assert made["read_back"]["passes"] is True and ladder_l5_noise.STEPS["ladder_l5_noise_measure_mean_constant"](config)["model"] == "mean_constant"
        # ... but not when one of its models' adapters is gone.
        store.path("ladder_l5_noise_average_decay.done.json").unlink()
        with pytest.raises(RuntimeError, match="adapters/decay_3.. are not there: the adapters `mean_decay` is the mean of are kept by this stage. Nothing was averaged"):
            ladder_l5_noise.STEPS["ladder_l5_noise_average_decay"](config)
        assert not store.is_done("ladder_l5_noise_average_decay")


def test_the_first_step_refuses_before_anything_is_written(noise, monkeypatch):  # noqa: F811
    config, root = noise.config, noise.l5
    own_root = noise.runs / "ladder_l5_noise_pre_r64_seed0"
    task = "the task of stage `ladder_l5_deep` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l5_deep --seeds 0`; for the smoke run, stage `ladder_l5_deep_smoke`."
    why = "L5n draws no row: its fixed rows are the set L5's finished run of seed 0 trained `published` on, read from that run"

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l5_noise.ladder_l5_noise_prepare(config)
        assert not own_root.exists() or not any(own_root.iterdir())             # NOTHING WAS WRITTEN

    def with_changed(path, change, match):
        kept = path.read_text()
        path.write_text(change(kept))
        refused(match)
        path.write_text(kept)

    def without(path, match):
        path.rename(path.with_name(path.name + ".away"))
        refused(match)
        path.with_name(path.name + ".away").rename(path)

    as_json = lambda change: (lambda text: json.dumps(change(json.loads(text))))      # noqa: E731
    as_rows = lambda change: (lambda text: "".join(json.dumps(row) + "\n" for row in change([json.loads(line) for line in text.splitlines()])))      # noqa: E731
    # ---- L5's RUN MISSING; UNFINISHED: a step's marker, its report or a draw is not there (L5's own reading, in this stage's words)
    root.rename(root.with_name("elsewhere"))
    refused(f"ladder_l5_deep_pre_r64_seed0 does not hold .*the FINISHED run of the deep search. {why}. Run {task} to its end first; nothing was written")
    root.with_name("elsewhere").rename(root)
    for name in ("ladder_l5_prepare.done.json", "ladder_l5_report.done.json", "report_ladder_l5.json", "l5_search_problems.jsonl"):
        without(root / name, f"ladder_l5_deep_pre_r64_seed0 does not hold .*{name}.*the FINISHED run of the deep search. L5n draws no row")
    # ---- A REPORT NOT TO BE READ, or INCONCLUSIVE: no model here stands beside such a run
    with_changed(root / "report_ladder_l5.json", as_json(lambda report: {**report, "ok": False}), "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is not to be read .*: no model here stands beside such a run")
    with_changed(root / "report_ladder_l5.json", as_json(lambda report: {**report, "inconclusive": True, "failed_checks": ["`shallow`: its_training_ran"]}),
                 "the report of the deep search's run ladder_l5_deep_pre_r64_seed0 is INCONCLUSIVE .`shallow`: its_training_ran.: no model here stands beside such a run")
    # ---- MADE OTHERWISE (L5's own holds): another rank recorded for the start adapter
    with_changed(root / "ladder_l5_search_prepare.done.json", as_json(lambda own: {**own, "start_recipe": {**own["start_recipe"], "rank": 16}}),
                 "was made with another start model, rank, search or held-aside set than this task would make: the start adapter's rank and alpha .*L5n's models are trained on that run's rows")
    # ---- ITS MODEL `published`: no model trained; a marker, its set, its training's record or its stored first sampling not there
    with_changed(root / "ladder_l5_prepare.done.json", as_json(lambda own: {**own, "trainings": {"run": False, "why": "too few deep solves"}}),
                 "L5's run ladder_l5_deep_pre_r64_seed0 trained no model .too few deep solves.: there is no set of `published` to train on again")
    for name in ("ladder_l5_train_published.done.json", "ladder_l5_measure_published.done.json", "l5_training_published.jsonl", "l5_loss_published.json",
                 "episodes_l4_reach_published_problems.jsonl", "episodes_l4_reach_published.done.json"):
        without(root / name, f"ladder_l5_deep_pre_r64_seed0 does not hold .*{name}.*what L5's run recorded of `published`.*Run {task} to its end first; nothing was written")
    # ---- A STORED SET THAT DOES NOT AGREE WITH ITS RUN: fewer rows than its training recorded; rows its own step did not count
    with_changed(root / "l5_training_published.jsonl", as_rows(lambda own: own[:-1]), "stored 7 rows for `published`, its training recorded 8 and its own step counted 8 .twice its rows of each kind.: the run does not agree with itself")
    with_changed(root / "ladder_l5_train_published.done.json", as_json(lambda own: {**own, "rows": 6}), "stored 8 rows for `published`, its training recorded 6 and its own step counted 8")
    # ---- ITS STORED FIRST SAMPLING would not pair: another sampling seed, other attempts
    with_changed(root / "episodes_l4_reach_published.done.json", as_json(lambda own: {**own, "sampling_seed": 7}), "ladder_l5_deep_pre_r64_seed0 measured `published` .reach. with sampling seed 7 and 32 episodes")
    # ---- THE FIXED ROWS held to the file, to L5's rule and to what that training recorded (the rule's refusals, with the run they are of)
    said = ".L5's run ladder_l5_deep_pre_r64_seed0.. Nothing was written"
    with_changed(root / "l5_training_published.jsonl", as_rows(lambda own: [{**own[0], "completion": "by simp"}, *own[1:]]), "1 rehearsal rows of the stored set of L5's `published` carry a text field.*" + said)
    with_changed(root / "l5_training_published.jsonl", as_rows(lambda own: [{**own[0], "row_of_the_file": own[1]["row_of_the_file"]}, *own[1:]]),
                 "1 rows of the stored set of L5's `published` are not the rows of the pretraining file at their place.*" + said)
    with_changed(root / "l5_training_published.jsonl", as_rows(lambda own: [own[1], own[0], *own[2:]]), "does not stand in the content-hash order of seed 0, or its training's own record .8 rows. is not these rows in this order.*" + said)
    with_changed(root / "l5_loss_published.json", as_json(lambda own: {**own, "rows_trained": own["rows_trained"][::-1]}), "its training's own record .8 rows. is not these rows in this order.*" + said)
    # ---- AN ORDER THAT IS THE ONE L5 TRAINED THE ROWS IN is refused in a real run (a smoke run alone trains it all the same): here order seed 3 is made to give it
    in_its_order = l5_noise.training_order
    monkeypatch.setattr(l5_noise, "training_order", lambda rows_, seed: list(rows_) if seed == 3 else in_its_order(rows_, seed))
    refused(r"of the order seeds \[1, 2, 3, 4\], \[3\] give the order L5's own task trained these 8 rows in, and the pairs \[\] give one order.*" + said)
    monkeypatch.setattr(l5_noise, "training_order", in_its_order)
    # ---- THE RECIPE is the one that trained L5's `published`: another learning rate, warm-up or batch now is refused
    for key, value in (("learning_rate", 5e-5), ("warmup_steps", 3), ("effective_batch", 8)):
        kept, config["training"][key] = config["training"][key], value
        refused(f"L5's run ladder_l5_deep_pre_r64_seed0 trained its models with .* and this task would train with .*'{key}': {value!r}.*`constant` is the recipe as it is, the one that trained that run's `published`")
        config["training"][key] = kept
    # ---- THE SAMPLING is the start model's FIRST, with its recorded attempts: a setting that names another, or other attempts, is refused
    settings = config["ladder_loop"]["l5_noise"]
    for key, value, match in (("sampling", "more", "names the sampling `more` of 32 attempts a problem, and the run of `pre_r64` .ladder_l4_pretrain_r64_seed0. recorded its first sampling of the goal set as `reach` of 32"),
                              ("attempts", 48, "names the sampling `reach` of 48 attempts a problem, and the run of `pre_r64` .* recorded its first sampling of the goal set as `reach` of 32")):
        kept, settings[key] = settings[key], value
        refused(match)
        settings[key] = kept
    # ---- A TASK OF ANOTHER SEED: nothing here is drawn by the task's seed
    monkeypatch.setenv(SEEDS, "1")
    with pytest.raises(RuntimeError, match="this task's seed is 1 and ladder_loop.l5.search_seed is 0: the fixed rows are that seed's, the orders are the setting's .ladder_loop.l5_noise.orders. and nothing here is drawn by the task's seed"):
        ladder_l5_noise.ladder_l5_noise_prepare(config)
    assert not (noise.runs / "ladder_l5_noise_pre_r64_seed1").exists() or not any((noise.runs / "ladder_l5_noise_pre_r64_seed1").iterdir())
    monkeypatch.delenv(SEEDS)
    # ---- THE SETTINGS: orders or recipes that are not the stage's; an order that is L5's own seed; seeds of L5 without its own; attempts that are no number
    for key, value, match in (("orders", [1, 2, 3], "ladder_loop.l5_noise.orders is .1, 2, 3. and recipes .'constant', 'decay'.; the stage trains and measures .1, 2, 3, 4. with .'constant', 'decay'.: change both together"),
                              ("recipes", ["decay", "constant"], "the stage trains and measures .1, 2, 3, 4. with .'constant', 'decay'.: change both together"),
                              ("attempts", 0, "ladder_loop.l5_noise.attempts is 0 and sampling 'reach': a whole number of at least 1"),
                              ("attempts", True, "ladder_loop.l5_noise.attempts is True"), ("sampling", "", "ladder_loop.l5_noise.attempts is 32 and sampling ''"),
                              ("published_seeds", [1, 2], "ladder_loop.l5_noise.published_seeds is .1, 2.: distinct whole numbers of at least 0, the seed whose task trained the fixed rows .ladder_loop.l5.search_seed, 0. among them"),
                              ("published_seeds", [0, 1, 1], "ladder_loop.l5_noise.published_seeds is .0, 1, 1.: distinct whole numbers")):
        kept, settings[key] = settings[key], value
        refused(match, ValueError)
        settings[key] = kept
    kept, config["ladder_loop"]["l5"]["search_seed"] = config["ladder_loop"]["l5"]["search_seed"], 3
    refused("ladder_loop.l5_noise.orders is .1, 2, 3, 4. and ladder_loop.l5.search_seed is 3: that order is the one L5's own task trained the fixed rows in", ValueError)
    config["ladder_loop"]["l5"]["search_seed"] = kept
    # ---- THE SMOKE-ONLY VARIABLE in a run of the seed's own directory: refused (the smoke stage's test holds the rest of its guard)
    monkeypatch.setenv(SMOKE, "2")
    refused("RLVR_LEAN_LADDER_L5_NOISE_SMOKE_ORDERS is set .2.: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_NOISE_RUN. and on the fixtures' base map")
    monkeypatch.delenv(SMOKE)
    # ---- and with nothing changed it prepares: every refusal above left the runs as they were
    assert ladder_l5_noise.ladder_l5_noise_prepare(config)["the_rows"]["rows"] == 8


def test_a_training_that_was_not_made_with_its_schedule_is_refused_and_its_step_is_not_marked(noise, monkeypatch):  # noqa: F811
    config = noise.config
    _stage5n(config, monkeypatch, until="ladder_l5_noise_measure_constant_1")
    store, calls = noise.store(), []
    # A TRAINING ASKED FOR THE DECAY THAT RECORDED THE RECIPE'S OWN RATES (the change did not reach it): refused, nothing of it is stored but its adapter, and its step is not marked.
    _trainings(monkeypatch, calls, rates=lambda more, recorded: learning_rates(RATE, WARMUP, len(recorded), False))
    with pytest.raises(RuntimeError, match="`decay_1`: the learning rate recorded at step 6 is 0.0001 and the schedule with the decay gives 7.5\\d*e-05 .3 of 8 steps are not the schedule's.: the training was not "
                                           "made with the schedule asked for. Nothing of this model is measured, and the step is not marked done"):
        _on_the_gpu(monkeypatch, ladder_l5_noise.STEPS["ladder_l5_noise_train_decay_1"], config)
    assert not store.is_done("ladder_l5_noise_train_decay_1") and not store.path("l5_noise_loss_decay_1.json").exists() and calls[0]["more"]["decay"] is True
    with pytest.raises(RuntimeError, match="the measurement of L5n's model `decay_1` needs the step ladder_l5_noise_train_decay_1 of this run, which is not done"):
        ladder_l5_noise.STEPS["ladder_l5_noise_measure_decay_1"](config)
    # A TRAINING OF THE RECIPE AS IT IS whose rate fell is refused the same way; so is one that recorded no rate, or fewer rates than it made steps.
    for rates, match in ((lambda more, recorded: learning_rates(RATE, WARMUP, len(recorded), True), "`constant_2`: the learning rate recorded at step 6 is 7.5\\d*e-05 and the schedule of the recipe as it is gives 0.0001"),
                         (lambda more, recorded: [], "`constant_2`: the training recorded no learning rate"), (lambda more, recorded: recorded[:-1], "`constant_2`: the training recorded no learning rate")):
        _trainings(monkeypatch, calls, rates=rates)
        with pytest.raises(RuntimeError, match=match):
            _on_the_gpu(monkeypatch, ladder_l5_noise.STEPS["ladder_l5_noise_train_constant_2"], config)
        assert not store.is_done("ladder_l5_noise_train_constant_2")
    # Made with its schedule, the same step is marked.
    _trainings(monkeypatch, calls)
    assert _on_the_gpu(monkeypatch, ladder_l5_noise.STEPS["ladder_l5_noise_train_decay_1"], config)["learning_rate"]["decay"] is True and store.is_done("ladder_l5_noise_train_decay_1")
    # A MODEL THAT IS NONE OF THIS RUN'S (a step the stage registers for no order of the setting; an averaged adapter is trained by nothing) has no step.
    for model, match in (("constant_9", "`constant_9` is none of this run's models"), ("mean_constant", "`mean_constant` is no trained model of L5n")):
        with pytest.raises((RuntimeError, ValueError), match=match):
            ladder_l5_noise.ladder_l5_noise_train(config, model)
    with pytest.raises(RuntimeError, match="`decay_9` is none of this run's models"):
        ladder_l5_noise.ladder_l5_noise_measure(config, "decay_9")


def test_an_averaged_adapter_is_refused_unless_it_is_the_mean_of_its_models_none_of_them_and_at_the_start_adapters_rank(noise, monkeypatch):  # noqa: F811
    config = noise.config
    trained = [step for step in STEPS if "_train_" in step or step == "ladder_l5_noise_prepare"]
    average = lambda recipe="constant": _on_the_gpu(monkeypatch, ladder_l5_noise.STEPS[f"ladder_l5_noise_average_{recipe}"], config)      # noqa: E731
    _stage5n(config, monkeypatch, only=trained[:-1])                               # every training but `decay_4`'s
    store = noise.store()
    kept_here = store.root / "adapters"
    # A MODEL OF THE RECIPE THAT IS NOT TRAINED YET: its averaged adapter cannot be made (the other recipe's can).
    with pytest.raises(RuntimeError, match="the averaged adapter `mean_decay` needs the step ladder_l5_noise_train_decay_4 of this run, which is not done"):
        average("decay")
    assert not (kept_here / "mean_decay").exists() and not store.is_done("ladder_l5_noise_average_decay")
    # AN AVERAGED MODEL IS MEASURED ONLY ONCE ITS OWN STEP HAS MADE ITS ADAPTER.
    with pytest.raises(RuntimeError, match="the measurement of L5n's model `mean_constant` needs the step ladder_l5_noise_average_constant of this run, which is not done"):
        ladder_l5_noise.STEPS["ladder_l5_noise_measure_mean_constant"](config)
    assert not store.is_done("ladder_l5_noise_measure_mean_constant") and noise.solver.measured == []
    # THE STAND-IN trains no adapter, and none is averaged: said, and marked (a pre-flight without a GPU).
    made = ladder_l5_noise.STEPS["ladder_l5_noise_average_constant"](config)
    assert (made["averaged"], made["stand_in_engine"]) == (False, True) and "none was averaged" in made["note"] and not (kept_here / "mean_constant").exists()
    store.path("ladder_l5_noise_average_constant.done.json").unlink()

    def refused(match):
        with pytest.raises(RuntimeError, match=match):
            average()
        assert not store.is_done("ladder_l5_noise_average_constant")

    # ADAPTERS THAT DO NOT HOLD THE SAME TENSORS (one saved at another shape): refused by the mean itself.
    weights = kept_here / "constant_2" / "adapter_model.safetensors"
    kept = weights.read_bytes()
    _write(weights, [("base_model.model.layers.0.q_proj.lora_A.weight", "F32", (1, 2), _of("F32", [[1.0, 2.0]]))])
    refused("`mean_constant`: .*constant_2/adapter_model.safetensors and .*constant_1/adapter_model.safetensors do not hold the same tensors .*Nothing of this model is measured, and the step is not marked done")
    weights.write_bytes(kept)
    # ADAPTERS SAVED WITH ANOTHER CONFIG (another alpha): the averaged adapter is saved with the config its models SHARE.
    of_config = kept_here / "constant_3" / "adapter_config.json"
    kept = of_config.read_text()
    of_config.write_text(json.dumps({**json.loads(kept), "lora_alpha": 64}))
    refused("the adapters of .'constant_1', 'constant_2', 'constant_3', 'constant_4'. were not saved with one adapter config")
    of_config.write_text(kept)
    # A FILE THAT, READ BACK, IS NOT THE MEAN (the writer wrote the first adapter's numbers): refused, not marked.
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l5_noise, "mean_of_adapters", lambda files, target: (shutil.copyfile(files[0], target), {"how": "copied", "adapters": 4, "tensors": 4, "elements": 1600})[1])
        refused("the adapter `mean_constant` saved at .*adapters/mean_constant, read back, has 1,600 of 1,600 numbers that are not the mean of its models', differs from each of them in .0, 1600, 1600, 1600. numbers")
    # FOUR MODELS THAT ARE ONE ADAPTER: their mean is each of them, and that is refused (an average of nothing).
    others = {name: (kept_here / name / "adapter_model.safetensors").read_bytes() for name in ("constant_2", "constant_3", "constant_4")}
    for name in others:
        shutil.copyfile(kept_here / "constant_1" / "adapter_model.safetensors", kept_here / name / "adapter_model.safetensors")
    refused("read back, has 0 of 1,600 numbers that are not the mean of its models', differs from each of them in .0, 0, 0, 0. numbers")
    for name, content in others.items():
        (kept_here / name / "adapter_model.safetensors").write_bytes(content)
    # MODELS AT ANOTHER RANK THAN THE START ADAPTER'S (each saved at rank 16): their mean is theirs, and is refused for its rank.
    whole = {name: _all(kept_here / name) for name in ("constant_1", "constant_2", "constant_3", "constant_4")}
    for index, name in enumerate(whole):
        _adapter_of(kept_here / name, 16, 32, float(index + 1))
    refused("has rank 16 and alpha 32 with matrices of rank .16. .the start adapter `pre_r64`: 64 and 128.")
    for name, files in whole.items():
        for path, content in files.items():
            (kept_here / name / path).write_bytes(content)
    # With its models as they were trained, it is made, read back and marked.
    made = average()
    assert made["averaged"] is True and made["read_back"]["passes"] is True and store.is_done("ladder_l5_noise_average_constant") and adapter_files.MEAN_HOW == made["read_back"]["how"]
    with pytest.raises(RuntimeError, match="`mean` is none of this run's models"):
        ladder_l5_noise.ladder_l5_noise_average(config, "mean")


def test_l5s_published_at_its_other_seeds_is_read_by_the_report_when_those_runs_are_there_and_finished(noise, monkeypatch):  # noqa: F811
    config = noise.config
    summaries = _stage5n(config, monkeypatch)
    store = noise.store()
    constant = summaries["ladder_l5_noise_report"]["the_spread_of_the_recipe_as_it_is"]["standard_deviation"]
    assert summaries["ladder_l5_noise_report"]["branches"]["the_spread_of_the_recipe_as_it_is"] == NOT_READ and constant == round(figure_sd([4, 8, 12, 16]), 6)
    report = lambda: ladder_l5_noise.ladder_l5_noise_report(config)      # noqa: E731
    # SEED 1 THERE, SEED 2 NOT: still NOT READ (the standard deviation is of L5's three seeds), and seed 1's rows are copied into this run.
    seed_1 = _another_seed_of_l5(noise, 1, 20)
    one = report()
    assert one["branches"]["the_spread_of_the_recipe_as_it_is"] == NOT_READ and one["beside"]["the_published_of_l5"]["1"]["successes"] == 20 and one["beside"]["the_published_of_l5"]["2"]["there"] is False
    assert store.read_rows("l5_noise_stored_published_seed1.jsonl") == _file_rows(seed_1 / "episodes_l4_reach_published_problems.jsonl") and not store.path("l5_noise_stored_published_seed2.jsonl").exists()
    assert one["beside"]["the_published_of_l5"]["1"]["run"] == str(seed_1) and "seed 2: ladder_l5_deep_pre_r64_seed2 does not hold" in one["the_spread_of_the_recipe_as_it_is"]["reason"]
    # ALL THREE THERE: the name is read. L5's `published` at 12, 20 and 4 of 32 is as spread as the orders here: ORDER ALONE MAKES THE SPREAD.
    seed_2 = _another_seed_of_l5(noise, 2, 4)
    three = report()
    as_it_is = three["the_spread_of_the_recipe_as_it_is"]
    assert as_it_is["name"] == ORDER_ALONE and (as_it_is["seeds"], as_it_is["of_published_over_the_seeds"], as_it_is["standard_deviation"]) == (3, round(figure_sd([12, 20, 4]), 6), constant)
    assert store.read_rows("l5_noise_stored_published_seed2.jsonl") == _file_rows(seed_2 / "episodes_l4_reach_published_problems.jsonl")
    assert {seed: own["successes"] for seed, own in three["beside"]["the_published_of_l5"].items()} == {"0": OF_L5, "1": 20, "2": 4} and three["branches"]["what_follows"] == WITH_THE_DECAY
    # ... and far more spread than the orders: THE ROWS DRAWN MAKE MOST OF IT.
    path = seed_2 / "episodes_l4_reach_published_problems.jsonl"
    path.write_text("".join(json.dumps({**row, "resolved": 32}) + "\n" for row in _file_rows(path)))
    seed_1.joinpath("episodes_l4_reach_published_problems.jsonl").write_text("".join(json.dumps({**row, "resolved": 0}) + "\n" for row in _file_rows(seed_1 / "episodes_l4_reach_published_problems.jsonl")))
    assert report()["branches"]["the_spread_of_the_recipe_as_it_is"] == ROWS_DRAWN
    # A SEED'S RUN THAT IS NOT FINISHED, NOT TO BE READ, OR WOULD NOT PAIR is NOT THERE, with why; nothing is refused for it, and the report is written.
    def not_there(change, why):
        kept = {name: content for name, content in _all(seed_2).items()}
        change()
        found = report()
        assert found["ok"] is True and found["beside"]["the_published_of_l5"]["2"]["there"] is False and why in found["beside"]["the_published_of_l5"]["2"]["why"], found["beside"]["the_published_of_l5"]["2"]
        assert found["branches"]["the_spread_of_the_recipe_as_it_is"] == NOT_READ
        for name, content in kept.items():
            (seed_2 / name).write_bytes(content)

    not_there(lambda: (seed_2 / "ladder_l5_report.done.json").unlink(), "ladder_l5_deep_pre_r64_seed2 does not hold ['ladder_l5_report.done.json']")
    not_there(lambda: (seed_2 / "ladder_l5_measure_published.done.json").unlink(), "does not hold ['ladder_l5_measure_published.done.json']")
    not_there(lambda: (seed_2 / "report_ladder_l5.json").write_text(json.dumps({"ok": False})), "the report of ladder_l5_deep_pre_r64_seed2 is not to be read")
    not_there(lambda: (seed_2 / "episodes_l4_reach_published.done.json").write_text(json.dumps({**json.loads((seed_2 / "episodes_l4_reach_published.done.json").read_text()), "sampling_seed": 9})),
              "measured `published` of seed 2 with sampling seed 9 and 32 episodes")
    not_there(lambda: path.write_text("".join(json.dumps({**row, "problem_id": "another"}) + "\n" for row in _file_rows(path))), "on other problems")
    # A seed whose own report read INCONCLUSIVE is read, and said (a rise of a training's loss decides nothing of its figure).
    (seed_2 / "report_ladder_l5.json").write_text(json.dumps({**json.loads((seed_2 / "report_ladder_l5.json").read_text()), "inconclusive": True}))
    said = report()
    assert said["beside"]["the_published_of_l5"]["2"]["inconclusive"] is True and said["branches"]["the_spread_of_the_recipe_as_it_is"] == ROWS_DRAWN
    assert any("seed 2: 1000.0 per 1,000, 1 solved (its own report reads INCONCLUSIVE)" in line for line in said["lines"])
    # THE RUNS OF L5 WERE ONLY READ, the seed's own among them.
    assert sorted(path_.name for path_ in noise.runs.iterdir() if path_.name.startswith("ladder_l5")) == ["ladder_l5_deep_pre_r64_seed0", "ladder_l5_deep_pre_r64_seed1", "ladder_l5_deep_pre_r64_seed2",
                                                                                                         "ladder_l5_noise_pre_r64_seed0"]


def test_a_set_lean_did_not_answer_makes_the_run_inconclusive_fails_the_report_and_is_sampled_again_from_its_adapter(noise, monkeypatch):  # noqa: F811
    config = noise.config
    _stage5n(config, monkeypatch, until="ladder_l5_noise_measure_mean_decay")
    store = noise.store()
    path = store.path("episodes_l4_reach_decay_2_problems.jsonl")
    kept = path.read_text()
    path.write_text("".join(json.dumps({**row, "attempts_without_an_answer": 4}) + "\n" for row in _file_rows(path)))       # 4 of 32: more than 0.02
    report = ladder_l5_noise.ladder_l5_noise_report(config)
    assert report["ok"] is False and report["not_to_be_read"] == ["decay_2"] and report["inconclusive"] is True and report["failed_checks"] == ["`decay_2`: lean_answered"]
    assert (report["branches"]["does_decay_lower_it"], report["branches"]["what_follows"]) == (INCONCLUSIVE, NOT_READ)
    assert any("NOT TO BE READ: too many attempts without a verdict from Lean in the sets of `decay_2`" in line for line in report["lines"]) and store.done_summary("ladder_l5_noise_report")["ok"] is False
    # QUEUED AGAIN: that set, and no other, is sampled again from the kept adapter; no model is trained or averaged again.
    asked, trained = len(noise.solver.measured), len(noise.calls)
    again = _stage5n(config, monkeypatch)
    assert len(noise.solver.measured) == asked + 1 and len(noise.calls) == trained and again["ladder_l5_noise_measure_decay_2"]["measured_again"] == ["l4_reach_decay_2"]
    assert again["ladder_l5_noise_report"]["ok"] is True and again["ladder_l5_noise_report"]["inconclusive"] is False and path.read_text() != kept


def test_the_steps_run_through_the_gpu_entry_point(noise, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(milestone2, "load_config", lambda path: noise.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    # In the stand-in's world no adapter is trained or averaged: every step runs, and the report says INCONCLUSIVE (a pre-flight without a GPU reads nothing).
    report = json.loads((tmp_path / "ladder_l5_noise_report.json").read_text())
    assert noise.store().is_done(ladder_l5_noise.REPORT) and report["inconclusive"] is True and report["stand_in_engine"] is True and len(report["failed_checks"]) == 10


# ---------------------------------------------------------------------------------------- the smoke-only path
def test_the_smoke_stage_goes_on_from_l5s_inconclusive_smoke_run_trains_two_orders_of_its_two_rows_and_says_so(deep, monkeypatch):  # noqa: F811
    for name in (RUN, SMOKE):
        monkeypatch.delenv(name, raising=False)
    config = deep.config
    first = _smoke_world(deep, monkeypatch)
    runs = ladder_l5._runs(config)
    of_l5 = runs / "ladder_l5_deep_smoke"
    failed = first["ladder_l5_report"]["failed_checks"]
    stored = _file_rows(of_l5 / "l5_training_published.jsonl")
    assert len(stored) == 2 and failed != []
    calls, measured = [], []
    _trainings(monkeypatch, calls, value=lambda number: 1.0)          # AS ON THE BOX: one optimizer step on the same two rows, whatever their order, is ONE training: four adapters alike
    solver = NoiseSolver(deep.solver, (4, 8, 8, 10, 6, 9))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (measured.append(given["vllm"]["max_lora_rank"]), (solver, stand_in_parameters))[1]))
    # ---- A REAL RUN IS REFUSED FOR SUCH A RUN OF L5 (its report is INCONCLUSIVE), in the seed's own directory and in one the task names.
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is INCONCLUSIVE .*: no model here stands beside such a run"):
        ladder_l5_noise.ladder_l5_noise_prepare(config)
    # THE SMOKE-ONLY VARIABLE without a run directory of the task's naming; naming fewer than two orders or more than the setting's; and on a map that is no fixture's: refused.
    said = "RLVR_LEAN_LADDER_L5_NOISE_SMOKE_ORDERS is set .{}.: it is for a SMOKE run alone, in a run directory the task names .RLVR_LEAN_LADDER_L5_NOISE_RUN. and on the fixtures' base map"
    monkeypatch.setenv(SMOKE, "2")
    with pytest.raises(RuntimeError, match=said.format(2) + ", and it names 2 to 4 of the setting's orders.*reads a run of L5 whose checks passed and trains every order. Nothing was written"):
        ladder_l5_noise.ladder_l5_noise_prepare(config)
    assert not (runs / "ladder_l5_noise_pre_r64_seed0").exists() or not any((runs / "ladder_l5_noise_pre_r64_seed0").iterdir())
    monkeypatch.setenv(RUN, "ladder_l5_noise_smoke")
    for orders in ("1", "5"):
        monkeypatch.setenv(SMOKE, orders)
        with pytest.raises(RuntimeError, match=said.format(orders)):
            ladder_l5_noise.ladder_l5_noise_prepare(config)
    monkeypatch.setenv(SMOKE, "2")
    marker = deep.map / "ladder_l4b_map.done.json"
    recorded = json.loads(marker.read_text())
    marker.write_text(json.dumps({**recorded, "fixture": False}))
    with pytest.raises(RuntimeError, match=said.format(2) + ". A run on the pool's problems reads a run of L5 whose checks passed and trains every order"):
        ladder_l5_noise.ladder_l5_noise_prepare(config)
    assert not (runs / "ladder_l5_noise_smoke").exists() or not any((runs / "ladder_l5_noise_smoke").iterdir())
    marker.write_text(json.dumps(recorded))
    # A report NOT TO BE READ is refused for a smoke run too.
    report_file = of_l5 / "report_ladder_l5.json"
    kept = report_file.read_text()
    report_file.write_text(json.dumps({**json.loads(kept), "ok": False}))
    with pytest.raises(RuntimeError, match="the report of the deep search's run ladder_l5_deep_smoke is not to be read"):
        ladder_l5_noise.ladder_l5_noise_prepare(config)
    report_file.write_text(kept)
    # ---- THE SMOKE RUN: its own directory, the fixtures' map, the variable. L5's run is read all the same, and that is RECORDED; two of the four orders are trained.
    read_only = {path: _all(path) for path in (of_l5, deep.map, deep.l4t2, deep.rank)}
    summaries = _stage5n(config, monkeypatch, stage_name="ladder_l5_noise_smoke")
    store = ladder_l5_noise._store(config, ladder_l5_noise.the_settings(config))
    assert store.root.name == "ladder_l5_noise_smoke" and list(summaries) == SMOKE_STEPS and {path: _all(path) for path in read_only} == read_only
    read = summaries["ladder_l5_noise_prepare"]
    smoke = read["smoke"]
    assert (smoke["orders"], smoke["orders_of_the_setting"], smoke["l5_read_though_inconclusive"], smoke["l5s_failed_checks"]) == (2, [1, 2, 3, 4], True, failed)
    assert (read["orders"], read["models"], read["the_rows"]["rows"], read["the_rows"]["optimizer_steps"]) == ([1, 2], list(models_of((1, 2))), 2, 1) and read["the_published_of_l5"]["inconclusive"] is True
    # TWO ROWS HAVE TWO ORDERS: an order seed that gives the order L5 trained them in, or the other seed's, is trained all the same, and RECORDED.
    before = [row["id"] for row in stored]
    again = [order for order in (1, 2) if [row["id"] for row in training_order(stored, order)] == before]
    alike = [[1, 2]] if [row["id"] for row in training_order(stored, 1)] == [row["id"] for row in training_order(stored, 2)] else []
    assert (smoke["the_order_trained_before"], smoke["alike"]) == (again, alike) == (read["the_orders"]["the_order_trained_before"], read["the_orders"]["alike"]) and (again or alike)
    for order in (1, 2):
        assert [row["id"] for row in store.read_rows(f"l5_noise_training_constant_{order}.jsonl")] == [row["id"] for row in training_order(stored, order)]
    # FOUR TRAININGS of one optimizer step, inside the warm-up: the decay is ASKED FOR in two of them, and its one recorded rate is the recipe's own.
    assert [call["more"].get("decay", False) for call in calls] == [False, True, False, True] and [call["schedule"]["checkpoint_steps"] for call in calls] == [{model: 1} for model in models_of((1, 2))[:4]]
    assert all(summaries[f"ladder_l5_noise_train_{model}"]["learning_rate"]["of_the_last_step"] == RATE / WARMUP for model in models_of((1, 2))[:4])
    assert summaries["ladder_l5_noise_train_decay_2"]["learning_rate"]["decay"] is True and summaries["ladder_l5_noise_train_constant_2"]["learning_rate"]["decay"] is False
    # TWO AVERAGED ADAPTERS, each the mean of TWO, read back; SIX measurements of the one goal problem's first sampling, each at rank 64. THE MEAN OF TWO ADAPTERS THAT ARE ONE
    # IS EACH OF THEM: a real run refuses it (an average of nothing), a smoke run keeps it, serves it, and SAYS so.
    for recipe in RECIPES:
        made = summaries[f"ladder_l5_noise_average_{recipe}"]
        assert (made["of"], made["read_back"]["adapters"], made["read_back"]["is_the_mean"]) == ([f"{recipe}_1", f"{recipe}_2"], 2, True)
        assert (made["read_back"]["is_none_of_them"], made["read_back"]["elements_not_those_of_each_adapter"], made["one_of_its_models_in_a_smoke_run"]) == (False, [0, 0], True)
        assert (store.root / "adapters" / f"mean_{recipe}" / "adapter_model.safetensors").read_bytes() == (store.root / "adapters" / f"{recipe}_1" / "adapter_model.safetensors").read_bytes()
    assert solver.measured == [(GOAL, 32, 1001, 1024)] * 6 and measured == [64] * 6 and sorted(path.name for path in (store.root / "adapters").iterdir()) == sorted(models_of((1, 2)))
    # THE REPORT says A SMOKE RUN, reads two orders at one degree of freedom, and L5's other seeds are NOT THERE.
    report = summaries["ladder_l5_noise_report"]
    assert report["ok"] is True and report["inconclusive"] is False and report["models"] == list(models_of((1, 2))) and report["smoke"] == smoke and report["and_its_level"]["degrees_of_freedom"] == 1
    assert any(f"A SMOKE RUN: L5's smoke run reads INCONCLUSIVE ({', '.join(failed)}) and was read all the same; 2 of the setting's 4 orders are trained" in line for line in report["lines"])
    of_the_mean = report["can_this_run_be_read"]["mean_constant"]["its_numbers_are_the_mean_of_its_models_and_it_is_served"]
    assert (of_the_mean["is_the_mean"], of_the_mean["is_none_of_them"], of_the_mean["one_of_its_models_in_a_smoke_run"], of_the_mean["passes"]) == (True, False, True, True)
    assert any("IT IS ONE OF ITS MODELS: a smoke run's trainings of one optimizer step on the same rows are one training; kept all the same" in line and line.endswith(": PASS") for line in report["lines"])
    assert report["branches"]["the_spread_of_the_recipe_as_it_is"] == NOT_READ and report["beside"]["the_published_of_l5"]["1"] == {
        "there": False, "why": "ladder_l5_deep_smoke_seed1 does not hold " + str(["ladder_l5_report.done.json", "report_ladder_l5.json", "ladder_l5_train_published.done.json",
                                                                                 "ladder_l5_measure_published.done.json", "episodes_l4_reach_published_problems.jsonl",
                                                                                 "episodes_l4_reach_published.done.json"])}
    # A step of an order this run does not train is no step of it.
    with pytest.raises(RuntimeError, match="`constant_3` is none of this run's models"):
        ladder_l5_noise.STEPS["ladder_l5_noise_train_constant_3"](config)
    # WITHOUT THE VARIABLE every order is the setting's, and nothing is taken loosely.
    monkeypatch.delenv(SMOKE)
    assert ladder_l5_noise.smoke_orders() == 0 and CONFIG["ladder_loop"]["l5_noise"]["orders"] == [1, 2, 3, 4]


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_the_smoke_variable_is_the_smoke_stages_alone():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_noise"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l5_noise.STEPS) == STEPS and ladder_l5_noise.ENTRY_STAGE == "ladder_l5_noise" and len(STEPS) == 22
    # THE SMOKE STAGE: the same steps for the first two orders (fourteen), under the runs L5's smoke stage names (WITHOUT its smoke-only variable), a run of its own, and ITS smoke-only variable.
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l5_noise_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", SMOKE_STEPS[0]),
                                                                       *(pair for step in SMOKE_STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == smoke[0][2] for _, _, options in smoke) and len(SMOKE_STEPS) == 14 and all(step in STEPS for step in SMOKE_STEPS)
    of_l5 = entry.step_fields(entry.STAGES["ladder_l5_deep_smoke"][2])[2]["environment"]
    assert smoke[0][2]["environment"] == {**{name: value for name, value in of_l5.items() if name != ladder_l5.SMOKE_VARIABLE}, RUN: "ladder_l5_noise_smoke", SMOKE: "2"}
    assert smoke[0][2]["environment"][ladder_l5.RUN_VARIABLE] == "ladder_l5_deep_smoke" and ladder_l5.SMOKE_VARIABLE in of_l5
    # THE ORDER IS THE SETTING'S: each order seed in turn, trained and measured with the recipe as it is, then with the decay; then each recipe's averaged adapter; then the report.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings = whole["ladder_loop"]["l5_noise"]
    assert settings == {"orders": [1, 2, 3, 4], "recipes": ["constant", "decay"], "sampling": "reach", "attempts": 32, "published_seeds": [0, 1, 2]}
    assert [step for _, step, _ in steps if step and "_train_" in step] == [f"ladder_l5_noise_train_{recipe}_{order}" for order in settings["orders"] for recipe in settings["recipes"]] == list(TRAININGS)
    assert [step for _, step, _ in steps if step and "_average_" in step] == [f"ladder_l5_noise_average_{recipe}" for recipe in settings["recipes"]] == list(AVERAGES)
    own = ladder_l5_noise.the_settings(whole)
    assert (own.start, own.orders, own.recipes, own.sampling, own.attempts, own.seeds) == ("pre_r64", ORDERS, RECIPES, "reach", 32, (0, 1, 2)) and own.deep == ladder_l5.the_settings(whole)
    assert ladder_l5_noise.run_name(own, 0) == "ladder_l5_noise_pre_r64_seed0" and ladder_l5.search_directory(whole, own.deep).name == "ladder_l5_deep_pre_r64_seed0"
    assert [ladder_l5.own_run(whole, own.deep, seed) for seed in own.seeds] == [f"ladder_l5_deep_pre_r64_seed{seed}" for seed in (0, 1, 2)]
    assert (ladder_l5_noise.the_set("mean_decay", "reach"), ladder_l5_noise.average_marker("decay"), ladder_l5_noise.stored_published_file(2)) == (
        "l4_reach_mean_decay", "ladder_l5_noise_average_decay", "l5_noise_stored_published_seed2.jsonl")
    # ONE reader of its settings; its steps are its own, and only its two stages run them.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["ladder_loop"]["l5_noise"]' in path.read_text()] == ["gpu/ladder_l5_noise.py"]
    assert not set(ladder_l5_noise.STEPS) & (set(ladder_l5.STEPS) | set(ladder_l6.STEPS) | set(ladder_l4_rows.STEPS)) and all(name.startswith("ladder_l5_noise_") for name in ladder_l5_noise.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l5_noise.STEPS for step in entry.STAGES[name])) == ["ladder_l5_noise", "ladder_l5_noise_smoke"]
    # ITS VARIABLES are named by its smoke stage alone, and no other module reads them.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(RUN) == names(SMOKE) == ["ladder_l5_noise_smoke"] and SMOKE == "RLVR_LEAN_LADDER_L5_NOISE_SMOKE_ORDERS" and RUN == "RLVR_LEAN_LADDER_L5_NOISE_RUN"
    assert sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if SMOKE in path.read_text()) == ["gpu/ladder_l5_noise.py", "runner/entry.py"]
    # ONE BODY EACH: its trainings are L4t's own training body, its measurements the ceiling's own measuring body with no rung, and L5's run is read by L5's own reading.
    assert ladder_l5_noise.train_a_model is ladder_l4_rows.train_a_model and ladder_l5_noise.measure_model is ladder_ceiling.measure_model
    own_stage = ladder_l5_noise.THE_STAGE
    assert (own_stage.name, own_stage.steps, own_stage.files, own_stage.reader, own_stage.prepare, own_stage.start_said_by) == (
        "ladder_l5_noise", "ladder_l5_noise", "l5_noise", "L5n", "ladder_l5_noise_prepare", "ladder_loop.l5.start")
    source = (PACKAGE / "gpu" / "ladder_l5_noise.py").read_text()
    assert "ladder_l5.read_the_search(" in source and "ladder_l5.read_for_the_search(" in source and "rungs=False" in source and "_episodes(" not in source and "LoRARequest" not in source
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l5_noise.STEPS" in main and "ladder_l5_noise," in main
    # THE DECAY IS AN OPTION OF THE TRAINING THAT IS OFF: each of the three functions it goes through takes it with a default that asks for nothing.
    import inspect

    for function, off in ((ladder_dose._train_with_readings, {"decay": False, "learning_rates": False}), (ladder_ceiling.one_pass, {"decay": False, "learning_rates": False}),
                          (ladder_l4_rows.train_a_model, {"schedule": None})):
        parameters = inspect.signature(function).parameters
        assert {name: parameters[name].default for name in off} == off


# ------------------------------------------------------------------- the stages that existed before this one
def test_l5_and_l6_write_what_they_wrote_with_this_stage_run_beside_them(noise, monkeypatch):  # noqa: F811
    config, deep_, first = noise.config, noise.deep, noise.first
    as_it_wrote = _all(noise.l5)
    # L5's run is L5's OWN PATH at the search's seed: its files and steps are what `test_ladder_l5_stage.py` holds them to; nothing of it names this stage.
    assert sorted(name for name in as_it_wrote if not name.startswith("episodes_")) == L5_FILES and list(first) == L5_STEPS
    assert not [name for name in as_it_wrote if "l5_noise" in name] and "l5_noise" not in json.dumps(first, default=str) and not [line for line in first["ladder_l5_report"]["lines"] if "L5n" in line]
    # A TRAINING OF L5 ASKS THE PASS FOR NOTHING MORE THAN IT DID: no decay, no learning rate recorded, in its call, its summary or its loss file.
    assert all(set(call["more"]) == {"start"} for call in deep_.calls[-3:]) and len(deep_.calls) >= 3
    for model in ("published", "shallow", "deep"):
        train, loss = first[f"ladder_l5_train_{model}"], json.loads((noise.l5 / f"l5_loss_{model}.json").read_text())
        assert "learning_rate" not in train and "learning_rates" not in train and all("learning_rate" not in row for row in loss["training_steps"])
        assert train["order"] == "a content hash of the task's seed and each row's id (`assembly.training_order`)"
    _stage5n(config, monkeypatch)
    root = noise.store().root
    own = _all(root)
    # L5's own task, queued again AFTER this stage ran in the same store: every step returns what is stored, nothing is trained or sampled, no file of either run moves.
    trained, asked = len(noise.calls), len(deep_.solver.asked)
    again = _stage5(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in first.items() if "report" not in name}
    assert again["ladder_l5_report"]["lines"] == first["ladder_l5_report"]["lines"] and again["ladder_l5_report"]["branches"] == first["ladder_l5_report"]["branches"]
    assert (len(noise.calls), len(deep_.solver.asked)) == (trained, asked) and _all(noise.l5) == as_it_wrote and _all(root) == own
    # A FURTHER SEED OF L5, run AFTER this stage: it reads the search's run as it always did and writes its own files, and no other: this stage's run is nothing to it.
    monkeypatch.setenv(SEEDS, "1")
    further = _stage5(config, monkeypatch)
    of_the_seed = ladder_l5._store(config, ladder_l5.the_settings(config)).root
    assert of_the_seed.name == "ladder_l5_deep_pre_r64_seed1" and list(further) == L5_STEPS and all(further[step]["run"] is False for step in NOT_MADE_AGAIN)
    assert sorted(name for name in _all(of_the_seed) if not name.startswith("episodes_")) == sorted({*L5_FILES, FURTHER_FILE} - {f"{step}.done.json" for step in NOT_MADE_AGAIN})
    assert all(set(call["more"]) == {"start"} for call in noise.calls[trained:]) and len(noise.calls) == trained + 3 and "l5_noise" not in json.dumps(further, default=str)
    assert all("learning_rate" not in further[f"ladder_l5_train_{model}"] for model in ("published", "shallow", "deep")) and _all(noise.l5) == as_it_wrote and _all(root) == own
    monkeypatch.delenv(SEEDS)
    # L5's names are its own as they were, and a seed's run directory is named as it was for the task's own seed.
    of_l5 = ladder_l5.THE_STAGE
    assert (of_l5.name, of_l5.steps, of_l5.files, of_l5.reader, of_l5.prepare, of_l5.start_said_by) == ("ladder_l5_deep", "ladder_l5", "l5", "L5", "ladder_l5_prepare", "ladder_loop.l5.start")
    settings = ladder_l5.the_settings(config)
    assert ladder_l5.own_run(config, settings) == "ladder_l5_deep_pre_r64_seed0" == ladder_l5.own_run(config, settings, 0) and ladder_l5.own_run(config, settings, 2) == "ladder_l5_deep_pre_r64_seed2"
    monkeypatch.setenv(SEEDS, "1")
    assert ladder_l5.own_run(config, settings) == "ladder_l5_deep_pre_r64_seed1" and ladder_l5.own_run(config, settings, 0) == "ladder_l5_deep_pre_r64_seed0"
    monkeypatch.setenv(ladder_l5.RUN_VARIABLE, "ladder_l5_deep_smoke")
    assert (ladder_l5.own_run(config, settings), ladder_l5.own_run(config, settings, 0), ladder_l5.own_run(config, settings, 2)) == ("ladder_l5_deep_smoke_seed1", "ladder_l5_deep_smoke", "ladder_l5_deep_smoke_seed2")
    monkeypatch.delenv(SEEDS)
    assert ladder_l5.own_run(config, settings) == "ladder_l5_deep_smoke"


def test_l6_writes_what_it_wrote_and_asks_the_pass_for_nothing_more(cycle, monkeypatch):  # noqa: F811
    # L6 IN ITS OWN WORLD, with the training's two new options in the code: its seventeen steps, its own files and no other; each of its four trainings hands the pass the
    # start adapter and NOTHING MORE (no decay, no learning rate recorded), and neither its summaries nor its loss files hold a learning rate.
    summaries = _stage6(cycle.config, monkeypatch)
    root = cycle.store().root
    start = cycle.deep.rank / "adapters" / "pre_r64"
    assert list(summaries) == L6_STEPS and sorted(name for name in _all(root) if not name.startswith("episodes_")) == L6_FILES and not [name for name in _all(root) if "l5_noise" in name]
    assert [call["more"] for call in cycle.calls] == [{"start": start}] * 4 and "l5_noise" not in json.dumps(summaries, default=str)
    for model in L6_MODELS:
        train, loss = summaries[f"ladder_l6_train_{model}"], json.loads((root / f"l6_loss_{model}.json").read_text())
        assert "learning_rate" not in train and "learning_rates" not in train and all("learning_rate" not in row for row in loss["training_steps"])
        assert train["order"] == "a content hash of the task's seed and each row's id (`assembly.training_order`)" and list(summaries[f"ladder_l6_measure_{model}"]["goal"]) == ["reach", "more"]
        assert "rungs" in summaries[f"ladder_l6_measure_{model}"]
