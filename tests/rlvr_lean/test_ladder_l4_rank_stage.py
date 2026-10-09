"""L4r, the stage end to end on what an L1 run, two L2 runs and the pretraining stored: `pre`'s pass again at another
adapter rank (the refusals of its prepare step, the ONE change held, the rank reaching the training, the adapter saved
read back and held to the check, an out-of-memory failure that fails the step and names the fallback's stage, the model
server's largest adapter rank raised for the measurement and for no other stage, the report against `pre`'s stored rows,
a set Lean did not answer sampled again); the pretraining stage left as it was; the fallback; the smoke stage; the
registration. Spec: docs/spec/ladder-loop.spec.md, "L4r: is the pretrained model capped by the size of its
adapter?". The engine and Lean are scripted; nothing touches a GPU or the network. The rules and the report on hand-made
rows are `test_ladder_l4_rank_report.py`."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adapter_files import _of, _write  # noqa: E402 - a weights file written by hand, from the format's definition
from test_ladder_assembly import arm  # noqa: E402, F401
from test_ladder_l2_stage import loop  # noqa: E402, F401
from test_ladder_l4_stage import FIXTURE, _file_rows, _files, _on_the_gpu, l4  # noqa: E402, F401 - L4's world: the fixtures, the stored runs, the 12-row file
from test_ladder_round import ScriptedLean, _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.dose import run_dose  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rank import BRANCHES, INCONCLUSIVE  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import (  # noqa: E402
    ladder_ceiling,
    ladder_dose,
    ladder_l2,
    ladder_l3d1,
    ladder_l3d2,
    ladder_l4,
    ladder_l4_rank,
    ladder_loop,
    ladder_round,
    milestone2,
    pipeline,
)
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
SAY = "l4 (pretrained on published proofs): "
CHECK, RUN = ladder_l4_rank.CHECK_VARIABLE, ladder_l4_rank.RUN_VARIABLE
SEEDS = "RLVR_LEAN_TRAINING_SEEDS"          # the task's seed (`--seeds`), as the entry shim hands it to every step
# The scripted adapter: two modules, whose matrices A are (rank x inputs) and B (outputs x rank). One more unit of rank is 6 + 10 + 5 + 4 numbers.
MODULES = (("q_proj", 6, 10), ("down_proj", 5, 4))
A_UNIT_OF_RANK = 25


class OutOfMemoryError(RuntimeError):
    """What PyTorch raises when an allocation would pass the cap (`torch.OutOfMemoryError`): read by its name."""


def _adapter(directory, rank, alpha, config_rank=None):
    """An adapter's directory as PEFT saves one: the weights file (its header written by hand) and `adapter_config.json`."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors = []
    for name, inputs, outputs in MODULES:
        tensors += [(f"base_model.model.layers.0.{name}.lora_A.weight", "F32", (rank, inputs), _of("F32", [[0.5] * inputs] * rank)),
                    (f"base_model.model.layers.0.{name}.lora_B.weight", "F32", (outputs, rank), _of("F32", [[0.25] * rank] * outputs))]
    _write(directory / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    (directory / "adapter_config.json").write_text(json.dumps({"r": rank if config_rank is None else config_rank, "lora_alpha": alpha, "peft_type": "LORA"}))


def _trainings(monkeypatch, calls, saves=None, fails=None):
    """The training's GPU path with the model taken out: `_train_with_readings` is the schedule alone (stand-in losses).
    AS THE REAL ONE DOES, it attaches the adapter the config it is HANDED asks for (`config["lora"]`): each adapter it
    saves is a weights file of that rank with that alpha beside it. `saves`: another (rank, alpha, rank in the config
    file) to save than the config's (a training the change did not reach). `fails`: an error to raise before any step."""
    def train_with_readings(config, examples, sample, pairs, schedule, seed, directory, tensorboard_run, orders=None, per_example=False, positions=False, **more):
        calls.append({"lora": dict(config["lora"]), "max_lora_rank": config["vllm"]["max_lora_rank"], "examples": examples, "schedule": schedule, "seed": seed,
                      "directory": directory, "tensorboard_run": tensorboard_run, "orders": orders, "more": more,
                      "training": {key: config["training"][key] for key in ("learning_rate", "effective_batch", "max_sequence_tokens", "warmup_steps")}})
        if fails is not None:
            raise fails
        train_step, read, _ = ladder_dose._stand_in_calls(len(examples), [], [])
        rank, alpha, in_the_file = saves or (config["lora"]["rank"], config["lora"]["alpha"], None)
        return {**run_dose(len(examples), config["training"]["effective_batch"], schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"],
                           train_step, read, lambda name, step_number: _adapter(directory / name, rank, alpha, in_the_file), orders=orders, per_example=per_example,
                           positions=positions), "target_format": "native", "seconds": 1.0, "peak_allocated_gb": 12.3}

    monkeypatch.setattr(ladder_dose, "_train_with_readings", train_with_readings)
    monkeypatch.setattr(pipeline, "_cap_torch_memory", lambda config: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0,
                                                                                   max_memory_reserved=lambda: 12_640_000_000)))


def _steps(stage_name):
    return [step for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]) if environment == "gpu" and step != "fix_tokenizers"]


def _pretrain(l4, monkeypatch, calls):  # noqa: F811
    """What the check stands on: the stored runs, then the pretraining stage, its training by the GPU path (so that `pre`'s adapter is a file on disk)."""
    l4.stored()
    _trainings(monkeypatch, calls)
    summaries = {}
    for step in _steps("ladder_l4_pretrain"):
        summaries[step] = _on_the_gpu(monkeypatch, ladder_l4.STEPS[step], l4.config) if step == ladder_l4.TRAIN else ladder_l4.STEPS[step](l4.config)
    return summaries


def _check(config, monkeypatch, until=None, stage_name="ladder_l4_rank"):
    """The check's steps in the stage's order, the training by its GPU path."""
    summaries = {}
    for step in _steps(stage_name):
        run = ladder_l4_rank.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step == ladder_l4_rank.TRAIN else run(config)
        if step == until:
            break
    return summaries


@pytest.fixture
def rank(l4, monkeypatch):  # noqa: F811
    """L4's world with nothing run yet and the stage's own variables as a task of `ladder_l4_rank` has them."""
    for name in (RUN, SEEDS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(CHECK, "rank")
    check = lambda: ladder_l4_rank.the_check(l4.config)      # noqa: E731
    return SimpleNamespace(config=l4.config, l4=l4, check=check, store=lambda: ladder_l4_rank._store(l4.config, check()), pre=lambda: l4.pretrain_store().root)


# ------------------------------------------------------------------------------------------------ the stage
def test_the_check_is_pres_pass_again_at_rank_64_measured_as_pre_was_and_read_against_pres_stored_rows(rank, monkeypatch):
    config, calls, kits = rank.config, [], []
    of_pre = _pretrain(rank.l4, monkeypatch, calls)
    pre, check = rank.pre(), rank.check()
    assert (check.which, check.rank, check.alpha, check.name, check.stage, check.in_the_place_of, check.then) == ("rank", 64, 128, "pre_r64", "ladder_l4_rank", None, (32, 64))
    read_only = {path: _files(path) for path in (pre, ladder_l4.source_directory(config), *ladder_ceiling.stored_directories(config, ladder_l4.STORED_VARIABLE, ladder_l4.SETTING).values())}
    of_the_adapter = _files(pre / "adapters" / "pre")
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append((self.enable_lora, adapter, given["vllm"]["max_lora_rank"], given["lora"]["rank"])), (rank.l4.solver, stand_in_parameters))[1]))
    del calls[:]
    summaries = _check(config, monkeypatch)
    store = rank.store()
    assert store.root.name == "ladder_l4_pretrain_r64_seed0" and store.root.parent == pre.parent and list(summaries) == list(ladder_l4_rank.STEPS) == [
        "ladder_l4_rank_prepare", "ladder_l4_rank_train", "ladder_l4_rank_measure", "ladder_l4_rank_report"]
    # `pre`'s run and the three it read were only READ, and `pre`'s adapter was not touched.
    assert {path: _files(path) for path in read_only} == read_only and _files(pre / "adapters" / "pre") == of_the_adapter
    assert config["lora"]["rank"] == 16 and config["vllm"]["max_lora_rank"] == 16                 # the config itself was not changed

    # ---- prepare: what the pretraining's prepare step records, with the check's rank and alpha in the recipe, and the check's own entry
    prepare, pre_prepare = summaries["ladder_l4_rank_prepare"], of_pre["ladder_l4_pretrain_prepare"]
    own = prepare["rank_check"]
    assert list(prepare) == [key if key != "minimums" else "rank_check" for key in pre_prepare]   # the pretraining's summary, the check's entry where its minimums stand
    assert prepare["recipe"] == {**pre_prepare["recipe"], "lora": {**config["lora"], "rank": 64, "alpha": 128}} and pre_prepare["recipe"]["lora"] == config["lora"]
    assert {key: value for key, value in prepare.items() if key not in ("recipe", "rank_check")} == {
        key: value for key, value in pre_prepare.items() if key not in ("recipe", "minimums")}    # the same file, rows, order, steps, seeds, samplings and stored runs
    assert (own["check"], own["stage"], own["model"], own["rank"], own["alpha"], own["in_the_place_of_rank"]) == ("rank", "ladder_l4_rank", "pre_r64", 64, 128, None)
    assert (own["max_lora_rank"], own["max_lora_rank_of_every_other_stage"]) == (64, 16) and own["never_solved_file"] == "l4_rank_never_solved.jsonl"
    # The adapter's trained numbers, from the shapes of `pre`'s stored adapter: rank x (inputs + outputs) over its modules.
    assert (own["trained_parameters"], own["trained_parameters_of_pre"], own["modules"]) == (64 * A_UNIT_OF_RANK, 16 * A_UNIT_OF_RANK, 2) and "adapters/pre/adapter_model.safetensors" in own["counted_from"]
    change = own["the_one_change"]
    assert (change["rank"], change["alpha"], change["alpha_over_rank"]) == ({"pre": 16, "check": 64}, {"pre": 32, "check": 128}, 2.0)
    assert "pretraining_file_sha256" in change["held_the_same"] and "recipe.learning_rate" in change["held_the_same"] and "recipe.lora.target_modules" in change["held_the_same"]
    assert own["pre"] == {"run": str(pre), "seed": 0, "rows": 12, "parts": ["rungs", REACH, MORE], "report_checks_pass": True, "rank": 16, "alpha": 32}
    # What its report reads of `pre` is copied into its own run: `pre`'s rows on each set, what it wrote, its rows' losses.
    for part in ("rungs", REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("pre", part, "l4")) == _file_rows(pre / f"episodes_l4_{part}_pre_problems.jsonl")
    assert json.loads(store.path(ladder_l4.PRE_FILE).read_text())["distinct_attempts"] == of_pre["ladder_l4_pretrain_measure"]["distinct_attempts"]
    of_pre_loss = json.loads((pre / ladder_l4.LOSS_FILE).read_text())
    copied = json.loads(store.path(ladder_l4_rank.PRE_LOSS_FILE).read_text())
    assert copied["row_losses"] == of_pre_loss["row_losses"] and (copied["model"], copied["run"], copied["rows"], copied["label"]) == ("pre", str(pre), 12, "pretrained on published proofs")
    rows = _file_rows(FIXTURE)
    assert not any(row["proof"].strip() in content.decode(errors="ignore") for row in rows for content in _files(store.root).values())      # no published proof is copied into the run

    # ---- train: `pre`'s pass (from the base, one pass, the file's order, the task's seed, the same recipe) with the check's rank and alpha
    (call,), train = calls, summaries["ladder_l4_rank_train"]
    assert call["lora"] == {**config["lora"], "rank": 64, "alpha": 128} and call["more"] == {}                                # FROM THE BASE: no start adapter
    assert call["examples"] == [{"problem_id": row["problem_id"], "side": "statement", "theorem": row["statement"], "completion": row["proof"]} for row in rows]
    assert call["orders"] == [list(range(12))] and call["schedule"] == {"passes": 1, "reading_steps": [], "checkpoint_steps": {"pre_r64": 2}}
    assert (call["seed"], call["tensorboard_run"], call["directory"]) == (0, "ladder_l4_pretrain_r64_seed0", store.root / "adapters")
    assert call["training"] == {"learning_rate": 1.0e-4, "effective_batch": 8, "max_sequence_tokens": 2048, "warmup_steps": 5}
    adapter = store.root / "adapters" / "pre_r64"
    assert (train["model"], train["rows"], train["steps"], train["passes"], train["label"], train["adapter"]) == ("pre_r64", 12, 2, 1, "pretrained on published proofs", str(adapter))
    # What it SAVED was read back from its files and is the check's: the rank and alpha PEFT recorded, the rank of every matrix, the numbers.
    saved = train["adapter_saved"]
    assert (saved["rank"], saved["alpha"], saved["ranks"], saved["numbers"], saved["modules"], saved["tensors"]) == (64, 128, [64], own["trained_parameters"], 2, 4)
    assert (train["peak_allocated_gb"], train["peak_reserved_gb"]) == (12.3, 12.64)
    loss = json.loads(store.path(ladder_l4.LOSS_FILE).read_text())
    assert len(loss["row_losses"]) == 12 and loss["label"] == "pretrained on published proofs"
    assert list(train)[:list(train).index("adapter_saved")] == list(of_pre["ladder_l4_pretrain_train"])                       # the pretraining's summary, then the check's own two entries

    # ---- measure: as `pre` was, with `pre`'s sampling seeds, under a model server whose largest adapter rank is the check's
    measured, of_l2 = summaries["ladder_l4_rank_measure"], ladder_l2.sampling_seeds(config)
    assert (measured["stage"], measured["model"], measured["label"]) == ("l4", "pre_r64", "pretrained on published proofs")
    assert kits and all(kit == (True, None, 64, 64) for kit in kits)                                                          # (the stand-in serves no adapter)
    of_pre_measure = of_pre["ladder_l4_pretrain_measure"]
    for of, own_set, pre_set in ((measured["rungs"], "l4_rungs_pre_r64", "l4_rungs_pre"), (measured["goal"][REACH], "l4_reach_pre_r64", "l4_reach_pre"),
                                 (measured["goal"][MORE], "l4_more_pre_r64", "l4_more_pre")):
        stored_marker = json.loads((pre / f"episodes_{pre_set}.done.json").read_text())
        assert of["set"] == own_set and (of["sampling_seed"], of["episodes_each"], of["problems"]) == (stored_marker["sampling_seed"], stored_marker["episodes_each"], stored_marker["problems"])
    assert (measured["rungs"]["sampling_seed"], measured["goal"][REACH]["sampling_seed"], measured["goal"][MORE]["sampling_seed"]) == (of_l2["rungs"], of_l2["reach"], of_l2["control"])
    assert (of_pre_measure["rungs"]["episodes_each"], measured["goal"][REACH]["episodes_each"]) == (8, 32)
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {"l4_rungs_pre_r64", "l4_reach_pre_r64", "l4_more_pre_r64"}      # NO MAP: no set of the base map's problems
    assert not [name for name in _files(store.root) if "map" in name] and not store.is_done(ladder_l4.MAP)

    # ---- the report: the checks first, then the read against `pre`; every line says what the model is
    report = summaries["ladder_l4_rank_report"]
    assert report["stage"] == "l4" and report["label"] == "pretrained on published proofs" and report["ok"] is True and report["branch"]["name"] in BRANCHES
    assert all(line.startswith(SAY) for line in report["lines"]) and "pretrained on published proofs" in report["headline"] and "`pre_r64` at rank 64" in report["headline"]
    heads = [line[len(SAY):].split(",")[0].split(".")[0].split(":")[0] for line in report["lines"]]
    assert heads[:4] == ["L4r", "CHECK 1", "CHECK 2", "CHECK 3"]
    read = report["measured_and_not_read"] if report["inconclusive"] else report
    primary = read["primary"]
    goal = [row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE) if row["group"] == "goal"]
    assert primary["problems"] == len(goal) == 1 and primary["attempts_each"] == prepare["attempts_a_goal_problem"] == 32 + measured["goal"][MORE]["episodes_each"]
    by_problem = lambda name: sum(row["resolved"] for part in (REACH, MORE) for row in store.read_rows(name(part)) if row["problem_id"] in goal)      # noqa: E731
    assert primary["successes"] == by_problem(lambda part: f"episodes_l4_{part}_pre_r64_problems.jsonl")
    assert primary["successes_of_the_base"] == by_problem(lambda part: ladder_ceiling.stored_file("pre", part, "l4"))         # the side compared with is `pre`
    assert read["secondary"]["never_solved_before"]["given"] is False                                                         # no file of ids was given
    assert report["the_check"]["adapter_saved"] == saved and report["the_check"]["the_one_change"] == change and report["the_check"]["peak_reserved_gb"] == 12.64
    assert json.loads(store.path(ladder_l4_rank.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_l4_rank.REPORT) == {"stage": "l4", "label": "pretrained on published proofs", "check": "rank", "model": "pre_r64",
                                                         "headline": report["headline"], "branch": report["branch"], "ok": True}
    # THE ADAPTER IS KEPT: no step deletes it, and the report, built once and again, deletes nothing.
    assert report["adapter"] == {"kept": True, "what": report["adapter"]["what"], "directory": str(adapter), "there": True} and "KEPT until the read" in report["adapter"]["what"]
    assert ladder_l4_rank.ladder_l4_rank_report(config)["adapter"]["there"] is True and (adapter / "adapter_model.safetensors").exists()
    # Every file of the run says l4, but the one name the shared episode step reads; and nothing of it is in another run's directory.
    assert [name for name in _files(store.root) if "l4" not in name] == ["problems.jsonl"]
    assert not any(marker.startswith("ladder_l4_rank") for path in read_only for marker in _files(path))
    # A rerun returns what is stored: nothing is trained or sampled again.
    sent = ScriptedLean.submitted
    again = _check(config, monkeypatch)
    assert len(calls) == 1 and ScriptedLean.submitted == sent and {name: summary for name, summary in again.items() if "report" not in name} == {
        name: summary for name, summary in summaries.items() if "report" not in name}


def test_the_goal_problems_nothing_stored_has_ever_solved_are_read_from_a_file_given_in_the_run_directory(rank, monkeypatch):
    config = rank.config
    _pretrain(rank.l4, monkeypatch, [])
    summaries = _check(config, monkeypatch)
    store = rank.store()
    goal = [row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE) if row["group"] == "goal"]
    not_given = summaries["ladder_l4_rank_report"]
    read = lambda report: (report["measured_and_not_read"] if report["inconclusive"] else report)["secondary"]["never_solved_before"]      # noqa: E731
    assert read(not_given) == {"given": False, "what": read(not_given)["what"]} and "does not compute it" in read(not_given)["what"]
    # The ids are GIVEN: a file in the run directory, one `problem_id` a row. The report is built again and reads it.
    store.write_rows(ladder_l4_rank.NEVER_SOLVED_FILE, [{"problem_id": goal[0]}, {"problem_id": "not_a_goal_problem"}])
    given = read(ladder_l4_rank.ladder_l4_rank_report(config))
    on_g = sum(row["resolved"] for part in (REACH, MORE) for row in store.read_rows(f"episodes_l4_{part}_pre_r64_problems.jsonl"))
    assert (given["given"], given["ids"], given["goal_problems"], given["not_goal_problems_of_this_run"]) == (True, 2, 1, 1)
    assert given["solved"] == int(on_g > 0) and given["problem_ids"] == ([goal[0]] if on_g else [])


def test_the_report_can_be_built_again_from_what_the_task_sent_out_with_the_ids_given_beside(rank, monkeypatch, tmp_path):
    config, out = rank.config, tmp_path / "out" / "steps"
    _pretrain(rank.l4, monkeypatch, [])
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(out))                         # where a job runner collects a task's files from: what a pulled copy holds
    report = _check(config, monkeypatch)["ladder_l4_rank_report"]
    monkeypatch.delenv("RLVR_LEAN_STEP_DIR")
    assert not (out / "adapters").exists() and not [path for path in out.iterdir() if path.is_dir()]       # no adapter leaves the box
    # Another machine: no box's store, no `pre`'s run, no adapter. The pulled files are the run directory, and the list of ids is put beside them.
    monkeypatch.setattr(pipeline, "STORE", tmp_path / "another_machine")
    pulled = rank.store()
    assert not _files(pulled.root) and not ladder_l4_rank.pre_directory(config).exists()
    for path in out.iterdir():
        (pulled.root / path.name).write_bytes(path.read_bytes())
    again = ladder_l4_rank.ladder_l4_rank_report(config)
    assert {**again, "adapter": None} == {**report, "adapter": None} and again["adapter"]["there"] is False and report["adapter"]["there"] is True
    goal = [row["problem_id"] for row in pulled.read_rows(ladder_l4.GROUPS_FILE) if row["group"] == "goal"]
    pulled.write_rows(ladder_l4_rank.NEVER_SOLVED_FILE, [{"problem_id": problem_id} for problem_id in goal])
    with_the_ids = ladder_l4_rank.ladder_l4_rank_report(config)
    read = lambda built: (built["measured_and_not_read"] if built["inconclusive"] else built)      # noqa: E731
    assert read(with_the_ids)["secondary"]["never_solved_before"]["goal_problems"] == len(goal) == 1 and read(with_the_ids)["primary"] == read(report)["primary"]


# ------------------------------------------------------------------------------------------------- prepare
def test_the_prepare_step_refuses_before_anything_is_written(rank, monkeypatch, tmp_path):
    config, root = rank.config, rank.store().root

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l4_rank.ladder_l4_rank_prepare(config)
        assert root.name == "ladder_l4_pretrain_r64_seed0" and not _files(root)                    # nothing was written: no marker, no file

    # ---- a task that names no check, or one the setting does not have: these steps are run by their stages
    for value in (None, "64", "RANK"):
        monkeypatch.delenv(CHECK) if value is None else monkeypatch.setenv(CHECK, value)
        for step in ladder_l4_rank.STEPS.values():
            with pytest.raises(RuntimeError, match=f"{CHECK} is {value!r}: it names the check of ladder_loop.l4.rank_check a task runs, one of .'rank', 'fallback'.. The steps of "
                                                   "L4r are run by the stages `ladder_l4_rank` and `ladder_l4_rank_fallback`, which set it"):
                step(config)
    monkeypatch.setenv(CHECK, "rank")
    # ---- `pre`'s run is not there: refused, naming the task to run
    refused(r"ladder_l4_pretrain_seed0 does not hold .'report_ladder_l4_pretrain.json', 'ladder_l4_pretrain_prepare.done.json', 'ladder_l4_pretrain_measure.done.json', "
            r"'l4_pretrain_loss.json'.\. L4r is read against `pre`'s stored rows and its training record, which the task of stage `ladder_l4_pretrain` for seed 0 "
            r".`python -m rlvr_lean.runner.entry --stage ladder_l4_pretrain --seeds 0`; for the smoke run, stage `ladder_l4_pretrain_smoke`. wrote on this box. Run that task to its end first; "
            "nothing was written")
    _pretrain(rank.l4, monkeypatch, [])
    pre = rank.pre()
    as_stored = _files(pre)

    def with_pre(change, match, error=RuntimeError):
        """`pre`'s run with one thing of it changed, refused; then put back as it was."""
        change()
        refused(match, error)
        for name, content in as_stored.items():
            (pre / name).write_bytes(content)

    def rewritten(name, **changed):
        return lambda: (pre / name).write_text(json.dumps({**json.loads(as_stored[name]), **changed}))

    # ---- each thing of `pre`'s run the check reads: its report, the markers of its prepare and measure steps, its training record
    for name in ("report_ladder_l4_pretrain.json", "ladder_l4_pretrain_prepare.done.json", "ladder_l4_pretrain_measure.done.json", "l4_pretrain_loss.json"):
        with_pre((pre / name).unlink, f"ladder_l4_pretrain_seed0 does not hold .'{name}'.. L4r is read against `pre`'s stored rows and its training record")
    # ... and, for each set it was measured on, its per-problem rows and the set's own marker
    for part in ("rungs", REACH, MORE):
        for name in (f"episodes_l4_{part}_pre_problems.jsonl", f"episodes_l4_{part}_pre.done.json"):
            with_pre((pre / name).unlink, f"ladder_l4_pretrain_seed0 does not hold .'{name}'.: `pre`'s per-problem rows on a set it was measured on, or that set's own marker")
    prepared = json.loads(as_stored["ladder_l4_pretrain_prepare.done.json"])
    with_pre(rewritten("ladder_l4_pretrain_prepare.done.json", seed=3),
             "the pretraining run ladder_l4_pretrain_seed0 was made at seed 3 and ladder_loop.l4.pretraining_seed is 0: the check is read against the ONE pretraining")
    with_pre(rewritten("report_ladder_l4_pretrain.json", ok=False),
             "the report of the pretraining run ladder_l4_pretrain_seed0 is not to be read .Lean gave no verdict on too much of a set.: `pre`'s rows cannot stand on the other side")
    # ---- THE ONE CHANGE: anything else that is not `pre`'s is refused, and named
    with_pre(rewritten("ladder_l4_pretrain_prepare.done.json", pretraining_file_sha256="0" * 64, rows=13),
             r"the check is `pre`'s pass again with ONE change, the adapter's rank and alpha, and 2 other things are not `pre`'s: pretraining_file_sha256 is '[0-9a-f]{64}' here "
             r"and '0{64}' in `pre`'s run; rows is 12 here and 13 in `pre`'s run\. Nothing was written")
    with_pre(rewritten("ladder_l4_pretrain_prepare.done.json", recipe={**prepared["recipe"], "learning_rate": 2.0e-4}),
             r"and 1 other thing is not `pre`'s: recipe.learning_rate is 0.0001 here and 0.0002 in `pre`'s run\. Nothing was written")
    with_pre(rewritten("ladder_l4_pretrain_prepare.done.json", recipe={**prepared["recipe"], "lora": {**prepared["recipe"]["lora"], "target_modules": ["q_proj"]}}),
             "1 other thing is not `pre`'s: recipe.lora.target_modules is ")
    with_pre(rewritten("ladder_l4_pretrain_prepare.done.json", recipe={**prepared["recipe"], "lora": {**prepared["recipe"]["lora"], "rank": 64, "alpha": 128}}),
             "the check's rank is 64 and `pre`'s is 64: the check asks whether a LARGER adapter reaches further. Nothing was written")
    with_pre(rewritten("ladder_l4_pretrain_prepare.done.json", recipe={**prepared["recipe"], "lora": {**prepared["recipe"]["lora"], "alpha": 16}}),
             "the check's alpha / rank is 128 / 64 and `pre`'s is 16 / 16: the adapter's scale would be a second change. Nothing was written")
    # ---- `pre`'s stored rows must pair by problem with what this run will measure
    with_pre(rewritten("ladder_l4_pretrain_prepare.done.json", sampling_seeds={**prepared["sampling_seeds"], "more": 5}),
             "this run would measure `pre_r64` with .* and ladder_l4_pretrain_seed0 recorded .* for `pre`: the two models would not pair by problem. Nothing was written")
    marker = json.loads(as_stored["episodes_l4_more_pre.done.json"])
    with_pre(rewritten("episodes_l4_more_pre.done.json", sampling_seed=5),
             f"ladder_l4_pretrain_seed0 measured `pre` .more. with sampling seed 5 and {marker['episodes_each']} episodes; this config gives {marker['sampling_seed']} and "
             f"{marker['episodes_each']}: the check's model would not pair with the stored results")
    rows = [json.loads(line) for line in as_stored["episodes_l4_rungs_pre_problems.jsonl"].decode().splitlines()]
    with_pre(lambda: (pre / "episodes_l4_rungs_pre_problems.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:-1])),
             "ladder_l4_pretrain_seed0 measured `pre` .rungs. on other problems than L1's run holds for it: the check's model would not pair with it")
    unanswered = [{**rows[0], "attempts_without_an_answer": 8}, *rows[1:]]
    with_pre(lambda: (pre / "episodes_l4_rungs_pre_problems.jsonl").write_text("".join(json.dumps(row) + "\n" for row in unanswered)),
             "ladder_l4_pretrain_seed0 measured `pre` .rungs. with too many attempts without a verdict from Lean: that set is `pre`'s side of every comparison")
    losses = json.loads(as_stored["l4_pretrain_loss.json"])
    with_pre(rewritten("l4_pretrain_loss.json", row_losses=losses["row_losses"][:-1]),
             "ladder_l4_pretrain_seed0 stored the losses of 11 rows and this run trains on 12: `pre`'s training record is not of this file. Nothing was written")
    # ---- the check is `pre`'s pass again at the SAME seed: a task of another seed is refused before anything is read
    monkeypatch.setenv(SEEDS, "1")
    with pytest.raises(RuntimeError, match="this task's seed is 1 and `pre` is the pretraining of seed 0 .ladder_loop.l4.pretraining_seed.: the check is `pre`'s pass again, the "
                                           "SAME seed, with one change. Queue it with --seeds 0; nothing was written"):
        ladder_l4_rank.ladder_l4_rank_prepare(config)
    assert not _files(rank.store().root) and rank.store().root.name == "ladder_l4_pretrain_r64_seed1"
    monkeypatch.delenv(SEEDS)
    # ---- what the pretraining's own prepare step refuses is refused here too: the same file, the same refusals
    held = _file_rows(ladder_l4.source_directory(config) / "heldout_groups.jsonl")[0]["problem_id"]
    changed = _file_rows(FIXTURE)
    changed[7]["problem_id"] = held
    (tmp_path / "changed.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changed))
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(tmp_path / "changed.jsonl"))
    refused(f"1 rows of L4's pretraining file are held-out problems .first: {held}", ValueError)
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    source = ladder_l4.source_directory(config)
    source.rename(source.with_name("elsewhere"))
    refused("L4 reads the run directory the L1 task for seed 0 wrote on this box .stage `ladder_l1` with --seeds 0")
    source.with_name("elsewhere").rename(source)
    # ---- a setting that is no check: a rank that is not a number, a fallback that is not the smaller adapter
    settings = config["ladder_loop"]["l4"]["rank_check"]
    monkeypatch.setitem(settings, "fallback_rank", 64)
    refused("ladder_loop.l4.rank_check.fallback_rank is 64 and its rank is 64: the fallback is the SMALLER adapter", ValueError)
    monkeypatch.setitem(settings, "fallback_rank", 32)
    monkeypatch.setitem(settings, "alpha", 128.5)
    refused("ladder_loop.l4.rank_check gives the check `rank` rank 64 and alpha 128.5: each is a positive whole number", ValueError)
    monkeypatch.setitem(settings, "alpha", 128)
    # ---- and with everything as it was, it prepares; the later steps refuse out of their order
    for step, match in ((ladder_l4_rank.ladder_l4_rank_train, "the check's training needs the step ladder_l4_rank_prepare of this run, which is not done: the stage `ladder_l4_rank`"),
                        (ladder_l4_rank.ladder_l4_rank_measure, "the measurement of L4r's model `pre_r64` needs the step ladder_l4_rank_prepare"),
                        (ladder_l4_rank.ladder_l4_rank_report, "the check's report needs the step ladder_l4_rank_prepare")):
        with pytest.raises(RuntimeError, match=match):
            step(config)
    assert ladder_l4_rank.ladder_l4_rank_prepare(config)["rank_check"]["rank"] == 64 and _files(pre) == as_stored
    with pytest.raises(RuntimeError, match="the check's report needs the step ladder_l4_rank_train"):
        ladder_l4_rank.ladder_l4_rank_report(config)


def test_the_trained_numbers_are_not_counted_when_pres_adapter_is_not_a_weights_file_on_this_box(rank, monkeypatch):
    config = rank.config
    _pretrain(rank.l4, monkeypatch, [])
    adapter = rank.pre() / "adapters" / "pre" / "adapter_model.safetensors"
    for change in (lambda: adapter.write_text("an adapter"), lambda: adapter.write_bytes(b"short"), adapter.unlink):       # not a weights file; too short for one; not there
        change()
        rank.store().path(f"{ladder_l4_rank.PREPARE}.done.json").unlink(missing_ok=True)
        own = ladder_l4_rank.ladder_l4_rank_prepare(config)["rank_check"]
        assert (own["trained_parameters"], own["trained_parameters_of_pre"], own["counted_from"]) == (None, None, None)
        assert "could not be read as an adapter's weights file" in own["not_counted"] and "the training step counts the adapter it saves" in own["not_counted"]
    # The check does not need `pre`'s adapter: it trains from the base, and counts the adapter it saves.
    train = _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)
    assert train["adapter_saved"]["numbers"] == 64 * A_UNIT_OF_RANK


# --------------------------------------------------------------------------------------------------- train
def test_the_adapter_saved_is_read_back_and_held_to_the_check_before_the_step_is_marked_done(rank, monkeypatch):
    config, calls = rank.config, []
    _pretrain(rank.l4, monkeypatch, calls)
    ladder_l4_rank.ladder_l4_rank_prepare(config)
    store = rank.store()
    adapter = store.root / "adapters" / "pre_r64"
    # A training the ONE change did not reach: it saved `pre`'s rank again, or the check's rank with another alpha, or says one rank and holds another.
    for saves, match in (((16, 32, None), r"has rank 16 and alpha 32 in its adapter_config.json and matrices of rank .16.; the check `rank` is rank 64 and alpha 128. The ONE change "
                                          "did not reach the training: nothing of this run is measured, and the step is not marked done"),
                         ((64, 32, None), r"has rank 64 and alpha 32 in its adapter_config.json and matrices of rank .64.; the check `rank` is rank 64 and alpha 128"),
                         ((16, 128, 64), r"has rank 64 and alpha 128 in its adapter_config.json and matrices of rank .16.; the check `rank` is rank 64 and alpha 128")):
        _trainings(monkeypatch, calls, saves=saves)
        with pytest.raises(RuntimeError, match=match):
            _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)
        assert not store.is_done(ladder_l4_rank.TRAIN)
        with pytest.raises(RuntimeError, match="the measurement of L4r's model `pre_r64` needs the step ladder_l4_rank_train of this run, which is not done"):
            ladder_l4_rank.ladder_l4_rank_measure(config)
    # A TRAINING THAT DID NOT FINISH STARTS AGAIN FROM THE BASE: nothing of a pass is kept but its adapter, and no start adapter is handed to the next.
    _trainings(monkeypatch, calls)
    train = _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)
    assert train["adapter_saved"]["ranks"] == [64] and store.is_done(ladder_l4_rank.TRAIN) and json.loads((adapter / "adapter_config.json").read_text())["r"] == 64
    of_the_check = calls[1:]                                                                       # (the first call was the pretraining's own)
    assert len(of_the_check) == 4 and all(call["more"] == {} and call["lora"]["rank"] == 64 and call["orders"] == [list(range(12))] for call in of_the_check)
    # A training that is done is not made again.
    assert _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config) == train and len(calls) == 5
    # The stand-in trains nothing and saves nothing: nothing is read back, and that is said.
    store.path(f"{ladder_l4_rank.TRAIN}.done.json").unlink()
    stood_in = ladder_l4_rank.ladder_l4_rank_train(config)
    assert stood_in["adapter_saved"] is None and "peak_reserved_gb" not in stood_in and "made up" in stood_in["note"] and len(calls) == 5
    # A file that changed since the run was prepared is not trained on (the pretraining's own refusal).
    store.path(f"{ladder_l4_rank.TRAIN}.done.json").unlink()
    changed = store.root.parent / "changed.jsonl"
    changed.write_text(FIXTURE.read_text() + "\n")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    with pytest.raises(RuntimeError, match="L4's pretraining file has SHA-256 .* a run trains on the file its prepare step checked"):
        _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)
    assert len(calls) == 5


def test_an_out_of_memory_failure_fails_the_step_and_names_the_fallbacks_stage_and_nothing_smaller_is_tried_inside_the_task(rank, monkeypatch):
    config, calls = rank.config, []
    _pretrain(rank.l4, monkeypatch, calls)
    ladder_l4_rank.ladder_l4_rank_prepare(config)
    store = rank.store()
    del calls[:]
    out = OutOfMemoryError("CUDA out of memory. Tried to allocate 44.00 MiB. GPU 0 has a total capacity of 15.45 GiB\nof which 11.94 GiB is allowed")
    _trainings(monkeypatch, calls, fails=out)
    with pytest.raises(RuntimeError, match=r"the box cannot hold an adapter of rank 64 at the sequence limit: the training ran out of GPU memory .CUDA out of memory. Tried to "
                                           r"allocate 44.00 MiB. GPU 0 has a total capacity of 15.45 GiB.\. By the spec this run is VOID, nothing of it is read, and rank 32 with "
                                           r"alpha 64 is run in its place under the same read\. NO fallback is made inside this task: queue "
                                           r"stage `ladder_l4_rank_fallback` .`python -m rlvr_lean.runner.entry --stage ladder_l4_rank_fallback --seeds 0`: the model `pre_r32`, the run directory "
                                           r"ladder_l4_pretrain_r32_seed0., and say in the note which ran\. .If the memory was another process's") as raised:
        _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)
    assert raised.value.__cause__ is out and "No adapter was saved" in str(raised.value)
    # ONE training was tried, at the check's rank: nothing smaller inside the task. Nothing is marked, nothing was saved, and no other run directory was made.
    assert [call["lora"]["rank"] for call in calls] == [64] and not store.is_done(ladder_l4_rank.TRAIN) and not (store.root / "adapters").exists()
    assert not (store.root.parent / "ladder_l4_pretrain_r32_seed0").exists()
    # A CUDA error that says so is read the same way; any other error is raised as it is.
    _trainings(monkeypatch, calls, fails=RuntimeError("CUDA error: out of memory"))
    with pytest.raises(RuntimeError, match="the box cannot hold an adapter of rank 64 at the sequence limit"):
        _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)
    other = RuntimeError("the tokenizer dropped every space")
    _trainings(monkeypatch, calls, fails=other)
    with pytest.raises(RuntimeError, match="the tokenizer dropped every space") as raised:
        _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)
    assert raised.value is other and ladder_l4_rank.out_of_memory(out) and not ladder_l4_rank.out_of_memory(other)
    # ---- THE FALLBACK is a task of its own: the smaller rank, a model and a run directory of its own, the same steps and the same read
    monkeypatch.setenv(CHECK, "fallback")
    check = rank.check()
    assert (check.which, check.rank, check.alpha, check.name, check.stage, check.in_the_place_of, check.then) == ("fallback", 32, 64, "pre_r32", "ladder_l4_rank_fallback", 64, None)
    _trainings(monkeypatch, calls)
    del calls[:]
    summaries = _check(config, monkeypatch, stage_name="ladder_l4_rank_fallback")
    fallback = rank.store()
    assert fallback.root.name == "ladder_l4_pretrain_r32_seed0" and (fallback.root / "adapters" / "pre_r32" / "adapter_model.safetensors").exists()
    own = summaries["ladder_l4_rank_prepare"]["rank_check"]
    assert (own["check"], own["stage"], own["model"], own["rank"], own["alpha"], own["in_the_place_of_rank"], own["max_lora_rank"]) == (
        "fallback", "ladder_l4_rank_fallback", "pre_r32", 32, 64, 64, 32)
    assert [call["lora"]["rank"] for call in calls] == [32] and summaries["ladder_l4_rank_train"]["adapter_saved"]["numbers"] == 32 * A_UNIT_OF_RANK
    report = summaries["ladder_l4_rank_report"]
    assert "THIS IS THE FALLBACK: rank 32 is run in the place of rank 64, which the box could not hold (that run is VOID)" in report["lines"][0] and "`pre_r32` at rank 32" in report["headline"]
    assert {row["set"] for row in fallback.read_rows("problems.jsonl")} == {"l4_rungs_pre_r32", "l4_reach_pre_r32", "l4_more_pre_r32"}
    assert not store.is_done(ladder_l4_rank.TRAIN)                                                 # the VOID run is as it was left
    # The fallback out of memory too: no smaller rank is configured, and it says so.
    fallback.path(f"{ladder_l4_rank.TRAIN}.done.json").unlink()
    _trainings(monkeypatch, calls, fails=out)
    with pytest.raises(RuntimeError, match="the box cannot hold an adapter of rank 32 at the sequence limit either .the fallback of ladder_loop.l4.rank_check, run in the place of "
                                           "rank 64.: the training ran out of GPU memory .* This run is VOID too, no smaller rank is configured"):
        _on_the_gpu(monkeypatch, ladder_l4_rank.ladder_l4_rank_train, config)


# ------------------------------------------------------------------------------------------------- measure
def test_the_model_servers_largest_adapter_rank_is_raised_for_the_checks_measurement_and_for_no_other_stage(rank, monkeypatch):
    config, kits = rank.config, []
    _pretrain(rank.l4, monkeypatch, [])
    _check(config, monkeypatch, until=ladder_l4_rank.TRAIN)
    store, pre = rank.store(), rank.pre()
    adapter = store.root / "adapters" / "pre_r64"
    served = lambda self, adapter=None: (lambda given: (kits.append({"lora": self.enable_lora, "adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"],      # noqa: E731
                                                                     "vllm": given["vllm"], "rank": given["lora"]["rank"]}), (rank.l4.solver, stand_in_parameters))[1])
    # ---- as on the GPU: the kept adapter served beside the base under a name of its own, by an engine started with the check's rank as its largest
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        patched.setattr(ladder_round.Engines, "kit", served)
        measured = ladder_l4_rank.ladder_l4_rank_measure(config)
        # Without its adapter the measurement cannot be made.
        store.path(f"{ladder_l4_rank.MEASURE}.done.json").unlink()
        adapter.rename(adapter.with_name("elsewhere"))
        with pytest.raises(RuntimeError, match="adapters/pre_r64 is not there: the adapter `pre_r64` is kept until the read, and one trained again would not be the model"):
            ladder_l4_rank.ladder_l4_rank_measure(config)
        adapter.with_name("elsewhere").rename(adapter)
        store.mark_done(ladder_l4_rank.MEASURE, measured)
    assert len(kits) == 3 and all(kit["lora"] is True and kit["adapter"] == ("ladder_l4_rank_pre_r64", 1, str(adapter)) and kit["max_lora_rank"] == 64 for kit in kits)
    # Nothing else of the model server's settings moved, and the config itself was not changed.
    assert all({**kit["vllm"], "max_lora_rank": 16} == config["vllm"] for kit in kits) and config["vllm"]["max_lora_rank"] == 16 and config["lora"]["rank"] == 16
    # ---- NO OTHER STAGE: the pretraining's own steps, run by a task that carries the check's variable, serve `pre` under the config's own limit
    del kits[:]
    monkeypatch.setattr(ladder_round.Engines, "kit", served)
    for marker in (ladder_l4.MEASURE, ladder_l4.MAP, "episodes_l4_rungs_pre", "episodes_l4_reach_pre", "episodes_l4_more_pre", "episodes_l4_map_pre"):
        (pre / f"{marker}.done.json").unlink()
    for path in pre.glob("episodes_l4_*_block_*.done.json"):
        path.unlink()
    ladder_l4.ladder_l4_pretrain_measure(config)
    ladder_l4.ladder_l4_pretrain_map(config)
    assert len(kits) == 4 and all(kit["max_lora_rank"] == 16 and kit["rank"] == 16 and kit["vllm"] is config["vllm"] for kit in kits)
    # ... and where a rank is put into a config is ONE function of ONE module: no other step of the package writes the model server's limit.
    writers = sorted(path.name for path in (PACKAGE / "gpu").glob("*.py") if re.search(r"[\"']max_lora_rank[\"']\s*:", path.read_text()))
    readers = sorted(path.name for path in (PACKAGE / "gpu").glob("*.py") if "max_lora_rank" in path.read_text())
    assert writers == ["ladder_l4_rank.py"] and readers == ["ladder_l4_rank.py", "milestone2.py", "pipeline.py"]
    # (A stage that starts from this check's model calls it ONCE, in the module L4t and L4b's arm both ask: `ladder_l4_start.config_from`; test_ladder_l4_rows_r64.py)
    callers = {path.name: path.read_text().count("config_of(") for path in (PACKAGE / "gpu").glob("*.py") if "config_of(" in path.read_text()}
    assert sorted(callers) == ["ladder_l4_rank.py", "ladder_l4_start.py"] and callers["ladder_l4_start.py"] == 1
    # The setting has ONE reader in the package (`a_check`), and the pretraining stage's module names neither the setting nor the check's variable.
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["l4"]["rank_check"]' in path.read_text()] == ["gpu/ladder_l4_rank.py"]
    assert (PACKAGE / "gpu" / "ladder_l4_rank.py").read_text().count('["l4"]["rank_check"]') == 1
    assert not [word for word in ("rank_check", CHECK, "max_lora_rank") if word in (PACKAGE / "gpu" / "ladder_l4.py").read_text()]
    of_the_check = ladder_l4_rank.config_of(config, rank.check())
    assert {key: value for key, value in of_the_check.items() if key not in ("lora", "vllm")} == {key: value for key, value in config.items() if key not in ("lora", "vllm")}
    assert of_the_check["lora"] == {**config["lora"], "rank": 64, "alpha": 128} and of_the_check["vllm"] == {**config["vllm"], "max_lora_rank": 64}


def test_a_set_lean_did_not_answer_fails_the_report_and_is_sampled_again_from_the_kept_adapter_when_the_task_is_queued_again(rank, monkeypatch, tmp_path):
    config, calls = rank.config, []
    _pretrain(rank.l4, monkeypatch, calls)
    first = _check(config, monkeypatch, until=ladder_l4_rank.MEASURE)
    store = rank.store()
    adapter = store.root / "adapters" / "pre_r64"
    # The pool was in trouble while one set was measured: Lean gave no verdict on a fifth of its attempts.
    unanswered = "l4_more_pre_r64"
    name = f"episodes_{unanswered}_problems.jsonl"
    as_measured, rows = store.path(name).read_bytes(), store.read_rows(name)
    rows[0]["attempts_without_an_answer"] = 20
    store.write_rows(name, rows)
    # ---- the report is written, its third check FAILS, it is INCONCLUSIVE and not to be read, and the step fails; the adapter is kept
    monkeypatch.setattr(milestone2, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", ladder_l4_rank.REPORT, "--out", str(tmp_path / "report.json")])
    assert gpu_main.main() != 0
    report = json.loads(store.path(ladder_l4_rank.REPORT_FILE).read_text())
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_pre_r64_2"] and report["inconclusive"] is True and report["branch"]["name"] == INCONCLUSIVE
    assert report["can_this_run_see_a_win"]["lean_answered"]["passes"] is False and report["branch"]["failed_checks"][-1] == "lean_answered"
    assert report["primary"] is None and report["secondary"] is None and "NOT TO BE READ" in report["lines"][-1] and report["lines"][3].endswith("FAIL")
    assert report["adapter"]["there"] is True and adapter.is_dir()
    # ---- the task queued again: that set, and no other, is sampled again from the kept adapter; nothing is trained
    untouched = {key: content for key, content in _files(store.root).items() if unanswered not in key and "report" not in key and "rank_measure" not in key}
    sent, trained = ScriptedLean.submitted, len(calls)
    again = _check(config, monkeypatch)
    assert len(calls) == trained and again["ladder_l4_rank_measure"]["measured_again"] == [unanswered]
    assert ScriptedLean.submitted - sent == first["ladder_l4_rank_measure"]["goal"][MORE]["pipeline"]["sent_to_lean"]         # that one set's checks, and no other's
    after = _files(store.root)
    assert all(after[key] == content for key, content in untouched.items()) and after[name] == as_measured                    # the stand-in samples the same again
    readable = again["ladder_l4_rank_report"]
    assert readable["ok"] is True and readable["not_to_be_read"] == [] and readable["can_this_run_see_a_win"]["lean_answered"]["passes"] is True and adapter.is_dir()


# ------------------------------------------------------------------- the pretraining stage is what it was
PREPARE_KEYS = ["stage", "label", "seed", "source_run", "stored_runs", "loop_arm", "loop_target_rate", "stand_in_engine", "ceiling", "pretraining_file",
                "pretraining_file_sha256", "rows", "half", "half_seed", "rows_by_kind", "rows_by_proof_lines", "tokens", "longest_example_tokens", "tokens_counted_by",
                "max_sequence_tokens", "order", "effective_batch", "steps", "recipe", "minimums", "goal_set", "rungs", "rung_problems", "sampling_seeds", "rung_episodes",
                "goal_samplings", "attempts_a_goal_problem", "stored_measurements", "l1_contradicted_side_setting", "contradicted_side_setting"]
TRAIN_KEYS = ["stage", "label", "model", "seed", "rows", "passes", "order", "stand_in_engine", "adapter", "first_step_loss", "last_step_loss", "mean_loss_over_the_first_rows",
              "mean_loss_over_the_last_rows", "rows_compared", "the_training_took"]
PRETRAINING_FILES = [
    "adapters/pre/adapter_config.json", "adapters/pre/adapter_model.safetensors", "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", "l4_map_pre.jsonl", "l4_map_problems.json",
    "l4_pretrain_loss.json", "l4_pretraining_rows.jsonl", "l4_stored_base_more.jsonl", "l4_stored_base_reach.jsonl", "l4_stored_base_rungs.jsonl", "l4_stored_loop_more.jsonl",
    "l4_stored_loop_reach.jsonl", "l4_stored_loop_rungs.jsonl", "l4_stored_models.json", "ladder_l4_pretrain_map.done.json", "ladder_l4_pretrain_measure.done.json",
    "ladder_l4_pretrain_prepare.done.json", "ladder_l4_pretrain_report.done.json", "ladder_l4_pretrain_train.done.json", "problems.jsonl", "report_ladder_l4_pretrain.json"]
TIMED = re.compile(r"seconds|per_second")


def _plain(value, store):
    """A stored value with what differs from run to run taken out: wall-clock figures, and the store's own path."""
    if isinstance(value, dict):
        return {key: ("<t>" if TIMED.search(key) and isinstance(item, (int, float)) else _plain(item, store)) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item, store) for item in value]
    return value.replace(store, "<store>") if isinstance(value, str) else value


def _written(directory):
    """Every file of a run directory as it was written, key order included, by its path under the directory."""
    store, written = str(pipeline.STORE), {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            text = path.read_bytes().decode(errors="replace")
            parsed = ([json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text) if path.suffix == ".json" else text)
            written[str(path.relative_to(directory))] = json.dumps(_plain(parsed, store), ensure_ascii=False)
    return written


def test_the_pretraining_stage_writes_what_it_wrote_whatever_the_checks_variable_says_and_after_a_check_ran(rank, monkeypatch, tmp_path):
    config, calls = rank.config, []
    monkeypatch.delenv(CHECK)
    # ---- a task of the pretraining stage as it always was: no variable of the check
    plain = _pretrain(rank.l4, monkeypatch, calls)
    pre = rank.pre()
    as_it_wrote = _written(pre)
    episodes = [name for name in as_it_wrote if name.startswith("episodes_")]
    assert sorted(name for name in as_it_wrote if name not in episodes) == PRETRAINING_FILES and pre.name == "ladder_l4_pretrain_seed0"
    assert list(plain["ladder_l4_pretrain_prepare"]) == PREPARE_KEYS and list(plain["ladder_l4_pretrain_train"])[:len(TRAIN_KEYS)] == TRAIN_KEYS
    assert "adapter_saved" not in plain["ladder_l4_pretrain_train"] and "rank_check" not in plain["ladder_l4_pretrain_prepare"]
    assert (calls[0]["lora"], calls[0]["max_lora_rank"], calls[0]["tensorboard_run"], calls[0]["schedule"]["checkpoint_steps"]) == (
        config["lora"], 16, "ladder_l4_pretrain_seed0", {"pre": 2}) and config["lora"]["rank"] == 16
    assert json.loads((pre / "adapters" / "pre" / "adapter_config.json").read_text())["r"] == 16
    # ---- the check ran beside it: `pre`'s run is, file for file, what the pretraining wrote
    monkeypatch.setenv(CHECK, "rank")
    _check(config, monkeypatch)
    assert _written(pre) == as_it_wrote
    # ---- the pretraining stage AGAIN, in another store, by a task that carries the check's variable: the same files, the same contents, rank 16
    monkeypatch.setattr(pipeline, "STORE", tmp_path / "another_store")
    del calls[:]
    with_the_variable = _pretrain(rank.l4, monkeypatch, calls)
    again = rank.pre()
    assert again != pre and again.name == "ladder_l4_pretrain_seed0" and _written(again) == as_it_wrote
    assert list(with_the_variable["ladder_l4_pretrain_prepare"]) == PREPARE_KEYS and (calls[0]["lora"], calls[0]["max_lora_rank"]) == (config["lora"], 16)
    assert not [path.name for path in again.parent.iterdir() if "_r64_" in path.name or "_r32_" in path.name]                 # and it made no run directory of a check


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_runs_at_the_real_rank_on_the_pretraining_smoke_runs_world(arm, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ladder_l2.L2_ARM_VARIABLE)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings(monkeypatch, calls)
    # ---- the pretraining's smoke task, which leaves `pre`'s smoke run on the box
    options = entry.step_fields(entry.STAGES["ladder_l4_pretrain_smoke"][2])[2]
    with monkeypatch.context() as patched:                                     # one task's environment: gone when the task is
        patched.delenv(ladder_loop.DATA_VARIABLE)
        for name, value in options["environment"].items():
            patched.setenv(name, value)
        for step in _steps("ladder_l4_pretrain_smoke"):
            _on_the_gpu(patched, ladder_l4.STEPS[step], config) if step == ladder_l4.TRAIN else ladder_l4.STEPS[step](config)
        pre = ladder_l4._store(config).root
    assert pre.name == "ladder_l4_pretrain_smoke"
    as_stored = _files(pre)
    # ---- the check's smoke task: the same world, the pretraining SMOKE run as `pre`'s run, THE REAL RANK
    options = entry.step_fields(entry.STAGES["ladder_l4_rank_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_rank_smoke"])
    of_the_pretraining = entry.step_fields(entry.STAGES["ladder_l4_pretrain_smoke"][2])[2]["environment"]
    assert options["environment"] == {ladder_loop.DATA_VARIABLE: of_the_pretraining[ladder_loop.DATA_VARIABLE], ladder_l4.SOURCE_VARIABLE: "ladder_l1_smoke",
                                      ladder_l4.STORED_VARIABLE: "none", ladder_l4.FILE_VARIABLE: str(FIXTURE), ladder_l4.RUN_VARIABLE: "ladder_l4_pretrain_smoke",
                                      RUN: "ladder_l4_rank_smoke", CHECK: "rank"}
    assert all(of_the_pretraining[name] == options["environment"][name] for name in (ladder_loop.DATA_VARIABLE, ladder_l4.SOURCE_VARIABLE, ladder_l4.STORED_VARIABLE,
                                                                                     ladder_l4.FILE_VARIABLE, ladder_l4.RUN_VARIABLE))
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (kits.append(given["vllm"]["max_lora_rank"]), (arm.solver, stand_in_parameters))[1]))
    del calls[:]
    summaries = _check(config, monkeypatch, stage_name="ladder_l4_rank_smoke")
    check = ladder_l4_rank.the_check(config)
    store = ladder_l4_rank._store(config, check)
    assert store.root.name == "ladder_l4_rank_smoke" and not [path.name for path in store.root.parent.iterdir() if "_r64_" in path.name] and _files(pre) == as_stored
    prepare = summaries["ladder_l4_rank_prepare"]
    assert prepare["stored_runs"] is None and prepare["rows"] == 12 and prepare["attempts_a_goal_problem"] == 32 and [sampling["name"] for sampling in prepare["goal_samplings"]] == [REACH]
    assert (prepare["rank_check"]["rank"], prepare["rank_check"]["alpha"], prepare["rank_check"]["pre"]["run"], prepare["rank_check"]["pre"]["parts"]) == (64, 128, str(pre), ["rungs", REACH])
    assert [call["lora"]["rank"] for call in calls] == [64] and summaries["ladder_l4_rank_train"]["adapter_saved"]["ranks"] == [64] and kits and set(kits) == {64}
    assert (store.root / "adapters" / "pre_r64" / "adapter_model.safetensors").exists()
    report = summaries["ladder_l4_rank_report"]
    assert report["branch"]["name"] in BRANCHES and report["label"] == "pretrained on published proofs" and all(line.startswith(SAY) for line in report["lines"])
    read = report["measured_and_not_read"] if report["inconclusive"] else report
    assert read["primary"]["attempts_each"] == 32 and read["primary"]["problems"] == 1 and store.is_done(ladder_l4_rank.REPORT)


# -------------------------------------------------------------------------------------------- registration
def test_the_stages_are_registered_with_a_guard_before_every_gpu_step_and_only_they_name_the_check():
    expected = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l4_rank_prepare"), ("guard", None), ("gpu", "ladder_l4_rank_train"),
                ("guard", None), ("gpu", "ladder_l4_rank_measure"), ("guard", None), ("gpu", "ladder_l4_rank_report")]
    for name, which in (("ladder_l4_rank", "rank"), ("ladder_l4_rank_fallback", "fallback")):
        steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES[name]]
        assert [(environment, step) for environment, step, _ in steps] == expected and all(options == {"environment": {CHECK: which}} for _, _, options in steps)
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_rank_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == expected and all(options == smoke[0][2] for _, _, options in smoke)
    # It is the pretraining stage's steps under the check's names, WITHOUT the map: a map is made only for a model an arm will start from.
    of_the_pretraining = [(environment, step) for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l4_pretrain"])]
    as_the_pretraining = [(environment, step.replace("ladder_l4_rank_", "ladder_l4_pretrain_") if step else step) for environment, step in expected]
    without_the_map = [pair for index, pair in enumerate(of_the_pretraining) if pair != ("gpu", "ladder_l4_pretrain_map") and of_the_pretraining[index:index + 2] != [
        ("guard", None), ("gpu", "ladder_l4_pretrain_map")]]
    assert as_the_pretraining == without_the_map
    assert (ladder_l4_rank.STAGE, ladder_l4_rank.FALLBACK_STAGE, ladder_l4_rank.STAGE_OF) == ("ladder_l4_rank", "ladder_l4_rank_fallback", {"rank": "ladder_l4_rank", "fallback": "ladder_l4_rank_fallback"})
    assert list(ladder_l4_rank.STEPS) == ["ladder_l4_rank_prepare", "ladder_l4_rank_train", "ladder_l4_rank_measure", "ladder_l4_rank_report"]
    others = (set(ladder_round.STEPS) | set(ladder_dose.STEPS) | set(ladder_l2.STEPS) | set(ladder_loop.STEPS) | set(ladder_ceiling.STEPS) | set(ladder_l3d1.STEPS)
              | set(ladder_l3d2.STEPS) | set(ladder_l4.STEPS))
    assert not set(ladder_l4_rank.STEPS) & others and all(name.startswith("ladder_l4_rank_") for name in ladder_l4_rank.STEPS)
    # The pretraining stage and the arm are what they were: neither names the check, and their steps are L4's own.
    assert list(ladder_l4.STEPS) == ["ladder_l4_pretrain_prepare", "ladder_l4_pretrain_train", "ladder_l4_pretrain_measure", "ladder_l4_pretrain_map",
                                     "ladder_l4_pretrain_report", "ladder_l4_prepare", "ladder_l4_report"]
    assert all(options == {} for _, _, options in map(entry.step_fields, entry.STAGES["ladder_l4_pretrain"]))
    of = lambda variable: sorted(name for name, steps in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in steps))      # noqa: E731
    named_by = of(CHECK)
    assert named_by == ["ladder_l4_rank", "ladder_l4_rank_fallback", "ladder_l4_rank_smoke"]
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l4_rank.STEPS for step in entry.STAGES[name])) == named_by
    # The check's SMOKE run is named by its own smoke stage, and by the stages that READ it and run no step of the check: L4t's smoke from `pre_r64`, and the smoke
    # stages of L4b's arm (one a training rule), which starts from `pre_r64`.
    assert of(RUN) == ["ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke", "ladder_l4b_old_smoke", "ladder_l4b_rehearse_smoke", "ladder_l4b_reward_rows_smoke"]
    # The setting: rank 64 with alpha 128, the fallback 32 with 64; alpha / rank is `pre`'s 2 in both.
    settings, lora = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")["ladder_loop"]["l4"]["rank_check"], milestone2.load_config(PACKAGE / "config" / "experiment.yaml")["lora"]
    assert settings == {"rank": 64, "alpha": 128, "fallback_rank": 32, "fallback_alpha": 64} and (lora["rank"], lora["alpha"]) == (16, 32)
    assert milestone2.load_config(PACKAGE / "config" / "experiment.yaml")["vllm"]["max_lora_rank"] == 16


def test_the_steps_run_through_the_gpu_entry_point(rank, monkeypatch, tmp_path):
    _pretrain(rank.l4, monkeypatch, [])
    monkeypatch.setattr(milestone2, "load_config", lambda path: rank.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in _steps("ladder_l4_rank"):
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert rank.store().is_done(ladder_l4_rank.REPORT)
