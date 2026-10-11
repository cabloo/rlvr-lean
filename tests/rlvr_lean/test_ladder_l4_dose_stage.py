"""L4p, the stage end to end on what L4t2's FINISHED run and the start model's run stored: the three training sets along
the one hash order (the refusals of its prepare step), each model trained from `pre_r64` at rank 64 and measured by L4t's
own two step bodies, the report; the job that trains and measures `matched_1262` alone, and the whole stage after it in
the same run directory; the runs it reads left file for file as they were, and L4t (from `pre` and from `pre_r64`) and
L4t2 writing what they wrote; the smoke stage; the registration. Spec:
docs/spec/ladder-loop.spec.md, "L4p: how much more pretraining? The dose of a second pass". The engine and Lean are
scripted; nothing touches a GPU or the network. The rules and the report on hand-made rows are `test_ladder_l4_dose.py`."""

import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import arm  # noqa: E402, F401
from test_ladder_l2_stage import ARM, loop  # noqa: E402, F401
from test_ladder_l4_rows2_stage import OWN_FILES as L4T2_FILES  # noqa: E402 - what L4t2 leaves, frozen there
from test_ladder_l4_rows2_stage import STEPS as L4T2_STEPS  # noqa: E402
from test_ladder_l4_rows2_stage import _stage2 as _l4t2  # noqa: E402 - L4t2's own stage, as its tests run it
from test_ladder_l4_rows2_stage import rows2  # noqa: E402, F401
from test_ladder_l4_rows_r64 import OWN_FILES as L4T_FILES_FROM_PRE_R64  # noqa: E402 - what L4t from `pre_r64` leaves, frozen there
from test_ladder_l4_rows_r64 import STEPS as L4T_STEPS_FROM_PRE_R64  # noqa: E402
from test_ladder_l4_rows_r64 import _stage as _l4t  # noqa: E402
from test_ladder_l4_rows_r64 import _trainings, r64  # noqa: E402, F401
from test_ladder_l4_rows_stage import OWN_FILES as L4T_FILES_FROM_PRE  # noqa: E402
from test_ladder_l4_rows_stage import STEPS as L4T_STEPS_FROM_PRE  # noqa: E402
from test_ladder_l4_rows_stage import _all, rows  # noqa: E402, F401
from test_ladder_l4_rows_stage import _stage as _l4t_from_pre  # noqa: E402
from test_ladder_l4_stage import FIXTURE, _file_rows, _on_the_gpu, l4  # noqa: E402, F401
from test_ladder_round import _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, LOOP, NOT_READ, goal_set_again, half_of  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_dose import BRANCHES, MATCHED_BRANCHES, MODELS, MORE_COSTS, dose_sets  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import REHEARSAL_LABEL  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_dose, ladder_l4_rank, ladder_l4_rows, ladder_l4_rows2, ladder_l4b, ladder_loop, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.ladder_l3d1 import WROTE  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
RUN, L4T2_RUN = ladder_l4_dose.RUN_VARIABLE, ladder_l4_dose.L4T2_RUN_VARIABLE
L4T2_OWN_RUN, L4T2_L4T_RUN = ladder_l4_rows2.RUN_VARIABLE, ladder_l4_rows2.L4T_RUN_VARIABLE
L4T_START, L4T_MINIMUM, L4T_OWN_RUN, L4T_ARM_RUN, L4T_START_CHECKS = (ladder_l4_rows.START_VARIABLE, ladder_l4_rows.MINIMUM_VARIABLE, ladder_l4_rows.RUN_VARIABLE,
                                                                     ladder_l4_rows.ARM_RUN_VARIABLE, ladder_l4_rows.START_CHECKS_VARIABLE)
RANK_RUN = ladder_l4_rank.RUN_VARIABLE
SEEDS = "RLVR_LEAN_TRAINING_SEEDS"
MATCHED, HALF, FULL = MODELS
STEPS = ["ladder_l4_dose_prepare", "ladder_l4_dose_train_matched_1262", "ladder_l4_dose_measure_matched_1262", "ladder_l4_dose_train_second_half",
         "ladder_l4_dose_measure_second_half", "ladder_l4_dose_train_second_full", "ladder_l4_dose_measure_second_full", "ladder_l4_dose_report"]
MATCHED_STEPS = [*STEPS[:3], "ladder_l4_dose_report_matched"]      # the stage `ladder_l4_dose_r64_matched`: `matched_1262` alone, and the report with it alone measured
TRAININGS = ("ladder_l4_dose_train_matched_1262", "ladder_l4_dose_train_second_half", "ladder_l4_dose_train_second_full")
RANK_64 = {"rank": 64, "alpha": 128}
L4T2S = ("rehearse_only", "reward_rehearse")               # L4t2's two models, read from the rows its run stored
STORED = ("pre_r64", *L4T2S)                               # the stored models the reads here are against: the start model, the dose L4t2 measured, and the model `matched_1262` is the control of
# What the stage leaves in its run directory, beside the episodes of the sets it samples.
OWN_FILES = sorted([
    *(f"adapters/{model}/{name}" for model in MODELS for name in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"l4_dose_loss_{model}.json" for model in MODELS), "l4_dose_stored_models.json", *(f"l4_dose_training_{model}.jsonl" for model in MODELS),
    "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", *(f"l4_stored_{who}_{part}.jsonl" for who in STORED for part in ("rungs", "reach", "more")),
    *(f"ladder_l4_dose_{step}_{model}.done.json" for model in MODELS for step in ("train", "measure")), "ladder_l4_dose_prepare.done.json", "ladder_l4_dose_report.done.json",
    "problems.jsonl", "report_ladder_l4_dose.json"])
PREPARE_KEYS = ["stage", "label", "check", "seed", "start", "start_adapter", "start_run", "l4t2_run", "models", "stand_in_engine", "effective_batch",
                "rows_the_start_was_pretrained_on", "rehearse_only", "reward_rehearse", "training_sets", "pretraining_file", "pretraining_file_sha256", "recipe", "goal_set", "goal_set_again",
                "rung_problems", "sampling_seeds", "rung_episodes", "goal_samplings", "attempts_a_goal_problem", "parts", "stored_models", "never_solved_file",
                "contradicted_side_setting", "start_recipe", "start_run_read", "l4t2_read"]
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "rank", "measured", "not_measured_yet", "branches", "branch",
               "inconclusive", "primary", "the_rounds_hard_rows", "models", "overfitting", "the_dose", "the_three_reads", "what_it_cannot_say", "the_stored_models", "training_sets", "heldout", "sizes", "start_recipe", "attempts",
               "not_to_be_read", "not_measured", "adapters"]


def _stage_p(config, monkeypatch, until=None, stage_name="ladder_l4_dose_r64"):
    """The stage's steps in its order, the trainings by their GPU path."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l4_dose.STEPS:
            continue
        run = ladder_l4_dose.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in TRAININGS else run(config)
        if step == until:
            break
    return summaries


def _draw_order(rows_, seed=0):
    """Rows of the pretraining file in the order L4t's rehearsal draw takes them: the smallest content hash of the seed and the row's id first."""
    return sorted(rows_, key=lambda row: rank(seed, REHEARSAL_LABEL, row["id"]))


def _as_named(row):
    return {key: value for key, value in row.items() if key != "row"}


def _placed(row):
    return row["id"], row["problem_id"], row["row_of_the_file"]


@pytest.fixture
def dose(rows2, monkeypatch):  # noqa: F811
    """`rows2`'s world (`pre`'s run, the arm's, the run of the check of the adapter's rank, L4t's run from `pre_r64`) with
    L4t2's own stage RUN TO ITS END in it: what this stage reads on the box. Then this stage's task, which carries NO
    variable (the start is its own setting's). `l4t2_summaries`: what L4t2's steps returned."""
    for name in (RUN, L4T2_RUN):
        monkeypatch.delenv(name, raising=False)
    config = rows2.config
    l4t2_summaries = _l4t2(config, monkeypatch)
    l4t2 = rows2.store().root
    assert l4t2.name == "ladder_l4_rows2_pre_r64_seed0" and list(l4t2_summaries) == L4T2_STEPS and l4t2_summaries["ladder_l4_rows2_report"]["ok"] is True
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l4_dose_r64"])        # THE STAGE names no variable
    del rows2.calls[:], rows2.kits[:]
    return SimpleNamespace(config=config, rows2=rows2, calls=rows2.calls, kits=rows2.kits, l4t2=l4t2, l4t=rows2.l4t, rank=rows2.rank, arm=rows2.arm, pre=rows2.pre,
                           l4t2_summaries=l4t2_summaries, solver=rows2.r64.l4.solver, lean=rows2.r64.l4.lean,
                           store=lambda: ladder_l4_dose._store(config, ladder_l4_dose.the_settings(config)))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_builds_the_three_sets_along_the_one_order_trains_each_from_pre_r64_at_rank_64_measures_them_and_reports(dose, monkeypatch):
    config, calls, kits = dose.config, dose.calls, dose.kits
    l4t2, rank_run = dose.l4t2, dose.rank
    read_only = {path: _all(path) for path in (l4t2, rank_run, dose.l4t, dose.arm, dose.pre)}
    summaries = _stage_p(config, monkeypatch)
    store = dose.store()
    assert store.root.name == "ladder_l4_dose_pre_r64_seed0" and store.root.parent == l4t2.parent and list(summaries) == STEPS and list(ladder_l4_dose.STEPS) == [*STEPS, MATCHED_STEPS[-1]]
    assert {path: _all(path) for path in read_only} == read_only               # L4t2's RUN AND THE START MODEL's WERE ONLY READ: file for file, adapters included; so were the others
    assert (config["lora"]["rank"], config["vllm"]["max_lora_rank"], config["sampling"]["temperature"]) == (16, 16, 1.0)      # and the config itself was not changed

    # ---- prepare: the start, its rank, what L4t2's run stored, the three training sets
    prepare = summaries["ladder_l4_dose_prepare"]
    start = rank_run / "adapters" / "pre_r64"
    assert list(prepare) == PREPARE_KEYS and (prepare["stage"], prepare["label"], prepare["check"], prepare["seed"]) == ("l4", "pretrained on published proofs", "L4p", 0)
    assert (prepare["start"], prepare["start_adapter"], prepare["start_run"], prepare["l4t2_run"], prepare["models"]) == ("pre_r64", str(start), str(rank_run), str(l4t2), list(MODELS))
    of_l4t2 = json.loads((l4t2 / "ladder_l4_rows2_prepare.done.json").read_text())
    of_start = json.loads((rank_run / "ladder_l4_rank_prepare.done.json").read_text())
    recipe = prepare["start_recipe"]
    assert (recipe["rank"], recipe["alpha"], recipe["rank_of_the_config"], recipe["alpha_of_the_config"]) == (64, 128, 16, 32) == (
        of_start["recipe"]["lora"]["rank"], of_start["recipe"]["lora"]["alpha"], config["lora"]["rank"], config["lora"]["alpha"])
    assert (recipe["rank"], recipe["alpha"]) == (of_l4t2["start_recipe"]["rank"], of_l4t2["start_recipe"]["alpha"]) and prepare["recipe"]["lora"] == {**config["lora"], **RANK_64}
    assert prepare["start_run_read"]["checks_waived_as_l4t2s_run_recorded"] is False and prepare["pretraining_file_sha256"] == of_l4t2["pretraining_file_sha256"] == of_start["pretraining_file_sha256"]
    assert all(prepare[key] == of_l4t2[key] == of_start[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings")) and prepare["goal_set_again"] == of_l4t2["goal_set_again"]
    # L4t2's OWN report, as it read its two models: carried here.
    of_its_report = json.loads((l4t2 / "report_ladder_l4_rows2.json").read_text())
    assert prepare["l4t2_read"]["branches"] == of_its_report["branches"] and prepare["l4t2_read"]["run"] == str(l4t2)
    assert prepare["l4t2_read"]["failed_checks"] == {model: [check for check, said in of_its_report["models"][model]["can_this_model_be_read"].items() if check != "what" and not said["passes"]]
                                                     for model in L4T2S}
    # THE THREE SETS, ALONG THE ONE HASH ORDER: `matched_1262`'s first, then `rehearse_only`'s stored rows, then `second_half`'s, inside `second_full`'s; by id and by place.
    file_rows = _file_rows(FIXTURE)
    rehearse_only, reward_rehearse = (_file_rows(l4t2 / f"l4_rows2_training_{model}.jsonl") for model in L4T2S)
    matched, half, full = (store.read_rows(f"l4_dose_training_{model}.jsonl") for model in MODELS)
    assert (len(file_rows), len(half), len(full)) == (12, 6, 12) and 0 < len(matched) < len(rehearse_only) < len(half)       # four doses that differ: every nesting is of rows that differ
    order = _draw_order(full)
    # `matched_1262` IS THE PUBLISHED ROWS OF L4t2's STORED `reward_rehearse` SET AND NOTHING ELSE: the same rows key for key, in that set's order with its rounds' rows left out.
    published = [row for row in reward_rehearse if row["origin"] == "pretraining"]
    assert published and len(published) < len(reward_rehearse) and [_as_named(row) for row in matched] == [_as_named(row) for row in published]
    assert list(map(_placed, _draw_order(matched))) == list(map(_placed, order[:len(matched)])) == list(map(_placed, _draw_order(rehearse_only)[:len(matched)]))
    assert list(map(_placed, _draw_order(rehearse_only))) == list(map(_placed, order[:len(rehearse_only)])) and list(map(_placed, _draw_order(half))) == list(map(_placed, order[:6]))
    assert [_as_named(row) for row in _draw_order(rehearse_only)] == [_as_named(row) for row in order[:len(rehearse_only)]]                    # the same rows, key for key
    # `second_full` holds EVERY row of the pretraining file exactly once, each naming the problem at its place of the file.
    assert sorted(row["row_of_the_file"] for row in full) == list(range(12)) and sorted(row["problem_id"] for row in full) == sorted(row["problem_id"] for row in file_rows)
    assert all(file_rows[row["row_of_the_file"]]["problem_id"] == row["problem_id"] and row["id"] == f"{row['problem_id']}#pretraining" for row in (*half, *full))
    # Each set is in the arm's content-hash order, numbered by its place in its own set; a smaller dose is the larger one's order with rows left out.
    for own in (matched, half, full):
        assert [row["id"] for row in own] == [row["id"] for row in training_order(own, 0)] and [row["row"] for row in own] == list(range(len(own)))
        assert all(row["origin"] == "pretraining" and row["k"] is None for row in own)
    assert [row["id"] for row in half] == [row["id"] for row in full if row["id"] in {given["id"] for given in half}]
    assert [row["id"] for row in rehearse_only] == [row["id"] for row in half if row["id"] in {given["id"] for given in rehearse_only}]
    sets = prepare["training_sets"]
    assert list(sets) == list(MODELS) and (sets[HALF]["rows"], sets[FULL]["rows"], prepare["rows_the_start_was_pretrained_on"], prepare["rehearse_only"]["rows"]) == (6, 12, 12, len(rehearse_only))
    assert (sets[MATCHED]["rows"], sets[MATCHED]["published_rows_of_reward_rehearse"]) == (len(published), len(published)) and prepare["reward_rehearse"] == {
        "what": prepare["reward_rehearse"]["what"], "file": "l4_rows2_training_reward_rehearse.jsonl", "rows": len(reward_rehearse), "published_rows": len(published),
        "rows_of_the_rounds": len(reward_rehearse) - len(published), "run": str(l4t2)}
    assert (sets[HALF]["rows_of_rehearse_only_they_begin_with"], sets[FULL]["rows_of_second_half_they_begin_with"]) == (len(rehearse_only), 6)
    assert all(sets[model]["optimizer_steps"] == -(-sets[model]["rows"] // 8) and sets[model]["barred"]["held_out"] == [] and sets[model]["rows_by_origin"] == {"pretraining": sets[model]["rows"]}
               and sets[model]["barred"]["rehearsal_rows"] == sets[model]["rows"] and sets[model]["barred"]["rehearsal_rows_not_of_the_pretrain_half"] == [] for model in MODELS)
    assert prepare["rehearse_only"]["file"] == "l4_rows2_training_rehearse_only.jsonl" and prepare["rehearse_only"]["run"] == str(l4t2)
    # NO PUBLISHED PROOF IS COPIED INTO THE RUN: a row is named by its id and its place, and its text stays in the pretraining file.
    assert not any(field in row for row in (*matched, *half, *full) for field in TEXT_FIELDS)
    assert not any(proof in content.decode(errors="ignore") for proof in (*(json.dumps(row["proof"], ensure_ascii=False)[1:-1] for row in file_rows), *(row["proof"].strip() for row in file_rows))
                   for content in _all(store.root).values())
    # G' is the START MODEL's, made again from its rows, and it is L4t2's; the stored rows are the start model's own run's and L4t2's run's.
    goal = [row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE) if row["group"] == "goal"]
    again = [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)]
    assert again == goal_set_again(_file_rows(rank_run / "episodes_l4_reach_pre_r64_problems.jsonl"), goal) == [row["problem_id"] for row in _file_rows(l4t2 / ladder_l4.AGAIN_FILE)]
    assert store.read_rows(ladder_l4.GROUPS_FILE) == _file_rows(l4t2 / ladder_l4.GROUPS_FILE)
    for part in ("rungs", REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("pre_r64", part, "l4")) == _file_rows(rank_run / f"episodes_l4_{part}_pre_r64_problems.jsonl")
        for model in L4T2S:
            assert store.read_rows(ladder_ceiling.stored_file(model, part, "l4")) == _file_rows(l4t2 / f"episodes_l4_{part}_{model}_problems.jsonl")
    assert prepare["stored_models"] == {"pre_r64": {"run": str(rank_run), "sets": {"rungs": "l4_rungs_pre_r64", REACH: "l4_reach_pre_r64", MORE: "l4_more_pre_r64"}},
                                        **{model: {"run": str(l4t2), "sets": {"rungs": f"l4_rungs_{model}", REACH: f"l4_reach_{model}", MORE: f"l4_more_{model}"}} for model in L4T2S}}
    of_the_stored = json.loads(store.path(ladder_l4_dose.STORED_MODELS_FILE).read_text())
    assert set(of_the_stored) == {"stage", "label", *STORED}
    assert of_the_stored["pre_r64"] == {key: json.loads((rank_run / "ladder_l4_rank_measure.done.json").read_text())[key] for key in WROTE}
    for model in L4T2S:
        assert of_the_stored[model] == {key: json.loads((l4t2 / f"ladder_l4_rows2_measure_{model}.done.json").read_text())[key] for key in WROTE}

    # ---- train: each model FROM `pre_r64`, AT ITS RANK, by L4t's own training body; `matched_1262` first; EVERY row's text read from the pretraining file
    by_problem = {row["problem_id"]: row for row in file_rows}
    assert [next(iter(call["schedule"]["checkpoint_steps"])) for call in calls] == list(MODELS) and len(calls) == 3
    for call, model, set_rows in zip(calls, MODELS, (matched, half, full)):
        train = summaries[f"ladder_l4_dose_train_{model}"]
        assert call["lora"] == {**config["lora"], **RANK_64} and call["max_lora_rank"] == 16      # the adapter's rank is the start adapter's; a training starts no model server
        assert call["more"] == {"start": start} and call["orders"] == [list(range(len(set_rows)))] and call["tensorboard_run"] == f"ladder_l4_dose_pre_r64_seed0_{model}"
        assert call["schedule"] == {"passes": 1, "reading_steps": [], "checkpoint_steps": {model: -(-len(set_rows) // 8)}} and call["directory"] == store.root / "adapters"
        assert call["examples"] == [{"problem_id": row["problem_id"], "side": "statement", "theorem": by_problem[row["problem_id"]]["statement"], "completion": by_problem[row["problem_id"]]["proof"]}
                                    for row in set_rows]
        assert (train["stage"], train["label"], train["model"], train["rows"], train["trained_from"], train["adapter"]) == (
            "l4", "pretrained on published proofs", model, len(set_rows), f"the stored adapter {start}", str(store.root / "adapters" / model))
        assert train["rows_by_origin"] == {"pretraining": len(set_rows)} and "a content hash of the task's seed and each row's id" in train["order"]
        saved = train["adapter_saved"]
        assert (saved["rank"], saved["alpha"], saved["ranks"]) == (64, 128, [64]) and set(train["against_the_start_adapter"]) == {model}
        assert json.loads((store.root / "adapters" / model / "adapter_config.json").read_text())["r"] == 64
        loss = json.loads(store.path(f"l4_dose_loss_{model}.json").read_text())
        assert loss["rows_trained"] == [row["id"] for row in set_rows] and len(loss["row_losses"]) == len(set_rows) and loss["label"] == "pretrained on published proofs"

    # ---- measure: as L4t's models were, by L4t's own measuring body: the rungs and G's samplings, the start model's sampling seeds, the config's temperature, rank 64
    assert len(kits) == 3 * 3 and all(kit["temperature"] == 1.0 and kit["max_lora_rank"] == 64 and kit["rank"] == 64 and {**kit["vllm"], "max_lora_rank": 16} == config["vllm"] for kit in kits)
    for model in MODELS:
        measured = summaries[f"ladder_l4_dose_measure_{model}"]
        assert (measured["stage"], measured["label"], measured["model"]) == ("l4", "pretrained on published proofs", model) and set(measured["goal"]) == {REACH, MORE}
        for of, own_set, start_set in ((measured["rungs"], f"l4_rungs_{model}", "l4_rungs_pre_r64"), (measured["goal"][REACH], f"l4_reach_{model}", "l4_reach_pre_r64"),
                                       (measured["goal"][MORE], f"l4_more_{model}", "l4_more_pre_r64")):
            marker = json.loads((rank_run / f"episodes_{start_set}.done.json").read_text())
            assert of["set"] == own_set and (of["sampling_seed"], of["episodes_each"], of["problems"]) == (marker["sampling_seed"], marker["episodes_each"], marker["problems"])
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l4_{part}_{model}" for part in ("rungs", REACH, MORE) for model in MODELS}
    assert not [name for name in _all(store.root) if "_hot_" in name]         # no measurement at another temperature, and no set of one

    # ---- the report: each model's four checks, the rounds' hard rows, the primary and its branch, overfitting, the dose
    report = summaries["ladder_l4_dose_report"]
    assert list(report) == REPORT_KEYS and (report["stage"], report["label"], report["check"], report["ok"], report["start"], report["rank"]) == (
        "l4", "pretrained on published proofs", "L4p", True, "pre_r64", 64)
    assert all(line.startswith(f"{SAY}: ") for line in report["lines"]) and "pretrained on published proofs" in report["headline"] and "from `pre_r64`" in report["headline"]
    assert report["branch"]["name"] in BRANCHES and list(report["models"]) == list(MODELS) == list(report["training_sets"]) and report["the_dose"]["rows"] == {
        MATCHED: len(matched), "rehearse_only": len(rehearse_only), HALF: 6, FULL: 12}
    assert (report["measured"], report["not_measured_yet"]) == (list(MODELS), []) and report["branches"] == {"the_dose": report["branch"]["name"], "the_rounds_hard_rows": report["the_rounds_hard_rows"]["name"]}
    assert report["the_rounds_hard_rows"]["name"] in MATCHED_BRANCHES and (report["the_rounds_hard_rows"]["published_rows"], report["the_rounds_hard_rows"]["rows_of_reward_rehearse"]) == (
        len(matched), len(reward_rehearse))
    solved_in = lambda name: sum(sum(counts) > 0 for counts in zip(*([row["resolved"] for row in store.read_rows(name(part))] for part in (REACH, MORE))))      # noqa: E731
    by_the_start = solved_in(lambda part: ladder_ceiling.stored_file("pre_r64", part, "l4"))
    for model in MODELS:
        read = report["models"][model]
        assert set(read["can_this_model_be_read"]) == {"what", "its_training_ran", "it_still_writes_proofs", "lean_answered", "no_barred_row"}
        secondary = read["secondary"] if not read["inconclusive"] else read["measured_and_not_read"]["secondary"]
        assert secondary["trained_on"]["rows"] == sets[model]["rows"] and secondary["trained_on"]["rows_by_origin"] == {"pretraining": sets[model]["rows"]}
        assert secondary["goal_problems_solved_by_attempts_alone"].keys() >= {"pre_r64", "rehearse_only", model} and ("reward_rehearse" in secondary["distinct_attempts"]) is (model == MATCHED)
        assert read["the_three_reads"] is report["the_three_reads"][model] and read["overfitting"] is report["overfitting"][model] and read["overfitting"]["against"] == "pre_r64"
        # Its reads stand on the rows THIS run measured for it, against the start model's stored rows.
        against = report["the_dose"]["against_the_start"][model]
        figures = against if against["read"] else against["measured_and_not_read"]
        assert (figures["solved"]["resolved_after"], figures["solved"]["resolved_before"], figures["solved"]["problems"]) == (
            solved_in(lambda part: f"episodes_l4_{part}_{model}_problems.jsonl"), by_the_start, len(goal))
    assert json.loads(store.path(ladder_l4_dose.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_l4_dose.REPORT) == {"stage": "l4", "label": "pretrained on published proofs", "check": "L4p", "headline": report["headline"],
                                                         "branches": report["branches"], "measured": list(MODELS), "ok": True}
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == sorted(MODELS) and "`pre_r64` is its own run's and was only read" in report["adapters"]["what"]
    # The goal problems nothing stored had solved are GIVEN: a file in the run directory, one `problem_id` a row. The report is built again and reads it.
    store.write_rows(ladder_l4_dose.NEVER_SOLVED_FILE, [{"problem_id": goal[0]}, {"problem_id": "not_a_goal_problem"}])
    assert ladder_l4_dose.NEVER_SOLVED_FILE == "l4_dose_never_solved.jsonl" == prepare["never_solved_file"]
    given = ladder_l4_dose.ladder_l4_dose_report(config)["models"][FULL]
    given = (given["measured_and_not_read"] if given["inconclusive"] else given)["secondary"]["never_solved_before"]
    assert (given["given"], given["ids"], given["goal_problems"], given["not_goal_problems_of_this_run"]) == (True, 2, 1, 1) and "solved_by_rehearse_only" in given
    store.path(ladder_l4_dose.NEVER_SOLVED_FILE).unlink()
    # What it leaves: its own files; nothing of it is in another run's directory.
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert {path.name for path in store.root.parent.iterdir() if path.name.startswith("ladder_l4_dose") or path.name.startswith("ladder_l4_rows")} == {
        "ladder_l4_rows_pre_r64_seed0", "ladder_l4_rows2_pre_r64_seed0", "ladder_l4_dose_pre_r64_seed0"}
    # A rerun returns what is stored: nothing is trained or sampled again.
    sent, sampled = len(dose.lean.sources), dose.solver.calls
    rerun = _stage_p(config, monkeypatch)
    assert len(calls) == 3 and (len(dose.lean.sources), dose.solver.calls) == (sent, sampled) and {path: _all(path) for path in read_only} == read_only
    assert {name: summary for name, summary in rerun.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}


def test_what_each_stored_model_wrote_and_its_rows_are_its_own_runs_and_no_other_models(dose):
    """The scripted solver writes the same for every model, so each stored record and each stored set is marked here with
    a figure of its own before the prepare step reads it: each of L4t2's two is its own, and never the other's."""
    config, l4t2, rank_run = dose.config, dose.l4t2, dose.rank
    marks = {"pre_r64": 0.811, "rehearse_only": 0.722, "reward_rehearse": 0.633}
    markers = {"pre_r64": rank_run / "ladder_l4_rank_measure.done.json", **{model: l4t2 / f"ladder_l4_rows2_measure_{model}.done.json" for model in ("rehearse_only", "reward_rehearse")}}
    rungs = {"pre_r64": rank_run / "episodes_l4_rungs_pre_r64_problems.jsonl", **{model: l4t2 / f"episodes_l4_rungs_{model}_problems.jsonl" for model in ("rehearse_only", "reward_rehearse")}}
    for place, (name, mark) in enumerate(marks.items()):
        recorded = json.loads(markers[name].read_text())
        recorded["distinct_attempts"]["goal"]["mean_share_distinct"] = mark
        markers[name].write_text(json.dumps(recorded))
        own = _file_rows(rungs[name])
        rungs[name].write_text("".join(json.dumps({**row, "resolved": place if index == 0 else row["resolved"]}) + "\n" for index, row in enumerate(own)))
    ladder_l4_dose.ladder_l4_dose_prepare(config)
    store = dose.store()
    of_the_stored = json.loads(store.path(ladder_l4_dose.STORED_MODELS_FILE).read_text())
    assert {name: of_the_stored[name]["distinct_attempts"]["goal"]["mean_share_distinct"] for name in STORED} == marks and list(marks) == list(STORED)
    assert all(set(of_the_stored[name]) == set(WROTE) for name in STORED) and set(of_the_stored) == {"stage", "label", *STORED}
    assert [store.read_rows(ladder_ceiling.stored_file(name, "rungs", "l4"))[0]["resolved"] for name in STORED] == [0, 1, 2]
    assert not [name for name in _all(store.root) if "old_rule" in name or "reward_rows" in name]


def test_the_prepare_step_refuses_before_anything_is_written(dose, monkeypatch, tmp_path):
    config, l4t2, rank_run = dose.config, dose.l4t2, dose.rank
    as_stored = {path: _all(path) for path in (l4t2, rank_run)}
    root = dose.store().root
    l4t2s_task = "the task of stage `ladder_l4_rows2_r64` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rows2_r64 --seeds 0`; for the smoke run, stage `ladder_l4_rows2_r64_smoke`."
    ranks_task = "the task of stage `ladder_l4_rank` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rank --seeds 0`; for the smoke run, stage `ladder_l4_rank_smoke`."

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l4_dose.ladder_l4_dose_prepare(config)
        assert root.name == "ladder_l4_dose_pre_r64_seed0" and not _all(root)  # nothing was written: no marker, no file

    def with_a_change(directory, change, match, error=RuntimeError):
        """One of the two runs with one thing of it changed, refused; then put back as it was."""
        change()
        refused(match, error)
        for name, content in as_stored[directory].items():
            (directory / name).write_bytes(content)

    def rewritten(directory, name, change):
        return lambda: (directory / name).write_text(json.dumps(change(json.loads(as_stored[directory][name]))))

    def rows_of(directory, name, change):
        own = [json.loads(line) for line in as_stored[directory][name].decode().splitlines() if line.strip()]
        return lambda: (directory / name).write_text("".join(json.dumps(row) + "\n" for row in change(own)))

    change_of = lambda **changed: (lambda recorded: {**recorded, **changed})      # noqa: E731
    # ---- L4t2's RUN, FINISHED: what its steps recorded, its report, its problems, groups and G', `rehearse_only`'s stored set, its rows and markers
    for name in ("ladder_l4_rows2_prepare.done.json", "ladder_l4_rows2_report.done.json", "report_ladder_l4_rows2.json", "problems.jsonl", "l4_heldout_groups.jsonl", "l4_goal_set_again.jsonl",
                 "l4_rows2_training_rehearse_only.jsonl", "l4_rows2_training_reward_rehearse.jsonl", "ladder_l4_rows2_measure_rehearse_only.done.json",
                 "ladder_l4_rows2_measure_reward_rehearse.done.json"):
        with_a_change(l4t2, (l4t2 / name).unlink, f"ladder_l4_rows2_pre_r64_seed0 does not hold .'{name}'.: what L4t2's steps recorded, its report, .* or one of its two stored training sets. "
                                                  f"L4p makes no round and no pretraining of its own: it reads what {l4t2s_task} wrote on this box. Run that task to its end first; nothing was written")
    for name in ("episodes_l4_rungs_rehearse_only_problems.jsonl", "episodes_l4_reach_rehearse_only.done.json", "episodes_l4_more_rehearse_only_problems.jsonl",
                 "episodes_l4_rungs_reward_rehearse.done.json", "episodes_l4_more_reward_rehearse_problems.jsonl"):
        with_a_change(l4t2, (l4t2 / name).unlink, f"ladder_l4_rows2_pre_r64_seed0 does not hold .'{name}'.: the per-problem rows of one of L4t2's two models on a set it was measured on, or "
                                                  f"that set's own marker. L4p makes no round and no pretraining of its own: it reads what {l4t2s_task} wrote")
    # ---- it is L4t2's run of THIS seed, from THIS start, with its two models
    for changed in ({"seed": 3}, {"start": "pre"}, {"start": "pre_r32"}, {"models": ["reward_rehearse"]}, {"models": ["rehearse_only"]}, {"models": None}):
        with_a_change(l4t2, rewritten(l4t2, "ladder_l4_rows2_prepare.done.json", change_of(**changed)),
                      "L4t2's run ladder_l4_rows2_pre_r64_seed0 recorded seed .*; this task's seed is 0, ladder_loop.l4.dose names the start 'pre_r64', and .'rehearse_only', "
                      "'reward_rehearse'. are the two models read from it: the sets and the rows read here are that run's")
    # ---- a report of L4t2's that is NOT TO BE READ
    with_a_change(l4t2, rewritten(l4t2, "report_ladder_l4_rows2.json", change_of(ok=False)),
                  f"the report of L4t2's run ladder_l4_rows2_pre_r64_seed0 is not to be read .Lean gave no verdict on too much of a set.: its models' rows cannot stand in a "
                  f"comparison. Queue {l4t2s_task} again")
    # ---- L4t2's run and the start model's run must be of ONE start model: its rank, how it was measured, its G', the file it was pretrained on
    for lora, said in (({"rank": 32, "alpha": 64}, "rank 32 and alpha 64"), ({"rank": 64, "alpha": 64}, "rank 64 and alpha 64")):
        with_a_change(l4t2, rewritten(l4t2, "ladder_l4_rows2_prepare.done.json", lambda recorded, lora=lora: {**recorded, "start_recipe": {**recorded["start_recipe"], **lora}}),
                      f"the run ladder_l4_pretrain_r64_seed0 recorded an adapter of rank 64 and alpha 128 for `pre_r64`, and L4t2's run ladder_l4_rows2_pre_r64_seed0 trained "
                      f"its models at {said}: every model here is at the rank L4t2's were")
    with_a_change(l4t2, rewritten(l4t2, "ladder_l4_rows2_prepare.done.json", lambda recorded: {**recorded, "sampling_seeds": {**recorded["sampling_seeds"], "rungs": 7}}),
                  "the run ladder_l4_pretrain_r64_seed0 recorded how `pre_r64` was measured .*, and L4t2's run ladder_l4_rows2_pre_r64_seed0 measured its models otherwise .*: "
                  "L4p's models would not pair with both")
    with_a_change(l4t2, rows_of(l4t2, "l4_goal_set_again.jsonl", lambda own: [*own, {"problem_id": "nowhere"}]),
                  "G' made from the rows of `pre_r64` .ladder_l4_pretrain_r64_seed0: .* goal problems it does not solve in its first sampling. is not the G' L4t2's run "
                  "ladder_l4_rows2_pre_r64_seed0 stored: the two runs are not of one start model's rows")
    # ... G' is made from the START MODEL's first sampling and no other model's: with a goal problem it no longer solves there, it is not the G' L4t2 stored.
    assert _file_rows(l4t2 / "l4_goal_set_again.jsonl") == [] and all(row["resolved"] > 0 for row in _file_rows(l4t2 / "episodes_l4_reach_rehearse_only_problems.jsonl"))
    with_a_change(rank_run, rows_of(rank_run, "episodes_l4_reach_pre_r64_problems.jsonl", lambda own: [{**row, "resolved": 0} for row in own]),
                  "G' made from the rows of `pre_r64` .ladder_l4_pretrain_r64_seed0: 1 goal problems it does not solve in its first sampling. is not the G' L4t2's run")
    with_a_change(l4t2, rewritten(l4t2, "ladder_l4_rows2_prepare.done.json", change_of(pretraining_file_sha256="0" * 64)),
                  "L4t2's run ladder_l4_rows2_pre_r64_seed0 was prepared on a pretraining file of SHA-256 0{64} and `pre_r64` was trained on .*: L4t2's sets name each published "
                  "row by its place in the file it was prepared on")
    # ---- the stored rows must pair with this run: the seeds, the problems, Lean having answered
    marker = json.loads(as_stored[l4t2]["episodes_l4_reach_rehearse_only.done.json"])
    with_a_change(l4t2, rewritten(l4t2, "episodes_l4_reach_rehearse_only.done.json", change_of(sampling_seed=5)),
                  f"ladder_l4_rows2_pre_r64_seed0 measured `rehearse_only` .reach. with sampling seed 5 and {marker['episodes_each']} episodes; this config gives {marker['sampling_seed']} and "
                  f"{marker['episodes_each']}: L4p's models would not pair with the stored results")
    with_a_change(l4t2, rows_of(l4t2, "episodes_l4_rungs_rehearse_only_problems.jsonl", lambda own: own[:-1]), "measured `rehearse_only` .rungs. on other problems than L1's run holds for it")
    with_a_change(l4t2, rows_of(l4t2, "episodes_l4_reach_reward_rehearse_problems.jsonl", lambda own: [{**own[0], "attempts_without_an_answer": 30}, *own[1:]]),
                  "measured `reward_rehearse` .reach. with too many attempts without a verdict from Lean: that set stands on the other side of a comparison")
    with_a_change(l4t2, rows_of(l4t2, "episodes_l4_more_rehearse_only_problems.jsonl", lambda own: [{**own[0], "attempts_without_an_answer": 60}, *own[1:]]),
                  "measured `rehearse_only` .more. with too many attempts without a verdict from Lean: that set stands on the other side of a comparison")
    with_a_change(rank_run, rows_of(rank_run, "episodes_l4_rungs_pre_r64_problems.jsonl", lambda own: own[:-1]), "ladder_l4_pretrain_r64_seed0 measured `pre_r64` .rungs. on other problems")
    # ---- THE START MODEL's RUN: its report, what its three steps recorded, the rows it was pretrained on, its rows and their markers; refused as L4t2 refuses them
    for name in ("report_ladder_l4_rank.json", "ladder_l4_rank_prepare.done.json", "ladder_l4_rank_train.done.json", "ladder_l4_rank_measure.done.json"):
        with_a_change(rank_run, (rank_run / name).unlink, f"ladder_l4_pretrain_r64_seed0 does not hold .'{name}'.: the report of the check that made `pre_r64`, or what its prepare, train "
                                                          f"or measure step recorded. L4p makes no round and no pretraining of its own: it reads what {ranks_task} wrote on this box")
    for name in ("l4_pretraining_rows.jsonl", "episodes_l4_reach_pre_r64.done.json", "episodes_l4_more_pre_r64_problems.jsonl"):
        with_a_change(rank_run, (rank_run / name).unlink, f"ladder_l4_pretrain_r64_seed0 does not hold .'{name}'.: the rows `pre_r64` was pretrained on .every row trained on here is one of "
                                                          f"them., its per-problem rows on a set it was measured on, or that set's own marker. L4p makes no round and no pretraining of its "
                                                          f"own: it reads what {ranks_task} wrote")
    with_a_change(rank_run, rewritten(rank_run, "report_ladder_l4_rank.json", change_of(ok=False)), "the run ladder_l4_pretrain_r64_seed0 cannot carry L4p: its report is not to be read")
    failing = lambda report: {**report, "inconclusive": True, "can_this_run_see_a_win": {**report["can_this_run_see_a_win"], "the_training_took": {"passes": False}}}      # noqa: E731
    with_a_change(rank_run, rewritten(rank_run, "report_ladder_l4_rank.json", failing), "the run ladder_l4_pretrain_r64_seed0 cannot carry L4p: its checks failed .the_training_took.")
    with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_prepare.done.json", change_of(seed=3)), "the run ladder_l4_pretrain_r64_seed0 recorded seed 3 and the model 'pre_r64'")
    with_a_change(rank_run, rows_of(rank_run, "l4_pretraining_rows.jsonl", lambda own: own[:-1]), "stored 11 rows `pre_r64` was trained on and L4's pretraining file holds 12")
    # ---- the pretraining file must be the one the start model was trained on
    changed = tmp_path / "changed.jsonl"
    changed.write_text(FIXTURE.read_text() + "\n")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    refused("L4's pretraining file has SHA-256 .* and `pre_r64` was trained on .* .ladder_l4_pretrain_r64_seed0.: the rehearsal rows are rows `pre_r64` was trained on")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    # ---- `rehearse_only`'s STORED SET must be the first rows of the one order: another row of the file in the place of one; a row with text; a row of a round; out of its order
    rehearse_only = _file_rows(l4t2 / "l4_rows2_training_rehearse_only.jsonl")
    not_drawn = _draw_order([{**row, "id": f"{row['problem_id']}#pretraining"} for row in _file_rows(rank_run / "l4_pretraining_rows.jsonl")])[-1]       # the LAST of the hash order
    another = {**rehearse_only[0], "id": not_drawn["id"], "problem_id": not_drawn["problem_id"], "row_of_the_file": not_drawn["row"]}
    assert not_drawn["id"] not in {row["id"] for row in rehearse_only}
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_rehearse_only.jsonl", lambda own: [another if row["id"] == rehearse_only[0]["id"] else row for row in own]),
                  f"L4t2's `rehearse_only` set holds {len(rehearse_only)} rows, and they are not the first {len(rehearse_only)} of the hash order among the 12 rows the start model was "
                  "pretrained on .*: that set was not drawn from this file with this seed, and the dose would not be along one order. Refused .L4t2's run ladder_l4_rows2_pre_r64_seed0; "
                  "`pre_r64`'s ladder_l4_pretrain_r64_seed0.. Nothing was written")
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_rehearse_only.jsonl", lambda own: [{**row, "row_of_the_file": (row["row_of_the_file"] + 1) % 12} if row["id"] == rehearse_only[0]["id"] else row
                                                                                           for row in own]),
                  "they are not the first .* of the hash order among the 12 rows")                    # the right id at another place of the file
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_rehearse_only.jsonl", lambda own: [{**row, "origin": "attempt"} if row["id"] == rehearse_only[0]["id"] else row for row in own]),
                  f"L4t2's `rehearse_only` set holds {len(rehearse_only)} rows and 1 of them are not rehearsal rows .first: {re.escape(rehearse_only[0]['id'])}.")
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_rehearse_only.jsonl", lambda own: list(reversed(own))),
                  "L4t2's `rehearse_only` set does not stand in the content-hash order of this seed")
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_rehearse_only.jsonl", lambda own: []), "L4t2's `rehearse_only` set holds 0 rows")
    # ---- `reward_rehearse`'s STORED SET: its published rows must be the first of the one order, by id and place, which `matched_1262` is trained on and nothing else
    reward_rehearse = _file_rows(l4t2 / "l4_rows2_training_reward_rehearse.jsonl")
    published = [row for row in reward_rehearse if row["origin"] == "pretraining"]
    elsewhere = {**published[0], "id": not_drawn["id"], "problem_id": not_drawn["problem_id"], "row_of_the_file": not_drawn["row"]}
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_reward_rehearse.jsonl", lambda own: [elsewhere if row["id"] == published[0]["id"] else row for row in own]),
                  f"L4t2's `reward_rehearse` set holds {len(published)} published rows, and they are not the first {len(published)} of the hash order among the 12 rows the start model "
                  "was pretrained on .*: `matched_1262` would not be the control of that set. Refused .L4t2's run ladder_l4_rows2_pre_r64_seed0; `pre_r64`'s "
                  "ladder_l4_pretrain_r64_seed0.. Nothing was written")
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_reward_rehearse.jsonl", lambda own: [{**row, "row_of_the_file": (row["row_of_the_file"] + 1) % 12} if row["id"] == published[0]["id"] else row
                                                                                             for row in own]),
                  "`matched_1262` would not be the control of that set")                                # the right id at another place of the file
    with_a_change(l4t2, rows_of(l4t2, "l4_rows2_training_reward_rehearse.jsonl", lambda own: [row for row in own if row["origin"] != "pretraining"]),
                  f"L4t2's `reward_rehearse` set holds 0 published rows among its {len(reward_rehearse) - len(published)} and `rehearse_only` holds {len(rehearse_only)}")
    # ---- a row that carries text is refused, and so is a set with a barred problem (the stage's own sets are made by the rule: the refusals are the functions L4t2 uses)
    held = _file_rows(l4t2 / "l4_heldout_groups.jsonl")[0]["problem_id"]
    of_the_loop_half = next(name for name in (f"p{index}" for index in range(200)) if half_of(name, config["ladder_loop"]["l4"]["half_seed"]) == LOOP)
    for changed, match in (({"proof": "text"}, "1 rehearsal rows of the training set of `{model}` carry a text field .first: .*#pretraining.: a rehearsal row is a published proof"),
                           ({"problem_id": held}, f"1 problems of the training set of `{{model}}` are held-out problems .first: {held}.: refused, nothing was written .L4t2's run "
                                                  "ladder_l4_rows2_pre_r64_seed0; `pre_r64`'s ladder_l4_pretrain_r64_seed0.. Nothing was written"),
                           ({"problem_id": of_the_loop_half}, f"1 problems of the training set of `{{model}}` are rehearsal rows that are not of the `pretrain` half .first: {of_the_loop_half}.")):
        for wrong in MODELS:                                                   # in each of the three sets, the two others as the rule made them
            with monkeypatch.context() as patched:
                patched.setattr(ladder_l4_dose, "dose_sets", lambda *given, changed=changed, wrong=wrong: {
                    model: [{**row, **changed} if index == 0 and model == wrong else row for index, row in enumerate(own)] for model, own in dose_sets(*given).items()})
                refused(match.format(model=wrong))
    # ---- the settings are the stage's: its two models in their order; a start that is not the one L4t2 ran from
    settings = config["ladder_loop"]["l4"]["dose"]
    for others in ([HALF, MATCHED, FULL], [HALF, FULL]):
        monkeypatch.setitem(settings, "models", others)
        refused("ladder_loop.l4.dose.models is .* and the stage trains and measures .'matched_1262', 'second_half', 'second_full'.: change both together", ValueError)
    monkeypatch.setitem(settings, "models", list(MODELS))
    for other in ("pre", "pre_r32", "the_base"):
        monkeypatch.setitem(settings, "start", other)
        with pytest.raises(ValueError, match=f"ladder_loop.l4.dose.start is `{other}` and ladder_loop.l4.rows2.start is `pre_r64`: `rehearse_only`, the smallest dose, is L4t2's model, "
                                             "trained from that start, and the dose is of ONE start model"):
            ladder_l4_dose.ladder_l4_dose_prepare(config)
        assert not (root.parent / f"ladder_l4_dose_{other}_seed0").exists()
    monkeypatch.setitem(settings, "start", "pre_r64")
    # ---- a task of another seed reads that seed's run of L4t2, which is not on the box
    monkeypatch.setenv(SEEDS, "1")
    with pytest.raises(RuntimeError, match="ladder_l4_rows2_pre_r64_seed1 does not hold .*the task of stage `ladder_l4_rows2_r64` for seed 1 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rows2_r64 --seeds 1`"):
        ladder_l4_dose.ladder_l4_dose_prepare(config)
    assert dose.store().root.name == "ladder_l4_dose_pre_r64_seed1" and not _all(dose.store().root) and not (root.parent / "ladder_l4_rows2_pre_r64_seed1").exists()
    monkeypatch.delenv(SEEDS)
    # ---- as on the GPU: without the start adapter nothing is prepared
    adapter = rank_run / "adapters" / "pre_r64"
    adapter.rename(adapter.with_name("elsewhere"))
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        refused(f"adapters/pre_r64 is not there: the adapter `pre_r64` is what every model here is trained from. Run {ranks_task} again; nothing was written")
    adapter.with_name("elsewhere").rename(adapter)
    # ---- the later steps refuse out of their order; and with everything as it was, it prepares, the two runs untouched; L4t2's adapters and L4t's whole run are NOT read
    for step, match in ((ladder_l4_dose.STEPS["ladder_l4_dose_train_second_half"], "the training of `second_half` needs the step ladder_l4_dose_prepare of this run, which is not "
                                                                                   "done: the stage `ladder_l4_dose_r64`"),
                        (ladder_l4_dose.STEPS["ladder_l4_dose_measure_second_full"], "the measurement of L4p's model `second_full` needs the step ladder_l4_dose_prepare"),
                        (ladder_l4_dose.ladder_l4_dose_report, "L4p's report needs the step ladder_l4_dose_prepare of this run, which is not done: the stage `ladder_l4_dose_r64`"),
                        (ladder_l4_dose.ladder_l4_dose_report_matched, "L4p's report needs the step ladder_l4_dose_prepare of this run, which is not done: the stage "
                                                                       "`ladder_l4_dose_r64_matched`")):
        with pytest.raises(RuntimeError, match=match):
            step(config)
    shutil.rmtree(l4t2 / "adapters")
    for directory in (dose.l4t, dose.arm, dose.pre):
        directory.rename(directory.with_name(f"{directory.name}_elsewhere"))
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        assert ladder_l4_dose.ladder_l4_dose_prepare(config)["start"] == "pre_r64"
    assert _all(rank_run) == as_stored[rank_run] and {name: content for name, content in as_stored[l4t2].items() if not name.startswith("adapters/")} == _all(l4t2)
    with pytest.raises(RuntimeError, match="the measurement of L4p's model `second_half` needs the step ladder_l4_dose_train_second_half of this run, which is not done"):
        ladder_l4_dose.STEPS["ladder_l4_dose_measure_second_half"](config)
    for report_step in (ladder_l4_dose.ladder_l4_dose_report, ladder_l4_dose.ladder_l4_dose_report_matched):
        with pytest.raises(RuntimeError, match="L4p's report needs the step ladder_l4_dose_train_matched_1262"):
            report_step(config)


def test_a_smoke_run_goes_on_from_a_start_whose_checks_failed_only_where_l4t2s_own_run_recorded_that_it_did(dose, monkeypatch):
    config, l4t2, rank_run = dose.config, dose.l4t2, dose.rank
    failing = json.loads((rank_run / "report_ladder_l4_rank.json").read_text())
    failing = {**failing, "inconclusive": True, "can_this_run_see_a_win": {**failing["can_this_run_see_a_win"], "the_training_took": {"passes": False}}}
    (rank_run / "report_ladder_l4_rank.json").write_text(json.dumps(failing))
    # L4t2's run recorded no waiver (a real run): refused, whatever L4t's own variable says in this task's environment.
    monkeypatch.setenv(L4T_START_CHECKS, "smoke")
    with pytest.raises(RuntimeError, match="the run ladder_l4_pretrain_r64_seed0 cannot carry L4p: its checks failed .the_training_took."):
        ladder_l4_dose.ladder_l4_dose_prepare(config)
    monkeypatch.delenv(L4T_START_CHECKS)
    # L4t2's run recorded that IT went on from the failed checks (a smoke run's alone does): this stage goes on too, sets nothing for it, and records it.
    of_l4t2 = json.loads((l4t2 / "ladder_l4_rows2_prepare.done.json").read_text())
    (l4t2 / "ladder_l4_rows2_prepare.done.json").write_text(json.dumps({**of_l4t2, "start_run_read": {**of_l4t2["start_run_read"], "checks_waived_as_l4ts_run_recorded": True}}))
    read = ladder_l4_dose.ladder_l4_dose_prepare(config)["start_run_read"]
    assert (read["checks_failed"], read["checks_waived_as_l4t2s_run_recorded"]) == (["the_training_took"], True)
    # A report that is not to be read is refused for it too.
    shutil.rmtree(dose.store().root)
    (rank_run / "report_ladder_l4_rank.json").write_text(json.dumps({**failing, "ok": False}))
    with pytest.raises(RuntimeError, match="cannot carry L4p: its report is not to be read"):
        ladder_l4_dose.ladder_l4_dose_prepare(config)
    assert not [name for name, own in entry.STAGES.items() if name.startswith("ladder_l4_dose") and any(L4T_START_CHECKS in entry.step_fields(step)[2].get("environment", {}) for step in own)]


def test_rehearse_only_read_as_inconclusive_by_l4t2s_own_report_is_carried_and_the_primary_is_not_read_against_it(dose, monkeypatch):
    config, l4t2 = dose.config, dose.l4t2
    report = json.loads((l4t2 / "report_ladder_l4_rows2.json").read_text())
    own = report["models"]["rehearse_only"]
    report["models"]["rehearse_only"] = {**own, "can_this_model_be_read": {key: ({**said, "passes": key != "its_training_ran"} if isinstance(said, dict) else said)
                                                                             for key, said in own["can_this_model_be_read"].items()}}
    other = report["models"]["reward_rehearse"]
    report["models"]["reward_rehearse"] = {**other, "can_this_model_be_read": {key: ({**said, "passes": key != "lean_answered"} if isinstance(said, dict) else said)
                                                                               for key, said in other["can_this_model_be_read"].items()}}      # each model's is its own
    (l4t2 / "report_ladder_l4_rows2.json").write_text(json.dumps(report))
    summaries = _stage_p(config, monkeypatch)
    carried = {"rehearse_only": ["its_training_ran"], "reward_rehearse": ["lean_answered"]}
    assert summaries["ladder_l4_dose_prepare"]["l4t2_read"]["failed_checks"] == carried
    built = summaries["ladder_l4_dose_report"]
    assert (built["branch"]["name"], built["branch"]["not_read_against"]) == (NOT_READ, "rehearse_only") and built["primary"]["mean"] is None and built["inconclusive"] is False
    assert built["the_dose"]["against_the_start"]["rehearse_only"]["read"] is False and built["the_stored_models"]["what_l4t2_read"]["failed_checks"] == carried
    # ... and the rounds' hard rows: `matched_1262`'s OWN check fails in this world (the scripted loss of a training of one optimizer step rises), which is said first.
    assert built["models"][MATCHED]["failed_checks"] == ["its_training_ran"] and (built["the_rounds_hard_rows"]["name"], built["the_rounds_hard_rows"]["failed_checks"]) == (
        INCONCLUSIVE, ["its_training_ran"]) and "not_read_against" not in built["the_rounds_hard_rows"]
    # (In this world the scripted loss of a training of ONE optimizer step rises: `matched_1262` and `second_half` are each INCONCLUSIVE by itself; `second_full`, twelve rows, is read.)
    assert not any(": PRIMARY. " in line for line in built["lines"]) and (built["models"][HALF]["failed_checks"], built["models"][FULL]["failed_checks"]) == (["its_training_ran"], [])
    # ... and with a `rehearse_only` that report read as passing every check, the primary IS read, against the rows L4t2's run stored for it: 32 and 48 attempts on the one
    # goal problem, of which `rehearse_only` is given two more successes than the scripted solver gives every model.
    shutil.rmtree(dose.store().root)
    for model, entry in (("rehearse_only", own), ("reward_rehearse", other)):
        report["models"][model] = {**entry, "can_this_model_be_read": {key: ({**said, "passes": True} if isinstance(said, dict) else said) for key, said in entry["can_this_model_be_read"].items()}}
    (l4t2 / "report_ladder_l4_rows2.json").write_text(json.dumps(report))
    stored = _file_rows(l4t2 / "episodes_l4_more_rehearse_only_problems.jsonl")
    (l4t2 / "episodes_l4_more_rehearse_only_problems.jsonl").write_text("".join(json.dumps({**row, "resolved": row["resolved"] + 2}) + "\n" for row in stored))
    of_the_other = _file_rows(l4t2 / "episodes_l4_more_reward_rehearse_problems.jsonl")       # ... and `reward_rehearse` three FEWER on its fresh attempts than the solver gives every model
    (l4t2 / "episodes_l4_more_reward_rehearse_problems.jsonl").write_text("".join(json.dumps({**row, "resolved": row["resolved"] - 3}) + "\n" for row in of_the_other))
    built = _stage_p(config, monkeypatch)["ladder_l4_dose_report"]
    attempts = built["sizes"]["attempts_a_goal_problem"]
    assert built["branch"]["name"] == MORE_COSTS and (built["primary"]["mean"], built["primary"]["high"], built["primary"]["problems"]) == (round(-2 / attempts, 5), round(-2 / attempts, 5), 1)
    assert built["primary"]["successes_of_the_base"] == built["primary"]["successes"] + 2 and built["the_dose"]["against_the_start"]["rehearse_only"]["per_attempt"]["mean"] == round(2 / attempts, 5)
    assert any(f": PRIMARY. All of G (1 problems), successes per attempt over the {attempts} attempts a problem, `second_full` minus `rehearse_only`" in line for line in built["lines"])
    # `second_half` and `matched_1262` are each INCONCLUSIVE by itself: the primary does not stand on them; the steps of the shape do, and are not read (what was measured is kept).
    assert [step["read"] for step in built["the_dose"]["steps"].values()] == [False, False, False] and built["overfitting"][HALF]["narrower"] is None and built["overfitting"][FULL]["narrower"] is False
    assert built["the_dose"]["steps"]["rehearse_only_minus_matched_1262"]["measured_and_not_read"]["mean"] == round(2 / attempts, 5)
    # THE ROUNDS' HARD ROWS stand on the stored `reward_rehearse`'s rows and `matched_1262`'s own (this run's). `matched_1262` is INCONCLUSIVE here, so nothing of it is read; what
    # was measured is kept apart: no G' in this world (the start model solves its one goal problem at once), and on all of G the difference is the three fewer.
    hard = built["the_rounds_hard_rows"]
    kept = hard["beside"]["measured_and_not_read"]["on_all_of_g"]
    assert (hard["name"], hard["difference"]["mean"], hard["difference"]["measured_and_not_read"]["problems"], kept["mean"], kept["problems"]) == (INCONCLUSIVE, None, 0, round(-3 / attempts, 5), 1)
    assert kept["successes_of_the_base"] == kept["successes"] + 3 and "not_read_against" not in hard and built["branches"] == {"the_dose": MORE_COSTS, "the_rounds_hard_rows": INCONCLUSIVE}


def test_as_on_the_gpu_each_measure_step_serves_its_adapter_at_rank_64_and_a_model_at_another_rank_fails_its_training_unmarked(dose, monkeypatch):
    config, calls, kits = dose.config, dose.calls, dose.kits
    ladder_l4_dose.ladder_l4_dose_prepare(config)
    store = dose.store()
    train, measure = ladder_l4_dose.STEPS["ladder_l4_dose_train_second_half"], ladder_l4_dose.STEPS["ladder_l4_dose_measure_second_half"]
    # A training whose saved adapter is not the start adapter's rank fails its step, unmarked, and nothing of it is measured (L4t's own body, under this stage's names).
    _trainings(monkeypatch, calls, saves={HALF: (16, 32, None)})
    with pytest.raises(RuntimeError, match=r"the adapter `second_half` saved at .*adapters/second_half has rank 16 and alpha 32 in its adapter_config.json and matrices of rank .16.; "
                                           r"the start adapter `pre_r64` is rank 64 and alpha 128"):
        _on_the_gpu(monkeypatch, train, config)
    assert not store.is_done("ladder_l4_dose_train_second_half")
    with pytest.raises(RuntimeError, match="the measurement of L4p's model `second_half` needs the step ladder_l4_dose_train_second_half of this run, which is not done: the stage "
                                           "`ladder_l4_dose_r64`"):
        measure(config)
    shutil.rmtree(store.root / "adapters" / HALF)
    _trainings(monkeypatch, calls)
    assert _on_the_gpu(monkeypatch, train, config)["adapter_saved"]["ranks"] == [64] and store.is_done("ladder_l4_dose_train_second_half")
    # The stand-in trains and saves nothing: nothing is read back.
    assert "adapter_saved" not in ladder_l4_dose.STEPS["ladder_l4_dose_train_second_full"](config)
    # As on the GPU: the measure step serves ITS adapter, under a name of its own, from a model server started for rank 64 at the config's temperature.
    del kits[:]
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        measure(config)
        with pytest.raises(RuntimeError, match="adapters/second_full is not there: the adapter `second_full` is kept by this stage"):
            ladder_l4_dose.STEPS["ladder_l4_dose_measure_second_full"](config)
    assert [(kit["temperature"], kit["adapter"], kit["max_lora_rank"]) for kit in kits] == [(1.0, ("ladder_l4_dose_r64_second_half", 1, str(store.root / "adapters" / HALF)), 64)] * 3
    # The check's setting moved after the run was prepared: refused before anything is trained or sampled.
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "alpha", 64)
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "fallback_alpha", 32)
    with pytest.raises(RuntimeError, match="this run was prepared from `pre_r64` at rank 64 and alpha 128 .what the run that made it recorded., and ladder_loop.l4.rank_check now gives it "
                                           "rank 64 and alpha 64"):
        ladder_l4_dose.STEPS["ladder_l4_dose_measure_second_full"](config)


def test_a_set_lean_did_not_answer_fails_the_report_and_is_sampled_again_from_the_kept_adapter_when_the_task_is_queued_again(dose, monkeypatch, tmp_path):
    config, calls = dose.config, dose.calls
    first = _stage_p(config, monkeypatch, until="ladder_l4_dose_measure_second_full")
    store = dose.store()
    unanswered = "l4_reach_second_full"
    name = f"episodes_{unanswered}_problems.jsonl"
    as_measured, own = store.path(name).read_bytes(), store.read_rows(name)
    own[0]["attempts_without_an_answer"] = 20
    store.write_rows(name, own)
    monkeypatch.setattr(milestone2, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", ladder_l4_dose.REPORT, "--out", str(tmp_path / "report.json")])
    assert gpu_main.main() != 0                                                # the report is written, is not to be read, and its step fails
    report = json.loads(store.path(ladder_l4_dose.REPORT_FILE).read_text())
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_second_full_1"] and "NOT TO BE READ" in report["lines"][-1] and report["branch"]["name"] == INCONCLUSIVE
    assert "lean_answered" in report["models"][FULL]["failed_checks"] and "lean_answered" not in report["models"][HALF]["failed_checks"]
    assert report["models"][FULL]["can_this_model_be_read"]["lean_answered"]["sets_not_answered"] == ["goal_second_full_1"] and store.done_summary(ladder_l4_dose.REPORT)["ok"] is False
    # The task queued again: that set, and no other, is sampled again from the kept adapter; nothing is trained.
    untouched = {key: content for key, content in _all(store.root).items() if unanswered not in key and "report" not in key and "measure_second_full" not in key}
    sampled = dose.solver.calls
    again = _stage_p(config, monkeypatch)
    assert len(calls) == 3 and again["ladder_l4_dose_measure_second_full"]["measured_again"] == [unanswered] and dose.solver.calls > sampled
    after = _all(store.root)
    assert all(after[key] == content for key, content in untouched.items()) and after[name] == as_measured
    assert again["ladder_l4_dose_report"]["ok"] is True and again["ladder_l4_dose_report"]["not_to_be_read"] == []
    assert first["ladder_l4_dose_measure_second_half"] == again["ladder_l4_dose_measure_second_half"]


def test_the_steps_run_through_the_gpu_entry_point(dose, monkeypatch, tmp_path):
    monkeypatch.setattr(milestone2, "load_config", lambda path: dose.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert dose.store().is_done(ladder_l4_dose.REPORT)


# ------------------------------------------------------------------------ two jobs, one run directory
def test_the_job_of_matched_1262_alone_writes_its_read_and_the_whole_stage_then_finds_it_done_and_goes_on(dose, monkeypatch):
    config, calls, kits = dose.config, dose.calls, dose.kits
    # ---- THE FIRST JOB: the prepare step, `matched_1262` trained and measured, and the report with it alone measured
    first = _stage_p(config, monkeypatch, stage_name="ladder_l4_dose_r64_matched")
    store = dose.store()
    assert list(first) == MATCHED_STEPS and store.root.name == "ladder_l4_dose_pre_r64_seed0" and [next(iter(call["schedule"]["checkpoint_steps"])) for call in calls] == [MATCHED]
    assert len(kits) == 3 and sorted(path.name for path in (store.root / "adapters").iterdir()) == [MATCHED]
    alone = first["ladder_l4_dose_report_matched"]
    assert list(alone) == REPORT_KEYS and (alone["measured"], alone["not_measured_yet"], alone["ok"], alone["inconclusive"]) == ([MATCHED], [HALF, FULL], True, False)
    assert (alone["branch"]["name"], alone["branch"]["not_measured"], alone["branches"]["the_dose"]) == (NOT_READ, FULL, NOT_READ) and alone["primary"]["mean"] is None
    assert alone["branches"]["the_rounds_hard_rows"] in MATCHED_BRANCHES and alone["models"][MATCHED]["measured"] is True and list(alone["models"]) == list(MODELS)
    assert all(alone["models"][model] == {"model": alone["models"][model]["model"], "measured": False, "said": f"NOT READ: `{model}` is not measured yet (the job that trains and "
                                          f"measures `matched_1262` alone writes this report)"} for model in (HALF, FULL))
    assert any("THIS IS THE REPORT OF THE JOB THAT TRAINS AND MEASURES `matched_1262` ALONE: `second_half` and `second_full` are NOT MEASURED YET" in line for line in alone["lines"])
    assert INCONCLUSIVE not in alone["branch"]["reason"] and alone["adapters"]["there"] == [MATCHED] and "`matched_1262`" in alone["adapters"]["what"] and "`second_half`" not in alone["adapters"]["what"]
    assert json.loads(store.path(ladder_l4_dose.REPORT_FILE).read_text()) == json.loads(json.dumps(alone, default=str))
    assert store.done_summary("ladder_l4_dose_report_matched") == {"stage": "l4", "label": "pretrained on published proofs", "check": "L4p", "headline": alone["headline"],
                                                                   "branches": alone["branches"], "measured": [MATCHED], "ok": True}
    # The prepare step wrote ALL THREE sets and the problems of every set the stage samples: the whole stage has nothing to prepare again.
    assert all(store.path(f"l4_dose_training_{model}.jsonl").exists() for model in MODELS) and not store.is_done(ladder_l4_dose.REPORT)
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l4_{part}_{model}" for part in ("rungs", REACH, MORE) for model in MODELS}
    # The whole stage's own report REFUSES while the two other models are not done: it never writes a report with a model missing.
    with pytest.raises(RuntimeError, match="L4p's report needs the step ladder_l4_dose_train_second_half of this run, which is not done: the stage `ladder_l4_dose_r64`"):
        ladder_l4_dose.ladder_l4_dose_report(config)
    # A model is MEASURED when both its steps are done: with a measure step's marker alone in the store (nothing trained), the first job's report is what it was.
    store.mark_done("ladder_l4_dose_measure_second_half", {"stage": "l4", "model": HALF})
    assert ladder_l4_dose.ladder_l4_dose_report_matched(config)["measured"] == [MATCHED]
    store.path("ladder_l4_dose_measure_second_half.done.json").unlink()
    after_the_first = _all(store.root)
    # ---- THE SECOND JOB, the whole stage: it finds the prepare step and `matched_1262` done, trains and measures the two others, and writes the whole report
    sampled = dose.solver.calls
    second = _stage_p(config, monkeypatch)
    assert list(second) == STEPS and [next(iter(call["schedule"]["checkpoint_steps"])) for call in calls] == [MATCHED, HALF, FULL] and len(kits) == 3 * 3 and dose.solver.calls > sampled
    assert all(second[step] == first[step] for step in MATCHED_STEPS[:3])                          # what the first job's three steps returned, returned again: nothing of them ran
    kept = {name: content for name, content in after_the_first.items() if name not in ("report_ladder_l4_dose.json",)}
    assert all(_all(store.root)[name] == content for name, content in kept.items())                # every file of the first job is as it left it, its report's marker among them
    whole = second["ladder_l4_dose_report"]
    assert (whole["measured"], whole["not_measured_yet"]) == (list(MODELS), []) and "not_measured" not in whole["branch"] and whole["branch"]["name"] in BRANCHES
    assert whole["the_rounds_hard_rows"] == alone["the_rounds_hard_rows"] and whole["models"][MATCHED] == alone["models"][MATCHED]      # `matched_1262`'s read is what the first job read
    assert all(whole["models"][model]["measured"] is True for model in MODELS) and store.done_summary(ladder_l4_dose.REPORT)["measured"] == list(MODELS)
    assert json.loads(store.path(ladder_l4_dose.REPORT_FILE).read_text()) == json.loads(json.dumps(whole, default=str))
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == sorted([*OWN_FILES, "ladder_l4_dose_report_matched.done.json"])
    # ---- THE FIRST JOB QUEUED AGAIN after the whole stage ran: nothing is trained or sampled, and its report is the WHOLE one (every model that is measured is read)
    trained, sampled = len(calls), dose.solver.calls
    again = _stage_p(config, monkeypatch, stage_name="ladder_l4_dose_r64_matched")
    assert (len(calls), dose.solver.calls) == (trained, sampled) and again["ladder_l4_dose_report_matched"]["measured"] == list(MODELS)
    assert {key: value for key, value in again["ladder_l4_dose_report_matched"].items() if key != "adapters"} == {key: value for key, value in whole.items() if key != "adapters"}


def test_the_whole_stage_alone_runs_all_three_and_leaves_no_marker_of_the_other_job(dose, monkeypatch):
    summaries = _stage_p(dose.config, monkeypatch)
    store = dose.store()
    assert list(summaries) == STEPS and not store.is_done("ladder_l4_dose_report_matched") and sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert summaries["ladder_l4_dose_report"]["measured"] == list(MODELS)


# ------------------------------------------------- L4t2, and L4t from `pre` and from `pre_r64`, are what they were
def test_l4t2_and_l4t_from_pre_r64_write_what_they_wrote_and_are_left_as_they_were_after_this_stage_ran_beside_them(dose, monkeypatch):
    config, l4t2, l4t = dose.config, dose.l4t2, dose.l4t
    as_it_wrote = {path: _all(path) for path in (l4t2, l4t)}
    # What L4t2 leaves: the files its own tests froze, 61 in all in this world; its six steps; its report's three branches. What L4t from `pre_r64` leaves: its 100.
    assert sorted(name for name in as_it_wrote[l4t2] if not name.startswith("episodes_")) == L4T2_FILES and len(as_it_wrote[l4t2]) == 61
    assert list(dose.l4t2_summaries) == L4T2_STEPS == list(ladder_l4_rows2.STEPS) and list(dose.l4t2_summaries["ladder_l4_rows2_report"]["branches"]) == ["question_1", "question_2", "the_arms_rule"]
    assert sorted(name for name in as_it_wrote[l4t] if not name.startswith("episodes_")) == sorted(L4T_FILES_FROM_PRE_R64) and len(as_it_wrote[l4t]) == 100
    assert list(dose.rows2.l4t_summaries) == L4T_STEPS_FROM_PRE_R64 == ladder_l4_rows.steps_of("pre_r64")
    _stage_p(config, monkeypatch)
    assert {path: _all(path) for path in as_it_wrote} == as_it_wrote and not [name for files in as_it_wrote.values() for name in files if "dose" in name or any(model in name for model in MODELS)]
    # L4t2's own task, queued again AFTER this stage ran in the same store: every step returns what is stored, nothing is trained or sampled, no file moves.
    trained, sampled = len(dose.calls), dose.solver.calls
    again = _l4t2(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in dose.l4t2_summaries.items() if "report" not in name}
    assert {key: value for key, value in again["ladder_l4_rows2_report"].items() if key != "adapters"} == {
        key: value for key, value in dose.l4t2_summaries["ladder_l4_rows2_report"].items() if key != "adapters"}
    # ... and L4t's own, from `pre_r64`.
    monkeypatch.setenv(L4T_START, "pre_r64")
    monkeypatch.setenv(L4T_MINIMUM, "2")
    of_l4t = _l4t(config, monkeypatch)
    assert {name: summary for name, summary in of_l4t.items() if "report" not in name} == {name: summary for name, summary in dose.rows2.l4t_summaries.items() if "report" not in name}
    assert {key: value for key, value in of_l4t["ladder_l4_rows_report"].items() if key != "adapters"} == {
        key: value for key, value in dose.rows2.l4t_summaries["ladder_l4_rows_report"].items() if key != "adapters"}
    assert (len(dose.calls), dose.solver.calls) == (trained, sampled) and {path: _all(path) for path in as_it_wrote} == as_it_wrote
    # Their names are their own as they were: the stage, its files, its markers; this stage's are beside them and never the same.
    of_l4t_stage, of_l4t2_stage, own = ladder_l4_rows.THE_STAGE, ladder_l4_rows2.THE_STAGE, ladder_l4_dose.THE_STAGE
    assert (of_l4t_stage.name, of_l4t_stage.steps, of_l4t_stage.files, of_l4t_stage.reader, of_l4t_stage.prepare) == ("ladder_l4_rows", "ladder_l4_rows", "l4_rows", "L4t", "ladder_l4_rows_prepare")
    assert (of_l4t2_stage.name, of_l4t2_stage.steps, of_l4t2_stage.files, of_l4t2_stage.reader, of_l4t2_stage.prepare, of_l4t2_stage.start_said_by) == (
        "ladder_l4_rows2_r64", "ladder_l4_rows2", "l4_rows2", "L4t2", "ladder_l4_rows2_prepare", "ladder_loop.l4.rows2.start")
    assert (own.name, own.steps, own.files, own.reader, own.prepare, own.start_said_by) == ("ladder_l4_dose_r64", "ladder_l4_dose", "l4_dose", "L4p", "ladder_l4_dose_prepare",
                                                                                        "ladder_loop.l4.dose.start")
    assert not (set(ladder_l4_rows.STEPS) | set(ladder_l4_rows2.STEPS)) & set(ladder_l4_dose.STEPS) and (own.training_file(HALF), own.loss_file(FULL)) == (
        "l4_dose_training_second_half.jsonl", "l4_dose_loss_second_full.json")
    assert ladder_l4_rows2.the_settings(config).models == ("rehearse_only", "reward_rehearse") and config["ladder_loop"]["l4"]["rows2"] == {"start": "pre_r64", "models": ["rehearse_only", "reward_rehearse"]}


def test_l4t_from_pre_writes_what_it_wrote(rows, monkeypatch):  # noqa: F811
    config = rows.config
    summaries = _l4t_from_pre(config, monkeypatch)
    written = _all(rows.store().root)
    # What L4t from `pre` leaves: the files its own tests froze, 79 in all in this world; its eight steps; no step, file or key of another start or of this stage.
    assert rows.store().root.name == "ladder_l4_rows_seed0" and sorted(name for name in written if not name.startswith("episodes_")) == L4T_FILES_FROM_PRE and len(written) == 79
    assert list(summaries) == L4T_STEPS_FROM_PRE == ladder_l4_rows.steps_of("pre") and not [name for name in written if "dose" in name or "second_" in name or "rows2" in name]
    assert [call["tensorboard_run"] for call in rows.calls] == ["ladder_l4_rows_seed0_rehearse", "ladder_l4_rows_seed0_reward_rows"]
    assert ladder_l4_rows.the_settings(config).start == "pre" and config["ladder_loop"]["l4"]["rows"]["start"] == "pre"
    # This stage does not run from `pre`: L4t2 is specified from `pre_r64` alone, and the dose is of its start.
    assert ladder_l4_dose.the_settings(config).start == "pre_r64" == ladder_l4_dose.the_settings(config).l4t2.start


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_reads_l4t2s_smoke_run_and_the_rank_checks_and_nothing_else(arm, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RUN, L4T2_RUN, L4T2_OWN_RUN, L4T2_L4T_RUN, L4T_OWN_RUN, L4T_ARM_RUN, L4T_MINIMUM, L4T_START, L4T_START_CHECKS, ladder_l4_rank.CHECK_VARIABLE, RANK_RUN):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings(monkeypatch, calls)
    engines = lambda self, adapter=None: (lambda given: (kits.append((given["sampling"]["temperature"], given["vllm"]["max_lora_rank"])), (arm.solver, stand_in_parameters))[1])      # noqa: E731
    # ---- the five smoke tasks before it, each in ITS OWN environment: the pretraining's, the arm's, the check's, L4t's from `pre_r64`, then L4t2's
    steps = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS, **ladder_l4_rank.STEPS, **ladder_l4_rows.STEPS, **ladder_l4_rows2.STEPS}
    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
        options = entry.step_fields(entry.STAGES[stage_name][2])[2]
        with monkeypatch.context() as patched:                                 # one task's environment: gone when the task is
            patched.delenv(ladder_loop.DATA_VARIABLE)
            for name, value in options["environment"].items():
                patched.setenv(name, value)
            if stage_name == "ladder_l4_smoke":
                arm.with_lean()
                patched.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
            if stage_name in ("ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"):
                patched.setattr(ladder_round.Engines, "kit", engines)
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                if environment == "gpu" and step in steps:
                    _on_the_gpu(patched, steps[step], config) if "_train" in step else steps[step](config)
    runs = pipeline.STORE / "runs-v4.27"
    l4t2, rank_run = runs / "ladder_l4_rows2_r64_smoke", runs / "ladder_l4_rank_smoke"
    assert (l4t2 / "report_ladder_l4_rows2.json").exists() and (rank_run / "adapters" / "pre_r64" / "adapter_config.json").exists()
    as_stored = {path: _all(path) for path in runs.iterdir() if path.is_dir()}
    assert set(path.name for path in as_stored) >= {"ladder_l1_smoke", "ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke", "ladder_l4_rows2_r64_smoke"}
    # ---- this stage's smoke task: L4t2's smoke run and the check's as the two it reads, the fixtures, a run directory of its own; NO variable of L4t's or of L4t2's
    options = entry.step_fields(entry.STAGES["ladder_l4_dose_r64_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_dose_r64_smoke"])
    of_the_pretraining = entry.step_fields(entry.STAGES["ladder_l4_pretrain_smoke"][2])[2]["environment"]
    assert options["environment"] == {ladder_loop.DATA_VARIABLE: of_the_pretraining[ladder_loop.DATA_VARIABLE], ladder_l4.FILE_VARIABLE: str(FIXTURE), RANK_RUN: "ladder_l4_rank_smoke",
                                      L4T2_RUN: "ladder_l4_rows2_r64_smoke", RUN: "ladder_l4_dose_r64_smoke"}
    of_l4t2s_smoke = entry.step_fields(entry.STAGES["ladder_l4_rows2_r64_smoke"][2])[2]["environment"]
    assert (of_l4t2s_smoke[L4T2_OWN_RUN], of_l4t2s_smoke[RANK_RUN]) == (options["environment"][L4T2_RUN], options["environment"][RANK_RUN])      # the two runs it reads are the ones those tasks write
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ladder_round.Engines, "kit", engines)
    del calls[:], kits[:]
    # It reads NOTHING else: with the four other smoke runs gone from the box (L4t's among them) it runs to its end.
    others = ("ladder_l1_smoke", "ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rows_r64_smoke")
    for name in others:
        (runs / name).rename(runs / f"{name}_elsewhere")
    summaries = _stage_p(config, monkeypatch, stage_name="ladder_l4_dose_r64_smoke")
    for name in others:
        (runs / f"{name}_elsewhere").rename(runs / name)
    store = ladder_l4_dose._store(config, ladder_l4_dose.the_settings(config))
    assert store.root.name == "ladder_l4_dose_r64_smoke" and list(summaries) == STEPS and {path: _all(path) for path in as_stored} == as_stored
    assert sorted(path.name for path in runs.iterdir() if path.name.startswith("ladder_l4_dose")) == ["ladder_l4_dose_r64_smoke"]
    prepare = summaries["ladder_l4_dose_prepare"]
    assert (prepare["start"], prepare["start_run"], prepare["start_adapter"], prepare["l4t2_run"]) == ("pre_r64", str(rank_run), str(rank_run / "adapters" / "pre_r64"), str(l4t2))
    assert [sampling["name"] for sampling in prepare["goal_samplings"]] == [REACH] and prepare["attempts_a_goal_problem"] == 32 and prepare["start_recipe"]["rank"] == 64
    # Whether it went on from a start model whose own checks failed is what L4t2's smoke run RECORDED of itself, and nothing this stage sets.
    of_l4t2 = json.loads((l4t2 / "ladder_l4_rows2_prepare.done.json").read_text())
    assert prepare["start_run_read"]["checks_waived_as_l4t2s_run_recorded"] is of_l4t2["start_run_read"]["checks_waived_as_l4ts_run_recorded"]
    assert prepare["start_run_read"]["checks_failed"] == of_l4t2["start_run_read"]["checks_failed"] and L4T_START_CHECKS not in options["environment"]
    # THE SMOKE'S SIZES are the rule's own on the 12-row fixture: `reward_rehearse`'s published rows, then 6 and 12, with `rehearse_only`'s rows between: four doses that differ.
    sets = prepare["training_sets"]
    least, published = of_l4t2["training_sets"]["rehearse_only"]["rows"], of_l4t2["training_sets"]["reward_rehearse"]["rehearsal_rows"]
    assert (sets[HALF]["rows"], sets[FULL]["rows"], prepare["rows_the_start_was_pretrained_on"], prepare["rehearse_only"]["rows"]) == (6, 12, 12, least) and 0 < published < least < 6
    assert (sets[MATCHED]["rows"], prepare["reward_rehearse"]["published_rows"]) == (published, published)
    matched, half, full = (store.read_rows(f"l4_dose_training_{model}.jsonl") for model in MODELS)
    order = _draw_order(full)
    assert sorted(map(_placed, matched)) == sorted(map(_placed, (row for row in _file_rows(l4t2 / "l4_rows2_training_reward_rehearse.jsonl") if row["origin"] == "pretraining")))
    assert list(map(_placed, _draw_order(matched))) == list(map(_placed, order[:published]))
    assert list(map(_placed, _draw_order(_file_rows(l4t2 / "l4_rows2_training_rehearse_only.jsonl")))) == list(map(_placed, order[:least])) and list(map(_placed, _draw_order(half))) == list(map(_placed, order[:6]))
    assert sorted(row["row_of_the_file"] for row in full) == list(range(12))
    # BOTH models are trained from the smoke `pre_r64` AT RANK 64 and measured under a model server started for rank 64, at the config's temperature.
    assert [(call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(64, 128, {"start": rank_run / "adapters" / "pre_r64"})] * 3
    assert [list(call["schedule"]["checkpoint_steps"].items()) for call in calls] == [[(MATCHED, 1)], [(HALF, 1)], [(FULL, 2)]]
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == sorted(MODELS) and kits == [(1.0, 64)] * 6       # the rungs and G for each model
    report = summaries["ladder_l4_dose_report"]
    assert report["label"] == "pretrained on published proofs" and all(line.startswith(f"{SAY}: ") for line in report["lines"]) and store.is_done(ladder_l4_dose.REPORT)
    # ONE sampling of G: no fresh attempts, so the rounds' hard rows and ADDS are not read; the primary is read on the one sampling, or is INCONCLUSIVE.
    assert report["branch"]["name"] in BRANCHES and report["the_three_reads"]["fresh_attempts"] is None and report["the_rounds_hard_rows"]["name"] in (NOT_READ, INCONCLUSIVE)
    assert report["measured"] == list(MODELS) and list(summaries) == STEPS
    assert any("the stored models have ONE sampling of G here (a smoke run)" in line for line in report["lines"])


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_names_no_variable():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_dose_r64"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l4_dose.STEPS) == [*STEPS, MATCHED_STEPS[-1]] and ladder_l4_dose.ENTRY_STAGE == "ladder_l4_dose_r64"
    # THE OTHER STAGE: the prepare step, `matched_1262` trained and measured, the report with it alone; a guard before each; no variable; the SAME run directory.
    alone = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_dose_r64_matched"]]
    assert [(environment, step) for environment, step, _ in alone] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", MATCHED_STEPS[0]),
                                                                       *(pair for step in MATCHED_STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in alone) and ladder_l4_dose.MATCHED_STAGE == "ladder_l4_dose_r64_matched" and ladder_l4_dose.REPORT_MATCHED == MATCHED_STEPS[-1]
    assert MATCHED_STEPS[:3] == STEPS[:3] and MATCHED_STEPS[-1] not in [step for _, step, _ in steps]
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_dose_r64_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    # THE ORDER IS THE SETTING'S: each model trained, then measured, before the next; `matched_1262` first, then the smaller dose.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings = whole["ladder_loop"]["l4"]["dose"]
    assert settings == {"start": "pre_r64", "models": ["matched_1262", "second_half", "second_full"]} and [step for _, step, _ in steps if step and "_train_" in step] == [
        f"ladder_l4_dose_train_{model}" for model in settings["models"]] == list(TRAININGS)
    own = ladder_l4_dose.the_settings(whole)
    assert (own.start, own.models, own.l4t2.start, own.l4t2.models) == ("pre_r64", MODELS, "pre_r64", ("rehearse_only", "reward_rehearse")) and ladder_l4_dose.run_name(own, 0) == "ladder_l4_dose_pre_r64_seed0"
    assert ladder_l4_rows2.run_name(own.l4t2, 0) == "ladder_l4_rows2_pre_r64_seed0" and ladder_l4_rank.check_of(whole, settings["start"]).rank == 64
    # L4t's and L4t2's own settings are what they were, and this stage's are its own block: one reader each.
    assert whole["ladder_loop"]["l4"]["rows"]["start"] == "pre" and whole["ladder_loop"]["l4"]["rows2"] == {"start": "pre_r64", "models": ["rehearse_only", "reward_rehearse"]}
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["l4"]["dose"]' in path.read_text()] == ["gpu/ladder_l4_dose.py"]
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["l4"]["rows2"]' in path.read_text()] == ["gpu/ladder_l4_rows2.py"]
    # Its steps are its own, and only its three stages run them; no other module's step has one of its names.
    others = set()
    for module in (ladder_round, ladder_l2, ladder_loop, ladder_ceiling, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_rows2, ladder_l4b):
        others |= set(module.STEPS)
    assert not set(ladder_l4_dose.STEPS) & others and all(name.startswith("ladder_l4_dose_") for name in ladder_l4_dose.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l4_dose.STEPS for step in entry.STAGES[name])) == ["ladder_l4_dose_r64", "ladder_l4_dose_r64_matched", "ladder_l4_dose_r64_smoke"]
    # Its two variables are named by its smoke stage alone, and it names NONE of L4t's or L4t2's.
    names = lambda variable: sorted(name for name, own_ in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own_))      # noqa: E731
    assert names(RUN) == names(L4T2_RUN) == ["ladder_l4_dose_r64_smoke"] and names(L4T2_OWN_RUN) == names(L4T2_L4T_RUN) == ["ladder_l4_rows2_r64_smoke"]
    for variable in (L4T_START, L4T_MINIMUM, L4T_OWN_RUN, L4T_ARM_RUN, L4T_START_CHECKS, L4T2_OWN_RUN, L4T2_L4T_RUN):
        assert not [name for name in names(variable) if name.startswith("ladder_l4_dose")], variable
    gpu = {path.name: path.read_text() for path in (PACKAGE / "gpu").glob("*.py")}
    assert not [variable for variable in (L4T_START, L4T_MINIMUM, L4T_ARM_RUN, L4T2_OWN_RUN, L4T2_L4T_RUN) if variable in gpu["ladder_l4_dose.py"]]
    # ONE BODY EACH: this stage's train and measure steps are L4t's own two bodies and its report reads its models by L4t's own reader.
    assert ladder_l4_dose.train_a_model is ladder_l4_rows.train_a_model and ladder_l4_dose.measure_a_model is ladder_l4_rows.measure_a_model
    assert ladder_l4_dose.read_the_trained is ladder_l4_rows.read_the_trained
    # The steps of the stage are in the GPU entry point's own table.
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l4_dose.STEPS" in main and "ladder_l4_dose," in main
