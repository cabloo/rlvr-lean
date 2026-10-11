"""L4t2, the stage end to end on what L4t's FINISHED run from `pre_r64` and the start model's run stored: the two training
sets built from L4t's two stored sets (the refusals of its prepare step), each model trained from `pre_r64` at rank 64 and
measured by L4t's own two step bodies, the report; the runs it reads left file for file as they were, and L4t from `pre`
and from `pre_r64` writing what they wrote; the smoke stage; the registration. Spec:
docs/spec/ladder-loop.spec.md, "L4t2: two more one-change checks, on what L4t left open". The engine and Lean are
scripted; nothing touches a GPU or the network. The rules and the report on hand-made rows are `test_ladder_l4_rows2.py`."""

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
from test_ladder_l4_rows_r64 import OWN_FILES as L4T_FILES_FROM_PRE_R64  # noqa: E402 - what L4t from `pre_r64` leaves, frozen there
from test_ladder_l4_rows_r64 import STEPS as L4T_STEPS_FROM_PRE_R64  # noqa: E402
from test_ladder_l4_rows_r64 import _stage as _l4t  # noqa: E402 - L4t's own stage from `pre_r64`, as its tests run it
from test_ladder_l4_rows_r64 import _trainings, r64  # noqa: E402, F401
from test_ladder_l4_rows_stage import OWN_FILES as L4T_FILES_FROM_PRE  # noqa: E402
from test_ladder_l4_rows_stage import STEPS as L4T_STEPS_FROM_PRE  # noqa: E402
from test_ladder_l4_rows_stage import _all, rows  # noqa: E402, F401
from test_ladder_l4_rows_stage import _stage as _l4t_from_pre  # noqa: E402
from test_ladder_l4_stage import FIXTURE, _file_rows, _on_the_gpu, l4  # noqa: E402, F401
from test_ladder_round import _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import INCONCLUSIVE, NOT_READ, goal_set_again  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import REHEARSAL_LABEL  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows2 import ALL_THREE, LOOPS_ROWS_ADD, MODELS, NO_ARM, NO_RULE, NOT_SEPARATED, QUESTION_1, SECOND_PASS, STORED, TEXT_FIELDS  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_rows2, ladder_l4b, ladder_loop, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.ladder_l3d1 import WROTE  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402
from rlvr_lean.reporting.ladder_l4 import SAY  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
RUN, L4T_RUN = ladder_l4_rows2.RUN_VARIABLE, ladder_l4_rows2.L4T_RUN_VARIABLE
L4T_START, L4T_MINIMUM, L4T_OWN_RUN, L4T_ARM_RUN, L4T_START_CHECKS = (ladder_l4_rows.START_VARIABLE, ladder_l4_rows.MINIMUM_VARIABLE, ladder_l4_rows.RUN_VARIABLE,
                                                                     ladder_l4_rows.ARM_RUN_VARIABLE, ladder_l4_rows.START_CHECKS_VARIABLE)
RANK_RUN = ladder_l4_rank.RUN_VARIABLE
SEEDS = "RLVR_LEAN_TRAINING_SEEDS"
STEPS = ["ladder_l4_rows2_prepare", "ladder_l4_rows2_train_rehearse_only", "ladder_l4_rows2_measure_rehearse_only", "ladder_l4_rows2_train_reward_rehearse",
         "ladder_l4_rows2_measure_reward_rehearse", "ladder_l4_rows2_report"]
TRAININGS = ("ladder_l4_rows2_train_rehearse_only", "ladder_l4_rows2_train_reward_rehearse")
RANK_64 = {"rank": 64, "alpha": 128}
# What the stage leaves in its run directory, beside the episodes of the sets it samples.
OWN_FILES = sorted([
    *(f"adapters/{model}/{name}" for model in MODELS for name in ("adapter_config.json", "adapter_model.safetensors")),
    "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl", "l4_rows2_loss_rehearse_only.json", "l4_rows2_loss_reward_rehearse.json", "l4_rows2_stored_models.json",
    "l4_rows2_training_rehearse_only.jsonl", "l4_rows2_training_reward_rehearse.jsonl",
    *(f"l4_stored_{who}_{part}.jsonl" for who in ("pre_r64", *STORED) for part in ("rungs", "reach", "more")),
    "ladder_l4_rows2_measure_rehearse_only.done.json", "ladder_l4_rows2_measure_reward_rehearse.done.json", "ladder_l4_rows2_prepare.done.json", "ladder_l4_rows2_report.done.json",
    "ladder_l4_rows2_train_rehearse_only.done.json", "ladder_l4_rows2_train_reward_rehearse.done.json", "problems.jsonl", "report_ladder_l4_rows2.json"])
PREPARE_KEYS = ["stage", "label", "check", "seed", "start", "start_adapter", "start_run", "l4t_run", "models", "stand_in_engine", "effective_batch", "training_sets",
                "pretraining_file", "pretraining_file_sha256", "recipe", "goal_set", "goal_set_again", "rung_problems", "sampling_seeds", "rung_episodes", "goal_samplings",
                "attempts_a_goal_problem", "parts", "stored_models", "never_solved_file", "contradicted_side_setting", "start_recipe", "start_run_read", "l4t_read"]
REPORT_KEYS = ["spec", "stage", "label", "check", "headline", "lines", "ok", "seed", "stand_in_engine", "start", "rank", "branches", "models", "the_three_reads", "question_1",
               "question_2", "the_arms_rule", "what_it_cannot_say", "the_stored_models", "training_sets", "heldout", "sizes", "start_recipe", "attempts", "not_to_be_read",
               "not_measured", "adapters"]
ARMS_RULES = ("reward_rehearse", "rehearse", NO_ARM, NO_RULE)


def _stage2(config, monkeypatch, until=None, stage_name="ladder_l4_rows2_r64"):
    """The stage's steps in its order, the trainings by their GPU path."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l4_rows2.STEPS:
            continue
        run = ladder_l4_rows2.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in TRAININGS else run(config)
        if step == until:
            break
    return summaries


def _draw_order(rows_, seed=0):
    """Rehearsal rows in the order L4t's rehearsal draw takes them: the smallest content hash of the seed and the row's id first."""
    return sorted(rows_, key=lambda row: rank(seed, REHEARSAL_LABEL, row["id"]))


def _as_named(row):
    return {key: value for key, value in row.items() if key != "row"}


@pytest.fixture
def rows2(r64, monkeypatch):  # noqa: F811
    """`r64`'s world (`pre`'s run, the arm's from it, the run of the check of the adapter's rank, every training by its
    GPU path) with L4t's own stage from `pre_r64` RUN TO ITS END in it: what this stage reads on the box. Then this
    stage's task, which carries NO variable of L4t's (the start is its own setting's). `l4t_summaries`: what L4t's
    steps returned."""
    for name in (RUN, L4T_RUN):
        monkeypatch.delenv(name, raising=False)
    config = r64.config
    l4t_summaries = _l4t(config, monkeypatch)
    l4t = r64.store().root
    assert l4t.name == "ladder_l4_rows_pre_r64_seed0" and list(l4t_summaries) == L4T_STEPS_FROM_PRE_R64 and l4t_summaries["ladder_l4_rows_report"]["ok"] is True
    for name in (L4T_START, L4T_MINIMUM):
        monkeypatch.delenv(name)
    assert all(entry.step_fields(step)[2] == {} for step in entry.STAGES["ladder_l4_rows2_r64"])       # THE STAGE names no variable
    del r64.calls[:], r64.kits[:]
    return SimpleNamespace(config=config, r64=r64, calls=r64.calls, kits=r64.kits, l4t=l4t, rank=r64.rank, arm=r64.arm, pre=r64.pre, l4t_summaries=l4t_summaries,
                           store=lambda: ladder_l4_rows2._store(config, ladder_l4_rows2.the_settings(config)))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_builds_the_two_sets_from_l4ts_stored_sets_trains_each_from_pre_r64_at_rank_64_measures_them_and_reports(rows2, monkeypatch):
    config, calls, kits = rows2.config, rows2.calls, rows2.kits
    l4t, rank_run, arm_run, pre = rows2.l4t, rows2.rank, rows2.arm, rows2.pre
    read_only = {path: _all(path) for path in (l4t, rank_run, arm_run, pre)}
    summaries = _stage2(config, monkeypatch)
    store = rows2.store()
    assert store.root.name == "ladder_l4_rows2_pre_r64_seed0" and store.root.parent == l4t.parent and list(summaries) == STEPS == list(ladder_l4_rows2.STEPS)
    assert {path: _all(path) for path in read_only} == read_only               # L4t's RUN AND THE START MODEL's WERE ONLY READ: file for file, adapters included
    assert (config["lora"]["rank"], config["vllm"]["max_lora_rank"], config["sampling"]["temperature"]) == (16, 16, 1.0)      # and the config itself was not changed

    # ---- prepare: the start, its rank, what L4t's run stored, the two training sets
    prepare = summaries["ladder_l4_rows2_prepare"]
    start = rank_run / "adapters" / "pre_r64"
    assert list(prepare) == PREPARE_KEYS and (prepare["stage"], prepare["label"], prepare["check"], prepare["seed"]) == ("l4", "pretrained on published proofs", "L4t2", 0)
    assert (prepare["start"], prepare["start_adapter"], prepare["start_run"], prepare["l4t_run"], prepare["models"]) == ("pre_r64", str(start), str(rank_run), str(l4t), list(MODELS))
    of_l4t = json.loads((l4t / "ladder_l4_rows_prepare.done.json").read_text())
    of_start = json.loads((rank_run / "ladder_l4_rank_prepare.done.json").read_text())
    recipe = prepare["start_recipe"]
    assert (recipe["rank"], recipe["alpha"], recipe["rank_of_the_config"], recipe["alpha_of_the_config"]) == (64, 128, 16, 32) == (
        of_start["recipe"]["lora"]["rank"], of_start["recipe"]["lora"]["alpha"], config["lora"]["rank"], config["lora"]["alpha"])
    assert (recipe["rank"], recipe["alpha"]) == (of_l4t["start_recipe"]["rank"], of_l4t["start_recipe"]["alpha"]) and prepare["recipe"]["lora"] == {**config["lora"], **RANK_64}
    assert prepare["start_run_read"]["checks_waived_as_l4ts_run_recorded"] is False and prepare["pretraining_file_sha256"] == of_l4t["pretraining_file_sha256"] == of_start["pretraining_file_sha256"]
    assert all(prepare[key] == of_l4t[key] == of_start[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings")) and prepare["goal_set_again"] == of_l4t["goal_set_again"]
    # L4t's OWN report, as it read its three models: carried here, and a model it read as INCONCLUSIVE is not read here either.
    of_its_report = json.loads((l4t / "report_ladder_l4_rows.json").read_text())
    assert prepare["l4t_read"]["branches"] == of_its_report["branches"] == rows2.l4t_summaries["ladder_l4_rows_report"]["branches"] and prepare["l4t_read"]["run"] == str(l4t)
    assert prepare["l4t_read"]["failed_checks"] == {model: [check for check, said in of_its_report["models"][model]["can_this_model_be_read"].items()
                                                            if check != "what" and not said["passes"]] for model in STORED}
    # `rehearse_only`: THE REHEARSAL ROWS OF L4t's `rehearse` SET AND NOTHING ELSE, in that set's order, no row changed.
    rehearse, reward_rows = _file_rows(l4t / "l4_rows_training_rehearse.jsonl"), _file_rows(l4t / "l4_rows_training_reward_rows.jsonl")
    of_rehearse = [row for row in rehearse if row["origin"] == "pretraining"]
    rehearse_only, reward_rehearse = store.read_rows("l4_rows2_training_rehearse_only.jsonl"), store.read_rows("l4_rows2_training_reward_rehearse.jsonl")
    assert of_rehearse and [_as_named(row) for row in rehearse_only] == [_as_named(row) for row in of_rehearse] and [row["row"] for row in rehearse_only] == list(range(len(of_rehearse)))
    assert all(row["origin"] == "pretraining" and row["k"] is None for row in rehearse_only) and len(rehearse_only) == len(rehearse) // 2
    # `reward_rehearse`: L4t's `reward_rows` set, every row unchanged, and AS MANY rehearsal rows as it has ONE-SHOT rows: the FIRST of the draw's order, so a prefix of `rehearse`'s.
    one_shot = [row for row in reward_rows if row["origin"] == "attempt"]
    of_the_rounds, rehearsal = [row for row in reward_rehearse if row["origin"] != "pretraining"], [row for row in reward_rehearse if row["origin"] == "pretraining"]
    assert one_shot and sorted(row["id"] for row in of_the_rounds) == sorted(row["id"] for row in reward_rows) and len(rehearsal) == len(one_shot) <= len(of_rehearse)
    assert all(_as_named(row) == _as_named(next(given for given in reward_rows if given["id"] == row["id"])) for row in of_the_rounds)
    assert [_as_named(row) for row in _draw_order(rehearsal)] == [_as_named(row) for row in _draw_order(of_rehearse)[:len(one_shot)]]
    assert [row["id"] for row in reward_rehearse] == [row["id"] for row in training_order(reward_rehearse, 0)] and [row["row"] for row in reward_rehearse] == list(range(len(reward_rehearse)))
    file_rows = _file_rows(FIXTURE)
    assert all(file_rows[row["row_of_the_file"]]["problem_id"] == row["problem_id"] for row in (*rehearse_only, *rehearsal))
    sets = prepare["training_sets"]
    assert list(sets) == list(MODELS) and (sets["rehearse_only"]["rows"], sets["rehearse_only"]["rehearsal_rows"], sets["rehearse_only"]["rows_of_rehearse"]) == (
        len(of_rehearse), len(of_rehearse), len(rehearse)) and sets["rehearse_only"]["rows_by_origin"] == {"pretraining": len(of_rehearse)}
    assert (sets["reward_rehearse"]["rows"], sets["reward_rehearse"]["rows_of_reward_rows"], sets["reward_rehearse"]["one_shot_rows_of_reward_rows"], sets["reward_rehearse"]["rehearsal_rows"],
            sets["reward_rehearse"]["rehearsal_rows_of_rehearse"]) == (len(reward_rows) + len(one_shot), len(reward_rows), len(one_shot), len(one_shot), len(of_rehearse))
    assert all(sets[model]["optimizer_steps"] == -(-sets[model]["rows"] // 8) and sets[model]["barred"]["held_out"] == [] for model in MODELS)
    assert sets["rehearse_only"]["barred"]["rehearsal_rows"] == len(of_rehearse) and sets["reward_rehearse"]["barred"]["rows_of_the_rounds"] == len(reward_rows)
    # NO PUBLISHED PROOF IS COPIED INTO THE RUN: a rehearsal row is named by its id and its place, and its text stays in the pretraining file.
    assert not any(field in row for row in (*rehearse_only, *rehearsal) for field in TEXT_FIELDS)
    assert not any(proof in content.decode(errors="ignore") for proof in (*(json.dumps(row["proof"], ensure_ascii=False)[1:-1] for row in file_rows), *(row["proof"].strip() for row in file_rows))
                   for content in _all(store.root).values())
    # G' is the START MODEL's, made again from its rows, and it is L4t's; the stored rows are the start model's own run's and L4t's run's.
    goal = [row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE) if row["group"] == "goal"]
    again = [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)]
    assert again == goal_set_again(_file_rows(rank_run / "episodes_l4_reach_pre_r64_problems.jsonl"), goal) == [row["problem_id"] for row in _file_rows(l4t / ladder_l4.AGAIN_FILE)]
    assert store.read_rows(ladder_l4.GROUPS_FILE) == _file_rows(l4t / ladder_l4.GROUPS_FILE)
    for part in ("rungs", REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("pre_r64", part, "l4")) == _file_rows(rank_run / f"episodes_l4_{part}_pre_r64_problems.jsonl")
        for model in STORED:
            assert store.read_rows(ladder_ceiling.stored_file(model, part, "l4")) == _file_rows(l4t / f"episodes_l4_{part}_{model}_problems.jsonl")
    assert list(prepare["stored_models"]) == ["pre_r64", *STORED] and prepare["stored_models"]["pre_r64"]["run"] == str(rank_run) and prepare["stored_models"]["rehearse"] == {
        "run": str(l4t), "sets": {"rungs": "l4_rungs_rehearse", REACH: "l4_reach_rehearse", MORE: "l4_more_rehearse"}}
    # What each stored model WROTE is its own measure step's record: the start model's from the check's run, L4t's three from L4t's run.
    of_the_stored = json.loads(store.path(ladder_l4_rows2.STORED_MODELS_FILE).read_text())
    assert set(of_the_stored) == {"stage", "label", "pre_r64", *STORED}
    assert of_the_stored["pre_r64"] == {key: json.loads((rank_run / "ladder_l4_rank_measure.done.json").read_text())[key] for key in WROTE}
    for model in STORED:                                                       # (in this world the scripted solver writes the same for every model: the next test tells them apart)
        assert of_the_stored[model] == {key: json.loads((l4t / f"ladder_l4_rows_measure_{model}.done.json").read_text())[key] for key in WROTE}

    # ---- train: each model FROM `pre_r64`, AT ITS RANK, by L4t's own training body; `rehearse_only` first; the rehearsal rows' text read from the pretraining file
    by_problem = {row["problem_id"]: row for row in file_rows}
    assert [next(iter(call["schedule"]["checkpoint_steps"])) for call in calls] == list(MODELS) and len(calls) == 2
    for call, model, set_rows in zip(calls, MODELS, (rehearse_only, reward_rehearse)):
        train = summaries[f"ladder_l4_rows2_train_{model}"]
        assert call["lora"] == {**config["lora"], **RANK_64} and call["max_lora_rank"] == 16      # the adapter's rank is the start adapter's; a training starts no model server
        assert call["more"] == {"start": start} and call["orders"] == [list(range(len(set_rows)))] and call["tensorboard_run"] == f"ladder_l4_rows2_pre_r64_seed0_{model}"
        assert call["schedule"] == {"passes": 1, "reading_steps": [], "checkpoint_steps": {model: -(-len(set_rows) // 8)}} and call["directory"] == store.root / "adapters"
        assert call["examples"] == [{"problem_id": row["problem_id"], "side": "statement", "theorem": by_problem[row["problem_id"]]["statement"], "completion": by_problem[row["problem_id"]]["proof"]}
                                    if row["origin"] == "pretraining" else {key: row[key] for key in ("problem_id", "side", "theorem", "completion")} for row in set_rows]
        assert (train["stage"], train["label"], train["model"], train["rows"], train["trained_from"], train["adapter"]) == (
            "l4", "pretrained on published proofs", model, len(set_rows), f"the stored adapter {start}", str(store.root / "adapters" / model))
        assert train["rows_by_origin"] == sets[model]["rows_by_origin"] and "a content hash of the task's seed and each row's id" in train["order"]
        saved = train["adapter_saved"]
        assert (saved["rank"], saved["alpha"], saved["ranks"], train["peak_reserved_gb"]) == (64, 128, [64], 12.64) and set(train["against_the_start_adapter"]) == {model}
        assert json.loads((store.root / "adapters" / model / "adapter_config.json").read_text())["r"] == 64
        loss = json.loads(store.path(f"l4_rows2_loss_{model}.json").read_text())
        assert loss["rows_trained"] == [row["id"] for row in set_rows] and len(loss["row_losses"]) == len(set_rows) and loss["label"] == "pretrained on published proofs"

    # ---- measure: as L4t's models were, by L4t's own measuring body: the rungs and G's samplings, the start model's sampling seeds, the config's temperature, rank 64
    assert len(kits) == 2 * 3 and all(kit["temperature"] == 1.0 and kit["max_lora_rank"] == 64 and kit["rank"] == 64 and {**kit["vllm"], "max_lora_rank": 16} == config["vllm"] for kit in kits)
    for model in MODELS:
        measured = summaries[f"ladder_l4_rows2_measure_{model}"]
        assert (measured["stage"], measured["label"], measured["model"]) == ("l4", "pretrained on published proofs", model) and set(measured["goal"]) == {REACH, MORE}
        for of, own_set, start_set in ((measured["rungs"], f"l4_rungs_{model}", "l4_rungs_pre_r64"), (measured["goal"][REACH], f"l4_reach_{model}", "l4_reach_pre_r64"),
                                       (measured["goal"][MORE], f"l4_more_{model}", "l4_more_pre_r64")):
            marker = json.loads((rank_run / f"episodes_{start_set}.done.json").read_text())
            assert of["set"] == own_set and (of["sampling_seed"], of["episodes_each"], of["problems"]) == (marker["sampling_seed"], marker["episodes_each"], marker["problems"])
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l4_{part}_{model}" for part in ("rungs", REACH, MORE) for model in MODELS}
    assert not [name for name in _all(store.root) if "_hot_" in name]         # no measurement at another temperature, and no set of one

    # ---- the report: each model's four checks, the three reads of the five, the two questions, the arm's rule
    report = summaries["ladder_l4_rows2_report"]
    assert list(report) == REPORT_KEYS and (report["stage"], report["label"], report["check"], report["ok"], report["start"], report["rank"]) == (
        "l4", "pretrained on published proofs", "L4t2", True, "pre_r64", 64)
    assert all(line.startswith(f"{SAY}: ") for line in report["lines"]) and "pretrained on published proofs" in report["headline"] and "from `pre_r64`" in report["headline"]
    assert list(report["branches"]) == ["question_1", "question_2", "the_arms_rule"] and report["branches"]["question_1"] in QUESTION_1 and report["branches"]["the_arms_rule"] in ARMS_RULES
    assert report["branches"]["question_2"] in (ALL_THREE, INCONCLUSIVE, NOT_READ) or report["branches"]["question_2"].startswith("IT FAILS: ")
    assert set(report["the_three_reads"]) >= {*MODELS, *STORED} and list(report["models"]) == list(MODELS) and report["the_three_reads"]["goal_set_again"] == len(again)
    solved_in = lambda name: sum(sum(counts) > 0 for counts in zip(*([row["resolved"] for row in store.read_rows(name(part))] for part in (REACH, MORE))))      # noqa: E731
    by_the_start = solved_in(lambda part: ladder_ceiling.stored_file("pre_r64", part, "l4"))
    for model in MODELS:
        read = report["models"][model]
        assert set(read["can_this_model_be_read"]) == {"what", "its_training_ran", "it_still_writes_proofs", "lean_answered", "no_barred_row"}
        secondary = read["secondary"] if not read["inconclusive"] else read["measured_and_not_read"]["secondary"]
        assert secondary["trained_on"]["rows"] == sets[model]["rows"] and secondary["trained_on"]["rows_by_origin"] == sets[model]["rows_by_origin"]
        long = secondary["by_length_group"]["minus_pre_r64"]["4_or_more"]
        assert secondary["goal_problems_solved_by_attempts_alone"].keys() >= {"pre_r64", *STORED, model} and long["problems"] == 0      # the fixture's goal problem has no known length
        assert read["the_three_reads"] is report["the_three_reads"][model] and read["the_three_reads"]["against"] == "pre_r64"
        # Its reads stand on the rows THIS run measured for it, against the start model's stored rows: the goal problems each solves in all the attempts.
        against = [entry for group, entry in secondary["by_length_group"]["against_pre_r64"].items() if group != "4_or_more"]
        assert (sum(entry["resolved_after"] for entry in against), sum(entry["resolved_before"] for entry in against), sum(entry["problems"] for entry in against)) == (
            solved_in(lambda part: f"episodes_l4_{part}_{model}_problems.jsonl"), by_the_start, len(goal))
        figures = read["the_three_reads"].get("measured_and_not_read") or read["the_three_reads"]
        assert figures["not_narrower"]["problems"] == len(goal) == 1 and figures["adds"]["problems"] == len(again)
    for model in STORED:                                                       # FROM THE ROWS L4t's RUN STORED: a model its own report read as INCONCLUSIVE is not read
        assert report["the_three_reads"][model]["failed_checks"] == prepare["l4t_read"]["failed_checks"][model]
    assert json.loads(store.path(ladder_l4_rows2.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_l4_rows2.REPORT) == {"stage": "l4", "label": "pretrained on published proofs", "check": "L4t2", "headline": report["headline"],
                                                          "branches": report["branches"], "ok": True}
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == sorted(MODELS) and "`pre_r64` is its own run's and was only read; so were L4t's" in report["adapters"]["what"]
    # The goal problems nothing stored had solved are GIVEN: a file in the run directory, one `problem_id` a row. The report is built again and reads it.
    store.write_rows(ladder_l4_rows2.NEVER_SOLVED_FILE, [{"problem_id": goal[0]}, {"problem_id": "not_a_goal_problem"}])
    given = ladder_l4_rows2.ladder_l4_rows2_report(config)["models"]["rehearse_only"]
    given = (given["measured_and_not_read"] if given["inconclusive"] else given)["secondary"]["never_solved_before"]
    assert (given["given"], given["ids"], given["goal_problems"], given["not_goal_problems_of_this_run"]) == (True, 2, 1, 1) and {f"solved_by_{model}" for model in STORED} <= set(given)
    store.path(ladder_l4_rows2.NEVER_SOLVED_FILE).unlink()
    # What it leaves: its own files; nothing of it is in another run's directory.
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert {path.name for path in store.root.parent.iterdir() if path.name.startswith("ladder_l4_rows")} == {"ladder_l4_rows_pre_r64_seed0", "ladder_l4_rows2_pre_r64_seed0"}
    # A rerun returns what is stored: nothing is trained or sampled again.
    sent, sampled = len(rows2.r64.l4.lean.sources), rows2.r64.l4.solver.calls
    rerun = _stage2(config, monkeypatch)
    assert len(calls) == 2 and (len(rows2.r64.l4.lean.sources), rows2.r64.l4.solver.calls) == (sent, sampled) and {path: _all(path) for path in read_only} == read_only
    assert {name: summary for name, summary in rerun.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}


def test_what_each_stored_model_wrote_and_its_rows_are_its_own_runs_and_no_other_models(rows2):
    """The scripted solver writes the same for every model, so each stored record and each stored set is marked here
    with a figure of its own before the prepare step reads it."""
    config, l4t, rank_run = rows2.config, rows2.l4t, rows2.rank
    marks = {"pre_r64": 0.811, "old_rule": 0.722, "reward_rows": 0.773, "rehearse": 0.737}
    markers = {"pre_r64": rank_run / "ladder_l4_rank_measure.done.json", **{model: l4t / f"ladder_l4_rows_measure_{model}.done.json" for model in STORED}}
    rungs = {"pre_r64": rank_run / "episodes_l4_rungs_pre_r64_problems.jsonl", **{model: l4t / f"episodes_l4_rungs_{model}_problems.jsonl" for model in STORED}}
    for place, (name, mark) in enumerate(marks.items()):
        recorded = json.loads(markers[name].read_text())
        recorded["distinct_attempts"]["goal"]["mean_share_distinct"] = mark
        markers[name].write_text(json.dumps(recorded))
        own = _file_rows(rungs[name])
        rungs[name].write_text("".join(json.dumps({**row, "resolved": place if index == 0 else row["resolved"]}) + "\n" for index, row in enumerate(own)))
    ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    store = rows2.store()
    of_the_stored = json.loads(store.path(ladder_l4_rows2.STORED_MODELS_FILE).read_text())
    assert {name: of_the_stored[name]["distinct_attempts"]["goal"]["mean_share_distinct"] for name in marks} == marks and all(set(of_the_stored[name]) == set(WROTE) for name in marks)
    assert [store.read_rows(ladder_ceiling.stored_file(name, "rungs", "l4"))[0]["resolved"] for name in marks] == [0, 1, 2, 3]


def test_the_prepare_step_refuses_before_anything_is_written(rows2, monkeypatch, tmp_path):
    config, l4t, rank_run = rows2.config, rows2.l4t, rows2.rank
    as_stored = {path: _all(path) for path in (l4t, rank_run)}
    root = rows2.store().root
    l4ts_task = "the task of stage `ladder_l4_rows_r64` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rows_r64 --seeds 0`; for the smoke run, stage `ladder_l4_rows_r64_smoke`."
    ranks_task = "the task of stage `ladder_l4_rank` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rank --seeds 0`; for the smoke run, stage `ladder_l4_rank_smoke`."

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l4_rows2.ladder_l4_rows2_prepare(config)
        assert root.name == "ladder_l4_rows2_pre_r64_seed0" and not _all(root)  # nothing was written: no marker, no file

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
    # ---- L4t's RUN, FINISHED: what its steps recorded, its report, its problems, groups and G', its two stored sets, its three models' rows and markers
    for name in ("ladder_l4_rows_prepare.done.json", "ladder_l4_rows_report.done.json", "report_ladder_l4_rows.json", "problems.jsonl", "l4_heldout_groups.jsonl", "l4_goal_set_again.jsonl",
                 "l4_rows_training_rehearse.jsonl", "l4_rows_training_reward_rows.jsonl", "ladder_l4_rows_measure_old_rule.done.json", "ladder_l4_rows_measure_rehearse.done.json"):
        with_a_change(l4t, (l4t / name).unlink, f"ladder_l4_rows_pre_r64_seed0 does not hold .'{name}'.: what L4t's steps recorded, its report, .* or one of its two stored training sets. "
                                                f"L4t2 makes no round and no pretraining: it reads what {l4ts_task} wrote on this box. Run that task to its end first; nothing was written")
    for name in ("episodes_l4_rungs_old_rule_problems.jsonl", "episodes_l4_reach_rehearse.done.json", "episodes_l4_more_reward_rows_problems.jsonl"):
        with_a_change(l4t, (l4t / name).unlink, f"ladder_l4_rows_pre_r64_seed0 does not hold .'{name}'.: the per-problem rows of one of L4t's three models on a set it was measured on, or "
                                                f"that set's own marker. L4t2 makes no round and no pretraining: it reads what {l4ts_task} wrote")
    # ---- it is L4t's run of THIS seed, from THIS start, with its three models
    for changed in ({"seed": 3}, {"start": "pre"}, {"start": "pre_r32"}, {"models": ["rehearse", "reward_rows"]}, {"models": ["old_rule", "rehearse", "reward_rows"]}):
        with_a_change(l4t, rewritten(l4t, "ladder_l4_rows_prepare.done.json", change_of(**changed)),
                      "L4t's run ladder_l4_rows_pre_r64_seed0 recorded seed .*; this task's seed is 0, ladder_loop.l4.rows2 names the start 'pre_r64', and from it L4t trains and measures "
                      ".'old_rule', 'reward_rows', 'rehearse'.: the sets and the rows read here are that run's")
    # ---- a report of L4t's that is NOT TO BE READ
    with_a_change(l4t, rewritten(l4t, "report_ladder_l4_rows.json", change_of(ok=False)),
                  f"the report of L4t's run ladder_l4_rows_pre_r64_seed0 is not to be read .Lean gave no verdict on too much of a set.: its models' rows cannot stand in a comparison. "
                  f"Queue {l4ts_task} again")
    # ---- L4t's run and the start model's run must be of ONE start model: its rank, how it was measured, its G', the file it was pretrained on
    for lora, said in (({"rank": 32, "alpha": 64}, "rank 32 and alpha 64"), ({"rank": 64, "alpha": 64}, "rank 64 and alpha 64")):
        with_a_change(l4t, rewritten(l4t, "ladder_l4_rows_prepare.done.json", lambda recorded, lora=lora: {**recorded, "start_recipe": {**recorded["start_recipe"], **lora}}),
                      f"the run ladder_l4_pretrain_r64_seed0 recorded an adapter of rank 64 and alpha 128 for `pre_r64`, and L4t's run ladder_l4_rows_pre_r64_seed0 trained its models "
                      f"at {said}: every model here is at the rank L4t's were")
    with_a_change(l4t, rewritten(l4t, "ladder_l4_rows_prepare.done.json", lambda recorded: {**recorded, "sampling_seeds": {**recorded["sampling_seeds"], "rungs": 7}}),
                  "the run ladder_l4_pretrain_r64_seed0 recorded how `pre_r64` was measured .*, and L4t's run ladder_l4_rows_pre_r64_seed0 measured its models otherwise .*: L4t2's "
                  "models would not pair with both")
    with_a_change(l4t, rows_of(l4t, "l4_goal_set_again.jsonl", lambda own: [*own, {"problem_id": "nowhere"}]),
                  "G' made from the rows of `pre_r64` .ladder_l4_pretrain_r64_seed0: .* goal problems it does not solve in its first sampling. is not the G' L4t's run "
                  "ladder_l4_rows_pre_r64_seed0 stored: the two runs are not of one start model's rows")
    # ... G' is made from the START MODEL's first sampling and no other model's: with a goal problem it no longer solves there, it is not the G' L4t stored.
    assert _file_rows(l4t / "l4_goal_set_again.jsonl") == [] and all(row["resolved"] > 0 for row in _file_rows(l4t / "episodes_l4_reach_old_rule_problems.jsonl"))
    with_a_change(rank_run, rows_of(rank_run, "episodes_l4_reach_pre_r64_problems.jsonl", lambda own: [{**row, "resolved": 0} for row in own]),
                  "G' made from the rows of `pre_r64` .ladder_l4_pretrain_r64_seed0: 1 goal problems it does not solve in its first sampling. is not the G' L4t's run")
    with_a_change(l4t, rewritten(l4t, "ladder_l4_rows_prepare.done.json", change_of(pretraining_file_sha256="0" * 64)),
                  "L4t's run ladder_l4_rows_pre_r64_seed0 was prepared on a pretraining file of SHA-256 0{64} and `pre_r64` was trained on .*: L4t's sets name each rehearsal row by its "
                  "place in the file it was prepared on")
    # ---- the stored rows must pair with this run: the seeds, the problems, Lean having answered
    marker = json.loads(as_stored[l4t]["episodes_l4_reach_rehearse.done.json"])
    with_a_change(l4t, rewritten(l4t, "episodes_l4_reach_rehearse.done.json", change_of(sampling_seed=5)),
                  f"ladder_l4_rows_pre_r64_seed0 measured `rehearse` .reach. with sampling seed 5 and {marker['episodes_each']} episodes; this config gives {marker['sampling_seed']} and "
                  f"{marker['episodes_each']}: L4t2's models would not pair with the stored results")
    with_a_change(l4t, rows_of(l4t, "episodes_l4_rungs_reward_rows_problems.jsonl", lambda own: own[:-1]), "measured `reward_rows` .rungs. on other problems than L1's run holds for it")
    with_a_change(l4t, rows_of(l4t, "episodes_l4_more_old_rule_problems.jsonl", lambda own: [{**own[0], "attempts_without_an_answer": 60}, *own[1:]]),
                  "measured `old_rule` .more. with too many attempts without a verdict from Lean: that set stands on the other side of a comparison")
    with_a_change(rank_run, rows_of(rank_run, "episodes_l4_rungs_pre_r64_problems.jsonl", lambda own: own[:-1]), "ladder_l4_pretrain_r64_seed0 measured `pre_r64` .rungs. on other problems")
    # ---- THE START MODEL's RUN: its report, what its three steps recorded, the rows it was pretrained on, its rows and their markers; refused as L4t refuses them
    for name in ("report_ladder_l4_rank.json", "ladder_l4_rank_prepare.done.json", "ladder_l4_rank_train.done.json", "ladder_l4_rank_measure.done.json"):
        with_a_change(rank_run, (rank_run / name).unlink, f"ladder_l4_pretrain_r64_seed0 does not hold .'{name}'.: the report of the check that made `pre_r64`, or what its prepare, train "
                                                          f"or measure step recorded. L4t2 makes no round and no pretraining: it reads what {ranks_task} wrote on this box")
    for name in ("l4_pretraining_rows.jsonl", "episodes_l4_reach_pre_r64.done.json", "episodes_l4_more_pre_r64_problems.jsonl"):
        with_a_change(rank_run, (rank_run / name).unlink, f"ladder_l4_pretrain_r64_seed0 does not hold .'{name}'.: the rows `pre_r64` was pretrained on .the rehearsal rows are rows of "
                                                          f"them., its per-problem rows on a set it was measured on, or that set's own marker. L4t2 makes no round and no pretraining: it "
                                                          f"reads what {ranks_task} wrote")
    with_a_change(rank_run, rewritten(rank_run, "report_ladder_l4_rank.json", change_of(ok=False)), "the run ladder_l4_pretrain_r64_seed0 cannot carry L4t2: its report is not to be read")
    failing = lambda report: {**report, "inconclusive": True, "can_this_run_see_a_win": {**report["can_this_run_see_a_win"], "the_training_took": {"passes": False}}}      # noqa: E731
    with_a_change(rank_run, rewritten(rank_run, "report_ladder_l4_rank.json", failing), "the run ladder_l4_pretrain_r64_seed0 cannot carry L4t2: its checks failed .the_training_took.")
    with_a_change(rank_run, rewritten(rank_run, "ladder_l4_rank_prepare.done.json", change_of(seed=3)), "the run ladder_l4_pretrain_r64_seed0 recorded seed 3 and the model 'pre_r64'")
    with_a_change(rank_run, rows_of(rank_run, "l4_pretraining_rows.jsonl", lambda own: own[:-1]), "stored 11 rows `pre_r64` was trained on and L4's pretraining file holds 12")
    # ---- the pretraining file must be the one the start model was trained on
    changed = tmp_path / "changed.jsonl"
    changed.write_text(FIXTURE.read_text() + "\n")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    refused("L4's pretraining file has SHA-256 .* and `pre_r64` was trained on .* .ladder_l4_pretrain_r64_seed0.: the rehearsal rows are rows `pre_r64` was trained on")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    # ---- L4t's STORED SETS must be what this stage reads them as: `rehearse` drawn from this file with this seed; a set with a barred problem; a rehearsal row with text
    rehearsal = [row for row in _file_rows(l4t / "l4_rows_training_rehearse.jsonl") if row["origin"] == "pretraining"]
    not_drawn = next(row for row in _file_rows(rank_run / "l4_pretraining_rows.jsonl") if row["problem_id"] not in {given["problem_id"] for given in rehearsal})
    another = {**rehearsal[0], "id": f"{not_drawn['problem_id']}#pretraining", "problem_id": not_drawn["problem_id"], "row_of_the_file": not_drawn["row"]}
    with_a_change(l4t, rows_of(l4t, "l4_rows_training_rehearse.jsonl", lambda own: [another if row["id"] == rehearsal[0]["id"] else row for row in own]),
                  f"L4t's `rehearse` set holds {len(rehearsal)} rehearsal rows, and they are not the {len(rehearsal)} rows of the smallest hash among the 12 rows the start model was "
                  "pretrained on .*: that set was not drawn from this file with this seed. Refused .L4t's run ladder_l4_rows_pre_r64_seed0; `pre_r64`'s ladder_l4_pretrain_r64_seed0.. "
                  "Nothing was written")
    with_a_change(l4t, rows_of(l4t, "l4_rows_training_rehearse.jsonl", lambda own: [row for row in own if row["origin"] != "pretraining"]),
                  "L4t's `rehearse` set holds no rehearsal row among its .*: there is nothing to train `rehearse_only` on")
    with_a_change(l4t, rows_of(l4t, "l4_rows_training_rehearse.jsonl", lambda own: [{**row, "proof": "text"} if row["id"] == rehearsal[0]["id"] else row for row in own]),
                  f"1 rehearsal rows of the training set of `rehearse_only` carry a text field .first: {re.escape(rehearsal[0]['id'])}.: a rehearsal row is a published proof")
    held = _file_rows(l4t / "l4_heldout_groups.jsonl")[0]["problem_id"]
    one_shot = next(row for row in _file_rows(l4t / "l4_rows_training_reward_rows.jsonl") if row["origin"] == "attempt")
    with_a_change(l4t, rows_of(l4t, "l4_rows_training_reward_rows.jsonl", lambda own: [{**row, "problem_id": held} if row["id"] == one_shot["id"] else row for row in own]),
                  f"1 problems of the training set of `reward_rehearse` are held-out problems .first: {held}.: refused, nothing was written")
    with_a_change(l4t, rows_of(l4t, "l4_rows_training_reward_rows.jsonl", lambda own: [{**row, "problem_id": "fixture_c3"} if row["id"] == one_shot["id"] else row for row in own]),
                  "1 problems of the training set of `reward_rehearse` are rounds' rows of the `pretrain` half .first: fixture_c3.")
    # ---- the settings are the stage's: its two models in their order; a start that is `pre` or no model of the check
    settings = config["ladder_loop"]["l4"]["rows2"]
    monkeypatch.setitem(settings, "models", ["reward_rehearse", "rehearse_only"])
    refused("ladder_loop.l4.rows2.models is .'reward_rehearse', 'rehearse_only'. and the stage trains and measures .'rehearse_only', 'reward_rehearse'.: change both together", ValueError)
    monkeypatch.setitem(settings, "models", list(MODELS))
    monkeypatch.setitem(settings, "start", "pre")
    with pytest.raises(ValueError, match="ladder_loop.l4.rows2.start is `pre`: L4t2 is specified from a model of ladder_loop.l4.rank_check alone .`pre_r64`.. L4t from `pre` trains no "
                                         "`old_rule`, and the three reads are against that model"):
        ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    assert not (root.parent / "ladder_l4_rows2_pre_seed0").exists()
    monkeypatch.setitem(settings, "start", "the_base")
    with pytest.raises(RuntimeError, match="ladder_loop.l4.rows2.start is 'the_base': it names `pre` or a model of ladder_loop.l4.rank_check"):
        ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    monkeypatch.setitem(settings, "start", "pre_r32")                          # the fallback's model: neither its own run nor L4t's from it is on the box
    with pytest.raises(RuntimeError, match="ladder_l4_rows_pre_r32_seed0 does not hold .*it reads what the task of stage `ladder_l4_rows_r32` for seed 0"):
        ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    assert not (root.parent / "ladder_l4_rows_pre_r32_seed0").exists()          # and no directory was made for the run it lacks
    monkeypatch.setitem(settings, "start", "pre_r64")
    # ---- a task of another seed reads that seed's run of L4t, which is not on the box
    monkeypatch.setenv(SEEDS, "1")
    with pytest.raises(RuntimeError, match="ladder_l4_rows_pre_r64_seed1 does not hold .*the task of stage `ladder_l4_rows_r64` for seed 1 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rows_r64 --seeds 1`"):
        ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    assert rows2.store().root.name == "ladder_l4_rows2_pre_r64_seed1" and not _all(rows2.store().root)
    monkeypatch.delenv(SEEDS)
    # ---- as on the GPU: without the start adapter nothing is prepared; L4t's own adapters are not read
    adapter = rank_run / "adapters" / "pre_r64"
    adapter.rename(adapter.with_name("elsewhere"))
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        refused(f"adapters/pre_r64 is not there: the adapter `pre_r64` is what every model here is trained from. Run {ranks_task} again; nothing was written")
    adapter.with_name("elsewhere").rename(adapter)
    # ---- the later steps refuse out of their order; and with everything as it was, it prepares, the two runs untouched
    for step, match in ((ladder_l4_rows2.STEPS["ladder_l4_rows2_train_rehearse_only"], "the training of `rehearse_only` needs the step ladder_l4_rows2_prepare of this run, which is not "
                                                                                       "done: the stage `ladder_l4_rows2_r64`"),
                        (ladder_l4_rows2.STEPS["ladder_l4_rows2_measure_reward_rehearse"], "the measurement of L4t2's model `reward_rehearse` needs the step ladder_l4_rows2_prepare"),
                        (ladder_l4_rows2.ladder_l4_rows2_report, "L4t2's report needs the step ladder_l4_rows2_prepare of this run, which is not done: the stage `ladder_l4_rows2_r64`")):
        with pytest.raises(RuntimeError, match=match):
            step(config)
    shutil.rmtree(l4t / "adapters")                                            # L4t's adapters are NOT read
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        assert ladder_l4_rows2.ladder_l4_rows2_prepare(config)["start"] == "pre_r64"
    assert _all(rank_run) == as_stored[rank_run] and {name: content for name, content in as_stored[l4t].items() if not name.startswith("adapters/")} == _all(l4t)
    with pytest.raises(RuntimeError, match="the measurement of L4t2's model `rehearse_only` needs the step ladder_l4_rows2_train_rehearse_only of this run, which is not done"):
        ladder_l4_rows2.STEPS["ladder_l4_rows2_measure_rehearse_only"](config)
    with pytest.raises(RuntimeError, match="L4t2's report needs the step ladder_l4_rows2_train_rehearse_only"):
        ladder_l4_rows2.ladder_l4_rows2_report(config)


def test_a_smoke_run_goes_on_from_a_start_whose_checks_failed_only_where_l4ts_own_run_recorded_that_it_did(rows2, monkeypatch):
    config, l4t, rank_run = rows2.config, rows2.l4t, rows2.rank
    failing = json.loads((rank_run / "report_ladder_l4_rank.json").read_text())
    failing = {**failing, "inconclusive": True, "can_this_run_see_a_win": {**failing["can_this_run_see_a_win"], "the_training_took": {"passes": False}}}
    (rank_run / "report_ladder_l4_rank.json").write_text(json.dumps(failing))
    # L4t's run recorded no waiver (a real run): refused, whatever L4t's own variable says in this task's environment.
    monkeypatch.setenv(L4T_START_CHECKS, "smoke")
    with pytest.raises(RuntimeError, match="the run ladder_l4_pretrain_r64_seed0 cannot carry L4t2: its checks failed .the_training_took."):
        ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    monkeypatch.delenv(L4T_START_CHECKS)
    # L4t's run recorded that IT went on from the failed checks (a smoke run's alone does): this stage goes on too, sets nothing for it, and records it.
    of_l4t = json.loads((l4t / "ladder_l4_rows_prepare.done.json").read_text())
    (l4t / "ladder_l4_rows_prepare.done.json").write_text(json.dumps({**of_l4t, "start_run_read": {**of_l4t["start_run_read"], "checks_waived_for_a_smoke_run": True}}))
    read = ladder_l4_rows2.ladder_l4_rows2_prepare(config)["start_run_read"]
    assert (read["checks_failed"], read["checks_waived_as_l4ts_run_recorded"]) == (["the_training_took"], True)
    # A report that is not to be read is refused for it too.
    shutil.rmtree(rows2.store().root)
    (rank_run / "report_ladder_l4_rank.json").write_text(json.dumps({**failing, "ok": False}))
    with pytest.raises(RuntimeError, match="cannot carry L4t2: its report is not to be read"):
        ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    assert not [name for name, own in entry.STAGES.items() if name.startswith("ladder_l4_rows2") and any(L4T_START_CHECKS in entry.step_fields(step)[2].get("environment", {}) for step in own)]


def test_as_on_the_gpu_each_measure_step_serves_its_adapter_at_rank_64_and_a_model_at_another_rank_fails_its_training_unmarked(rows2, monkeypatch):
    config, calls, kits = rows2.config, rows2.calls, rows2.kits
    ladder_l4_rows2.ladder_l4_rows2_prepare(config)
    store = rows2.store()
    train, measure = ladder_l4_rows2.STEPS["ladder_l4_rows2_train_rehearse_only"], ladder_l4_rows2.STEPS["ladder_l4_rows2_measure_rehearse_only"]
    # A training whose saved adapter is not the start adapter's rank fails its step, unmarked, and nothing of it is measured (L4t's own body, under this stage's names).
    _trainings(monkeypatch, calls, saves={"rehearse_only": (16, 32, None)})
    with pytest.raises(RuntimeError, match=r"the adapter `rehearse_only` saved at .*adapters/rehearse_only has rank 16 and alpha 32 in its adapter_config.json and matrices of rank .16.; "
                                           r"the start adapter `pre_r64` is rank 64 and alpha 128"):
        _on_the_gpu(monkeypatch, train, config)
    assert not store.is_done("ladder_l4_rows2_train_rehearse_only")
    with pytest.raises(RuntimeError, match="the measurement of L4t2's model `rehearse_only` needs the step ladder_l4_rows2_train_rehearse_only of this run, which is not done: the stage "
                                           "`ladder_l4_rows2_r64`"):
        measure(config)
    shutil.rmtree(store.root / "adapters" / "rehearse_only")
    _trainings(monkeypatch, calls)
    assert _on_the_gpu(monkeypatch, train, config)["adapter_saved"]["ranks"] == [64] and store.is_done("ladder_l4_rows2_train_rehearse_only")
    # The stand-in trains and saves nothing: nothing is read back.
    assert "adapter_saved" not in ladder_l4_rows2.STEPS["ladder_l4_rows2_train_reward_rehearse"](config)
    # As on the GPU: the measure step serves ITS adapter, under a name of its own, from a model server started for rank 64 at the config's temperature.
    del kits[:]
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        measure(config)
        with pytest.raises(RuntimeError, match="adapters/reward_rehearse is not there: the adapter `reward_rehearse` is kept by this stage"):
            ladder_l4_rows2.STEPS["ladder_l4_rows2_measure_reward_rehearse"](config)
    assert [(kit["temperature"], kit["adapter"], kit["max_lora_rank"]) for kit in kits] == [(1.0, ("ladder_l4_rows2_r64_rehearse_only", 1, str(store.root / "adapters" / "rehearse_only")), 64)] * 3
    # The check's setting moved after the run was prepared: refused before anything is trained or sampled.
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "alpha", 64)
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "fallback_alpha", 32)
    with pytest.raises(RuntimeError, match="this run was prepared from `pre_r64` at rank 64 and alpha 128 .what the run that made it recorded., and ladder_loop.l4.rank_check now gives it "
                                           "rank 64 and alpha 64"):
        ladder_l4_rows2.STEPS["ladder_l4_rows2_measure_reward_rehearse"](config)


def test_a_set_lean_did_not_answer_fails_the_report_and_is_sampled_again_from_the_kept_adapter_when_the_task_is_queued_again(rows2, monkeypatch, tmp_path):
    config, calls = rows2.config, rows2.calls
    first = _stage2(config, monkeypatch, until="ladder_l4_rows2_measure_reward_rehearse")
    store = rows2.store()
    unanswered = "l4_reach_reward_rehearse"
    name = f"episodes_{unanswered}_problems.jsonl"
    as_measured, own = store.path(name).read_bytes(), store.read_rows(name)
    own[0]["attempts_without_an_answer"] = 20
    store.write_rows(name, own)
    monkeypatch.setattr(milestone2, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", ladder_l4_rows2.REPORT, "--out", str(tmp_path / "report.json")])
    assert gpu_main.main() != 0                                                # the report is written, is not to be read, and its step fails
    report = json.loads(store.path(ladder_l4_rows2.REPORT_FILE).read_text())
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_reward_rehearse_1"] and "NOT TO BE READ" in report["lines"][-1] and report["branches"]["question_2"] == INCONCLUSIVE
    assert "lean_answered" in report["models"]["reward_rehearse"]["failed_checks"] and "lean_answered" not in report["models"]["rehearse_only"]["failed_checks"]
    assert report["models"]["reward_rehearse"]["can_this_model_be_read"]["lean_answered"]["sets_not_answered"] == ["goal_reward_rehearse_1"]
    # The task queued again: that set, and no other, is sampled again from the kept adapter; nothing is trained.
    untouched = {key: content for key, content in _all(store.root).items() if unanswered not in key and "report" not in key and "measure_reward_rehearse" not in key}
    sampled = rows2.r64.l4.solver.calls
    again = _stage2(config, monkeypatch)
    assert len(calls) == 2 and again["ladder_l4_rows2_measure_reward_rehearse"]["measured_again"] == [unanswered] and rows2.r64.l4.solver.calls > sampled
    after = _all(store.root)
    assert all(after[key] == content for key, content in untouched.items()) and after[name] == as_measured
    assert again["ladder_l4_rows2_report"]["ok"] is True and again["ladder_l4_rows2_report"]["not_to_be_read"] == []
    assert first["ladder_l4_rows2_measure_rehearse_only"] == again["ladder_l4_rows2_measure_rehearse_only"]


def test_the_steps_run_through_the_gpu_entry_point(rows2, monkeypatch, tmp_path):
    monkeypatch.setattr(milestone2, "load_config", lambda path: rows2.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert rows2.store().is_done(ladder_l4_rows2.REPORT)


# ----------------------------------------------------------- L4t, from `pre` and from `pre_r64`, is what it was
def test_l4t_from_pre_r64_writes_what_it_wrote_and_is_left_as_it_was_after_this_stage_ran_beside_it(rows2, monkeypatch):
    config, l4t = rows2.config, rows2.l4t
    as_it_wrote = _all(l4t)
    # What L4t from `pre_r64` leaves: the files its own tests froze, 100 in all in this world; its ten steps; its report's branches.
    assert sorted(name for name in as_it_wrote if not name.startswith("episodes_")) == sorted(L4T_FILES_FROM_PRE_R64) and len(as_it_wrote) == 100
    assert list(rows2.l4t_summaries) == L4T_STEPS_FROM_PRE_R64 == ladder_l4_rows.steps_of("pre_r64") and list(rows2.l4t_summaries["ladder_l4_rows_report"]["branches"]) == list(STORED)
    _stage2(config, monkeypatch)
    assert _all(l4t) == as_it_wrote and not [name for name in as_it_wrote if "rows2" in name or any(model in name for model in MODELS)]
    # L4t's own task, queued again AFTER this stage ran in the same store: every step returns what is stored, nothing is trained or sampled, no file moves.
    monkeypatch.setenv(L4T_START, "pre_r64")
    monkeypatch.setenv(L4T_MINIMUM, "2")
    trained, sampled = len(rows2.calls), rows2.r64.l4.solver.calls
    again = _l4t(config, monkeypatch)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in rows2.l4t_summaries.items() if "report" not in name}
    assert {key: value for key, value in again["ladder_l4_rows_report"].items() if key != "adapters"} == {
        key: value for key, value in rows2.l4t_summaries["ladder_l4_rows_report"].items() if key != "adapters"}
    assert (len(rows2.calls), rows2.r64.l4.solver.calls) == (trained, sampled) and _all(l4t) == as_it_wrote
    # Its names are its own as they were: the stage, its files, its markers; this stage's are beside them and never the same.
    of_l4t, own = ladder_l4_rows.THE_STAGE, ladder_l4_rows2.THE_STAGE
    assert (of_l4t.name, of_l4t.steps, of_l4t.files, of_l4t.reader, of_l4t.prepare) == ("ladder_l4_rows", "ladder_l4_rows", "l4_rows", "L4t", "ladder_l4_rows_prepare") == (
        ladder_l4_rows.STAGE, ladder_l4_rows.STAGE, ladder_l4_rows.L4T, "L4t", ladder_l4_rows.PREPARE)
    assert (ladder_l4_rows.training_file("rehearse"), ladder_l4_rows.loss_file("rehearse"), ladder_l4_rows.train_marker("rehearse"), ladder_l4_rows.measure_marker("rehearse")) == (
        "l4_rows_training_rehearse.jsonl", "l4_rows_loss_rehearse.json", "ladder_l4_rows_train_rehearse", "ladder_l4_rows_measure_rehearse")
    assert (own.name, own.steps, own.files, own.reader, own.prepare, own.start_said_by) == ("ladder_l4_rows2_r64", "ladder_l4_rows2", "l4_rows2", "L4t2", "ladder_l4_rows2_prepare",
                                                                                        "ladder_loop.l4.rows2.start")
    assert not set(ladder_l4_rows.STEPS) & set(ladder_l4_rows2.STEPS) and (own.training_file("rehearse_only"), own.loss_file("rehearse_only")) == (
        "l4_rows2_training_rehearse_only.jsonl", "l4_rows2_loss_rehearse_only.json")


def test_l4t_from_pre_writes_what_it_wrote(rows, monkeypatch):  # noqa: F811
    config = rows.config
    summaries = _l4t_from_pre(config, monkeypatch)
    written = _all(rows.store().root)
    # What L4t from `pre` leaves: the files its own tests froze, 79 in all in this world; its eight steps; no step, file or key of another start or of this stage.
    assert rows.store().root.name == "ladder_l4_rows_seed0" and sorted(name for name in written if not name.startswith("episodes_")) == L4T_FILES_FROM_PRE and len(written) == 79
    assert list(summaries) == L4T_STEPS_FROM_PRE == ladder_l4_rows.steps_of("pre") and not [name for name in written if "rows2" in name or "old_rule" in name or "rehearse_only" in name]
    assert [call["tensorboard_run"] for call in rows.calls] == ["ladder_l4_rows_seed0_rehearse", "ladder_l4_rows_seed0_reward_rows"]
    assert all(summaries[f"ladder_l4_rows_train_{model}"]["model"] == model and "adapter_saved" not in summaries[f"ladder_l4_rows_train_{model}"] for model in ("rehearse", "reward_rows"))
    # This stage does not run from `pre`: its setting refuses it, and L4t's settings from `pre` are read as they were (the setting, no variable).
    assert ladder_l4_rows.the_settings(config).start == "pre" and ladder_l4_rows.the_settings(config, "pre_r64").start == "pre_r64" and ladder_l4_rows.the_settings(config, None).start == "pre"
    monkeypatch.setenv(L4T_START, "pre_r64")
    assert ladder_l4_rows.the_settings(config).start == "pre_r64" and ladder_l4_rows.the_settings(config, "pre").start == "pre"      # a start that is NAMED is the start, whatever the variable says


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_reads_l4ts_smoke_run_from_pre_r64_and_the_rank_checks_and_nothing_else(arm, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RUN, L4T_RUN, L4T_OWN_RUN, L4T_ARM_RUN, L4T_MINIMUM, L4T_START, L4T_START_CHECKS, ladder_l4_rank.CHECK_VARIABLE, RANK_RUN):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings(monkeypatch, calls)
    engines = lambda self, adapter=None: (lambda given: (kits.append((given["sampling"]["temperature"], given["vllm"]["max_lora_rank"])), (arm.solver, stand_in_parameters))[1])      # noqa: E731
    # ---- the four smoke tasks before it, each in ITS OWN environment: the pretraining's, the arm's, the check's, then L4t's from `pre_r64`
    steps = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS, **ladder_l4_rank.STEPS, **ladder_l4_rows.STEPS}
    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke"):
        options = entry.step_fields(entry.STAGES[stage_name][2])[2]
        with monkeypatch.context() as patched:                                 # one task's environment: gone when the task is
            patched.delenv(ladder_loop.DATA_VARIABLE)
            for name, value in options["environment"].items():
                patched.setenv(name, value)
            if stage_name == "ladder_l4_smoke":
                arm.with_lean()
                patched.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
            if stage_name == "ladder_l4_rows_r64_smoke":
                patched.setattr(ladder_round.Engines, "kit", engines)
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                if environment == "gpu" and step in steps:
                    _on_the_gpu(patched, steps[step], config) if "_train" in step else steps[step](config)
    runs = pipeline.STORE / "runs-v4.27"
    l4t, rank_run = runs / "ladder_l4_rows_r64_smoke", runs / "ladder_l4_rank_smoke"
    assert (l4t / "report_ladder_l4_rows.json").exists() and (rank_run / "adapters" / "pre_r64" / "adapter_config.json").exists()
    as_stored = {path: _all(path) for path in runs.iterdir() if path.is_dir()}
    assert set(path.name for path in as_stored) >= {"ladder_l1_smoke", "ladder_l4_pretrain_smoke", "ladder_l4_smoke", "ladder_l4_rank_smoke", "ladder_l4_rows_r64_smoke"}
    # ---- this stage's smoke task: L4t's smoke run and the check's as the two it reads, the fixtures, a run directory of its own; NO variable of L4t's
    options = entry.step_fields(entry.STAGES["ladder_l4_rows2_r64_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_rows2_r64_smoke"])
    of_the_pretraining = entry.step_fields(entry.STAGES["ladder_l4_pretrain_smoke"][2])[2]["environment"]
    assert options["environment"] == {ladder_loop.DATA_VARIABLE: of_the_pretraining[ladder_loop.DATA_VARIABLE], ladder_l4.FILE_VARIABLE: str(FIXTURE), RANK_RUN: "ladder_l4_rank_smoke",
                                      L4T_RUN: "ladder_l4_rows_r64_smoke", RUN: "ladder_l4_rows2_r64_smoke"}
    of_l4ts_smoke = entry.step_fields(entry.STAGES["ladder_l4_rows_r64_smoke"][2])[2]["environment"]
    assert (of_l4ts_smoke[L4T_OWN_RUN], of_l4ts_smoke[RANK_RUN]) == (options["environment"][L4T_RUN], options["environment"][RANK_RUN])      # the two runs it reads are the ones those tasks write
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ladder_round.Engines, "kit", engines)
    del calls[:], kits[:]
    # It reads NOTHING else: with the three other smoke runs gone from the box it runs to its end.
    for name in ("ladder_l1_smoke", "ladder_l4_pretrain_smoke", "ladder_l4_smoke"):
        (runs / name).rename(runs / f"{name}_elsewhere")
    summaries = _stage2(config, monkeypatch, stage_name="ladder_l4_rows2_r64_smoke")
    for name in ("ladder_l1_smoke", "ladder_l4_pretrain_smoke", "ladder_l4_smoke"):
        (runs / f"{name}_elsewhere").rename(runs / name)
    store = ladder_l4_rows2._store(config, ladder_l4_rows2.the_settings(config))
    assert store.root.name == "ladder_l4_rows2_r64_smoke" and list(summaries) == STEPS and {path: _all(path) for path in as_stored} == as_stored
    assert sorted(path.name for path in runs.iterdir() if path.name.startswith("ladder_l4_rows")) == ["ladder_l4_rows2_r64_smoke", "ladder_l4_rows_r64_smoke"]
    prepare = summaries["ladder_l4_rows2_prepare"]
    assert (prepare["start"], prepare["start_run"], prepare["start_adapter"], prepare["l4t_run"]) == ("pre_r64", str(rank_run), str(rank_run / "adapters" / "pre_r64"), str(l4t))
    assert [sampling["name"] for sampling in prepare["goal_samplings"]] == [REACH] and prepare["attempts_a_goal_problem"] == 32 and prepare["start_recipe"]["rank"] == 64
    # Whether it went on from a start model whose own checks failed is what L4t's smoke run RECORDED of itself, and nothing this stage sets (on the box that report is
    # INCONCLUSIVE and L4t's smoke run recorded the waiver; in this world the scripted training's loss falls, and there is nothing to waive).
    of_l4t = json.loads((l4t / "ladder_l4_rows_prepare.done.json").read_text())
    assert prepare["start_run_read"]["checks_waived_as_l4ts_run_recorded"] is of_l4t["start_run_read"]["checks_waived_for_a_smoke_run"]
    assert prepare["start_run_read"]["checks_failed"] == of_l4t["start_run_read"]["checks_failed"] and L4T_START_CHECKS not in options["environment"]
    sets = prepare["training_sets"]
    assert sets["rehearse_only"]["rows"] == sets["rehearse_only"]["rehearsal_rows"] == of_l4t["training_sets"]["rehearse"]["rehearsal_rows"] > 0
    assert sets["reward_rehearse"]["rows"] == of_l4t["training_sets"]["reward_rows"]["rows"] + sets["reward_rehearse"]["one_shot_rows_of_reward_rows"]
    # BOTH models are trained from the smoke `pre_r64` AT RANK 64 and measured under a model server started for rank 64, at the config's temperature.
    assert [(call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(64, 128, {"start": rank_run / "adapters" / "pre_r64"})] * 2
    assert [list(call["schedule"]["checkpoint_steps"]) for call in calls] == [["rehearse_only"], ["reward_rehearse"]]
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == sorted(MODELS) and kits == [(1.0, 64)] * 4       # the rungs and G for each model
    report = summaries["ladder_l4_rows2_report"]
    assert report["label"] == "pretrained on published proofs" and all(line.startswith(f"{SAY}: ") for line in report["lines"]) and store.is_done(ladder_l4_rows2.REPORT)
    # ONE sampling of G: no fresh attempts, so ADDS and Question 1 are not read, and no rule is named.
    assert (report["branches"]["question_1"], report["branches"]["the_arms_rule"]) in ((NOT_READ, NO_RULE), (INCONCLUSIVE, NO_RULE)) and report["the_three_reads"]["fresh_attempts"] is None
    assert any("the stored models have ONE sampling of G here (a smoke run)" in line for line in report["lines"]) and SECOND_PASS not in report["headline"]
    assert LOOPS_ROWS_ADD not in report["headline"] and NOT_SEPARATED not in report["headline"]


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_names_no_variable_of_l4ts():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_rows2_r64"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    assert all(options == {} for _, _, options in steps) and list(ladder_l4_rows2.STEPS) == STEPS and ladder_l4_rows2.ENTRY_STAGE == "ladder_l4_rows2_r64"
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_rows2_r64_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    # THE ORDER IS THE SETTING'S: each model trained, then measured, before the next; `rehearse_only` first (Question 1 is what an arm with rehearsal rows waits on).
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings = whole["ladder_loop"]["l4"]["rows2"]
    assert settings == {"start": "pre_r64", "models": ["rehearse_only", "reward_rehearse"]} and [step for _, step, _ in steps if step and "_train_" in step] == [
        f"ladder_l4_rows2_train_{model}" for model in settings["models"]] == list(TRAININGS)
    rows_ = ladder_l4_rows2.the_settings(whole)
    assert (rows_.start, rows_.models, rows_.l4t.start, rows_.l4t.models) == ("pre_r64", MODELS, "pre_r64", STORED) and ladder_l4_rows2.run_name(rows_, 0) == "ladder_l4_rows2_pre_r64_seed0"
    assert ladder_l4_rows.run_name(rows_.l4t, 0) == "ladder_l4_rows_pre_r64_seed0" and ladder_l4_rank.check_of(whole, settings["start"]).rank == 64
    # L4t's own settings are what they were, and this stage's are its own block: one reader each.
    assert whole["ladder_loop"]["l4"]["rows"]["start"] == "pre" and ladder_l4_rows.the_settings(whole).start == "pre"
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["l4"]["rows2"]' in path.read_text()] == ["gpu/ladder_l4_rows2.py"]
    # Its steps are its own, and only its two stages run them; no other module's step has one of its names.
    others = set()
    for module in (ladder_round, ladder_l2, ladder_loop, ladder_ceiling, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4b):
        others |= set(module.STEPS)
    assert not set(ladder_l4_rows2.STEPS) & others and all(name.startswith("ladder_l4_rows2_") for name in ladder_l4_rows2.STEPS)
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l4_rows2.STEPS for step in entry.STAGES[name])) == ["ladder_l4_rows2_r64", "ladder_l4_rows2_r64_smoke"]
    # Its two variables are named by its smoke stage alone, and it names NONE of L4t's (the start is a setting; the smoke start's failed checks are waived by L4t's record).
    names = lambda variable: sorted(name for name, own in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own))      # noqa: E731
    assert names(RUN) == names(L4T_RUN) == ["ladder_l4_rows2_r64_smoke"]
    for variable in (L4T_START, L4T_MINIMUM, L4T_OWN_RUN, L4T_ARM_RUN, L4T_START_CHECKS):
        assert not [name for name in names(variable) if name.startswith("ladder_l4_rows2")], variable
    gpu = {path.name: path.read_text() for path in (PACKAGE / "gpu").glob("*.py")}
    assert not [variable for variable in (L4T_START, L4T_MINIMUM, L4T_ARM_RUN) if variable in gpu["ladder_l4_rows2.py"]]
    # ONE BODY EACH: this stage's train and measure steps are L4t's own two bodies and its report reads its models by L4t's own reader; it calls no training or measuring function itself.
    assert all(name in gpu["ladder_l4_rows.py"] for name in ("def train_a_model(", "def measure_a_model(", "def read_the_trained(")) and gpu["ladder_l4_rows2.py"].count("train_a_model(") == 1
    assert gpu["ladder_l4_rows2.py"].count("measure_a_model(") == 1 and gpu["ladder_l4_rows2.py"].count("read_the_trained(") == 1
    assert not [name for name in ("one_pass(", "measure_model(", "examples_of(", "the_adapter_saved(", "hot_config(", "max_lora_rank") if name in gpu["ladder_l4_rows2.py"]]
    assert ladder_l4_rows2.train_a_model is ladder_l4_rows.train_a_model and ladder_l4_rows2.measure_a_model is ladder_l4_rows.measure_a_model
    # The steps of the stage are in the GPU entry point's own table.
    main = (PACKAGE / "gpu" / "__main__.py").read_text()
    assert "**ladder_l4_rows2.STEPS" in main and "ladder_l4_rows2," in main
