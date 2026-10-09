"""L4t, the stage end to end on what the arm's run and `pre`'s stored: the two training sets built from the stored rounds
(the refusals of its prepare step), each model trained FROM the start adapter and measured as `with` was, `with` and `pre`
sampled again on G alone at the hot temperature (and no other step of any stage at another temperature than the
config's), the report; the two runs it reads left as they were, and the stages before it writing what they wrote; the
smoke stage; the registration. Spec: docs/spec/ladder-loop.spec.md, "L4t: what should a round train on? Three
one-change checks on the rounds already made". The engine and Lean are scripted; nothing touches a GPU or the network. The
rules and the report on hand-made rows are `test_ladder_l4_rows.py`."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import arm  # noqa: E402, F401
from test_ladder_l2_stage import ARM, loop  # noqa: E402, F401
from test_ladder_l4_rank_stage import CHECK, PRETRAINING_FILES, _check, _trainings, _written  # noqa: E402 - the check's own steps, and what the pretraining writes
from test_ladder_l4_stage import CHANGED, FIXTURE, _file_rows, _files, _on_the_gpu, _pretrained, _run, _trainings_as_the_gpu_would, l4  # noqa: E402, F401
from test_ladder_round import ScriptedLean, _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.assembly import training_order  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import BRANCHES, MODELS, THE_MINIMUM, THE_RULE, reward_draws, uniform  # noqa: E402
from rlvr_lean.domain.problem_pool.episodes import reward  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_loop, ladder_round, milestone2, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
SAY = "l4 (pretrained on published proofs): "
RUN, ARM_RUN, MINIMUM = ladder_l4_rows.RUN_VARIABLE, ladder_l4_rows.ARM_RUN_VARIABLE, ladder_l4_rows.MINIMUM_VARIABLE
SEEDS = "RLVR_LEAN_TRAINING_SEEDS"
STEPS = ["ladder_l4_rows_prepare", "ladder_l4_rows_train_rehearse", "ladder_l4_rows_measure_rehearse", "ladder_l4_rows_train_reward_rows", "ladder_l4_rows_measure_reward_rows",
         "ladder_l4_rows_measure_hot_with", "ladder_l4_rows_measure_hot_pre", "ladder_l4_rows_report"]
TRAININGS = ("ladder_l4_rows_train_rehearse", "ladder_l4_rows_train_reward_rows")
# What the stage leaves in its run directory, beside the episodes of the sets it samples.
OWN_FILES = [
    "adapters/rehearse/adapter_model.safetensors", "adapters/reward_rows/adapter_model.safetensors", "l4_goal_set_again.jsonl", "l4_heldout_groups.jsonl",
    "l4_rows_loss_rehearse.json", "l4_rows_loss_reward_rows.json", "l4_rows_reward_draws.jsonl", "l4_rows_stored_models.json", "l4_rows_training_rehearse.jsonl",
    "l4_rows_training_reward_rows.jsonl", "l4_stored_pre_more.jsonl", "l4_stored_pre_reach.jsonl", "l4_stored_pre_rungs.jsonl", "l4_stored_with_more.jsonl",
    "l4_stored_with_reach.jsonl", "l4_stored_with_rungs.jsonl", "l4_stored_without_more.jsonl", "l4_stored_without_reach.jsonl", "l4_stored_without_rungs.jsonl",
    "ladder_l4_rows_measure_hot_pre.done.json", "ladder_l4_rows_measure_hot_with.done.json", "ladder_l4_rows_measure_rehearse.done.json",
    "ladder_l4_rows_measure_reward_rows.done.json", "ladder_l4_rows_prepare.done.json", "ladder_l4_rows_report.done.json", "ladder_l4_rows_train_rehearse.done.json",
    "ladder_l4_rows_train_reward_rows.done.json", "problems.jsonl", "report_ladder_l4_rows.json"]
# What the check of the adapter's rank (L4r) leaves in ITS run directory, beside its episodes: frozen here, so that a stage built after it is seen not to move it.
RANK_FILES = [
    "adapters/pre_r64/adapter_config.json", "adapters/pre_r64/adapter_model.safetensors", "l4_heldout_groups.jsonl", "l4_pretrain_loss.json", "l4_pretraining_rows.jsonl",
    "l4_rank_loss_of_pre.json", "l4_stored_base_more.jsonl", "l4_stored_base_reach.jsonl", "l4_stored_base_rungs.jsonl", "l4_stored_loop_more.jsonl", "l4_stored_loop_reach.jsonl",
    "l4_stored_loop_rungs.jsonl", "l4_stored_models.json", "l4_stored_pre.json", "l4_stored_pre_more.jsonl", "l4_stored_pre_reach.jsonl", "l4_stored_pre_rungs.jsonl",
    "ladder_l4_rank_measure.done.json", "ladder_l4_rank_prepare.done.json", "ladder_l4_rank_report.done.json", "ladder_l4_rank_train.done.json", "problems.jsonl",
    "report_ladder_l4_rank.json"]


def _stage(config, monkeypatch, until=None, stage_name="ladder_l4_rows"):
    """The stage's steps in its order, the trainings by their GPU path."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in ladder_l4_rows.STEPS:
            continue
        run = ladder_l4_rows.STEPS[step]
        summaries[step] = _on_the_gpu(monkeypatch, run, config) if step in TRAININGS else run(config)
        if step == until:
            break
    return summaries


def _all(directory):
    """Every file under a run directory, by its path under it."""
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


@pytest.fixture
def rows(l4, monkeypatch):  # noqa: F811
    """L4's world with what the stage reads on the box: `pre`'s run, then the arm's (two rounds of two problems), every
    training by its GPU path so that the adapters are on disk. `kits`: the sampling temperature every engine was built
    with from the arm's first round on, and the adapter it served."""
    for name in (RUN, ARM_RUN, MINIMUM, SEEDS):
        monkeypatch.delenv(name, raising=False)
    calls, kits = [], []
    _pretrained(l4, monkeypatch, calls)
    l4.the_arm()
    monkeypatch.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (
        kits.append({"temperature": given["sampling"]["temperature"], "adapter": adapter, "sampling": given["sampling"]}), (l4.solver, stand_in_parameters))[1]))
    with monkeypatch.context() as patched:                                     # the trainings, and only they, take their GPU path
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        _run(l4.config, "ladder_l4", rounds=2, until="ladder_l3d2_train_without")
    _run(l4.config, "ladder_l4", rounds=2)
    arm_run, pre = l4.store().root, l4.pretrain_store().root
    monkeypatch.delenv(ARM)                                                    # a task of this stage names no arm and no number of rounds
    monkeypatch.delenv(ladder_l2.L2_ROUNDS_VARIABLE)
    monkeypatch.setenv(MINIMUM, "2")                                           # the fixtures' rounds make a handful of rows: as the smoke run, at least two are kept
    del calls[:]
    settings = lambda: ladder_l4_rows.the_settings(l4.config)      # noqa: E731
    return SimpleNamespace(config=l4.config, l4=l4, calls=calls, kits=kits, arm=arm_run, pre=pre, store=lambda: ladder_l4_rows._store(l4.config, settings()))


# ------------------------------------------------------------------------------------------------ the stage
def test_the_stage_builds_the_two_sets_from_the_stored_rounds_trains_each_from_pre_measures_them_and_samples_with_and_pre_again_at_the_hot_temperature(rows, monkeypatch):
    config, calls, kits = rows.config, rows.calls, rows.kits
    arm_run, pre = rows.arm, rows.pre
    assert arm_run.name == "ladder_l2_t010_assembly_pre_seed0" and pre.name == "ladder_l4_pretrain_seed0"
    assert kits and all(kit["temperature"] == 1.0 for kit in kits)             # the arm's rounds and its two measurements: the config's temperature
    read_only = {path: _all(path) for path in (arm_run, pre)}
    del kits[:]
    summaries = _stage(config, monkeypatch)
    store = rows.store()
    assert store.root.name == "ladder_l4_rows_seed0" and store.root.parent == pre.parent and list(summaries) == ladder_l4_rows.steps_of("pre") == STEPS
    assert {path: _all(path) for path in read_only} == read_only               # THE TWO RUNS WERE ONLY READ: file for file, adapters included
    assert config["sampling"]["temperature"] == 1.0                            # and the config itself was not changed

    # ---- prepare: the two training sets, from the stored rounds
    prepare = summaries["ladder_l4_rows_prepare"]
    of_the_arm = json.loads((arm_run / "ladder_l2_prepare.done.json").read_text())
    start, with_adapter = pre / "adapters" / "pre", arm_run / "adapters" / "m2"
    assert (prepare["stage"], prepare["label"], prepare["check"], prepare["seed"], prepare["start"], prepare["arm"]) == ("l4", "pretrained on published proofs", "L4t", 0, "pre", "t010_assembly_pre")
    assert (prepare["start_adapter"], prepare["with_adapter"], prepare["arm_run"], prepare["pretraining_run"]) == (str(start), str(with_adapter), str(arm_run), str(pre))
    assert (prepare["target_rate"], prepare["rounds"], prepare["models"]) == (0.10, [1, 2], ["rehearse", "reward_rows"]) == (of_the_arm["target_rate"], of_the_arm["rounds"], list(MODELS))
    twin = _file_rows(arm_run / ladder_l3d2.TRAINING_WITHOUT_FILE)
    of_the_rounds = [row for number in (1, 2) for row in _file_rows(arm_run / f"training_examples_r{number}.jsonl")]
    picks = {row["problem_id"]: row for number in (1, 2) for batch in (1, 2) for row in _file_rows(arm_run / f"episodes_round_r{number}_b{batch}_problems.jsonl")}
    assert twin and len(of_the_rounds) > len(twin)                             # the fixtures' rounds: one-shot rows, and an assembled one
    # `rehearse`: exactly the twin's rows, and as many rows of the pretraining file, no row twice, in ONE order (the arm's content hash).
    rehearse = store.read_rows("l4_rows_training_rehearse.jsonl")
    of_the_twin, rehearsal = [row for row in rehearse if row["origin"] == "attempt"], [row for row in rehearse if row["origin"] == "pretraining"]
    assert sorted(row["id"] for row in of_the_twin) == sorted(row["id"] for row in twin) and len(rehearsal) == len(twin) == len({row["problem_id"] for row in rehearsal})
    assert [row["id"] for row in rehearse] == [row["id"] for row in training_order(rehearse, 0)] and [row["row"] for row in rehearse] == list(range(len(rehearse)))
    file_rows = _file_rows(FIXTURE)
    assert {row["problem_id"] for row in rehearsal} <= {row["problem_id"] for row in file_rows} and all(file_rows[row["row_of_the_file"]]["problem_id"] == row["problem_id"] for row in rehearsal)
    assert all(row["k"] is None and "theorem" not in row and "completion" not in row for row in rehearsal)
    assert all(row["k"] == picks[row["problem_id"]]["resolved"] and row["theorem"] and row["completion"] for row in of_the_twin)       # each with its problem's k, and its text
    # `reward_rows`: each row of the rounds with its k (an assembled row: 1), its reward and its draw; kept when the draw is under the reward.
    draws = store.read_rows("l4_rows_reward_draws.jsonl")
    assert sorted(row["id"] for row in draws) == sorted(row["id"] for row in of_the_rounds) and not any("completion" in row for row in draws)
    for row in draws:
        pick = picks[row["problem_id"]]
        assert row["k"] == (1 if row["origin"] == "assembled" else pick["resolved"]) and (pick["resolved"] == 0) == (row["origin"] == "assembled") and row["n"] == 8
        assert row["reward"] == round(reward(row["k"], 8, 0.10), 6) and row["draw"] == uniform(0, "l4_rows_keep", row["id"]) and row["kept"] is (row["draw"] < reward(row["k"], 8, 0.10))
    reward_rows = store.read_rows("l4_rows_training_reward_rows.jsonl")
    by_rule = [row["id"] for row in draws if row["kept"]]
    assert sorted(row["id"] for row in reward_rows if row["kept_by"] == THE_RULE) == sorted(by_rule) and len(reward_rows) == max(2, len(by_rule))
    assert all(row["kept_by"] in (THE_RULE, THE_MINIMUM) and row["theorem"] and row["completion"] for row in reward_rows)
    assert [row["id"] for row in reward_rows] == [row["id"] for row in training_order(reward_rows, 0)]
    sets = prepare["training_sets"]
    assert (sets["rehearse"]["rows"], sets["rehearse"]["twin_rows"], sets["rehearse"]["rehearsal_rows"]) == (2 * len(twin), len(twin), len(twin))
    assert sets["rehearse"]["rows_by_origin"] == {"attempt": len(twin), "pretraining": len(twin)} and sets["rehearse"]["optimizer_steps"] == -(-2 * len(twin) // 8)
    assert (sets["reward_rows"]["rows"], sets["reward_rows"]["rows_of_the_rounds"], sets["reward_rows"]["kept_by_the_rule"], sets["reward_rows"]["minimum_of_a_smoke_run"]) == (
        len(reward_rows), len(of_the_rounds), len(by_rule), 2) and sets["reward_rows"]["kept_by_the_runs_minimum"] == len(reward_rows) - len(by_rule)
    assert sum(sets["reward_rows"]["rows_by_k"].values()) == len(reward_rows) and sets["rehearse"]["lines"]["proofs"] == 2 * len(twin)
    assert sets["rehearse"]["barred"]["rehearsal_rows"] == len(twin) and sets["reward_rows"]["barred"]["held_out"] == []
    # NO PUBLISHED PROOF IS COPIED INTO THE RUN: the rehearsal rows are named, and their text stays in the pretraining file.
    as_stored_in_a_row = [json.dumps(row["proof"], ensure_ascii=False)[1:-1] for row in file_rows]     # a proof as a JSON row would hold it (its line ends escaped)
    assert not any(proof in content.decode(errors="ignore") for proof in (*as_stored_in_a_row, *(row["proof"].strip() for row in file_rows)) for content in _all(store.root).values())
    assert not any(key in row for row in rehearsal for key in ("proof", "statement"))
    # How every model is measured is what `pre`'s run recorded; the hot samplings have the setting's seeds, G alone.
    of_pre = json.loads((pre / "ladder_l4_pretrain_prepare.done.json").read_text())
    assert all(prepare[key] == of_pre[key] for key in ("sampling_seeds", "rung_episodes", "goal_samplings")) and prepare["attempts_a_goal_problem"] == of_pre["attempts_a_goal_problem"]
    hot = prepare["hot"]
    assert (hot["temperature"], hot["temperature_of_every_other_measurement"], hot["models"]) == (1.2, 1.0, ["with", "pre"])
    assert hot["samplings"] == [{"name": sampling["name"], "episodes": sampling["episodes"], "sampling_seed": seed} for sampling, seed in zip(of_pre["goal_samplings"], (1040, 1041))]
    for who, source, name in (("pre", pre, "episodes_l4_{part}_pre_problems.jsonl"), ("with", arm_run, "episodes_l3d2_{part}_with_problems.jsonl"),
                              ("without", arm_run, "episodes_l3d2_{part}_without_problems.jsonl")):
        for part in ("rungs", REACH, MORE):
            assert store.read_rows(ladder_ceiling.stored_file(who, part, "l4")) == _file_rows(source / name.format(part=part))
    assert [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)] == [row["problem_id"] for row in _file_rows(pre / ladder_l4.AGAIN_FILE)]

    # ---- train: each model FROM THE START ADAPTER, one pass over its stored set in its order, the arm's recipe
    by_problem = {row["problem_id"]: row for row in file_rows}
    for call, model, set_rows in zip(calls, MODELS, (rehearse, reward_rows)):
        train = summaries[f"ladder_l4_rows_train_{model}"]
        assert call["more"] == {"start": start} and call["orders"] == [list(range(len(set_rows)))] and call["schedule"] == {
            "passes": 1, "reading_steps": [], "checkpoint_steps": {model: -(-len(set_rows) // 8)}}
        assert (call["seed"], call["tensorboard_run"], call["directory"]) == (0, f"ladder_l4_rows_seed0_{model}", store.root / "adapters")
        assert call["examples"] == [{"problem_id": row["problem_id"], "side": "statement", "theorem": by_problem[row["problem_id"]]["statement"], "completion": by_problem[row["problem_id"]]["proof"]}
                                    if row["origin"] == "pretraining" else {key: row[key] for key in ("problem_id", "side", "theorem", "completion")} for row in set_rows]
        assert (train["model"], train["rows"], train["trained_from"], train["label"]) == (model, len(set_rows), f"the stored adapter {start}", "pretrained on published proofs")
        assert train["against_the_start_adapter"] == {model: CHANGED} and train["adapter"] == str(store.root / "adapters" / model) and (store.root / "adapters" / model).is_dir()
        loss = json.loads(store.path(f"l4_rows_loss_{model}.json").read_text())
        assert loss["rows_trained"] == [row["id"] for row in set_rows] and len(loss["row_losses"]) == len(set_rows) and loss["label"] == "pretrained on published proofs"
    assert len(calls) == 2

    # ---- measure: as `with` was, with `pre`'s sampling seeds, at the config's temperature; then `with` and `pre` again on G ALONE at the hot one
    assert len(kits) == 3 + 3 + 2 + 2                                          # in the order the steps ran: the rungs and G's two samplings for each trained model; G's two for each hot measurement
    assert [kit["temperature"] for kit in kits] == [1.0] * 6 + [1.2] * 4
    assert all({**kit["sampling"], "temperature": 1.0} == config["sampling"] for kit in kits)      # nothing else of the sampling settings moved
    for model in MODELS:
        measured = summaries[f"ladder_l4_rows_measure_{model}"]
        assert (measured["stage"], measured["label"], measured["model"]) == ("l4", "pretrained on published proofs", model)
        for of, own_set, pre_set in ((measured["rungs"], f"l4_rungs_{model}", "l4_rungs_pre"), (measured["goal"][REACH], f"l4_reach_{model}", "l4_reach_pre"),
                                     (measured["goal"][MORE], f"l4_more_{model}", "l4_more_pre")):
            marker = json.loads((pre / f"episodes_{pre_set}.done.json").read_text())
            assert of["set"] == own_set and (of["sampling_seed"], of["episodes_each"], of["problems"]) == (marker["sampling_seed"], marker["episodes_each"], marker["problems"])
    for who in ("with", "pre"):
        measured = summaries[f"ladder_l4_rows_measure_hot_{who}"]
        assert (measured["model"], measured["temperature"], measured["label"]) == (who, 1.2, "pretrained on published proofs") and "rungs" not in measured
        assert measured["adapter"] == str(with_adapter if who == "with" else start) and set(measured["goal"]) == {REACH, MORE}
        assert [(measured["goal"][sampling["name"]]["set"], measured["goal"][sampling["name"]]["sampling_seed"], measured["goal"][sampling["name"]]["episodes_each"])
                for sampling in hot["samplings"]] == [(f"l4_{sampling['name']}_hot_{who}", sampling["sampling_seed"], sampling["episodes"]) for sampling in hot["samplings"]]
        assert not store.path(f"episodes_l4_rungs_hot_{who}_problems.jsonl").exists()              # no rung is sampled, and no set of the rungs is named
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l4_{part}_{model}" for part in ("rungs", REACH, MORE) for model in MODELS} | {
        f"l4_{part}_hot_{who}" for part in (REACH, MORE) for who in ("with", "pre")}

    # ---- the report: each trained model by itself, then the table of `hot`; every line says what the models are
    report = summaries["ladder_l4_rows_report"]
    assert report["stage"] == "l4" and report["label"] == "pretrained on published proofs" and report["check"] == "L4t" and report["ok"] is True
    assert all(line.startswith(SAY) for line in report["lines"]) and "pretrained on published proofs" in report["headline"]
    assert set(report["branches"]) == set(MODELS) and all(name in BRANCHES for name in report["branches"].values())
    goal = [row["problem_id"] for row in store.read_rows(ladder_l4.GROUPS_FILE) if row["group"] == "goal"]
    for model in MODELS:
        read = report["models"][model]
        read = read["measured_and_not_read"] if read["inconclusive"] else read
        assert read["primary"]["problems"] == len(goal) == 1 and read["beside_the_primary"]["attempts_each"] == prepare["attempts_a_goal_problem"]
        own = lambda name: sum(row["resolved"] for part in (REACH, MORE) for row in store.read_rows(name(part)))      # noqa: E731
        assert read["beside_the_primary"]["successes"] == own(lambda part: f"episodes_l4_{part}_{model}_problems.jsonl")
        assert read["beside_the_primary"]["successes_of_the_base"] == own(lambda part: ladder_ceiling.stored_file("pre", part, "l4"))
        assert read["secondary"]["never_solved_before"]["given"] is False and read["secondary"]["trained_on"]["rows"] == sets[model]["rows"]
        assert read["secondary"]["trained_on"]["rows_by_origin"] == sets[model]["rows_by_origin"]
    assert report["hot"]["with"]["hot"]["all_of_g"]["attempts"] == hot["attempts_a_goal_problem"] and report["hot"]["temperature"] == 1.2
    assert json.loads(store.path(ladder_l4_rows.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_l4_rows.REPORT) == {"stage": "l4", "label": "pretrained on published proofs", "check": "L4t", "headline": report["headline"],
                                                         "branches": report["branches"], "ok": True}
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == ["rehearse", "reward_rows"]
    # The goal problems nothing stored had solved are GIVEN: a file in the run directory, one `problem_id` a row. The report is built again and reads it.
    store.write_rows(ladder_l4_rows.NEVER_SOLVED_FILE, [{"problem_id": goal[0]}, {"problem_id": "not_a_goal_problem"}])
    given = ladder_l4_rows.ladder_l4_rows_report(config)["models"]["rehearse"]
    given = (given["measured_and_not_read"] if given["inconclusive"] else given)["secondary"]["never_solved_before"]
    assert (given["given"], given["ids"], given["goal_problems"], given["not_goal_problems_of_this_run"]) == (True, 2, 1, 1)
    store.path(ladder_l4_rows.NEVER_SOLVED_FILE).unlink()
    # What it leaves: its own files, every one of which says l4 but the one name the shared episode step reads; nothing of it is in another run's directory.
    assert sorted(name for name in _all(store.root) if not name.startswith("episodes_")) == OWN_FILES
    assert [name for name in _files(store.root) if "l4" not in name] == ["problems.jsonl"]
    # A rerun returns what is stored: nothing is trained or sampled again.
    sent, sampled = len(rows.l4.lean.sources), rows.l4.solver.calls
    again = _stage(config, monkeypatch)
    assert len(calls) == 2 and (len(rows.l4.lean.sources), rows.l4.solver.calls) == (sent, sampled)
    assert {name: summary for name, summary in again.items() if "report" not in name} == {name: summary for name, summary in summaries.items() if "report" not in name}
    assert {path: _all(path) for path in read_only} == read_only


def test_the_prepare_step_refuses_before_anything_is_written(rows, monkeypatch, tmp_path):
    config, arm_run, pre = rows.config, rows.arm, rows.pre
    as_stored = {arm_run: _all(arm_run), pre: _all(pre)}
    root = rows.store().root

    def refused(match, error=RuntimeError):
        with pytest.raises(error, match=match):
            ladder_l4_rows.ladder_l4_rows_prepare(config)
        assert root.name == "ladder_l4_rows_seed0" and not _all(root)          # nothing was written: no marker, no file

    def with_a_change(directory, change, match, error=RuntimeError):
        """One of the two runs with one thing of it changed, refused; then put back as it was."""
        change()
        refused(match, error)
        for name, content in as_stored[directory].items():
            (directory / name).write_bytes(content)

    def rewritten(directory, name, **changed):
        return lambda: (directory / name).write_text(json.dumps({**json.loads(as_stored[directory][name]), **changed}))

    def rows_of(directory, name, change):
        own = [json.loads(line) for line in as_stored[directory][name].decode().splitlines() if line.strip()]
        return lambda: (directory / name).write_text("".join(json.dumps(row) + "\n" for row in change(own)))

    # ---- what is read of the arm's run: its steps' markers, the rounds' training examples and picks, the twin's rows, the stored rows with their markers
    for name, what in (("ladder_l2_prepare.done.json", "what the arm's steps recorded"), ("l3d2_training_without.jsonl", "the twin's rows"),
                       ("ladder_l3d2_measure_with.done.json", "what the arm's steps recorded"), ("training_examples_r2.jsonl", "a round's training examples"),
                       ("episodes_round_r1_b2_problems.jsonl", "a batch's picks with their k of n"), ("ladder_l2_assembly_r2_b1.done.json", "the marker of a batch's assembly"),
                       ("episodes_l3d2_more_without_problems.jsonl", "the per-problem rows of `with` or of `without`"), ("episodes_l3d2_rungs_with.done.json", "that set's own marker")):
        with_a_change(arm_run, (arm_run / name).unlink, f"ladder_l2_t010_assembly_pre_seed0 does not hold .'{name}'.: .*{what}.*L4t makes no round and no pretraining: it reads what "
                                                        "the task of stage `ladder_l4` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4 --seeds 0`; for the smoke run, "
                                                        "stage `ladder_l4_smoke`. wrote on this box. Run that task to its end first; nothing was written")
    # ---- ... and of `pre`'s: its record, its report, G', the rows it was trained on, its stored rows with their markers
    for name in ("ladder_l4_pretrain_prepare.done.json", "report_ladder_l4_pretrain.json", "l4_goal_set_again.jsonl", "l4_pretraining_rows.jsonl", "episodes_l4_reach_pre_problems.jsonl",
                 "episodes_l4_more_pre.done.json"):
        with_a_change(pre, (pre / name).unlink, f"ladder_l4_pretrain_seed0 does not hold .'{name}'.: .*it reads what the task of stage `ladder_l4_pretrain` for seed 0 "
                                                ".`python -m rlvr_lean.runner.entry --stage ladder_l4_pretrain --seeds 0`")
    # ---- the arm's run recorded another seed, another arm or ANOTHER START than the setting's
    for changed in ({"seed": 3}, {"arm": "t010_assembly"}, {"start": "pre_r64"}, {"start": None}):
        with_a_change(arm_run, rewritten(arm_run, "ladder_l2_prepare.done.json", **changed),
                      "the arm's run ladder_l2_t010_assembly_pre_seed0 recorded seed .* this task's seed is 0 and ladder_loop.l4.rows names the arm 't010_assembly_pre' and the "
                      "start 'pre': every model here is trained from the adapter the arm's own models were")
    with_a_change(arm_run, rewritten(arm_run, "ladder_l2_prepare.done.json", start_adapter=str(pre.with_name("another_run") / "adapters" / "pre")),
                  "was trained from .*another_run/adapters/pre and this task's start adapter is .*ladder_l4_pretrain_seed0/adapters/pre: not the same adapter")
    with_a_change(pre, rewritten(pre, "ladder_l4_pretrain_prepare.done.json", seed=3), "the pretraining run ladder_l4_pretrain_seed0 was made at seed 3 and "
                  "ladder_loop.l4.pretraining_seed is 0: every model here starts from the ONE pretraining of that seed")
    with_a_change(pre, rewritten(pre, "report_ladder_l4_pretrain.json", ok=False), "the report of the pretraining run ladder_l4_pretrain_seed0 is not to be read")
    # ---- the stored rows must pair by problem: `with` sampled otherwise than `pre` was, other problems, a set Lean did not answer
    marker = json.loads(as_stored[arm_run]["episodes_l3d2_reach_with.done.json"])
    with_a_change(arm_run, rewritten(arm_run, "episodes_l3d2_reach_with.done.json", sampling_seed=5),
                  f"ladder_l2_t010_assembly_pre_seed0 measured `with` .reach. with sampling seed 5 and {marker['episodes_each']} episodes; this config gives {marker['sampling_seed']} "
                  f"and {marker['episodes_each']}: L4t's models would not pair with the stored results")
    with_a_change(arm_run, rows_of(arm_run, "episodes_l3d2_rungs_without_problems.jsonl", lambda own: own[:-1]),
                  "measured `without` .rungs. on other problems than L1's run holds for it: L4t's models would not pair with it")
    with_a_change(pre, rows_of(pre, "episodes_l4_rungs_pre_problems.jsonl", lambda own: [{**own[0], "attempts_without_an_answer": 8}, *own[1:]]),
                  "measured `pre` .rungs. with too many attempts without a verdict from Lean: that set stands on the other side of a comparison")
    with_a_change(pre, rows_of(pre, "l4_goal_set_again.jsonl", lambda own: [*own, {"problem_id": "nowhere"}]), "1 problems of G' .ladder_l4_pretrain_seed0. are not goal problems")
    # ---- the pretraining file must be the one `pre` was trained on
    changed = tmp_path / "changed.jsonl"
    changed.write_text(FIXTURE.read_text() + "\n")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    refused("L4's pretraining file has SHA-256 .* and `pre` was trained on .* .ladder_l4_pretrain_seed0.: the rehearsal rows are rows `pre` was trained on")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    with_a_change(pre, rows_of(pre, "l4_pretraining_rows.jsonl", lambda own: own[:-1]), "stored 11 rows `pre` was trained on and L4's pretraining file holds 12")
    # ---- A TRAINING SET WITH A BARRED PROBLEM: a round's row of the `pretrain` half (its published proof was pretrained on), and a held-out problem
    one_shot = [json.loads(line) for line in as_stored[arm_run]["l3d2_training_without.jsonl"].decode().splitlines()][0]["problem_id"]
    held = _file_rows(arm_run / "l3d2_heldout_groups.jsonl")[0]["problem_id"]

    def renamed(new):
        def change():
            for name, content in as_stored[arm_run].items():
                if name.endswith(".jsonl") and (name.startswith(("training_examples_r", "l3d2_training_without")) or (name.startswith("episodes_round_") and "_problems" in name)):
                    (arm_run / name).write_text(content.decode().replace(f'"{one_shot}', f'"{new}'))
        return change

    with_a_change(arm_run, renamed("fixture_c3"), "1 problems of the training set of `rehearse` are rounds' rows of the `pretrain` half .first: fixture_c3.: refused, nothing was "
                                                  "written .the arm's run ladder_l2_t010_assembly_pre_seed0; `pre`'s ladder_l4_pretrain_seed0.. Nothing was written")
    with_a_change(arm_run, renamed(held), f"1 problems of the training set of `rehearse` are held-out problems .first: {held}")
    # ---- the stored rounds must agree with themselves: the twin's file is the rounds' one-shot rows
    with_a_change(arm_run, rows_of(arm_run, "l3d2_training_without.jsonl", lambda own: own[:-1]), "the twin's file holds .* rows and the rounds' training examples .* one-shot rows")
    # ---- with no minimum (every run but a smoke run) a rule that keeps no row is refused; here the fixtures' rounds may keep one
    monkeypatch.delenv(MINIMUM)
    kept = sum(row["kept"] for row in reward_draws([{**row, "k": 1 if row["origin"] == "assembled" else 3, "n": 8} for number in (1, 2)
                                                    for row in _file_rows(arm_run / f"training_examples_r{number}.jsonl")], 0.10, 0))
    if not kept:
        refused("the rule keeps none of the rounds' .* rows .each is kept when its draw is under its problem's reward.: there is nothing to train `reward_rows` on")
    monkeypatch.setenv(MINIMUM, "2")
    # ---- the hot measurements' sampling seeds are their own; the settings are the stage's
    settings = config["ladder_loop"]["l4"]["rows"]
    monkeypatch.setitem(settings, "sampling_seeds_hot", [1040, 1001])
    refused(r"ladder_loop.l4.rows.sampling_seeds_hot is .1040, 1001. and .*`pre`'s reach.* already sampled with one of them: the hot measurements have sampling seeds of their own")
    for value, match in (([1040, 1040], "two distinct whole numbers"), ([1040], "two distinct whole numbers"), ([1040, "1041"], "two distinct whole numbers")):
        monkeypatch.setitem(settings, "sampling_seeds_hot", value)
        refused(f"ladder_loop.l4.rows.sampling_seeds_hot is .*: {match}, one for each sampling of G", ValueError)
    monkeypatch.setitem(settings, "sampling_seeds_hot", [1040, 1041])
    for value in (1.0, 0, -1.2, "1.2", True):
        monkeypatch.setitem(settings, "temperature_hot", value)
        refused("ladder_loop.l4.rows.temperature_hot is .*: a positive number other than sampling.temperature .1.0., which every other measurement keeps", ValueError)
    monkeypatch.setitem(settings, "temperature_hot", 1.2)
    monkeypatch.setitem(settings, "models", ["reward_rows", "rehearse"])
    refused("ladder_loop.l4.rows.models is .'reward_rows', 'rehearse'. and the stage trains and measures .'rehearse', 'reward_rows'.: change both together", ValueError)
    monkeypatch.setitem(settings, "models", ["rehearse", "reward_rows"])
    # ---- ANOTHER START than `pre`: a run directory of its own; it is the model of the check of the adapter's rank, whose run is not on this box
    # (what a run from `pre_r64` reads, refuses and writes is test_ladder_l4_rows_r64.py)
    monkeypatch.setitem(settings, "start", "pre_r64")
    assert rows.store().root.name == "ladder_l4_rows_pre_r64_seed0"
    with pytest.raises(RuntimeError, match="ladder_l4_pretrain_r64_seed0 does not hold .*the report of the check that made `pre_r64`.* it reads what the task of stage "
                                           "`ladder_l4_rank` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rank --seeds 0`; for the smoke run, stage `ladder_l4_rank_smoke`. wrote"):
        ladder_l4_rows.ladder_l4_rows_prepare(config)
    assert not _all(rows.store().root) and not (rows.store().root.parent / "ladder_l4_pretrain_r64_seed0").exists()        # nothing was written, and no directory made for the run it lacks
    monkeypatch.setitem(settings, "start", "pre_r32")                          # the fallback's model: its run is not there either
    with pytest.raises(RuntimeError, match="ladder_l4_pretrain_r32_seed0 does not hold .*the task of stage `ladder_l4_rank_fallback` for seed 0 .`python -m rlvr_lean.runner.entry --stage ladder_l4_rank_fallback --seeds 0`"):
        ladder_l4_rows.ladder_l4_rows_prepare(config)
    monkeypatch.setitem(settings, "start", "the_base")                         # a name that is neither `pre` nor a model of the check
    with pytest.raises(RuntimeError, match="ladder_loop.l4.rows.start .or RLVR_LEAN_LADDER_L4_ROWS_START. is 'the_base': it names `pre` or a model of ladder_loop.l4.rank_check"):
        ladder_l4_rows.ladder_l4_rows_prepare(config)
    monkeypatch.setitem(settings, "start", "pre")
    # ---- a task of another seed reads that seed's arm, which is not on the box
    monkeypatch.setenv(SEEDS, "1")
    with pytest.raises(RuntimeError, match="ladder_l2_t010_assembly_pre_seed1 does not hold .*the task of stage `ladder_l4` for seed 1 .`python -m rlvr_lean.runner.entry --stage ladder_l4 --seeds 1`"):
        ladder_l4_rows.ladder_l4_rows_prepare(config)
    assert rows.store().root.name == "ladder_l4_rows_seed1" and not _all(rows.store().root)
    monkeypatch.delenv(SEEDS)
    # ---- as on the GPU: without the start adapter, or without `with`'s, nothing is prepared
    for adapter, match in ((pre / "adapters" / "pre", "adapters/pre is not there: the adapter `pre` is what every model here is trained from"),
                           (arm_run / "adapters" / "m2", "adapters/m2 is not there: the adapter `with` is what the arm's last model is, and it is sampled again here")):
        adapter.rename(adapter.with_name("elsewhere"))
        with monkeypatch.context() as patched:
            patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
            refused(match)
        adapter.with_name("elsewhere").rename(adapter)
    # ---- and with everything as it was, it prepares; the later steps refuse out of their order
    for step, match in ((ladder_l4_rows.STEPS["ladder_l4_rows_train_rehearse"], "the training of `rehearse` needs the step ladder_l4_rows_prepare of this run, which is not done: the stage `ladder_l4_rows`"),
                        (ladder_l4_rows.STEPS["ladder_l4_rows_measure_reward_rows"], "the measurement of L4t's model `reward_rows` needs the step ladder_l4_rows_prepare"),
                        (ladder_l4_rows.STEPS["ladder_l4_rows_measure_hot_pre"], "the measurement of `pre` at temperature 1.2 needs the step ladder_l4_rows_prepare"),
                        (ladder_l4_rows.ladder_l4_rows_report, "L4t's report needs the step ladder_l4_rows_prepare")):
        with pytest.raises(RuntimeError, match=match):
            step(config)
    assert ladder_l4_rows.ladder_l4_rows_prepare(config)["start"] == "pre" and {arm_run: _all(arm_run), pre: _all(pre)} == as_stored
    with pytest.raises(RuntimeError, match="the measurement of L4t's model `rehearse` needs the step ladder_l4_rows_train_rehearse of this run, which is not done"):
        ladder_l4_rows.STEPS["ladder_l4_rows_measure_rehearse"](config)
    with pytest.raises(RuntimeError, match="L4t's report needs the step ladder_l4_rows_train_rehearse"):
        ladder_l4_rows.ladder_l4_rows_report(config)


def test_the_hot_temperature_reaches_the_two_hot_steps_and_no_other_step_of_any_stage(rows, monkeypatch):
    config, kits = rows.config, rows.kits
    _stage(config, monkeypatch, until="ladder_l4_rows_measure_reward_rows")
    store, start, with_adapter = rows.store(), rows.pre / "adapters" / "pre", rows.arm / "adapters" / "m2"
    assert all(kit["temperature"] == 1.0 for kit in kits)                      # the arm's rounds and measurements, and this stage's two trained models
    # ---- as on the GPU: each hot step serves ITS model's stored adapter, under a name of its own, from an engine built with the hot temperature
    del kits[:]
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        hot_with = ladder_l4_rows.STEPS["ladder_l4_rows_measure_hot_with"](config)
        hot_pre = ladder_l4_rows.STEPS["ladder_l4_rows_measure_hot_pre"](config)
        # Without its adapter a measurement cannot be made.
        store.path("ladder_l4_rows_measure_hot_pre.done.json").unlink()
        start.rename(start.with_name("elsewhere"))
        with pytest.raises(RuntimeError, match="adapters/pre is not there: the adapter of `pre` is its own run's, only read here"):
            ladder_l4_rows.STEPS["ladder_l4_rows_measure_hot_pre"](config)
        start.with_name("elsewhere").rename(start)
        store.mark_done("ladder_l4_rows_measure_hot_pre", hot_pre)
    assert [(kit["temperature"], kit["adapter"]) for kit in kits] == [(1.2, ("ladder_l4_rows_hot_with", 1, str(with_adapter)))] * 2 + [(1.2, ("ladder_l4_rows_hot_pre", 1, str(start)))] * 2
    assert hot_with["temperature"] == hot_pre["temperature"] == 1.2 and config["sampling"]["temperature"] == 1.0
    # ---- NO OTHER STEP: the arm's own measurement and the pretraining's, run again by a task that carries this stage's variables, sample at the config's temperature
    del kits[:]
    for directory, markers in ((rows.pre, (ladder_l4.MEASURE, "episodes_l4_rungs_pre", "episodes_l4_reach_pre", "episodes_l4_more_pre")),
                               (rows.arm, (ladder_l3d2.measure_marker("with"), "episodes_l3d2_rungs_with", "episodes_l3d2_reach_with", "episodes_l3d2_more_with"))):
        for marker in markers:
            (directory / f"{marker}.done.json").unlink()
        for path in [*directory.glob("episodes_l4_*_pre_block_*.done.json"), *directory.glob("episodes_l3d2_*_with_block_*.done.json")]:
            path.unlink()
    ladder_l4.ladder_l4_pretrain_measure(config)
    monkeypatch.setenv(ARM, "t010_assembly_pre")
    monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, "2")
    ladder_l3d2.STEPS["ladder_l3d2_measure_with"](config)
    assert len(kits) == 6 and all(kit["temperature"] == 1.0 and kit["sampling"] is config["sampling"] for kit in kits)
    # ---- where a temperature is put into a config is ONE function of ONE module, called by the two hot steps alone; the setting has ONE reader
    gpu = {path.name: path.read_text() for path in (PACKAGE / "gpu").glob("*.py")}
    assert sorted(name for name, text in gpu.items() if "hot_config(" in text) == ["ladder_l4_rows.py"] and gpu["ladder_l4_rows.py"].count("hot_config(") == 2      # defined, and called once
    assert "measure_model(hot_config(config, rows)" in gpu["ladder_l4_rows.py"].split("def ladder_l4_rows_measure_hot")[1].split("\ndef ")[0]
    assert sorted(name for name, text in gpu.items() if re.search(r"[\"']temperature[\"']\s*:\s*rows\.temperature", text)) == ["ladder_l4_rows.py"]
    assert sorted(name for name, text in gpu.items() if re.search(r"[\"']sampling[\"']\s*:\s*\{", text)) == ["ladder_l4_rows.py"]              # no other step builds a sampling config
    assert [str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py") if '["l4"]["rows"]' in path.read_text()] == ["gpu/ladder_l4_rows.py"]
    assert gpu["ladder_l4_rows.py"].count('["l4"]["rows"]') == 1 and not [name for name in ("ladder_l4.py", "ladder_l4_rank.py", "ladder_l3d2.py", "ladder_l2.py") if "temperature_hot" in gpu[name]]
    of_the_hot = ladder_l4_rows.hot_config(config, ladder_l4_rows.the_settings(config))
    assert {key: value for key, value in of_the_hot.items() if key != "sampling"} == {key: value for key, value in config.items() if key != "sampling"}
    assert of_the_hot["sampling"] == {**config["sampling"], "temperature": 1.2}


def test_a_set_lean_did_not_answer_fails_the_report_and_is_sampled_again_from_the_kept_adapter_when_the_task_is_queued_again(rows, monkeypatch, tmp_path):
    config, calls = rows.config, rows.calls
    first = _stage(config, monkeypatch, until="ladder_l4_rows_measure_hot_pre")
    store = rows.store()
    # The pool was in trouble while one hot set was measured: Lean gave no verdict on most of its attempts.
    unanswered = "l4_reach_hot_with"
    name = f"episodes_{unanswered}_problems.jsonl"
    as_measured, own = store.path(name).read_bytes(), store.read_rows(name)
    own[0]["attempts_without_an_answer"] = 20
    store.write_rows(name, own)
    monkeypatch.setattr(milestone2, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", ladder_l4_rows.REPORT, "--out", str(tmp_path / "report.json")])
    assert gpu_main.main() != 0                                                # the report is written, is not to be read, and its step fails
    report = json.loads(store.path(ladder_l4_rows.REPORT_FILE).read_text())
    assert report["ok"] is False and report["not_to_be_read"] == ["goal_hot_with_1"] and "NOT TO BE READ" in report["lines"][-1] and report["hot"]["with"]["not_to_be_read"] == ["goal_hot_with_1"]
    # The task queued again: that set, and no other, is sampled again from the kept adapter; nothing is trained.
    untouched = {key: content for key, content in _all(store.root).items() if unanswered not in key and "report" not in key and "measure_hot_with" not in key}
    sampled = rows.l4.solver.calls
    again = _stage(config, monkeypatch)
    assert len(calls) == 2 and again["ladder_l4_rows_measure_hot_with"]["measured_again"] == [unanswered] and rows.l4.solver.calls > sampled
    after = _all(store.root)
    assert all(after[key] == content for key, content in untouched.items()) and after[name] == as_measured                    # the scripted solver writes the same again
    assert again["ladder_l4_rows_report"]["ok"] is True and again["ladder_l4_rows_report"]["not_to_be_read"] == []
    assert first["ladder_l4_rows_measure_hot_pre"] == again["ladder_l4_rows_measure_hot_pre"]


# ------------------------------------------------------------------- the stages before it are what they were
def test_the_stages_before_it_write_what_they_wrote_after_this_stage_ran_beside_them(rows, monkeypatch):
    config = rows.config
    arm_run, pre = rows.arm, rows.pre
    as_they_wrote = {path: _written(path) for path in (arm_run, pre)}
    # `ladder_l4_pretrain`: its files are the list the check of the adapter's rank froze; `ladder_l4`: its two measured models have their rungs and G's two samplings.
    assert sorted(name for name in as_they_wrote[pre] if not name.startswith("episodes_")) == [name for name in PRETRAINING_FILES if name != "adapters/pre/adapter_config.json"]      # (this world's training writes the weights' file alone)
    for model in ("with", "without"):
        measured = json.loads((arm_run / f"ladder_l3d2_measure_{model}.done.json").read_text())
        assert set(measured["goal"]) == {REACH, MORE} and measured["rungs"]["set"] == f"l3d2_rungs_{model}" and (arm_run / f"episodes_l3d2_rungs_{model}_problems.jsonl").exists()
    assert json.loads((pre / "ladder_l4_pretrain_measure.done.json").read_text())["rungs"]["set"] == "l4_rungs_pre"
    assert ladder_ceiling.Model.__dataclass_fields__["rungs"].default is True                      # every caller but the two hot steps measures the rungs
    _stage(config, monkeypatch)
    assert {path: _written(path) for path in (arm_run, pre)} == as_they_wrote
    # `ladder_l4_rank`, run AFTER this stage in the same store by a task that carries this stage's variables: the files it always wrote, and `pre`'s run untouched.
    _trainings(monkeypatch, [])
    monkeypatch.setenv(CHECK, "rank")
    summaries = _check(config, monkeypatch)
    check = ladder_l4_rank._store(config, ladder_l4_rank.the_check(config)).root
    assert check.name == "ladder_l4_pretrain_r64_seed0" and list(summaries) == list(ladder_l4_rank.STEPS)
    assert sorted(name for name in _all(check) if not name.startswith("episodes_")) == RANK_FILES
    assert {row["set"] for row in _file_rows(check / "problems.jsonl")} == {"l4_rungs_pre_r64", "l4_reach_pre_r64", "l4_more_pre_r64"}
    assert summaries["ladder_l4_rank_measure"]["rungs"]["set"] == "l4_rungs_pre_r64" and _written(pre) == as_they_wrote[pre] and _written(arm_run) == as_they_wrote[arm_run]
    # The steps and the stages of the three are what they were.
    assert list(ladder_l4.STEPS) == ["ladder_l4_pretrain_prepare", "ladder_l4_pretrain_train", "ladder_l4_pretrain_measure", "ladder_l4_pretrain_map",
                                     "ladder_l4_pretrain_report", "ladder_l4_prepare", "ladder_l4_report"]
    assert list(ladder_l4_rank.STEPS) == ["ladder_l4_rank_prepare", "ladder_l4_rank_train", "ladder_l4_rank_measure", "ladder_l4_rank_report"]


# ------------------------------------------------------------------------------------------- the smoke stage
def test_the_smoke_stage_trains_both_models_from_the_smoke_pre_measures_them_and_runs_the_two_hot_measurements(arm, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RUN, ARM_RUN, MINIMUM):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings_as_the_gpu_would(monkeypatch, calls)
    # ---- the pretraining's smoke task, then the arm's: what the two L4 smoke tasks leave on the box
    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_smoke"):
        options = entry.step_fields(entry.STAGES[stage_name][2])[2]
        with monkeypatch.context() as patched:                                 # one task's environment: gone when the task is
            patched.delenv(ladder_loop.DATA_VARIABLE)
            for name, value in options["environment"].items():
                patched.setenv(name, value)
            if stage_name == "ladder_l4_smoke":
                arm.with_lean()
                patched.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                steps = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS}
                if environment == "gpu" and step in steps:
                    _on_the_gpu(patched, steps[step], config) if "_train" in step else steps[step](config)
    runs = pipeline.STORE / "runs-v4.27"
    pre, arm_run = runs / "ladder_l4_pretrain_smoke", runs / "ladder_l4_smoke"
    assert (pre / "adapters" / "pre").is_dir() and (arm_run / "adapters" / "m2").is_dir() and (arm_run / "ladder_l4_report.done.json").exists()
    as_stored = {path: _all(path) for path in (pre, arm_run)}
    # ---- this stage's smoke task: the two smoke runs as `pre`'s and the arm's, the fixture, at least two rows in `reward_rows`
    options = entry.step_fields(entry.STAGES["ladder_l4_rows_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_rows_smoke"])
    of_the_pretraining = entry.step_fields(entry.STAGES["ladder_l4_pretrain_smoke"][2])[2]["environment"]
    assert options["environment"] == {ladder_loop.DATA_VARIABLE: of_the_pretraining[ladder_loop.DATA_VARIABLE], ladder_l4.FILE_VARIABLE: str(FIXTURE),
                                      ladder_l4.RUN_VARIABLE: "ladder_l4_pretrain_smoke", ARM_RUN: "ladder_l4_smoke", RUN: "ladder_l4_rows_smoke", MINIMUM: "2"}
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (kits.append(given["sampling"]["temperature"]), (arm.solver, stand_in_parameters))[1]))
    del calls[:]
    summaries = _stage(config, monkeypatch, stage_name="ladder_l4_rows_smoke")
    store = ladder_l4_rows._store(config, ladder_l4_rows.the_settings(config))
    assert store.root.name == "ladder_l4_rows_smoke" and not (runs / "ladder_l4_rows_seed0").exists() and {path: _all(path) for path in as_stored} == as_stored
    assert list(summaries) == STEPS
    prepare = summaries["ladder_l4_rows_prepare"]
    assert (prepare["arm_run"], prepare["pretraining_run"], prepare["start_adapter"]) == (str(arm_run), str(pre), str(pre / "adapters" / "pre"))
    assert [sampling["name"] for sampling in prepare["goal_samplings"]] == [REACH] and prepare["attempts_a_goal_problem"] == 32       # no L2 run beside: one sampling of G
    assert prepare["hot"]["samplings"] == [{"name": REACH, "episodes": 32, "sampling_seed": 1040}] and prepare["hot"]["temperature"] == 1.2
    sets = prepare["training_sets"]
    assert sets["rehearse"]["rows"] == 2 * sets["rehearse"]["twin_rows"] > 0 and sets["reward_rows"]["rows"] >= min(2, sets["reward_rows"]["rows_of_the_rounds"])
    assert sets["reward_rows"]["minimum_of_a_smoke_run"] == 2
    # BOTH models are trained from the smoke `pre` and measured; the two hot measurements run at the hot temperature.
    assert [call["more"] for call in calls] == [{"start": pre / "adapters" / "pre"}] * 2 and [list(call["schedule"]["checkpoint_steps"]) for call in calls] == [["rehearse"], ["reward_rows"]]
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == ["rehearse", "reward_rows"]
    assert kits == [1.0] * 4 + [1.2] * 2                                       # the rungs and G for each trained model; G alone for `with`, then for `pre`
    assert all(summaries[f"ladder_l4_rows_measure_hot_{who}"]["goal"][REACH]["set"] == f"l4_reach_hot_{who}" for who in ("with", "pre"))
    report = summaries["ladder_l4_rows_report"]
    assert report["label"] == "pretrained on published proofs" and all(line.startswith(SAY) for line in report["lines"]) and set(report["branches"].values()) <= set(BRANCHES)
    assert any("the stored models have ONE sampling of G here (a smoke run)" in line for line in report["lines"]) and store.is_done(ladder_l4_rows.REPORT)


# -------------------------------------------------------------------------------------------- registration
def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_only_it_names_its_variables():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_rows"]]
    assert [(environment, step) for environment, step, _ in steps] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", STEPS[0]),
                                                                       *(pair for step in STEPS[1:] for pair in (("guard", None), ("gpu", step)))]
    # The module's steps: this start's eight first, in its order; then what a run from another start has beside (test_ladder_l4_rows_r64.py).
    assert all(options == {} for _, _, options in steps) and list(ladder_l4_rows.STEPS)[:len(STEPS)] == STEPS == ladder_l4_rows.steps_of("pre")
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_rows_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps] and all(options == smoke[0][2] for _, _, options in smoke)
    others = set()
    for module in (ladder_round, ladder_l2, ladder_loop, ladder_ceiling, ladder_l3d2, ladder_l4, ladder_l4_rank):
        others |= set(module.STEPS)
    assert not set(ladder_l4_rows.STEPS) & others and all(name.startswith("ladder_l4_rows_") for name in ladder_l4_rows.STEPS)
    named_by = sorted(name for name, own in entry.STAGES.items() if any(variable in entry.step_fields(step)[2].get("environment", {}) for step in own for variable in (RUN, ARM_RUN, MINIMUM)))
    assert named_by == ["ladder_l4_rows_r64_smoke", "ladder_l4_rows_smoke"]
    assert sorted(name for name in entry.STAGES if any(entry.step_fields(step)[1] in ladder_l4_rows.STEPS for step in entry.STAGES[name])) == [
        "ladder_l4_rows", "ladder_l4_rows_r64", "ladder_l4_rows_r64_smoke", "ladder_l4_rows_smoke"]
    # THE START IS THE SETTING'S for this stage and its smoke: neither names another, and neither has a step of another start.
    of_another_start = set(ladder_l4_rows.STEPS) - set(STEPS)
    for name in ("ladder_l4_rows", "ladder_l4_rows_smoke"):
        assert not [step for _, step, options in map(entry.step_fields, entry.STAGES[name]) if step in of_another_start or ladder_l4_rows.START_VARIABLE in options.get("environment", {})]
    # The stages before it are registered as they were: no step of this stage, no variable of it.
    for name in ("ladder_l4", "ladder_l4_pretrain", "ladder_l4_rank", "ladder_l4_rank_fallback", "ladder_l4_smoke", "ladder_l4_pretrain_smoke", "ladder_l4_rank_smoke", "ladder_l3d2"):
        own = entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES[name][2])[2])
        assert not [variable for variable in (RUN, ARM_RUN, MINIMUM) if variable in own] and not [step for _, step, _ in map(entry.step_fields, entry.STAGES[name]) if step in ladder_l4_rows.STEPS]
    # The settings: from `pre`, on the arm's rounds, two models, 1.2 against the config's 1.0, and hot sampling seeds no other measurement has.
    whole = milestone2.load_config(PACKAGE / "config" / "experiment.yaml")
    settings, ladder = whole["ladder_loop"]["l4"]["rows"], whole["ladder_loop"]
    assert settings == {"start": "pre", "arm_run": "t010_assembly_pre", "models": ["rehearse", "reward_rows"], "models_from_another_start": ["old_rule", "reward_rows", "rehearse"],
                        "temperature_hot": 1.2, "sampling_seeds_hot": [1040, 1041]}
    assert whole["sampling"]["temperature"] == 1.0 and settings["arm_run"] == ladder["l4"]["arm"] and ladder["l2_assembly_arms"][settings["arm_run"]]["start"] == settings["start"]
    first = ladder["round"]["sampling_seed"]
    places = {*range(len(ladder_round.SAMPLING_KINDS)), *(ladder_l2.ROUND_SEED_PLACE + number for number in range(1, ladder_l2.MOST_ROUNDS + 1)), ladder_l2.CONTROL_SEED_PLACE, 30, 31, 32, 33}
    used = {first + 100 * seed + place for seed in (0, 1, 2) for place in places} | {ladder["goal"]["sampling_seed"], ladder["base_map"]["sampling_seed"],
                                                                                   whole["ladder"]["fresh_sampling_seed"], whole["ladder"]["reach_sampling_seed"]}
    assert not used & set(settings["sampling_seeds_hot"]) and {1001, 1002, 1011, 1016, 1020, 1033, 100, 101} <= used
    rows = ladder_l4_rows.the_settings(whole)
    assert (rows.start, rows.arm, rows.temperature, rows.hot_seeds, rows.models) == ("pre", "t010_assembly_pre", 1.2, (1040, 1041), ("rehearse", "reward_rows"))
    assert ladder_l4_rows.run_name(rows, 0) == "ladder_l4_rows_seed0" and ladder_l4_rows.run_name(ladder_l4_rows.Rows("pre_r64", rows.arm, 1.2, (1040, 1041), rows.models), 0) == "ladder_l4_rows_pre_r64_seed0"


def test_the_steps_run_through_the_gpu_entry_point(rows, monkeypatch, tmp_path):
    monkeypatch.setattr(milestone2, "load_config", lambda path: rows.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for name in STEPS:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert rows.store().is_done(ladder_l4_rows.REPORT) and ScriptedLean is not None
    # A step of another start is not one of a run from `pre`: refused, and nothing of it is written.
    as_written = _all(rows.store().root)
    for name in sorted(set(ladder_l4_rows.STEPS) - set(STEPS)):
        with pytest.raises(RuntimeError, match=f"the step {name} is not one of a run from `pre` .ladder_loop.l4.rows.start, or RLVR_LEAN_LADDER_L4_ROWS_START.: its steps are "):
            ladder_l4_rows.STEPS[name](rows.config)
    assert _all(rows.store().root) == as_written
