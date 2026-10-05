"""The ladder loop's L2, the stage end to end on what an L1 run stored: three rounds in batches, the training sets, the
measurements with L1's sampling seeds, the control, the report; resume after an interruption between batches; the
embeddings stored once; the stop rule; the refusals; the alarm. Spec: docs/spec/ladder-loop.spec.md, "L2: three
rounds"; fixtures 5 to 8, 10, 11. The engine, the model and Lean are stand-ins;
nothing touches a GPU or the network. The rules themselves are `test_ladder_l2.py`."""

import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_round import KNOWN_FALSE, L0_FIXTURE, L1_FIXTURE, ScriptedLean, _run_stage, stage  # noqa: E402, F401 - the L1 stage on the fixtures, with stand-ins

from rlvr_lean.domain.ladder_round.challenger import expected_reward  # noqa: E402
from rlvr_lean.domain.ladder_round.read import STOP_AND_DIAGNOSE, VOID, arm_reward  # noqa: E402
from rlvr_lean.domain.problem_pool import SoundnessAlarm  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_dose, ladder_l2, ladder_loop, ladder_round, pipeline  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
SETTINGS = CONFIG["ladder_loop"]
ROUNDS = (1, 2, 3)
ARM = ladder_l2.L2_ARM_VARIABLE
L2_VARIABLES = (ladder_l2.L2_RUN_VARIABLE, ladder_l2.L2_SOURCE_VARIABLE, ladder_l2.L2_PROBLEMS_VARIABLE, ladder_l2.L2_BATCHES_VARIABLE, ARM)
BATCHES = [(number, batch) for number in ROUNDS for batch in (1, 2)]


@pytest.fixture
def loop(stage, monkeypatch):  # noqa: F811
    """The L1 fixture stage, with L2 reading the fixture's twelve candidates as the whole pool: three rounds of four
    problems, in two batches of two, use them up."""
    monkeypatch.setenv(ladder_l2.L2_DATA_VARIABLE, str(L1_FIXTURE))
    for name in L2_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    stage.config["ladder_loop"]["round"].update({"problems": 4, "batches": 2})
    return stage


def _run_loop(config, until=None):
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l2"]):
        if environment == "gpu" and step in ladder_l2.STEPS:
            summaries[step] = ladder_l2.STEPS[step](config)
            if step == until:
                break
    return summaries


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def test_the_whole_stage_runs_on_what_an_l1_run_stored_and_writes_a_run_of_its_own(loop):
    config = loop.config
    with pytest.raises(RuntimeError, match="stage `ladder_l1` with --seeds 0"):           # nothing to read yet: refused, and it says what to run
        ladder_l2.ladder_l2_prepare(config)
    assert not ladder_l2.source_directory(config).exists()                                # and the refusal created nothing
    _run_stage(config)
    source = ladder_l2.source_directory(config)
    assert source == ladder_round._store(config).root
    before = _files(source)
    summaries = _run_loop(config)
    assert _files(source) == before                                                        # L1's run directory was only read
    store = ladder_l2._store(config)
    assert store.root.name == "ladder_l2_seed0" and store.root.parent == source.parent
    prepare = summaries["ladder_l2_prepare"]
    assert prepare["goal_set"] == 1 and prepare["rungs"] == {"below": 2, "in": 2, "above": 1} and prepare["data"]["candidates"] == 12
    assert prepare["batch_sizes"] == [2, 2] and prepare["candidates_short_by"] == 0 and prepare["fixture"] is True
    embed = summaries["ladder_l2_embed"]
    assert embed["statements"] == 16 and embed["embeddings_reused"] is False and Path(embed["directory"]).parent == pipeline.STORE / "ladder_embeddings"

    # ---- the rounds: every candidate proposed once, in equal batches, each with its n episodes
    candidates = [json.loads(line)["problem_id"] for line in (L1_FIXTURE / "candidates.jsonl").read_text().splitlines()]
    held = {json.loads(line)["problem_id"] for name in ("heldout.jsonl", "base_map.jsonl") for line in (L0_FIXTURE / name).read_text().splitlines()}
    picks = {key: store.read_rows(f"proposals_r{key[0]}_b{key[1]}.jsonl") for key in BATCHES}
    proposed = [row["problem_id"] for key in BATCHES for row in picks[key]]
    assert [len(picks[key]) for key in BATCHES] == [2] * 6 and sorted(proposed) == sorted(candidates) and not set(proposed) & held
    for number, batch in BATCHES:
        results = {row["problem_id"]: row for row in store.read_rows(f"episodes_round_r{number}_b{batch}_problems.jsonl")}
        # Every proposed problem gets its n episodes: none is dropped or kept by a pass-rate estimate first.
        assert set(results) == {row["problem_id"] for row in picks[number, batch]} and all(row["episodes"] == 8 for row in results.values())
        assert all(row["round"] == number and row["batch"] == batch and "predicted_rate" in row for row in picks[number, batch])
        assert store.done_summary(f"ladder_l2_batch_r{number}_b{batch}")["episodes"]["sampling_seed"] == prepare["sampling_seeds"][f"round_{number}"]
    # ---- the refit before each batch saw the base map and every batch finished before it, weighted by its round
    fits = {key: store.done_summary(f"ladder_l2_propose_r{key[0]}_b{key[1]}") for key in BATCHES}
    assert [fits[key]["observations"] for key in BATCHES] == [4, 6, 8, 10, 12, 14]
    seen = {key: fits[key]["observations_by_the_round_that_gave_them"] for key in BATCHES}
    assert seen[1, 1] == {"1": {"observations": 4, "weight": 1.0}} and seen[1, 2] == {"1": {"observations": 6, "weight": 1.0}}
    assert seen[2, 1] == {"1": {"observations": 8, "weight": 0.5}}
    assert seen[2, 2] == {"1": {"observations": 8, "weight": 0.5}, "2": {"observations": 2, "weight": 1.0}}
    assert seen[3, 2] == {"1": {"observations": 8, "weight": 0.25}, "2": {"observations": 4, "weight": 0.5}, "3": {"observations": 2, "weight": 1.0}}
    assert [fits[key]["attempted_by"] for key in BATCHES] == ["M(0)", "M(0)", "M(1)", "M(1)", "M(2)", "M(2)"]
    assert [fits[key]["candidates_not_yet_proposed"]["problems"] for key in BATCHES] == [12, 10, 8, 6, 4, 2]
    assert len(store.read_rows("candidate_scores_r1.jsonl")) == 12 and len(store.read_rows("candidate_scores_r3.jsonl")) == 4

    # ---- training: M(r) from the base on every round's training set so far; the proofs are the solver's own
    so_far = []
    for number in ROUNDS:
        examples = store.read_rows(f"training_examples_r{number}.jsonl")
        attempts = [json.loads(line) for path in sorted(store.root.glob(f"episodes_round_r{number}_b*_attempts_*.jsonl")) for line in path.read_text().splitlines()]
        verified = {(row["problem_id"], row["completion"]) for row in attempts if row["status"] == "verified"}
        resolved = {row["problem_id"] for batch in (1, 2) for row in store.read_rows(f"episodes_round_r{number}_b{batch}_problems.jsonl") if row["resolved"] > 0}
        assert {example["problem_id"] for example in examples} == resolved and all((example["problem_id"], example["completion"]) in verified for example in examples)
        assert all(example["side"] == "negation" for example in examples if example["problem_id"] in KNOWN_FALSE)
        so_far += examples
        trained = summaries[f"ladder_l2_train_{number}"]
        assert trained["examples"] == len(so_far) and trained["rounds_trained_on"] == list(range(1, number + 1)) and trained["model"] == f"M({number})"
        assert trained["trained_from"] == "the base" and "no adapter was trained" in trained["note"]
    assert so_far and summaries["ladder_l2_train_3"]["examples_by_round"] == {str(number): len(store.read_rows(f"training_examples_r{number}.jsonl")) for number in ROUNDS}

    # ---- measured after each round's training, with L1's sampling seeds: each pairs by problem with L1's base results
    base_rungs = {row["problem_id"]: row for row in store.read_rows("base_rungs.jsonl")}
    l1_seeds = {"rungs": ladder_round.sampling_seed(config, "rungs_base"), "reach": ladder_round.sampling_seed(config, "reach_base")}
    for number in ROUNDS:
        measured = summaries[f"ladder_l2_measure_{number}"]
        rows = store.read_rows(f"episodes_rungs_m{number}_problems.jsonl")
        assert {row["problem_id"]: row["sides"] for row in rows} == {key: row["sides"] for key, row in base_rungs.items()} and all(row["episodes"] == 8 for row in rows)
        assert measured["rungs"]["sampling_seed"] == l1_seeds["rungs"] and measured["reach"]["sampling_seed"] == l1_seeds["reach"]
        assert measured["reach"]["episodes_each"] == 32 and measured["distinct_attempts_on_the_rungs"]["prompts"] >= 5 and measured["stop_rule_fires"] is False
    assert not list(store.root.glob("episodes_gain*"))                                    # the gain by k is not measured in L2

    # ---- the control: the base, the loop's attempt episodes on G's one problem, a seed of its own
    control = summaries["ladder_l2_control"]
    assert control["loop_attempt_episodes"] == 3 * 4 * 8 and control["episodes_each"] == 96 and control["episodes_not_spent"] == 0
    assert control["episodes"]["sampling_seed"] == prepare["sampling_seeds"]["control"] and control["episodes"]["episodes_each"] == 96
    assert [row["episodes"] for row in store.read_rows("episodes_control_problems.jsonl")] == [96]

    # ---- the same extra attempts for the trained model: M(3), the control's number of episodes on every problem of
    # G and the control's sampling seed, in a set of its own
    trained = summaries["ladder_l2_control_trained"]
    assert (trained["model"], trained["set"], trained["episodes_each"]) == ("M(3)", "control_m3", control["episodes_each"])
    assert trained["sampling_seed"] == trained["episodes"]["sampling_seed"] == control["episodes"]["sampling_seed"] and trained["episodes"]["set"] == "control_m3"
    own, of_the_control = store.read_rows("episodes_control_m3_problems.jsonl"), store.read_rows("episodes_control_problems.jsonl")
    assert [(row["problem_id"], row["episodes"], row["sides"]) for row in own] == [(row["problem_id"], row["episodes"], row["sides"]) for row in of_the_control]
    without_the_set = lambda name: [{key: value for key, value in row.items() if key != "set"} for row in store.read_rows("problems.jsonl") if row["set"] == name]      # noqa: E731
    assert without_the_set("control_m3") == without_the_set("control") != []           # the same problems, with the same negations
    assert list(store.root.glob("episodes_control_m3_attempts_*.jsonl")) and store.is_done("episodes_control_m3")

    # ---- the report
    report = summaries["ladder_l2_report"]
    assert report["fixture"] is True and report["stand_in_engine"] is True and report["ok"] is True and report["rounds_measured"] == [1, 2, 3]
    # The stand-in "trains" nothing, so every model is the base: round 1 left the upper rungs where they were, and that reads VOID.
    assert report["branch"]["name"] == VOID and "did not train" in report["branch"]["reason"]
    assert report["void_conditions"]["round_1_left_both_the_in_band_and_the_above_band_rung_at_or_below_zero"] is True
    assert report["primary"]["problems"] == 2 and set(report["climb"]["by_rung"]) == {"below", "in", "above"}
    assert [(row["round"], row["batch"]) for row in report["challenger"]["by_batch"]] == BATCHES and set(report["challenger"]["by_round"]) == {"1", "2", "3"}
    assert all(row["all"]["picks"] == 2 for row in report["challenger"]["by_batch"]) and report["challenger"]["by_round"]["2"]["all"]["picks"] == 4
    assert report["challenger"]["trajectory"]["compared"] == [1, 3] and set(report["rounds"]) == {"1", "2", "3"}
    assert report["reach_on_g"]["last_model_against_the_base_afresh"]["problems"] == 1 and report["reach_on_g"]["last_model_against_the_one_round_model"]["problems"] == 1
    assert (report["control"]["read"]["episodes_after"], report["control"]["read"]["episodes_before"]) == (32, 32 + 96)
    # Beside the control's read, the two models at EQUAL attempts: M(3)'s reach episodes plus its own extra ones.
    equal = report["control"]["equal_attempts"]
    assert equal["step"]["set"] == "control_m3" and (equal["read"]["episodes_after"], equal["read"]["episodes_before"]) == (32 + 96, 32 + 96)
    assert equal["read"]["attempts_of_the_last_model"] == equal["read"]["attempts_of_the_base"] == 32 + 96 and equal["read"]["extra_episodes_each"] == 96
    # The stand-in's M(3) IS the base: with the control's seed its extra attempts are the control's own, sample for sample.
    assert equal["read"]["successful_episodes_of_the_last_model"] == equal["read"]["successful_episodes_of_the_base"]
    assert (equal["read"]["gained"], equal["read"]["lost"]) == (0, 0) and "at equal attempts (128 each)" in report["headline"] and "control_m3" in report["attempts"]
    assert set(report["distinct_attempts_on_the_rungs"]) == {"base", "M(1)", "M(2)", "M(3)"} and "M(0) is the base" in report["naming"]
    assert json.loads(store.path("report_ladder_l2.json").read_text())["headline"] == report["headline"]

    # ---- a rerun returns what is stored: nothing is trained or sampled again
    sent = ScriptedLean.submitted
    again = _run_loop(config)
    assert ScriptedLean.submitted == sent and again == summaries
    assert not ladder_round._store(config).is_done("ladder_l2_prepare") and _files(source) == before


def test_the_report_step_hands_a_rerun_everything_a_reader_needs(loop, monkeypatch, tmp_path):
    """A rerun is another task with an output directory of its own: the report step delivers the per-problem files and
    what each step recorded again, and neither the attempts nor the per-block files."""
    _run_stage(loop.config)
    _run_loop(loop.config)
    out = tmp_path / "rerun_out" / "steps"
    monkeypatch.setenv("RLVR_LEAN_STEP_DIR", str(out))
    ladder_l2.ladder_l2_report(loop.config)
    delivered = {path.name for path in out.iterdir()}
    assert {"report_ladder_l2.json", "heldout_groups.jsonl", "base_rungs.jsonl", "base_reach.jsonl", "proposals_r1_b1.jsonl", "proposals_r3_b2.jsonl",
            "training_examples_r2.jsonl", "candidate_scores_r1.jsonl", "episodes_round_r2_b1_problems.jsonl", "episodes_rungs_m3_problems.jsonl",
            "episodes_reach_m1_problems.jsonl", "episodes_control_problems.jsonl", "ladder_l2_prepare.done.json", "ladder_l2_propose_r2_b2.done.json",
            "ladder_l2_round_1.done.json", "ladder_l2_train_3.done.json", "ladder_l2_measure_2.done.json", "ladder_l2_control.done.json",
            "episodes_control_m3_problems.jsonl", "episodes_control_m3.done.json", "ladder_l2_control_trained.done.json"} <= delivered
    assert not [name for name in delivered if "_attempts_" in name or "_block_" in name or name.rsplit("_", 1)[-1][:4].isdigit()]


def test_a_run_that_finished_before_the_trained_models_extra_attempts_existed_gets_them_alone_when_run_again(loop, monkeypatch):
    """Spec, "Added 2026-10-05": a run that has finished gets this measurement alone when its task is queued again;
    nothing else is sampled. Seeds 0, 1 and 2 finished with the stage as it was: every step but the new one."""
    config = loop.config
    _run_stage(config)
    as_it_was = [step for step in ladder_l2.STEPS if step != "ladder_l2_control_trained"]
    finished = {step: ladder_l2.STEPS[step](config) for step in as_it_was}
    store = ladder_l2._store(config)
    assert not store.is_done("ladder_l2_control_trained") and not any(row["set"] == "control_m3" for row in store.read_rows("problems.jsonl"))
    before = finished["ladder_l2_report"]
    assert "equal_attempts" not in before["control"] and "at equal attempts" not in before["headline"] and "control_m3" not in before["attempts"]
    kept, sent = _files(store.root), ScriptedLean.submitted
    sample, fit, embed, sampled, refits, embedded = ladder_l2._episodes, ladder_l2.fit_pass_rate_model, ladder_round.stand_in_embeddings, [], [], []
    monkeypatch.setattr(ladder_l2, "_episodes", lambda config, store, set_name, *rest: sampled.append(set_name) or sample(config, store, set_name, *rest))
    monkeypatch.setattr(ladder_l2, "fit_pass_rate_model", lambda *arguments, **named: refits.append(1) or fit(*arguments, **named))
    monkeypatch.setattr(ladder_round, "stand_in_embeddings", lambda statements: embedded.append(1) or embed(statements))

    again = _run_loop(config)                           # the task queued again, with the stage as it is now
    # The new step alone did work: one set sampled, nothing embedded, no refit, and every other step returned what it had stored.
    assert sampled == ["control_m3"] and refits == [] and embedded == []
    assert all(again[step] == finished[step] for step in as_it_was if step != "ladder_l2_report")
    added = again["ladder_l2_control_trained"]
    assert added["episodes_each"] == finished["ladder_l2_control"]["episodes_each"] and added["sampling_seed"] == finished["ladder_l2_control"]["sampling_seed"]
    # Every file the finished run had is byte for byte what it was, but the three that must take the new part in.
    after = _files(store.root)
    rewritten = {"problems.jsonl", "report_ladder_l2.json", "ladder_l2_report.done.json"}
    assert all(after[name] == content for name, content in kept.items() if name not in rewritten)
    assert set(after) - set(kept) == {"ladder_l2_control_trained.done.json", "episodes_control_m3.done.json", "episodes_control_m3_problems.jsonl",
                                      "episodes_control_m3_problems_0000.jsonl", "episodes_control_m3_attempts_0000.jsonl", "episodes_control_m3_block_0000.done.json"}
    # `problems.jsonl`: every row it held, in place, and after them the goal problems under the new set's name.
    held = [json.loads(line) for line in kept["problems.jsonl"].decode().splitlines()]
    rows = store.read_rows("problems.jsonl")
    assert rows[:len(held)] == held and {row["set"] for row in rows[len(held):]} == {"control_m3"} and len(rows) - len(held) == sum(row["set"] == "control" for row in held)
    # Lean was sent the new attempts and nothing else.
    attempts = [json.loads(line) for line in after["episodes_control_m3_attempts_0000.jsonl"].decode().splitlines()]
    assert ScriptedLean.submitted - sent == sum(attempt["status"] != "capped_tokens" for attempt in attempts) > 0
    # The report: every key it had, unchanged, and the new part beside the control's read; one more clause in the headline.
    report = again["ladder_l2_report"]
    assert {key: value for key, value in report.items() if key not in ("headline", "control", "attempts")} == \
           {key: value for key, value in before.items() if key not in ("headline", "control", "attempts")}
    assert {key: value for key, value in report["control"].items() if key != "equal_attempts"} == before["control"]
    assert {key: value for key, value in report["attempts"].items() if key != "control_m3"} == before["attempts"]
    assert re.sub(r"; at equal attempts \(128 each\).*? successes per 1,000 attempts", "", report["headline"]) == before["headline"] != report["headline"]
    assert report["control"]["equal_attempts"]["read"]["episodes_after"] == report["control"]["equal_attempts"]["read"]["episodes_before"] == 32 + 96
    assert json.loads(store.path("report_ladder_l2.json").read_text()) == report
    # Queued a third time it does nothing at all.
    sent, sampled[:] = ScriptedLean.submitted, []
    assert _run_loop(config) == again and sampled == [] and ScriptedLean.submitted == sent and _files(store.root) == after


def test_a_run_interrupted_between_batches_resumes_at_the_first_batch_not_done(loop, monkeypatch):
    config = loop.config
    _run_stage(config)
    whole = _run_loop(config)                                   # the same loop, uninterrupted, in a run directory of its own
    reference = ladder_l2._store(config).root
    monkeypatch.setenv(ladder_l2.L2_RUN_VARIABLE, "ladder_l2_interrupted")
    sample, sampled = ladder_l2._episodes, []

    def until_the_box_goes_away(config, store, set_name, *rest):
        if set_name == "round_r2_b2":
            raise RuntimeError("the box went away")
        return sample(config, store, set_name, *rest)

    def counting(config, store, set_name, *rest):
        sampled.append(set_name)
        return sample(config, store, set_name, *rest)

    monkeypatch.setattr(ladder_l2, "_episodes", until_the_box_goes_away)
    with pytest.raises(RuntimeError, match="the box went away"):
        _run_loop(config)
    store = ladder_l2._store(config)
    assert store.root.name == "ladder_l2_interrupted" and store.done_summary("ladder_l2_embed")["embeddings_reused"] is True      # stored once per box
    assert store.is_done("ladder_l2_batch_r2_b1") and not store.is_done("ladder_l2_batch_r2_b2") and not store.is_done("ladder_l2_round_2")
    assert store.is_done("ladder_l2_propose_r2_b2") and not store.path("episodes_round_r2_b2_problems.jsonl").exists()
    kept = _files(store.root)
    monkeypatch.setattr(ladder_l2, "_episodes", counting)
    fit, refits = ladder_l2.fit_pass_rate_model, []
    monkeypatch.setattr(ladder_l2, "fit_pass_rate_model", lambda *arguments, **named: refits.append(len(arguments[0])) or fit(*arguments, **named))
    resumed = _run_loop(config)
    # It resumed AT the batch that was not done: no earlier batch was sampled again, and no stored file of theirs changed.
    assert sampled[:2] == ["round_r2_b2", "rungs_m2"] and not [name for name in sampled if name.startswith("round_r1") or name == "round_r2_b1"]
    # The interrupted batch's proposals were made before the box went away and are the stored ones: only round 3's two
    # batches were still to be proposed (with 12 and 14 observations).
    assert refits == [12, 14]
    after = _files(store.root)
    assert all(after[name] == content for name, content in kept.items() if name != "problems.jsonl")
    # And it ended where the uninterrupted loop did: the same proposals, the same results, the same read.
    for name, content in _files(reference).items():
        if name.startswith(("proposals_", "training_examples_", "candidate_scores_")) or name.endswith("_problems.jsonl"):
            assert after[name] == content, name
    for key in ("branch", "primary", "climb", "reach_on_g", "challenger", "rounds"):
        assert resumed["ladder_l2_report"][key] == whole["ladder_l2_report"][key], key
    assert resumed["ladder_l2_report"]["control"]["read"] == whole["ladder_l2_report"]["control"]["read"]


def test_the_statements_are_embedded_once_and_another_seeds_task_reuses_them(loop, monkeypatch):
    config, calls = loop.config, []
    embed = ladder_round.stand_in_embeddings
    monkeypatch.setattr(ladder_round, "stand_in_embeddings", lambda statements: calls.append(len(statements)) or embed(statements))
    first = ladder_l2.ladder_l2_embed(config)
    assert calls == [16] and first["embeddings_reused"] is False and first["projection_reused"] is False
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "1")                                    # seed 1's task: another run directory, the same statements
    second = ladder_l2.ladder_l2_embed(config)
    assert ladder_l2._store(config).root.name == "ladder_l2_seed1" and calls == [16]       # nothing was embedded again
    assert second["embeddings_reused"] is True and second["projection_reused"] is True and (second["key"], second["directory"]) == (first["key"], first["directory"])
    assert ladder_l2.ladder_l2_embed(config) == second
    # The key is the model and every prompt: another model, other statements or the stand-in is another key.
    ids, prompts = ["a", "b"], ["prompt a", "prompt b"]
    key = ladder_l2.embedding_key(config, ids, prompts, False)
    other_model = {**config, "model": {**config["model"], "revision": "another"}}
    assert len({key, ladder_l2.embedding_key(other_model, ids, prompts, False), ladder_l2.embedding_key(config, ids, ["prompt a", "prompt c"], False),
                ladder_l2.embedding_key(config, ids[::-1], prompts, False), ladder_l2.embedding_key(config, ids, prompts, True)}) == 5
    assert ladder_l2.embedding_key(config, ids, prompts, False) == key
    # A projection asked for with another number of directions is fitted from the stored embeddings, not embedded again.
    config["ladder_loop"]["challenger"]["embedding_components"] = 4
    third = ladder_l2.ladder_l2_embed(config)
    assert calls == [16] and third["embeddings_reused"] is True and third["projection_reused"] is False and third["projected"] == "projected_4.npy"


def test_a_round_whose_below_band_interval_lies_entirely_below_zero_stops_the_loop(loop):
    """Spec, "A round" step 6 and L2's branches: the loop stops for diagnosis. The steps after it run nothing, the
    control is not run, and the report says where it stopped."""
    config = loop.config
    _run_stage(config)
    ladder_l2.ladder_l2_prepare(config)
    store = ladder_l2._store(config)
    below = {row["problem_id"] for row in store.read_rows("heldout_groups.jsonl") if row["group"] == "below"}
    # A base that resolved the below-band problems in every fresh episode and the others in none: any model that
    # does less on the first is below it, and round 1 is seen to have trained on the others.
    store.write_rows("base_rungs.jsonl", [{**row, "resolved": row["episodes"] if row["problem_id"] in below else 0} for row in store.read_rows("base_rungs.jsonl")])
    summaries = _run_loop(config)
    first = summaries["ladder_l2_measure_1"]
    assert first["stop_rule_fires"] is True and first["below_band_minus_base"]["high"] < 0 and ladder_l2.stopped_after(store) == 1
    later = ["ladder_l2_round_2", "ladder_l2_train_2", "ladder_l2_measure_2", "ladder_l2_round_3", "ladder_l2_train_3", "ladder_l2_measure_3", "ladder_l2_control",
             "ladder_l2_control_trained"]
    assert all(summaries[step]["stopped_after_round"] == 1 and "was not run" in summaries[step]["skipped"] for step in later)
    assert not any(store.is_done(step) for step in later) and not list(store.root.glob("proposals_r2_*")) and not list(store.root.glob("episodes_control*"))
    report = summaries["ladder_l2_report"]
    assert report["branch"]["name"] == STOP_AND_DIAGNOSE and "M(1)" in report["branch"]["reason"] and report["rounds_measured"] == [1]
    assert report["stop_rule"]["fires_at_rounds"] == [1] and report["stop_rule"]["the_loop_stopped_after_round"] == 1
    assert report["control"]["read"] is None and report["climb"]["by_rung"] is None and "STOPPED after round 1" in report["headline"]
    assert set(report["rounds"]) == {"1"} and report["challenger"]["trajectory"]["compared"] is None


def test_a_loop_that_would_not_pair_with_l1s_results_or_holds_out_another_set_is_refused(loop, monkeypatch):
    config = loop.config
    _run_stage(config)
    config["ladder_loop"]["round"]["sampling_seed"] += 7                       # another sampling seed than L1 stored
    with pytest.raises(RuntimeError, match="would not pair with L1's stored results"):
        ladder_l2.ladder_l2_prepare(config)
    config["ladder_loop"]["round"]["sampling_seed"] -= 7
    for key, value, message in (("rounds", 2, "change both together"), ("batches", 3, "must be equal")):
        wrong = {**config, "ladder_loop": {**config["ladder_loop"], "round": {**config["ladder_loop"]["round"], key: value}}}
        with pytest.raises(ValueError, match=message):
            ladder_l2.ladder_l2_prepare(wrong)
    # An L1 run that placed another held-out set than the snapshot's H: the candidates would be held out against one
    # set and the rungs measured on another.
    source = ladder_l2.source_directory(config)
    other = source.parent / "ladder_l1_other"
    shutil.copytree(source, other)
    rows = [json.loads(line) for line in (other / "heldout_groups.jsonl").read_text().splitlines()]
    (other / "heldout_groups.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
    monkeypatch.setenv(ladder_l2.L2_SOURCE_VARIABLE, "ladder_l1_other")
    with pytest.raises(RuntimeError, match="another held-out set"):
        ladder_l2.ladder_l2_prepare(config)
    assert not ladder_l2._store(config).is_done("ladder_l2_prepare")
    # A step asked for out of the stage's order says which step it needs.
    monkeypatch.delenv(ladder_l2.L2_SOURCE_VARIABLE)
    ladder_l2.ladder_l2_prepare(config)
    with pytest.raises(RuntimeError, match="needs the step ladder_l2_embed"):
        ladder_l2.STEPS["ladder_l2_round_1"](config)
    with pytest.raises(RuntimeError, match="needs the step ladder_l2_round_1"):
        ladder_l2.STEPS["ladder_l2_train_1"](config)
    with pytest.raises(RuntimeError, match="needs the step ladder_l2_control of this run"):      # the number of episodes is the control's
        ladder_l2.STEPS["ladder_l2_control_trained"](config)
    # A run keeps the sizes it was prepared with: its batches are stored by their number.
    ladder_l2.ladder_l2_embed(config)
    config["ladder_loop"]["round"]["problems"] = 2
    with pytest.raises(RuntimeError, match="keeps the sizes it began with"):
        ladder_l2.STEPS["ladder_l2_round_1"](config)
    assert not list(ladder_l2._store(config).root.glob("proposals_*"))


def test_a_proof_on_the_side_a_certificate_contradicts_stops_the_round(loop):
    config = loop.config
    config["ladder_loop"]["episode"]["contradicted_side"] = "all"            # every problem's ruled-out side is checked
    _run_stage(config)
    ScriptedLean.alarm = "fixture_c1"
    with pytest.raises(SoundnessAlarm, match="fixture_c1 is known true"):
        _run_loop(config)
    store = ladder_l2._store(config)
    number, batch = next(key for key in BATCHES if store.path(f"proposals_r{key[0]}_b{key[1]}.jsonl").exists()
                         and "fixture_c1" in {row["problem_id"] for row in store.read_rows(f"proposals_r{key[0]}_b{key[1]}.jsonl")})
    assert not store.is_done(f"ladder_l2_batch_r{number}_b{batch}") and not store.is_done(f"ladder_l2_round_{number}")
    assert list(store.root.glob(f"episodes_round_r{number}_b{batch}_attempts_*.jsonl"))      # the alarm leaves its evidence
    assert not store.path("report_ladder_l2.json").exists()


def test_the_rounds_and_the_control_sample_with_seeds_of_their_own_and_the_rungs_and_g_with_l1s(loop, monkeypatch):
    config = loop.config
    seeds = ladder_l2.sampling_seeds(config)
    assert seeds["rungs"] == ladder_round.sampling_seed(config, "rungs_base") and seeds["reach"] == ladder_round.sampling_seed(config, "reach_base")
    own = [seeds["round_1"], seeds["round_2"], seeds["round_3"], seeds["control"]]
    of_l1 = {ladder_round.sampling_seed(config, name) for name in ladder_round.SAMPLED_SETS}
    assert len(set(own)) == 4 and not set(own) & of_l1                         # no measurement of L1 or L1b shares them, nor one another
    assert not set(own) & {SETTINGS["goal"]["sampling_seed"], SETTINGS["base_map"]["sampling_seed"]}      # never L0's placing seeds
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "1")
    other = ladder_l2.sampling_seeds(config)
    assert not set(other.values()) & set(seeds.values()) and not set(other.values()) & of_l1
    assert ladder_l2._store(config).root.name == "ladder_l2_seed1" and ladder_l2.source_directory(config).name == "ladder_l1_seed1"
    monkeypatch.setenv(ladder_l2.L2_RUN_VARIABLE, "ladder_l2_smoke")
    monkeypatch.setenv(ladder_l2.L2_SOURCE_VARIABLE, "ladder_l1_smoke")
    assert ladder_l2._store(config).root.name == "ladder_l2_smoke" and ladder_l2.source_directory(config).name == "ladder_l1_smoke"
    monkeypatch.setenv(ladder_l2.L2_PROBLEMS_VARIABLE, "6")
    monkeypatch.setenv(ladder_l2.L2_BATCHES_VARIABLE, "3")
    assert ladder_l2.loop_sizes(config) == {"problems": 6, "batches": 3, "batch_sizes": [2, 2, 2], "solvers": 8}
    assert ladder_l2.loop_sizes(CONFIG)["batch_sizes"] == [2, 2, 2] and ladder_l2.data_directory() == L1_FIXTURE
    for name in (ladder_l2.L2_PROBLEMS_VARIABLE, ladder_l2.L2_BATCHES_VARIABLE, ladder_l2.L2_DATA_VARIABLE):
        monkeypatch.delenv(name)
    assert ladder_l2.loop_sizes(CONFIG) == {"problems": 1000, "batches": 4, "batch_sizes": [250] * 4, "solvers": 8}
    assert ladder_l2.data_directory() == PACKAGE / "data" / "ladder_l2"


# ------------------------------------------------------------------------------ L2t: an arm of the stage
def test_an_arm_changes_the_target_rate_wherever_it_is_read_and_the_run_directory_and_nothing_else(loop, monkeypatch):
    """Spec, "L2t: the lower target": L2 again with ONE setting changed, `challenger.target_rate` 0.10 for 0.25, in a
    run directory of its own. Same candidates, batches, seeds and sampling seeds; the held-out rungs stay L1's."""
    config = loop.config
    assert CONFIG["ladder_loop"]["l2_arms"] == {"t010": {"target_rate": 0.10}} and config["ladder_loop"]["challenger"]["target_rate"] == 0.25
    statements = (["a", "b"], ["prompt a", "prompt b"], False)
    of_the_run = lambda: {"seeds": ladder_l2.sampling_seeds(config), "sizes": ladder_l2.loop_sizes(config), "source": ladder_l2.source_directory(config),      # noqa: E731
                          "data": ladder_l2.data_directory(), "embedding_key": ladder_l2.embedding_key(ladder_l2.arm_config(config), *statements)}
    # No arm: the config itself, and the run directory the stage has always had.
    assert ladder_l2.arm_name() is None and ladder_l2.arm_config(config) is config and ladder_l2._store(config).root.name == "ladder_l2_seed0"
    plain = of_the_run()
    _run_stage(config)
    without = _run_loop(config)
    plain_root = ladder_l2._store(config).root
    held = _files(plain_root)

    monkeypatch.setenv(ARM, "t010")
    armed = ladder_l2.arm_config(config)
    assert armed["ladder_loop"]["challenger"]["target_rate"] == 0.10 and config["ladder_loop"]["challenger"]["target_rate"] == 0.25     # the config given is not touched
    assert {**armed, "ladder_loop": {**armed["ladder_loop"], "challenger": {**armed["ladder_loop"]["challenger"], "target_rate": 0.25}}} == config
    assert ladder_l2.arm_config(armed) == armed and of_the_run() == plain                 # the seeds, the sizes, L1's run, the data and the embeddings' key do not move
    store = ladder_l2._store(config)
    assert store.root.name == "ladder_l2_t010_seed0" and store.root.parent == plain_root.parent
    summaries = _run_loop(config)
    assert _files(plain_root) == held                                                     # the run with no arm was not touched

    # ---- prepare records the arm and its target rate; everything else it recorded is what the run with no arm recorded
    prepare = summaries["ladder_l2_prepare"]
    assert (prepare["arm"], prepare["target_rate"]) == ("t010", 0.10) and "arm" not in without["ladder_l2_prepare"] and "target_rate" not in without["ladder_l2_prepare"]
    assert {key: value for key, value in prepare.items() if key not in ("arm", "target_rate", "the_heldout_rungs_are")} == without["ladder_l2_prepare"]
    # ---- the held-out sets are L1's stored groups, byte for byte the ones the run with no arm read: they do not move with t
    after = _files(store.root)
    for name in ("heldout_groups.jsonl", "base_rungs.jsonl", "base_reach.jsonl"):
        assert after[name] == held[name], name
    assert summaries["ladder_l2_embed"]["embeddings_reused"] is True and summaries["ladder_l2_embed"]["key"] == without["ladder_l2_embed"]["key"]
    # ---- the challenger scores by the expected reward AT THE ARM'S TARGET RATE: the proposals are chosen by it
    for arm_store, target, other in ((store, 0.10, 0.25), (ladder_l2.ArtifactStore(plain_root), 0.25, 0.10)):
        for number, batch in BATCHES:
            fit = arm_store.done_summary(f"ladder_l2_propose_r{number}_b{batch}")
            picks = arm_store.read_rows(f"proposals_r{number}_b{batch}.jsonl")
            at = lambda rate: [expected_reward(row["predicted_rate"], 8, rate, fit["fit"]["dispersion"]) for row in picks]      # noqa: E731
            assert [row["score"] for row in picks] == pytest.approx(at(target), abs=1e-5) and [row["score"] for row in picks] != pytest.approx(at(other), abs=1e-5)
    # The first refit of round 1 saw the same observations and is the same predictor: only what it is asked to maximise moved.
    first, first_plain = store.done_summary("ladder_l2_propose_r1_b1"), ladder_l2.ArtifactStore(plain_root).done_summary("ladder_l2_propose_r1_b1")
    assert first["fit"] == first_plain["fit"] and first["observations_by_the_round_that_gave_them"] == first_plain["observations_by_the_round_that_gave_them"]
    assert first["candidates_not_yet_proposed"]["mean_predicted_rate"] == first_plain["candidates_not_yet_proposed"]["mean_predicted_rate"]
    # ---- the reward recorded for a round and a batch is the arm's
    for number in ROUNDS:
        results = [row for batch in (1, 2) for row in store.read_rows(f"episodes_round_r{number}_b{batch}_problems.jsonl")]
        assert summaries[f"ladder_l2_round_{number}"]["reward"] == arm_reward(results, 0.10, CONFIG["ladder_loop"]["challenger"]["band_reward"])
        assert summaries[f"ladder_l2_round_{number}"]["reward"] != arm_reward(results, 0.25, CONFIG["ladder_loop"]["challenger"]["band_reward"])
        # The same sampling seeds as the run with no arm, for the round's attempts and for the measurements.
        assert summaries[f"ladder_l2_round_{number}"]["sampling_seed"] == without[f"ladder_l2_round_{number}"]["sampling_seed"]
        for name in (f"episodes_rungs_m{number}_problems.jsonl", f"episodes_reach_m{number}_problems.jsonl"):
            assert after[name] == held[name], name                  # the stand-in's models are the base in both runs: the same seeds give the same episodes
    assert summaries["ladder_l2_control"]["sampling_seed"] == without["ladder_l2_control"]["sampling_seed"] and after["episodes_control_problems.jsonl"] == held["episodes_control_problems.jsonl"]
    assert summaries["ladder_l2_control_trained"]["episodes"]["sampling_seed"] == without["ladder_l2_control_trained"]["episodes"]["sampling_seed"]
    # ---- the report: the arm and its target rate at the top, every reward figure at that rate, the held-out read where it was
    report, plain_report = summaries["ladder_l2_report"], without["ladder_l2_report"]
    assert list(report)[:5] == ["spec", "headline", "ok", "seed", "arm"] and report["arm"]["name"] == "t010" and report["target_rate"] == 0.10 and "arm" not in plain_report
    assert report["headline"].startswith("L2 arm t010 (target rate 0.1) seed 0: ") and plain_report["headline"].startswith("L2 seed 0: ") and plain_report["target_rate"] == 0.25
    assert report["band"] == {"low": 0.0485, "high": 0.1749} and report["challenger"]["trajectory"]["target_rate"] == 0.10
    for key in ("heldout", "primary", "climb", "reach_on_g", "branch", "void_conditions", "stop_rule", "rounds_measured", "distinct_attempts_on_the_rungs"):
        assert report[key] == plain_report[key], key
    assert report["control"]["read"] == plain_report["control"]["read"] and report["control"]["equal_attempts"]["read"] == plain_report["control"]["equal_attempts"]["read"]
    scored = report["challenger"]["by_round"]["1"]["all"]
    assert {"share_at_k_0", "share_at_k_1_to_3", "share_at_k_4_or_more"} <= set(scored) and scored["share_at_k_0"] + scored["share_at_k_1_to_3"] + scored["share_at_k_4_or_more"] == pytest.approx(1)
    # A rerun of the arm returns what is stored; so does a rerun of the run with no arm.
    sent = ScriptedLean.submitted
    assert _run_loop(config) == summaries and ScriptedLean.submitted == sent
    monkeypatch.delenv(ARM)
    assert _run_loop(config) == without and ScriptedLean.submitted == sent and _files(plain_root) == held


def test_an_unknown_arm_and_a_changed_target_rate_on_a_prepared_run_are_refused(loop, monkeypatch):
    config = loop.config
    _run_stage(config)
    runs = ladder_round._store(config).root.parent
    monkeypatch.setenv(ARM, "t005")                                    # an arm the config does not name: refused, and nothing is made
    for step in ("ladder_l2_prepare", "ladder_l2_embed", "ladder_l2_report"):
        with pytest.raises(ValueError, match=r"names the arm 't005' and ladder_loop.l2_arms has \['t010'\]"):
            ladder_l2.STEPS[step](config)
    assert sorted(path.name for path in runs.iterdir()) == ["ladder_l1_seed0"]
    config["ladder_loop"]["l2_arms"]["wide"] = {"target_rate": 0.10, "random_share": 0.5}      # an arm changes the target rate and nothing else
    monkeypatch.setenv(ARM, "wide")
    with pytest.raises(ValueError, match="and nothing else"):
        ladder_l2.ladder_l2_prepare(config)
    # A run prepared under one target rate does not continue under another.
    monkeypatch.setenv(ARM, "t010")
    ladder_l2.ladder_l2_prepare(config)
    ladder_l2.ladder_l2_embed(config)
    config["ladder_loop"]["l2_arms"]["t010"]["target_rate"] = 0.08
    with pytest.raises(RuntimeError, match="prepared at the target rate 0.1 and the config now gives 0.08"):
        ladder_l2.ladder_l2_prepare(config)                            # the stage's first step: the task stops at once
    with pytest.raises(RuntimeError, match="keeps the target rate it began with"):
        ladder_l2.STEPS["ladder_l2_round_1"](config)
    assert not list(ladder_l2._store(config).root.glob("proposals_*"))
    config["ladder_loop"]["l2_arms"]["t010"]["target_rate"] = 0.10
    assert ladder_l2.ladder_l2_prepare(config)["target_rate"] == 0.10
    # Nor does a run change its arm: one prepared with no arm is not continued as an arm (a run-directory override), nor the reverse.
    monkeypatch.delenv(ARM)
    ladder_l2.ladder_l2_prepare(config)
    monkeypatch.setenv(ARM, "t010")
    monkeypatch.setenv(ladder_l2.L2_RUN_VARIABLE, "ladder_l2_seed0")
    with pytest.raises(RuntimeError, match="prepared with no arm and this task runs arm t010"):
        ladder_l2.ladder_l2_prepare(config)
    monkeypatch.delenv(ARM)
    monkeypatch.setenv(ladder_l2.L2_RUN_VARIABLE, "ladder_l2_t010_seed0")
    with pytest.raises(RuntimeError, match="prepared as arm t010 and this task runs no arm"):
        ladder_l2.ladder_l2_prepare(config)


def test_the_arms_stage_is_the_l2_stage_with_the_arms_variable():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l2"]]
    of_the_arm = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l2_t010"]]
    assert [(environment, step) for environment, step, _ in of_the_arm] == [(environment, step) for environment, step, _ in steps]
    assert "ladder_l2_control_trained" in [step for _, step, _ in of_the_arm]
    assert all(options == {"environment": {ARM: "t010"}} for _, _, options in of_the_arm) and "t010" in CONFIG["ladder_loop"]["l2_arms"]
    variables = entry.child_environment("gpu", "key", of_the_arm[2][2])
    assert variables[ARM] == "t010" and not [name for name in (ladder_l2.L2_DATA_VARIABLE, ladder_l2.L2_RUN_VARIABLE, ladder_l2.L2_PROBLEMS_VARIABLE) if name in variables]
    assert ARM not in entry.child_environment("gpu", "key", steps[2][2])
    assert ARM not in entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES["ladder_l2_smoke"][2])[2])


def test_the_stage_is_registered_with_a_guard_before_every_gpu_step_and_its_smoke_run_reads_the_l1_smoke_run():
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l2"]]
    gpu_steps = [step for environment, step, _ in steps if environment == "gpu"]
    by_round = [f"ladder_l2_{name}_{number}" for number in ROUNDS for name in ("round", "train", "measure")]
    assert gpu_steps == ["fix_tokenizers", "ladder_l2_prepare", "ladder_l2_embed", *by_round, "ladder_l2_control", "ladder_l2_control_trained", "ladder_l2_report"]
    assert gpu_steps[1:] == list(ladder_l2.STEPS) and all(options == {} for _, _, options in steps)
    assert tuple(range(1, SETTINGS["round"]["rounds"] + 1)) == ladder_l2.ROUNDS
    # Every step that loads a model or an engine starts on a card the step before it has given back.
    for position, (environment, step, _) in enumerate(steps):
        if environment == "gpu" and step not in ("fix_tokenizers", "ladder_l2_prepare"):
            assert steps[position - 1][0] == "guard", step
    assert not set(ladder_l2.STEPS) & (set(ladder_round.STEPS) | set(ladder_dose.STEPS) | set(ladder_loop.STEPS))
    smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l2_smoke"]]
    assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in steps]
    variables = entry.child_environment("gpu", "key", smoke[2][2])
    assert Path(variables[ladder_loop.DATA_VARIABLE]) == L0_FIXTURE and Path(variables[ladder_l2.L2_DATA_VARIABLE]) == L1_FIXTURE
    assert variables[ladder_l2.L2_SOURCE_VARIABLE] == "ladder_l1_smoke" and variables[ladder_l2.L2_RUN_VARIABLE] == "ladder_l2_smoke"
    # The smoke run's sizes: three rounds of four problems in two batches use the fixture's twelve candidates up.
    assert (variables[ladder_l2.L2_PROBLEMS_VARIABLE], variables[ladder_l2.L2_BATCHES_VARIABLE]) == ("4", "2")
    candidates = len((L1_FIXTURE / "candidates.jsonl").read_text().splitlines())
    assert len(ROUNDS) * int(variables[ladder_l2.L2_PROBLEMS_VARIABLE]) == candidates
    real = entry.child_environment("gpu", "key", steps[2][2])
    assert not [name for name in (ladder_l2.L2_DATA_VARIABLE, *L2_VARIABLES) if name in real]


def test_every_step_the_stage_names_is_one_the_step_runner_knows(tmp_path, monkeypatch):
    """`python -m rlvr_lean.gpu <step>` is how the entry shim runs a step (it cannot import them): each of the
    stage's steps reaches its own function there, and a skipped step is not a failed one."""
    from rlvr_lean.gpu import __main__ as gpu_main

    ran = []
    for name in list(ladder_l2.STEPS):
        monkeypatch.setitem(ladder_l2.STEPS, name, lambda config, name=name: ran.append(name) or {"skipped": "the stop rule fired after round 1"})
    names = [step for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l2"]) if environment == "gpu" and step != "fix_tokenizers"]
    for name in names:
        monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
        assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert ran == names
