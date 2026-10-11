"""L4c, the arm from `pre_r64` with a rule L4t2 names, end to end in the scripted world: with each of the two rules whose
sets hold rehearsal rows the arm trains ONE more model after its last round (`rehearsal_alone`: from the start adapter at
its rank, on the rehearsal rows of the last model's set alone, in that set's order, the text read from the pretraining
file) and measures it as the twin is, before its report; the steps' refusals; a run directory L4b's build left finished
gets the two steps and a new report when its stage is run again; and under `old` and `reward_rows` the arm has no such
model and leaves what L4b's build left. Spec: docs/spec/ladder-loop.spec.md, "L4c: the arm from `pre_r64` with a
rule L4t2 names". The engine and Lean are scripted; nothing touches a GPU or the network. The rules and the report on
hand-made rows are `test_ladder_l4c.py`; the fourth rule inside the rounds, the smoke stages and the registration are
asserted with the three other rules in `test_ladder_l4b_stage.py`."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import arm  # noqa: E402, F401
from test_ladder_l2_stage import ARM, loop  # noqa: E402, F401
from test_ladder_l4_rows_r64 import _trainings, r64  # noqa: E402, F401
from test_ladder_l4_rows_stage import _all, rows  # noqa: E402, F401
from test_ladder_l4_stage import FIXTURE, _file_rows, _on_the_gpu, l4  # noqa: E402, F401
from test_ladder_l4b_stage import ALONE, L4_ARM, L4B_ARM, ORDER, RULE, STEPS, _order, _stage, l4b  # noqa: E402, F401
from test_ladder_round import stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.ceiling import training_example  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import BRANCHES, NOT_READ  # noqa: E402
from rlvr_lean.domain.ladder_round.l4b import OLD, REHEARSE, REWARD_REHEARSE, REWARD_ROWS, RULES, WITH_REHEARSAL  # noqa: E402
from rlvr_lean.domain.ladder_round.l4c import REHEARSAL_ALONE  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4b  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.ladder_l3d1 import WROTE  # noqa: E402

TRAIN, MEASURE, REPORT = ladder_l4b.TRAIN_ALONE, ladder_l4b.MEASURE_ALONE, ladder_l4b.REPORT
SET_FILE, LOSS_FILE = ladder_l4b.TRAINING_ALONE_FILE, ladder_l4b.LOSS_ALONE_FILE
PARTS = ("rungs", REACH, MORE)
ROW_KEYS = {"row", "row_of_with", "id", "problem_id", "kind", "proof_lines", "lines", "side", "origin", "k", "row_of_the_file"}     # a rehearsal row: its id and its place, NEVER its text
# What an arm of L4b's build leaves in its run directory in this world (two rounds of two batches), beside the episodes of the sets it samples: frozen from L4b's own build.
L4B_FILES = sorted([
    *(f"adapters/{model}/{name}" for model in ("m1", "m2", "without") for name in ("adapter_config.json", "adapter_model.safetensors")),
    *(f"assembly_r{number}_b{batch}.jsonl" for number in (1, 2) for batch in (1, 2)), "base_reach.jsonl", "base_rungs.jsonl", "candidate_scores_r1.jsonl", "candidate_scores_r2.jsonl",
    "heldout_groups.jsonl", "l3d2_heldout_groups.jsonl", "l3d2_loss_without.json", *(f"l3d2_stored_{who}_{part}.jsonl" for who in ("base", "loop") for part in ("more", "reach", "rungs")),
    "l3d2_stored_models.json", "l3d2_training_without.jsonl", "l4_goal_set_again.jsonl", "l4_stored_pre_r64.json", *(f"l4_stored_pre_r64_{part}.jsonl" for part in ("more", "reach", "rungs")),
    "l4b_rank_16.json", *(f"ladder_l2_{what}_r{number}_b{batch}.done.json" for what in ("assembly", "batch", "propose") for number in (1, 2) for batch in (1, 2)),
    "ladder_l2_embed.done.json", "ladder_l2_prepare.done.json", "ladder_l2_round_1.done.json", "ladder_l2_round_2.done.json", "ladder_l2_train_1.done.json", "ladder_l2_train_2.done.json",
    "ladder_l3d2_measure_with.done.json", "ladder_l3d2_measure_without.done.json", "ladder_l3d2_prepare.done.json", "ladder_l3d2_train_without.done.json", "ladder_l4b_prepare.done.json",
    "ladder_l4b_report.done.json", "problems.jsonl", *(f"proposals_r{number}_b{batch}.jsonl" for number in (1, 2) for batch in (1, 2)), "report_ladder_l4b.json",
    "training_examples_r1.jsonl", "training_examples_r2.jsonl", "training_loss_m1.json", "training_loss_m2.json", "training_set_m1.jsonl", "training_set_m2.jsonl"])
OF_THE_ALONE = sorted([f"adapters/{REHEARSAL_ALONE}/adapter_config.json", f"adapters/{REHEARSAL_ALONE}/adapter_model.safetensors", LOSS_FILE, SET_FILE, f"{MEASURE}.done.json",
                       f"{TRAIN}.done.json"])
L4B_REPORT_KEYS = ["spec", "stage", "label", "headline", "lines", "ok", "seed", "arm", "rounds", "stand_in_engine", "branch", "inconclusive", "can_this_run_see_a_win",
                   "the_other_trainings", "primary", "beside_the_primary", "breadth_beside_the_primary", "secondary", "measured_and_not_read", "models", "trainings", "the_pretraining",
                   "the_arm", "heldout", "sizes", "attempts", "not_to_be_read", "not_measured", "check", "start", "rank", "alpha", "rule", "the_start_model", "what_it_cannot_say",
                   "adapters"]
L4B_SETS = sorted([*(f"round_r{number}_b{batch}" for number in (1, 2) for batch in (1, 2)), *(f"l3d2_{part}_{model}" for part in PARTS for model in ("with", "without"))])


def _own_files(store):
    """What a run directory holds beside its sets' episodes, and the sets it sampled."""
    names = sorted(str(path.relative_to(store.root)) for path in store.root.rglob("*") if path.is_file())
    sets = sorted(name[len("episodes_"):-len(".done.json")] for name in names if name.startswith("episodes_") and name.endswith(".done.json") and "_block_" not in name)
    return [name for name in names if not name.startswith("episodes_")], sets


def _as_on_the_gpu(monkeypatch, step, config):
    """A measure step by its GPU path: the model server is handed the adapter (a `LoRARequest`, here the three things it is made of)."""
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        return step(config)


# ------------------------------------------------------------------------------------------------ the two steps
@pytest.mark.parametrize("rule", WITH_REHEARSAL)
def test_an_arm_with_rehearsal_rows_trains_one_more_model_on_the_last_models_rehearsal_rows_alone_and_measures_it_as_the_twin_is(l4b, monkeypatch, rule):
    config, calls, kits, r64_ = l4b.config, l4b.calls, l4b.kits, l4b.r64
    name = l4b.task(rule)
    assert _order(rule) == [*ORDER[:-1], TRAIN, MEASURE, REPORT] == [*ORDER[:-1], *ALONE, ORDER[-1]]         # AFTER the arm's own steps, BEFORE its report
    before = _stage(config, monkeypatch, name, until="ladder_l3d2_measure_without")
    store, start = l4b.store(), r64_.rank / "adapters" / "pre_r64"
    assert list(before) == ORDER[:-1] and not [path.name for path in store.root.iterdir() if REHEARSAL_ALONE in path.name]
    last, twin = store.read_rows(ladder_assembly.training_set_file(2)), store.read_rows(ladder_l3d2.TRAINING_WITHOUT_FILE)
    problems_before = store.read_rows("problems.jsonl")
    trained_before, sampled_before = len(calls), len(kits)

    # ---- THE TRAINING: from the start adapter at ITS rank, one pass, by the twin's code, on the rehearsal rows of M(2)'s set ALONE, in that set's order
    trained = _on_the_gpu(monkeypatch, STEPS[TRAIN], config)
    alone = store.read_rows(SET_FILE)
    of_the_last = [row for row in last if row["origin"] == "pretraining"]
    assert [row["id"] for row in alone] == [row["id"] for row in of_the_last] and 0 < len(alone) < len(twin) < len(last)
    assert [(row["row"], row["row_of_with"]) for row in alone] == [(position, row["row"]) for position, row in enumerate(of_the_last)]
    assert all(set(row) == ROW_KEYS and row["origin"] == "pretraining" for row in alone)       # by id and place, never with text
    assert [{key: value for key, value in row.items() if key not in ("row", "row_of_with")} for row in alone] == [{key: value for key, value in row.items() if key != "row"} for row in of_the_last]
    assert len(calls) == trained_before + 1 and len(kits) == sampled_before
    call, of_the_twin = calls[-1], calls[-2]
    fixture = _file_rows(FIXTURE)
    assert call["examples"] == [training_example(fixture[row["row_of_the_file"]]) for row in alone]        # the text is the pretraining file's, row for row, in the set's order
    assert (call["lora"]["rank"], call["lora"]["alpha"], call["more"], call["max_lora_rank"]) == (64, 128, {"start": start}, 16)
    assert call["schedule"] == {"passes": 1, "reading_steps": [], "checkpoint_steps": {REHEARSAL_ALONE: -(-len(alone) // config["training"]["effective_batch"])}}
    assert call["orders"] == [list(range(len(alone)))] and (call["seed"], call["directory"]) == (of_the_twin["seed"], of_the_twin["directory"]) == (0, store.root / "adapters")
    assert call["tensorboard_run"] == f"ladder_l4b_{L4B_ARM}_rehearsal_alone_seed0" and list(of_the_twin["schedule"]["checkpoint_steps"]) == ["without"]
    assert (trained["model"], trained["check"], trained["rule"], trained["rows"], trained["rows_of_with"], trained["left_out"]) == (REHEARSAL_ALONE, "L4c", rule, len(alone), len(last), len(last) - len(alone))
    assert trained["trained_from"] == f"the stored adapter {start}" and trained["order"] == "M(2)'s, with every row that is not a rehearsal row left out and no row moved"
    assert (trained["adapter_saved"]["rank"], trained["adapter_saved"]["alpha"], trained["adapter_saved"]["ranks"], trained["peak_reserved_gb"]) == (64, 128, [64], 12.64)
    assert json.loads((store.root / "adapters" / REHEARSAL_ALONE / "adapter_config.json").read_text())["r"] == 64 and list(trained["against_the_start_adapter"]) == [REHEARSAL_ALONE]
    losses = json.loads(store.path(LOSS_FILE).read_text())
    assert (losses["model"], losses["rows"], losses["rows_trained"], len(losses["row_losses"])) == (REHEARSAL_ALONE, len(alone), [row["id"] for row in alone], len(alone))
    assert store.read_rows("problems.jsonl") == problems_before                 # the training samples nothing and adds no set

    # ---- THE MEASUREMENT: as the twin is (the rungs, G twice, the twin's sampling seeds), ITS adapter served under a model server started for rank 64
    measured = _as_on_the_gpu(monkeypatch, STEPS[MEASURE], config)
    assert [kit["adapter"] for kit in kits[sampled_before:]] == [("ladder_l4b_rehearsal_alone", 1, str(store.root / "adapters" / REHEARSAL_ALONE))] * 3
    assert all(kit["max_lora_rank"] == 64 and kit["rank"] == 64 for kit in kits[sampled_before:]) and len(calls) == trained_before + 1
    assert (measured["model"], measured["check"], measured["stage"], measured["label"]) == (REHEARSAL_ALONE, "L4c", "l4", "pretrained on published proofs") and set(WROTE) < set(measured)
    for part in PARTS:
        mine, of_the_twin = (json.loads(store.path(f"episodes_l3d2_{part}_{model}.done.json").read_text()) for model in (REHEARSAL_ALONE, "without"))
        assert (mine["sampling_seed"], mine["episodes_each"], mine["problems"]) == (of_the_twin["sampling_seed"], of_the_twin["episodes_each"], of_the_twin["problems"])
        assert ladder_l4b.alone_set(part) == f"l3d2_{part}_rehearsal_alone" and ladder_l4b.alone_set(part, "with") == f"l3d2_{part}_with"
    problems = store.read_rows("problems.jsonl")
    of = lambda model: [{key: value for key, value in row.items() if key != "set"} | {"part": row["set"].split("_")[1]} for row in problems if row["set"].endswith(f"_{model}")]      # noqa: E731
    assert problems[:len(problems_before)] == problems_before and of(REHEARSAL_ALONE) == of("with") and len(of(REHEARSAL_ALONE)) == len(problems) - len(problems_before) > 0
    assert measured["goal"].keys() == before["ladder_l3d2_measure_without"]["goal"].keys() == {REACH, MORE}

    # ---- THE REPORT: L4b's, with the long proofs and the arm's own separation beside the primary; `rehearsal_alone` is kept with the arm's other adapters
    report = STEPS[REPORT](config)
    assert list(report) == [*L4B_REPORT_KEYS[:17], "the_long_proofs_beside_the_primary", "the_rounds_apart_from_the_rehearsal", *L4B_REPORT_KEYS[17:-2], "spec_of_the_separation",
                            *L4B_REPORT_KEYS[-2:]]
    section = report["measured_and_not_read"] if report["inconclusive"] else report
    apart = section["the_rounds_apart_from_the_rehearsal"]
    assert (apart["rehearsal_alone"]["rows"], apart["rehearsal_alone"]["rows_by_origin"], apart["rehearsal_alone"]["training"]["model"]) == (len(alone), {"pretraining": len(alone)}, REHEARSAL_ALONE)
    assert list(apart["checks_of_rehearsal_alone"]) == ["what", "its_training_ran", "it_still_writes_proofs", "lean_answered"]
    assert apart["checks_of_rehearsal_alone"]["lean_answered"]["share_without_an_answer"] == {"rungs_rehearsal_alone": 0.0, "goal_rehearsal_alone_1": 0.0, "goal_rehearsal_alone_2": 0.0}
    assert apart["checks_of_rehearsal_alone"]["its_training_ran"]["the_adapter_differs_from_the_start"]["passes"] is True and apart["difference"]["problems"] == section["primary"]["problems"]
    assert report["adapters"]["there"] == ["m1", "m2", REHEARSAL_ALONE, "without"] and REHEARSAL_ALONE in report["models"] and "THE ARM CARRIES ITS OWN SEPARATION" in report["lines"][0]
    assert not [name for name in report["attempts"] if REHEARSAL_ALONE in name] and report["ok"] is True
    files, sets = _own_files(store)
    assert files == sorted([*L4B_FILES, *OF_THE_ALONE]) and sets == sorted([*L4B_SETS, *(f"l3d2_{part}_rehearsal_alone" for part in PARTS)])

    # ---- a rerun returns what is stored: nothing is trained or sampled again, and no file but the report is written again
    stored = {path: content for path, content in _all(store.root).items() if "report" not in path}
    again = _stage(config, monkeypatch, name)
    assert list(again) == _order(rule) and (again[TRAIN], again[MEASURE]) == (trained, json.loads(json.dumps(measured, default=str))) and len(calls) == trained_before + 1
    assert len(kits) == sampled_before + 3 and {path: content for path, content in _all(store.root).items() if "report" not in path} == stored
    # ---- ITS checks are read on ITS OWN rows (the stand-in gives every model the same attempts: here its rung rows are made to differ, and the twin's are not touched)
    rungs_file = store.path("episodes_l3d2_rungs_rehearsal_alone_problems.jsonl")
    as_measured = rungs_file.read_text()
    rungs_file.write_text("".join(json.dumps({**row, "attempts_capped": row["episodes"] * row["sides"]}) + "\n" for row in _file_rows(rungs_file)))
    capped = STEPS[REPORT](config)
    apart = (capped["measured_and_not_read"] if capped["inconclusive"] else capped)["the_rounds_apart_from_the_rehearsal"]
    assert (apart["name"], apart["checks_of_rehearsal_alone"]["it_still_writes_proofs"]["share"], capped["ok"]) == (NOT_READ, 1.0, True) and "it_still_writes_proofs" in apart["failed_checks"]
    assert capped["can_this_run_see_a_win"] == report["can_this_run_see_a_win"]         # the run's own fourth check reads `with` and the twin, and did not move
    rungs_file.write_text(as_measured)
    # ---- A SET OF ITS OWN LEAN DID NOT ANSWER makes the separation NOT READ and nothing else: the report is still to be read; the task queued again samples that set again
    rows_file = store.path(f"episodes_l3d2_{REACH}_rehearsal_alone_problems.jsonl")
    answered = rows_file.read_text()
    rows_file.write_text("".join(json.dumps({**row, "attempts_without_an_answer": row["episodes"] * row["sides"]}) + "\n" for row in _file_rows(rows_file)))
    unanswered = STEPS[REPORT](config)
    apart = (unanswered["measured_and_not_read"] if unanswered["inconclusive"] else unanswered)["the_rounds_apart_from_the_rehearsal"]
    assert (apart["name"], unanswered["ok"], unanswered["not_to_be_read"]) == (NOT_READ, True, []) and "lean_answered" in apart["failed_checks"]
    assert apart["checks_of_rehearsal_alone"]["lean_answered"]["sets_not_answered"] == ["goal_rehearsal_alone_1"] and unanswered["branch"]["name"] == report["branch"]["name"]
    again = STEPS[MEASURE](config)
    assert again["measured_again"] == [f"l3d2_{REACH}_rehearsal_alone"] and len(kits) == sampled_before + 3 + 1 and rows_file.read_text() == answered      # that set, and no other
    # ---- without its adapter the measurement cannot be made again (as the twin's)
    store.path(f"{MEASURE}.done.json").unlink()
    (store.root / "adapters" / REHEARSAL_ALONE).rename(store.root / "adapters" / "elsewhere")
    with pytest.raises(RuntimeError, match="adapters/rehearsal_alone is not there: this run keeps its adapters"):
        _as_on_the_gpu(monkeypatch, STEPS[MEASURE], config)


# ------------------------------------------------------------------------------------------------ the refusals
def test_the_two_steps_refuse_out_of_their_order_a_set_without_rehearsal_rows_another_file_and_an_adapter_at_another_rank(l4b, monkeypatch, tmp_path):
    config, r64_ = l4b.config, l4b.r64
    name = l4b.task(REWARD_REHEARSE)
    _stage(config, monkeypatch, name, until="ladder_l2_round_2")
    store = l4b.store()
    # BEFORE THE LAST ROUND'S MODEL IS TRAINED there is no set to take the rehearsal rows of; before `rehearsal_alone` is trained there is nothing to measure, and no set is made for it.
    with pytest.raises(RuntimeError, match="the training of `rehearsal_alone` needs the step ladder_l2_train_2 of this run, which is not done: the stage `ladder_l4b` runs the steps in order"):
        STEPS[TRAIN](config)
    _stage(config, monkeypatch, name, until="ladder_l3d2_measure_without")
    problems = store.read_rows("problems.jsonl")
    with pytest.raises(RuntimeError, match="the measurement of `rehearsal_alone` needs the step ladder_l4b_train_rehearsal_alone of this run, which is not done"):
        STEPS[MEASURE](config)
    assert store.read_rows("problems.jsonl") == problems
    # THE REPORT of an arm with rehearsal rows is not written without its separation.
    with pytest.raises(RuntimeError, match=r"L4b's report \(the arm's own separation\) needs the step ladder_l4b_train_rehearsal_alone of this run, which is not done"):
        STEPS[REPORT](config)
    assert not store.is_done(REPORT) and not store.path(ladder_l4b.REPORT_FILE).exists()
    # A LAST SET WITHOUT A REHEARSAL ROW: there is nothing to train the model on; nothing is written.
    last_file = store.path(ladder_assembly.training_set_file(2))
    as_stored = last_file.read_text()
    last_file.write_text("".join(json.dumps(row) + "\n" for row in _file_rows(last_file) if row["origin"] != "pretraining"))
    with pytest.raises(RuntimeError, match=r"M\(2\)'s training set \(training_set_m2.jsonl\): the last model's set holds no rehearsal row among its \d+: there is nothing to train `rehearsal_alone` on. "
                                           "Refused. Nothing was trained"):
        STEPS[TRAIN](config)
    assert not store.path(SET_FILE).exists() and not store.is_done(TRAIN)
    # A REHEARSAL ROW THAT CARRIES TEXT is refused: the set names a published proof by its id and its place, and nothing else.
    last_file.write_text("".join(json.dumps({**row, "proof": "a published proof"} if row["origin"] == "pretraining" else row) + "\n" for row in map(json.loads, as_stored.splitlines())))
    with pytest.raises(RuntimeError, match="rehearsal rows of `rehearsal_alone` carry a text field .* Nothing was trained"):
        STEPS[TRAIN](config)
    last_file.write_text(as_stored)
    assert not store.path(SET_FILE).exists()
    # THE TEXT IS READ FROM THE PRETRAINING FILE, HELD TO THE SHA-256 THE START MODEL'S RUN RECORDED: another file is refused, and the step is not marked done.
    changed = tmp_path / "changed" / "pretraining.jsonl"
    changed.parent.mkdir()
    changed.write_text("".join(json.dumps(row) + "\n" for row in _file_rows(FIXTURE)[:-1]))
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    with pytest.raises(RuntimeError, match="L4's pretraining file has SHA-256 .* and this run was prepared on .*: a run trains on the file its prepare step checked"):
        _on_the_gpu(monkeypatch, STEPS[TRAIN], config)
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    assert not store.is_done(TRAIN) and not (store.root / "adapters" / REHEARSAL_ALONE).exists()
    # AN ADAPTER SAVED AT ANOTHER RANK THAN THE START ADAPTER'S fails the step unmarked, and nothing of the model is measured.
    _trainings(monkeypatch, [], saves={REHEARSAL_ALONE: (16, 32, None)})
    with pytest.raises(RuntimeError, match="the adapter `rehearsal_alone` saved at .*adapters/rehearsal_alone has rank 16 and alpha 32 .* the start adapter `pre_r64` is rank 64 and alpha 128"):
        _on_the_gpu(monkeypatch, STEPS[TRAIN], config)
    assert not store.is_done(TRAIN)
    with pytest.raises(RuntimeError, match="the measurement of `rehearsal_alone` needs the step ladder_l4b_train_rehearsal_alone"):
        STEPS[MEASURE](config)
    # The steps are for THE STAGE'S arm: named by another arm, they are refused as L4b's own are.
    monkeypatch.delenv(RULE)
    monkeypatch.setenv(ARM, L4_ARM)
    for step in (TRAIN, MEASURE):
        with pytest.raises(RuntimeError, match=f"the stages `ladder_l4b_.` run the arm {L4B_ARM} of ladder_loop.l2_assembly_arms and {ARM} names '{L4_ARM}'"):
            STEPS[step](config)
    assert r64_.rank.exists()


def test_under_reward_rehearse_too_the_pretraining_file_is_held_to_the_start_models_record_before_anything_is_sampled(l4b, monkeypatch, tmp_path):
    config, r64_ = l4b.config, l4b.r64
    name = l4b.task(REWARD_REHEARSE)
    _stage(config, monkeypatch, name, until="ladder_l2_prepare")
    store, sampled = l4b.store(), r64_.l4.solver.calls
    changed = tmp_path / "changed" / "pretraining.jsonl"
    changed.parent.mkdir()
    changed.write_text("".join(json.dumps(row) + "\n" for row in _file_rows(FIXTURE)[:-1]))
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    # Its rehearsal rows are rows the START MODEL was pretrained on, as `rehearse`'s are: another file is refused by the arm's own prepare step, with nothing written.
    with pytest.raises(RuntimeError, match="L4's pretraining file has SHA-256 .* and `pre_r64` was trained on .* .ladder_l4_pretrain_r64_seed0.: the rehearsal rows are rows `pre_r64` was trained on"):
        ladder_l4b.ladder_l4b_prepare(config)
    assert not store.is_done(ladder_l4b.PREPARE) and not store.path(ladder_l4.AGAIN_FILE).exists() and r64_.l4.solver.calls == sampled
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    own = ladder_l4b.ladder_l4b_prepare(config)
    assert (own["rule"], store.root.name) == (REWARD_REHEARSE, "ladder_l2_t010_assembly_pre_r64_reward_rehearse_seed0") and name == "ladder_l4b_reward_rehearse"


# -------------------------------------------------------------- a run directory that L4b's build left finished
def test_a_run_l4bs_build_left_finished_gets_the_two_steps_and_a_new_report_when_its_stage_is_run_again(l4b, monkeypatch):
    config, calls, kits, r64_ = l4b.config, l4b.calls, l4b.kits, l4b.r64
    name = l4b.task(REHEARSE)
    earlier = _stage(config, monkeypatch, name, until="ladder_l3d2_measure_without")
    store = l4b.store()
    # What L4b's build left: every step of its own done, its report written and marked (here by hand: that build's report had no separation).
    store.path(ladder_l4b.REPORT_FILE).write_text(json.dumps({"headline": "what L4b's build wrote", "check": "L4b"}))
    store.mark_done(REPORT, {"stage": "l4", "headline": "what L4b's build wrote", "ok": True})
    files, sets = _own_files(store)
    assert files == L4B_FILES and sets == L4B_SETS
    left = _all(store.root)
    sampled, sent, trained, served = r64_.l4.solver.calls, len(r64_.l4.lean.sources), len(calls), len(kits)
    # THE STAGE AGAIN (the same task queued again): every step that was done returns what is stored, and nothing of it is sampled, checked or trained again ...
    again = _stage(config, monkeypatch, name, until="ladder_l3d2_measure_without")
    assert again == earlier and (r64_.l4.solver.calls, len(r64_.l4.lean.sources), len(calls), len(kits)) == (sampled, sent, trained, served) and _all(store.root) == left
    # ... THE TWO NEW STEPS RUN, and the report is built again with the separation.
    summaries = _stage(config, monkeypatch, name)
    assert list(summaries) == _order(REHEARSE) and len(calls) == trained + 1 and len(kits) == served + 3 and r64_.l4.solver.calls > sampled
    assert list(calls[-1]["schedule"]["checkpoint_steps"]) == [REHEARSAL_ALONE] and summaries[TRAIN]["rows"] == sum(row["origin"] == "pretraining" for row in store.read_rows(ladder_assembly.training_set_file(2)))
    now = _all(store.root)
    changed = sorted(path for path in left if now[path] != left[path])
    assert changed == ["ladder_l4b_report.done.json", "problems.jsonl", "report_ladder_l4b.json"]            # the report and its marker; and `problems.jsonl` gained the model's sets
    assert now["problems.jsonl"].startswith(left["problems.jsonl"]) and store.done_summary(REPORT)["headline"] == summaries[REPORT]["headline"] != "what L4b's build wrote"
    files, sets = _own_files(store)
    assert files == sorted([*L4B_FILES, *OF_THE_ALONE]) and sets == sorted([*L4B_SETS, *(f"l3d2_{part}_rehearsal_alone" for part in PARTS)])
    section = summaries[REPORT]["measured_and_not_read"] if summaries[REPORT]["inconclusive"] else summaries[REPORT]
    assert "the_rounds_apart_from_the_rehearsal" in section and "the_long_proofs_beside_the_primary" in section and summaries[REPORT]["branch"]["name"] in BRANCHES


# ------------------------------------------------------------------------------------- the two other rules
@pytest.mark.parametrize("rule", (OLD, REWARD_ROWS))
def test_under_old_and_reward_rows_the_arm_has_no_such_model_and_leaves_what_l4bs_build_left(l4b, monkeypatch, rule):
    config, calls, kits = l4b.config, l4b.calls, l4b.kits
    name = l4b.task(rule)
    summaries = _stage(config, monkeypatch, name)
    store = l4b.store()
    # The stage's steps are L4b's: none of `rehearsal_alone`. Its files, the sets it samples and its report's keys are the ones L4b's build wrote.
    assert list(summaries) == ORDER and not set(ALONE) & {step for _, step, _ in map(entry.step_fields, entry.STAGES[name])}
    files, sets = _own_files(store)
    assert files == L4B_FILES and sets == L4B_SETS and list(summaries[REPORT]) == L4B_REPORT_KEYS and len(calls) == 3 and len(kits) == 1 + 2 + 2 + 3 + 3
    assert not [text for text in (json.dumps(summaries, default=str), *(path.read_text() for path in store.root.glob("*.json*"))) if REHEARSAL_ALONE in text]
    left = _all(store.root)
    # The two steps are no steps of such an arm: each refuses, and nothing is written.
    for step, what in ((TRAIN, "the training of `rehearsal_alone`"), (MEASURE, "the measurement of `rehearsal_alone`")):
        with pytest.raises(RuntimeError, match=f"{what} is not a step of the arm {L4B_ARM} with the training rule `{rule}`: `rehearsal_alone` is trained on the rehearsal rows of the "
                                               r"last model's set, and only a rule of \['rehearse', 'reward_rehearse'\] adds any"):
            STEPS[step](config)
    assert _all(store.root) == left and len(calls) == 3


def test_the_stages_of_the_two_rules_with_rehearsal_rows_and_no_other_carry_the_two_steps():
    assert tuple(entry._LADDER_L4B_RULES) == RULES and tuple(entry._LADDER_L4B_WITH_REHEARSAL) == WITH_REHEARSAL == (REHEARSE, REWARD_REHEARSE)
    assert (TRAIN, MEASURE) == tuple(ALONE) == ("ladder_l4b_train_rehearsal_alone", "ladder_l4b_measure_rehearsal_alone") and (SET_FILE, LOSS_FILE) == (
        "l4b_training_rehearsal_alone.jsonl", "l4b_loss_rehearsal_alone.json")
    of = lambda name: [(environment, step) for environment, step, _ in map(entry.step_fields, entry.STAGES[name])]      # noqa: E731
    for kind in ("", "_smoke"):
        plain = of(f"ladder_l4b_old{kind}")
        assert of(f"ladder_l4b_reward_rows{kind}") == plain and not [step for _, step in plain if step and REHEARSAL_ALONE in step]
        for rule in WITH_REHEARSAL:
            steps = of(f"ladder_l4b_{rule}{kind}")
            # ... the stage of `old`, with the two steps (a guard before each, as before every GPU step) between the twin's measurement and the report
            assert steps == [*plain[:-1], ("gpu", TRAIN), ("guard", None), ("gpu", MEASURE), ("guard", None), plain[-1]] and plain[-3:] == [
                ("gpu", "ladder_l3d2_measure_without"), ("guard", None), ("gpu", "ladder_l4b_report")]
            assert steps[-6:] == [("guard", None), ("gpu", TRAIN), ("guard", None), ("gpu", MEASURE), ("guard", None), ("gpu", REPORT)]
    # No other stage runs them, and every one of them is a step the GPU entry point knows.
    for name, steps in entry.STAGES.items():
        assert (name.replace("_smoke", "") in (f"ladder_l4b_{rule}" for rule in WITH_REHEARSAL)) == bool([step for step in steps if entry.step_fields(step)[1] in ALONE]), name
    assert set(ALONE) < set(ladder_l4b.STEPS) and not set(ALONE) & (set(ladder_l2.STEPS) | set(ladder_l3d2.STEPS) | set(ladder_l4.STEPS))
