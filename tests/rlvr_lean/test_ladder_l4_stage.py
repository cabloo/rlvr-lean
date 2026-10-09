"""L4, the two stages end to end on what an L1 run and two L2 runs stored: the pretraining (the refusals of its prepare
step, one pass from the base in the file's order, the adapter `pre` kept, the measurement, `pre`'s own map of the base
map's problems, the goal set drawn again and the two checks), and the arm from it (the refusals of its prepare steps,
the `loop` half as its only candidates, `pre`'s map in the base map's place, round 1 attempted by `pre`, every model
and the twin trained from `pre`, no H0, the report with the base arm's own gain beside the primary); the base arm left
as it was; the smoke stages; the registration. Spec:
docs/spec/ladder-loop.spec.md, "L4: the loop from a model pretrained on published proofs". The engine and Lean are
scripted; nothing touches a GPU or the network. The rules and the reports on hand-made rows are `test_ladder_l4.py`."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import ARM_NAME as BASE_ARM  # noqa: E402
from test_ladder_assembly import arm  # noqa: E402, F401 - the arm's world on the fixtures, with the scripted solver and Lean
from test_ladder_ceiling_stage import _run_ceiling  # noqa: E402
from test_ladder_l2_stage import ARM, _run_loop, loop  # noqa: E402, F401
from test_ladder_round import _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.dose import run_dose  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import BRANCHES, INCONCLUSIVE, LOOP, MAP_FIELDS, NOT_READ, PRETRAIN, half_of  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import (  # noqa: E402
    ladder_assembly,
    ladder_ceiling,
    ladder_dose,
    ladder_l2,
    ladder_l3d1,
    ladder_l3d2,
    ladder_l4,
    ladder_loop,
    ladder_round,
    milestone2,
    pipeline,
)
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
FIXTURE = PACKAGE / "data" / "ladder_l4_fixture" / "pretraining.jsonl"
L4_ARM = "t010_assembly_pre"
SAY = "l4 (pretrained on published proofs): "
OWN_VARIABLES = (ladder_l4.RUN_VARIABLE, ladder_l4.SOURCE_VARIABLE, ladder_l4.STORED_VARIABLE, ladder_l4.FILE_VARIABLE, ladder_l4.MINIMUMS_VARIABLE)
VARIABLES = (*OWN_VARIABLES, ladder_l3d2.STORED_VARIABLE, ladder_l3d2.MINIMUM_VARIABLE, ladder_l2.L2_ROUNDS_VARIABLE)
STEPS = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS}
LOOP_HALF = ["fixture_c1", "fixture_c2", "fixture_c6", "fixture_c11", "fixture_c12"]       # of the fixtures' twelve candidates, by the hash
BASE_MAP = ["fixture_p1", "fixture_p2", "fixture_p3", "fixture_p4"]                         # the fixtures' base map
CHANGED = {"tensors": 2, "tensors_not_compared": 0, "elements": 64, "elements_changed": 64, "largest_change": 0.01}       # what a training from a start adapter records


def _file_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _files(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def _run(config, stage_name, until=None, rounds=None):
    """A stage's GPU steps in its order; with `rounds`, the steps of the rounds past them are left out (a test's arm runs two)."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
        if environment != "gpu" or step not in STEPS or (rounds and step.rsplit("_", 1)[-1].isdigit() and int(step.rsplit("_", 1)[-1]) > rounds):
            continue
        summaries[step] = STEPS[step](config)
        if step == until:
            break
    return summaries


@pytest.fixture
def l4(arm, monkeypatch):  # noqa: F811
    """The arm's world (the fixtures' twelve candidates, rounds of two problems, the scripted solver and Lean) with
    nothing run yet, the 12-row fixture in the pretraining file's place and the two checks' minimums at zero.
    `stored()` runs what both stages read: L1, then L2 and L2's arm t010. `the_arm()` names L4's arm with two rounds
    (five of the twelve candidates are of the `loop` half) and puts the scripted solver and Lean in."""
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(ARM)
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    monkeypatch.setenv(ladder_l4.MINIMUMS_VARIABLE, "0,0")
    monkeypatch.setenv(ladder_l3d2.MINIMUM_VARIABLE, "3")

    def stored():
        _run_stage(arm.config)
        _run_loop(arm.config)
        monkeypatch.setenv(ARM, "t010")
        _run_loop(arm.config)
        monkeypatch.delenv(ARM)

    def the_arm(rounds=2, name=L4_ARM):
        monkeypatch.setenv(ARM, name)
        monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, str(rounds)) if rounds else monkeypatch.delenv(ladder_l2.L2_ROUNDS_VARIABLE, raising=False)
        arm.with_lean()

    return SimpleNamespace(config=arm.config, stored=stored, the_arm=the_arm, pretrain_store=lambda: ladder_l4._store(arm.config), store=arm.store, solver=arm.solver,
                           lean=arm.lean, harvest=arm.harvest)


def _pretraining_file(tmp_path, monkeypatch, change):
    """A copy of the fixture with its rows changed by `change(rows)`, in the pretraining file's place."""
    rows = _file_rows(FIXTURE)
    change(rows)
    path = tmp_path / "changed" / "pretraining.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(path))
    return path


def _trainings_as_the_gpu_would(monkeypatch, calls):
    """The training's GPU path with the model taken out: `_train_with_readings` is the schedule alone (stand-in
    losses); it records what it was given (`more`: whatever it was passed beside the round's own arguments, which is
    where a start adapter shows) and writes each checkpoint's adapter. A training from a start adapter also records,
    as the real one does, what of each adapter it saved is not the start adapter's."""
    def train_with_readings(config, examples, sample, pairs, schedule, seed, directory, tensorboard_run, orders=None, per_example=False, positions=False, **more):
        calls.append({"examples": examples, "schedule": schedule, "seed": seed, "directory": directory, "tensorboard_run": tensorboard_run, "orders": orders, "more": more})
        train_step, read, _ = ladder_dose._stand_in_calls(len(examples), [], [])

        def save(name, step_number):
            (directory / name).mkdir(parents=True)
            (directory / name / "adapter_model.safetensors").write_text("an adapter")

        return {**run_dose(len(examples), config["training"]["effective_batch"], schedule["passes"], seed, schedule["reading_steps"], schedule["checkpoint_steps"],
                           train_step, read, save, orders=orders, per_example=per_example, positions=positions), "target_format": "native", "seconds": 1.0,
                **({"against_the_start_adapter": {name: dict(CHANGED) for name in schedule["checkpoint_steps"]}} if "start" in more else {})}

    monkeypatch.setattr(ladder_dose, "_train_with_readings", train_with_readings)
    monkeypatch.setattr(pipeline, "_cap_torch_memory", lambda config: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, memory_allocated=lambda: 0)))


def _on_the_gpu(monkeypatch, step, *arguments):
    """One training step by its GPU path (the trainings, and only they, read `ladder_ceiling._stand_in`)."""
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        return step(*arguments)


# ------------------------------------------------------------------------------------------- the pretraining
def test_the_pretraining_stage_runs_on_what_l1_and_l2_stored_keeps_pre_and_stores_the_goal_set_again(l4):
    config = l4.config
    with pytest.raises(RuntimeError, match="L4 reads the run directory the L1 task for seed 0 wrote on this box .stage `ladder_l1` with --seeds 0"):
        ladder_l4.ladder_l4_pretrain_prepare(config)                                              # nothing to read yet: refused, and it says what to run
    _run_stage(config)
    with pytest.raises(RuntimeError, match=r"L4 reads the run directory the task of stage `ladder_l2` for seed 0 .*the base's attempts on G"):
        ladder_l4.ladder_l4_pretrain_prepare(config)
    assert not l4.pretrain_store().is_done(ladder_l4.PREPARE) and not l4.pretrain_store().path("problems.jsonl").exists()
    l4.stored()
    source, stored = ladder_l4.source_directory(config), ladder_ceiling.stored_directories(config, ladder_l4.STORED_VARIABLE, ladder_l4.SETTING)
    assert source == ladder_round._store(config).root and [path.name for path in stored.values()] == ["ladder_l2_seed0", "ladder_l2_t010_seed0"]
    before = {path: _files(path) for path in (source, *stored.values())}
    summaries = _run(config, "ladder_l4_pretrain")
    assert {path: _files(path) for path in before} == before                                      # the three run directories were only read
    store, rows = l4.pretrain_store(), _file_rows(FIXTURE)
    assert store.root.name == "ladder_l4_pretrain_seed0" and store.root.parent == source.parent and list(summaries) == list(ladder_l4.STEPS)[:5] == [
        "ladder_l4_pretrain_prepare", "ladder_l4_pretrain_train", "ladder_l4_pretrain_measure", "ladder_l4_pretrain_map", "ladder_l4_pretrain_report"]

    # ---- prepare: every row of the file, in its order; the counts, the tokens, the file's hash; L2's sampling seeds
    prepare = summaries["ladder_l4_pretrain_prepare"]
    assert (prepare["stage"], prepare["label"], prepare["rows"], prepare["half"], prepare["half_seed"], prepare["steps"]) == ("l4", "pretrained on published proofs", 12, "pretrain", 0, 2)
    assert prepare["rows_by_kind"] == {"lean_workbook": 5, "stp_conjecture": 7} and prepare["rows_by_proof_lines"] == {"1": 1, "2-3": 1, "4-7": 5, "8+": 5}
    assert prepare["pretraining_file"] == str(FIXTURE) and prepare["pretraining_file_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    stored_rows = store.read_rows(ladder_l4.ROWS_FILE)
    assert [row["problem_id"] for row in stored_rows] == [row["problem_id"] for row in rows] and [row["row"] for row in stored_rows] == list(range(12))
    assert prepare["tokens"] == sum(row["tokens"] for row in stored_rows) and prepare["longest_example_tokens"] == max(row["tokens"] for row in stored_rows) < 2048
    assert set(stored_rows[0]) == {"row", "problem_id", "kind", "proof_lines", "length_group", "tokens"}                               # no statement, no proof
    assert not any(row["proof"].strip() in content.decode() for row in rows for content in _files(store.root).values())               # no published proof is copied into the run
    of_l2 = ladder_l2.sampling_seeds(config)
    control = json.loads((stored["base"] / "episodes_control.done.json").read_text())
    assert prepare["sampling_seeds"] == {"rungs": of_l2["rungs"], "reach": of_l2["reach"], "more": of_l2["control"]}
    assert prepare["goal_samplings"] == [{"name": "reach", "episodes": 32, "sampling_seed": of_l2["reach"]},
                                         {"name": "more", "episodes": control["episodes_each"], "sampling_seed": of_l2["control"]}]
    assert (prepare["loop_arm"], prepare["loop_target_rate"], prepare["minimums"]) == ("t010", 0.1, {"goal_problems_solved_by_pre": 0, "goal_set_again": 0})
    assert prepare["ceiling"]["read"] is False and "ladder_ceiling_seed0" in prepare["ceiling"]["why"]                                 # no ceiling run on this box

    # ---- train: one pass in the file's order (stand-in losses), every row's loss stored
    train = summaries["ladder_l4_pretrain_train"]
    assert (train["model"], train["rows"], train["steps"], train["passes"], train["label"]) == ("pre", 12, 2, 1, "pretrained on published proofs") and "made up" in train["note"]
    assert train["adapter"] == str(store.root / "adapters" / "pre") and train["rows_compared"] == 1                                    # a tenth of 12 rows, at least one
    loss = json.loads(store.path(ladder_l4.LOSS_FILE).read_text())
    assert loss["label"] == "pretrained on published proofs" and len(loss["row_losses"]) == 12 and [row["rows_seen"] for row in loss["training_steps"]] == [8, 12]

    # ---- measure: `pre` as the ceiling's models are, pairing by problem with the stored base
    measured = summaries["ladder_l4_pretrain_measure"]
    assert (measured["stage"], measured["model"], measured["label"]) == ("l4", "pre", "pretrained on published proofs")
    assert measured["rungs"]["sampling_seed"] == of_l2["rungs"] and measured["rungs"]["episodes_each"] == 8
    assert (measured["goal"][REACH]["sampling_seed"], measured["goal"][REACH]["episodes_each"]) == (of_l2["reach"], 32)
    assert (measured["goal"][MORE]["sampling_seed"], measured["goal"][MORE]["episodes_each"]) == (of_l2["control"], control["episodes_each"])
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {"l4_rungs_pre", "l4_reach_pre", "l4_more_pre", "l4_map_pre"}
    for part in ("rungs", REACH, MORE):
        assert {row["problem_id"] for row in store.read_rows(f"episodes_l4_{part}_pre_problems.jsonl")} == {
            row["problem_id"] for row in store.read_rows(ladder_ceiling.stored_file("base", part, "l4"))}
    assert set(measured["opens_with_a_have"]) == {"rungs", "goal"} and len(measured["episodes_of_8"]) == 1

    # ---- `pre`'s OWN MAP: the base map's problems attempted again by `pre`, with what the stored map (the base's) was made with
    made, of_the_stored = summaries["ladder_l4_pretrain_map"], config["ladder_loop"]["base_map"]
    assert (made["model"], made["label"], made["set"], made["problems"], made["episodes_each"], made["sampling_seed"]) == (
        "pre", "pretrained on published proofs", "l4_map_pre", 4, of_the_stored["episodes"], of_the_stored["sampling_seed"]) and (made["episodes_each"], made["sampling_seed"]) == (8, 101)
    episodes = json.loads(store.path("episodes_l4_map_pre.done.json").read_text())
    assert (episodes["sampling_seed"], episodes["episodes_each"], episodes["problems"]) == (101, 8, 4)                                  # the episode step's own record
    the_map = store.read_rows("l4_map_pre.jsonl")
    assert [row["problem_id"] for row in the_map] == BASE_MAP and all(list(row) == [*MAP_FIELDS, "set"] and row["set"] == "base_map" and row["episodes"] == 8 for row in the_map)
    by_problem = {row["problem_id"]: row for row in store.read_rows("episodes_l4_map_pre_problems.jsonl")}
    assert all(row["resolved"] == by_problem[row["problem_id"]]["resolved"] and row["sides"] == by_problem[row["problem_id"]]["sides"] for row in the_map)
    stored_map = [row for row in _file_rows(PACKAGE / "data" / "ladder_l1_fixture" / "base_results.jsonl") if row["set"] == "base_map"]
    assert made["map"]["problems"] == 4 and made["map_of_the_base"]["mean_pass_rate"] == round(sum(row["resolved"] / 8 for row in stored_map) / 4, 5)
    assert made["map"]["mean_pass_rate"] == round(sum(row["resolved"] / 8 for row in the_map) / 4, 5) and sum(made["map"]["problems_by_k"].values()) == 4
    # Its problems carry their exact negations by the rule that gave the stored map's; a fixture's stored rows are hand-made, so its sides are not held to them (and that is said).
    built = made["problems_built"]
    assert (built["problems"], built["exactness_checks"], built["two_sided"], built["sides_held_to_the_stored_map"]) == (4, 4, 4, False) and "hand-made" in built["why_not"]
    assert built == json.loads(store.path(ladder_l4.MAP_PROBLEMS_FILE).read_text()) and all(row["negation"] for row in store.read_rows("problems.jsonl") if row["set"] == "l4_map_pre")

    # ---- the report: `pre` against the base, G' with its ids stored for the arm, the two checks; every line says what the model is
    report = summaries["ladder_l4_pretrain_report"]
    assert report["stage"] == "l4" and report["label"] == "pretrained on published proofs" and report["ok"] is True and report["checks_pass"] is True
    assert all(line.startswith(SAY) for line in report["lines"]) and "pretrained on published proofs" in report["headline"]
    first = {row["problem_id"]: row["resolved"] for row in store.read_rows("episodes_l4_reach_pre_problems.jsonl")}
    again = [problem_id for problem_id, resolved in first.items() if resolved == 0]
    assert report["goal_set_again"]["problem_ids"] == again == [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)]
    assert list(report["against_the_base"]["by_length_group"]["models"]) == ["pre", "loop"] and report["models"]["loop"] == "the stored three-round model at t = 1/10"
    assert report["against_the_base"]["by_length_group"]["models"]["pre"]["all"]["attempts_each"] == prepare["attempts_a_goal_problem"]
    assert report["adapter"]["kept"] is True and report["adapter"]["directory"] == str(store.root / "adapters" / "pre")
    assert report["the_map_the_arm_starts_from"]["pre"] == made["map"] and report["the_map_the_arm_starts_from"]["base"] == made["map_of_the_base"]
    assert (report["the_map_the_arm_starts_from"]["file"], report["the_map_the_arm_starts_from"]["sampling_seed"]) == ("l4_map_pre.jsonl", 101)
    assert any("THE MAP THE ARM'S CHALLENGER STARTS FROM is `pre`'s own (l4_map_pre.jsonl): the base map's 4 problems attempted again by `pre`, 8 attempts each" in line
               for line in report["lines"])
    assert json.loads(store.path(ladder_l4.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str))
    assert store.done_summary(ladder_l4.REPORT) == {"stage": "l4", "label": "pretrained on published proofs", "headline": report["headline"], "ok": True, "checks_pass": True}
    # Every stored file and every set says l4, but the one name the shared episode step reads.
    assert [name for name in _files(store.root) if "l4" not in name] == ["problems.jsonl"]
    # A rerun returns what is stored: nothing is trained or sampled again.
    assert {name: summary for name, summary in _run(config, "ladder_l4_pretrain").items() if "report" not in name} == {
        name: summary for name, summary in summaries.items() if "report" not in name}
    assert not ladder_round._store(config).is_done(ladder_l4.PREPARE) and not ladder_l2._store(config).is_done(ladder_l4.PREPARE)


def test_the_pretraining_prepare_step_refuses_a_file_it_must_not_train_on_and_writes_nothing(l4, monkeypatch, tmp_path):
    config = l4.config
    l4.stored()
    held = _file_rows(ladder_l4.source_directory(config) / "heldout_groups.jsonl")[0]["problem_id"]
    mapped = json.loads((PACKAGE / "data" / "ladder_l0_fixture" / "base_map.jsonl").read_text().splitlines()[0])["problem_id"]
    of_the_loop_half = next(row["problem_id"] for row in _file_rows(PACKAGE / "data" / "ladder_ceiling_fixture" / "training.jsonl") if half_of(row["problem_id"], 0) == LOOP)

    def refused(match, error=ValueError):
        with pytest.raises(error, match=match):
            ladder_l4.ladder_l4_pretrain_prepare(config)
        assert not l4.pretrain_store().is_done(ladder_l4.PREPARE) and not l4.pretrain_store().path("problems.jsonl").exists()

    for change, match in (
            (lambda rows: rows[7].update(problem_id=held), f"1 rows of L4's pretraining file are held-out problems .first: {held}"),
            (lambda rows: rows[7].update(problem_id=mapped), f"1 rows of L4's pretraining file are problems of the base map .first: {mapped}"),
            (lambda rows: rows[7].update(problem_id=of_the_loop_half), f"1 problems of L4's pretraining file are not of the `pretrain` half .first: {of_the_loop_half}.: refused"),
            (lambda rows: rows[7].update(side="false"), "1 rows of L4's pretraining file are not on the side `true`"),
            (lambda rows: rows.append(dict(rows[2])), f"1 problems appear twice in L4's pretraining file .first: {_file_rows(FIXTURE)[2]['problem_id']}"),
            (lambda rows: rows[3].update(proof="  nlinarith [sq_nonneg (a - b)]\n" * 400), "1 rows of L4's pretraining file are longer than the 2048 tokens a training example is cut at"),
            (lambda rows: rows[5].pop("proof"), "row 5 of L4's pretraining file has no .'proof'."),
            (lambda rows: rows.clear(), "L4's pretraining file holds no row: refused")):
        _pretraining_file(tmp_path, monkeypatch, change)
        refused(match)
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(tmp_path / "nowhere" / "pretraining.jsonl"))
    refused(r"L4's pretraining file .*nowhere/pretraining\.jsonl is not in this snapshot: it is built on the dev machine by `tools/ladder_ceiling_set.py --file l4_pretrain`", RuntimeError)
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    monkeypatch.setenv(ladder_l4.STORED_VARIABLE, "ladder_l2_smoke")
    refused(f"{ladder_l4.STORED_VARIABLE} is 'ladder_l2_smoke': it is `none`")
    monkeypatch.delenv(ladder_l4.STORED_VARIABLE)
    # The snapshot's own held-out set and base map (the real ones here) bar their problems too, whatever the run placed.
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, what in (("heldout.jsonl", "held-out problems"), ("base_map.jsonl", "problems of the base map")):
        barred = json.loads((PACKAGE / "data" / "ladder_l0" / name).read_text().splitlines()[0])["problem_id"]
        _pretraining_file(tmp_path, monkeypatch, lambda rows, barred=barred: rows[11].update(problem_id=barred))
        refused(f"1 rows of L4's pretraining file are {what} .first: {barred}")
    # ... and so do the RUN's own held-out groups, whatever the snapshot's set holds (the real one here knows no problem of the fixtures).
    _pretraining_file(tmp_path, monkeypatch, lambda rows: rows[11].update(problem_id=held))
    refused(f"1 rows of L4's pretraining file are held-out problems .first: {held}")
    # The package's own file is the one a task reads when no other is named; the minimums are the spec's.
    monkeypatch.delenv(ladder_l4.FILE_VARIABLE)
    monkeypatch.delenv(ladder_l4.MINIMUMS_VARIABLE)
    assert ladder_l4.pretraining_file() == ladder_l4.PACKAGE_FILE == PACKAGE / "data" / "ladder_l4" / "pretraining.jsonl" and ladder_l4.minimums(config) == (150, 80)


def test_the_pretraining_is_one_pass_from_the_base_in_the_files_order_and_its_adapter_is_kept(l4, monkeypatch):
    config, calls = l4.config, []
    l4.stored()
    with pytest.raises(RuntimeError, match="the pretraining needs the step ladder_l4_pretrain_prepare of this run, which is not done: the stage `ladder_l4_pretrain`"):
        ladder_l4.ladder_l4_pretrain_train(config)
    ladder_l4.ladder_l4_pretrain_prepare(config)
    _trainings_as_the_gpu_would(monkeypatch, calls)
    train = _on_the_gpu(monkeypatch, ladder_l4.ladder_l4_pretrain_train, config)
    (call,), rows, store = calls, _file_rows(FIXTURE), l4.pretrain_store()
    # The rows as the round's training examples, in the file's order; ONE pass, nothing shuffled; FROM THE BASE (no start adapter); one adapter, at the end.
    assert call["examples"] == [{"problem_id": row["problem_id"], "side": "statement", "theorem": row["statement"], "completion": row["proof"]} for row in rows]
    assert call["orders"] == [list(range(12))] and call["schedule"] == {"passes": 1, "reading_steps": [], "checkpoint_steps": {"pre": 2}} and call["more"] == {}
    assert (call["seed"], call["tensorboard_run"], call["directory"]) == (0, "ladder_l4_pretrain_seed0", store.root / "adapters")
    adapter = store.root / "adapters" / "pre"
    assert (adapter / "adapter_model.safetensors").exists() and train["adapter"] == str(adapter) and "note" not in train and train["allocated_after_cleanup_gb"] == 0
    recipe = store.done_summary(ladder_l4.PREPARE)["recipe"]
    assert (recipe["learning_rate"], recipe["warmup_steps"], recipe["effective_batch"], recipe["lora"]) == (1.0e-4, 5, 8, config["lora"])
    # A training that is done is not made again; one whose marker is gone starts again FROM THE BASE (nothing of a pass is kept but its adapter).
    assert ladder_l4.ladder_l4_pretrain_train(config) == train and len(calls) == 1
    # A measurement of `pre` without its adapter is refused.
    adapter.rename(adapter.with_name("elsewhere"))
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        with pytest.raises(RuntimeError, match="the adapter `pre` is kept by this stage, and one trained again would not be the model"):
            ladder_l4.ladder_l4_pretrain_measure(config)
    adapter.with_name("elsewhere").rename(adapter)
    with pytest.raises(RuntimeError, match="the pretraining's report needs the step ladder_l4_pretrain_measure"):
        ladder_l4.ladder_l4_pretrain_report(config)
    ladder_l4.ladder_l4_pretrain_measure(config)
    with pytest.raises(RuntimeError, match="the pretraining's report needs the step ladder_l4_pretrain_map"):
        ladder_l4.ladder_l4_pretrain_report(config)
    # The map is `pre`'s: without the adapter it cannot be made (as on the GPU), and nothing is sampled for it.
    adapter.rename(adapter.with_name("elsewhere"))
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l4, "_stand_in", lambda: False)
        with pytest.raises(RuntimeError, match="adapters/pre is not there: the adapter `pre` is kept by this stage, and the map is `pre`'s"):
            ladder_l4.ladder_l4_pretrain_map(config)
    adapter.with_name("elsewhere").rename(adapter)
    assert not store.is_done("episodes_l4_map_pre") and not store.path("l4_map_pre.jsonl").exists()
    # As on the GPU: the map's attempts are `pre`'s, the kept adapter served beside the base under a name of its own.
    kits = []
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l4, "_stand_in", lambda: False)
        for name in ("vllm", "vllm.lora"):
            patched.setitem(sys.modules, name, SimpleNamespace())
        patched.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
        patched.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (kits.append((self.enable_lora, adapter)), lambda config: (l4.solver, stand_in_parameters))[1])
        made = ladder_l4.ladder_l4_pretrain_map(config)
    assert kits == [(True, ("ladder_l4_pretrain_pre", 1, str(adapter)))] and made["problems"] == 4 and store.is_done("episodes_l4_map_pre")
    report = ladder_l4.ladder_l4_pretrain_report(config)
    # THE ADAPTER IS KEPT: the report, built once and again, deletes nothing.
    assert report["adapter"]["kept"] is True and ladder_l4.ladder_l4_pretrain_report(config)["adapter"]["kept"] is True and (adapter / "adapter_model.safetensors").exists()
    # A file that changed since the run was prepared is not trained on.
    store.path(f"{ladder_l4.TRAIN}.done.json").unlink()
    changed = store.root.parent / "changed.jsonl"
    changed.write_text(FIXTURE.read_text() + "\n")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    with pytest.raises(RuntimeError, match="L4's pretraining file has SHA-256 .* a run trains on the file its prepare step checked"):
        _on_the_gpu(monkeypatch, ladder_l4.ladder_l4_pretrain_train, config)
    assert len(calls) == 1


def test_the_ceilings_two_models_stand_beside_pre_when_its_run_is_on_the_box(l4, monkeypatch):
    config = l4.config
    l4.stored()
    monkeypatch.setenv(ladder_ceiling.CEILING_TRAINING_VARIABLE, str(PACKAGE / "data" / "ladder_ceiling_fixture" / "training.jsonl"))
    monkeypatch.setenv(ladder_ceiling.CEILING_CHECKPOINTS_VARIABLE, "12,24")
    _run_ceiling(config)
    ceiling = ladder_ceiling._store(config).root
    before = _files(ceiling)
    summaries = _run(config, "ladder_l4_pretrain")
    assert _files(ceiling) == before                                                               # the ceiling's run was only read
    read = summaries["ladder_l4_pretrain_prepare"]["ceiling"]
    assert read == {"read": True, "run": str(ceiling), "models": {"ceiling_small": {"rows": 16}, "ceiling_full": {"rows": 24}}}
    store = l4.pretrain_store()
    for name in ("small", "full"):
        for part in ("rungs", REACH, MORE):
            assert store.read_rows(ladder_ceiling.stored_file(f"ceiling_{name}", part, "l4")) == _file_rows(ceiling / f"episodes_ceiling_{part}_{name}_problems.jsonl")
    report = summaries["ladder_l4_pretrain_report"]
    assert list(report["against_the_base"]["by_length_group"]["models"]) == ["pre", "loop", "ceiling_small", "ceiling_full"]
    assert report["models"]["ceiling_small"] == "the ceiling's 16-proof model" and report["models"]["ceiling_full"] == "the ceiling's 24-proof model"
    assert set(report["goal_problems_solved_by_attempts_alone"]) == {"what", "base", "pre", "loop", "ceiling_small", "ceiling_full"}
    assert report["the_ceiling"]["read"] is True and any("the ceiling's 24-proof model minus the base" in line for line in report["lines"])
    # A ceiling's set sampled otherwise than this run samples is not read, with the reason: the pretraining is not refused for it.
    store.path(f"{ladder_l4.PREPARE}.done.json").unlink()
    done = ceiling / "episodes_ceiling_more_full.done.json"
    done.write_text(json.dumps({**json.loads(done.read_text()), "sampling_seed": 5}))
    again = ladder_l4.ladder_l4_pretrain_prepare(config)["ceiling"]
    assert again["read"] is False and "ladder_ceiling_seed0 measured the ceiling's model full (more) with sampling seed 5" in again["why"]


# --------------------------------------------------------------------------------------------------- the arm
def _pretrained(l4, monkeypatch, calls=None):
    """What the arm stands on: the stored runs and the pretraining stage, its training by the GPU path (so `pre` is on disk)."""
    l4.stored()
    _trainings_as_the_gpu_would(monkeypatch, [] if calls is None else calls)
    ladder_l4.ladder_l4_pretrain_prepare(l4.config)
    _on_the_gpu(monkeypatch, ladder_l4.ladder_l4_pretrain_train, l4.config)
    ladder_l4.ladder_l4_pretrain_measure(l4.config)
    ladder_l4.ladder_l4_pretrain_map(l4.config)
    return ladder_l4.ladder_l4_pretrain_report(l4.config)


def test_the_arm_runs_from_pre_on_the_loop_half_alone_and_its_report_reads_the_goal_set_again(l4, monkeypatch):
    config, calls, kits, fitted_on = l4.config, [], [], []
    of_pre = _pretrained(l4, monkeypatch, calls)
    pre = l4.pretrain_store().root / "adapters" / "pre"
    # `pre`'s own map, as its stage stored it; here it is given other numbers than the base's stored map has, so that which of the two a fit read can be told.
    the_map = [{**row, "resolved": resolved, "resolved_by_statement": resolved} for row, resolved in zip(l4.pretrain_store().read_rows("l4_map_pre.jsonl"), (7, 5, 1, 4))]
    l4.pretrain_store().write_rows("l4_map_pre.jsonl", the_map)
    l4.the_arm()
    refit = ladder_l2.refit_observations
    monkeypatch.setattr(ladder_l2, "refit_observations", lambda base, finished, number, decay: (fitted_on.append([dict(row) for row in base]), refit(base, finished, number, decay))[1])
    monkeypatch.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (kits.append((self.enable_lora, adapter)), lambda config: (l4.solver, stand_in_parameters))[1])
    assert ladder_l2.assembly_arm(config) == {"target_rate": 0.10, "rounds": 2, "start": "pre", "candidates": "loop_half", "h0": False, "map": "start_model"}
    assert ladder_l2.start_directory(config) == pre and ladder_l2.arm_config(config)["ladder_loop"]["challenger"]["target_rate"] == 0.10
    del calls[:]
    with monkeypatch.context() as patched:                                                         # the trainings, and only they, take their GPU path
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        _run(config, "ladder_l4", rounds=2, until="ladder_l3d2_train_without")
    summaries = _run(config, "ladder_l4", rounds=2)
    store = l4.store()
    assert store.root.name == "ladder_l2_t010_assembly_pre_seed0" and list(summaries) == [
        "ladder_l2_prepare", "ladder_l4_prepare", "ladder_l3d2_prepare", "ladder_l2_embed", "ladder_l2_round_1", "ladder_l2_train_1", "ladder_l2_round_2", "ladder_l2_train_2",
        "ladder_l3d2_train_without", "ladder_l3d2_measure_with", "ladder_l3d2_measure_without", "ladder_l4_report"]

    # ---- the arm's own prepare step: the `loop` half as the whole of its candidates, the start adapter, no H0
    prepare = summaries["ladder_l2_prepare"]
    assert (prepare["arm"], prepare["target_rate"], prepare["rounds"], prepare["assembly"]) == (L4_ARM, 0.10, [1, 2], True)
    assert (prepare["candidates"], prepare["data"]["candidates"], prepare["candidates_of_the_whole_pool"], prepare["candidates_short_by"]) == ("loop_half", 5, 12, 0)
    assert (prepare["start"], prepare["start_adapter"], prepare["h0_file"], prepare["h0_rows"], prepare["h0"]) == ("pre", str(pre), None, 0, "this arm reads no H0")
    assert summaries["ladder_l2_embed"]["candidates"] == 5 and summaries["ladder_l2_embed"]["statements"] == 5 + 4                 # the base map's four, and the `loop` half
    # ---- THE CHALLENGER STARTS FROM `pre`'s OWN MAP: its first fit, and every refit, read that map's rows where the base arm's read the base's stored ones
    assert (prepare["map"], prepare["map_file"]) == ("start_model", str(l4.pretrain_store().root / "l4_map_pre.jsonl"))
    assert prepare["map_of_the_start_model"]["problems_by_k"] == {"0": 0, "1": 1, "2": 0, "3": 0, "4": 1, "5": 1, "6": 0, "7": 1, "8": 0} and prepare["map_of_the_start_model"]["mean_pass_rate"] == round(17 / 32, 5)
    stored_map = {row["problem_id"]: row["resolved"] for row in _file_rows(PACKAGE / "data" / "ladder_l1_fixture" / "base_results.jsonl") if row["set"] == "base_map"}
    assert len(fitted_on) == 4 and all(observed == the_map for observed in fitted_on)              # one fit a batch, each on the map of the model the arm starts from
    assert [row["resolved"] for row in fitted_on[0]] == [7, 5, 1, 4] != [stored_map[name] for name in BASE_MAP] and [row["problem_id"] for row in fitted_on[0]] == BASE_MAP
    assert [row for row in ladder_l2._data(ladder_l2.arm_config(config))["base_results"] if row["set"] == "base_map"] == the_map
    own = summaries["ladder_l4_prepare"]
    assert (own["label"], own["arm"], own["pretraining_run"], own["start_adapter"], own["pretraining_rows"]) == (
        "pretrained on published proofs", L4_ARM, str(l4.pretrain_store().root), str(pre), 12)
    assert own["the_two_checks_of_the_pretraining"] == {name: check for name, check in of_pre["can_this_run_see_a_win"].items() if isinstance(check, dict)}
    assert own["goal_set_again"] == of_pre["goal_set_again"]["problems"] and own["base_arm_run"] is None                              # no base arm's run on this box
    assert [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)] == of_pre["goal_set_again"]["problem_ids"]
    for part in ("rungs", REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("pre", part, "l4")) == l4.pretrain_store().read_rows(f"episodes_l4_{part}_pre_problems.jsonl")

    # ---- NO PROBLEM OF THE `pretrain` HALF is proposed, scored or trained on
    batches = [(number, batch) for number in (1, 2) for batch in (1, 2)]
    proposed = [row["problem_id"] for number, batch in batches for row in store.read_rows(f"proposals_r{number}_b{batch}.jsonl")]
    assert len(proposed) == 4 == len(set(proposed)) and set(proposed) <= set(LOOP_HALF) and all(half_of(name, 0) == LOOP for name in LOOP_HALF)
    assert {row["problem_id"] for number in (1, 2) for row in store.read_rows(f"candidate_scores_r{number}.jsonl")} == set(LOOP_HALF)
    trained_on = {row["problem_id"] for number in (1, 2) for row in store.read_rows(ladder_assembly.training_set_file(number))}
    assert trained_on and trained_on <= set(proposed) and not any(row["origin"] == "h0" for number in (1, 2) for row in store.read_rows(ladder_assembly.training_set_file(number)))

    # ---- ROUND 1 IS ATTEMPTED BY `pre`: the engine serves the start adapter; round 2 by M(1) (the stand-in has no adapter to hand)
    assert kits[:2] == [(True, ("the start adapter", pre)), (True, None)] and kits[2:] == [(True, None), (True, None)]                # then the two measurements
    assert (summaries["ladder_l2_round_1"]["attempted_by_the_start_adapter"], summaries["ladder_l2_round_2"]["attempted_by_the_start_adapter"]) == (True, False)
    assert summaries["ladder_l2_round_1"]["start_adapter"] == str(pre) == summaries["ladder_l2_round_2"]["start_adapter"]

    # ---- EVERY MODEL OF THE ARM, AND THE TWIN, IS TRAINED FROM `pre`: the start adapter is what the training loads, not a fresh one on the base
    assert [call["tensorboard_run"] for call in calls] == [f"ladder_l2_{L4_ARM}_m1_seed0", f"ladder_l2_{L4_ARM}_m2_seed0", f"ladder_l3d2_{L4_ARM}_without_seed0"]
    assert all(call["more"] == {"start": pre} and call["directory"] == store.root / "adapters" and call["seed"] == 0 for call in calls)
    assert [list(call["schedule"]["checkpoint_steps"]) for call in calls] == [["m1"], ["m2"], ["without"]]
    assert sorted(path.name for path in (store.root / "adapters").iterdir()) == ["m1", "m2", "without"] and (pre / "adapter_model.safetensors").exists()
    for name in ("ladder_l2_train_1", "ladder_l2_train_2", "ladder_l3d2_train_without"):
        assert summaries[name]["trained_from"] == f"the stored adapter {pre}"
    with_rows, without_rows = store.read_rows(ladder_assembly.training_set_file(2)), store.read_rows(ladder_l3d2.TRAINING_WITHOUT_FILE)
    assert [row["id"] for row in without_rows] == [row["id"] for row in with_rows if row["origin"] == "attempt"]                       # the twin: the same order, the others left out
    assert calls[1]["examples"] == [{key: row[key] for key in ("problem_id", "side", "theorem", "completion")} for row in with_rows]

    # ---- the measurements and the report
    assert {row["set"] for row in store.read_rows("problems.jsonl")} == {f"l3d2_{part}_{model}" for part in ("rungs", REACH, MORE) for model in ("with", "without")} | {
        f"round_r{number}_b{batch}" for number, batch in batches}
    assert summaries["ladder_l3d2_measure_with"]["is"] == "M(2)" and summaries["ladder_l3d2_measure_without"]["is"] == "the twin of M(2)"
    report = summaries["ladder_l4_report"]
    assert report["stage"] == "l4" and report["label"] == "pretrained on published proofs" and report["arm"] == L4_ARM and report["ok"] is True
    assert all(line.startswith(SAY) for line in report["lines"]) and report["branch"]["name"] in BRANCHES
    assert [line[len(SAY):][:7] for line in report["lines"][1:6]] == ["CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "CHECK 5"]
    checks = report["can_this_run_see_a_win"]
    assert list(checks)[1:] == ["the_pretraining_took", "the_goal_set_again_is_large_enough", "the_two_measured_trainings_ran", "each_measured_model_still_writes_proofs",
                                "no_training_row_is_of_the_pretrain_half_or_held_out"]
    # The third check reads the two measured trainings, each with what its training recorded of the adapter it saved against `pre`'s; M(1) is printed and decides nothing.
    ran = checks["the_two_measured_trainings_ran"]
    assert set(ran) == {"what", "M(2)", "`without`", "passes"} and set(report["the_other_trainings"]) == {"what", "M(1)"}
    assert all(ran[name]["the_adapter_differs_from_the_start"] == {"compared": True, "elements": 64, "elements_changed": 64, "largest_change": 0.01, "passes": True}
               for name in ("M(2)", "`without`"))
    assert ran["M(2)"]["the_loss_did_not_rise"]["rows"] == len(with_rows) and summaries["ladder_l2_train_2"]["against_the_start_adapter"] == {"m2": CHANGED}
    assert summaries["ladder_l3d2_train_without"]["against_the_start_adapter"] == {"without": CHANGED} and report["lines"][6][len(SAY):].startswith("for information, deciding nothing")
    assert set(checks["each_measured_model_still_writes_proofs"]) >= {"pre", "with", "without"}
    barred = checks["no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert barred["passes"] is True and barred["rows"] == {"M(1)": len(store.read_rows(ladder_assembly.training_set_file(1))), "M(2)": len(with_rows), "`without`": len(without_rows)}
    assert report["inconclusive"] == (report["branch"]["name"] == INCONCLUSIVE) == (not all(check["passes"] for check in checks.values() if isinstance(check, dict)))
    read = report["measured_and_not_read"] if report["inconclusive"] else report
    # THE PRIMARY is read on G' over the second sampling alone: `with`'s and `pre`'s stored rows of it, problem by problem.
    again = of_pre["goal_set_again"]["problem_ids"]
    of = lambda rows: sum(row["resolved"] for row in rows if row["problem_id"] in again)      # noqa: E731
    assert read["primary"]["goal_set_again"] == len(again) == read["primary"]["problems"]
    if again:
        assert read["primary"]["successes"] == of(store.read_rows("episodes_l3d2_more_with_problems.jsonl"))
        assert read["primary"]["successes_of_the_base"] == of(l4.pretrain_store().read_rows("episodes_l4_more_pre_problems.jsonl"))
    assert read["beside_the_primary"] is None and set(read["secondary"]["goal_problems_solved_by_attempts_alone"]) == {"what", "base", "pre", "with", "without", "loop"}
    assert [row["round"] for row in read["secondary"]["by_round"]["rows"]] == [1, 2] and all(row["picks"] == 2 for row in read["secondary"]["by_round"]["rows"])
    assert read["secondary"]["with_minus_without"]["assembled_in_the_rounds"] == sum(row["origin"] == "assembled" for row in with_rows)
    assert report["trainings"]["M(2)"]["trained_from"] == f"the stored adapter {pre}" and report["the_arm"]["candidates"] == "loop_half" and report["the_arm"]["start"] == "pre"
    assert report["adapters"]["kept"] is True and report["adapters"]["there"] == ["m1", "m2", "without"] and report["adapters"]["start_adapter"] == str(pre)
    assert json.loads(store.path(ladder_l4.ARM_REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str)) and store.is_done(ladder_l4.ARM_REPORT)
    assert not store.path(ladder_l3d2.REPORT_FILE).exists()                                         # Step 2's report is the base arm's: this arm writes its own
    # A rerun returns what is stored: nothing is sampled, checked or trained again.
    solver_calls, sent, trainings = l4.solver.calls, len(l4.lean.sources), len(calls)
    again_run = _run(config, "ladder_l4", rounds=2)
    assert (l4.solver.calls, len(l4.lean.sources), len(calls)) == (solver_calls, sent, trainings)
    assert all(again_run[step] == summaries[step] for step in summaries if step != "ladder_l4_report")
    # A training row of the `pretrain` half (its published proof was pretrained on): the fifth check fails, and the report says INCONCLUSIVE and nothing else.
    rows = store.read_rows(ladder_assembly.training_set_file(2))
    store.write_rows(ladder_assembly.training_set_file(2), [{**rows[0], "problem_id": "fixture_c3"}, *rows[1:]])
    broken = ladder_l4.ladder_l4_report(config)
    assert broken["branch"]["name"] == INCONCLUSIVE and "no_training_row_is_of_the_pretrain_half_or_held_out" in broken["branch"]["failed_checks"]
    assert broken["primary"] is None and not any("PRIMARY" in line or "SECONDARY" in line for line in broken["lines"]) and broken["lines"][5].endswith("FAIL")
    assert broken["can_this_run_see_a_win"]["no_training_row_is_of_the_pretrain_half_or_held_out"]["first"] == ["fixture_c3"]


def test_the_arms_prepare_steps_refuse_before_anything_is_sampled(l4, monkeypatch):
    config = l4.config
    l4.stored()
    l4.the_arm()
    pretrain = l4.pretrain_store().root
    # THE MAP of the model the arm starts from must be there: the arm's own prepare step reads it before anything else, and says what to run.
    with pytest.raises(RuntimeError, match=r"ladder_l4_pretrain_seed0 does not hold the map of the model this arm starts from .l4_map_pre.jsonl, and the marker of the step "
                                           r"ladder_l4_pretrain_map.\. .* Run the task of stage `ladder_l4_pretrain` for seed 0 to its end first"):
        ladder_l2.ladder_l2_prepare(config)
    assert not l4.store().is_done(ladder_l2.PREPARE)
    monkeypatch.delenv(ARM)
    monkeypatch.setenv(ladder_l4.MINIMUMS_VARIABLE, "150,80")                                      # the spec's minimums: this small world fails both
    failed = _run(config, "ladder_l4_pretrain")["ladder_l4_pretrain_report"]
    assert failed["checks_pass"] is False and failed["ok"] is True
    # ... and it must be the stored map of another model: one made with another sampling seed, number of attempts or set of problems is refused.
    l4.the_arm()
    marker, file = pretrain / "ladder_l4_pretrain_map.done.json", pretrain / "l4_map_pre.jsonl"
    as_made, rows = marker.read_text(), _file_rows(file)

    def refused(match):
        with pytest.raises(RuntimeError, match=match):
            ladder_l2.ladder_l2_prepare(config)
        assert not l4.store().is_done(ladder_l2.PREPARE)
        marker.write_text(as_made)
        file.write_text("".join(json.dumps(row) + "\n" for row in rows))

    marker.write_text(json.dumps({**json.loads(as_made), "sampling_seed": 5}))
    refused("l4_map_pre.jsonl cannot stand in the base map's place: it was made with sampling seed 5 and 8 attempts a problem; the stored map's are 101 and 8")
    marker.write_text(json.dumps({**json.loads(as_made), "episodes_each": 4}))
    refused("it was made with sampling seed 101 and 4 attempts a problem; the stored map's are 101 and 8")
    file.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    refused("it is of 3 problems and the base map holds 4; they are not the same problems .first that is in one alone: fixture_p4")
    file.write_text("".join(json.dumps(row) + "\n" for row in [*rows[:-1], {**rows[-1], "problem_id": "fixture_c1"}]))
    refused("it is of 4 problems and the base map holds 4; they are not the same problems")
    file.write_text("".join(json.dumps(row) + "\n" for row in [*rows[:-1], {**rows[-1], "episodes": 16}]))
    refused("1 of its problems do not have 8 attempts .first: fixture_p4")
    file.unlink()
    refused("does not hold the map of the model this arm starts from")
    marker.unlink()                                                                                # the file without the marker of the step that made it: the same refusal
    refused("does not hold the map of the model this arm starts from .l4_map_pre.jsonl, and the marker of the step ladder_l4_pretrain_map.")
    # The `loop` half must hold the arm's rounds; a candidate of the `pretrain` half among the arm's candidates is refused.
    l4.the_arm(rounds=None)                                                                        # the arm's own six rounds
    with pytest.raises(RuntimeError, match="the `loop` half holds 5 candidates and 6 rounds of 2 need 12: refused, nothing was written"):
        ladder_l2.ladder_l2_prepare(config)
    assert not l4.store().is_done(ladder_l2.PREPARE)
    l4.the_arm()
    with pytest.raises(RuntimeError, match="L4's own prepare step needs the step ladder_l2_prepare of this run, which is not done: the stage `ladder_l4`"):
        ladder_l4.ladder_l4_prepare(config)
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l2, "half_of", lambda problem_id, seed: LOOP)                       # a cut gone wrong: it lets every candidate through
        with pytest.raises(ValueError, match="7 problems of the arm's candidates are not of the `loop` half .first: fixture_c10.: refused, nothing was written"):
            ladder_l2.ladder_l2_prepare(config)
    assert not l4.store().is_done(ladder_l2.PREPARE)
    # An arm that asks for the map of the model it starts from and names no start is a configuration error.
    arms = config["ladder_loop"]["l2_assembly_arms"]
    start = arms[L4_ARM].pop("start")
    with pytest.raises(ValueError, match="asks for the map of the model it starts from and names no start"):
        ladder_l2.assembly_arm(config)
    arms[L4_ARM]["start"] = start
    ladder_l2.ladder_l2_prepare(config)
    # L4's own prepare step: the pretraining run's report must be there, its two checks passed, and its adapter on disk.
    report_file = pretrain / ladder_l4.REPORT_FILE
    report_file.rename(report_file.with_name("elsewhere.json"))
    with pytest.raises(RuntimeError, match=r"ladder_l4_pretrain_seed0 does not hold \['report_ladder_l4_pretrain.json'\]. L4's arm starts from the model the task of stage "
                                           r"`ladder_l4_pretrain` for seed 0 .*pretrained, and reads its report"):
        ladder_l4.ladder_l4_prepare(config)
    report_file.with_name("elsewhere.json").rename(report_file)
    for lost in (file, marker):                                                                    # ... and its map, with the marker of the step that made it
        lost.rename(lost.with_name("elsewhere"))
        with pytest.raises(RuntimeError, match=rf"ladder_l4_pretrain_seed0 does not hold \['{lost.name}'\]"):
            ladder_l4.ladder_l4_prepare(config)
        lost.with_name("elsewhere").rename(lost)
    with pytest.raises(RuntimeError, match="the pretraining run ladder_l4_pretrain_seed0 cannot carry the arm: its checks failed "
                                           r".the_pretraining_took, the_goal_set_again_is_large_enough.\. The arm is not run on it; nothing was written"):
        ladder_l4.ladder_l4_prepare(config)
    monkeypatch.setenv(ladder_l4.MINIMUMS_VARIABLE, "0,0")
    monkeypatch.delenv(ARM)
    (pretrain / f"{ladder_l4.REPORT}.done.json").unlink()
    assert ladder_l4.ladder_l4_pretrain_report(config)["checks_pass"] is True
    l4.the_arm()
    sampled = l4.solver.calls                                                                      # the pretraining's own measurement, and no more from here on
    with monkeypatch.context() as patched:                                                         # as on the GPU: the adapter must be on disk (the stand-in trained none)
        patched.setattr(ladder_l4, "_stand_in", lambda: False)
        with pytest.raises(RuntimeError, match=r"adapters/pre is not there: the adapter `pre` is what every model of the arm is trained from and what attempts round 1"):
            ladder_l4.ladder_l4_prepare(config)
    ladder_l2.ladder_l2_embed(config)
    with monkeypatch.context() as patched:                                                         # ... and a round asks for it too, before it samples
        patched.setattr(ladder_l2, "_stand_in", lambda: False)
        with pytest.raises(RuntimeError, match=r"adapters/pre is not there: the models of this arm are trained from that adapter and its first round is attempted by it"):
            ladder_l2.ladder_l2_round(config, 1)
    assert l4.solver.calls == sampled and not l4.store().is_done(ladder_l4.ARM_PREPARE)            # nothing was sampled
    own = ladder_l4.ladder_l4_prepare(config)
    assert (own["pretraining_run"], own["map"], own["map_file"]) == (str(pretrain), "start_model", str(file))
    assert ladder_l4.ladder_l4_prepare(config) == l4.store().done_summary(ladder_l4.ARM_PREPARE)   # made once
    # A round handed the whole pool by a cut gone wrong finds that its stored embeddings are of the `loop` half alone, and samples nothing.
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l2, "half_of", lambda problem_id, seed: LOOP)
        with pytest.raises(RuntimeError, match="does not hold the embeddings of this data's statements"):
            ladder_l2.ladder_l2_round(config, 1)
    assert l4.solver.calls == sampled
    # The step named by another arm, or by none, is refused.
    monkeypatch.setenv(ARM, BASE_ARM)
    with pytest.raises(RuntimeError, match=f"the stage `ladder_l4` runs the arm {L4_ARM} of ladder_loop.l2_assembly_arms and {ARM} names '{BASE_ARM}'"):
        ladder_l4.ladder_l4_prepare(config)
    # The real start request: vLLM serves `pre` beside the base under an id no round's model has.
    monkeypatch.setenv(ARM, L4_ARM)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.lora", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda name, number, path: (name, number, path)))
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l2, "_stand_in", lambda: False)
        assert ladder_l2._start_request(pretrain / "adapters" / "pre") == ("ladder_l4_pre", 100, str(pretrain / "adapters" / "pre"))
        assert ladder_l2._adapter(l4.store(), 6) == ("ladder_l2_m6", 6, str(l4.store().root / "adapters" / "m6"))


def test_the_map_is_held_to_the_stored_maps_sides_and_one_lean_did_not_answer_is_not_kept(l4, monkeypatch):
    config = l4.config
    l4.stored()
    ladder_l4.ladder_l4_pretrain_prepare(config)
    with pytest.raises(RuntimeError, match="`pre`'s own map needs the step ladder_l4_pretrain_train of this run, which is not done"):
        ladder_l4.ladder_l4_pretrain_map(config)
    ladder_l4.ladder_l4_pretrain_train(config)
    store = l4.pretrain_store()
    built, with_negations = [], ladder_round.with_negations
    monkeypatch.setattr(ladder_round, "with_negations", lambda given, rows: (built.append(len(rows)), with_negations(given, rows))[1])
    # On REAL data the stored map's rows are what an episode step sampled: a problem that would now be attempted on other sides stops the step before anything is sampled.
    real = ladder_l4.stored_map(config)
    assert [row["problem_id"] for row in real[0]] == BASE_MAP and len(real[1]) == 4 and real[2] is True                               # the fixture says it is one
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l4, "stored_map", lambda given: (real[0], real[1], False))
        with pytest.raises(RuntimeError, match=r"4 of the base map's 4 problems would be attempted on other sides than the stored map's were .first: fixture_p1, 1 against 2.: "
                                               "the episode settings or an exactness check are not what they were when the stored map was made. Nothing was sampled"):
            ladder_l4.ladder_l4_pretrain_map(config)
        assert not store.is_done("episodes_l4_map_pre") and not any(row["set"] == "l4_map_pre" for row in store.read_rows("problems.jsonl"))
        # With the stored map's own sides (here one a problem, as this world's settings give) the same step runs, and records that they were held to them.
        patched.setattr(ladder_l4, "stored_map", lambda given: (real[0], [{**row, "sides": 1} for row in real[1]], False))
        held = ladder_l4.ladder_l4_pretrain_map(config)
        assert held["problems_built"]["sides_held_to_the_stored_map"] is True and held["problems_built"]["problems_on_other_sides_than_the_stored_map"] == 0
        assert held["problems_built"]["sides"] == {"1": 4} == held["problems_built"]["sides_of_the_stored_map"] and "why_not" not in held["problems_built"]
    # A stored map that is not 8 attempts on each of the base map's problems is refused; so is a map Lean did not answer, which a rerun samples again.
    store.path(f"{ladder_l4.MAP}.done.json").unlink()
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l4, "stored_map", lambda given: (real[0], [{**row, "episodes": 16} for row in real[1]], True))
        with pytest.raises(RuntimeError, match="the stored map is not 8 attempts on each of the base map's 4 problems .4 rows; 4 with another number of attempts."):
            ladder_l4.ladder_l4_pretrain_map(config)
        patched.setattr(ladder_l4, "stored_map", lambda given: real)
        patched.setattr(ladder_l4, "lean_did_not_answer", lambda rows: True)
        with pytest.raises(RuntimeError, match="Lean gave no verdict on too many attempts of l4_map_pre: the map is not kept. Queue the task again"):
            ladder_l4.ladder_l4_pretrain_map(config)
    assert not store.is_done(ladder_l4.MAP) and not store.is_done("episodes_l4_map_pre")           # forgotten: the next run samples it again
    again = ladder_l4.ladder_l4_pretrain_map(config)
    assert again["map"] == held["map"] and store.is_done("episodes_l4_map_pre") and ladder_l4.ladder_l4_pretrain_map(config) == again
    # THE PROBLEMS ARE BUILT ONCE: by the step that was refused for its sides (which wrote none) and by the one that then ran; the reruns after it asked Lean for
    # no exactness check again and read the problems the first run wrote, so that a set sampled in two runs is sampled on one plan.
    assert built == [4, 4] and sum(row["set"] == "l4_map_pre" for row in store.read_rows("problems.jsonl")) == 4


def test_a_batch_cannot_propose_a_candidate_of_the_pretrain_half(l4, monkeypatch):
    config = l4.config
    _pretrained(l4, monkeypatch)
    l4.the_arm()
    for step in (ladder_l2.ladder_l2_prepare, ladder_l4.ladder_l4_prepare, ladder_l3d2.ladder_l3d2_prepare, ladder_l2.ladder_l2_embed):
        step(config)
    store = l4.store()
    data, statements, sizes = ladder_l2._data(ladder_l2.arm_config(config)), None, ladder_l2._sizes(ladder_l2.arm_config(config), store)
    assert [row["problem_id"] for row in data["candidates"]] == LOOP_HALF and data["counts"]["candidates"] == 5 and data["counts"]["candidates_of_the_whole_pool"] == 12
    statements = ladder_l2._statements(store, data)
    assert set(statements["position"]) == {*LOOP_HALF, "fixture_p1", "fixture_p2", "fixture_p3", "fixture_p4"}        # no statement of the other half is embedded or scored
    # The whole pool handed to a batch by mistake: its proposals are refused before any is made.
    whole = ladder_round._data(config, ladder_l2.data_directory())
    with pytest.raises(ValueError, match="7 problems of a batch's candidates are not of the `loop` half .first: fixture_c10."):
        ladder_l2._propose(ladder_l2.arm_config(config), store, whole, statements, sizes, 1, 1)
    assert not store.is_done("ladder_l2_propose_r1_b1") and not store.path("proposals_r1_b1.jsonl").exists()
    picks = ladder_l2._propose(ladder_l2.arm_config(config), store, data, statements, sizes, 1, 1)
    assert len(picks) == 1 and picks[0]["problem_id"] in LOOP_HALF


# ------------------------------------------------------------------------- the base arm is what it was
def test_the_base_arm_is_what_it_was_and_its_own_gain_stands_beside_the_primary(l4, monkeypatch):
    config, calls, kits = l4.config, [], []
    # ---- its settings and its step lists, literally
    assert config["ladder_loop"]["l2_assembly_arms"][BASE_ARM] == {"target_rate": 0.10, "rounds": 6} and config["ladder_loop"]["l3d"]["step_2"]["arm"] == BASE_ARM
    assert [(environment, step) for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l3d2"]) if environment == "gpu"] == [
        ("gpu", "fix_tokenizers"), ("gpu", "ladder_l2_prepare"), ("gpu", "ladder_l3d2_prepare"), ("gpu", "ladder_l2_embed"),
        ("gpu", "ladder_l2_round_1"), ("gpu", "ladder_l2_train_1"), ("gpu", "ladder_l2_round_2"), ("gpu", "ladder_l2_train_2"), ("gpu", "ladder_l2_round_3"),
        ("gpu", "ladder_l2_train_3"), ("gpu", "ladder_l2_round_4"), ("gpu", "ladder_l2_train_4"), ("gpu", "ladder_l2_round_5"), ("gpu", "ladder_l2_train_5"),
        ("gpu", "ladder_l2_round_6"), ("gpu", "ladder_l2_train_6"), ("gpu", "ladder_l3d2_train_without"), ("gpu", "ladder_l3d2_measure_with"),
        ("gpu", "ladder_l3d2_measure_without"), ("gpu", "ladder_l3d2_report")]
    assert all(options == {"environment": {ARM: BASE_ARM}} for _, _, options in map(entry.step_fields, entry.STAGES["ladder_l3d2"]))
    assert list(ladder_l3d2.STEPS) == ["ladder_l3d2_prepare", *(f"ladder_l2_{name}_{number}" for number in (4, 5, 6) for name in ("round", "train")),
                                       "ladder_l3d2_train_without", "ladder_l3d2_measure_with", "ladder_l3d2_measure_without", "ladder_l3d2_report"]
    l4.stored()
    l4.the_arm(rounds=None, name=BASE_ARM)
    assert ladder_l2.assembly_arm(config) == {"target_rate": 0.10, "rounds": 6} and ladder_l2.start_directory(config) is None
    whole = ladder_l2._data(ladder_l2.arm_config(config))
    assert len(whole["candidates"]) == 12 and whole == ladder_round._data(config, ladder_l2.data_directory())                         # the whole pool, as it reads it
    # ---- it runs as it did: round 1 by the base with no adapter served, every model and the twin from the base, H0 read
    _trainings_as_the_gpu_would(monkeypatch, calls)
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (kits.append((self.enable_lora, adapter)), lambda config: (l4.solver, stand_in_parameters))[1])
    fitted_on, refit = [], ladder_l2.refit_observations
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l2, "refit_observations", lambda base, finished, number, decay: (fitted_on.append([dict(row) for row in base]), refit(base, finished, number, decay))[1])
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        _run(config, "ladder_l3d2", until="ladder_l3d2_train_without")
    of_the_base_arm = _run(config, "ladder_l3d2")
    stored_map = [row for row in _file_rows(PACKAGE / "data" / "ladder_l1_fixture" / "base_results.jsonl") if row["set"] == "base_map"]
    assert len(fitted_on) == 12 and all(observed == stored_map for observed in fitted_on)          # every fit of the base arm is on the BASE's stored map
    base_store = l4.store()
    assert base_store.root.name == "ladder_l2_t010_assembly_seed0" and kits[0] == (False, None) and kits[1:6] == [(True, None)] * 5
    assert len(calls) == 7 and all(call["more"] == {} for call in calls)                           # no start adapter is handed to any of its trainings
    assert [call["tensorboard_run"] for call in calls] == [*(f"ladder_l2_{BASE_ARM}_m{number}_seed0" for number in range(1, 7)), "ladder_l3d2_without_seed0"]
    assert all(of_the_base_arm[step]["trained_from"] == "the base" for step in (*(f"ladder_l2_train_{number}" for number in range(1, 7)), "ladder_l3d2_train_without"))
    prepare = of_the_base_arm["ladder_l2_prepare"]
    assert (prepare["h0_file"], prepare["h0_rows"], prepare["data"]["candidates"]) == (str(l4.harvest), 3, 12)
    assert not [key for key in prepare if key in ("candidates", "candidates_of_the_whole_pool", "start", "start_adapter", "h0", "map", "map_file", "map_of_the_start_model")]
    assert not any("against_the_start_adapter" in of_the_base_arm[step] or "start_adapter_loaded" in of_the_base_arm[step]
                   for step in (*(f"ladder_l2_train_{number}" for number in range(1, 7)), "ladder_l3d2_train_without"))
    assert not [key for key in of_the_base_arm["ladder_l2_round_1"] if "start" in key] and sum(row["origin"] == "h0" for row in base_store.read_rows(ladder_assembly.training_set_file(6))) == 2
    names = set(_files(base_store.root))
    by_round = ("episodes_", "proposals_", "assembly_", "candidate_scores_", "training_", "ladder_l2_")       # what a round, a batch or a training of the arm writes
    assert sorted(name for name in names if not name.startswith(by_round)) == [
        "base_reach.jsonl", "base_rungs.jsonl", "heldout_groups.jsonl", "l3d2_heldout_groups.jsonl", "l3d2_loss_without.json", "l3d2_stored_base_more.jsonl",
        "l3d2_stored_base_reach.jsonl", "l3d2_stored_base_rungs.jsonl", "l3d2_stored_loop_more.jsonl", "l3d2_stored_loop_reach.jsonl", "l3d2_stored_loop_rungs.jsonl",
        "l3d2_stored_models.json", "l3d2_training_without.jsonl", "ladder_l3d2_measure_with.done.json", "ladder_l3d2_measure_without.done.json",
        "ladder_l3d2_prepare.done.json", "ladder_l3d2_report.done.json", "ladder_l3d2_train_without.done.json", "problems.jsonl", "report_ladder_l3d2.json"]
    assert sorted(name for name in names if name.startswith("training_")) == sorted(
        f"training_{kind}{number}.{suffix}" for kind, suffix in (("examples_r", "jsonl"), ("loss_m", "json"), ("set_m", "jsonl")) for number in range(1, 7))
    assert not [name for name in names if "l4" in name] and "report_ladder_l3d2.json" in names and of_the_base_arm["ladder_l3d2_report"]["stage"] == "l3d2"
    assert all(line.startswith("l3d2: ") for line in of_the_base_arm["ladder_l3d2_report"]["lines"])
    as_it_wrote = _files(base_store.root)

    # ---- L4's arm beside it: the base arm's run is only read, and its last model's stored rows give the figure beside the primary
    monkeypatch.delenv(ARM)
    for step in (ladder_l4.ladder_l4_pretrain_prepare, ladder_l4.ladder_l4_pretrain_measure, ladder_l4.ladder_l4_pretrain_map, ladder_l4.ladder_l4_pretrain_report):
        if step is ladder_l4.ladder_l4_pretrain_measure:
            _on_the_gpu(monkeypatch, ladder_l4.ladder_l4_pretrain_train, config)
        step(config)
    l4.the_arm()
    with monkeypatch.context() as patched:
        patched.setattr(ladder_ceiling, "_stand_in", lambda: False)
        _run(config, "ladder_l4", rounds=2, until="ladder_l3d2_train_without")
    summaries = _run(config, "ladder_l4", rounds=2)
    assert _files(base_store.root) == as_it_wrote                                                  # nothing of the base arm's run was written
    store = l4.store()
    assert summaries["ladder_l4_prepare"]["base_arm_run"] == str(base_store.root) and summaries["ladder_l4_prepare"]["base_arm"] == BASE_ARM
    for part in (REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("base_arm", part, "l4")) == base_store.read_rows(f"episodes_l3d2_{part}_with_problems.jsonl")
    report = summaries["ladder_l4_report"]
    read = report["measured_and_not_read"] if report["inconclusive"] else report
    beside = read["beside_the_primary"]
    base_first = {row["problem_id"]: row["resolved"] for row in base_store.read_rows(ladder_ceiling.stored_file("base", REACH, "l3d2"))}
    base_again = [problem_id for problem_id, resolved in base_first.items() if resolved == 0]
    assert beside["run"] == str(base_store.root) and beside["goal_set_again_of_the_base"] == len(base_again) == beside["problems"]
    if base_again:
        assert beside["successes"] == sum(row["resolved"] for row in base_store.read_rows("episodes_l3d2_more_with_problems.jsonl") if row["problem_id"] in base_again)
        assert beside["successes_of_the_base"] == sum(row["resolved"] for row in base_store.read_rows(ladder_ceiling.stored_file("base", MORE, "l3d2")) if row["problem_id"] in base_again)
    if not report["inconclusive"]:
        assert any(line.startswith(f"{SAY}BESIDE IT, the base arm's own gain from its stored rows") for line in report["lines"])
    # With the stored-runs variable saying none (a smoke run), the base arm's run is not read even when it is on the box.
    store.path(f"{ladder_l4.ARM_PREPARE}.done.json").unlink()
    monkeypatch.setenv(ladder_l3d2.STORED_VARIABLE, "none")
    assert ladder_l4.ladder_l4_prepare(config)["base_arm_run"] is None


# ------------------------------------------------------------------------------------------ the smoke stages
def test_the_two_smoke_stages_run_on_the_l1_smoke_run_with_the_fixture_and_two_rounds_of_the_loop_half(arm, monkeypatch):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config = arm.config
    monkeypatch.delenv(ARM)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    of_l2_smoke = entry.step_fields(entry.STAGES["ladder_l2_smoke"][2])[2]["environment"]
    # ---- the pretraining's smoke: the 12-row fixture, the L1 smoke run alone, the two minimums at zero
    options = entry.step_fields(entry.STAGES["ladder_l4_pretrain_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_pretrain_smoke"])
    # Its data are the L2 smoke run's fixtures: the map is then of the fixture's four base-map problems, the ones the arm's smoke run reads.
    assert options["environment"] == {ladder_l4.SOURCE_VARIABLE: "ladder_l1_smoke", ladder_l4.RUN_VARIABLE: "ladder_l4_pretrain_smoke", ladder_l4.STORED_VARIABLE: "none",
                                      ladder_l4.MINIMUMS_VARIABLE: "0,0", ladder_l4.FILE_VARIABLE: str(FIXTURE),
                                      ladder_loop.DATA_VARIABLE: of_l2_smoke[ladder_loop.DATA_VARIABLE], ladder_l2.L2_DATA_VARIABLE: of_l2_smoke[ladder_l2.L2_DATA_VARIABLE]}
    with monkeypatch.context() as patched:                                     # one task's environment: gone when the task is
        patched.delenv(ladder_loop.DATA_VARIABLE)
        for name, value in options["environment"].items():
            patched.setenv(name, value)
        pretrained = _run(config, "ladder_l4_pretrain_smoke")
        pretrain = ladder_l4._store(config)
        assert pretrain.root.name == "ladder_l4_pretrain_smoke" and not (pretrain.root.parent / "ladder_l4_pretrain_seed0").exists()
        prepare = pretrained["ladder_l4_pretrain_prepare"]
        assert prepare["stored_runs"] is None and prepare["rows"] == 12 and prepare["attempts_a_goal_problem"] == 32 and prepare["ceiling"]["read"] is False
        assert [sampling["name"] for sampling in prepare["goal_samplings"]] == [REACH] and prepare["minimums"] == {"goal_problems_solved_by_pre": 0, "goal_set_again": 0}
        of_pre = pretrained["ladder_l4_pretrain_report"]
        assert of_pre["checks_pass"] is True and list(of_pre["models"]) == ["base", "pre"] and any("L2's stored runs were not read (a smoke run)" in line for line in of_pre["lines"])
        made = pretrained["ladder_l4_pretrain_map"]
        assert (made["problems"], made["episodes_each"], made["sampling_seed"], made["fixture"]) == (4, 8, 101, True) and [
            row["problem_id"] for row in pretrain.read_rows("l4_map_pre.jsonl")] == BASE_MAP
    # ---- the arm's smoke: the L2 smoke run's world, the pretraining SMOKE run's adapter and report, two rounds of two problems
    options = entry.step_fields(entry.STAGES["ladder_l4_smoke"][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES["ladder_l4_smoke"])
    assert options["environment"] == {**of_l2_smoke, ladder_l2.L2_RUN_VARIABLE: "ladder_l4_smoke", ARM: L4_ARM, ladder_l2.L2_ROUNDS_VARIABLE: "2",
                                      ladder_l2.L2_PROBLEMS_VARIABLE: "2", ladder_l3d2.STORED_VARIABLE: "none", ladder_l4.RUN_VARIABLE: "ladder_l4_pretrain_smoke"}
    assert not [name for name in OWN_VARIABLES if name in options["environment"] and name != ladder_l4.RUN_VARIABLE]
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    config["ladder_loop"]["round"].update({"problems": 4000, "batches": 4})    # the smoke's sizes are the environment's, whatever the config's
    arm.with_lean()
    summaries = _run(config, "ladder_l4_smoke")
    store = ladder_l3d2._store(config)
    assert store.root.name == "ladder_l4_smoke" and not (store.root.parent / f"ladder_l2_{L4_ARM}_seed0").exists()
    assert [step for step in summaries if "round" in step] == ["ladder_l2_round_1", "ladder_l2_round_2"] and summaries["ladder_l2_prepare"]["rounds"] == [1, 2]
    arm_prepare = summaries["ladder_l2_prepare"]
    assert (arm_prepare["problems_a_round"], arm_prepare["batch_sizes"], arm_prepare["h0_rows"], arm_prepare["data"]["candidates"]) == (2, [1, 1], 0, 5)
    assert arm_prepare["start_adapter"] == str(pretrain.root / "adapters" / "pre") and summaries["ladder_l4_prepare"]["pretraining_run"] == str(pretrain.root)
    assert (arm_prepare["map"], arm_prepare["map_file"]) == ("start_model", str(pretrain.root / "l4_map_pre.jsonl")) and arm_prepare["map_of_the_start_model"] == made["map"]
    assert summaries["ladder_l4_prepare"]["base_arm_run"] is None and summaries["ladder_l3d2_prepare"]["stored_runs"] is None
    proposed = [row["problem_id"] for number in (1, 2) for batch in (1, 2) for row in store.read_rows(f"proposals_r{number}_b{batch}.jsonl")]
    assert len(proposed) == 4 and set(proposed) <= set(LOOP_HALF)
    report = summaries["ladder_l4_report"]
    assert report["branch"]["name"] in (NOT_READ, INCONCLUSIVE) and "loop" not in report["models"] and report["rounds"] == [1, 2] and report["label"] == "pretrained on published proofs"
    read = report["measured_and_not_read"] if report["inconclusive"] else report
    assert read["primary"]["mean"] is None and read["beside_the_primary"] is None            # one sampling of G: no fresh attempts for the primary
    assert any("G was attempted 32 times a problem in ONE sampling" in line for line in report["lines"]) and store.is_done(ladder_l4.ARM_REPORT)


# -------------------------------------------------------------------------------------------- registration
def test_the_two_stages_are_registered_with_a_guard_before_every_gpu_step():
    pretrain = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4_pretrain"]]
    assert [(environment, step) for environment, step, _ in pretrain] == [
        ("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l4_pretrain_prepare"), ("guard", None), ("gpu", "ladder_l4_pretrain_train"),
        ("guard", None), ("gpu", "ladder_l4_pretrain_measure"), ("guard", None), ("gpu", "ladder_l4_pretrain_map"), ("guard", None), ("gpu", "ladder_l4_pretrain_report")]
    assert all(options == {} for _, _, options in pretrain)
    assert list(ladder_l4.STEPS) == ["ladder_l4_pretrain_prepare", "ladder_l4_pretrain_train", "ladder_l4_pretrain_measure", "ladder_l4_pretrain_map",
                                     "ladder_l4_pretrain_report", "ladder_l4_prepare", "ladder_l4_report"]
    assert ladder_l4.MAP == "ladder_l4_pretrain_map" == ladder_l2.MAP_STEP and ladder_l2.MAP_FILE == "l4_map_pre.jsonl"       # one name, read by the arm and written by the stage
    steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES["ladder_l4"]]
    gpu_steps = [step for environment, step, _ in steps if environment == "gpu"]
    assert gpu_steps == ["fix_tokenizers", "ladder_l2_prepare", "ladder_l4_prepare", "ladder_l3d2_prepare", "ladder_l2_embed",
                         *(f"ladder_l2_{name}_{number}" for number in range(1, 7) for name in ("round", "train")),
                         "ladder_l3d2_train_without", "ladder_l3d2_measure_with", "ladder_l3d2_measure_without", "ladder_l4_report"]
    assert [(environment, step) for environment, step, _ in steps][:2] == [("sync", "gpu"), ("gpu", "fix_tokenizers")]
    assert all(steps[index - 1][0] == "guard" for index, (environment, _, _) in enumerate(steps) if environment == "gpu" and index > 2)
    assert all(options == {"environment": {ARM: L4_ARM}} for _, _, options in steps) and all(step in STEPS for step in gpu_steps[1:])
    # It is Step 2's stage with `ladder_l4_prepare` after the arm's prepare step and its own report in the place of Step 2's.
    of_step_2 = [step for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l3d2"]) if environment == "gpu"]
    assert [step for step in gpu_steps if step != "ladder_l4_prepare"][:-1] == of_step_2[:-1] and of_step_2[-1] == "ladder_l3d2_report"
    others = set(ladder_round.STEPS) | set(ladder_dose.STEPS) | set(ladder_l2.STEPS) | set(ladder_loop.STEPS) | set(ladder_ceiling.STEPS) | set(ladder_l3d1.STEPS) | set(ladder_l3d2.STEPS)
    assert not set(ladder_l4.STEPS) & others and all("ladder_l4" in name for name in ladder_l4.STEPS)
    for name, of in (("ladder_l4_pretrain_smoke", pretrain), ("ladder_l4_smoke", None)):
        smoke = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES[name]]
        assert all(options == smoke[0][2] for _, _, options in smoke)
        if of:
            assert [(environment, step) for environment, step, _ in smoke] == [(environment, step) for environment, step, _ in of]
    smoke_steps = [step for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l4_smoke"]) if environment == "gpu"]
    assert smoke_steps == [step for step in gpu_steps if not any(step.endswith(f"_{number}") for number in (3, 4, 5, 6))]
    for other in ("ladder_l2", "ladder_l2_t010", "ladder_l2_smoke", "ladder_l3d1", "ladder_l3d2", "ladder_l3d2_smoke", "ladder_ceiling", "ladder_ceiling_smoke"):
        own = entry.child_environment("gpu", "key", entry.step_fields(entry.STAGES[other][2])[2])   # no other stage's task names the arm or this stage's variables
        assert own.get(ARM) != L4_ARM and not [name for name in OWN_VARIABLES if name in own]
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    # The fixture: 12 published proofs of the `pretrain` half, on the side true, one a problem.
    rows = _file_rows(FIXTURE)
    assert len(rows) == 12 == len({row["problem_id"] for row in rows}) and all(half_of(row["problem_id"], 0) == PRETRAIN and row["half"] == "pretrain" for row in rows)
    assert all(row["side"] == "true" and row["predicted_rate"] is None and row["statement"].rstrip().endswith(":= by") and row["proof"].strip() for row in rows)


def test_the_steps_of_both_stages_run_through_the_gpu_entry_point(l4, monkeypatch, tmp_path):
    l4.stored()
    monkeypatch.setattr(milestone2, "load_config", lambda path: l4.config)
    monkeypatch.setattr(pipeline, "use_lean_pin", lambda config: None)
    for stage_name in ("ladder_l4_pretrain", "ladder_l4"):
        if stage_name == "ladder_l4":
            l4.the_arm()
        for environment, name, _ in map(entry.step_fields, entry.STAGES[stage_name]):
            if environment == "gpu" and name in STEPS and not (name.rsplit("_", 1)[-1].isdigit() and int(name.rsplit("_", 1)[-1]) > 2):
                monkeypatch.setattr(sys, "argv", ["rlvr_lean.gpu", name, "--out", str(tmp_path / f"{name}.json")])
                assert gpu_main.main() == 0 and json.loads((tmp_path / f"{name}.json").read_text())["ok"] is True
    assert l4.pretrain_store().is_done(ladder_l4.REPORT) and l4.store().is_done(ladder_l4.ARM_REPORT)
