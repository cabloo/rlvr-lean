"""L4b, the arm again from the larger pretrained model, end to end in the scripted world from a rank-64 start: the start
model resolved by ONE function for L4t and for the arm; `pre_r64`'s own map, made first in a run directory of its own by
`pre`'s map step under its names; the arm with each training rule (every training at the start adapter's rank, every
engine started for it, each saved adapter read back; the rule inside the rounds, nested across them; the twin; the fifth
check with rehearsal rows); the refusals of its prepare steps; its report; the smoke stages; the registration; and the
two arms that existed before it left as they were. Spec: docs/spec/ladder-loop.spec.md, "L4b: the arm again, from
the larger pretrained model". The engine and Lean are scripted; nothing touches a GPU or the network. The rules and the
report on hand-made rows are `test_ladder_l4b.py`."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_assembly import ARM_NAME as BASE_ARM  # noqa: E402
from test_ladder_assembly import arm  # noqa: E402, F401
from test_ladder_l2_stage import ARM, loop  # noqa: E402, F401
from test_ladder_l4_rows_r64 import _trainings, r64  # noqa: E402, F401 - L4's world with `pre`'s run, the arm's from it and the rank check's (`pre_r64`), every adapter on disk
from test_ladder_l4_rows_stage import _all, rows  # noqa: E402, F401
from test_ladder_l4_stage import FIXTURE, _file_rows, _on_the_gpu, l4  # noqa: E402, F401
from test_ladder_round import _run_stage, stage  # noqa: E402, F401

from rlvr_lean.domain.ladder_round.assembly import attempts_alone, training_order, training_set  # noqa: E402
from rlvr_lean.domain.ladder_round.ceiling import training_example  # noqa: E402
from rlvr_lean.domain.ladder_round.l4 import BRANCHES, INCONCLUSIVE, PRETRAIN, half_of  # noqa: E402
from rlvr_lean.domain.ladder_round.l4_rows import rehearsal_rows, uniform  # noqa: E402
from rlvr_lean.domain.ladder_round.l4b import RULES  # noqa: E402
from rlvr_lean.domain.problem_pool.episodes import reward  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import __main__ as gpu_main  # noqa: E402
from rlvr_lean.gpu import ladder_assembly, ladder_ceiling, ladder_l2, ladder_l3d2, ladder_l4, ladder_l4_rank, ladder_l4_rows, ladder_l4_start, ladder_l4b, ladder_loop, ladder_round, pipeline  # noqa: E402
from rlvr_lean.gpu.ladder_ceiling import MORE, REACH  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import stand_in_parameters  # noqa: E402

SAY = "l4 (pretrained on published proofs): "
L4B_ARM, L4_ARM = "t010_assembly_pre_r64", "t010_assembly_pre"
RULE, MINIMUM, START_CHECKS, MAP_RUN = ladder_l2.L2_RULE_VARIABLE, ladder_l2.L2_RULE_MINIMUM_VARIABLE, ladder_l2.L2_START_CHECKS_VARIABLE, ladder_l4_start.MAP_RUN_VARIABLE
STEPS = {**ladder_l2.STEPS, **ladder_l3d2.STEPS, **ladder_l4.STEPS, **ladder_l4_rank.STEPS, **ladder_l4b.STEPS}
ORDER = ["ladder_l4b_map", "ladder_l2_prepare", "ladder_l4b_prepare", "ladder_l3d2_prepare", "ladder_l2_embed", "ladder_l2_round_1", "ladder_l2_train_1", "ladder_l2_round_2",
         "ladder_l2_train_2", "ladder_l3d2_train_without", "ladder_l3d2_measure_with", "ladder_l3d2_measure_without", "ladder_l4b_report"]
RANK_64 = {"rank": 64, "alpha": 128}
# What every training set of an arm WITHOUT a rule holds of a row: what it held before a rule could be said.
ROW_KEYS = {"row", "id", "problem_id", "side", "attempt_id", "theorem", "completion", "verified_attempts", "round", "batch", "origin", "resolved_after_attempt", "minimised"}


def _stage(config, monkeypatch, name, until=None, rounds=2):
    """A stage's GPU steps in its order, the trainings by their GPU path; the steps of the rounds past `rounds` left out."""
    summaries = {}
    for environment, step, _ in map(entry.step_fields, entry.STAGES[name]):
        if environment != "gpu" or step not in STEPS or (step.rsplit("_", 1)[-1].isdigit() and int(step.rsplit("_", 1)[-1]) > rounds):
            continue
        summaries[step] = _on_the_gpu(monkeypatch, STEPS[step], config) if "_train" in step else STEPS[step](config)
        if step == until:
            break
    return summaries


@pytest.fixture
def l4b(r64, monkeypatch):  # noqa: F811
    """The world of a run from `pre_r64` (`pre`'s run, the rank-16 arm's, the rank check's) with a task of L4b's arm:
    `task(rule)` sets what the stage `ladder_l4b_<rule>` sets, two rounds, and the smoke run's minimum for `reward_rows`
    (the fixtures' problems are solved by every attempt or by none)."""
    for name in (RULE, MINIMUM, START_CHECKS, MAP_RUN, ladder_l4_rows.START_VARIABLE, ladder_l4_rows.MINIMUM_VARIABLE):
        monkeypatch.delenv(name, raising=False)

    def task(rule="old", rounds=2, minimum=2):
        options = entry.step_fields(entry.STAGES[f"ladder_l4b_{rule}"][2])[2]
        assert options == {"environment": {ARM: L4B_ARM, RULE: rule}} and all(entry.step_fields(step)[2] == options for step in entry.STAGES[f"ladder_l4b_{rule}"])
        for name, value in options["environment"].items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, str(rounds))
        monkeypatch.setenv(MINIMUM, str(minimum)) if minimum else monkeypatch.delenv(MINIMUM, raising=False)
        r64.l4.the_arm(rounds=rounds, name=L4B_ARM)
        monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (            # what every engine of this task is built with
            r64.kits.append({"adapter": adapter, "max_lora_rank": given["vllm"]["max_lora_rank"], "rank": given["lora"]["rank"], "vllm": given["vllm"], "lora": given["lora"]}),
            (r64.l4.solver, stand_in_parameters))[1]))
        return f"ladder_l4b_{rule}"

    return SimpleNamespace(config=r64.config, r64=r64, calls=r64.calls, kits=r64.kits, task=task, store=lambda: ladder_l3d2._store(r64.config),
                           map_run=lambda: ladder_l4_start.map_run(r64.config, "pre_r64"))


# -------------------------------------------------------------------------------- where a start model lives
def test_the_start_of_an_arm_and_of_l4t_is_resolved_by_one_function(l4b, monkeypatch):
    config, r64_ = l4b.config, l4b.r64
    assert ladder_l4_rows.Start is ladder_l4_start.Start and ladder_l4_rows.config_from is ladder_l4_start.config_from and ladder_l4_rows.start_set is ladder_l4_start.start_set
    # L4t's start (the stage names `pre_r64`) ...
    monkeypatch.setenv(ladder_l4_rows.START_VARIABLE, "pre_r64")
    of_l4t = ladder_l4_rows.the_start(config, ladder_l4_rows.the_settings(config))
    monkeypatch.delenv(ladder_l4_rows.START_VARIABLE)
    # ... and the arm's, by its setting: the same model, in the same run, with the same check, by the same function.
    l4b.task("old")
    assert ladder_l2.assembly_arm(config) == {"target_rate": 0.10, "rounds": 2, "start": "pre_r64", "candidates": "loop_half", "h0": False, "map": "start_model", "rule": "old"}
    of_the_arm = ladder_l2.the_start_of(config)
    assert of_the_arm == of_l4t == ladder_l4_start.the_start(config, "pre_r64") and of_the_arm.check == ladder_l4_rank.check_of(config, "pre_r64")
    assert (of_the_arm.name, of_the_arm.directory, of_the_arm.adapter) == ("pre_r64", r64_.rank, r64_.rank / "adapters" / "pre_r64") and ladder_l2.start_directory(config) == of_the_arm.adapter
    assert (of_the_arm.check.rank, of_the_arm.check.alpha, of_the_arm.prepare, of_the_arm.measure) == (64, 128, "ladder_l4_rank_prepare", "ladder_l4_rank_measure")
    # ITS OWN MAP is in a run directory of its own (the run that made the model is only read), at the pretraining's seed whatever the task's.
    assert (of_the_arm.map_directory, of_the_arm.map_file, of_the_arm.map_step) == (r64_.rank.parent / "ladder_l4_map_pre_r64_seed0", "l4_map_pre_r64.jsonl", "ladder_l4b_map")
    assert ladder_l2.start_map_directory(config) == of_the_arm.map_directory == l4b.map_run() and "the step `ladder_l4b_map` to its end first" in of_the_arm.map_task
    monkeypatch.setenv("RLVR_LEAN_TRAINING_SEEDS", "2")
    assert ladder_l2.the_start_of(config) == of_the_arm                           # a further seed of the arm reads the ONE model and the ONE map
    monkeypatch.delenv("RLVR_LEAN_TRAINING_SEEDS")
    monkeypatch.setenv(MAP_RUN, "ladder_l4b_map_smoke")
    assert ladder_l2.the_start_of(config).map_directory == r64_.rank.parent / "ladder_l4b_map_smoke"
    monkeypatch.delenv(MAP_RUN)
    # The arm from `pre`: the ONE pretraining's run and adapter, its map beside it under the names it always had, the config handed on AS IT IS.
    monkeypatch.delenv(RULE)
    monkeypatch.setenv(ARM, L4_ARM)
    from_pre = ladder_l2.the_start_of(config)
    assert from_pre == ladder_l4_start.the_start(config, "pre") and (from_pre.name, from_pre.check, from_pre.directory, from_pre.adapter) == ("pre", None, r64_.pre, r64_.pre / "adapters" / "pre")
    assert (from_pre.map_directory, from_pre.map_file, from_pre.map_step) == (r64_.pre, "l4_map_pre.jsonl", "ladder_l4_pretrain_map") == (r64_.pre, ladder_l2.MAP_FILE, ladder_l2.MAP_STEP)
    store = ladder_l3d2._store(config)
    assert ladder_l2.config_for(config, store) is config and ladder_l2.config_for(config, store, serving=True) is config and ladder_l2.adapter_saved(config, store.root, "m1") == {}
    assert ladder_l2.rule_of(config) is None and ladder_l2.rule_inputs(config, store, 1) is None and store.root.name == "ladder_l2_t010_assembly_pre_seed0"
    # The base arm has no start at all.
    monkeypatch.setenv(ARM, BASE_ARM)
    monkeypatch.delenv(ladder_l2.L2_ROUNDS_VARIABLE)
    assert ladder_l2.the_start_of(config) is None and ladder_l2.start_directory(config) is None and ladder_l2.rule_of(config) is None
    assert ladder_l2.assembly_arm(config) == {"target_rate": 0.10, "rounds": 6} and ladder_l2.config_for(config, None) is config
    # A start that is neither `pre` nor a model of the check is a configuration error; so is a rule that is none of the three, or one named for an arm that has none.
    arms = config["ladder_loop"]["l2_assembly_arms"]
    monkeypatch.setenv(ARM, L4B_ARM)
    monkeypatch.setitem(arms[L4B_ARM], "start", "pre_r128")
    with pytest.raises(ValueError, match="an arm with assembly says .'target_rate', 'rounds'. and nothing else, but for .*for a start that is a model of the check of the adapter's rank"):
        ladder_l2.assembly_arm(config)
    monkeypatch.setitem(arms[L4B_ARM], "start", "pre_r64")
    monkeypatch.setitem(arms[L4B_ARM], "rule", "newest")
    with pytest.raises(ValueError, match="and for its training rule, one of .'old', 'reward_rows', 'rehearse'."):
        ladder_l2.assembly_arm(config)
    monkeypatch.setitem(arms[L4B_ARM], "rule", "old")
    monkeypatch.setenv(RULE, "newest")
    with pytest.raises(ValueError, match=f"{RULE} names the training rule 'newest' for the arm {L4B_ARM}: a rule is one of"):
        ladder_l2.assembly_arm(config)
    monkeypatch.setenv(ARM, L4_ARM)
    monkeypatch.setenv(RULE, "rehearse")
    with pytest.raises(ValueError, match=f"for the arm {L4_ARM}: that arm states no rule of its own .it is trained as every arm was., and none is given it"):
        ladder_l2.assembly_arm(config)
    # THE RULE IS ONE SETTING: the arm's own, or the one the stage names; a run with another rule than the old one is another run directory.
    monkeypatch.setenv(ARM, L4B_ARM)
    names = {}
    for rule in RULES:
        monkeypatch.setenv(RULE, rule)
        names[rule] = (ladder_l2.rule_of(config), ladder_l2._store(ladder_l2.arm_config(config)).root.name)
    assert names == {"old": ("old", "ladder_l2_t010_assembly_pre_r64_seed0"), "reward_rows": ("reward_rows", "ladder_l2_t010_assembly_pre_r64_reward_rows_seed0"),
                     "rehearse": ("rehearse", "ladder_l2_t010_assembly_pre_r64_rehearse_seed0")}
    monkeypatch.delenv(RULE)
    assert ladder_l2.rule_of(config) == "old" == arms[L4B_ARM]["rule"]            # no variable: the arm's own setting
    gpu = {path.name: path.read_text() for path in (Path(ladder_l2.__file__).parent).glob("*.py")}
    assert sorted(name for name, text in gpu.items() if RULE in text) == ["ladder_l2.py"] and gpu["ladder_l2.py"].count("os.environ.get(L2_RULE_VARIABLE)") == 1
    assert sorted(name for name, text in gpu.items() if '["l4"]["again"]' in text) == ["ladder_l4b.py"] and gpu["ladder_l4b.py"].count('["l4"]["again"]') == 1


# ------------------------------------------------------------------------------------- the stage, each rule
@pytest.mark.parametrize("rule", RULES)
def test_the_arm_from_pre_r64_runs_with_each_rule_every_model_at_rank_64(l4b, monkeypatch, rule):
    config, calls, kits, r64_ = l4b.config, l4b.calls, l4b.kits, l4b.r64
    read_only = {path: _all(path) for path in (r64_.arm, r64_.pre, r64_.rank)}
    name = l4b.task(rule)
    summaries = _stage(config, monkeypatch, name)
    store, map_run, start = l4b.store(), l4b.map_run(), r64_.rank / "adapters" / "pre_r64"
    assert list(summaries) == ORDER and store.root.name == "ladder_l2_t010_assembly_pre_r64_" + ("" if rule == "old" else f"{rule}_") + "seed0"
    assert {path: _all(path) for path in read_only} == read_only                  # `pre`'s run, the rank-16 arm's and the rank check's are only read: file for file what they were

    # ---- THE MAP FIRST, in a run directory of its own: `pre`'s map step under `pre_r64`'s names, the model server started for rank 64
    made = summaries["ladder_l4b_map"]
    assert map_run.name == "ladder_l4_map_pre_r64_seed0" and sorted(path.name for path in map_run.iterdir() if not path.name.startswith("episodes_")) == [
        "l4_map_pre_r64.jsonl", "l4_map_problems.json", "ladder_l4b_map.done.json", "problems.jsonl"]
    assert (made["model"], made["set"], made["problems"], made["episodes_each"], made["sampling_seed"]) == ("pre_r64", "l4_map_pre_r64", 4, 8, 101)
    assert (made["start_run"], made["start_adapter"], made["start_recipe"]) == (str(r64_.rank), str(start), RANK_64)
    of_pre = json.loads((r64_.pre / "ladder_l4_pretrain_map.done.json").read_text())
    assert [key for key in made if key not in of_pre] == ["start_run", "start_adapter", "start_recipe", "read_by", "checks_of_the_start_run_waived_for_a_smoke_run"]
    assert [key for key in of_pre if key not in made] == [] and made["map_of_the_base"] == of_pre["map_of_the_base"] and made["problems_built"]["sides"] == of_pre["problems_built"]["sides"]
    the_map = _file_rows(map_run / "l4_map_pre_r64.jsonl")
    assert [row["problem_id"] for row in the_map] == [row["problem_id"] for row in _file_rows(r64_.pre / "l4_map_pre.jsonl")] and all(row["episodes"] == 8 and row["set"] == "base_map" for row in the_map)
    assert kits[0]["max_lora_rank"] == 64 and kits[0]["rank"] == 64 and {**kits[0]["vllm"], "max_lora_rank": 16} == config["vllm"]

    # ---- the arm's own prepare step: the start, ITS rank as its run recorded it, its own map in the base map's place, the rule
    prepare = summaries["ladder_l2_prepare"]
    assert (prepare["arm"], prepare["start"], prepare["start_adapter"], prepare["start_run"], prepare["rule"], prepare["rule_of_the_arms_own_setting"]) == (
        L4B_ARM, "pre_r64", str(start), str(r64_.rank), rule, "old")
    assert {key: prepare["start_recipe"][key] for key in ("rank", "alpha", "check", "stage", "rank_of_the_config", "alpha_of_the_config")} == {
        **RANK_64, "check": "rank", "stage": "ladder_l4_rank", "rank_of_the_config": 16, "alpha_of_the_config": 32}
    assert (prepare["map"], prepare["map_file"], prepare["candidates"], prepare["h0_rows"]) == ("start_model", str(map_run / "l4_map_pre_r64.jsonl"), "loop_half", 0)
    assert [row for row in ladder_l2._data(ladder_l2.arm_config(config))["base_results"] if row["set"] == "base_map"] == the_map

    # ---- EVERY ENGINE of the arm is started for rank 64 (the map, the two rounds, the two measurements), and round 1 is attempted by `pre_r64`
    # (one engine's kit a sampled set: the map; two batches a round; the rungs and G's two samplings for each measured model)
    assert len(kits) == 1 + 2 + 2 + 3 + 3 and all(kit["max_lora_rank"] == 64 and kit["rank"] == 64 and {**kit["vllm"], "max_lora_rank": 16} == config["vllm"] for kit in kits)
    assert [kit["adapter"] for kit in kits[1:3]] == [("the start adapter", start)] * 2 and not any(kit["adapter"] == ("the start adapter", start) for kit in kits[3:])
    assert (config["lora"]["rank"], config["lora"]["alpha"], config["vllm"]["max_lora_rank"]) == (16, 32, 16)                      # and the config itself is not changed
    assert (summaries["ladder_l2_round_1"]["attempted_by_the_start_adapter"], summaries["ladder_l2_round_2"]["attempted_by_the_start_adapter"]) == (True, False)

    # ---- EVERY TRAINING (M(1), M(2), the twin) attaches the START ADAPTER's rank and alpha, starts from it, and the adapter it saved is read back
    assert [(call["lora"]["rank"], call["lora"]["alpha"], call["more"], call["max_lora_rank"]) for call in calls] == [(64, 128, {"start": start}, 16)] * 3
    assert [list(call["schedule"]["checkpoint_steps"]) for call in calls] == [["m1"], ["m2"], ["without"]]
    trainings = {"m1": summaries["ladder_l2_train_1"], "m2": summaries["ladder_l2_train_2"], "without": summaries["ladder_l3d2_train_without"]}
    for model, summary in trainings.items():
        assert summary["adapter_saved"]["rank"] == 64 and summary["adapter_saved"]["alpha"] == 128 and summary["adapter_saved"]["ranks"] == [64] and summary["peak_reserved_gb"] == 12.64
        assert summary["trained_from"] == f"the stored adapter {start}" and json.loads((store.root / "adapters" / model / "adapter_config.json").read_text())["r"] == 64

    # ---- THE RULE, where each model's set is built from the rows of the rounds so far
    by_round = {number: store.read_rows(f"training_examples_r{number}.jsonl") for number in (1, 2)}
    sets = {number: store.read_rows(ladder_assembly.training_set_file(number)) for number in (1, 2)}
    twin = store.read_rows(ladder_l3d2.TRAINING_WITHOUT_FILE)
    old = {number: training_order(training_set(by_round, number), 0) for number in (1, 2)}
    picks = {row["problem_id"]: row for number in (1, 2) for batch in (1, 2) for row in store.read_rows(f"episodes_round_r{number}_b{batch}_problems.jsonl")}
    for number in (1, 2):
        said = summaries[f"ladder_l2_train_{number}"]["rule"]
        of_the_rounds = [row for row in sets[number] if row["origin"] != "pretraining"]
        assert (said["rule"], said["rows"], said["rows_of_the_rounds"], said["target_rate"]) == (rule, len(sets[number]), len(old[number]), 0.10)
        assert said["rows_by_origin"] == {origin: sum(row["origin"] == origin for row in sets[number]) for origin in ("attempt", "assembled", "pretraining")}
        assert said["kept_of_the_rounds_rows"]["rows"] == len(old[number]) and said["kept_of_the_rounds_rows"]["kept"] == len(of_the_rounds)
        assert sum(entry["rows"] for entry in said["kept_of_the_rounds_rows"]["by_k"].values()) == len(old[number])
        assert [row["id"] for row in of_the_rounds] == [row["id"] for row in old[number] if row["id"] in {own["id"] for own in of_the_rounds}]      # the arm's order, rows left out
    if rule == "old":           # what every arm does: the same rows, the same order, the same fields
        assert all([{key: value for key, value in row.items() if key != "row"} for row in sets[number]] == old[number] and set(sets[number][0]) <= ROW_KEYS for number in (1, 2))
        assert [row["id"] for row in twin] == [row["id"] for row in attempts_alone(sets[2])]
    if rule == "reward_rows":   # a row is kept when its draw is under r(k); kept by the smoke run's minimum where the rule keeps too few, and said so
        for number in (1, 2):
            for row in sets[number]:
                k = max(1, picks[row["problem_id"]]["resolved"])
                assert (row["k"], row["n"], row["draw"]) == (k, 8, uniform(0, "l4_rows_keep", row["id"])) and row["reward"] == round(reward(k, 8, 0.10), 6)
                assert row["kept_by"] == ("the rule" if row["draw"] < reward(k, 8, 0.10) else "the run's minimum")
            by_the_rule = [row["id"] for row in sets[number] if row["kept_by"] == "the rule"]
            assert by_the_rule == [row["id"] for row in old[number] if uniform(0, "l4_rows_keep", row["id"]) < reward(max(1, picks[row["problem_id"]]["resolved"]), 8, 0.10)]
            assert summaries[f"ladder_l2_train_{number}"]["rule"]["kept_by_the_runs_minimum"] == len(sets[number]) - len(by_the_rule)
        assert {row["id"] for row in sets[1] if row["kept_by"] == "the rule"} <= {row["id"] for row in sets[2] if row["kept_by"] == "the rule"}       # kept for M(1): kept for M(2)
        assert [row["id"] for row in twin] == [row["id"] for row in sets[2] if row["origin"] == "attempt"]                              # the twin: the same rule, the assembled rows left out
    if rule == "rehearse":      # all the rounds' rows, and as many rows of the pretraining file as the model has one-shot rows, the first of the same hash order
        fixture = _file_rows(FIXTURE)
        in_order = rehearsal_rows([{"problem_id": row["problem_id"]} for row in fixture], len(fixture), 0)
        for number in (1, 2):
            rehearsal = [row for row in sets[number] if row["origin"] == "pretraining"]
            one_shot = sum(row["origin"] == "attempt" for row in sets[number])
            assert len(rehearsal) == one_shot and (number == 1 or one_shot > 0) and sorted(row["id"] for row in sets[number] if row["origin"] != "pretraining") == sorted(row["id"] for row in old[number])
            assert sorted(row["id"] for row in rehearsal) == sorted(row["id"] for row in in_order[:one_shot])
            # STORED BY ID AND PLACE, never with text; of the `pretrain` half; and what the training was handed is the file's own example for it.
            assert all({"theorem", "completion", "statement", "proof"}.isdisjoint(row) and half_of(row["problem_id"], 0) == PRETRAIN for row in rehearsal)
            assert all(fixture[row["row_of_the_file"]]["problem_id"] == row["problem_id"] and row["id"] == f"{row['problem_id']}#pretraining" and row["k"] is None for row in rehearsal)
            handed = dict(zip([row["id"] for row in sets[number]], calls[number - 1]["examples"]))
            assert all(handed[row["id"]] == training_example(fixture[row["row_of_the_file"]]) for row in rehearsal)
        first = [row["id"] for row in in_order]
        assert {row["id"] for row in sets[1] if row["origin"] == "pretraining"} <= {row["id"] for row in sets[2] if row["origin"] == "pretraining"}
        assert sorted((first.index(row["id"]) for row in sets[2] if row["origin"] == "pretraining")) == list(range(sum(row["origin"] == "pretraining" for row in sets[2])))
        # The twin: the same rule with the assembled rows left out. Its rehearsal rows are the last model's, and it was handed their text too.
        assert [row["id"] for row in twin] == [row["id"] for row in sets[2] if row["origin"] != "assembled"] and summaries["ladder_l3d2_train_without"]["rule"] == {
            "rule": "rehearse", "what": summaries["ladder_l3d2_train_without"]["rule"]["what"], "one_shot_rows": sum(row["origin"] == "attempt" for row in twin),
            "rehearsal_rows": sum(row["origin"] == "pretraining" for row in twin)}
        assert len(calls[2]["examples"]) == len(twin) and not any("completion" in row for row in twin if row["origin"] == "pretraining")
    else:
        assert not any(row["origin"] == "pretraining" for row in (*sets[1], *sets[2], *twin))
    for number in (1, 2):       # what each training was handed is its stored set, row for row
        assert [example["problem_id"] for example in calls[number - 1]["examples"]] == [row["problem_id"] for row in sets[number]]

    # ---- L4b's own prepare step: G' made for `pre_r64` from its first sampling; L4's first two checks read on it; the rank-16 arm's figures beside
    own = summaries["ladder_l4b_prepare"]
    first_sampling = _file_rows(r64_.rank / "episodes_l4_reach_pre_r64_problems.jsonl")
    again = [row["problem_id"] for row in first_sampling if row["resolved"] == 0]
    assert (own["check"], own["arm"], own["rule"], own["start"], own["pretraining_run"], own["start_adapter"], own["start_recipe"]) == (
        "L4b", L4B_ARM, rule, "pre_r64", str(r64_.rank), str(start), prepare["start_recipe"])
    assert [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)] == again and own["goal_set_again"] == len(again) == own["goal_set_again_made_here"]["problems"]
    checks = own["the_two_checks_of_the_pretraining"]
    assert list(checks) == ["the_pretraining_took", "the_goal_set_again_is_large_enough"] and "`pre_r64` solves at least 0 goal problems" in checks["the_pretraining_took"]["what"]
    assert checks["the_pretraining_took"]["checks_of_its_own_report"] and checks["the_pretraining_took"]["passes"] is all(checks["the_pretraining_took"]["checks_of_its_own_report"].values())
    for part in ("rungs", REACH, MORE):
        assert store.read_rows(ladder_ceiling.stored_file("pre_r64", part, "l4")) == _file_rows(r64_.rank / f"episodes_l4_{part}_pre_r64_problems.jsonl")
    assert json.loads(store.path("l4_stored_pre_r64.json").read_text())["distinct_attempts"] == json.loads((r64_.rank / "ladder_l4_rank_measure.done.json").read_text())["distinct_attempts"]
    assert (own["rank_16"]["read"], own["rank_16"]["arm"], own["rank_16"]["run"]) == (True, L4_ARM, str(r64_.arm)) and json.loads(store.path("l4b_rank_16.json").read_text())["read"] is True
    assert not store.path("l4_stored_pre.json").exists() and not [path.name for path in store.root.iterdir() if "stored_pre_" in path.name and "pre_r64" not in path.name]

    # ---- the report: L4's, with `pre_r64` in `pre`'s place, the breadth beside the primary, the rule's kept share, rank 16 beside
    report = summaries["ladder_l4b_report"]
    assert (report["check"], report["start"], report["rank"], report["alpha"], report["rule"], report["label"], report["arm"]) == (
        "L4b", "pre_r64", 64, 128, rule, "pretrained on published proofs", L4B_ARM) and report["branch"]["name"] in BRANCHES
    assert all(line.startswith(SAY) for line in report["lines"]) and f"THE TRAINING RULE is `{rule}`" in report["lines"][0] and "rank 64, alpha 128" in report["lines"][0]
    assert [line[len(SAY):][:7] for line in report["lines"][2:7]] == ["CHECK 1", "CHECK 2", "CHECK 3", "CHECK 4", "CHECK 5"] and "`pre_r64` solves" in report["lines"][2]
    assert not [line for line in report["lines"] if "`pre`" in line and "RANK-16 ARM" not in line and "in `pre`'s place" not in line]
    section = report["measured_and_not_read"] if report["inconclusive"] else report
    assert set(section["secondary"]["goal_problems_solved_by_attempts_alone"]) == {"what", "base", "pre_r64", "with", "without", "loop"}
    breadth = section["breadth_beside_the_primary"]
    with_rows, start_rows = (store.read_rows(f"episodes_l3d2_{part}_with_problems.jsonl") for part in (REACH, MORE)), (store.read_rows(f"l4_stored_pre_r64_{part}.jsonl") for part in (REACH, MORE))
    solved = lambda both: {row["problem_id"] for rows_ in both for row in rows_ if row["resolved"] > 0}      # noqa: E731
    by_with, by_start = solved(with_rows), solved(start_rows)
    assert (breadth["solved_by_with"], breadth["solved_by_the_start"], breadth["gained"], breadth["lost"]) == (len(by_with), len(by_start), len(by_with - by_start), len(by_start - by_with))
    assert breadth["narrower"] is False and set(breadth["share_of_distinct_attempts_on_g"]) == {"with", "pre_r64"} and report["branch"]["narrower"] in (False, None)
    assert section["primary"]["goal_set_again"] == len(again) and section["secondary"]["the_rule"]["rule"] == rule
    assert section["secondary"]["the_rule"]["kept"] == sum(row["origin"] != "pretraining" for row in sets[2]) and section["secondary"]["the_rule"]["rows"] == len(old[2])
    assert all(row["kept_by_the_rule"] == section["secondary"]["the_rule"]["by_round"][str(row["round"])] for row in section["secondary"]["by_round"]["rows"])
    assert section["secondary"]["beside_the_rank_16_arm"]["read"] is True and section["secondary"]["beside_the_rank_16_arm"]["run"] == str(r64_.arm)
    assert section["secondary"]["never_solved"]["given"] is False
    # THE FIFTH CHECK: no row of the seven... here three trainings is barred; the rehearsal rows are of the `pretrain` half and are NAMED, not barred.
    barred = report["can_this_run_see_a_win"]["no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert barred["passes"] is True and barred["barred_problems"] == 0
    if rule == "rehearse":
        assert barred["rehearsal_rows_of_the_pretrain_half"] == {"M(1)": sum(row["origin"] == "pretraining" for row in sets[1]), "M(2)": sum(row["origin"] == "pretraining" for row in sets[2]),
                                                                  "`without`": sum(row["origin"] == "pretraining" for row in twin)} and barred["barred_rehearsal_rows"] == 0
        assert barred["rows"] == {"M(1)": sum(row["origin"] != "pretraining" for row in sets[1]), "M(2)": sum(row["origin"] != "pretraining" for row in sets[2]),
                                  "`without`": sum(row["origin"] != "pretraining" for row in twin)}
    else:
        assert "rehearsal_rows_of_the_pretrain_half" not in barred and barred["rows"] == {"M(1)": len(sets[1]), "M(2)": len(sets[2]), "`without`": len(twin)}
    assert json.loads(store.path(ladder_l4b.REPORT_FILE).read_text()) == json.loads(json.dumps(report, default=str)) and store.is_done(ladder_l4b.REPORT)
    assert not store.path(ladder_l4.ARM_REPORT_FILE).exists() and report["adapters"]["there"] == ["m1", "m2", "without"] and report["adapters"]["start_adapter"] == str(start)

    # ---- a rerun returns what is stored: nothing is sampled, checked or trained again; and the map is not made again
    solver_calls, sent, trained = r64_.l4.solver.calls, len(r64_.l4.lean.sources), len(calls)
    again_run = _stage(config, monkeypatch, name)
    assert (r64_.l4.solver.calls, len(r64_.l4.lean.sources), len(calls)) == (solver_calls, sent, trained)
    assert all(again_run[step] == summaries[step] for step in summaries if step != "ladder_l4b_report")
    # ---- a round's row of the `pretrain` half, or a held-out problem among the rehearsal rows: the fifth check FAILS and the report says INCONCLUSIVE and nothing else
    stored = store.read_rows(ladder_assembly.training_set_file(2))
    store.write_rows(ladder_assembly.training_set_file(2), [{**stored[0], "problem_id": "fixture_c3"} if stored[0]["origin"] != "pretraining" else stored[0],
                                                             *({**row, "problem_id": "fixture_c3"} if index == 0 and stored[0]["origin"] == "pretraining" and row["origin"] != "pretraining" else row
                                                               for index, row in enumerate(stored[1:]))] if stored[0]["origin"] != "pretraining" else [
        *(row for row in stored if row["origin"] == "pretraining"), *({**row, "problem_id": "fixture_c3"} for row in stored if row["origin"] != "pretraining")])
    broken = ladder_l4b.ladder_l4b_report(config)
    assert broken["branch"]["name"] == INCONCLUSIVE and "no_training_row_is_of_the_pretrain_half_or_held_out" in broken["branch"]["failed_checks"] and broken["primary"] is None
    assert "fixture_c3" in broken["can_this_run_see_a_win"]["no_training_row_is_of_the_pretrain_half_or_held_out"]["first"] and half_of("fixture_c3", 0) == PRETRAIN
    assert broken["breadth_beside_the_primary"] is None and "breadth_beside_the_primary" in broken["measured_and_not_read"] and not any("SECONDARY" in line or "WITHOUT OVERFITTING" in line for line in broken["lines"])
    store.write_rows(ladder_assembly.training_set_file(2), stored)
    if rule == "rehearse":
        heldout = store.read_rows(ladder_l3d2.GROUPS_FILE)[0]["problem_id"]
        index = next(position for position, row in enumerate(stored) if row["origin"] == "pretraining")
        for wrong in (heldout, "fixture_c1"):                                   # a held-out problem; a problem of the `loop` half: neither is a rehearsal row
            store.write_rows(ladder_assembly.training_set_file(2), [{**row, "problem_id": wrong} if position == index else row for position, row in enumerate(stored)])
            barred = ladder_l4b.ladder_l4b_report(config)["can_this_run_see_a_win"]["no_training_row_is_of_the_pretrain_half_or_held_out"]
            assert barred["passes"] is False and barred["first"] == [wrong] and barred["barred_rehearsal_rows"] >= 1
        store.write_rows(ladder_assembly.training_set_file(2), stored)
    # ---- the goal problems nothing stored had solved are read from a file GIVEN in the run directory
    goal = [row["problem_id"] for row in first_sampling]
    store.write_rows(ladder_l4b.NEVER_SOLVED_FILE, [{"problem_id": problem_id} for problem_id in (*goal, "elsewhere")])
    given = ladder_l4b.ladder_l4b_report(config)
    never = (given["measured_and_not_read"] if given["inconclusive"] else given)["secondary"]["never_solved"]
    assert (never["given"], never["problems"], never["of_this_runs_goal_set"]) == (True, len(goal) + 1, len(goal)) and set(never["solved"]) == {"pre_r64", "with", "without"}
    assert never["solved"]["with"] == len(by_with) and never["solved"]["pre_r64"] == len(by_start)


def test_the_prepare_steps_refuse_before_anything_is_sampled_and_a_run_keeps_its_rule(l4b, monkeypatch, tmp_path):
    config, r64_ = l4b.config, l4b.r64
    name = l4b.task("rehearse")
    sampled = r64_.l4.solver.calls

    def refused(step, match, error=RuntimeError):
        with pytest.raises(error, match=match):
            step(config)
        assert r64_.l4.solver.calls == sampled and not l4b.store().is_done(ladder_l2.PREPARE) and not l4b.store().is_done(ladder_l4b.PREPARE)

    # THE MAP of the model the arm starts from must be there: the arm's own prepare step reads it before anything else, and names the step that makes it.
    refused(ladder_l2.ladder_l2_prepare, r"ladder_l4_map_pre_r64_seed0 does not hold the map of the model this arm starts from .l4_map_pre_r64.jsonl, and the marker of the step "
                                         r"ladder_l4b_map.\. .* Run the step `ladder_l4b_map` to its end first")
    # The map step itself refuses a run of the check that cannot carry the arm, before anything is sampled.
    report_file = r64_.rank / ladder_l4_rank.REPORT_FILE
    as_written = report_file.read_text()
    report_file.rename(report_file.with_name("elsewhere.json"))
    refused(ladder_l4b.ladder_l4b_map, r"ladder_l4_pretrain_r64_seed0 does not hold \['report_ladder_l4_rank.json'\].* L4b's arm starts from the model that task kept: it reads what "
                                       r"the task of stage `ladder_l4_rank` for seed 0")
    report_file.with_name("elsewhere.json").rename(report_file)
    failing = json.loads(as_written)
    failing["can_this_run_see_a_win"] = {**failing["can_this_run_see_a_win"], "the_training_took": {"passes": False}}
    report_file.write_text(json.dumps(failing))
    refused(ladder_l4b.ladder_l4b_map, "the run ladder_l4_pretrain_r64_seed0 cannot carry L4b's arm: its checks failed .the_training_took.")
    assert not list(l4b.map_run().iterdir())                                      # nothing was written there
    # A SMOKE run's task alone goes on from a start model whose own report's checks failed: the map is made, and says so.
    monkeypatch.setenv(START_CHECKS, "smoke")
    assert ladder_l4b.ladder_l4b_map(config)["checks_of_the_start_run_waived_for_a_smoke_run"] is True
    monkeypatch.delenv(START_CHECKS)
    sampled = r64_.l4.solver.calls
    # ... and the arm's own prepare step refuses that run as the map step did (the arm is not run on it); L4b's own could not be reached.
    refused(ladder_l2.ladder_l2_prepare, "the run ladder_l4_pretrain_r64_seed0 cannot carry the arm: its checks failed .the_training_took.")
    report_file.write_text(as_written)
    # Another rank than the check's in what that run recorded; a map made with another seed: refused, nothing written.
    prepared = r64_.rank / "ladder_l4_rank_prepare.done.json"
    as_prepared = prepared.read_text()
    other = json.loads(as_prepared)
    other["recipe"]["lora"] = {**other["recipe"]["lora"], "rank": 32, "alpha": 64}
    prepared.write_text(json.dumps(other))
    refused(ladder_l2.ladder_l2_prepare, "recorded an adapter of rank 32 and alpha 64 for `pre_r64`; ladder_loop.l4.rank_check gives that model rank 64 and alpha 128")
    prepared.write_text(as_prepared)
    marker = l4b.map_run() / "ladder_l4b_map.done.json"
    as_made = marker.read_text()
    marker.write_text(json.dumps({**json.loads(as_made), "sampling_seed": 5}))
    refused(ladder_l2.ladder_l2_prepare, "l4_map_pre_r64.jsonl cannot stand in the base map's place: it was made with sampling seed 5 and 8 attempts a problem; the stored map's are 101 and 8")
    marker.write_text(as_made)
    # An arm with a rule reads no H0, and `rehearse` needs a start: configuration errors, before anything is written.
    settings = config["ladder_loop"]["l2_assembly_arms"][L4B_ARM]
    monkeypatch.setitem(settings, "h0", True)
    monkeypatch.setenv(ladder_assembly.H0_VARIABLE, str(r64_.l4.harvest))
    refused(ladder_l2.ladder_l2_prepare, f"the arm {L4B_ARM} has a training rule and reads H0")
    monkeypatch.setitem(settings, "h0", False)
    # L4b's own prepare step needs the arm's; then it refuses a start model whose stored rows are not there, and (as on the GPU) a missing adapter.
    with pytest.raises(RuntimeError, match="L4b's own prepare step needs the step ladder_l2_prepare of this run, which is not done: the stage `ladder_l4b`"):
        ladder_l4b.ladder_l4b_prepare(config)
    ladder_l2.ladder_l2_prepare(config)
    sampled = r64_.l4.solver.calls

    def refused_own(match):
        with pytest.raises(RuntimeError, match=match):
            ladder_l4b.ladder_l4b_prepare(config)
        assert r64_.l4.solver.calls == sampled and not l4b.store().is_done(ladder_l4b.PREPARE) and not l4b.store().path(ladder_l4.AGAIN_FILE).exists()

    rows_file = r64_.rank / "episodes_l4_more_pre_r64_problems.jsonl"
    rows_file.rename(rows_file.with_name("elsewhere.jsonl"))
    refused_own(r"ladder_l4_pretrain_r64_seed0 does not hold \['episodes_l4_more_pre_r64_problems.jsonl'\]: `pre_r64`'s per-problem rows on a set it was measured on")
    rows_file.with_name("elsewhere.jsonl").rename(rows_file)
    with monkeypatch.context() as patched:
        patched.setattr(ladder_l4b, "_stand_in", lambda: False)
        (r64_.rank / "adapters" / "pre_r64").rename(r64_.rank / "adapters" / "elsewhere")
        refused_own(r"adapters/pre_r64 is not there: the adapter `pre_r64` is what every model of the arm is trained from and what attempts round 1")
        (r64_.rank / "adapters" / "elsewhere").rename(r64_.rank / "adapters" / "pre_r64")
    # The two checks at the spec's minimums: this small world fails both, and the arm is not run on it; a smoke run's task goes on and the report will say INCONCLUSIVE.
    monkeypatch.setenv(ladder_l4.MINIMUMS_VARIABLE, "150,80")
    refused_own("`pre_r64` .ladder_l4_pretrain_r64_seed0. cannot carry the arm: its checks failed .the_pretraining_took, the_goal_set_again_is_large_enough.")
    monkeypatch.setenv(ladder_l4.MINIMUMS_VARIABLE, "0,0")
    # The rule `rehearse` draws among the rows the START MODEL was pretrained on: the pretraining file must be the one its run recorded.
    changed = tmp_path / "changed" / "pretraining.jsonl"
    changed.parent.mkdir()
    changed.write_text("".join(json.dumps(row) + "\n" for row in _file_rows(FIXTURE)[:-1]))
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(changed))
    refused_own("L4's pretraining file has SHA-256 .* and `pre_r64` was trained on .* .ladder_l4_pretrain_r64_seed0.: the rehearsal rows are rows `pre_r64` was trained on")
    monkeypatch.setenv(ladder_l4.FILE_VARIABLE, str(FIXTURE))
    own = ladder_l4b.ladder_l4b_prepare(config)
    assert own["rule"] == "rehearse" and ladder_l4b.ladder_l4b_prepare(config) == l4b.store().done_summary(ladder_l4b.PREPARE)       # made once
    # A RUN KEEPS ITS RULE: the same run directory named with another rule is refused (another rule is another run directory, by its name).
    monkeypatch.setenv(ladder_l2.L2_RUN_VARIABLE, l4b.store().root.name)
    monkeypatch.setenv(RULE, "reward_rows")
    with pytest.raises(RuntimeError, match="was prepared with the training rule 'rehearse' and this task runs 'reward_rows': a run keeps the rule it began with"):
        ladder_l2.ladder_l2_prepare(config)
    monkeypatch.delenv(ladder_l2.L2_RUN_VARIABLE)
    monkeypatch.setenv(RULE, "rehearse")
    # The steps of this stage are for ITS arm: named by another arm, they are refused.
    monkeypatch.delenv(RULE)
    monkeypatch.setenv(ARM, L4_ARM)
    for step in (ladder_l4b.ladder_l4b_map, ladder_l4b.ladder_l4b_prepare, ladder_l4b.ladder_l4b_report):
        with pytest.raises(RuntimeError, match=f"the stages `ladder_l4b_.` run the arm {L4B_ARM} of ladder_loop.l2_assembly_arms and {ARM} names '{L4_ARM}'"):
            step(config)
    # ... and L4's own two steps refuse L4b's arm: each arm has its own prepare step and report.
    monkeypatch.setenv(ARM, L4B_ARM)
    with pytest.raises(RuntimeError, match=f"the stage `ladder_l4` runs the arm {L4_ARM} of ladder_loop.l2_assembly_arms and {ARM} names '{L4B_ARM}'"):
        ladder_l4.ladder_l4_prepare(config)
    assert name == "ladder_l4b_rehearse"


def test_a_training_whose_saved_adapter_is_not_at_the_start_adapters_rank_fails_its_step_unmarked(l4b, monkeypatch):
    config, r64_ = l4b.config, l4b.r64
    name = l4b.task("old")
    _stage(config, monkeypatch, name, until="ladder_l2_round_1")
    _trainings(monkeypatch, [], saves={"m1": (16, 32, None)})                     # a training the start adapter's rank did not reach
    with pytest.raises(RuntimeError, match="the adapter `m1` saved at .*adapters/m1 has rank 16 and alpha 32 .* the start adapter `pre_r64` is rank 64 and alpha 128"):
        _on_the_gpu(monkeypatch, STEPS["ladder_l2_train_1"], config)
    assert not l4b.store().is_done("ladder_l2_train_1")
    with pytest.raises(RuntimeError, match="round 2 needs the step ladder_l2_train_1 of this run, which is not done"):
        STEPS["ladder_l2_round_2"](config)
    # The setting's check moved after the run was prepared: a training is refused before it attaches anything.
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "rank", 128)
    monkeypatch.setitem(config["ladder_loop"]["l4"]["rank_check"], "alpha", 256)
    monkeypatch.setitem(config["ladder_loop"]["l2_assembly_arms"][L4B_ARM], "start", "pre_r128")
    monkeypatch.setenv(ladder_l4_rank.RUN_VARIABLE, r64_.rank.name)
    with pytest.raises(RuntimeError, match="this run was prepared from `pre_r128` at rank 64 and alpha 128 .what the run that made it recorded., and ladder_loop.l4.rank_check now gives it rank 128"):
        _on_the_gpu(monkeypatch, STEPS["ladder_l2_train_1"], config)


def test_the_goal_set_again_is_the_start_models_made_from_its_first_sampling_and_check_1_reads_its_own_reports_checks(l4b, monkeypatch):
    config, r64_ = l4b.config, l4b.r64
    name = l4b.task("old")
    _stage(config, monkeypatch, name, until="ladder_l2_prepare")
    store, rank_run = l4b.store(), r64_.rank
    goal = [row["problem_id"] for row in store.read_rows("heldout_groups.jsonl") if row["group"] == "goal"]
    stored = {part: _file_rows(rank_run / f"episodes_l4_{part}_pre_r64_problems.jsonl") for part in (REACH, MORE)}

    def prepared_with(first, more):
        """The start model's stored rows with these successes on the goal problem in its first sampling and in its second."""
        for part, resolved in ((REACH, first), (MORE, more)):
            (rank_run / f"episodes_l4_{part}_pre_r64_problems.jsonl").write_text("".join(json.dumps({**row, "resolved": resolved}) + "\n" for row in stored[part]))
        store.path(f"{ladder_l4b.PREPARE}.done.json").unlink(missing_ok=True)
        summary = ladder_l4b.ladder_l4b_prepare(config)
        return summary, [row["problem_id"] for row in store.read_rows(ladder_l4.AGAIN_FILE)]

    # Not solved in its FIRST sampling (whatever its second gave): the problem is of G'. Solved there: it is not, whatever the second gave.
    summary, again = prepared_with(first=0, more=5)
    assert again == goal and len(goal) == 1 and (summary["goal_set_again"], summary["goal_set_again_made_here"]["solved_in_the_first_sampling"]) == (1, 0)
    assert summary["the_two_checks_of_the_pretraining"]["the_pretraining_took"]["goal_problems_solved"] == 1                         # over ALL its attempts
    summary, again = prepared_with(first=3, more=0)
    assert again == [] and (summary["goal_set_again"], summary["goal_set_again_made_here"]["solved_in_the_first_sampling"]) == (0, 1)
    # CHECK 1 is L4's count AND the checks of the start model's own report. A real task is refused a run whose report failed one (the arm's own prepare step, before
    # this one); a SMOKE run's task goes on, and then the check FAILS here, naming the report's check: the arm's report will read INCONCLUSIVE.
    took = summary["the_two_checks_of_the_pretraining"]["the_pretraining_took"]
    assert took["passes"] is True and took["checks_of_its_own_report"] and all(took["checks_of_its_own_report"].values()) and "and the checks of the report of the run that made it pass" in took["what"]
    report_file = rank_run / ladder_l4_rank.REPORT_FILE
    failing = json.loads(report_file.read_text())
    failing["can_this_run_see_a_win"] = {**failing["can_this_run_see_a_win"], "the_training_took": {**failing["can_this_run_see_a_win"]["the_training_took"], "passes": False}}
    report_file.write_text(json.dumps(failing))
    store.path(f"{ladder_l4b.PREPARE}.done.json").unlink()
    with pytest.raises(RuntimeError, match="the run ladder_l4_pretrain_r64_seed0 cannot carry L4b's arm: its checks failed .the_training_took."):
        ladder_l4b.ladder_l4b_prepare(config)
    monkeypatch.setenv(START_CHECKS, "smoke")
    waived = ladder_l4b.ladder_l4b_prepare(config)
    took = waived["the_two_checks_of_the_pretraining"]["the_pretraining_took"]
    assert took["passes"] is False and took["checks_of_its_own_report"]["the_training_took"] is False and took["goal_problems_solved"] >= took["minimum"]
    assert waived["start_run_read"]["checks_waived_for_a_smoke_run"] is True and waived["start_run_read"]["checks_failed"] == ["the_training_took"]


# ------------------------------------------------------------------- the two arms that existed before this one
def test_the_arm_from_pre_and_the_base_arm_state_no_rule_and_their_sets_and_summaries_are_what_they_were(l4b, monkeypatch):
    config, r64_ = l4b.config, l4b.r64
    arms = config["ladder_loop"]["l2_assembly_arms"]
    assert "rule" not in arms[L4_ARM] and "rule" not in arms[BASE_ARM] and arms[L4_ARM]["start"] == "pre"
    # The rank-16 arm's run, made by the code as it is now (the fixture ran its stage): no rule anywhere, the config's own rank everywhere, the old fields in every row.
    arm_run = r64_.arm
    prepared = json.loads((arm_run / "ladder_l2_prepare.done.json").read_text())
    assert not [key for key in prepared if "rule" in key or key in ("start_recipe", "start_run", "start_run_read")] and prepared["start"] == "pre"
    for marker in ("ladder_l2_train_1", "ladder_l2_train_2", "ladder_l3d2_train_without"):
        summary = json.loads((arm_run / f"{marker}.done.json").read_text())
        assert not [key for key in summary if key in ("rule", "adapter_saved", "peak_reserved_gb")]
    for number in (1, 2):
        assert all(set(row) <= ROW_KEYS for row in _file_rows(arm_run / ladder_assembly.training_set_file(number)))
    assert [row["id"] for row in _file_rows(arm_run / ladder_l3d2.TRAINING_WITHOUT_FILE)] == [
        row["id"] for row in _file_rows(arm_run / ladder_assembly.training_set_file(2)) if row["origin"] == "attempt"]
    report = json.loads((arm_run / ladder_l4.ARM_REPORT_FILE).read_text())
    assert not [key for key in ("check", "start", "rule", "rank", "what_it_cannot_say") if key in report] and "breadth_beside_the_primary" not in json.dumps(report)
    assert "rehearsal_rows_of_the_pretrain_half" not in report["can_this_run_see_a_win"]["no_training_row_is_of_the_pretrain_half_or_held_out"]
    assert not (arm_run.parent / "ladder_l4_map_pre_seed0").exists() and (r64_.pre / "l4_map_pre.jsonl").exists()                  # `pre`'s map is where it always was


# ---------------------------------------------------------------------------------------------- the smoke stages
@pytest.mark.parametrize("rule", RULES)
def test_the_smoke_stages_run_from_the_smoke_pre_r64_in_the_smoke_world(arm, monkeypatch, rule):  # noqa: F811
    if not FIXTURE.exists():
        pytest.skip("needs the published-proof fixture, which is not distributed")
    config, calls, kits = arm.config, [], []
    monkeypatch.delenv(ARM)
    for name in (RULE, MINIMUM, START_CHECKS, MAP_RUN, ladder_l4_rank.CHECK_VARIABLE, ladder_l4_rank.RUN_VARIABLE, ladder_l4.RUN_VARIABLE, ladder_l4.MINIMUMS_VARIABLE, ladder_l4.FILE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ladder_round.ROUND_RUN_VARIABLE, "ladder_l1_smoke")
    _run_stage(config)                                                         # what the L1 smoke task leaves on the box
    monkeypatch.delenv(ladder_round.ROUND_RUN_VARIABLE)
    _trainings(monkeypatch, calls)
    # ---- what must be on the box: the pretraining's smoke run, then the rank check's (`ladder_l4_smoke` is NOT needed: the rank-16 arm is read only when it is there)
    for stage_name in ("ladder_l4_pretrain_smoke", "ladder_l4_rank_smoke"):
        options = entry.step_fields(entry.STAGES[stage_name][2])[2]
        with monkeypatch.context() as patched:
            patched.delenv(ladder_loop.DATA_VARIABLE)
            for name, value in options["environment"].items():
                patched.setenv(name, value)
            for environment, step, _ in map(entry.step_fields, entry.STAGES[stage_name]):
                if environment == "gpu" and step in STEPS:
                    _on_the_gpu(patched, STEPS[step], config) if "_train" in step else STEPS[step](config)
    runs = pipeline.STORE / "runs-v4.27"
    pre, rank_run = runs / "ladder_l4_pretrain_smoke", runs / "ladder_l4_rank_smoke"
    as_stored = {path: _all(path) for path in (pre, rank_run)}
    # ---- this stage's smoke task
    stage_name = f"ladder_l4b_{rule}_smoke"
    options = entry.step_fields(entry.STAGES[stage_name][2])[2]
    assert all(entry.step_fields(step)[2] == options for step in entry.STAGES[stage_name])
    of_l4 = entry.step_fields(entry.STAGES["ladder_l4_smoke"][2])[2]["environment"]
    assert options["environment"] == {**of_l4, ladder_l2.L2_RUN_VARIABLE: stage_name, ARM: L4B_ARM, RULE: rule, ladder_l4_rank.RUN_VARIABLE: "ladder_l4_rank_smoke",
                                      MAP_RUN: "ladder_l4b_map_smoke", ladder_l4.MINIMUMS_VARIABLE: "0,0", ladder_l4.FILE_VARIABLE: str(FIXTURE), START_CHECKS: "smoke", MINIMUM: "2"}
    monkeypatch.delenv(ladder_loop.DATA_VARIABLE)
    for name, value in options["environment"].items():
        monkeypatch.setenv(name, value)
    arm.with_lean()
    monkeypatch.setattr(ladder_l2, "_start_request", lambda start: ("the start adapter", start))
    monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda given: (kits.append(given["vllm"]["max_lora_rank"]), (arm.solver, stand_in_parameters))[1]))
    del calls[:]
    summaries = _stage(config, monkeypatch, stage_name)
    store = ladder_l3d2._store(config)
    assert list(summaries) == ORDER and store.root == runs / stage_name and (runs / "ladder_l4b_map_smoke" / "l4_map_pre_r64.jsonl").exists()
    assert {path: _all(path) for path in as_stored} == as_stored                  # the two smoke runs it reads are only read
    assert not [path.name for path in runs.iterdir() if "pre_r64" in path.name]   # no run directory of a real run was made
    assert set(kits) == {64} and len(kits) == 1 + 2 + 2 + 2 + 2 and [(call["lora"]["rank"], call["lora"]["alpha"], call["more"]) for call in calls] == [(64, 128, {"start": rank_run / "adapters" / "pre_r64"})] * 3
    prepare, own, report = summaries["ladder_l2_prepare"], summaries["ladder_l4b_prepare"], summaries["ladder_l4b_report"]
    assert (prepare["rule"], prepare["start_run"], prepare["start_recipe"]["rank"], prepare["rule_minimum_of_a_smoke_run"]) == (rule, str(rank_run), 64, 2)
    assert (own["rank_16"]["read"], own["base_arm_run"], own["start_run_read"]["checks_waived_for_a_smoke_run"]) == (False, None, prepare["start_run_read"]["checks_waived_for_a_smoke_run"])
    assert summaries["ladder_l2_train_2"]["rule"]["rule"] == rule and summaries["ladder_l2_train_2"]["adapter_saved"]["ranks"] == [64]
    assert report["check"] == "L4b" and report["rule"] == rule and report["branch"]["name"] in BRANCHES and all(line.startswith(SAY) for line in report["lines"])
    assert any("L2's stored runs were not read (a smoke run)" in line for line in report["lines"]) and store.is_done(ladder_l4b.REPORT)
    section = report["measured_and_not_read"] if report["inconclusive"] else report
    assert section["secondary"]["beside_the_rank_16_arm"]["read"] is False and section["breadth_beside_the_primary"]["problems"] == 1
    # The map is made ONCE for every rule: the next rule's smoke task finds it done.
    sampled = arm.solver.calls
    assert ladder_l4b.ladder_l4b_map(config) == summaries["ladder_l4b_map"] and arm.solver.calls == sampled


# -------------------------------------------------------------------------------------------- registration
def test_the_stages_run_the_map_first_then_l4s_steps_with_a_guard_before_every_gpu_step():
    of_l4 = [(environment, step) for environment, step, _ in map(entry.step_fields, entry.STAGES["ladder_l4"])]
    for rule in RULES:
        steps = [entry.step_fields(stage_entry) for stage_entry in entry.STAGES[f"ladder_l4b_{rule}"]]
        pairs = [(environment, step) for environment, step, _ in steps]
        assert pairs[:4] == [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l4b_map"), ("guard", None)]
        # ... then L4's own stage, step for step, with this stage's prepare step and report in the place of L4's
        renamed = [(environment, {"ladder_l4_prepare": "ladder_l4b_prepare", "ladder_l4_report": "ladder_l4b_report"}.get(step, step)) for environment, step in of_l4]
        assert [pair for index, pair in enumerate(pairs) if index not in (2, 3)] == renamed
        assert all(steps[index - 1][0] == "guard" for index, (environment, _, _) in enumerate(steps) if environment == "gpu" and index > 2)
        assert all(options == {"environment": {ARM: L4B_ARM, RULE: rule}} for _, _, options in steps)
        gpu_steps = [step for environment, step in pairs if environment == "gpu"]
        assert gpu_steps[1:] == [*ORDER[:5], *(f"ladder_l2_{kind}_{number}" for number in range(1, 7) for kind in ("round", "train")), *ORDER[9:]] and all(step in STEPS for step in gpu_steps[1:])
        smoke = [(environment, step) for environment, step, _ in map(entry.step_fields, entry.STAGES[f"ladder_l4b_{rule}_smoke"])]
        assert [step for environment, step in smoke if environment == "gpu"][1:] == ORDER
    assert sorted(name for name in entry.STAGES if name.startswith("ladder_l4b")) == sorted([*(f"ladder_l4b_{rule}" for rule in RULES), *(f"ladder_l4b_{rule}_smoke" for rule in RULES)])
    assert list(ladder_l4b.STEPS) == ["ladder_l4b_map", "ladder_l4b_prepare", "ladder_l4b_report"]
    others = set(ladder_l2.STEPS) | set(ladder_l3d2.STEPS) | set(ladder_l4.STEPS) | set(ladder_l4_rank.STEPS) | set(ladder_l4_rows.STEPS)
    assert not set(ladder_l4b.STEPS) & others
    # No other stage's task names the rule, L4b's arm or the map's run: the stages that existed are what they were.
    for name, steps in entry.STAGES.items():
        if not name.startswith("ladder_l4b"):
            named = {key for step in steps for key in entry.step_fields(step)[2].get("environment", {})}
            assert not named & {RULE, MINIMUM, START_CHECKS, MAP_RUN} and all(entry.step_fields(step)[2].get("environment", {}).get(ARM) != L4B_ARM for step in steps)


def test_the_steps_run_through_the_gpu_entry_point(l4b, monkeypatch, tmp_path):
    l4b.task("old")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["gpu", "ladder_l4b_map", "--out", str(out)])
    monkeypatch.setattr(gpu_main, "load_config", lambda path: l4b.config, raising=False)
    from rlvr_lean.gpu import milestone2

    monkeypatch.setattr(milestone2, "load_config", lambda path: l4b.config)
    assert gpu_main.main() == 0 and (l4b.map_run() / "l4_map_pre_r64.jsonl").exists()
