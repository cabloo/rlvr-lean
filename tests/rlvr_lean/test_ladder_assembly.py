"""The round with assembly (L3d Step 2), the L2 stage's arm `t010_assembly` end to end on what an L1 run stored: six
rounds, the error positions kept from the round's own checks, the assembly after each batch, k = 1 for a problem only
assembly resolved (the reward, the refit), the training set with its origins and H0's rows, the hashed training order
and the loop's record, the alarm on an assembled proof, the resume, the refusals; and the three-round arms left as they
were. Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches", Step 2. The engine and Lean
are scripted (Lean reads a proof's lemmas and its closing step); nothing touches a GPU or the network."""

import contextlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ladder_l3c_stage as scripted  # noqa: E402 - its Lean reads lemmas, closing steps and a pool's goals
from test_ladder_l2_stage import ARM, _run_loop, loop  # noqa: E402, F401 - the L2 stage on the fixtures
from test_ladder_round import KNOWN_FALSE, _run_stage, stage  # noqa: E402, F401 - the L1 stage on the fixtures, with stand-ins

from rlvr_lean.domain.ladder_round import assembly as rules  # noqa: E402
from rlvr_lean.domain.ladder_round.read import arm_reward  # noqa: E402
from rlvr_lean.domain.problem_pool import SoundnessAlarm  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import rank  # noqa: E402
from rlvr_lean.domain.repair import replay  # noqa: E402
from rlvr_lean.runner import entry  # noqa: E402
from rlvr_lean.gpu import ladder_assembly, ladder_l2, ladder_loop, ladder_round  # noqa: E402
from rlvr_lean.gpu.stand_in_engine import StandInOutput, StandInSample, stand_in_parameters  # noqa: E402
from rlvr_lean.infrastructure.verification_service import CheckSession  # noqa: E402

ARM_NAME = "t010_assembly"
ROUNDS = (1, 2, 3, 4, 5, 6)
UNRELATED = "  have h₃ : fact_c := by good\n  bad_step\n"          # a lemma no closing step needs
FIRST_HALF = "  have h₁ : fact_a := by good\n  bad_step\n"
SECOND_HALF = "  have h₂ : fact_b := by good\n  combine\n"         # `combine` closes the goal once fact_a and fact_b are both above it
NOTHING = "  bad_step\n  done\n"
MINIMISED = "  have h₁_g2 : fact_a := by good\n  have h₂_g3 : fact_b := by good\n  combine\n"
ASSEMBLED = "  have h₃_g1 : fact_c := by good\n" + MINIMISED
# What the solver writes for a problem, by its place among the 8 samples: no attempt of an ASSEMBLES problem verifies, and
# the lemmas of its first three do; a SOLVED problem's first three attempts verify; the others never do.
ASSEMBLES, NEVER = {"fixture_c1", "fixture_c4", "fixture_c7", "fixture_c9"}, {"fixture_c5", "fixture_c12"}
SCRIPTS = {"assembles": [UNRELATED, FIRST_HALF, SECOND_HALF, None, *[NOTHING] * 4], "never": [NOTHING] * 8, "solved": ["  done\n"] * 3 + [NOTHING] * 5}


class ScriptedSolver:
    """n samples for each prompt, by the problem its theorem names and the sample's place; the fourth sample of an
    ASSEMBLES problem runs into the token cap."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompts, parameters, lora_request=None):
        outputs = []
        for prompt in prompts:
            self.calls += 1
            name = re.findall(r"theorem (\S+)", prompt)[-1]
            base = name[len("negation_of_"):] if name.startswith("negation_of_") else name
            script = SCRIPTS["assembles" if base in ASSEMBLES else "never" if base in NEVER else "solved"]
            outputs.append(StandInOutput([StandInSample("  step\n" * 40, list(range(parameters.max_tokens)), "length") if text is None
                                          else StandInSample(text, list(range(max(1, len(text) // 4))), "stop")
                                          for text in (script[index % len(script)] for index in range(parameters.n))]))      # a measurement asks for more than 8
        return outputs


def _h0_row(problem_id, statement=None):
    return {"problem_id": problem_id, "side": "statement", "theorem": statement or f"theorem {problem_id} : P := by\n",
            "completion": "  have k_g1 : fact_a := by good\n  done\n", "kind": "lean_workbook", "published_side": "true", "assembled": True}


@pytest.fixture
def arm(loop, monkeypatch, tmp_path):  # noqa: F811
    """The fixtures' twelve candidates as the whole pool, six rounds of two problems (two batches of one), the
    scripted solver and Lean, and an H0 of three rows: a pool problem no round proposes, one the rounds resolve by an
    attempt, one they never resolve."""
    for name in (ladder_l2.L2_ROUNDS_VARIABLE, ladder_assembly.H0_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ARM, ARM_NAME)
    loop.config["ladder_loop"]["round"].update({"problems": 2, "batches": 2})
    loop.config["ladder_loop"]["episode"]["contradicted_side_audit_share"] = 0.0      # no problem's ruled-out side is audited here: one side a problem
    solver, lean = ScriptedSolver(), scripted.PoolingLean()
    lean.session = lambda: CheckSession(lean)                       # the round's own checker over the scripted pool: it keeps Lean's raw answers
    harvest = tmp_path / "harvest_h0.jsonl"
    harvest.write_text("".join(json.dumps(row) + "\n" for row in (_h0_row("pool_x"), _h0_row("fixture_c2"), _h0_row("fixture_c5"))))
    monkeypatch.setenv(ladder_assembly.H0_VARIABLE, str(harvest))

    def with_lean():
        """L1 ran with its own scripted Lean; from here on the round's checker and the assembly share the pooling one."""
        monkeypatch.setattr(ladder_round.Engines, "kit", lambda self, adapter=None: (lambda config: (solver, stand_in_parameters)))
        monkeypatch.setattr(ladder_loop, "lean_sessions", lambda config: contextlib.nullcontext(lean.session))
        monkeypatch.setattr(ladder_assembly, "lean_pool", lambda config: contextlib.nullcontext(lean))

    return SimpleNamespace(config=loop.config, solver=solver, lean=lean, harvest=harvest, with_lean=with_lean,
                           store=lambda: ladder_l2._store(ladder_l2.arm_config(loop.config)))


def _run_arm(config, rounds=ROUNDS, until=None):
    summaries = {"prepare": ladder_l2.ladder_l2_prepare(config), "embed": ladder_l2.ladder_l2_embed(config)}
    for number in rounds:
        for name, step in (("round", ladder_l2.ladder_l2_round), ("train", ladder_l2.ladder_l2_train)):
            summaries[f"{name}_{number}"] = step(config, number)
            if until == (name, number):
                return summaries
    return summaries


def _started(arm):
    _run_stage(arm.config)
    arm.with_lean()
    return arm.config


# ---------------------------------------------------------------------------------------------- the settings
def test_the_arm_is_the_t010_arm_with_six_rounds_and_the_three_round_arms_are_what_they_were(arm, monkeypatch):
    config = arm.config
    assert config["ladder_loop"]["l2_assembly_arms"][ARM_NAME] == {"target_rate": 0.10, "rounds": 6} and config["ladder_loop"]["l2_arms"] == {"t010": {"target_rate": 0.10}}
    assert ladder_l2.assembly_arm(config) == {"target_rate": 0.10, "rounds": 6} and ladder_l2.rounds_of(config) == ROUNDS
    assert ladder_l2.arm_settings(config) == {"target_rate": 0.10} and ladder_l2.arm_config(config)["ladder_loop"]["challenger"]["target_rate"] == 0.10
    assert arm.store().root.name == "ladder_l2_t010_assembly_seed0"
    seeds = ladder_l2.sampling_seeds(config)
    first = config["ladder_loop"]["round"]["sampling_seed"]
    assert [seeds[f"round_{number}"] - first for number in ROUNDS] == [11, 12, 13, 14, 15, 16] and seeds["control"] - first == 20      # places of their own, after round 3's
    monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, "2")           # a smoke run's own number of rounds
    assert ladder_l2.rounds_of(config) == (1, 2)
    monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, "7")
    with pytest.raises(ValueError, match="would run 7 rounds: an arm runs 1 to 6"):
        ladder_l2.rounds_of(config)
    monkeypatch.delenv(ladder_l2.L2_ROUNDS_VARIABLE)
    config["ladder_loop"]["l2_assembly_arms"][ARM_NAME]["random_share"] = 0.5
    with pytest.raises(ValueError, match="an arm with assembly says .'target_rate', 'rounds'. and nothing else"):
        ladder_l2.assembly_arm(config)
    del config["ladder_loop"]["l2_assembly_arms"][ARM_NAME]["random_share"]
    # The three-round arms: no assembly, the stage's three rounds and seeds, whatever the rounds variable says.
    for name in (None, "t010"):
        monkeypatch.delenv(ARM, raising=False) if name is None else monkeypatch.setenv(ARM, name)
        monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, "2")
        assert ladder_l2.assembly_arm(config) is None and ladder_l2.rounds_of(config) == (1, 2, 3)
        assert sorted(ladder_l2.sampling_seeds(config)) == ["control", "reach", "round_1", "round_2", "round_3", "rungs"]
        monkeypatch.delenv(ladder_l2.L2_ROUNDS_VARIABLE)
    # The stage's own step lists, as they always were: three rounds, each measured, then the control and the report.
    of_a_round = [step for number in (1, 2, 3) for step in (f"ladder_l2_round_{number}", f"ladder_l2_train_{number}", f"ladder_l2_measure_{number}")]
    as_always = ["ladder_l2_prepare", "ladder_l2_embed", *of_a_round, "ladder_l2_control", "ladder_l2_control_trained", "ladder_l2_report"]
    assert list(ladder_l2.STEPS) == as_always
    for name in ("ladder_l2", "ladder_l2_t010", "ladder_l2_smoke"):
        assert [step for environment, step, _ in map(entry.step_fields, entry.STAGES[name]) if environment == "gpu"] == ["fix_tokenizers", *as_always]


# ------------------------------------------------------------------------------------------ the whole arm
def test_six_rounds_with_assembly_after_every_batch_and_what_it_resolves_counts_as_k_1(arm, monkeypatch):
    config = _started(arm)
    seen, refit = [], ladder_l2.refit_observations
    monkeypatch.setattr(ladder_l2, "refit_observations", lambda base, finished, number, decay: (seen.append([dict(row) for row in finished]), refit(base, finished, number, decay))[1])
    summaries = _run_arm(config)
    store = arm.store()
    prepare = summaries["prepare"]
    assert (prepare["arm"], prepare["target_rate"], prepare["rounds"], prepare["assembly"], prepare["h0_rows"]) == (ARM_NAME, 0.10, list(ROUNDS), True, 3)
    assert prepare["h0_file"] == str(arm.harvest) and prepare["candidates_short_by"] == 0 and "ladder_l3d2" in prepare["the_models_are_measured"]
    # No model of this arm is measured round by round, and it has no control: the prepare step wrote no set for them.
    assert not [row for row in store.read_rows("problems.jsonl") if not row["set"].startswith("round_r")]
    batches = [(number, batch) for number in ROUNDS for batch in (1, 2)]
    proposed = [row["problem_id"] for key in batches for row in store.read_rows(f"proposals_r{key[0]}_b{key[1]}.jsonl")]
    assert sorted(proposed) == sorted(f"fixture_c{index}" for index in range(1, 13))               # every candidate once, over the six rounds
    assert [store.done_summary(f"ladder_l2_propose_r{number}_b{batch}")["attempted_by"] for number, batch in batches] == [f"M({number - 1})" for number in ROUNDS for _ in (1, 2)]

    by_problem, rows_of = {}, {}
    for number, batch in batches:
        set_name = f"round_r{number}_b{batch}"
        (result,) = store.read_rows(f"episodes_{set_name}_problems.jsonl")
        attempts = [json.loads(line) for line in store.path(f"episodes_{set_name}_attempts_0000.jsonl").read_text().splitlines()]
        errors = {row["attempt_id"]: row["errors"] for row in store.read_rows(f"episodes_{set_name}_errors_0000.jsonl")}
        # THE ERROR POSITIONS are the round's own: one entry for each attempt Lean rejected, from the answer the checker had.
        assert set(errors) == {attempt["attempt_id"] for attempt in attempts if attempt["status"] == "lean_error"} and all(errors.values())
        assert all({"line", "column", "text"} <= set(found[0]) for found in errors.values())
        assembly, assembled = store.done_summary(f"ladder_l2_assembly_r{number}_b{batch}"), store.read_rows(f"assembly_r{number}_b{batch}.jsonl")
        marker = store.done_summary(f"ladder_l2_batch_r{number}_b{batch}")
        assert marker["assembly"] == assembly and assembly["checks_by_kind"]["rechecks_for_error_positions"] == 0 and "no attempt was checked again" in assembly["error_positions"]
        by_problem[result["problem_id"]], rows_of[result["problem_id"]] = result, assembled
        name = result["problem_id"]
        if name in ASSEMBLES:
            # None of its 8 attempts verified; the lemmas of the first three and the closing step of the third make a proof. It counts as k = 1.
            assert (result["resolved"], result["resolved_by_assembly"]) == (0, True) and assembly["resolved_by_assembly"] == 1 == assembly["unresolved_problems"]
            (row,) = assembled
            side = "negation" if name in KNOWN_FALSE else "statement"
            assert (row["problem_id"], row["side"], row["round"], row["batch"], row["resolved_after_attempt"]) == (name, side, number, batch, 3)
            assert row["assembled_proof"] == ASSEMBLED and row["completion"] == MINIMISED and row["minimised"] is True      # the lemma no step needs went
            assert (row["blocks_before"], row["blocks_after"], row["lines_before"], row["lines_after"], row["minimise_checks"]) == (3, 2, 4, 3, 4)
            assert row["theorem"].startswith(f"theorem {'negation_of_' if side == 'negation' else ''}{name}") and row["pooled_attempts"] == [1, 2, 3]
            assert assembly["attempts_replayed"] == 7 and assembly["sides_replayed"] == 1 and assembly["a_pool_stood_on"] == 1        # the capped attempt is not replayed
            assert assembly["checks_by_kind"]["minimise"] == 4 and assembly["lean_checks"] == sum(assembly["checks_by_kind"][kind] for kind in ("pool", "closer", "minimise"))
            assert marker["reward"] == arm_reward([{**result, "resolved": 1}], 0.10, config["ladder_loop"]["challenger"]["band_reward"]) and marker["reward"]["k_histogram"]["1"] == 1
        else:
            assert result["resolved_by_assembly"] is False and assembled == [] and assembly["resolved_by_assembly"] == 0
            assert (result["resolved"], assembly["unresolved_problems"], assembly["a_pool_stood_on"]) == ((0, 1, 0) if name in NEVER else (3, 0, 0))
            assert marker["reward"] == arm_reward([result], 0.10, config["ladder_loop"]["challenger"]["band_reward"])
    assert sum(result["resolved_by_assembly"] for result in by_problem.values()) == len(ASSEMBLES) == 4
    # THE REFIT: every finished batch is read with k as the challenger reads it. The last batch's refit saw the eleven problems before it.
    last = {row["problem_id"]: row["resolved"] for row in seen[-1]}
    assert len(seen) == 12 and len(last) == 11 and all(last[name] == (1 if name in ASSEMBLES else 0 if name in NEVER else 3) for name in last)
    assert store.done_summary("ladder_l2_propose_r6_b2")["observations"] == 4 + 11                # the base map's four, and those

    # ---- the training set of a round: its own rows by origin, and H0's rows for what the rounds have not resolved
    resolved_so_far, round_of = set(), {name: next(number for (number, batch), problem in zip(batches, proposed) if problem == name) for name in proposed}
    for number in ROUNDS:
        rows = store.read_rows(f"training_examples_r{number}.jsonl")
        own = [row for row in rows if row["origin"] != "h0"]
        mine = [name for name in proposed if round_of[name] == number and name not in NEVER]
        assert sorted(row["problem_id"] for row in own) == sorted(mine) and all(row["round"] == number for row in rows)
        for row in own:
            name = row["problem_id"]
            assert row["origin"] == ("assembled" if name in ASSEMBLES else "attempt")
            if name in ASSEMBLES:
                assert (row["completion"], row["id"], row["verified_attempts"], row["attempt_id"]) == (MINIMISED, f"{name}#assembled#r{number}", 0, None)
                assert row["theorem"] == rows_of[name][0]["theorem"] and row["side"] == rows_of[name][0]["side"]
            else:
                assert "done" in row["completion"] and "bad_step" not in row["completion"] and row["id"] == row["attempt_id"] and row["verified_attempts"] == 3
        resolved_so_far |= {row["problem_id"] for row in own}
        # H0: `pool_x` is no candidate and stays; fixture_c5 is never resolved and stays; fixture_c2 drops out once a round resolved it.
        assert [row["problem_id"] for row in rows if row["origin"] == "h0"] == [name for name in ("pool_x", "fixture_c2", "fixture_c5") if name not in resolved_so_far]
        assert all(row["id"] == f"{row['problem_id']}#h0" and row["completion"] == "  have k_g1 : fact_a := by good\n  done\n" for row in rows if row["origin"] == "h0")
        summary = summaries[f"round_{number}"]
        assert summary["training_rows_by_origin"] == {origin: sum(row["origin"] == origin for row in rows) for origin in ("attempt", "assembled", "h0")}
        assert summary["assembly"]["resolved_by_assembly"] == sum(row["origin"] == "assembled" for row in own)
        assert summary["reward"]["with_a_training_proof"] == len(own)                              # an assembled problem has its proof: k = 1 in the round's reward too

        # ---- M(number): from the base on one proof a problem, in the order of a content hash, with the loop's own record
        trained = store.read_rows(f"training_set_m{number}.jsonl")
        expected = [row for earlier in range(1, number + 1) for row in store.read_rows(f"training_examples_r{earlier}.jsonl") if row["origin"] != "h0"]
        expected += [row for row in rows if row["origin"] == "h0"]
        assert [row["id"] for row in trained] == sorted((row["id"] for row in expected), key=lambda key: rank(0, "l3d2_order", key))
        assert len({row["problem_id"] for row in trained}) == len(trained) and [row["row"] for row in trained] == list(range(len(trained)))
        loss = json.loads(store.path(f"training_loss_m{number}.json").read_text())
        assert loss["rows_trained"] == [row["id"] for row in trained] and len(loss["row_losses"]) == len(trained)
        train = summaries[f"train_{number}"]
        assert (train["rows"], train["model"], train["stand_in_engine"]) == (len(trained), f"M({number})", True) and "made up" in train["note"]
        assert train["rows_by_origin"] == {origin: sum(row["origin"] == origin for row in trained) for origin in ("attempt", "assembled", "h0")}
    final = store.read_rows("training_set_m6.jsonl")
    assert {row["problem_id"] for row in final if row["origin"] == "h0"} == {"pool_x", "fixture_c5"}             # fixture_c2's own proof took its H0 row's place
    assert next(row for row in final if row["problem_id"] == "fixture_c2")["origin"] == "attempt"
    assert [row["id"] for row in rules.attempts_alone(final)] == [row["id"] for row in final if row["origin"] == "attempt"] and len(rules.attempts_alone(final)) == 6
    # A rerun returns what is stored: nothing is sampled, checked or trained again.
    calls, sent = arm.solver.calls, len(arm.lean.sources)
    assert _run_arm(config) == summaries and (arm.solver.calls, len(arm.lean.sources)) == (calls, sent)
    # The stage's own measurements, control and report are not steps of this arm.
    for step, what in ((lambda: ladder_l2.ladder_l2_measure(config, 1), "the measurement of M.1."), (lambda: ladder_l2.ladder_l2_control(config), "the equal-compute control"),
                       (lambda: ladder_l2.ladder_l2_control_trained(config), "extra attempts on G"), (lambda: ladder_l2.ladder_l2_report(config), "the L2 report")):
        with pytest.raises(RuntimeError, match=f"{what} is not a step of the arm t010_assembly: an arm with assembly is run by the stage `ladder_l3d2`"):
            step()


def test_a_three_round_arm_writes_no_file_of_the_assembly_and_reads_k_as_it_did(loop, monkeypatch):  # noqa: F811
    config = loop.config
    _run_stage(config)
    monkeypatch.setenv(ARM, "t010")
    _run_loop(config)
    store = ladder_l2._store(ladder_l2.arm_config(config))
    names = sorted(path.name for path in store.root.iterdir())
    assert store.root.name == "ladder_l2_t010_seed0" and not [name for name in names if "_errors_" in name or "assembly" in name or "training_set" in name or "training_loss" in name]
    assert not any("origin" in row or "id" in row for number in (1, 2, 3) for row in store.read_rows(f"training_examples_r{number}.jsonl"))
    assert not any("resolved_by_assembly" in row for row in store.read_rows("episodes_round_r1_b1_problems.jsonl"))
    assert "assembly" not in store.done_summary("ladder_l2_prepare") and "assembly" not in store.done_summary("ladder_l2_batch_r1_b1")
    assert {row["set"] for row in store.read_rows("problems.jsonl")} >= {"rungs_m1", "reach_m3", "control"}


# -------------------------------------------------------------------------------------------- the refusals
def test_an_assembled_proof_on_the_side_a_published_answer_rules_out_is_the_soundness_alarm(arm, monkeypatch):
    config = _started(arm)
    config["ladder_loop"]["episode"]["contradicted_side"] = "all"                # both sides of every problem are sampled and checked
    monkeypatch.setattr(scripted, "KNOWN_FALSE", KNOWN_FALSE | ASSEMBLES)        # a verifier gone wrong: it takes an assembled proof of a true problem's negation
    ladder_l2.ladder_l2_prepare(config)
    ladder_l2.ladder_l2_embed(config)
    store, raised = arm.store(), None
    for number in ROUNDS:
        try:
            ladder_l2.ladder_l2_round(config, number)
            ladder_l2.ladder_l2_train(config, number)
        except SoundnessAlarm as alarm:
            raised = (number, str(alarm))
            break
    number, message = raised
    name = next(name for name in sorted(ASSEMBLES - KNOWN_FALSE) if name in message)
    assert f"{name} is known true" in message and "an assembled proof verified its negation: a proof of both sides" in message and "combine" in message
    # No attempt verified (the attempts' own alarm did not fire): the assembled proof did. Nothing of the batch's assembly is recorded, and the round is not done.
    batch = next(batch for batch in (1, 2) if store.read_rows(f"proposals_r{number}_b{batch}.jsonl")[0]["problem_id"] == name)
    assert not store.is_done(f"ladder_l2_assembly_r{number}_b{batch}") and not store.is_done(f"ladder_l2_batch_r{number}_b{batch}") and not store.is_done(f"ladder_l2_round_{number}")
    assert store.is_done(f"episodes_round_r{number}_b{batch}") and all(row["resolved"] == 0 for row in store.read_rows(f"episodes_round_r{number}_b{batch}_problems.jsonl"))


def test_a_round_interrupted_before_its_assembly_resumes_there_and_one_whose_errors_were_not_kept_is_refused(arm, monkeypatch):
    config = _started(arm)
    ladder_l2.ladder_l2_prepare(config)
    ladder_l2.ladder_l2_embed(config)
    real = ladder_assembly.lean_pool
    monkeypatch.setattr(ladder_assembly, "lean_pool", lambda config: (_ for _ in ()).throw(RuntimeError("the pool is away")))
    with pytest.raises(RuntimeError, match="the pool is away"):
        ladder_l2.ladder_l2_round(config, 1)
    store = arm.store()
    assert store.is_done("episodes_round_r1_b1") and not store.is_done("ladder_l2_assembly_r1_b1") and not store.is_done("ladder_l2_batch_r1_b1")
    calls = arm.solver.calls
    monkeypatch.setattr(ladder_assembly, "lean_pool", real)
    summary = ladder_l2.ladder_l2_round(config, 1)
    assert arm.solver.calls == calls + 1 and store.is_done("ladder_l2_batch_r1_b1") and summary["problems"] == 2        # only the second batch was sampled: one prompt
    # A check the pool does not take is asked again; one it keeps refusing fails the step with nothing recorded.
    asked = []

    class Refusing:
        def submit_sources(self, sources):
            asked.append(len(sources))
            return SimpleNamespace(result=lambda: [{"id": key, "error": "HTTP 503 from the proxy: the queue timed out"} for key in sources])

    with pytest.raises(ladder_loop.LeanPoolRefused, match="did not take 2 of 2 checks of a batch's assembly, asked again 2 times"):
        ladder_assembly.solver_check(Refusing(), config)({"a": "theorem a : P := by\n  done\n", "b": "theorem b : P := by\n  done\n"})
    assert asked == [2, 2, 2] and ladder_assembly.solver_check(Refusing(), config)({}) == {}
    # A batch whose rejected attempts have no stored errors cannot be assembled: refused, not passed over.
    ladder_l2.ladder_l2_train(config, 1)
    monkeypatch.setattr(ladder_loop, "lean_sessions", lambda config: contextlib.nullcontext(lambda: SimpleNamespace(
        submit=lambda attempts: None, results=lambda: {}, raw=None)))                 # a checker that keeps no answer
    with monkeypatch.context() as patched:
        patched.setattr(ladder_loop, "settle_block", lambda sampled, config, factory, errors=None: (
            [{**attempt, "status": attempt["status"] or "lean_error"} for attempt in sampled.attempts],
            {"problems": len(sampled.problems), "attempts": len(sampled.attempts), "statuses": {}, "generated_tokens": 0, "generation_seconds": 0.0, "sent_to_lean": 0}))
        with pytest.raises(RuntimeError, match="rejected attempts of round_r2_b1 have no stored errors"):
            ladder_l2.ladder_l2_round(config, 2)
    assert not store.is_done("ladder_l2_assembly_r2_b1")


def test_the_arm_refuses_an_h0_it_must_not_train_on_a_round_that_is_not_its_own_and_another_h0_than_it_began_with(arm, monkeypatch):
    config = _started(arm)
    held = json.loads((scripted.PACKAGE / "data" / "ladder_l0_fixture" / "heldout.jsonl").read_text().splitlines()[0])["problem_id"]
    mapped = json.loads((scripted.PACKAGE / "data" / "ladder_l0_fixture" / "base_map.jsonl").read_text().splitlines()[0])["problem_id"]
    for rows, match in (([_h0_row(held)], f"H0 holds 1 held-out problems .first: {held}"), ([_h0_row(mapped)], f"H0 holds 1 problems of the base map .first: {mapped}"),
                        ([_h0_row("pool_x"), _h0_row("pool_x")], "1 problems stand twice in H0"), ([{**_h0_row("pool_x"), "assembled": False}], "1 rows of H0 are not assembled proofs")):
        arm.harvest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError, match=match):
            ladder_l2.ladder_l2_prepare(config)
        assert not arm.store().is_done("ladder_l2_prepare")
    monkeypatch.setenv(ladder_assembly.H0_VARIABLE, str(arm.harvest.with_name("nowhere.jsonl")))
    with pytest.raises(RuntimeError, match="the harvest H0 .* is not in this snapshot"):
        ladder_l2.ladder_l2_prepare(config)
    monkeypatch.setenv(ladder_assembly.H0_VARIABLE, str(arm.harvest))
    arm.harvest.write_text(json.dumps(_h0_row("pool_x")) + "\n")
    ladder_l2.ladder_l2_prepare(config)
    ladder_l2.ladder_l2_embed(config)
    monkeypatch.setenv(ladder_l2.L2_ROUNDS_VARIABLE, "2")
    with pytest.raises(ValueError, match=r"round 3 is not one of the rounds \[1, 2\] this task's arm runs"):
        ladder_l2.ladder_l2_round(config, 3)
    monkeypatch.delenv(ladder_l2.L2_ROUNDS_VARIABLE)
    with pytest.raises(RuntimeError, match="round 2 needs the step ladder_l2_train_1 of this run"):        # the model before it, trained: no measurement between rounds
        ladder_l2.ladder_l2_round(config, 2)
    arm.harvest.write_text(json.dumps(_h0_row("pool_y")) + "\n")                   # another H0 than the run was prepared on
    with pytest.raises(RuntimeError, match="a run keeps the H0 its prepare step checked"):
        ladder_l2.ladder_l2_round(config, 1)


# ------------------------------------------------------------------------------------------------ the rules
def _result(problem_id, resolved, **more):
    return {"problem_id": problem_id, "side": "true", "episodes": 8, "resolved": resolved, **more}


def _attempt(problem_id, episode, status, completion, side="statement"):
    return {"attempt_id": f"{problem_id}#round_r1_b1#{side}#{episode}", "problem_id": problem_id, "side": side, "episode": episode, "status": status, "completion": completion}


def test_k_as_the_challenger_reads_it_is_the_verified_attempts_and_1_for_a_problem_only_assembly_resolved():
    rows = [_result("a", 0, resolved_by_assembly=True), _result("b", 0, resolved_by_assembly=False), _result("c", 3, resolved_by_assembly=False), _result("d", 0)]
    assert [row["resolved"] for row in rules.counted(rows)] == [1, 0, 3, 0] and rules.counted(rows)[1] is rows[1] and rules.counted(rows)[3] is rows[3]
    assert rows[0]["resolved"] == 0                                             # the stored count stays the verified attempts
    marked = rules.with_assembly([_result("a", 0), _result("b", 0), _result("c", 3)], [{"problem_id": "a"}, {"problem_id": "c"}])
    assert [row["resolved_by_assembly"] for row in marked] == [True, False, False]            # a problem an attempt resolved is not assembly's
    assert arm_reward(rules.counted(marked), 0.10, 0.8)["k_histogram"] == {"0": 1, "1": 1, "2": 0, "3": 1, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}


def test_an_episode_to_assemble_is_a_side_of_an_unresolved_problem_with_its_rejected_attempts_in_the_order_drawn():
    problems = [{"problem_id": name, "side": side, "statement": f"theorem {name} : P := by\n", "negation": f"theorem negation_of_{name} : ¬ P := by\n"}
                for name, side in (("a", "true"), ("b", "false"), ("c", "true"))]
    attempts = [_attempt("a", 2, "lean_error", FIRST_HALF), _attempt("a", 0, "lean_error", UNRELATED), _attempt("a", 1, "capped_tokens", "  step\n"),
                _attempt("a", 3, "timeout", "  slow\n"), _attempt("a", 4, "lean_error", "  sorry\n"), _attempt("a", 5, "no_answer", "  x\n"), _attempt("a", 6, "lean_error", ""),
                _attempt("b", 0, "lean_error", NOTHING, "negation"), _attempt("b", 0, "contradicted_side_not_checked", NOTHING), _attempt("c", 0, "lean_error", NOTHING)]
    errors = {attempt["attempt_id"]: [{"line": 5, "column": 2, "text": "failed"}] for attempt in attempts if attempt["status"] == "lean_error"}
    episodes, not_kept = rules.episodes_to_assemble(problems, [_result("a", 0), _result("b", 0), _result("c", 2)], attempts, errors)
    assert not_kept == [] and [(episode["problem_id"], episode["side"], episode["published_side"], episode["attempts"]) for episode in episodes] == [
        ("a", "statement", "true", 7), ("b", "negation", "false", 1), ("b", "statement", "false", 1)]      # c was resolved by an attempt: not replayed
    first = episodes[0]
    assert sorted(first["chain"]) == [1, 3] and first["chain"][3]["completion"] == FIRST_HALF and first["chain"][1]["errors"] == errors["a#round_r1_b1#statement#0"]
    assert first["statement"] == "theorem a : P := by\n" and episodes[1]["statement"] == "theorem negation_of_b : ¬ P := by\n" and episodes[2]["chain"] == {}
    assert (first["pool"], first["closers"], first["rejected"], first["resolved_at"]) == ([], [], set(), None)
    del errors["a#round_r1_b1#statement#2"]
    assert rules.episodes_to_assemble(problems, [_result("a", 0)], attempts, errors)[1] == ["a#round_r1_b1#statement#2"]
    rules.raise_on_the_ruled_out_side(episodes[1], "  done\n")                  # a refutation of a known-false problem is its own side
    with pytest.raises(SoundnessAlarm, match="b is known false .* an assembled proof verified its statement"):
        rules.raise_on_the_ruled_out_side(episodes[2], "  done\n")


def test_the_loop_over_episodes_resolves_each_at_the_attempt_its_closer_verified_after_and_goes_no_further():
    lean, statement = scripted.PoolingLean(), "theorem fixture_c1 (x : ℝ) : 0 ≤ x ^ 2 := by\n"

    def check(sources):
        return {raw["id"]: raw for raw in lean.submit_sources(dict(sources)).result()}

    def episode(*texts):
        chain = {}
        for place, text in enumerate(texts, start=1):
            (answer,) = check({"x": replay.build_proof_source(statement, text)}).values()
            chain[place] = {"completion": text, "errors": replay.rules.errors_of(answer)}
        return {**replay.new_episode(statement), "chain": chain, "resolved_at": None, "assembled_proof": None, "closer": None}

    # The first episode's fourth attempt brings a new lemma: it comes after the episode is resolved, and is not replayed (no pool check for it).
    episodes = [episode(UNRELATED, FIRST_HALF, SECOND_HALF, "  have h₄ : fact_d := by good\n  bad_step\n"), episode(NOTHING, NOTHING), episode(SECOND_HALF, FIRST_HALF)]
    sent, stats, called = len(lean.sources), Counter(), []
    counts = replay.assemble(episodes, check, {"pool_blocks": 12, "kept_closers": 8}, 4, stats, lambda episode, proof: called.append(proof))
    assert [(episode["resolved_at"], episode["assembled_proof"]) for episode in episodes] == [
        (3, ASSEMBLED), (None, None), (2, "  have h₂_g1 : fact_b := by good\n  have h₁_g2 : fact_a := by good\n  combine\n")]
    assert called == [episodes[2]["assembled_proof"], ASSEMBLED] and episodes[0]["closer"]["text"] == "  combine"
    assert counts["pool_checks"] == 5 and len(episodes[0]["pool"]) == 3 and len(lean.sources) - sent == counts["pool_checks"] + counts["closer_checks"] and stats["closer:verified"] == 2
    with pytest.raises(SoundnessAlarm):
        again = [episode(SECOND_HALF, FIRST_HALF)]
        replay.assemble(again, check, {"pool_blocks": 12, "kept_closers": 8}, 2, Counter(), lambda episode, proof: (_ for _ in ()).throw(SoundnessAlarm("both sides")))
    assert again[0]["resolved_at"] is None                                       # the alarm is raised before the resolution is recorded


def test_every_episode_advances_on_its_own_checks_and_an_alarm_is_raised_before_any_other_failure():
    def work(item, stats):
        stats[f"seen:{item % 2}"] += 1
        if item == 3:
            raise RuntimeError("the pool is away")
        if item == 5:
            raise SoundnessAlarm("a proof of both sides")
        return {"pool_checks": item, "closer_checks": 1}

    counts, stats = ladder_assembly.each_on_its_own([1, 2, 4], work)
    assert dict(counts) == {"pool_checks": 7, "closer_checks": 3} and dict(stats) == {"seen:1": 1, "seen:0": 2}
    assert ladder_assembly.each_on_its_own([], work) == (Counter(), Counter())
    with pytest.raises(RuntimeError, match="the pool is away"):
        ladder_assembly.each_on_its_own([1, 3, 4], work)
    with pytest.raises(SoundnessAlarm):                                         # the alarm first, whatever else failed before it
        ladder_assembly.each_on_its_own([1, 3, 5], work)
    # Two episodes side by side resolve as each would alone, and send the files each would alone.
    lean, statement = scripted.PoolingLean(), "theorem fixture_c1 (x : ℝ) : 0 ≤ x ^ 2 := by\n"
    check = ladder_assembly.solver_check(lean, {"ladder_loop": {"episode": {}}})

    def episode(*texts):
        chain = {place: {"completion": text, "errors": replay.rules.errors_of(lean.answer(replay.build_proof_source(statement, text)))} for place, text in enumerate(texts, start=1)}
        return {**replay.new_episode(statement), "chain": chain, "attempts": len(texts), "resolved_at": None, "assembled_proof": None, "closer": None}

    def assemble(episode, own):
        return replay.assemble([episode], check, {"pool_blocks": 12, "kept_closers": 8}, episode["attempts"], own)

    together = [episode(UNRELATED, FIRST_HALF, SECOND_HALF), episode(SECOND_HALF, FIRST_HALF)]
    counts, stats = ladder_assembly.each_on_its_own(together, assemble)
    sent = sorted(lean.sources)
    lean.sources.clear()
    alone = [episode(UNRELATED, FIRST_HALF, SECOND_HALF), episode(SECOND_HALF, FIRST_HALF)]
    one_after_the_other = [assemble(one, Counter()) for one in alone]
    assert [one["resolved_at"] for one in together] == [3, 2] == [one["resolved_at"] for one in alone] and sorted(lean.sources) == sent
    assert dict(counts) == {key: sum(one[key] for one in one_after_the_other) for key in ("pool_checks", "closer_checks")} and stats["closer:verified"] == 2


def test_the_training_set_has_one_proof_a_problem_and_its_order_does_not_move_when_rows_are_left_out():
    example = lambda name, number: {"problem_id": name, "side": "statement", "attempt_id": f"{name}#round_r{number}_b1#statement#3", "theorem": f"theorem {name} : P := by\n",      # noqa: E731
                                    "completion": "  done\n", "verified_attempts": 2}
    assembled = lambda name, number: {"problem_id": name, "side": "statement", "round": number, "batch": 2, "theorem": f"theorem {name} : P := by\n", "completion": MINIMISED,      # noqa: E731
                                      "resolved_after_attempt": 3, "minimised": True}
    harvest = [_h0_row("h_one"), _h0_row("p3"), _h0_row("q2"), _h0_row("h_two")]
    first = rules.round_rows([example("p1", 1), example("p2", 1)], [assembled("q1", 1), assembled("p1", 1)], 1, {"p1": 1, "p2": 2})
    assert [(row["id"], row["origin"], row["batch"]) for row in first] == [("p1#round_r1_b1#statement#3", "attempt", 1), ("p2#round_r1_b1#statement#3", "attempt", 2),
                                                                         ("q1#assembled#r1", "assembled", 2)]      # a problem an attempt resolved has the attempt's proof
    first += rules.h0_rows(harvest, {row["problem_id"] for row in first}, 1)
    second = rules.round_rows([example("p3", 2)], [assembled("q2", 2)], 2, {"p3": 1})
    second += rules.h0_rows(harvest, {row["problem_id"] for row in [*first, *second] if row["origin"] != "h0"}, 2)
    assert [row["problem_id"] for row in first if row["origin"] == "h0"] == ["h_one", "p3", "q2", "h_two"]          # from round 1, all of H0
    assert [row["problem_id"] for row in second if row["origin"] == "h0"] == ["h_one", "h_two"]                    # the rounds' own proofs took two rows' places
    by_round = {1: first, 2: second}
    assert [row["id"] for row in rules.training_set(by_round, 1)] == [row["id"] for row in first]
    whole = rules.training_set(by_round, 2)
    assert sorted(row["id"] for row in whole) == sorted(["p1#round_r1_b1#statement#3", "p2#round_r1_b1#statement#3", "q1#assembled#r1", "p3#round_r2_b1#statement#3",
                                                         "q2#assembled#r2", "h_one#h0", "h_two#h0"])
    assert rules.by_origin(whole) == {"attempt": 3, "assembled": 2, "h0": 2}
    with pytest.raises(ValueError, match="1 problems have two proofs in the training set of round 2 .first: p3"):     # an H0 row that should have dropped out
        rules.training_set({1: first, 2: [*second, *rules.h0_rows([_h0_row("p3")], set(), 2)]}, 2)
    ordered = rules.training_order(whole, 0)
    assert [row["id"] for row in ordered] == sorted((row["id"] for row in whole), key=lambda key: rank(0, "l3d2_order", key)) and rules.training_order(whole, 1) != ordered
    twin = rules.attempts_alone(ordered)
    assert [row["origin"] for row in twin] == ["attempt"] * 3 and twin == rules.training_order(twin, 0)             # the same order with rows left out: none moved
    assert [row["id"] for row in twin] == [row["id"] for row in ordered if row["origin"] == "attempt"]
    assert rules.FROM_ASSEMBLY == ("assembled", "h0") and rules.ORIGINS == ("attempt", "assembled", "h0")
