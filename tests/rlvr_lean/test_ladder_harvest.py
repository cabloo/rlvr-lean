"""The harvest H0 (`tools/ladder_harvest.py`, `domain/repair/replay.py`): which stored attempts are read and in what
order, which problems and sides are replayed, the episode's Lean-only part over them, the minimisation of an assembled
proof, the rows written, the alarm. Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches"
("The harvest H0", "A harvested proof is minimised"). Lean is the scripted one of the L3c stage's tests, handed to the
tool: nothing touches the pool or the network."""

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3c_stage import PoolingLean  # noqa: E402 - Lean by the text of a proof: lemmas, closing steps, a pool's goals

from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm  # noqa: E402
from rlvr_lean.domain.repair import accumulate as pooling  # noqa: E402
from rlvr_lean.domain.repair import replay  # noqa: E402
from rlvr_lean.domain.repair.alternate import checked_text  # noqa: E402
from rlvr_lean.domain.verification.lean_source import build_proof_source  # noqa: E402
from rlvr_lean.infrastructure import verification_service  # noqa: E402
from rlvr_lean.infrastructure.kimina_client import BACKGROUND_PRIORITY  # noqa: E402
from rlvr_lean.tools import ladder_harvest as tool  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
UNRELATED = "  have h₃ : fact_c := by good\n  bad_step\n"          # a lemma no closing step needs
FIRST_HALF = "  have h₁ : fact_a := by good\n  bad_step\n"
SECOND_HALF = "  have h₂ : fact_b := by good\n  combine\n"         # `combine` closes the goal once fact_a and fact_b are both above it
NOTHING_TO_KEEP = "  bad_step\n  done\n"
ASSEMBLED = "  have h₃_g1 : fact_c := by good\n  have h₁_g2 : fact_a := by good\n  have h₂_g3 : fact_b := by good\n  combine\n"
MINIMISED = "  have h₁_g2 : fact_a := by good\n  have h₂_g3 : fact_b := by good\n  combine\n"


def _statement(problem_id):
    return f"theorem {problem_id} (x : ℝ) : 0 ≤ x ^ 2 := by\n"


def _attempt(problem_id, status, completion, side="statement", episode=0):
    return {"attempt_id": f"{problem_id}#round#{side}#{episode}", "problem_id": problem_id, "side": side, "episode": episode, "completion": completion,
            "status": status, "token_count": 20, "finish_reason": "stop", "seconds": 0.5, "first_error": ""}


POOL = [{"problem_id": name, "kind": kind, "side": side, "statement": _statement(name)}
        for name, kind, side in (("pool_a", "lean_workbook", "true"), ("pool_b", "stp_conjecture", "true"), ("pool_c", "stp_conjecture", "false"),
                                 ("pool_solved", "lean_workbook", "true"), ("pool_untried", "lean_workbook", "true"), ("pool_d", "lean_workbook", "true"),
                                 ("held_1", "lean_workbook", "true"))]      # a held-out problem: were it among the candidates, it is still never read
STORED = {
    # One run, pulled twice: its second pull is the same file again and is read once.
    "ladder_l2_seed0_r1/steps/episodes_round_r1_b1_attempts_0000.jsonl": [
        _attempt("pool_a", "lean_error", UNRELATED), _attempt("pool_a", "lean_error", FIRST_HALF, episode=1), _attempt("pool_a", "capped_tokens", "  step\n" * 9, episode=2),
        _attempt("pool_a", "lean_error", FIRST_HALF.rstrip() + "   \n\n", episode=3), _attempt("pool_a", "lean_error", "  sorry\n", episode=4),
        _attempt("pool_solved", "verified", "  done\n"), _attempt("pool_solved", "lean_error", FIRST_HALF, episode=1),
        _attempt("pool_b", "lean_error", NOTHING_TO_KEEP), _attempt("pool_b", "lean_error", "  intro h\n  bad_step\n", side="negation"),
        _attempt("pool_c", "timeout", "  slow_step\n", side="negation"),
        _attempt("pool_d", "lean_error", "  done\n")],                 # stored as a failure; Lean verifies it when it is checked again
    "ladder_l2_seed0_r2/steps/episodes_round_r1_b1_attempts_0000.jsonl": [_attempt("pool_a", "lean_error", "  never_read\n  bad_step\n")],
    # Another run: its attempts come after the first run's, whatever a file is called.
    "ladder_l2_t010_seed0_r1/steps/episodes_round_r1_b1_attempts_0000.jsonl": [
        _attempt("pool_a", "lean_error", SECOND_HALF), _attempt("pool_c", "lean_error", "  have k : fact_a := by good\n  bad_step\n", side="negation")],
    # Held-out problems, the base map's, and what is not a ladder run or is a smoke or a pilot run: never read.
    "ladder_l1_seed0_r1/steps/episodes_reach_base_attempts_0000.jsonl": [_attempt("held_1", "lean_error", FIRST_HALF), _attempt("pool_a", "lean_error", "  after_a_held_out_row\n")],
    "ladder_l0b_r1/steps/episodes_base_map_attempts_0000.jsonl": [_attempt("base_map_1", "verified", "  done\n"), _attempt("pool_untried", "verified", "  done\n")],
    "ladder_l1_smoke_r1/steps/episodes_round_challenger_attempts_0000.jsonl": [_attempt("pool_a", "verified", "  done\n")],
    "ladder_l3c_pilot_seed0_r1/steps/episodes_round_r1_b1_attempts_0000.jsonl": [_attempt("pool_a", "verified", "  done\n")],
    "ladder_ceiling_seed0_r1/steps/episodes_ceiling_rungs_full_attempts_0000.jsonl": [_attempt("pool_a", "verified", "  done\n")],
    "phase_b_seed0/steps/episodes_round_r1_b1_attempts_0000.jsonl": [_attempt("pool_a", "verified", "  done\n")],
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    monkeypatch.setattr(tool, "POOL_CANDIDATES", write(tmp_path / "data" / "candidates.jsonl", POOL))
    monkeypatch.setattr(tool, "HELDOUT", write(tmp_path / "data" / "heldout.jsonl", [{"problem_id": "held_1"}]))
    for name, rows in STORED.items():
        write(tmp_path / "runs" / name, rows)
    lean = PoolingLean()
    sent = []

    def check(sources):
        sent.append(dict(sources))
        return {raw["id"]: raw for raw in lean.submit_sources({name: sources[name] for name in sorted(sources)}).result()} if sources else {}

    arguments = SimpleNamespace(runs_root=tmp_path / "runs", out=tmp_path / "out" / "harvest_h0.jsonl", limit=None, attempts=None, dry_run=False,
                                api_key_file=None, ca_file=None, in_flight=None)
    return SimpleNamespace(config=copy.deepcopy(CONFIG), arguments=arguments, check=check, lean=lean, sent=sent, write=write, tmp_path=tmp_path)


def _read(world_):
    statements = {row["problem_id"]: row["statement"] for row in POOL}
    return (*tool.stored_attempts(world_.arguments.runs_root, statements, {"held_1"}), statements)


# --------------------------------------------------------------------------------------- what is read
def test_the_sizes_are_l3cs_and_the_harvest_takes_at_most_48_attempts_a_side():
    ladder = CONFIG["ladder_loop"]
    assert ladder["l3d"]["harvest"] == {"attempts": 48} and (ladder["accumulate"]["pool_blocks"], ladder["accumulate"]["kept_closers"]) == (12, 8)
    assert ladder["episode"]["lean_seconds"] == 30 and tool.STORED_TASKS == "ladder_l" and tool.LEFT_OUT == ("smoke", "pilot")


def test_the_stored_training_side_attempts_are_read_a_run_once_and_a_held_out_problem_never(world):
    stored, solved, tasks, _ = _read(world)
    assert tasks == ["ladder_l0b_r1", "ladder_l1_seed0_r1", "ladder_l2_seed0_r1", "ladder_l2_t010_seed0_r1"]      # no smoke, pilot, ceiling or Phase B run; the second pull adds nothing
    assert solved == {"pool_solved"}                                # `pool_untried` stood after a base-map row: its file was left there
    assert set(stored) == {("pool_a", "statement"), ("pool_b", "statement"), ("pool_b", "negation"), ("pool_c", "negation"), ("pool_d", "statement"), ("pool_solved", "statement")}
    # Only `lean_error` rows are kept, each with its run, file and row; nothing after a held-out row, nothing of the second pull.
    assert [(run, row) for run, _, row, _ in stored["pool_a", "statement"]] == [("ladder_l2_seed0", 0), ("ladder_l2_seed0", 1), ("ladder_l2_seed0", 3), ("ladder_l2_seed0", 4),
                                                                                  ("ladder_l2_t010_seed0", 0)]
    assert not any("never_read" in text or "after_a_held_out_row" in text for rows in stored.values() for _, _, _, text in rows)
    assert not any(problem == "held_1" for problem, _ in stored) and [text for _, _, _, text in stored["pool_c", "negation"]] == ["  have k : fact_a := by good\n  bad_step\n"]


def test_a_side_is_its_distinct_failed_attempts_in_the_order_run_file_row(world):
    stored, solved, _, statements = _read(world)
    episodes = tool.episodes_of(stored, solved, statements, 48)
    assert [(episode["problem_id"], episode["side"]) for episode in episodes] == [("pool_a", "statement"), ("pool_b", "negation"), ("pool_b", "statement"), ("pool_c", "negation"),
                                                                                 ("pool_d", "statement")]
    first = episodes[0]
    # A text once (as it stands in the checked file), none with a forbidden token; the other run's attempt comes last.
    assert first["chain"] == {1: {"completion": UNRELATED, "from": "ladder_l2_seed0"}, 2: {"completion": FIRST_HALF, "from": "ladder_l2_seed0"},
                              3: {"completion": SECOND_HALF, "from": "ladder_l2_t010_seed0"}}
    assert first["stored_attempts"] == 5 and first["statement"] == _statement("pool_a") and (first["pool"], first["closers"], first["rejected"]) == ([], [], set())
    assert episodes[1]["statement"].startswith("theorem negation_of_pool_b : ¬ (∀ (x : ℝ), 0 ≤ x ^ 2) := by")      # the exact negation, built as the pool builds it
    assert list(tool.episodes_of(stored, solved, statements, 2)[0]["chain"]) == [1, 2]                              # at most so many a side
    assert [episode["problem_id"] for episode in tool.episodes_of(stored, solved, statements, 48, limit=2)] == ["pool_a", "pool_b"]


# ------------------------------------------------------------------------------------- the whole harvest
def test_the_harvest_assembles_a_proof_no_attempt_wrote_minimises_it_and_writes_a_training_example(world, capsys):
    assert tool.build(world.arguments, world.config, check=world.check) == 0
    rows = [json.loads(line) for line in world.arguments.out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    # A training example as a round stores one, and where it came from.
    assert (row["problem_id"], row["side"], row["theorem"], row["completion"]) == ("pool_a", "statement", _statement("pool_a"), MINIMISED)
    assert (row["kind"], row["published_side"], row["assembled"]) == ("lean_workbook", "true", True)
    assert (row["resolved_after_attempt"], row["attempts_replayed"], row["stored_attempts"]) == (3, 3, 5)
    assert row["pooled_attempts"] == [1, 2, 3] and row["runs"] == ["ladder_l2_seed0", "ladder_l2_t010_seed0"]
    # Minimised: the lemma no step needs went, the two the closing step needs stayed; four checks (one a block, and the text once more).
    assert (row["blocks_before"], row["blocks_after"], row["lines_before"], row["lines_after"], row["minimise_checks"], row["minimised"]) == (3, 2, 4, 3, 4, True)
    # The first batch is the re-check of every attempt replayed, each as an attempt's own file; no attempt's text is the proof.
    assert set(world.sent[0].values()) == {build_proof_source(episode_statement, text) for episode_statement, text in (
        (_statement("pool_a"), UNRELATED), (_statement("pool_a"), FIRST_HALF), (_statement("pool_a"), SECOND_HALF), (_statement("pool_b"), NOTHING_TO_KEEP),
        ("theorem negation_of_pool_b : ¬ (∀ (x : ℝ), 0 ≤ x ^ 2) := by\n", "  intro h\n  bad_step\n"),
        ("theorem negation_of_pool_c : ¬ (∀ (x : ℝ), 0 ≤ x ^ 2) := by\n", "  have k : fact_a := by good\n  bad_step\n"), (_statement("pool_d"), "  done\n"))}
    assert all(name == replay.name_of(source) for batch in world.sent for name, source in batch.items())
    assert build_proof_source(_statement("pool_a"), ASSEMBLED) in [source for batch in world.sent for source in batch.values()]
    summary = json.loads(tool.summary_path(world.arguments.out).read_text())
    assert summary["rows"] == 1 == summary["resolved_by_an_assembled_proof"] and summary["problems_no_stored_attempt_verified"] == 4 and summary["sides_replayed"] == 5
    assert summary["attempts_replayed"] == 7 and summary["task_directories_read"] == ["ladder_l0b_r1", "ladder_l1_seed0_r1", "ladder_l2_seed0_r1", "ladder_l2_t010_seed0_r1"]
    assert summary["minimisation"] == {"blocks_before": 3, "blocks_after": 2, "proofs_made_shorter": 1, "proofs_used_as_assembled": 0, "checks": 4}
    assert summary["check_statuses"]["recheck:lean_error"] == 6 and summary["check_statuses"]["closer:verified"] == 1 and summary["check_statuses"]["closer:own_proof"] >= 1
    # A stored failure that verifies on the re-check is counted apart: it is no assembled proof and is not written.
    assert summary["a_stored_failure_verified_on_the_recheck"]["problem_ids"] == ["pool_d"] and summary["check_statuses"]["recheck:verified"] == 1
    assert summary["a_stored_failure_verified_on_the_recheck"]["problems"] == 1 and summary["timeouts"] == {} and summary["sizes"] == {"pool_blocks": 12, "kept_closers": 8}
    assert not tool.partial_path(world.arguments.out).exists()                # what was kept as it went is gone once the harvest is written
    printed = capsys.readouterr().out
    assert "pool problems no stored attempt verified: 4 | sides to replay: 5 | distinct failed attempts used: 7 (at most 48 a side)" in printed
    assert "re-checks for error positions: 7" in printed and "after attempt 1: problems resolved by an assembled proof 0" in printed
    assert "assembly resolves 1 | a stored failure verified on the re-check: 1" in printed and "assembled proofs to minimise: 1 (3 pool blocks)" in printed


def test_what_is_resolved_so_far_is_kept_while_the_harvest_runs_and_removed_when_it_is_written(world):
    world.arguments.attempts, seen = 3, []                           # the last attempt replayed is one after which the count is kept
    partial = tool.partial_path(world.arguments.out)

    def check(sources):
        seen.append(json.loads(partial.read_text()) if partial.exists() else None)
        return world.check(sources)

    assert tool.build(world.arguments, world.config, check=check) == 0
    kept = [value for value in seen if value is not None]
    assert seen[:3] == [None, None, None] and kept and kept[-1]["attempts_done"] == 3 and kept[-1]["problems"] == 4
    assert [(row["problem_id"], row["resolved_at"]) for row in kept[-1]["resolved"]] == [("pool_a", 3), ("pool_d", -1)]
    assert kept[-1]["resolved"][0]["assembled_proof"] == ASSEMBLED and not partial.exists() and world.arguments.out.exists()
    assert json.loads(tool.summary_path(world.arguments.out).read_text())["attempts_a_side_at_most"] == 3


def test_the_checks_go_to_the_pins_pool_as_background_work_with_an_episodes_lean_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(verification_service, "resolve_lan_host", lambda name, server: ("192.0.2.9", name))
    key, certificate = tmp_path / "key", tmp_path / "ca.crt"
    key.write_text("pool-key\n")
    certificate.write_text("not read here")
    settings = tool.lean_check_settings(argparse.Namespace(api_key_file=key, ca_file=certificate, in_flight=24), copy.deepcopy(CONFIG))
    pool = CONFIG["lean"]["pool"]
    assert settings.base_url == f"https://192.0.2.9:{pool['port']}" and settings.tls_server_name == pool["name"] and settings.api_key == "pool-key"
    assert (settings.priority, settings.lean_timeout_seconds, settings.concurrent_requests) == (BACKGROUND_PRIORITY, 30, 24) and pool["lean_timeout_seconds"] != 30
    # One client answers a batch: the files go in the order of their names, and each answer is found by its name.
    lean = PoolingLean()
    check = tool.pool_check(lean)
    sources = {name: build_proof_source(_statement("pool_a"), text) for name, text in (("b", "  done\n"), ("a", "  bad_step\n"))}
    answers = check(sources)
    assert lean.sources == [sources["a"], sources["b"]] and replay.judged("b", answers["b"]).is_verified and not replay.judged("a", answers["a"]).is_verified
    assert check({}) == {} and len(lean.sources) == 2


def test_a_dry_run_reads_and_counts_and_sends_nothing(world, capsys):
    world.arguments.dry_run = True
    assert tool.build(world.arguments, world.config, check=world.check) == 0
    assert world.sent == [] and not world.arguments.out.parent.exists()
    printed = capsys.readouterr().out
    assert "sides to replay: 5" in printed and "dry run: 4 task directories read, 7 re-checks to send first; nothing was sent and nothing was written" in printed
    assert tool.main(["--runs-root", str(world.arguments.runs_root), "--dry-run", "--limit", "1", "--attempts", "2"]) == 0
    assert "sides to replay: 1 | distinct failed attempts used: 2 (at most 2 a side)" in capsys.readouterr().out
    with pytest.raises(SystemExit):                                  # without --dry-run it wants a file to write and the pool's key and certificate
        tool.main(["--runs-root", str(world.arguments.runs_root), "--out", str(world.arguments.out)])


def test_a_proof_on_the_side_the_published_answer_rules_out_stops_the_harvest(world, monkeypatch, capsys):
    world.lean.alarm = "pool_b"                                      # a verifier gone wrong: it accepts a proof of `negation_of_pool_b`, and pool_b is known true
    with pytest.raises(SoundnessAlarm, match="pool_b is known true .* verified its negation"):
        tool.build(world.arguments, world.config, check=world.check)
    assert not world.arguments.out.exists() and json.loads(tool.partial_path(world.arguments.out).read_text())["soundness_alarm"] is True
    # Through the command line the tool exits with the alarm's own code.
    monkeypatch.setattr(tool, "build", lambda arguments, config: (_ for _ in ()).throw(SoundnessAlarm("a proof of both sides")))
    assert tool.main(["--runs-root", str(world.arguments.runs_root), "--dry-run"]) == tool.SOUNDNESS_ALARM_EXIT == 3 and "SOUNDNESS ALARM" in capsys.readouterr().err
    tool.raise_on_the_ruled_out_side({"problem_id": "pool_c", "side": "negation"}, "false", "an assembled proof")      # a refutation of a known-false problem is its own side
    with pytest.raises(SoundnessAlarm):
        tool.raise_on_the_ruled_out_side({"problem_id": "pool_c", "side": "statement"}, "false", "an assembled proof")


# ------------------------------------------------------------------------------------------ minimisation
def _blocks(*facts):
    return [{"text": f"  have h{index}_g{index} : {fact} := by good", "name": f"h{index}_g{index}", "statement": f": {fact}", "generation": index}
            for index, fact in enumerate(facts, start=1)]


def _lean(calls=None, failing=()):
    lean = PoolingLean()

    def check(sources):
        if calls is not None:
            calls.append(sorted(sources.values()))
        return {name: ({"error": "Lean REPL command timed out in 30 seconds"} if source in failing else lean.answer(source)) for name, source in sources.items()}
    return check


def test_a_block_the_proof_needs_stays_and_one_it_does_not_need_goes_from_the_last_to_the_first():
    statement, closer = _statement("pool_a"), {"text": "  combine", "needs": [], "generation": 5}
    blocks = _blocks("fact_x", "fact_a", "fact_y", "fact_b", "fact_z")
    proofs = [{"statement": statement, "blocks": blocks, "closer": closer, "assembled": replay.without_a_block(blocks, range(5), None, closer)}]
    calls, stats = [], Counter()
    replay.minimise(proofs, _lean(calls), stats)
    (proof,) = proofs
    assert proof["kept"] == [1, 3] and proof["minimised"] is True and proof["minimise_checks"] == 6          # one check a block, and the text that is left once more
    assert proof["completion"] == "  have h2_g2 : fact_a := by good\n  have h4_g4 : fact_b := by good\n  combine\n"      # names as the pool made them
    # The last block first, each time from what is left: fact_z goes, fact_b stays, fact_y goes, fact_a stays, fact_x goes.
    tried = [source.split(":= by\n", 1)[1].split("\n#print")[0].count("have") for (source,) in calls]
    assert tried == [4, 3, 3, 2, 2, 2] and dict(stats) == {"minimise:verified": 3, "minimise:lean_error": 2, "minimised_text:verified": 1}
    # Two proofs go together, a place from the end at a time; a pool of one block and a proof with none are handled.
    short = {"statement": statement, "blocks": _blocks("fact_x"), "closer": {"text": "  done", "needs": [], "generation": 2}, "assembled": "  have h1_g1 : fact_x := by good\n  done\n"}
    bare = {"statement": statement, "blocks": [], "closer": {"text": "  done", "needs": [], "generation": 1}, "assembled": "  done\n"}
    again = [{**proofs[0], "blocks": blocks}, short, bare]
    calls.clear()
    replay.minimise(again, _lean(calls), Counter())
    # The last batch is each proof's text once more: `short` and `bare` come to one text, which is one file, read for both.
    assert [len(batch) for batch in calls] == [2, 1, 1, 1, 1, 2] and short["completion"] == "  done\n" == bare["completion"] and short["kept"] == []
    assert (short["minimise_checks"], bare["minimise_checks"], short["minimised"], bare["minimised"]) == (2, 1, True, True) and again[0]["kept"] == [1, 3]


def test_of_two_lemmas_that_would_each_do_the_later_one_goes():
    statement, closer = _statement("pool_a"), {"text": "  combine", "needs": [], "generation": 4}
    blocks = _blocks("fact_a", "fact_b", "fact_b")                  # the pool holds a statement once, but a proof may come to it with either
    proofs = [{"statement": statement, "blocks": blocks, "closer": closer, "assembled": replay.without_a_block(blocks, range(3), None, closer)}]
    replay.minimise(proofs, _lean(), Counter())
    assert proofs[0]["kept"] == [0, 1] and proofs[0]["completion"] == "  have h1_g1 : fact_a := by good\n  have h2_g2 : fact_b := by good\n  combine\n"
    assert replay.without_a_block(blocks, [0, 1, 2], 1, closer) == "  have h1_g1 : fact_a := by good\n  have h3_g3 : fact_b := by good\n  combine\n"


def test_a_minimised_text_that_fails_its_last_check_is_not_used_the_proof_as_assembled_is():
    statement, closer = _statement("pool_a"), {"text": "  combine", "needs": [], "generation": 3}
    blocks = _blocks("fact_x", "fact_a", "fact_b")
    assembled = replay.without_a_block(blocks, range(3), None, closer)
    minimised = build_proof_source(statement, "  have h2_g2 : fact_a := by good\n  have h3_g3 : fact_b := by good\n  combine\n")
    calls = []

    def flaky(sources):                         # Lean verifies the shorter text when a block is taken out, and times out when it is checked once more
        calls.append(len(sources))
        return _lean(failing=(minimised,) if len(calls) == 4 else ())(sources)

    proofs = [{"statement": statement, "blocks": blocks, "closer": closer, "assembled": assembled}]
    stats = Counter()
    replay.minimise(proofs, flaky, stats)
    assert proofs[0]["minimised"] is False and proofs[0]["completion"] == assembled and proofs[0]["kept"] == [0, 1, 2] and proofs[0]["minimise_checks"] == 4
    assert stats["minimised_text:timeout"] == 1 and stats["minimise:verified"] == 1


def test_one_proof_a_problem_and_two_would_be_a_proof_of_both_sides():
    def proof(side):
        episode = {"problem_id": "pool_a", "side": side, "statement": _statement("pool_a"), "resolved_at": 2, "stored_attempts": 4,
                   "chain": {1: {"completion": "a", "from": "ladder_l2_seed0"}, 2: {"completion": "b", "from": "ladder_l2_seed1"}}}
        return {"episode": episode, "blocks": _blocks("fact_a"), "closer": {"text": "  done", "needs": [], "generation": 2}, "assembled": "  have h1_g1 : fact_a := by good\n  done\n",
                "completion": "  done\n", "kept": [], "minimise_checks": 2, "minimised": True}

    kinds = {"pool_a": ("lean_workbook", "true")}
    (row,) = tool.harvest_rows([proof("statement")], kinds)
    assert list(row)[:4] == ["problem_id", "side", "theorem", "completion"] and row["runs"] == ["ladder_l2_seed0", "ladder_l2_seed1"] and row["lines_before"] == 2
    with pytest.raises(SoundnessAlarm, match="an assembled proof on both sides"):
        tool.harvest_rows([proof("statement"), proof("negation")], kinds)


# ------------------------------------------------------------------------------- one attempt, between checks
def test_a_closer_is_checked_after_the_pool_unless_lean_has_rejected_that_text_in_the_episode():
    statement, (block,) = _statement("pool_a"), _blocks("fact_a")
    closer = {"text": "  combine", "needs": [], "generation": 2}
    proof = pooling.assembled_proof([block], closer)

    def after_an_attempt(found, rejected=(), closers=()):
        episode = {**replay.new_episode(statement), "pool": [block], "closers": list(closers), "rejected": set(rejected)}
        after, stats = {"found": found, "pool": None, "pool_now": [block]}, Counter()
        return episode, after, replay.closers_to_send(episode, after, {}, 8, stats), dict(stats)

    # A closing step just kept is tried after the pool as it stands.
    episode, after, sources, stats = after_an_attempt(pooling.Harvest(closer=closer))
    assert sources == {replay.name_of(build_proof_source(statement, proof)): build_proof_source(statement, proof)} and stats == {}
    assert episode["closers"] == [closer] and after["checks"] == [(replay.name_of(build_proof_source(statement, proof)), proof, closer)]
    # ... but not when that text is one Lean rejected already (a known copy), holds a forbidden token, or is the failed proof itself under its pooled names.
    assert after_an_attempt(pooling.Harvest(closer=closer), rejected={checked_text(proof)})[2:] == ({}, {"closer:known_copy": 1})
    assert after_an_attempt(pooling.Harvest(closer={**closer, "text": "  sorry"}))[2:] == ({}, {"closer:forbidden": 1})
    episode, _, sources, stats = after_an_attempt(pooling.Harvest(closer=closer, own_proof=True))
    assert (sources, stats) == ({}, {"closer:own_proof": 1}) and checked_text(proof) in episode["rejected"] and episode["closers"] == [closer]
    # A closing step the episode holds already, or one past its eight, is not kept again and nothing is checked.
    assert after_an_attempt(pooling.Harvest(closer=closer), closers=[closer])[2:] == ({}, {})
    full = [{"text": f"  step_{index}", "needs": [], "generation": 1} for index in range(8)]
    episode, _, sources, _ = after_an_attempt(pooling.Harvest(closer=closer), closers=full)
    assert sources == {} and episode["closers"] == full


def test_the_first_closer_that_verifies_resolves_the_episode_and_the_others_are_rejected_texts():
    episode, stats = replay.new_episode(_statement("pool_a")), Counter()
    closers = [{"text": f"  step_{index}", "needs": [], "generation": index} for index in range(3)]
    after = {"checks": [("n0", "  a\n", closers[0]), ("n1", "  b\n", closers[1]), ("n2", "  c\n", closers[2])]}
    proved = {"response": {"messages": [{"severity": "info", "data": "'t' does not depend on any axioms"}]}}
    failed = {"response": {"messages": [{"severity": "error", "data": "unsolved goals"}]}}
    assert replay.first_verified(episode, after, {"n0": failed, "n1": proved, "n2": proved}, stats) == ("  b\n", closers[1])
    assert episode["rejected"] == {checked_text("  a\n")} and dict(stats) == {"closer:lean_error": 1, "closer:verified": 2}
    assert replay.first_verified(episode, after, {"n0": failed, "n1": failed, "n2": {"error": "Lean REPL command timed out in 30 seconds"}}, stats) is None
    assert episode["rejected"] == {checked_text(text) for text in ("  a\n", "  b\n", "  c\n")} and stats["closer:timeout"] == 1
    # An answer is read as a solver's attempt is at the ladder loop's pin: a warning that says "failed" is not an error there.
    warned = {"response": {"messages": [{"severity": "warning", "data": "aesop: failed to prove the goal after exhaustive search"}, *proved["response"]["messages"]]}}
    assert replay.judged("x", warned).is_verified and not replay.judged("x", failed).is_verified
    source = "theorem t : True := by\n  trivial\n"
    assert replay.name_of(source) == hashlib.sha256(source.encode()).hexdigest()[:24]      # a file's name is a hash of its text, as L3c names a pool's file
