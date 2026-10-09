"""A model's stored goal attempts as episodes of 8 with assembly (`tools/ladder_goal_assembly.py`), and the reliable
count (`domain/ladder_round/reliable.py`): how stored samplings are cut, what is checked again, how an episode is
resolved, the summary by problem and by length group, several models side by side, the alarm. Spec:
docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches" ("The aim, as a count"; "Measured").
Lean is the scripted one of the L3c stage's tests, handed to the tool: nothing touches the pool or the network."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ladder_l3c_stage import PoolingLean  # noqa: E402 - Lean by the text of a proof: lemmas, closing steps, a pool's goals

from rlvr_lean.domain.ladder_round import reliable  # noqa: E402
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm  # noqa: E402
from rlvr_lean.domain.verification.lean_source import build_proof_source  # noqa: E402
from rlvr_lean.tools import ladder_goal_assembly as tool  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
UNRELATED = "  have h₃ : fact_c := by good\n  bad_step\n"          # a lemma no closing step needs
FIRST_HALF = "  have h₁ : fact_a := by good\n  bad_step\n"
SECOND_HALF = "  have h₂ : fact_b := by good\n  combine\n"         # `combine` closes the goal once fact_a and fact_b are both above it
NOTHING = "  bad_step\n  done\n"
DONE = "  done\n"
FAILED, VERIFIED, CAPPED = "lean_error", "verified", "capped_tokens"


def _problem(name, group, side="true"):
    return {"problem_id": name, "group": group, "side": side, "statement": f"theorem {name} (x : ℝ) : 0 ≤ x ^ 2 := by\n",
            "negation": f"theorem negation_of_{name} : ¬ (∀ (x : ℝ), 0 ≤ x ^ 2) := by\n"}


def _sampling(problem_id, side, attempts):
    """Stored attempt rows of one side, in the order drawn."""
    return [{"problem_id": problem_id, "side": side, "episode": index, "status": status, "completion": completion}
            for index, (status, completion) in enumerate(attempts)]


PROBLEMS = [_problem("goal_1", "goal"), _problem("goal_2", "goal"), _problem("fixture_p3", "goal", "false"), _problem("rung_1", "below")]
REACH = [
    *_sampling("rung_1", "statement", [(VERIFIED, DONE)] * 8),                                     # not a goal problem: never read
    # goal_1: three episodes of 8 and one attempt left over.
    *_sampling("goal_1", "statement", [
        (FAILED, UNRELATED), (FAILED, FIRST_HALF), (FAILED, SECOND_HALF), (CAPPED, "  step\n" * 9), *[(FAILED, NOTHING)] * 4,      # no attempt verifies: assembled after 3
        (FAILED, NOTHING), (VERIFIED, DONE), *[(FAILED, NOTHING)] * 6,                                                             # the second attempt verifies
        (FAILED, FIRST_HALF), (FAILED, SECOND_HALF), (FAILED, NOTHING), (VERIFIED, DONE), *[(FAILED, UNRELATED)] * 4,              # assembled after 2, an attempt at 4
        (VERIFIED, DONE)]),                                                                                                        # left over: no episode
    *_sampling("goal_1", "negation", [(FAILED, "  intro h\n  bad_step\n")] * 8),
    *_sampling("goal_2", "statement", [(FAILED, FIRST_HALF), *[(CAPPED, "  step\n" * 9)] * 6, (FAILED, "  sorry\n")]),             # nothing closes it
    # A problem known false: its negation is the side that can be proved, and assembly proves it.
    *_sampling("fixture_p3", "negation", [(FAILED, SECOND_HALF), (FAILED, FIRST_HALF), *[(FAILED, NOTHING)] * 6]),
    *_sampling("fixture_p3", "statement", [(FAILED, NOTHING)] * 8),                                  # ... and its statement never: the episode is resolved all the same
]
# The rows of a stored file are not in the order drawn: `episode` is.
CONTROL = list(reversed(_sampling("goal_2", "statement", [(FAILED, NOTHING), (VERIFIED, DONE), *[(FAILED, NOTHING)] * 6, (FAILED, NOTHING), (VERIFIED, DONE)])))
LENGTHS = {"goal_1": {"length_group": "4-7"}, "fixture_p3": {"length_group": "1"}}                  # goal_2's shortest published proof is not known


@pytest.fixture
def world(tmp_path, monkeypatch):
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    monkeypatch.setattr(tool, "PROOF_LINES", write(tmp_path / "lines.jsonl", [{"problem_id": problem, **row} for problem, row in LENGTHS.items()]))
    write(tmp_path / "one" / "steps" / "episodes_reach_m_attempts_0000.jsonl", REACH[:20])
    write(tmp_path / "one" / "steps" / "episodes_reach_m_attempts_0001.jsonl", REACH[20:])
    write(tmp_path / "one" / "steps" / "episodes_reach_other_attempts_0000.jsonl", _sampling("goal_2", "statement", [(VERIFIED, DONE)] * 8))      # another set: not named, not read
    write(tmp_path / "two" / "steps" / "episodes_control_m_attempts_0000.jsonl", CONTROL)
    lean, sent = PoolingLean(), []

    def check(sources):
        sent.append(dict(sources))
        return {raw["id"]: raw for raw in lean.submit_sources({name: sources[name] for name in sorted(sources)}).result()} if sources else {}

    arguments = SimpleNamespace(model="m", problems=write(tmp_path / "problems.jsonl", PROBLEMS), out=tmp_path / "out" / "m.json", limit=None, dry_run=False,
                                attempts=[[str(tmp_path / "one" / "steps"), "reach_m"], [str(tmp_path / "two" / "steps"), "control_m"]],
                                api_key_file=None, ca_file=None, in_flight=None)
    return SimpleNamespace(config=copy.deepcopy(CONFIG), arguments=arguments, check=check, lean=lean, sent=sent, write=write, tmp_path=tmp_path)


def _episodes(limit=None):
    return tool.goal_episodes({row["problem_id"]: row for row in PROBLEMS}, [("reach_m", REACH), ("control_m", CONTROL)], limit)


# ---------------------------------------------------------------------------------------- the reliable count
def test_a_sampling_is_cut_into_episodes_of_8_in_the_order_drawn_and_the_rest_is_left_over():
    assert reliable.EPISODE_ATTEMPTS == 8 and (reliable.QUARTER, reliable.HALF, reliable.NINE_TENTHS) == (0.25, 0.5, 0.9)
    flags = [False] * 7 + [True] + [False] * 8 + [True] * 3
    assert reliable.episodes_of(flags) == [True, False] and reliable.episodes_of(flags[:7]) == [] and reliable.episodes_of(flags, 4) == [False, True, False, False]
    # A stored row's place is its `episode`, whatever the file's order; an attempt verified on either side counts.
    rows = [{"problem_id": "a", "side": "negation", "episode": 9, "status": "verified"}, {"problem_id": "b", "side": "statement", "episode": 0, "status": "verified"}]
    rows += [{"problem_id": "a", "side": "statement", "episode": index, "status": "lean_error"} for index in reversed(range(16))]
    found = reliable.attempt_flags(rows)
    assert found == {"a": [False] * 9 + [True] + [False] * 6, "b": [True]}
    # Each sampling is cut on its own: 16 + 8 attempts are three episodes, and 12 + 12 would be two.
    assert reliable.episodes_by_attempts([found, {"a": [True] * 8}], ["a", "b", "c"]) == {"a": [False, True, True], "b": [], "c": []}
    assert reliable.episodes_by_attempts([{"a": [False] * 11 + [True]}, {"a": [False] * 11 + [True]}], ["a"]) == {"a": [False, False]}


def test_a_problem_is_solved_reliably_when_it_is_resolved_in_at_least_half_of_its_episodes():
    episodes = {"never": [False] * 11, "once": [True] + [False] * 10, "quarter": [True] * 3 + [False] * 8, "under_half": [True] * 5 + [False] * 6,
                "half": [True] * 6 + [False] * 5, "nearly": [True] * 9 + [False] * 2, "always": [True] * 10 + [False], "none_stored": []}
    assert reliable.reliability(episodes, list(episodes)) == {"problems": 8, "episodes_a_problem": [0, 11], "episodes": 77, "resolved_episodes": 34,
                                                              "solved_at_least_once": 6, "in_a_quarter": 5, "reliably": 3, "in_nine_tenths": 1}
    # Exactly half is reliable (4 of 8), exactly a quarter counts (2 of 8), and a problem with no episode is counted in none.
    assert reliable.reliability({"a": [True] * 4 + [False] * 4, "b": [True] * 2 + [False] * 6}, ["a", "b", "c"]) == {
        "problems": 3, "episodes_a_problem": [0, 8], "episodes": 16, "resolved_episodes": 6, "solved_at_least_once": 2, "in_a_quarter": 2, "reliably": 1, "in_nine_tenths": 0}


# ------------------------------------------------------------------------------------------------ episodes
def test_the_goal_problems_stored_attempts_are_cut_by_side_and_by_sampling():
    episodes = _episodes()
    assert [(episode["problem_id"], episode["side"], episode["sampling"], episode["episode"], episode["blind_at"]) for episode in episodes] == [
        ("fixture_p3", "negation", "reach_m", 0, None), ("fixture_p3", "statement", "reach_m", 0, None), ("goal_1", "negation", "reach_m", 0, None),
        ("goal_1", "statement", "reach_m", 0, None), ("goal_1", "statement", "reach_m", 1, 2), ("goal_1", "statement", "reach_m", 2, 4),
        ("goal_2", "statement", "reach_m", 0, None), ("goal_2", "statement", "control_m", 0, 2)]                  # the control's 10 attempts are one episode; its ninth and tenth are left over
    assert all(sorted(episode["chain"]) == list(range(1, 9)) for episode in episodes) and not any(episode["problem_id"] == "rung_1" for episode in episodes)
    assert episodes[0]["statement"] == "theorem negation_of_fixture_p3 : ¬ (∀ (x : ℝ), 0 ≤ x ^ 2) := by\n" and episodes[3]["statement"].startswith("theorem goal_1 ")
    assert [episode["chain"][2]["status"] for episode in episodes[-1:]] == ["verified"] and len(_episodes(limit=3)) == 3
    goal_1 = episodes[3:6]                                           # its statement's three episodes
    # Checked again: each failed attempt before its episode's first verified one, a text once; never a capped attempt or one with a forbidden token.
    sources = tool.rechecks(episodes)
    statement = goal_1[0]["statement"]
    assert {build_proof_source(statement, text) for text in (UNRELATED, FIRST_HALF, SECOND_HALF, NOTHING)} <= set(sources.values())
    assert len(sources) == 11 and not any("sorry" in source.split(":= by\n", 1)[1] or "step\n  step" in source for source in sources.values())
    assert "file" in goal_1[1]["chain"][1] and "file" not in goal_1[1]["chain"][2] and "file" not in goal_1[1]["chain"][3]      # the episode ended at its second attempt
    assert "file" not in goal_1[0]["chain"][4] and "file" not in episodes[6]["chain"][8]


# -------------------------------------------------------------------------------------------- the replay
def test_an_episode_is_resolved_by_its_first_verified_attempt_or_earlier_by_an_assembled_proof(world, capsys):
    assert tool.run_replay(world.arguments, world.config, check=world.check) == 0
    output = json.loads(world.arguments.out.read_text(encoding="utf-8"))
    rows = {(row["problem_id"], row["side"], row["sampling"], row["episode"]): row for row in output["episodes"]}
    assert len(rows) == 8 == len(output["episodes"])
    how = {key: (row["blind_at"], row["resolved_at"], row["how"]) for key, row in rows.items()}
    assert how == {("fixture_p3", "negation", "reach_m", 0): (None, 2, "assembled"),          # the two halves in the other order: `combine` was kept, then the pool held both
                   ("fixture_p3", "statement", "reach_m", 0): (None, None, None),              # the side that cannot be proved
                   ("goal_1", "negation", "reach_m", 0): (None, None, None),
                   ("goal_1", "statement", "reach_m", 0): (None, 3, "assembled"),              # no attempt of the 8 verified
                   ("goal_1", "statement", "reach_m", 1): (2, 2, "attempt"),
                   ("goal_1", "statement", "reach_m", 2): (4, 2, "assembled"),                 # assembled two attempts before one verified
                   ("goal_2", "statement", "reach_m", 0): (None, None, None),
                   ("goal_2", "statement", "control_m", 0): (2, 2, "attempt")}
    proof = rows["goal_1", "statement", "reach_m", 0]
    assert proof["assembled_proof"] == "  have h₃_g1 : fact_c := by good\n  have h₁_g2 : fact_a := by good\n  have h₂_g3 : fact_b := by good\n  combine\n" and proof["pool_blocks"] == 3
    assert rows["goal_1", "statement", "reach_m", 2]["assembled_proof"] == "  have h₁_g1 : fact_a := by good\n  have h₂_g2 : fact_b := by good\n  combine\n"
    assert rows["fixture_p3", "negation", "reach_m", 0]["assembled_proof"] == "  have h₂_g1 : fact_b := by good\n  have h₁_g2 : fact_a := by good\n  combine\n"
    assert rows["goal_1", "statement", "reach_m", 1]["assembled_proof"] is None and rows["goal_1", "statement", "reach_m", 1]["pool_blocks"] == 0
    assert output["model"] == "m" and output["sizes"] == {"pool_blocks": 12, "kept_closers": 8} and [source["set"] for source in output["sources"]] == ["reach_m", "control_m"]
    # 37 failed attempts stood before their episode's end, in 11 distinct files; three closers verified after a pool.
    assert output["stats"]["recheck:lean_error"] == 37 and output["stats"]["closer:verified"] == 3 and output["stats"]["pool_checks"] == 8
    # The first batch is the re-checks; nothing an episode held after it was resolved is sent.
    assert len(world.sent[0]) == 11 and not any("fact_c" in source and "negation_of" in source for batch in world.sent for source in batch.values())
    summary = output["summary"]
    assert (summary["episodes_of_one_side"], summary["resolved_by_attempts_alone"], summary["resolved_with_assembly"], summary["resolved_by_an_assembled_proof"]) == (8, 3, 5, 3)
    assert summary["how"] == {"None": 3, "assembled": 3, "attempt": 2} and summary["assembled_proof_lines"] == {"median": 3, "longest": 4}
    alone, assembly = summary["goal_problems"]["by_attempts_alone"], summary["goal_problems"]["with_assembly"]
    # goal_1: 2 of its 3 episodes by attempts, 3 with assembly (its negation's episode is the same episode: either side). goal_2: 1 of 2. fixture_p3: 0 of 1,
    # then 1: its negation is assembled, and the statement's 8 failed attempts are the same episode.
    assert alone["all"] == {"problems": 3, "episodes_a_problem": [1, 2, 3], "episodes": 6, "resolved_episodes": 3, "solved_at_least_once": 2, "in_a_quarter": 2,
                            "reliably": 2, "in_nine_tenths": 0}
    assert assembly["all"] == {"problems": 3, "episodes_a_problem": [1, 2, 3], "episodes": 6, "resolved_episodes": 5, "solved_at_least_once": 3, "in_a_quarter": 3,
                               "reliably": 3, "in_nine_tenths": 2}
    assert list(alone) == ["all", "1", "2-3", "4-7", "8+", "not_known", "4_or_more"]
    assert (alone["4_or_more"]["problems"], alone["4_or_more"]["reliably"], assembly["4-7"]["in_nine_tenths"], alone["1"]["solved_at_least_once"], assembly["1"]["reliably"]) == (1, 1, 1, 0, 1)
    assert alone["not_known"]["problems"] == 1 and alone["8+"] == {"problems": 0, "episodes_a_problem": [], "episodes": 0, "resolved_episodes": 0, "solved_at_least_once": 0,
                                                                   "in_a_quarter": 0, "reliably": 0, "in_nine_tenths": 0}
    printed = capsys.readouterr().out
    assert "m: goal problems 3 | episodes of 8 attempts 8 | resolved by an attempt: 3" in printed and "re-checks for error positions: 11" in printed
    assert "within 8: with assembly 5 | attempts alone 3" in printed and "within 1: with assembly 0 | attempts alone 0" in printed
    assert "m: 3 problems, [1, 2, 3] episodes a problem; episodes resolved 3 -> 5 of 8 (3 by an assembled proof)" in printed
    assert "      all: 2 -> 3 / 2 -> 3 / 2 -> 3 / 0 -> 2   of 3" in printed and "4_or_more: 1 -> 1 / 1 -> 1 / 1 -> 1 / 0 -> 1   of 1" in printed


def test_several_models_are_printed_side_by_side_and_a_dry_run_sends_nothing(world, capsys):
    assert tool.run_replay(world.arguments, world.config, check=world.check) == 0
    other = json.loads(world.arguments.out.read_text(encoding="utf-8"))
    other["model"] = "another"
    (world.tmp_path / "out" / "another.json").write_text(json.dumps(other), encoding="utf-8")
    capsys.readouterr()
    assert tool.main(["report", str(world.arguments.out), str(world.tmp_path / "out" / "another.json")]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("goal problems solved at least once / in a quarter / in half (reliably) / in nine tenths of their episodes of 8: by attempts alone -> with assembly")
    assert [line.split(":")[0] for line in lines if not line.startswith(" ")][1:] == ["m", "another"] and len(lines) == 1 + 2 * 8
    # A dry run reads, cuts and counts.
    world.sent.clear()
    command = ["replay", "--model", "m", "--problems", str(world.arguments.problems), "--attempts", *world.arguments.attempts[0], "--attempts", *world.arguments.attempts[1]]
    assert tool.main([*command, "--dry-run", "--limit", "5"]) == 0 and world.sent == []
    printed = capsys.readouterr().out
    assert "m: goal problems 2 | episodes of 8 attempts 5 | resolved by an attempt: 1" in printed and "re-checks to send first; nothing was sent and nothing was written" in printed
    with pytest.raises(SystemExit):                                  # without --dry-run it wants a file to write and the pool's key and certificate
        tool.main([*command, "--out", str(world.tmp_path / "x.json")])
    # A set that is not stored where it was said to be is refused before anything is sent.
    world.arguments.attempts[1][1] = "control_m3"
    with pytest.raises(ValueError, match="holds no stored attempts of the set control_m3"):
        tool.run_replay(world.arguments, world.config, check=world.check)
    assert world.sent == []


def test_an_assembled_proof_on_the_side_the_published_answer_rules_out_stops_the_replay(world, monkeypatch, capsys):
    problems = [{**row, "side": "true"} if row["problem_id"] == "fixture_p3" else row for row in PROBLEMS]      # a published answer that says the statement is TRUE
    world.write(world.arguments.problems, problems)
    with pytest.raises(SoundnessAlarm, match="fixture_p3 is known true .* an assembled proof verified its negation"):
        tool.run_replay(world.arguments, world.config, check=world.check)
    assert not world.arguments.out.exists()
    # ... and so does a stored failure that verifies when it is checked again there.
    world.write(world.arguments.problems, PROBLEMS)
    world.lean.alarm = "goal_1"                                      # a verifier gone wrong: it accepts any proof of `negation_of_goal_1`
    with pytest.raises(SoundnessAlarm, match="goal_1 is known true .* a stored attempt, checked again, verified its negation"):
        tool.run_replay(world.arguments, world.config, check=world.check)
    monkeypatch.setattr(tool, "run_replay", lambda arguments, config: (_ for _ in ()).throw(SoundnessAlarm("a proof of both sides")))
    command = ["replay", "--model", "m", "--problems", str(world.arguments.problems), "--attempts", *world.arguments.attempts[0], "--dry-run"]
    assert tool.main(command) == tool.SOUNDNESS_ALARM_EXIT == 3 and "SOUNDNESS ALARM" in capsys.readouterr().err
