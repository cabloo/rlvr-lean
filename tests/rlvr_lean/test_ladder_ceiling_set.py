"""The tool that builds the ceiling's training file (`tools/ladder_ceiling_set.py`): who is eligible, who is chosen and
in what order, which published proofs are tried (the character rule among them), the file each is checked as, the
proof kept, the rows and their order. Spec: docs/spec/ladder-loop.spec.md, "The ceiling: a labelled diagnostic",
"The training file". Lean is a stand-in handed to the tool: nothing touches the pool or the network."""

import argparse
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rlvr_lean.domain.verification.lean_source import build_proof_source, imports_first
from rlvr_lean.infrastructure import verification_service
from rlvr_lean.infrastructure.kimina_client import BACKGROUND_PRIORITY
from rlvr_lean.tools import ladder_ceiling_set as tool

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
FIELDS = ["problem_id", "kind", "side", "statement", "proof", "proof_lines", "predicted_rate", "certificate_source"]
VERIFIED = {"response": {"messages": [{"severity": "info", "data": "'t' depends on axioms: [propext, Classical.choice, Quot.sound]"}]}}
VERIFIED_WITH_A_WARNING = {"response": {"messages": [{"severity": "warning", "data": "aesop: failed to prove the goal after exhaustive search"},
                                                     {"severity": "info", "data": "'t' does not depend on any axioms"}]}}
FAILED = {"response": {"messages": [{"severity": "error", "data": "unsolved goals"}]}}
TIMED_OUT = {"error": "Lean REPL command timed out in 60 seconds"}


def _statement(problem_id):
    return f"theorem {problem_id} (x : ℝ) : 0 ≤ x ^ 2 := by\n"


def _pool_row(problem_id, kind, side="true"):
    return {"problem_id": problem_id, "kind": kind, "side": side, "statement": _statement(problem_id), "rewritten": False, "certificate_source": "stp",
            "published_proof_chars": 20, "stp_round": None}


POOL = [_pool_row("wb_3", "lean_workbook"), _pool_row("stp_c", "stp_conjecture"), _pool_row("wb_1", "lean_workbook"), _pool_row("wb_2", "lean_workbook"),
        _pool_row("wb_refuted", "lean_workbook", side="false"), _pool_row("wb_easy", "lean_workbook"), _pool_row("wb_unscored", "lean_workbook"),
        *(_pool_row(f"stp_{letter}", "stp_conjecture") for letter in "abdef")]
SCORES = [{"problem_id": row["problem_id"], "predicted_rate": 0.3 if row["problem_id"] == "wb_easy" else 0.01, "expected_reward": 0.1}
          for row in POOL if row["problem_id"] != "wb_unscored"]
ONE_LINE, TWO_LINES, THREE_LINES = "  nlinarith [sq_nonneg x]\n", "  have h := sq_nonneg x\n  exact h\n", "  -- a comment is no line\n  intro\n  simp\n  done"
OTHER_TWO_LINES, FOUR_LINES = "  positivity\n  done\n", "  a\n  b\n  c\n  d\n"
TOO_LONG = "  nlinarith [" + ", ".join(["sq_nonneg x"] * 12) + "]\n"            # ONE line, and with its statement over the world's 140 characters
# The pool build's two files: the published proofs, then the same with Mathlib's renames applied (new certificates).
PUBLISHED = {
    "candidates.jsonl": [
        {"problem_id": "wb_1", "certificates": [{"source": "stp", "proof": THREE_LINES}, {"source": "goedel", "proof": ONE_LINE}, {"source": "statement"}]},
        {"problem_id": "wb_2", "certificates": [{"source": "stp", "proof": "  sorry\n"}]},                      # a forbidden token: never tried
        {"problem_id": "wb_3", "certificates": [{"source": "stp", "proof": TOO_LONG}, {"source": "stp", "proof": FOUR_LINES}, {"source": "stp", "proof": TWO_LINES},
                                                {"source": "stp", "proof": OTHER_TWO_LINES}, {"source": "stp", "proof": THREE_LINES}]},
        {"problem_id": "wb_easy", "certificates": [{"source": "stp", "proof": ONE_LINE}]},                      # not chosen: its line is never parsed
        *({"problem_id": f"stp_{letter}", "certificates": [{"source": "stp", "proof": f"  norm_num [{letter}]\n"}]} for letter in "abcdef"),
    ],
    "renamed.jsonl": [
        {"problem_id": "wb_1", "certificates": [{"source": "stp", "proof": ONE_LINE}, {"source": "stp", "proof": TWO_LINES}]},      # the first is held already
        {"problem_id": "stp_a", "certificates": [{"source": "stp", "proof": "  simp\n  norm_num [a]\n"}]},
    ],
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A small pool, its scores, the pool build's two files of published proofs, and no held-out problem among
    them; the tool's sizes made small: rows of at most 140 characters, 4 STP problems drawn, 5 rows written."""
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    monkeypatch.setattr(tool, "POOL_CANDIDATES", write(tmp_path / "data" / "candidates.jsonl", POOL))
    monkeypatch.setattr(tool, "HELDOUT", write(tmp_path / "data" / "heldout.jsonl", [{"problem_id": "held_1"}, {"problem_id": "held_2"}]))
    for name, rows in PUBLISHED.items():
        write(tmp_path / "steps" / name, rows)
    config = copy.deepcopy(CONFIG)
    config["ladder_loop"]["ceiling"]["checkpoints"] = [2, 5]
    config["ladder_loop"]["ceiling"]["training_file"].update({"stp_sample": 4, "maximum_characters": 140})
    arguments = SimpleNamespace(scores=write(tmp_path / "scores.jsonl", SCORES), candidates_store=tmp_path / "steps", out=tmp_path / "out" / "training.jsonl",
                                limit=None, dry_run=False, api_key_file=None, ca_file=None, in_flight=None)
    return SimpleNamespace(config=config, arguments=arguments, write=write, tmp_path=tmp_path)


def _lean(good, calls=None):
    """Lean by the proof's text: verified when it is one of `good`."""
    def check(snippets):
        if calls is not None:
            calls.append(list(snippets))
        return [VERIFIED if proof in good else FAILED for _, _, proof, _ in snippets]
    return check


# ---------------------------------------------------------------------------------------------- selection
def test_the_settings_are_the_scratch_scripts():
    own = CONFIG["ladder_loop"]["ceiling"]["training_file"]
    assert own == {"below_predicted_rate": 0.05, "stp_sample": 4800, "proofs_tried": 3, "maximum_characters": 2400, "seed": 0, "lean_seconds": 60}
    assert CONFIG["ladder_loop"]["ceiling"]["checkpoints"] == [2000, 8000] and tool.LEAN_WORKBOOK == "lean_workbook" and tool.TRUE_SIDE == "true"
    assert tool.PROOF_FILES == ("candidates.jsonl", "renamed.jsonl")              # as published first, then as renamed
    assert tool.POOL_CANDIDATES == PACKAGE / "data" / "ladder_l2" / "candidates.jsonl" and tool.HELDOUT == PACKAGE / "data" / "ladder_l0" / "heldout.jsonl"


def test_the_eligible_are_on_the_side_true_with_a_predicted_rate_below_the_bar():
    predicted = {row["problem_id"]: row["predicted_rate"] for row in SCORES}
    found = tool.eligible(POOL, predicted, 0.05)
    assert list(found) == ["wb_3", "stp_c", "wb_1", "wb_2", "stp_a", "stp_b", "stp_d", "stp_e", "stp_f"]      # the pool's order; no refuted, easy or unscored problem
    assert "wb_easy" in tool.eligible(POOL, predicted, 0.31) and "wb_easy" not in tool.eligible(POOL, predicted, 0.3)      # below, not at
    assert found["wb_1"] is POOL[2]


def test_a_held_out_problem_among_the_candidates_is_refused(world):
    predicted = {row["problem_id"]: row["predicted_rate"] for row in SCORES}
    tool.refuse_heldout(tool.eligible(POOL, predicted, 0.05), ["held_1", "held_2"])
    with pytest.raises(ValueError, match=r"2 held-out problems are among the candidates \(first: stp_c\)"):
        tool.refuse_heldout(tool.eligible(POOL, predicted, 0.05), ["wb_1", "held_1", "stp_c"])
    # The tool itself: a held-out problem among the pool's candidates stops it before anything is sent or written.
    world.write(tool.HELDOUT, [{"problem_id": "held_1"}, {"problem_id": "wb_3"}])
    calls = []
    with pytest.raises(ValueError, match=r"1 held-out problems are among the candidates \(first: wb_3\)"):
        tool.build(world.arguments, world.config, check=_lean(set(), calls))
    assert calls == [] and not world.arguments.out.exists()


def test_every_workbook_problem_is_chosen_in_the_order_of_its_id_then_a_seeded_draw_of_the_stp_ones():
    predicted = {row["problem_id"]: row["predicted_rate"] for row in SCORES}
    candidates = tool.eligible(POOL, predicted, 0.05)
    workbook, stp, chosen = tool.choose(candidates, 4, 0)
    assert workbook == ["wb_1", "wb_2", "wb_3"] and stp == ["stp_a", "stp_b", "stp_c", "stp_d", "stp_e", "stp_f"]
    assert chosen == workbook + random.Random(0).sample(stp, 4) and chosen[3:] != stp[:4] and len(set(chosen)) == 7
    assert tool.choose(candidates, 4800, 0)[2] == workbook + random.Random(0).sample(stp, 6)      # fewer than the draw: all of them, in the draw's order
    assert tool.choose(candidates, 4, 1)[2] != chosen
    # A trial (`--limit`): the first half of that many from the front and the rest from the back, as the scratch script cut it.
    assert tool.choose(candidates, 4, 0, limit=4)[2] == chosen[:2] + chosen[-2:]
    assert tool.choose(candidates, 4, 0, limit=5)[2] == chosen[:2] + chosen[-3:] and tool.choose(candidates, 4, 0, limit=0)[2] == chosen


# ------------------------------------------------------------------------------------------------- proofs
def _lines(rows):
    return [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]


def test_the_published_proofs_are_read_in_the_files_order_without_repeats_forbidden_tokens_or_proofs_too_long():
    candidates = {row["problem_id"]: row for row in POOL}
    files = [_lines(PUBLISHED[name]) for name in tool.PROOF_FILES]
    assert tool.id_at_the_start(files[0][0]) == "wb_1" and tool.id_at_the_start(files[1][1]) == "stp_a"
    proofs, too_long = tool.published_proofs(files, {"wb_1", "wb_2", "wb_3", "stp_a"}, candidates, 140)
    assert proofs["wb_1"] == [THREE_LINES, ONE_LINE, TWO_LINES]                   # as published, then the renamed ones that are new; a certificate with no proof is none
    assert proofs.get("wb_2", []) == [] and "wb_easy" not in proofs               # `sorry` is never tried; a problem that is not wanted is not read
    assert proofs["wb_3"] == [FOUR_LINES, TWO_LINES, OTHER_TWO_LINES, THREE_LINES] and too_long == 1
    assert proofs["stp_a"] == ["  norm_num [a]\n", "  simp\n  norm_num [a]\n"]
    # THE CHARACTER RULE: a proof is tried only if statement and proof together are at most the limit.
    exact = len(_statement("wb_3")) + len(TOO_LONG)
    assert exact > 140 and tool.published_proofs(files, {"wb_3"}, candidates, exact)[0]["wb_3"][0] == TOO_LONG
    assert TOO_LONG not in tool.published_proofs(files, {"wb_3"}, candidates, exact - 1)[0]["wb_3"]
    assert tool.published_proofs(files, {"wb_3"}, candidates, exact - 1)[1] == 1 and tool.published_proofs(files, {"wb_3"}, candidates, exact)[1] == 0


def test_at_most_the_three_shortest_by_lines_are_checked_as_a_solvers_attempt_is_and_the_rule_comes_first():
    candidates = {row["problem_id"]: row for row in POOL}
    files = [_lines(PUBLISHED[name]) for name in tool.PROOF_FILES]
    proofs, _ = tool.published_proofs(files, {"wb_1", "wb_3"}, candidates, 140)
    snippets = tool.checks(["wb_3", "wb_2", "wb_1"], proofs, candidates, 3)
    assert [(name, key) for name, key, _, _ in snippets] == [("wb_3|0", "wb_3"), ("wb_3|1", "wb_3"), ("wb_3|2", "wb_3"), ("wb_1|0", "wb_1"), ("wb_1|1", "wb_1"), ("wb_1|2", "wb_1")]
    # wb_3: the one-line proof is too long, so it takes no place among the three: the two two-line proofs in the files' order (a stable sort), then three lines.
    assert [proof for _, key, proof, _ in snippets if key == "wb_3"] == [TWO_LINES, OTHER_TWO_LINES, THREE_LINES]
    assert [proof for _, key, proof, _ in snippets if key == "wb_1"] == [ONE_LINE, TWO_LINES, THREE_LINES]
    # With the rule off the long one-line proof is the shortest by lines, and the three-line proof is not tried.
    without, _ = tool.published_proofs(files, {"wb_3"}, candidates, 10 ** 9)
    assert [proof for _, _, proof, _ in tool.checks(["wb_3"], without, candidates, 3)] == [TOO_LONG, TWO_LINES, OTHER_TWO_LINES]
    # The Lean file is the one a solver's attempt is checked as: the pool caches by its text.
    assert all(code == imports_first(build_proof_source(_statement(key), proof)) for _, key, proof, code in snippets)
    assert snippets[3][3].startswith("import Mathlib\n") and snippets[3][3].endswith(f"{_statement('wb_1')}{ONE_LINE}\n#print axioms wb_1\n")


def test_an_answer_is_judged_as_an_attempt_is_under_v4_27_and_the_proof_kept_is_the_fewest_lines_then_characters():
    snippets = [("wb_1|0", "wb_1", ONE_LINE, "file"), ("wb_1|1", "wb_1", TWO_LINES, "file"), ("wb_1|2", "wb_1", THREE_LINES, "file"),
                ("wb_3|0", "wb_3", TWO_LINES, "file"), ("wb_3|1", "wb_3", OTHER_TWO_LINES, "file"), ("stp_a|0", "stp_a", ONE_LINE, "file")]
    verified, statuses = tool.read_answers(snippets, [FAILED, VERIFIED, VERIFIED, VERIFIED, VERIFIED_WITH_A_WARNING, TIMED_OUT])
    assert dict(verified) == {"wb_1": [TWO_LINES, THREE_LINES], "wb_3": [TWO_LINES, OTHER_TWO_LINES]}      # a warning that says "failed" is not an error at v4.27
    assert dict(statuses) == {"lean_error": 1, "verified": 4, "timeout": 1}
    candidates = {row["problem_id"]: row for row in POOL}
    rows = tool.training_rows(["wb_3", "stp_a", "wb_1"], verified, candidates, {"wb_1": 0.011, "wb_3": 0.033, "stp_a": 0.02})
    assert [row["problem_id"] for row in rows] == ["wb_3", "wb_1"] and all(list(row) == FIELDS for row in rows)
    # Two proofs of two lines: the one with fewer characters. Fewer lines beat fewer characters.
    assert rows[0]["proof"] == OTHER_TWO_LINES and len(OTHER_TWO_LINES) < len(TWO_LINES) and rows[1]["proof"] == TWO_LINES
    assert rows[0] == {"problem_id": "wb_3", "kind": "lean_workbook", "side": "true", "statement": _statement("wb_3"), "proof": OTHER_TWO_LINES, "proof_lines": 2,
                       "predicted_rate": 0.033, "certificate_source": "stp"}
    fewest_lines = tool.training_rows(["wb_1"], {"wb_1": [OTHER_TWO_LINES, ONE_LINE]}, candidates, {"wb_1": 0.011})[0]
    assert fewest_lines["proof"] == ONE_LINE and len(ONE_LINE) > len(OTHER_TWO_LINES) and fewest_lines["proof_lines"] == 1      # one line, though it is the longer text
    kept = tool.training_rows(["wb_1"], {"wb_1": [THREE_LINES]}, candidates, {"wb_1": 0.011})[0]
    assert kept["proof"] == THREE_LINES + "\n" and kept["proof_lines"] == 3      # the proof ends in ONE newline; a comment line is not counted


def test_the_rows_written_are_every_workbook_row_then_stp_rows_to_the_number_shuffled_with_the_seed():
    rows = [{"problem_id": f"wb_{index}", "kind": "lean_workbook"} for index in range(3)] + [{"problem_id": f"stp_{index}", "kind": "stp_conjecture"} for index in range(6)]
    mixed = [rows[3], rows[0], rows[4], rows[1], rows[5], rows[2], *rows[6:]]      # the chosen order need not be by kind
    book, other, kept = tool.written(mixed, 5, 0)
    assert book == rows[:3] and other == rows[3:]
    expected = rows[:3] + rows[3:5]                                              # every Lean Workbook row, then the first STP rows to make up five
    random.Random(0).shuffle(expected)
    assert kept == expected and len(kept) == 5 and kept != rows[:5]
    assert len(tool.written(mixed, 2, 0)[2]) == 2 and {row["kind"] for row in tool.written(mixed, 2, 0)[2]} == {"lean_workbook"}      # more Lean Workbook rows than the number
    assert tool.written(mixed, 50, 0)[2] != tool.written(mixed, 50, 1)[2] and sorted(row["problem_id"] for row in tool.written(mixed, 50, 0)[2]) == sorted(row["problem_id"] for row in rows)


# ------------------------------------------------------------------------------------------- the whole tool
def test_the_whole_build_writes_the_file_the_stage_reads_and_prints_its_counts(world, capsys):
    calls = []
    good = {TWO_LINES, OTHER_TWO_LINES, THREE_LINES, *(f"  norm_num [{letter}]\n" for letter in "abcdef")}
    assert tool.build(world.arguments, world.config, check=_lean(good, calls)) == 0
    chosen = ["wb_1", "wb_2", "wb_3"] + random.Random(0).sample([f"stp_{letter}" for letter in "abcdef"], 4)
    (snippets,) = calls
    assert [key for _, key, _, _ in snippets if key.startswith("wb")] == ["wb_1"] * 3 + ["wb_3"] * 3 and "wb_2" not in {key for _, key, _, _ in snippets}
    assert TOO_LONG not in [proof for _, _, proof, _ in snippets]                  # over the world's 140 characters: not tried
    raw = world.arguments.out.read_bytes()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    # Every chosen Lean Workbook problem with a verified proof (wb_2 has none), then STP ones to five; shuffled with seed 0.
    expected = [key for key in chosen if key in ("wb_1", "wb_3")] + [key for key in chosen if key.startswith("stp")][:3]
    random.Random(0).shuffle(expected)
    assert [row["problem_id"] for row in rows] == expected and len(rows) == 5 and all(list(row) == FIELDS for row in rows)
    by_id = {row["problem_id"]: row for row in rows}
    assert by_id["wb_1"]["proof"] == TWO_LINES and by_id["wb_3"]["proof"] == OTHER_TWO_LINES and by_id["wb_1"]["predicted_rate"] == 0.01
    assert all(row["side"] == "true" and row["statement"] == _statement(row["problem_id"]) and row["proof"].endswith("\n") and not row["proof"].endswith("\n\n") for row in rows)
    assert "ℝ".encode("utf-8") in raw and b"\\u211d" not in raw and all(line.endswith("\n") for line in lines)      # as written: not escaped, one row a line
    assert raw == "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    printed = capsys.readouterr().out
    for said in ("eligible: 3 Lean Workbook, 6 STP | chosen 7", "problems with a published proof to try: 6 | proofs: 12",
                 "published proofs not tried for their length (statement and proof over 140 characters): 1", "checks to send: 11",
                 "check statuses: {'lean_error': 2, 'verified': 9}", "verified: 2 Lean Workbook, 4 STP | written 5 (2 Lean Workbook) to", "the first 2: "):
        assert said in printed, said
    assert "stp_a" in chosen            # the draw holds the problem with two published proofs: 3 + 4 + 5 proofs read, 3 + 3 + 5 checks, of which two fail
    # The stage's own check takes the file (no held-out problem, the side true, no problem twice, a statement and a proof each).
    from rlvr_lean.domain.ladder_round.ceiling import training_rows
    assert training_rows(rows, {"held_1"}, set(), 5) == rows


def test_a_dry_run_counts_and_sends_nothing_and_writes_nothing(world, capsys, monkeypatch):
    calls = []
    world.arguments.dry_run = True
    monkeypatch.setattr(tool, "send", lambda settings, snippets: calls.append("sent") or [])
    assert tool.build(world.arguments, world.config, check=_lean(set(), calls)) == 0
    printed = capsys.readouterr().out
    assert calls == [] and not world.arguments.out.exists() and "checks to send: 11" in printed and "dry run: nothing was sent and nothing was written" in printed
    # `--limit`: a trial on a few of the chosen problems.
    world.arguments.limit = 2
    assert tool.build(world.arguments, world.config) == 0 and "chosen 2" in capsys.readouterr().out
    # Through the command line: without --dry-run the tool wants a file to write and the pool's key and certificate.
    base = ["--scores", str(world.arguments.scores), "--candidates-store", str(world.arguments.candidates_store)]
    assert tool.main([*base, "--dry-run", "--limit", "4"]) == 0 and "chosen 4" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        tool.main([*base, "--out", str(world.arguments.out)])
    with pytest.raises(SystemExit):
        tool.main([*base, "--dry-run", "--in-flight", "none"])
    assert calls == [] and not world.arguments.out.exists()


def test_the_checks_go_to_the_pins_pool_as_background_work_with_a_lean_limit_of_sixty_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(verification_service, "resolve_lan_host", lambda name, server: ("192.0.2.9", name))
    key, certificate = tmp_path / "key", tmp_path / "ca.crt"
    key.write_text("pool-key\n")
    certificate.write_text("not read here")
    config = copy.deepcopy(CONFIG)
    settings = tool.lean_check_settings(argparse.Namespace(api_key_file=key, ca_file=certificate, in_flight=24), config, 60)
    pool = CONFIG["lean"]["pool"]
    assert settings.base_url == f"https://192.0.2.9:{pool['port']}" and settings.tls_server_name == pool["name"] and settings.api_key == "pool-key"
    assert (settings.priority, settings.lean_timeout_seconds, settings.concurrent_requests) == (BACKGROUND_PRIORITY, 60, 24)
    assert pool["lean_timeout_seconds"] != 60 and settings.server_queue_wait_seconds == pool["proxy_queue_seconds"] + pool["server_wait_seconds"]
    assert config["lean"]["pin"] == CONFIG["ladder_loop"]["lean_pin"] == "v4.27"      # the ladder loop's pin, whatever the config's default is
    default = tool.lean_check_settings(argparse.Namespace(api_key_file=key, ca_file=certificate, in_flight=None), copy.deepcopy(CONFIG), 60)
    assert default.concurrent_requests == pool["concurrent_requests"] and default.follow_pool_size is False
    # What is sent is the file as built, one snippet a check, named by problem and place.
    sent = {}

    class Verifier:
        def __init__(self, given):
            sent["settings"] = given

        async def __aenter__(self):
            return self

        async def __aexit__(self, *error):
            return False

        async def check(self, snippets):
            sent["snippets"] = [(snippet.snippet_id, snippet.code) for snippet in snippets]
            return [VERIFIED for _ in snippets]

    monkeypatch.setattr(tool, "KiminaVerifier", Verifier)
    assert tool.send(settings, [("wb_1|0", "wb_1", ONE_LINE, "the file's text")]) == [VERIFIED]
    assert sent == {"settings": settings, "snippets": [("wb_1|0", "the file's text")]}
