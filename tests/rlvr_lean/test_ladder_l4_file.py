"""L4's pretraining file (`tools/ladder_ceiling_set.py --file l4_pretrain`) and the two halves of the pool
(`domain/ladder_round/l4.py`): which half a problem is in, who is eligible and chosen, THE CERTIFICATE THE POOL BUILD
VERIFIED taken by the hash of its check (as published, and as renamed), what is left out and counted, the stop when the
hashes do not match, the rows written and their order, the summary, the refusals; and THE CONTROL, a seeded sample of
the written rows checked as solver's attempts. Spec: docs/spec/ladder-loop.spec.md, "L4: the loop from a model
pretrained on published proofs" ("The pool is cut in two", "The pretraining file"). No Lean check is made for the file;
the control's Lean is a stand-in handed to the tool: nothing touches the pool or the network. That the tool's `ceiling`
file is what it was is `test_ladder_ceiling_set.py`, unedited."""

import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rlvr_lean.domain.ladder_round import l4
from rlvr_lean.domain.ladder_round.l4 import LOOP, PRETRAIN, half_of, halves, refuse_the_other_half
from rlvr_lean.domain.problem_pool.certificates import Certificate, certificate_source, sha_of
from rlvr_lean.domain.verification.lean_source import build_proof_source, imports_first
from rlvr_lean.tools import ladder_ceiling_set as tool

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
CONFIG = yaml.safe_load((PACKAGE / "config" / "experiment.yaml").read_text())
VERIFIED = {"response": {"messages": [{"severity": "info", "data": "'t' depends on axioms: [propext, Classical.choice, Quot.sound]"}]}}
FAILED = {"response": {"messages": [{"severity": "error", "data": "unsolved goals"}]}}
TIMED_OUT = {"error": "Lean REPL command timed out in 60 seconds"}
ONE_LINE, TWO_LINES, THREE_LINES = "  nlinarith [sq_nonneg x]\n", "  have h := sq_nonneg x\n  exact h\n", "  intro\n  simp\n  done"
AS_RENAMED = "  have h := sq_nonneg x\n  exact le_div_iff₀ h\n"                       # a published proof with a library name the renaming pass replaced
FORBIDDEN_PROOF = "  run_cmd Lean.logInfo \"hello\"\n  simp\n"
LONG_PROOF = "  nlinarith [" + ", ".join(["sq_nonneg x"] * 220) + "]\n"               # with its statement, over 2,400 characters
WORKBOOK = [f"wb_{index}" for index in range(24)]
STP = [f"stp_{index}" for index in range(16)]
OF_THE_HALF = {half: sorted(key for key in (*WORKBOOK, *STP) if half_of(key, 0) == half) for half in (PRETRAIN, LOOP)}
# Of the pretrain half: one problem is known FALSE (never eligible); one's pool row names a hash none of its certificates has;
# one's verified certificate is a RENAMED one; one's certificate holds a forbidden token; one's is too long; one's proves another
# theorem than the statement L2 reads. Every other problem has three published proofs, and the pool names the SECOND.
REFUTED, NO_MATCH, RENAMED, FORBIDDEN, TOO_LONG, OTHER_THEOREM = OF_THE_HALF[PRETRAIN][:6]
PLAIN = OF_THE_HALF[PRETRAIN][6:]


def _statement(problem_id):
    return f"theorem {problem_id} (x : ℝ) : 0 ≤ x ^ 2 := by\n"


def _kind(problem_id):
    return "lean_workbook" if problem_id.startswith("wb") else "stp_conjecture"


def _entry(problem_id, proof, source="stp", renamed=(), theorem=None):
    entry = {"source": source, "theorem": theorem or _statement(problem_id), "proof": proof, "rewritten": 0}
    return {**entry, "renamed": list(renamed)} if renamed else entry


def _sha(problem_id, entry, side="true"):
    """The hash the pool build stores for a certificate: of its Lean file, with the fingerprint command for the side `true`."""
    certificate = Certificate(problem_id, side, entry["source"], entry["theorem"], entry["proof"], entry["rewritten"], tuple(entry.get("renamed", ())))
    return sha_of(certificate_source(certificate, with_fingerprint=side == "true"))


def _published(problem_id):
    """A problem's certificates as published (`candidates.jsonl`), as renamed (`renamed.jsonl`), and the one the pool build verified."""
    if problem_id == RENAMED:
        published, renamed = [_entry(problem_id, f"  bad_step {problem_id}\n", "goedel")], [_entry(problem_id, AS_RENAMED, "goedel", ["le_div_iff"])]
        return published, renamed, renamed[0]
    proof = {FORBIDDEN: FORBIDDEN_PROOF, TOO_LONG: LONG_PROOF}.get(problem_id, TWO_LINES)
    theorem = f"theorem {problem_id} (y : ℝ) : 0 ≤ y ^ 2 := by\n" if problem_id == OTHER_THEOREM else None
    published = [_entry(problem_id, THREE_LINES, "internlm_proofs", theorem=theorem), _entry(problem_id, proof, "goedel", theorem=theorem), _entry(problem_id, ONE_LINE, theorem=theorem)]
    return published, [], published[1]


L2_CANDIDATES = [{"problem_id": key, "kind": _kind(key), "side": "false" if key == REFUTED else "true", "statement": _statement(key), "certificate_source": "elsewhere"}
                 for key in (*STP, *WORKBOOK)]
CANDIDATES = [{"problem_id": key, "kind": _kind(key), "side": "false" if key == REFUTED else "true", "statement": _statement(key), "statement_published": _statement(key),
               "rewritten": 0, "group": f"group_{key}", "certificates": _published(key)[0]} for key in (*WORKBOOK, *STP)]
RENAMED_ROWS = [{"problem_id": key, "certificates": _published(key)[1]} for key in (*WORKBOOK, *STP) if _published(key)[1]]
POOL = [{"problem_id": key, "kind": _kind(key), "side": "false" if key == REFUTED else "true", "statement": _statement(key),
         "certificate_source": _published(key)[2]["source"], "certificate_renamed": _published(key)[2].get("renamed", []),
         "certificate_sha": _sha(key, _entry(key, "  another_text\n")) if key == NO_MATCH else _sha(key, _published(key)[2], "false" if key == REFUTED else "true")}
        for key in (*WORKBOOK, *STP)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A pool of forty problems in two halves, with the pool build's three files (its candidates with their published
    proofs, the renamed ones, the pool's rows with the hash of each verified certificate), no held-out problem among
    them. Half of the chosen may be without a match here (the spec's 1% is its own test); nothing may reach Lean."""
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    monkeypatch.setattr(tool, "POOL_CANDIDATES", write(tmp_path / "data" / "candidates.jsonl", L2_CANDIDATES))
    monkeypatch.setattr(tool, "HELDOUT", write(tmp_path / "data" / "heldout.jsonl", [{"problem_id": "held_1"}]))
    monkeypatch.setattr(tool, "send", lambda *given: (_ for _ in ()).throw(AssertionError("the file is made with no Lean check")))
    for name, rows in (("candidates.jsonl", CANDIDATES), ("renamed.jsonl", RENAMED_ROWS), ("pool.jsonl", POOL)):
        write(tmp_path / "steps" / name, rows)
    config = copy.deepcopy(CONFIG)
    config["ladder_loop"]["l4"]["pretraining_file"]["maximum_share_without_a_match"] = 0.5
    arguments = SimpleNamespace(file="l4_pretrain", scores=None, candidates_store=tmp_path / "steps", out=tmp_path / "out" / "pretraining.jsonl", limit=None, dry_run=False,
                                api_key_file=None, ca_file=None, in_flight=None, control=None)
    return SimpleNamespace(config=config, arguments=arguments, write=write, tmp_path=tmp_path)


# ----------------------------------------------------------------------------------------------- the halves
def test_the_settings_are_the_specs():
    own = CONFIG["ladder_loop"]["l4"]
    assert own["half_seed"] == 0 and own["pretraining_file"] == {"maximum_characters": 2400, "seed": 0, "maximum_share_without_a_match": 0.01,
                                                                 "control": {"rows": 300, "seed": 0, "lean_seconds": 60, "minimum_share_verified": 0.98}}
    of_the_ceiling = CONFIG["ladder_loop"]["ceiling"]["training_file"]
    assert (own["pretraining_file"]["maximum_characters"], own["pretraining_file"]["control"]["lean_seconds"]) == (of_the_ceiling["maximum_characters"], of_the_ceiling["lean_seconds"])
    assert (PRETRAIN, LOOP) == l4.HALVES == ("pretrain", "loop") and tool.POOL_FILE == "pool.jsonl" and tool.PROOF_FILES == ("candidates.jsonl", "renamed.jsonl")


def test_a_problems_half_is_a_hash_of_its_id_even_is_pretrain_and_odd_is_loop():
    for key in ("lean_workbook_1234", "stp_conjecture_9", "wb_0"):
        even = int(hashlib.sha256(f"0:l4_half:{key}".encode()).hexdigest(), 16) % 2 == 0
        assert half_of(key, 0) == (PRETRAIN if even else LOOP) == half_of(key, 0)                               # the same whenever it is asked
    assert any(half_of(key, 0) != half_of(key, 1) for key in WORKBOOK)                                          # another seed cuts the pool elsewhere
    assert len(OF_THE_HALF[PRETRAIN]) >= 12 and OF_THE_HALF[LOOP] and not set(OF_THE_HALF[PRETRAIN]) & set(OF_THE_HALF[LOOP])
    assert halves([*WORKBOOK, *STP], 0) == {PRETRAIN: len(OF_THE_HALF[PRETRAIN]), LOOP: len(OF_THE_HALF[LOOP])} and halves([], 0) == {PRETRAIN: 0, LOOP: 0}
    refuse_the_other_half([{"problem_id": key} for key in OF_THE_HALF[PRETRAIN]], PRETRAIN, 0, "the pretraining file")
    with pytest.raises(ValueError, match=f"1 problems of the pretraining file are not of the `pretrain` half .first: {OF_THE_HALF[LOOP][0]}.: refused"):
        refuse_the_other_half([{"problem_id": OF_THE_HALF[PRETRAIN][0]}, {"problem_id": OF_THE_HALF[LOOP][0]}], PRETRAIN, 0, "the pretraining file")
    if not (PACKAGE / "data" / "ladder_l2" / "candidates.jsonl").exists():
        pytest.skip("the rest needs the pool's data directory (src/rlvr_lean/data/ladder_l2: derived data), which is not shipped")
    # On the pool's own candidates the two halves are about even, and every problem is in exactly one.
    with (PACKAGE / "data" / "ladder_l2" / "candidates.jsonl").open(encoding="utf-8") as handle:
        ids = [json.loads(line)["problem_id"] for line in handle if line.strip()]
    counted = halves(ids, CONFIG["ladder_loop"]["l4"]["half_seed"])
    assert sum(counted.values()) == len(ids) == len(set(ids)) > 50000 and all(0.49 < count / len(ids) < 0.51 for count in counted.values())


# ------------------------------------------------------------------------------ the certificate, by its hash
def _stored(rows):
    return [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]


def test_the_certificate_is_the_one_whose_check_has_the_pools_hash_as_published_or_as_renamed():
    wanted = {key for key in OF_THE_HALF[PRETRAIN] if key != REFUTED}
    pool = {row["problem_id"]: row for row in POOL}
    found, unmatched = tool.certified(_stored(CANDIDATES), _stored(RENAMED_ROWS), pool, wanted)
    assert set(found) == wanted - {NO_MATCH} and unmatched == {NO_MATCH: "none of the checks of its 3 certificates has the hash the pool names"}
    # As published: the SECOND of its three proofs is the one the pool build verified, not the first and not the shortest.
    plain = found[PLAIN[0]]
    assert (plain.proof, plain.source, plain.renamed, plain.theorem) == (TWO_LINES, "goedel", (), _statement(PLAIN[0]))
    # As renamed: the proof of `renamed.jsonl`, which follows the published ones as in the pool's plan; it carries the names replaced.
    renamed = found[RENAMED]
    assert (renamed.proof, renamed.source, renamed.renamed) == (AS_RENAMED, "goedel", ("le_div_iff",))
    # The hash is of the pool build's own file for a problem known true: header, theorem, proof, the axiom report and the fingerprint command.
    assert pool[RENAMED]["certificate_sha"] == sha_of(certificate_source(renamed, with_fingerprint=True)) != sha_of(certificate_source(renamed, with_fingerprint=False))
    assert pool[PLAIN[0]]["certificate_sha"] == sha_of(certificate_source(plain, with_fingerprint=True))
    # Without the renamed file the renamed certificate is not found; a hash made without the fingerprint command matches nothing.
    assert RENAMED in tool.certified(_stored(CANDIDATES), [], pool, wanted)[1]
    # The renamed certificates FOLLOW the published ones, as in the pool's plan: a problem that has renamed ones keeps its published ones too.
    as_published = {**pool, RENAMED: {**pool[RENAMED], "certificate_sha": _sha(RENAMED, _published(RENAMED)[0][0])}}
    assert tool.certified(_stored(CANDIDATES), _stored(RENAMED_ROWS), as_published, wanted)[0][RENAMED].proof == f"  bad_step {RENAMED}\n"
    other = {**pool, PLAIN[0]: {**pool[PLAIN[0]], "certificate_sha": sha_of(certificate_source(plain, with_fingerprint=False))}}
    assert PLAIN[0] in tool.certified(_stored(CANDIDATES), _stored(RENAMED_ROWS), other, wanted)[1]
    # A wanted problem the pool's file or the pool build's candidates do not hold has no match, and says which.
    less = {key: row for key, row in pool.items() if key != PLAIN[1]}
    _, why = tool.certified(_stored([row for row in CANDIDATES if row["problem_id"] != PLAIN[2]]), _stored(RENAMED_ROWS), less, wanted)
    assert (why[PLAIN[1]], why[PLAIN[2]]) == ("the pool's file has no row for it", "the pool build's candidates have no row for it")
    # Only the wanted problems' lines are read: a line of another problem is not even parsed.
    assert tool.certified([*_stored(CANDIDATES), '{"problem_id": "elsewhere", not json\n', "\n"], [], pool, {PLAIN[0]})[0][PLAIN[0]].proof == TWO_LINES
    assert tool.wanted_rows(_stored(POOL), {PLAIN[0]}) == {PLAIN[0]: pool[PLAIN[0]]}


# ------------------------------------------------------------------------------------------- the whole build
def test_the_pretraining_file_is_every_true_problem_of_the_pretrain_half_with_the_certificate_the_pool_verified(world, capsys):
    assert tool.build(world.arguments, world.config) == 0                                      # no Lean check: `send` would raise
    eligible = [key for key in OF_THE_HALF[PRETRAIN] if key != REFUTED]
    written = [key for key in eligible if key not in (NO_MATCH, FORBIDDEN, TOO_LONG, OTHER_THEOREM)]
    expected = list(written)
    random.Random(0).shuffle(expected)
    rows = [json.loads(line) for line in world.arguments.out.read_text(encoding="utf-8").splitlines()]
    assert [row["problem_id"] for row in rows] == expected != written and RENAMED in written                    # shuffled with the seed over the list in the order of the ids
    pool = {row["problem_id"]: row for row in POOL}
    for row in rows:
        assert list(row) == ["problem_id", "kind", "side", "statement", "proof", "proof_lines", "predicted_rate", "certificate_source", "half", "certificate_sha",
                             "certificate_renamed"]
        assert (row["side"], row["half"], row["predicted_rate"], row["statement"]) == ("true", "pretrain", None, _statement(row["problem_id"]))
        assert row["kind"] == _kind(row["problem_id"]) and row["certificate_sha"] == pool[row["problem_id"]]["certificate_sha"] and row["certificate_source"] == "goedel"
        if row["problem_id"] == RENAMED:
            assert (row["proof"], row["proof_lines"], row["certificate_renamed"]) == (AS_RENAMED, 2, ["le_div_iff"])
        else:
            assert (row["proof"], row["proof_lines"], row["certificate_renamed"]) == (TWO_LINES, 2, [])
    summary = json.loads(tool.summary_path(world.arguments.out).read_text())
    assert tool.summary_path(world.arguments.out).name == "pretraining.summary.json" and summary["rows"] == len(rows) == len(eligible) - 4
    assert summary["sha256"] == hashlib.sha256(world.arguments.out.read_bytes()).hexdigest() and (summary["half_seed"], summary["seed"], summary["file"]) == (0, 0, "l4_pretrain")
    assert summary["candidates_by_half"] == {half: len(OF_THE_HALF[half]) for half in (PRETRAIN, LOOP)} and summary["eligible"] == summary["chosen"] == len(eligible)
    assert summary["candidates_on_the_side_true_by_half"] == {PRETRAIN: len(eligible), LOOP: len(OF_THE_HALF[LOOP])}
    assert (summary["matched"], summary["matched_with_a_renamed_certificate"], summary["matched_by_source"]) == (len(eligible) - 1, 1, {"goedel": len(eligible) - 1})
    assert summary["without_a_match"] == {"problems": 1, "first": {NO_MATCH: "none of the checks of its 3 certificates has the hash the pool names"}}
    assert summary["left_out"] == {"statement_is_not_the_certificates_theorem": {"problems": 1, "first": [OTHER_THEOREM]}, "forbidden_token": {"problems": 1, "first": [FORBIDDEN]},
                                   "too_long": {"problems": 1, "first": [TOO_LONG]}}
    assert summary["statements_that_are_not_the_pool_rows"] == 0 and summary["certificates_whose_source_or_renamed_names_are_not_the_pool_rows"] == 0
    assert summary["rows_by_half"] == {"pretrain": len(rows)} and sum(summary["rows_by_kind"].values()) == len(rows) and summary["proof_lines"]["by_length_group"] == {"2-3": len(rows)}
    assert (summary["rows_by_certificate_source"], summary["rows_with_a_renamed_certificate"]) == ({"goedel": len(rows)}, 1) and "no Lean check was made" in summary["the_proof"]
    # The summary is the file's alone: no check was made, so it holds no check status; and no partial file and no control are left beside it.
    assert not [key for key in summary if "check" in key or "timeout" in key] and sorted(path.name for path in world.arguments.out.parent.iterdir()) == [
        "pretraining.jsonl", "pretraining.summary.json"]
    printed = capsys.readouterr().out
    assert f"eligible (the pretrain half, on the side true): {len(eligible)} | chosen {len(eligible)}" in printed
    assert f"with the certificate the pool build verified (matched by the hash of its check): {len(eligible) - 1} of {len(eligible)} | without: 1\n  {NO_MATCH}: none of" in printed
    assert f"matched by source: {{'goedel': {len(eligible) - 1}}} | of them a renamed certificate: 1" in printed
    assert f"left out, the statement is not the theorem its certificate proves: 1 (first: {OTHER_THEOREM})" in printed
    assert f"left out, the proof holds a forbidden token: 1 (first: {FORBIDDEN})" in printed and f"left out, statement and proof are over 2400 characters: 1 (first: {TOO_LONG})" in printed
    assert f"rows: {len(rows)} of {len(eligible)} chosen" in printed and "THE CONTROL IS NOT MADE YET: --control 300" in printed


def test_more_than_one_in_a_hundred_without_a_match_stops_the_tool_and_nothing_is_written(world):
    world.config["ladder_loop"]["l4"]["pretraining_file"]["maximum_share_without_a_match"] = CONFIG["ladder_loop"]["l4"]["pretraining_file"]["maximum_share_without_a_match"]
    eligible = len(OF_THE_HALF[PRETRAIN]) - 1
    with pytest.raises(ValueError, match=f"1 of the {eligible} chosen problems have no certificate whose check has the hash the pool names, more than 1%: the hash is not being "
                                         f"rebuilt the way the pool build made it. Nothing was written .first: {NO_MATCH}: none of the checks"):
        tool.build(world.arguments, world.config)
    assert not world.arguments.out.parent.exists()
    world.arguments.dry_run = True                                   # a dry run stops the same way: it is where the stop is meant to be met first
    with pytest.raises(ValueError, match="the hash is not being rebuilt the way the pool build made it"):
        tool.build(world.arguments, world.config)
    # With every hash found (that one problem's pool row repaired) the real 1% lets the file be written.
    world.arguments.dry_run = False
    world.write(world.arguments.candidates_store / "pool.jsonl", [{**row, "certificate_sha": _sha(NO_MATCH, _published(NO_MATCH)[2])} if row["problem_id"] == NO_MATCH else row
                                                                  for row in POOL])
    assert tool.build(world.arguments, world.config) == 0 and json.loads(tool.summary_path(world.arguments.out).read_text())["without_a_match"]["problems"] == 0


def test_a_dry_run_reads_and_counts_and_writes_nothing_and_the_file_needs_no_scores_and_no_key(world, capsys):
    world.arguments.dry_run = True
    assert tool.build(world.arguments, world.config) == 0 and not world.arguments.out.parent.exists()
    printed = capsys.readouterr().out
    assert f"the pool's 40 candidates by half (seed 0): {{'pretrain': {len(OF_THE_HALF[PRETRAIN])}, 'loop': {len(OF_THE_HALF[LOOP])}}}" in printed
    assert "dry run: nothing was sent and nothing was written" in printed and "matched by source:" in printed and "left out, " in printed and "rows: " in printed
    command = ["--file", "l4_pretrain", "--candidates-store", str(world.arguments.candidates_store)]
    with pytest.raises(ValueError, match="the hash is not being rebuilt"):                    # through the command line the real 1% stands: this small world is over it
        tool.main([*command, "--dry-run"])
    world.write(world.arguments.candidates_store / "pool.jsonl", [{**row, "certificate_sha": _sha(NO_MATCH, _published(NO_MATCH)[2])} if row["problem_id"] == NO_MATCH else row
                                                                  for row in POOL])
    assert tool.main([*command, "--dry-run", "--limit", "4"]) == 0 and "chosen 4:" in capsys.readouterr().out   # a trial: two from the front, two from the back
    # The file is written with no key and no certificate of the pool's authority: no Lean check is made for it.
    out = world.tmp_path / "by_the_command" / "pretraining.jsonl"
    assert tool.main([*command, "--out", str(out)]) == 0 and out.exists() and tool.summary_path(out).exists()
    with pytest.raises(SystemExit):                                  # without --dry-run it wants a file to write
        tool.main(command)
    with pytest.raises(SystemExit):                                  # the pool build's steps are needed
        tool.main(["--file", "l4_pretrain", "--dry-run"])
    with pytest.raises(SystemExit):                                  # the ceiling's file still needs its scores
        tool.main(["--candidates-store", str(world.arguments.candidates_store), "--dry-run"])


def test_a_held_out_candidate_and_a_row_of_the_loop_half_are_refused_and_nothing_is_written(world):
    world.write(tool.HELDOUT, [{"problem_id": OF_THE_HALF[PRETRAIN][-1]}])
    with pytest.raises(ValueError, match="1 held-out problems are among the candidates"):
        tool.build(world.arguments, world.config)
    assert not world.arguments.out.parent.exists()
    candidates, pool = {row["problem_id"]: row for row in L2_CANDIDATES}, {row["problem_id"]: row for row in POOL}
    inside, outside = PLAIN[-1], OF_THE_HALF[LOOP][0]
    found, _ = tool.certified(_stored(CANDIDATES), _stored(RENAMED_ROWS), pool, {inside, outside})
    rows, left_out = tool.pretraining_rows([inside], found, candidates, pool, 0, 0, 2400)
    assert [(row["problem_id"], row["proof"], row["half"]) for row in rows] == [(inside, TWO_LINES, "pretrain")] and left_out == {rule: [] for rule in tool.LEFT_OUT}
    with pytest.raises(ValueError, match=f"1 problems of the pretraining file are not of the `pretrain` half .first: {outside}"):
        tool.pretraining_rows([inside, outside], found, candidates, pool, 0, 0, 2400)
    # The length rule is on the statement and the proof together, as the ceiling's file has it; exactly the limit is kept.
    exactly = len(_statement(inside)) + len(TWO_LINES)
    assert len(tool.pretraining_rows([inside], found, candidates, pool, 0, 0, exactly)[0]) == 1
    assert tool.pretraining_rows([inside], found, candidates, pool, 0, 0, exactly - 1) == ([], {**{rule: [] for rule in tool.LEFT_OUT}, "too_long": [inside]})
    assert list(tool.eligible_for_pretraining(L2_CANDIDATES, 0)) == [row["problem_id"] for row in L2_CANDIDATES if row["side"] == "true" and row["problem_id"] in OF_THE_HALF[PRETRAIN]]


# --------------------------------------------------------------------------------------------- the control
def _file_of(world, count):
    """A written file of `count` rows (hand-made: the control reads the file, whatever made it)."""
    rows = [{"problem_id": f"row_{index}", "kind": "stp_conjecture" if index % 3 else "lean_workbook", "side": "true", "statement": _statement(f"row_{index}"),
             "proof": TWO_LINES, "proof_lines": 2, "predicted_rate": None, "certificate_source": "stp", "half": "pretrain", "certificate_sha": f"{index:064d}",
             "certificate_renamed": ["le_div_iff"] if index % 10 == 0 else []} for index in range(count)]
    world.write(world.arguments.out, rows)
    return rows


def _lean(failing=(), timing_out=(), calls=None):
    def check(snippets):
        if calls is not None:
            calls.append(list(snippets))
        return [TIMED_OUT if key in timing_out else FAILED if key in failing else VERIFIED for _, key, _, _ in snippets]
    return check


def test_the_control_checks_a_seeded_sample_of_the_written_rows_as_solvers_attempts_and_stores_what_verified(world, capsys):
    rows, calls = _file_of(world, 500), []
    world.arguments.control = 100
    sample = tool.control_sample(rows, 100, 0)
    picked = sorted(random.Random(0).sample(range(500), 100))
    assert [row["problem_id"] for row in sample] == [f"row_{index}" for index in picked] and tool.control_sample(rows, 100, 1) != sample      # seeded, in the file's order
    assert tool.control_sample(rows[:7], 100, 0) == rows[:7]                                     # a file with no more rows than asked: all of them
    assert tool.build(world.arguments, world.config, check=_lean(calls=calls)) == 0
    (sent,) = calls
    assert [key for _, key, _, _ in sent] == [row["problem_id"] for row in sample]
    name, key, proof, source = sent[0]
    assert (name, proof, source) == (key, TWO_LINES, imports_first(build_proof_source(_statement(key), TWO_LINES)))                         # the file a solver's attempt is checked as
    stored = json.loads(tool.control_path(world.arguments.out).read_text())
    assert tool.control_path(world.arguments.out).name == "pretraining.control.json" and stored["sha256"] == hashlib.sha256(world.arguments.out.read_bytes()).hexdigest()
    assert (stored["rows_of_the_file"], stored["sampled"], stored["verified"], stored["share_verified"], stored["minimum_share_verified"], stored["passes"]) == (500, 100, 100, 1.0, 0.98, True)
    assert stored["statuses"] == {"verified": 100} and stored["not_verified"] == [] and (stored["seed"], stored["lean_seconds"], stored["control_of"]) == (0, 60, "pretraining.jsonl")
    assert "THE CONTROL PASSES: 100 of 100 verified (100.0%); at least 98% is asked" in capsys.readouterr().out
    # THE ARITHMETIC: 98 of 100 is enough, 97 is not; the exit code is the verdict, and those that did not verify are named with their statuses.
    two, three = [row["problem_id"] for row in sample[:2]], [row["problem_id"] for row in sample[:3]]
    assert tool.build(world.arguments, world.config, check=_lean(failing=two)) == 0 and json.loads(tool.control_path(world.arguments.out).read_text())["share_verified"] == 0.98
    assert tool.build(world.arguments, world.config, check=_lean(failing=two, timing_out=three[2:])) == 1
    failed = json.loads(tool.control_path(world.arguments.out).read_text())
    assert (failed["verified"], failed["share_verified"], failed["passes"], failed["statuses"]) == (97, 0.97, False, {"lean_error": 2, "verified": 97, "timeout": 1})
    assert [(entry["problem_id"], entry["status"]) for entry in failed["not_verified"]] == [(three[0], "lean_error"), (three[1], "lean_error"), (three[2], "timeout")]
    assert set(failed["not_verified"][0]) == {"problem_id", "status", "kind", "certificate_source", "certificate_renamed", "proof_lines"}
    printed = capsys.readouterr().out
    assert "THE CONTROL FAILS: 97 of 100 verified (97.0%); at least 98% is asked. THE FILE IS NOT TO BE USED" in printed and f"not verified: {three[2]}: timeout (stp, 2 lines" in printed
    # The file is built by the solver's attempt's own two functions, the header's imports moved first last of all.
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(tool, "build_proof_source", lambda statement, proof: f"<built {statement}{proof}>")
        patched.setattr(tool, "imports_first", lambda lean_file: f"<imports first {lean_file}>")
        calls = []
        tool.build(world.arguments, world.config, check=_lean(calls=calls))
        assert calls[0][0][3] == f"<imports first <built {_statement(sample[0]['problem_id'])}{TWO_LINES}>>"
    # The file itself is not touched by its control; with no number the sample is the settings' 300.
    assert [json.loads(line) for line in world.arguments.out.read_text(encoding="utf-8").splitlines()] == rows
    world.arguments.control, calls = 0, []
    assert tool.build(world.arguments, world.config, check=_lean(calls=calls)) == 0 and len(calls[0]) == 300 == CONFIG["ladder_loop"]["l4"]["pretraining_file"]["control"]["rows"]


def test_the_control_goes_to_the_pool_as_background_work_and_its_dry_run_sends_nothing(world, monkeypatch, capsys):
    _file_of(world, 40)
    seen = {}
    settings = SimpleNamespace(follow_pool_size=False, concurrent_requests=24, priority="background", lean_timeout_seconds=60)
    monkeypatch.setattr(tool, "lean_check_settings", lambda arguments, config, lean_seconds: (seen.update(lean_seconds=lean_seconds), settings)[1])
    monkeypatch.setattr(tool, "send", lambda given, snippets: (seen.update(settings=given, sent=len(snippets)), [VERIFIED] * len(snippets))[1])
    command = ["--file", "l4_pretrain", "--out", str(world.arguments.out), "--control", "10"]
    assert tool.main([*command, "--dry-run"]) == 0 and seen == {} and not tool.control_path(world.arguments.out).exists()
    assert "the control: 10 of the 40 rows" in capsys.readouterr().out
    key = world.write(world.tmp_path / "key", [])
    assert tool.main([*command, "--api-key-file", str(key), "--ca-file", str(key)]) == 0
    assert seen == {"lean_seconds": 60, "settings": settings, "sent": 10} and "priority background, Lean timeout 60 s" in capsys.readouterr().out
    for wrong in (command, ["--file", "l4_pretrain", "--control", "10", "--dry-run"], ["--out", str(world.arguments.out), "--control", "10", "--dry-run", "--scores", str(key)],
                  [*command[:-1], "-3", "--dry-run"]):                # no key; no written file; not L4's file; a negative number
        with pytest.raises(SystemExit):
            tool.main(wrong)
    world.write(world.arguments.out, [])
    with pytest.raises(ValueError, match="holds no row: there is nothing to check"):
        tool.main([*command, "--dry-run"])
