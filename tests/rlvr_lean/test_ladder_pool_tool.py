"""The ladder pool tool: the checker's rounds, resuming, the pool and the held-out draw.
Spec: docs/spec/ladder-loop.spec.md, L0 (fixtures 2, 3, 6, 7, 10). The Lean pool is a stand-in that answers
from a table; nothing touches the network."""

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rlvr_lean.data.published import candidate_row
from rlvr_lean.domain.problem_pool import (
    CERTIFICATE_CHECK,
    EXACTNESS_CHECK,
    FALSE_SIDE,
    GOEDEL,
    INTERNLM_PROOFS,
    INTERNLM_ROWS,
    LEAN_WORKBOOK,
    STATEMENT_CHECK,
    STP,
    STP_CONJECTURE,
    TRUE_SIDE,
    Candidate,
    Certificate,
)
from rlvr_lean.domain.problem_pool import SLICE_ALL, SLICE_PRESENT, SLICE_STP, SLICE_WORKBOOK, rank
from rlvr_lean.domain.verification import lean_pin
from rlvr_lean.infrastructure.artifact_store import ArtifactStore
from rlvr_lean.tools import ladder_pool

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[2] / "src" / "rlvr_lean" / "config" / "experiment.yaml").read_text())
CONFIG["ladder_loop"]["heldout"].update({"workbook_problems": 2, "stp_conjectures": 1})
# The stand-in library: `old_lemma` and `Real.old_sqrt` are gone from it; only the first is in the table.
CONFIG["ladder_loop"]["certificates"]["renames"] = {"old_lemma": "new_lemma"}
GONE = re.compile(r"(?<![\w.'])(old_lemma|old_sqrt)(?![\w'₀])")


def _candidate(problem, n, sources, kind=LEAN_WORKBOOK, group=None):
    statement = f"theorem {problem} (x : ℕ) : x + {n} = {n} + x := by\n"
    certificates = tuple(Certificate(problem, TRUE_SIDE, source, statement, f"  simp [{source}_{index}]") for index, source in enumerate(sources))
    return Candidate(problem, kind, TRUE_SIDE, statement, statement, 0, group or problem, certificates)


def _false_candidate(problem):
    statement = f"theorem {problem} : ∀ a : ℝ, a > 0 := by\n"
    negation = f"theorem negation_of_{problem} : ¬ (∀ a : ℝ, a > 0) := by\n"
    return Candidate(problem, LEAN_WORKBOOK, FALSE_SIDE, statement, statement, 0, problem,
                     (Certificate(problem, FALSE_SIDE, INTERNLM_ROWS, negation, "  push_neg\n  exact ⟨0, by norm_num⟩"),))


CANDIDATES = [
    _candidate("wb_a", 1, [GOEDEL, STP, INTERNLM_PROOFS]),      # verified by its first certificate
    _candidate("wb_b", 2, [GOEDEL, STP]),                       # its first fails, its second verifies
    _candidate("wb_c", 3, [INTERNLM_PROOFS]),                   # nothing verifies
    _false_candidate("wb_d"),                                   # exact, and its negation verifies
    _false_candidate("wb_e"),                                   # the built negation is not exact
    _candidate("wb_f", 6, [GOEDEL]),
    _candidate("wb_g", 7, [STP]),
    _candidate("stp_1", 11, [STP], kind=STP_CONJECTURE),
    _candidate("stp_2", 12, [STP], kind=STP_CONJECTURE),
]
SET_ASIDE = [{"statement_id": "wb_z", "statement": "theorem wb_z (x : ℕ) : x * 9 = 9 * x := by\n", "statement_published": "",
              "rewritten": 0, "group": "nl_z"}]
FAILS = {("wb_b", GOEDEL), ("wb_c", INTERNLM_PROOFS)}


def _raw(messages):
    return {"time": 1.5, "response": {"messages": [{"severity": severity, "data": data} for severity, data in messages]}}


def lean_stand_in(check, fingerprints):
    """What Lean would answer for one check of the table above."""
    if check.kind == EXACTNESS_CHECK:
        exact = check.problem_id != "wb_e"
        return _raw([("warning", "declaration uses 'sorry'"), ("info", f"rlvr-type-fingerprint {fingerprints[check.problem_id]}")]
                    + ([] if exact else [("error", "type mismatch")]))
    if check.kind == STATEMENT_CHECK:
        return _raw([("warning", "declaration uses 'sorry'"), ("info", "rlvr-type-fingerprint 999")])
    gone = GONE.search(check.lean_file)
    if gone is not None:        # as Lean reports it: the name as written, or in full when it resolved a namespace
        name = gone.group(1)
        report = "Unknown identifier `old_lemma`" if name == "old_lemma" else "Unknown constant `Real.old_sqrt`"
        return _raw([("error", report), ("error", f"Unknown constant `{check.problem_id}`")])
    if (check.problem_id, check.source) in FAILS:
        return _raw([("error", "unsolved goals")])
    axioms = ("info", "'t' depends on axioms: [propext]")
    return _raw([axioms] if check.problem_id in ("wb_d", "wb_e") else [axioms, ("info", f"rlvr-type-fingerprint {fingerprints[check.problem_id]}")])


class Fingerprints(dict):
    """The scripted Lean's type fingerprints: the given ones, and one of its own for any other problem."""

    def __missing__(self, key):
        return 10_000 + int.from_bytes(key.encode(), "big") % 1_000_000


@pytest.fixture
def store(tmp_path, monkeypatch):
    steps_store = ArtifactStore(tmp_path / "steps")
    steps_store.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in CANDIDATES))
    steps_store.write_rows("set_aside.jsonl", SET_ASIDE)
    steps_store.write_rows("present_holdout.jsonl", [{"statement_id": "wb_f"}, {"statement_id": "wb_z"}])
    sent = []
    fingerprints = {candidate.problem_id: 100 + index for index, candidate in enumerate(CANDIDATES)}

    async def fake_run_checks(settings, jobs, out, retry_passes, answer_of=None, on_answer=None):
        plan = ladder_pool.Plan(out.parent)              # the store this run is against
        by_sha = {check.sha: check for checks in plan.checks.values() for check in checks}
        by_sha.update({check.sha: check for check in plan.statement_checks})
        with (out / "checks.jsonl").open("a") as handle:
            for job in jobs:
                sent.append(by_sha[job.sha])
                row = answer_of(job, lean_stand_in(by_sha[job.sha], Fingerprints(fingerprints)), settings.pin)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                on_answer(row)
        return {"stopped_early": False, "answered": len(jobs)}

    monkeypatch.setattr(ladder_pool, "run_checks", fake_run_checks)
    monkeypatch.setattr(ladder_pool, "pin_settings", lambda arguments, config, background=False: SimpleNamespace(
        pin=lean_pin("v4.27"), concurrent_requests=8, lean_timeout_seconds=120, follow_pool_size=False,
        priority="background" if background else None))
    return SimpleNamespace(path=tmp_path, sent=sent, steps=steps_store, fingerprints=fingerprints)


def _arguments(store, **extra):
    return argparse.Namespace(**{"store": store.path, "max_checks": None, "retry_passes": 0, "dry_run": False, "out": None, **extra})


def _both_passes(store, config=CONFIG):
    """The first pass to its end, then the renaming pass to its end."""
    assert ladder_pool.check(_arguments(store), config) == 0
    return ladder_pool.rename(_arguments(store), config)


def test_the_checker_sends_one_check_per_open_problem_per_round_and_stops_at_the_first_verified(store):
    assert ladder_pool.check(_arguments(store), CONFIG) == 0
    sent = [(check.problem_id, check.kind, check.source) for check in store.sent]
    # wb_a was settled by its first certificate: its second and third were never sent.
    assert [entry for entry in sent if entry[0] == "wb_a"] == [("wb_a", CERTIFICATE_CHECK, GOEDEL)]
    assert [entry for entry in sent if entry[0] == "wb_b"] == [("wb_b", CERTIFICATE_CHECK, GOEDEL), ("wb_b", CERTIFICATE_CHECK, STP)]
    # A negation that is not exact is never given to its certificate.
    assert [entry[1] for entry in sent if entry[0] == "wb_e"] == [EXACTNESS_CHECK]
    assert [entry[1] for entry in sent if entry[0] == "wb_d"] == [EXACTNESS_CHECK, CERTIFICATE_CHECK]
    assert ("wb_z", STATEMENT_CHECK, "") in sent
    progress = json.loads((store.path / "checks" / "progress.json").read_text())
    assert (progress["state"], progress["in_pool"], progress["out"], progress["open"]) == ("finished", 7, 2, 0)


def test_a_second_run_sends_nothing(store):
    ladder_pool.check(_arguments(store), CONFIG)
    first = len(store.sent)
    assert ladder_pool.check(_arguments(store), CONFIG) == 0 and len(store.sent) == first


def test_a_run_cut_short_is_resumed_where_it_stopped(store):
    assert ladder_pool.check(argparse.Namespace(store=store.path, max_checks=3, retry_passes=0), CONFIG) == 1
    assert len(store.sent) == 3
    assert json.loads((store.path / "checks" / "progress.json").read_text())["state"] == "stopped"
    assert ladder_pool.check(_arguments(store), CONFIG) == 0
    shas = [check.sha for check in store.sent]
    assert len(shas) == len(set(shas))              # nothing was asked twice


def _mark_renaming_pass_built(store, config=CONFIG):
    store.steps.write_rows("renamed.jsonl", [])
    store.steps.mark_done(ladder_pool.RENAMED_STEP, {"table": ladder_pool.renames_table(config), "counts": {}})


def test_the_pool_is_refused_while_a_candidate_is_unsettled(store):
    """Fixture 2: nothing enters the pool without its check."""
    ladder_pool.check(_arguments(store, max_checks=3), CONFIG)
    _mark_renaming_pass_built(store)
    with pytest.raises(SystemExit, match="not settled"):
        ladder_pool.pool(_arguments(store), CONFIG)
    assert not (store.path / "steps" / "pool.jsonl").exists()


def test_the_pool_is_refused_until_both_passes_are_finished(store):
    """Spec: "The held-out sets are drawn only after both passes"."""
    assert ladder_pool.check(_arguments(store), CONFIG) == 0
    with pytest.raises(SystemExit, match="renaming pass has not run"):
        ladder_pool.pool(_arguments(store), CONFIG)
    assert not (store.path / "steps" / "heldout.jsonl").exists()
    assert ladder_pool.rename(_arguments(store), CONFIG) == 0
    assert ladder_pool.pool(_arguments(store), CONFIG) == 0
    # A table that changed since the pass was built: the pass must be run again first.
    other = json.loads(json.dumps(CONFIG))
    other["ladder_loop"]["certificates"]["renames"]["another_old_lemma"] = "another_new_lemma"
    with pytest.raises(SystemExit, match="table changed"):
        ladder_pool.pool(_arguments(store), other)


def test_the_pool_holds_verified_problems_only_and_h_is_apart_from_it(store):
    """Fixtures 2, 3, 6 and 7."""
    assert _both_passes(store) == 0
    assert ladder_pool.pool(_arguments(store), CONFIG) == 0
    pool = store.steps.read_rows("pool.jsonl")
    heldout = store.steps.read_rows("heldout.jsonl")
    in_pool, held = {row["problem_id"] for row in pool}, {row["problem_id"] for row in heldout}
    assert in_pool | held == {"wb_a", "wb_b", "wb_d", "wb_f", "wb_g", "stp_1", "stp_2"}      # wb_c and wb_e never verified
    assert not in_pool & held and "wb_z" not in in_pool | held
    assert "wb_f" in held                                               # the present holdout's verified problem is taken first
    assert [row["heldout_part"] for row in heldout] == [LEAN_WORKBOOK, LEAN_WORKBOOK, STP_CONJECTURE]
    by_id = {row["problem_id"]: row for row in pool + heldout}
    assert by_id["wb_b"]["certificate_source"] == STP and by_id["wb_d"]["side"] == FALSE_SIDE
    assert by_id["wb_d"]["type_fingerprint"] == store.fingerprints["wb_d"]          # from its exactness check
    assert all("simp [" not in json.dumps(row) and "push_neg" not in json.dumps(row) for row in pool + heldout)     # no published proof
    assert all(row["certificate_renamed"] == [] for row in pool + heldout)
    summary = store.steps.done_summary("pool")
    assert summary["verified"] == 7 and summary["not_verified_by_reason"] == {"no_certificate_verifies": 1, "negation_not_exact": 1}
    assert summary["set_aside"] == 1 and summary["set_aside_compiling"] == 1 and summary["verified_by_a_renamed_certificate"] == 0
    assert store.steps.read_rows("set_aside_checked.jsonl")[0]["type_fingerprint"] == 999


def test_h_is_drawn_once(store):
    _both_passes(store)
    ladder_pool.pool(_arguments(store), CONFIG)
    first = (store.path / "steps" / "heldout.jsonl").read_text()
    assert ladder_pool.pool(_arguments(store), CONFIG) == 0                         # the same draw again changes nothing
    assert (store.path / "steps" / "heldout.jsonl").read_text() == first
    other = json.loads(json.dumps(CONFIG))
    other["ladder_loop"]["heldout"]["seed"] += 1
    other["ladder_loop"]["heldout"]["workbook_problems"] = 3
    with pytest.raises(SystemExit, match="refused"):
        ladder_pool.pool(_arguments(store), other)
    assert (store.path / "steps" / "heldout.jsonl").read_text() == first


def test_problems_checked_after_h_was_drawn_join_the_pool_and_h_stays(store):
    """Spec, a round: "a chosen problem whose certificate has not yet been checked under the pin is checked
    now". More candidates later never move H; a copy of a member of H still stays out."""
    _both_passes(store)
    ladder_pool.pool(_arguments(store), CONFIG)
    heldout = (store.path / "steps" / "heldout.jsonl").read_text()
    held = json.loads(heldout.splitlines()[0])
    later = _candidate("stp_3", 13, [STP], kind=STP_CONJECTURE)
    copy = Candidate("stp_copy", STP_CONJECTURE, TRUE_SIDE, held["statement"].replace(held["problem_id"], "stp_copy"), "", 0, "stp_copy",
                     (Certificate("stp_copy", TRUE_SIDE, STP, held["statement"].replace(held["problem_id"], "stp_copy"), "  simp"),))
    store.steps.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in CANDIDATES + [later, copy]))
    store.fingerprints.update({"stp_3": 777, "stp_copy": 778})
    assert _both_passes(store) == 0
    assert ladder_pool.pool(_arguments(store), CONFIG) == 0
    assert (store.path / "steps" / "heldout.jsonl").read_text() == heldout
    in_pool = {row["problem_id"] for row in store.steps.read_rows("pool.jsonl")}
    assert "stp_3" in in_pool and "stp_copy" not in in_pool


def test_a_member_of_h_that_is_no_longer_verified_is_refused(store):
    _both_passes(store)
    ladder_pool.pool(_arguments(store), CONFIG)
    store.steps.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in CANDIDATES if candidate.problem_id != "wb_f"))
    with pytest.raises(SystemExit, match="members of H are not verified"):
        ladder_pool.pool(_arguments(store), CONFIG)


def test_a_statement_verified_both_ways_stops_the_checker(store):
    """Fixture 10 at L0: the alarm."""
    twin = Candidate("wb_twin", LEAN_WORKBOOK, TRUE_SIDE, "theorem wb_twin : ∀ a : ℝ, a > 0 := by\n", "", 0, "wb_twin",
                     (Certificate("wb_twin", TRUE_SIDE, GOEDEL, "theorem wb_twin : ∀ a : ℝ, a > 0 := by\n", "  simp"),))
    store.steps.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in CANDIDATES + [twin]))
    store.fingerprints["wb_twin"] = 555
    assert ladder_pool.check(_arguments(store), CONFIG) == 3
    assert json.loads((store.path / "checks" / "progress.json").read_text())["state"] == "SOUNDNESS ALARM"
    _mark_renaming_pass_built(store)
    with pytest.raises(Exception, match="both sides"):
        ladder_pool.pool(_arguments(store), CONFIG)


# ----------------------------------------------------------------------------------------- the renaming pass
def _citing(problem, n, proofs, kind=LEAN_WORKBOOK):
    """A known-true candidate whose published proofs are given as (source, proof text)."""
    statement = f"theorem {problem} (x : ℝ) (h : 0 < x) : x / {n} ≤ x := by\n"
    certificates = tuple(Certificate(problem, TRUE_SIDE, source, statement, proof) for source, proof in proofs)
    return Candidate(problem, kind, TRUE_SIDE, statement, statement, 0, problem, certificates)


RENAMING = [
    _citing("wb_r", 21, [(GOEDEL, "  rw [old_lemma (by norm_num)]\n  linarith"), (STP, "  nlinarith [(old_lemma h).mp le_rfl]")]),
    _citing("wb_s", 22, [(STP, "  rw [old_sqrt h]\n  nlinarith")]),                # gone, reported in full, NOT in the table
    _citing("stp_r", 23, [(STP, "  exact old_lemma.mpr h.le")], kind=STP_CONJECTURE),
]


@pytest.fixture
def renaming(store):
    store.steps.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in CANDIDATES + RENAMING))
    store.fingerprints.update({"wb_r": 801, "wb_s": 802, "stp_r": 803})
    return store


def test_the_renaming_pass_is_refused_while_the_first_pass_is_open(renaming):
    ladder_pool.check(_arguments(renaming, max_checks=3), CONFIG)
    with pytest.raises(SystemExit, match="first pass is not finished"):
        ladder_pool.rename(_arguments(renaming), CONFIG)
    assert not (renaming.path / "steps" / "renamed.jsonl").exists() and not renaming.steps.is_done(ladder_pool.RENAMED_STEP)


def test_a_dry_run_counts_from_what_is_stored_and_writes_nothing(renaming, capsys):
    ladder_pool.check(_arguments(renaming, max_checks=3), CONFIG)
    before = sorted(path.name for path in (renaming.path / "steps").iterdir()), len(renaming.sent)
    capsys.readouterr()
    assert ladder_pool.rename(_arguments(renaming, dry_run=True), CONFIG) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["dry_run"] is True and counts["first_pass_open"] > 0 and counts["table_entries"] == 1
    assert (sorted(path.name for path in (renaming.path / "steps").iterdir()), len(renaming.sent)) == before


def test_the_renaming_pass_checks_renamed_certificates_of_what_the_first_pass_left_unsettled(renaming):
    assert ladder_pool.check(_arguments(renaming), CONFIG) == 0
    first_pass = len(renaming.sent)
    progress = json.loads((renaming.path / "checks" / "progress.json").read_text())
    assert (progress["in_pool"], progress["out"]) == (7, 5)                       # wb_r, wb_s and stp_r failed on a name that is gone
    assert ladder_pool.rename(_arguments(renaming), CONFIG) == 0
    rows = {row["problem_id"]: row["certificates"] for row in renaming.steps.read_rows("renamed.jsonl")}
    # Only problems the first pass left unsettled AND whose proofs carry a tabled name: not wb_c (no such name),
    # not wb_s (its name is not in the table), not wb_e (its negation is not exact), not one that verified.
    assert sorted(rows) == ["stp_r", "wb_r"]
    assert [(entry["source"], entry["proof"], entry["renamed"]) for entry in rows["wb_r"]] == [
        (GOEDEL, "  rw [new_lemma (by norm_num)]\n  linarith", ["old_lemma"]),
        (STP, "  nlinarith [(new_lemma h).mp le_rfl]", ["old_lemma"])]
    assert rows["stp_r"][0]["proof"] == "  exact new_lemma.mpr h.le"               # a projection of the lemma goes with it
    assert all(entry["theorem"] == RENAMING[0].statement for entry in rows["wb_r"])       # the statement is never renamed
    # One check per problem: wb_r's first renamed certificate verifies, its second is never sent.
    second_pass = [(check.problem_id, check.source) for check in renaming.sent[first_pass:]]
    assert sorted(second_pass) == [("stp_r", STP), ("wb_r", GOEDEL)]
    assert all("new_lemma" in check.lean_file and "old_lemma" not in check.lean_file for check in renaming.sent[first_pass:])
    counts = renaming.steps.done_summary(ladder_pool.RENAMED_STEP)["counts"]
    assert counts["unsettled_no_certificate_verifies"] == 4 and counts["problems_with_a_renamed_certificate"] == 2
    assert counts["renamed_certificates"] == 3 and counts["problems_by_renamed_name"] == {"old_lemma": 2}
    # The originals' failures stay on record, and the problems are in the pool by a MARKED certificate.
    plan = ladder_pool.Plan(renaming.path)
    originals = [plan.answers[check.sha]["status"] for check in ladder_pool.Plan(renaming.path, with_renamed=False).checks["wb_r"]]
    assert originals == ["lean_error", "lean_error"]
    assert ladder_pool.pool(_arguments(renaming), CONFIG) == 0
    by_id = {row["problem_id"]: row for row in renaming.steps.read_rows("pool.jsonl") + renaming.steps.read_rows("heldout.jsonl")}
    assert by_id["wb_r"]["certificate_renamed"] == ["old_lemma"] and by_id["wb_r"]["certificate_source"] == GOEDEL
    assert by_id["stp_r"]["certificate_renamed"] == ["old_lemma"] and "wb_s" not in by_id and "wb_c" not in by_id
    assert by_id["wb_r"]["statement"] == RENAMING[0].statement
    summary = renaming.steps.done_summary("pool")
    assert summary["verified"] == 9 and summary["verified_by_a_renamed_certificate"] == 2


def test_the_renaming_pass_resumes_and_a_second_run_sends_nothing(renaming):
    assert _both_passes(renaming) == 0
    sent = len(renaming.sent)
    assert ladder_pool.rename(_arguments(renaming), CONFIG) == 0 and len(renaming.sent) == sent
    # The first pass alone still reads as it did: the renamed certificates never change who it left unsettled.
    first = ladder_pool.Plan(renaming.path, with_renamed=False).verdicts()
    assert first["wb_r"].state == "out" and ladder_pool.Plan(renaming.path).verdicts()["wb_r"].state == "in"


def test_a_table_that_grows_adds_certificates_and_keeps_the_answers(renaming):
    assert _both_passes(renaming) == 0
    sent = len(renaming.sent)
    grown = json.loads(json.dumps(CONFIG))
    grown["ladder_loop"]["certificates"]["renames"]["old_sqrt"] = "new_sqrt"        # the short form wb_s writes
    assert ladder_pool.rename(_arguments(renaming), grown) == 0
    assert [(check.problem_id, check.source) for check in renaming.sent[sent:]] == [("wb_s", STP)]      # only the new certificate
    assert ladder_pool.Plan(renaming.path).verdicts()["wb_s"].state == "in"


def test_the_alarm_stops_the_renaming_pass_too(renaming):
    """Fixture 10: a renamed certificate that proves a statement whose negation is verified."""
    twin_statement = "theorem wb_twin : ∀ a : ℝ, a > 0 := by\n"
    twin = Candidate("wb_twin", LEAN_WORKBOOK, TRUE_SIDE, twin_statement, "", 0, "wb_twin",
                     (Certificate("wb_twin", TRUE_SIDE, GOEDEL, twin_statement, "  exact old_lemma"),))
    renaming.steps.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in CANDIDATES + RENAMING + [twin]))
    renaming.fingerprints["wb_twin"] = 556
    assert ladder_pool.check(_arguments(renaming), CONFIG) == 0                    # the published proof fails: no alarm yet
    assert ladder_pool.rename(_arguments(renaming), CONFIG) == 3
    assert json.loads((renaming.path / "checks" / "progress.json").read_text())["state"] == "SOUNDNESS ALARM"


def test_the_tabulation_counts_the_names_lean_reports_by_source(renaming):
    assert ladder_pool.check(_arguments(renaming), CONFIG) == 0
    report = ladder_pool.names_report(ladder_pool.Plan(renaming.path), ladder_pool.renames_table(CONFIG))
    assert report["failed_certificates"] == 6 and report["naming_an_unknown_name"] == 4      # wb_b's and wb_c's name nothing
    assert report["naming_a_name_the_table_renames"] == 3
    by_name = {row["name"]: row for row in report["names"]}
    assert sorted(by_name) == ["Real.old_sqrt", "old_lemma"]                       # never the failed theorem's own name
    assert by_name["old_lemma"] == {
        "name": "old_lemma", "certificates": 3, "problems": 2,
        "by_source": {"lean_workbook/goedel": 1, "lean_workbook/stp": 1, "stp_conjecture/stp": 1},
        "written_as": {"old_lemma": 3}, "renamed_to": {"old_lemma": "new_lemma"}, "written_forms_not_in_table": []}
    # Reported in full, written without the namespace the header opens, and not in the table: listed as such.
    assert by_name["Real.old_sqrt"]["written_as"] == {"old_sqrt": 1}
    assert by_name["Real.old_sqrt"]["renamed_to"] == {} and by_name["Real.old_sqrt"]["written_forms_not_in_table"] == ["old_sqrt"]


def test_a_table_entry_without_a_new_name_is_a_configuration_error(store):
    bad = json.loads(json.dumps(CONFIG))
    bad["ladder_loop"]["certificates"]["renames"]["same"] = "same"
    with pytest.raises(ValueError, match="no new name"):
        ladder_pool.renames_table(bad)


# ------------------------------------------------------------------------------------------------ the export
def test_the_export_is_refused_before_the_pool_exists(renaming, tmp_path):
    assert ladder_pool.check(_arguments(renaming), CONFIG) == 0
    with pytest.raises(SystemExit, match="`pool` has not run"):
        ladder_pool.export(_arguments(renaming, out=tmp_path / "export"), CONFIG)
    with pytest.raises(SystemExit, match="needs --out"):
        ladder_pool.export(_arguments(renaming), CONFIG)
    assert not (tmp_path / "export").exists()


def test_the_export_carries_h_and_a_sample_of_the_pool_and_no_published_proof(renaming, tmp_path):
    """Fixtures 6 and 7: what travels to the GPU box is statements and ids; H is not in the pool sample."""
    from rlvr_lean.gpu.ladder_loop import load_problems

    config = json.loads(json.dumps(CONFIG))
    config["ladder_loop"]["base_map"]["problems"] = 3
    assert _both_passes(renaming) == 0 and ladder_pool.pool(_arguments(renaming), config) == 0
    assert [row["problem_id"] for row in renaming.steps.read_rows("base_map.jsonl")] == [
        row["problem_id"] for row in sorted(renaming.steps.read_rows("pool.jsonl"), key=lambda row: rank(config["ladder_loop"]["base_map"]["seed"], "base_map", row["problem_id"]))[:3]]
    assert ladder_pool.export(_arguments(renaming, out=tmp_path / "export"), config) == 0
    heldout, base_map, summary = load_problems(tmp_path / "export")                # the GPU step's own reader accepts it
    stored_h = renaming.steps.read_rows("heldout.jsonl")
    assert [row["problem_id"] for row in heldout] == [row["problem_id"] for row in stored_h]
    assert all(set(row) == {"problem_id", "kind", "side", "statement", "rewritten", "heldout_part", "in_present_holdout"} for row in heldout)
    assert [row["in_present_holdout"] for row in heldout if row["problem_id"] == "wb_f"] == [True]
    in_pool = {row["problem_id"] for row in renaming.steps.read_rows("pool.jsonl")}
    assert len(base_map) == 3 and {row["problem_id"] for row in base_map} <= in_pool
    assert not {row["problem_id"] for row in base_map} & {row["problem_id"] for row in heldout}
    text = "".join((tmp_path / "export" / name).read_text() for name in ("heldout.jsonl", "base_map.jsonl", "summary.json"))
    assert "new_lemma" not in text and "simp [" not in text and "push_neg" not in text and "certificate_sha" not in text
    assert summary["fixture"] is False and summary["pool"]["verified"] == 9 and summary["base_map_settings"]["problems"] == 3
    # The same store exports the same files.
    again = tmp_path / "again"
    ladder_pool.export(_arguments(renaming, out=again), config)
    assert (again / "heldout.jsonl").read_bytes() == (tmp_path / "export" / "heldout.jsonl").read_bytes()
    assert (again / "base_map.jsonl").read_bytes() == (tmp_path / "export" / "base_map.jsonl").read_bytes()


# ------------------------------------------------------------------------------------------------ the slice
def _many():
    """48 candidates: most verify as published, every fifth only once `old_lemma` is renamed, some never."""
    candidates = []
    for kind, prefix in ((LEAN_WORKBOOK, "lw"), (STP_CONJECTURE, "sc")):
        for index in range(24):
            proof = "  rw [old_lemma h]\n  linarith" if index % 5 == 0 else "  exact old_sqrt h" if index % 7 == 3 else "  nlinarith [h]"
            candidates.append(_citing(f"{prefix}_{index:02d}", (200 if kind == LEAN_WORKBOOK else 300) + index, [(GOEDEL if kind == LEAN_WORKBOOK else STP, proof)], kind=kind))
    return candidates


MANY = _many()
SLICED = json.loads(json.dumps(CONFIG))
SLICED["ladder_loop"]["slice"] = {"lean_workbook": 8, "stp": 8, "all": 16}
SLICED["ladder_loop"]["heldout"].update({"workbook_problems": 6, "stp_conjectures": 4})
SLICED["ladder_loop"]["base_map"]["problems"] = 5


def _write_many(steps, extra=()):
    steps.write_rows("candidates.jsonl", (candidate_row(candidate) for candidate in MANY + list(extra)))
    steps.write_rows("present_holdout.jsonl", [{"statement_id": name} for name in ("lw_00", "lw_01", "lw_02", "wb_z")])


@pytest.fixture
def sliced(store):
    _write_many(store.steps)
    return store


def _ids(steps, name):
    return [row["problem_id"] for row in steps.read_rows(name)]


def test_the_slice_is_checked_through_both_passes_and_nothing_else_is(sliced):
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0
    parts = ladder_pool.stored_slice(sliced.steps)
    assert [len(parts[part]) for part in (SLICE_PRESENT, SLICE_WORKBOOK, SLICE_STP, SLICE_ALL)] == [3, 8, 8, 16]
    assert not set(parts[SLICE_PRESENT]) & set(parts[SLICE_WORKBOOK]) and all(name.startswith("sc_") for name in parts[SLICE_STP])
    members = ladder_pool.members_of(parts)
    assert {check.problem_id for check in sliced.sent} == members | {"wb_z"}          # the slice and the set-aside statement, nothing else
    verdicts = ladder_pool.Plan(sliced.path).verdicts()
    assert all(verdicts[name].state != "open" for name in members)
    assert all(verdict.state == "open" for name, verdict in verdicts.items() if name not in members)
    # The renaming pass ran on what the first pass left unsettled IN THE SLICE: its marker says so.
    renamed = {row["problem_id"] for row in sliced.steps.read_rows("renamed.jsonl")}
    assert renamed and renamed <= members and all(int(name[3:]) % 5 == 0 for name in renamed)
    assert sliced.steps.done_summary(ladder_pool.RENAMED_STEP)["scope"] == ladder_pool.THE_SLICE
    assert all(verdicts[name].state == "in" and verdicts[name].certificate.renamed == ("old_lemma",) for name in renamed)
    progress = json.loads((sliced.path / "checks" / "slice_progress.json").read_text())
    assert (progress["state"], progress["problems"], progress["open"]) == ("finished", len(members), 0)


def test_the_slice_reuses_every_answer_the_bulk_check_stored(sliced):
    assert ladder_pool.check(_arguments(sliced, max_checks=30), SLICED) == 1         # the bulk check, stopped part of the way
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0
    shas = [check.sha for check in sliced.sent]
    assert len(shas) == len(set(shas))                                             # nothing was asked twice
    sent = len(sliced.sent)
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0 and len(sliced.sent) == sent     # a rerun sends nothing


def test_a_dry_run_of_the_slice_counts_and_writes_nothing(sliced, capsys):
    capsys.readouterr()
    assert ladder_pool.slice_first(_arguments(sliced, dry_run=True), SLICED) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["first_pass_open"] == counts["members"] and counts["new_checks_at_least"] == counts["members"] + 1
    assert not sliced.steps.is_done(ladder_pool.SLICE_STEP) and not (sliced.path / "steps" / "slice.jsonl").exists() and sliced.sent == []


def test_h_and_the_base_map_drawn_from_the_slice_are_the_whole_pools_draw(sliced, tmp_path_factory):
    """Spec: "A random sample of candidates, kept where verified, is a random sample of the verified: it is
    the same draw, made early"."""
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0 and ladder_pool.pool(_arguments(sliced), SLICED) == 0
    whole = SimpleNamespace(path=tmp_path_factory.mktemp("whole"))
    steps = ArtifactStore(whole.path / "steps")
    _write_many(steps)
    steps.write_rows("set_aside.jsonl", SET_ASIDE)
    assert _both_passes(whole, SLICED) == 0 and ladder_pool.pool(_arguments(whole), SLICED) == 0      # every candidate, both passes, no slice
    assert _ids(sliced.steps, "heldout.jsonl") == _ids(steps, "heldout.jsonl") and len(_ids(steps, "heldout.jsonl")) == 10
    assert _ids(sliced.steps, "base_map.jsonl") == _ids(steps, "base_map.jsonl") and len(_ids(steps, "base_map.jsonl")) == 5
    heldout = sliced.steps.read_rows("heldout.jsonl")
    assert [row["problem_id"] for row in heldout[:3]] == ladder_pool.stored_slice(sliced.steps)[SLICE_PRESENT]     # the present holdout first
    assert [row["heldout_part"] for row in heldout] == [LEAN_WORKBOOK] * 6 + [STP_CONJECTURE] * 4
    assert not set(_ids(sliced.steps, "base_map.jsonl")) & set(_ids(sliced.steps, "heldout.jsonl"))
    summary = sliced.steps.done_summary("pool")
    assert summary["drawn_from"] == "the slice" and summary["candidates_still_open"] > 0 and steps.done_summary("pool")["drawn_from"] == "the whole pool"


def test_the_pool_file_holds_what_is_verified_so_far_then_grows_and_h_stays(sliced):
    held_text = "theorem {name} (x : ℝ) (h : 0 < x) : x / 201 ≤ x := by\n"              # lw_01's statement: a present-holdout problem, so in H
    copy = Candidate("zz_copy", STP_CONJECTURE, TRUE_SIDE, held_text.format(name="zz_copy"), "", 0, "zz_copy",
                     (Certificate("zz_copy", TRUE_SIDE, STP, held_text.format(name="zz_copy"), "  nlinarith [h]"),))
    _write_many(sliced.steps, [copy])
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0 and ladder_pool.pool(_arguments(sliced), SLICED) == 0
    heldout, base_map = (sliced.path / "steps" / "heldout.jsonl").read_text(), (sliced.path / "steps" / "base_map.jsonl").read_text()
    early = set(_ids(sliced.steps, "pool.jsonl"))
    members = ladder_pool.members_of(ladder_pool.stored_slice(sliced.steps))
    assert early and early <= members and set(_ids(sliced.steps, "base_map.jsonl")) <= early and "lw_01" in heldout
    # The rest of the pool, afterwards: the bulk check skips nothing, then the whole renaming pass, then `pool` again.
    sent = len(sliced.sent)
    assert ladder_pool.check(_arguments(sliced), SLICED) == 0 and len(sliced.sent) > sent
    assert ladder_pool.rename(_arguments(sliced), SLICED) == 0 and ladder_pool.pool(_arguments(sliced), SLICED) == 0
    assert (sliced.path / "steps" / "heldout.jsonl").read_text() == heldout and (sliced.path / "steps" / "base_map.jsonl").read_text() == base_map
    late = set(_ids(sliced.steps, "pool.jsonl"))
    assert early < late and not late & set(_ids(sliced.steps, "heldout.jsonl"))
    verdicts = ladder_pool.Plan(sliced.path).verdicts()
    assert verdicts["zz_copy"].state == "in" and "zz_copy" not in late                # verified, and a copy of a member of H: never in the pool
    assert all(verdict.state != "open" for verdict in verdicts.values())
    assert sliced.steps.done_summary(ladder_pool.RENAMED_STEP)["scope"] == ladder_pool.WHOLE_POOL
    not_in_slice_renamed = [name for name, verdict in verdicts.items() if name not in members and verdict.state == "in" and verdict.certificate.renamed]
    assert not_in_slice_renamed and set(not_in_slice_renamed) <= late                  # the whole pass renamed outside the slice too
    shas = [check.sha for check in sliced.sent]
    assert len(shas) == len(set(shas))


def test_a_draw_that_comes_out_short_says_how_many_more_candidates_and_writes_nothing(sliced):
    small = json.loads(json.dumps(SLICED))
    small["ladder_loop"]["slice"] = {"lean_workbook": 2, "stp": 3, "all": 4}
    assert ladder_pool.slice_first(_arguments(sliced), small) == 0
    with pytest.raises(SystemExit) as refusal:
        ladder_pool.pool(_arguments(sliced), small)
    message = str(refusal.value)
    assert re.search(r"H is \d+ lean_workbook problems short: raise ladder_loop\.slice\.lean_workbook from 2 by about \d+", message)
    assert re.search(r"H is \d+ stp_conjecture problems short: raise ladder_loop\.slice\.stp from 3 by about \d+", message)
    assert re.search(r"the base-map sample is \d+ problems short: raise ladder_loop\.slice\.all from 4 by about \d+", message)
    assert message.endswith("Nothing was written.")
    for name in ("heldout.jsonl", "base_map.jsonl", "pool.jsonl"):
        assert not (sliced.path / "steps" / name).exists()
    assert not sliced.steps.is_done("heldout") and not sliced.steps.is_done("base_map") and not sliced.steps.is_done("pool")
    # A larger slice holds the smaller one: only the new candidates are checked, and then the draw is made.
    sent = {check.problem_id for check in sliced.sent}
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0
    newly = [check for check in sliced.sent if check.problem_id not in sent]
    assert newly and len([check.sha for check in sliced.sent]) == len({check.sha for check in sliced.sent})
    assert ladder_pool.pool(_arguments(sliced), SLICED) == 0 and len(_ids(sliced.steps, "heldout.jsonl")) == 10


def test_the_pool_waits_for_the_slice_and_only_for_the_slice(sliced):
    assert ladder_pool.slice_first(_arguments(sliced, max_checks=5), SLICED) == 1      # stopped part of the way
    with pytest.raises(SystemExit, match="renaming pass has not run"):
        ladder_pool.pool(_arguments(sliced), SLICED)
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0
    verdicts = ladder_pool.Plan(sliced.path).verdicts()
    assert any(verdict.state == "open" for verdict in verdicts.values())               # the rest of the pool is not checked
    assert ladder_pool.pool(_arguments(sliced), SLICED) == 0


def test_a_slice_stopped_between_its_passes_is_not_settled(sliced):
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0
    # Another slice in the config after the renaming pass was built for this one: `pool` refuses until `slice` ran again.
    other = json.loads(json.dumps(SLICED))
    other["ladder_loop"]["slice"]["all"] = 20
    sliced.steps.mark_done(ladder_pool.SLICE_STEP, {**sliced.steps.done_summary(ladder_pool.SLICE_STEP), "settings": ladder_pool.slice_settings(other)})
    with pytest.raises(SystemExit, match="built for another slice"):
        ladder_pool.pool(_arguments(sliced), other)


def test_the_slice_is_fixed_once_h_is_drawn_from_it(sliced):
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0 and ladder_pool.pool(_arguments(sliced), SLICED) == 0
    other = json.loads(json.dumps(SLICED))
    other["ladder_loop"]["slice"]["stp"] = 12
    with pytest.raises(SystemExit, match="H is already drawn from the slice"):
        ladder_pool.slice_first(_arguments(sliced), other)
    assert ladder_pool.slice_first(_arguments(sliced), SLICED) == 0                    # the same slice again: nothing to do


def test_the_sample_names_one_certificate_per_problem_and_stratum(store):
    plan = ladder_pool.Plan(store.path)
    chosen = ladder_pool.sample_plan(plan, dict.fromkeys(ladder_pool.STRATA, 5), seed=0)
    assert {stratum: sorted(candidate.problem_id for candidate, _ in members) for stratum, members in chosen.items()} == {
        "internlm_proofs": ["wb_a", "wb_c"], "internlm_rows_proved": [], "goedel": ["wb_a", "wb_b", "wb_f"],
        "stp_statement": ["wb_a", "wb_b", "wb_g"], "stp_conjecture": ["stp_1", "stp_2"],
        "disproved_no_variables": ["wb_d", "wb_e"], "disproved_with_variables": []}
    for stratum, members in chosen.items():
        for candidate, checks in members:
            assert [check.kind for check in checks] == ([EXACTNESS_CHECK] if candidate.side == FALSE_SIDE else []) + [CERTIFICATE_CHECK]
    assert len(ladder_pool.sample_plan(plan, {"goedel": 2}, seed=0)["goedel"]) == 2


def test_the_sample_report_reads_survival_by_stratum(store, capsys):
    arguments = argparse.Namespace(store=store.path, max_checks=600, retry_passes=0, sizes=[5] * len(ladder_pool.STRATA), seed=0, report_only=False)
    assert ladder_pool.sample(arguments, CONFIG) == 0
    report = json.loads((store.path / "checks" / "sample_report.json").read_text())["strata"]
    assert (report["goedel"]["verified"], report["goedel"]["answered"]) == (2, 3)                  # wb_b's Goedel proof fails
    assert (report["stp_statement"]["verified"], report["stp_statement"]["answered"]) == (3, 3)    # the sample asks wb_b's STP proof too
    assert report["internlm_proofs"]["statuses"] == {"verified": 1, "lean_error": 1}
    assert report["disproved_no_variables"]["statuses"] == {"verified": 1, "negation_not_exact": 1}
    assert report["goedel"]["failure_classes"] == {"other_error": 1}
    capsys.readouterr()
