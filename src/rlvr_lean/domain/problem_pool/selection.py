"""From published answers to the pool: which checks a problem needs, when it is settled, and what is held out.
Spec: docs/spec/ladder-loop.spec.md, "A problem, and the pool" and "Held-out sets".

Pure rules. A CANDIDATE is a statement with a published answer and its certificates in the order they are
tried. It enters the pool only with a certificate our Lean verified; a known-false one also needs its built
negation shown exact. A problem verified on both sides is a soundness alarm, never a pool row.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

from rlvr_lean.domain.conjecturing.dedup import StatementDeduplicator, normalize_statement
from rlvr_lean.domain.conjecturing.negation import build_negation_exactness_source
from rlvr_lean.domain.conjecturing.statement_checks import build_compile_check_source, fingerprint_command
from rlvr_lean.domain.problem_pool.certificates import FALSE_SIDE, TRUE_SIDE, Certificate, certificate_source, sha_of
from rlvr_lean.domain.verification.lean_source import theorem_name_of

LEAN_WORKBOOK, STP_CONJECTURE = "lean_workbook", "stp_conjecture"

# The kinds of check, and what each one's answer decides.
CERTIFICATE_CHECK = "certificate"       # verified: the problem has its answer
EXACTNESS_CHECK = "exactness"           # compiles: the built negation IS the negation of the statement
STATEMENT_CHECK = "statement"           # compiles: a set-aside statement elaborates under the pin

IN, OUT, OPEN = "in", "out", "open"

# Why a candidate that was checked is not in the pool.
NEGATION_NOT_EXACT = "negation_not_exact"
NO_CERTIFICATE_VERIFIES = "no_certificate_verifies"


class SoundnessAlarm(Exception):
    """Lean verified a proof of a statement AND of its negation. Nothing built on these checks may be trusted
    until it is explained (a server fault, a Lean bug, a wrong negation): the run stops."""


@dataclass(frozen=True)
class Candidate:
    problem_id: str
    kind: str                            # LEAN_WORKBOOK or STP_CONJECTURE
    side: str                            # what the published answer says: TRUE_SIDE or FALSE_SIDE
    statement: str                       # `theorem ... := by\n`, in the pin's notation
    statement_published: str             # as published, before any rewrite
    rewritten: int                       # places the pin's rewrite changed in the statement
    group: str                           # problems of one group are one problem written twice (natural language)
    certificates: tuple[Certificate, ...]   # in the order they are tried


@dataclass(frozen=True)
class Check:
    sha: str
    kind: str
    problem_id: str
    source: str                          # the certificate's published source; "" for the other kinds
    lean_file: str


def _check(kind: str, problem_id: str, source: str, lean_file: str) -> Check:
    return Check(sha_of(lean_file), kind, problem_id, source, lean_file)


def exactness_source(statement: str) -> str:
    """The exactness check of a statement's built negation, followed by the statement's fingerprint (the
    duplicate check's second key: a negation's certificate proves another theorem and cannot print it)."""
    return build_negation_exactness_source(statement) + "\n" + fingerprint_command(theorem_name_of(statement))


def steps(candidate: Candidate) -> list[Check]:
    """The checks that can settle a candidate, in order. Known true: its certificates, each also printing the
    statement's fingerprint. Known false: first that the built negation is exact, then its certificates."""
    checks = []
    if candidate.side == FALSE_SIDE:
        checks.append(_check(EXACTNESS_CHECK, candidate.problem_id, "", exactness_source(candidate.statement)))
    for certificate in candidate.certificates:
        source = certificate_source(certificate, with_fingerprint=candidate.side == TRUE_SIDE)
        checks.append(_check(CERTIFICATE_CHECK, candidate.problem_id, certificate.source, source))
    return checks


def statement_check(statement_id: str, statement: str) -> Check:
    return _check(STATEMENT_CHECK, statement_id, "", build_compile_check_source(statement))


@dataclass(frozen=True)
class Verdict:
    state: str                           # IN, OUT or OPEN
    reason: str = ""                     # for OUT
    certificate: Certificate | None = None
    certificate_sha: str = ""
    fingerprint: int | None = None
    next_check: Check | None = None      # for OPEN: the one check to run now
    checks_answered: int = 0


def settle(candidate: Candidate, checks: list[Check], answers: Mapping[str, Mapping], abandoned: Iterable[str] = ()) -> Verdict:
    """Where a candidate stands given the answers so far (sha -> stored row). The first certificate that
    verifies settles it IN and later ones are never asked. A check in `abandoned` (it crashed its worker in
    every run allowed) counts as failed. A candidate with a check still unanswered is OPEN."""
    abandoned = set(abandoned)
    fingerprint, answered = None, 0
    certificates = iter(candidate.certificates)
    for check in checks:
        certificate = next(certificates) if check.kind == CERTIFICATE_CHECK else None
        row = answers.get(check.sha)
        if row is None:
            if check.sha not in abandoned:
                return Verdict(OPEN, next_check=check, checks_answered=answered)
            if check.kind == EXACTNESS_CHECK:       # exactness never shown: no certificate may stand in for it
                return Verdict(OUT, reason=NEGATION_NOT_EXACT, checks_answered=answered)
            continue
        answered += 1
        if check.kind == EXACTNESS_CHECK:
            if not row.get("compiles"):
                return Verdict(OUT, reason=NEGATION_NOT_EXACT, checks_answered=answered)
            fingerprint = row.get("fingerprint")
        elif row.get("status") == "verified":
            return Verdict(IN, certificate=certificate, certificate_sha=check.sha, checks_answered=answered,
                           fingerprint=row.get("fingerprint") if candidate.side == TRUE_SIDE else fingerprint)
    return Verdict(OUT, reason=NO_CERTIFICATE_VERIFIES, checks_answered=answered)


def pool_row(candidate: Candidate, verdict: Verdict) -> dict:
    """One pool problem as stored. The certificate is named, not copied: it is a published proof, and a
    published proof is never a training target (the pool file is what later stages read)."""
    return {"problem_id": candidate.problem_id, "kind": candidate.kind, "side": candidate.side,
            "statement": candidate.statement, "statement_published": candidate.statement_published,
            "rewritten": candidate.rewritten, "group": candidate.group, "type_fingerprint": verdict.fingerprint,
            "certificate_source": verdict.certificate.source, "certificate_sha": verdict.certificate_sha,
            "certificate_renamed": list(verdict.certificate.renamed)}


def raise_on_both_sides(rows: Iterable[Mapping]) -> None:
    """Raise SoundnessAlarm if two verified rows are one statement (by text or by elaborated type) answered
    both ways."""
    by_text: dict[str, Mapping] = {}
    by_type: dict[int, Mapping] = {}
    for row in rows:
        keys = [(by_text, normalize_statement(row["statement"]))]
        if row.get("type_fingerprint") is not None:
            keys.append((by_type, row["type_fingerprint"]))
        for seen, key in keys:
            other = seen.setdefault(key, row)
            if other["side"] != row["side"]:
                raise SoundnessAlarm(f"{other['problem_id']} is verified {other['side']} and {row['problem_id']} is verified "
                                     f"{row['side']}, and they are one statement: a proof of both sides")


# ------------------------------------------------------------------------------------------------ held out
def rank(seed: int, label: str, key: str) -> str:
    """The seeded order every draw here uses: a content hash, so a rerun reproduces it exactly."""
    return hashlib.sha256(f"{seed}:{label}:{key}".encode()).hexdigest()


@dataclass(frozen=True)
class HeldOut:
    heldout: list[dict]                  # H: the rows held out, each with `heldout_part`
    pool: list[dict]                     # what stays in the pool
    counts: dict


def draw_heldout(rows: list[dict], present_holdout: set[str], set_aside: list[dict], seed: int,
                 workbook_problems: int, stp_conjectures: int) -> HeldOut:
    """H and the pool that remains, a function of (seed, the verified rows) and nothing else.

    H's Lean Workbook part takes the present holdout's verified problems first, then a seeded draw; its STP
    part is a seeded draw. No two members of H are one problem (same group, same normalized text, or same
    elaborated type). The pool is `pool_apart_from` H and the set-aside statements."""
    def order(label: str, kind: str, first: set[str] = frozenset()) -> list[dict]:
        chosen = [row for row in rows if row["kind"] == kind]
        return sorted(chosen, key=lambda row: (row["problem_id"] not in first, rank(seed, label, row["problem_id"])))

    members, groups, heldout = StatementDeduplicator(), set(), []
    counts = {"heldout_workbook_from_present_holdout": 0}

    def take(candidates: list[dict], wanted: int, part: str) -> None:
        taken = 0
        for row in candidates:
            if taken >= wanted:
                break
            if row["group"] in groups or not members.add(row["statement"], row.get("type_fingerprint")):
                continue
            groups.add(row["group"])
            heldout.append({**row, "heldout_part": part})
            taken += 1
            counts["heldout_workbook_from_present_holdout"] += part == LEAN_WORKBOOK and row["problem_id"] in present_holdout
        counts[f"heldout_{part}"] = taken
        counts[f"heldout_{part}_short_by"] = wanted - taken

    take(order("heldout_workbook", LEAN_WORKBOOK, present_holdout), workbook_problems, LEAN_WORKBOOK)
    take(order("heldout_stp", STP_CONJECTURE), stp_conjectures, STP_CONJECTURE)
    pool, dropped = pool_apart_from(rows, heldout, set_aside)
    return HeldOut(heldout=heldout, pool=pool, counts={**counts, **dropped})


class _Barred:
    """What may never be in the pool: a member of H, and any row that is one problem with a member of H or with
    a set-aside statement (same group, same normalized text, or same elaborated type)."""

    def __init__(self, heldout: list[dict], set_aside: list[dict]) -> None:
        self.held_ids = {row["problem_id"] for row in heldout}
        self.groups = {row["group"] for row in heldout + set_aside}
        self.texts = {normalize_statement(row["statement"]) for row in heldout + set_aside}
        self.types = {row["type_fingerprint"] for row in heldout + set_aside if row.get("type_fingerprint") is not None}

    def same_problem(self, row: Mapping) -> bool:
        return (row["group"] in self.groups or normalize_statement(row["statement"]) in self.texts
                or (row.get("type_fingerprint") is not None and row["type_fingerprint"] in self.types))


def pool_apart_from(rows: list[dict], heldout: list[dict], set_aside: list[dict], kept_first: Iterable[str] = ()) -> tuple[list[dict], dict]:
    """(the pool, counts): the verified rows minus H, minus every row that is one problem with a member of H or
    with a set-aside statement, minus every later copy of a row the pool already holds. H is given, never
    drawn here: verified rows added after H was fixed go through the same rule against the H that was written.
    `kept_first` (the base-map sample, once drawn) are the rows a copy is measured against first, so a problem
    that was drawn for the base map is never the copy that is dropped."""
    barred = _Barred(heldout, set_aside)
    first = set(kept_first)
    kept, kept_ids = StatementDeduplicator(), set()
    counts = {"dropped_same_problem_as_held_out_or_set_aside": 0, "dropped_copy_within_pool": 0}
    for row in [row for row in rows if row["problem_id"] in first] + [row for row in rows if row["problem_id"] not in first]:
        if row["problem_id"] in barred.held_ids:
            continue
        if barred.same_problem(row):
            counts["dropped_same_problem_as_held_out_or_set_aside"] += 1
        elif not kept.add(row["statement"], row.get("type_fingerprint")):
            counts["dropped_copy_within_pool"] += 1
        else:
            kept_ids.add(row["problem_id"])
    pool = [row for row in rows if row["problem_id"] in kept_ids]         # in the rows' own order, whoever was kept first
    counts["pool"] = len(pool)
    return pool, counts


# ------------------------------------------------------------------------------------------------ the slice
# Spec: "The held-out sets do not wait for the whole pool". A seeded slice of candidates is taken through both
# passes first. Its Lean Workbook and STP samples are PREFIXES of the very orders `draw_heldout` draws in, and
# its all-candidates sample a prefix of the base map's, so the first verified of a prefix are the first
# verified of the whole order: H and the base-map sample drawn from the slice are the ones the whole pool
# would have given. It is the same draw, made early.
SLICE_PRESENT, SLICE_WORKBOOK, SLICE_STP, SLICE_ALL = "present_holdout", "lean_workbook", "stp", "all"
SLICE_PARTS = (SLICE_PRESENT, SLICE_WORKBOOK, SLICE_STP, SLICE_ALL)
BASE_MAP_LABEL = "base_map"


def slice_parts(candidates: Iterable[tuple[str, str]], present_holdout: set[str], heldout_seed: int, base_map_seed: int,
                sizes: Mapping[str, int]) -> dict[str, list[str]]:
    """The slice's four parts, each a list of problem ids in its seeded order, from (problem id, kind) pairs:
    the present holdout's candidates; the first `sizes[lean_workbook]` other Lean Workbook candidates and the
    first `sizes[stp]` STP candidates of the held-out orders; the first `sizes[all]` of all candidates in the
    base map's order."""
    candidates = list(candidates)
    workbook = [key for key, kind in candidates if kind == LEAN_WORKBOOK and key not in present_holdout]
    conjectures = [key for key, kind in candidates if kind == STP_CONJECTURE]
    return {
        SLICE_PRESENT: sorted((key for key, kind in candidates if key in present_holdout), key=lambda key: rank(heldout_seed, "heldout_workbook", key)),
        SLICE_WORKBOOK: sorted(workbook, key=lambda key: rank(heldout_seed, "heldout_workbook", key))[:sizes[SLICE_WORKBOOK]],
        SLICE_STP: sorted(conjectures, key=lambda key: rank(heldout_seed, "heldout_stp", key))[:sizes[SLICE_STP]],
        SLICE_ALL: sorted((key for key, _ in candidates), key=lambda key: rank(base_map_seed, BASE_MAP_LABEL, key))[:sizes[SLICE_ALL]],
    }


def draw_base_map(rows: Mapping[str, dict], order: Iterable[str], heldout: list[dict], set_aside: list[dict], problems: int) -> tuple[list[dict], dict]:
    """(the base-map sample, counts): the first `problems` VERIFIED problems of `order` that are neither in H
    nor one problem with a member of H or with a set-aside statement, nor a copy of a problem already drawn.
    `rows` holds the verified rows by problem id; an id it does not hold did not verify."""
    barred = _Barred(heldout, set_aside)
    kept, sample = StatementDeduplicator(), []
    counts = {"considered": 0, "not_verified": 0, "in_h": 0, "same_problem_as_held_out_or_set_aside": 0, "copy_within_the_sample": 0}
    for problem_id in order:
        if len(sample) >= problems:
            break
        counts["considered"] += 1
        row = rows.get(problem_id)
        if row is None:
            counts["not_verified"] += 1
        elif problem_id in barred.held_ids:
            counts["in_h"] += 1
        elif barred.same_problem(row):
            counts["same_problem_as_held_out_or_set_aside"] += 1
        elif not kept.add(row["statement"], row.get("type_fingerprint")):
            counts["copy_within_the_sample"] += 1
        else:
            sample.append(row)
    counts.update({"base_map": len(sample), "base_map_short_by": problems - len(sample)})
    return sample, counts


def more_candidates_needed(short_by: int, taken: int, considered: int, margin: float = 1.25) -> int:
    """How many MORE candidates a part of the slice needs for `short_by` more problems, at the share of its
    candidates that were taken so far, with a margin (at least `short_by` itself)."""
    share = taken / considered if considered and taken else 0.5
    return max(short_by, int(short_by / share * margin) + 1)
