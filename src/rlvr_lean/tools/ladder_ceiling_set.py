"""The ceiling's training file: published proofs of pool problems the base cannot solve.
Spec: docs/spec/ladder-loop.spec.md, "The ceiling: a labelled diagnostic", "The training file". No GPU: only the
Lean pool is used, as BACKGROUND work, and only published statements and published proofs of them are sent.

AN EXCEPTION, NOT A CHANGE OF RULE. Published proofs are certificates and not training text. The file this writes is
the input of ONE labelled diagnostic (`gpu/ladder_ceiling.py`); it holds other people's proofs, no round reads it,
and it is not distributed.

  eligible   candidates of the pool as L2 reads it (`data/ladder_l2/candidates.jsonl`: never a held-out problem, never
             one of the base map's) on the side `true`, whose pass rate the challenger predicted for the base below
             `below_predicted_rate` (`--scores`: the round-1 scores of L2's seed 0, `candidate_scores_r1.jsonl`)
  chosen     every eligible Lean Workbook problem, in the order of their ids, then a seeded draw of `stp_sample` of the
             STP ones
  proofs     of each chosen problem, its published proofs from the pool build's steps (`--candidates-store`:
             `candidates.jsonl`, then `renamed.jsonl`, the same proofs with Mathlib's lemma renames applied), without
             the repeated ones, without one that holds a forbidden token, and WITHOUT ONE TOO LONG: a proof is tried
             only if statement and proof together are at most `maximum_characters` (so that no row can be refused for
             its length on the box). At most the `proofs_tried` shortest by lines go to Lean
  the check  the file a solver's attempt is checked as (`imports_first(build_proof_source(statement, proof))`: the
             pool caches by the file's text), judged as an attempt is under Lean v4.27
  the proof  of a problem's proofs that verified, the one with the fewest lines (then the fewest characters)
  written    as many rows as the ceiling's last checkpoint (8,000), shuffled with the seed: every chosen Lean Workbook
             problem with a verified proof, then STP ones to make up the number. The first rows are the smaller dose

Everything it decides with is in `ladder_loop.ceiling` (`training_file`, and `checkpoints` for the two sizes).
`--dry-run` selects, reads the proofs and counts the checks: it sends nothing and writes nothing.

THE SAME TOOL WRITES L4's PRETRAINING FILE (`--file l4_pretrain`; spec "L4: the loop from a model pretrained on
published proofs", "The pool is cut in two", "The pretraining file"; settings `ladder_loop.l4`). NO LEAN CHECK IS MADE
FOR IT: every pool problem is in the pool because one of its published proofs verified when the pool was built, and
the pool's file names that proof by the hash of its Lean file.

  eligible   every candidate on the side `true` that is in the `pretrain` HALF of the pool (`domain/ladder_round/l4.py`:
             a hash of the problem's id); no cut by predicted rate, so no `--scores`
  chosen     all of them, Lean Workbook and STP alike, in the order of their ids
  the proof  THE CERTIFICATE THE POOL BUILD VERIFIED: of the problem's published proofs (`candidates.jsonl`, then the
             renamed ones of `renamed.jsonl`), the one whose check has the hash its row of `pool.jsonl` names
             (`certificate_sha`). The candidate, its certificates and their checks are formed by the pool build's own
             functions, as its plan forms them (`certified`)
  left out   and counted: a problem with no such certificate (above `maximum_share_without_a_match` of the chosen
             the tool STOPS: the hash is not being rebuilt as the pool build made it); one whose statement, as L2 reads
             it, is not the theorem that certificate proves; one whose proof holds a forbidden token; one whose
             statement and proof together pass `maximum_characters`
  written    one row for every other chosen problem (the ceiling file's fields, `predicted_rate` null,
             `half: "pretrain"`, and the certificate's hash and renamed names), shuffled with the seed, to
             `src/rlvr_lean/data/ladder_l4/pretraining.jsonl`, and a summary beside it. A row of the `loop` half is
             refused before anything is written
  --control  THE CONTROL, by Lean: a seeded sample of the written file's rows (`--control 300`), each built as a
             solver's attempt and checked through the pool as background work; how many verified is printed and
             stored beside the file (`pretraining.control.json`, with the ids and statuses of those that did not),
             and the tool exits non-zero under `control.minimum_share_verified`

That file too holds other people's published proofs: it is not distributed.

    PYTHONPATH=src python -m rlvr_lean.tools.ladder_ceiling_set \\
        --scores <L2 seed 0's steps>/candidate_scores_r1.jsonl --candidates-store <the pool build's store>/steps \\
        --out src/rlvr_lean/data/ladder_ceiling/training.jsonl \\
        --api-key-file <the pool's key file> --ca-file <the pool authority's certificate> [--in-flight 24 | auto] [--limit N]
    PYTHONPATH=src python -m rlvr_lean.tools.ladder_ceiling_set --file l4_pretrain \\
        --candidates-store <the pool build's store>/steps --out src/rlvr_lean/data/ladder_l4/pretraining.jsonl [--limit N]
    PYTHONPATH=src python -m rlvr_lean.tools.ladder_ceiling_set --file l4_pretrain --control 300 \\
        --out src/rlvr_lean/data/ladder_l4/pretraining.jsonl \\
        --api-key-file <the pool's key file> --ca-file <the pool authority's certificate> [--in-flight 24 | auto]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import yaml

from rlvr_lean.data.heldout_proof_lines import length_group
from rlvr_lean.data.heldout_proof_lines import proof_lines as lines_of      # a proof's lines as every report here counts them
from rlvr_lean.data.published import candidate_from_row, certificate_from_entry      # a stored candidate and a stored certificate, as the pool build reads them
from rlvr_lean.domain.ladder_round.l4 import PRETRAIN, half_of, halves, refuse_the_other_half
from rlvr_lean.domain.problem_pool.certificates import Certificate
from rlvr_lean.domain.problem_pool.selection import CERTIFICATE_CHECK, LEAN_WORKBOOK, steps
from rlvr_lean.domain.verification.lean_source import build_proof_source, find_forbidden_token, imports_first
from rlvr_lean.domain.verification.status import classify_check_result
from rlvr_lean.infrastructure.kimina_client import KiminaVerifier, LeanSnippet
from rlvr_lean.tools.ladder_pool import in_flight_argument, in_flight_text, pin_settings

PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / "config" / "experiment.yaml"
POOL_CANDIDATES = PACKAGE / "data" / "ladder_l2" / "candidates.jsonl"       # the whole pool as L2 reads it: ids, kinds, sides, statements
HELDOUT = PACKAGE / "data" / "ladder_l0" / "heldout.jsonl"
PROOF_FILES = ("candidates.jsonl", "renamed.jsonl")                          # of the pool build's steps, in the order they are read
TRUE_SIDE = "true"
POOL_FILE = "pool.jsonl"                                                     # of the pool build's steps: each pool problem with the hash of its verified certificate's Lean file
CEILING_FILE, L4_PRETRAIN = "ceiling", "l4_pretrain"                        # the files this tool writes (`--file`)
SHOWN = 5                                                                    # ids printed of the problems a rule leaves out


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------------------------- selection
def eligible(pool: Iterable[Mapping], predicted: Mapping[str, float], below: float) -> dict[str, Mapping]:
    """The pool's candidates on the side `true` whose predicted pass rate is below `below`, in the pool's order. A
    problem the scores do not hold is not eligible."""
    return {row["problem_id"]: row for row in pool if row["side"] == TRUE_SIDE and predicted.get(row["problem_id"], 1.0) < below}


def refuse_heldout(candidates: Mapping[str, Mapping], heldout: Iterable[str]) -> None:
    """No held-out problem is ever trained on: one among the candidates stops the tool (ValueError)."""
    shared = sorted(set(heldout) & set(candidates))
    if shared:
        raise ValueError(f"{len(shared)} held-out problems are among the candidates (first: {shared[0]}): refused, nothing was sent or written")


def choose(candidates: Mapping[str, Mapping], stp_sample: int, seed: int, limit: int | None = None) -> tuple[list[str], list[str], list[str]]:
    """(the eligible Lean Workbook ids, the eligible STP ids, the chosen ids): every Lean Workbook problem in the
    order of its id, then a seeded draw of `stp_sample` of the STP ones (all of them when there are fewer).
    `limit` (a trial): the first half of that many from the front and the rest from the back."""
    workbook = sorted(key for key, row in candidates.items() if row["kind"] == LEAN_WORKBOOK)
    stp = sorted(key for key, row in candidates.items() if row["kind"] != LEAN_WORKBOOK)
    generator = random.Random(seed)
    chosen = workbook + generator.sample(stp, min(stp_sample, len(stp)))
    if limit:
        chosen = chosen[:limit // 2] + chosen[-limit // 2:]
    return workbook, stp, chosen


# ------------------------------------------------------------------------------------------------- proofs
def id_at_the_start(line: str) -> str:
    """The problem id a stored line begins with (`{"problem_id": "<id>", ...`), read without parsing the line: the
    two files hold every candidate of the pool (200 MB), and only the chosen problems' lines are parsed."""
    return line[:60].split('"')[3]


def published_proofs(files: Iterable[Iterable[str]], wanted: set[str], candidates: Mapping[str, Mapping], maximum_characters: int) -> tuple[dict[str, list[str]], int]:
    """(each wanted problem's published proofs in the order the files give them, how many were left out for their
    length). A proof already held is not held twice; one with a forbidden token is left out; and one is tried only
    if statement and proof together are at most `maximum_characters`."""
    proofs: dict[str, list[str]] = defaultdict(list)
    too_long: set[tuple[str, str]] = set()
    for lines in files:
        for line in lines:
            if id_at_the_start(line) not in wanted:
                continue
            row = json.loads(line)
            statement = candidates[row["problem_id"]]["statement"]
            for certificate in row.get("certificates", []):
                proof = certificate.get("proof")
                if proof and proof not in proofs[row["problem_id"]] and find_forbidden_token(proof) is None:
                    if len(statement) + len(proof) <= maximum_characters:
                        proofs[row["problem_id"]].append(proof)
                    else:
                        too_long.add((row["problem_id"], proof))
    return proofs, len(too_long)


def checks(chosen: Sequence[str], proofs: Mapping[str, Sequence[str]], candidates: Mapping[str, Mapping], tried: int) -> list[tuple[str, str, str, str]]:
    """(the check's name, the problem, the proof, the Lean file) for at most the `tried` shortest proofs of each
    chosen problem (by lines; a stable sort, so equals keep the files' order). The file is the one a solver's
    attempt is checked as: the pool caches by its text."""
    snippets = []
    for key in chosen:
        for index, proof in enumerate(sorted(proofs.get(key, ()), key=lines_of)[:tried]):
            snippets.append((f"{key}|{index}", key, proof, imports_first(build_proof_source(candidates[key]["statement"], proof))))
    return snippets


def read_answers(snippets: Sequence[tuple[str, str, str, str]], results: Sequence[Mapping]) -> tuple[dict[str, list[str]], Counter]:
    """(each problem's proofs that verified, in the order they were tried; the checks by status). Judged as a
    solver's attempt is under Lean v4.27: a warning that says "failed" is not an error there."""
    verified: dict[str, list[str]] = defaultdict(list)
    statuses: Counter = Counter()
    for (name, key, proof, _), result in zip(snippets, results):
        outcome = classify_check_result(name, result, failed_warning_is_error=False)
        statuses[outcome.status.value] += 1
        if outcome.is_verified:
            verified[key].append(proof)
    return verified, statuses


# --------------------------------------------------------------------------------------------------- rows
def training_rows(chosen: Sequence[str], verified: Mapping[str, Sequence[str]], candidates: Mapping[str, Mapping], predicted: Mapping[str, float]) -> list[dict]:
    """One row for every chosen problem with a verified proof, in the chosen order: its statement and, of its
    verified proofs, the one with the fewest lines (then the fewest characters), ending in one newline."""
    rows = []
    for key in chosen:
        if verified.get(key):
            proof = min(verified[key], key=lambda text: (lines_of(text), len(text)))
            row = candidates[key]
            rows.append({"problem_id": key, "kind": row["kind"], "side": TRUE_SIDE, "statement": row["statement"], "proof": proof.rstrip() + "\n",
                         "proof_lines": lines_of(proof), "predicted_rate": predicted[key], "certificate_source": row.get("certificate_source")})
    return rows


def written(rows: Sequence[Mapping], total: int, seed: int) -> tuple[list, list, list]:
    """(the Lean Workbook rows, the STP rows, the rows written): every Lean Workbook row, then STP rows to make up
    `total`, shuffled with the seed (a generator of its own), and at most `total` of them."""
    book = [row for row in rows if row["kind"] == LEAN_WORKBOOK]
    other = [row for row in rows if row["kind"] != LEAN_WORKBOOK]
    kept = book + other[:max(0, total - len(book))]
    random.Random(seed).shuffle(kept)
    return book, other, kept[:total]


# ---------------------------------------------------------------------------------------------- the tool
def lean_check_settings(arguments, config: dict, lean_seconds: int):
    """The pool's settings as `tools/ladder_pool` builds them for its bulk check (the pin's endpoint and limits from
    the config, the requests in flight asked for, BACKGROUND priority and a wait that covers the pool's queue),
    with this tool's own Lean limit for one proof."""
    return dataclasses.replace(pin_settings(arguments, config, background=True), lean_timeout_seconds=lean_seconds)


def send(settings, snippets: Sequence[tuple[str, str, str, str]]) -> list[dict]:
    async def run() -> list[dict]:
        async with KiminaVerifier(settings) as verifier:
            return await verifier.check([LeanSnippet(snippet_id=name, code=code) for name, _, _, code in snippets])

    return asyncio.run(run())


def _groups(rows: Sequence[Mapping]) -> dict:
    return dict(Counter(length_group(row["proof_lines"]) for row in rows))


def eligible_for_pretraining(pool: Iterable[Mapping], seed: int) -> dict[str, Mapping]:
    """L4's eligible: the pool's candidates on the side `true` that are in the `pretrain` half, in the pool's order."""
    return {row["problem_id"]: row for row in pool if row["side"] == TRUE_SIDE and half_of(row["problem_id"], seed) == PRETRAIN}


def wanted_rows(lines: Iterable[str], wanted: set[str]) -> dict[str, dict]:
    """The stored rows of the wanted problems, by problem. The files hold every candidate of the pool: only the
    wanted problems' lines are parsed."""
    return {row["problem_id"]: row for row in (json.loads(line) for line in lines if line.strip() and id_at_the_start(line) in wanted)}


def certified(candidate_lines: Iterable[str], renamed_lines: Iterable[str], pool: Mapping[str, Mapping], wanted: set[str]) -> tuple[dict[str, Certificate], dict[str, str]]:
    """(each wanted problem's CERTIFICATE: of its published proofs, as published and as renamed, the one whose check
    has the hash its pool row names; for every wanted problem without one, why). NOTHING IS SENT TO LEAN, and no Lean
    file is written here by hand: the candidate, its certificates and their checks are formed by the pool build's own
    functions, as its plan forms them (`tools/ladder_pool.Plan`): the stored row is read by `candidate_from_row`; the
    renamed certificates follow the published ones, each read by `certificate_from_entry`; and `selection.steps`
    gives the checks, one a certificate (for a problem on the side `true`, of the file
    `certificate_source(certificate, with_fingerprint=True)`; its hash is `sha_of` that file)."""
    renamed = wanted_rows(renamed_lines, wanted)
    found, unmatched = {}, {}
    for line in candidate_lines:
        if not line.strip() or id_at_the_start(line) not in wanted:
            continue
        candidate = candidate_from_row(json.loads(line))
        key = candidate.problem_id
        if key in renamed:
            candidate = dataclasses.replace(candidate, certificates=candidate.certificates + tuple(
                certificate_from_entry(key, candidate.side, entry) for entry in renamed[key]["certificates"]))
        if key not in pool:
            unmatched[key] = "the pool's file has no row for it"
            continue
        of_its_certificates = [check for check in steps(candidate) if check.kind == CERTIFICATE_CHECK]
        matched = next((certificate for certificate, check in zip(candidate.certificates, of_its_certificates) if check.sha == pool[key]["certificate_sha"]), None)
        if matched is None:
            unmatched[key] = f"none of the checks of its {len(candidate.certificates)} certificates has the hash the pool names"
        else:
            found[key] = matched
    unmatched.update({key: "the pool build's candidates have no row for it" for key in wanted if key not in found and key not in unmatched})
    return found, unmatched


LEFT_OUT = ("statement_is_not_the_certificates_theorem", "forbidden_token", "too_long")       # the rules that leave a matched problem out, in the order they are read


def pretraining_rows(chosen: Sequence[str], found: Mapping[str, Certificate], candidates: Mapping[str, Mapping], pool: Mapping[str, Mapping], half_seed: int, seed: int,
                     maximum_characters: int) -> tuple[list[dict], dict[str, list[str]]]:
    """(L4's rows; by rule, the problems it left out). One row for every chosen problem with its certificate, in the
    ceiling file's fields (the statement is the candidate's as L2 reads it; no predicted rate), with the half it is
    of and the certificate's hash and renamed names, shuffled with the seed over the list in the chosen order. Left
    out: a problem whose statement is not the theorem its certificate proves (the file the pool build verified held
    that theorem), one whose proof holds a forbidden token, one whose statement and proof together are longer than
    `maximum_characters`. A row of the `loop` half is refused."""
    rows, left_out = [], {rule: [] for rule in LEFT_OUT}
    for key in chosen:
        certificate = found.get(key)
        if certificate is None:
            continue
        row, statement = candidates[key], candidates[key]["statement"]
        if certificate.theorem != statement:
            left_out["statement_is_not_the_certificates_theorem"].append(key)
        elif find_forbidden_token(certificate.proof) is not None:
            left_out["forbidden_token"].append(key)
        elif len(statement) + len(certificate.proof) > maximum_characters:
            left_out["too_long"].append(key)
        else:
            rows.append({"problem_id": key, "kind": row["kind"], "side": TRUE_SIDE, "statement": statement, "proof": certificate.proof.rstrip() + "\n",
                         "proof_lines": lines_of(certificate.proof), "predicted_rate": None, "certificate_source": certificate.source, "half": PRETRAIN,
                         "certificate_sha": pool[key]["certificate_sha"], "certificate_renamed": list(certificate.renamed)})
    refuse_the_other_half(rows, PRETRAIN, half_seed, "the pretraining file")
    random.Random(seed).shuffle(rows)
    return rows, left_out


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def summary_path(out: Path) -> Path:
    return out.with_name(out.stem + ".summary.json")


def control_path(out: Path) -> Path:
    return out.with_name(out.stem + ".control.json")


def build_pretraining(arguments, config: dict) -> int:
    """L4's pretraining file, whole, from the pool's own certificates: no Lean check is made."""
    settings_of = config["ladder_loop"]["l4"]
    own, half_seed = settings_of["pretraining_file"], settings_of["half_seed"]
    pool_candidates = read_jsonl(POOL_CANDIDATES)
    of_the_pool = halves((row["problem_id"] for row in pool_candidates), half_seed)
    on_true = halves((row["problem_id"] for row in pool_candidates if row["side"] == TRUE_SIDE), half_seed)
    candidates = eligible_for_pretraining(pool_candidates, half_seed)
    refuse_heldout(candidates, (row["problem_id"] for row in read_jsonl(HELDOUT)))
    chosen = sorted(candidates)
    if arguments.limit:
        chosen = chosen[:arguments.limit // 2] + chosen[-arguments.limit // 2:]
    by_kind = Counter(candidates[key]["kind"] for key in chosen)
    print(f"the pool's {len(pool_candidates)} candidates by half (seed {half_seed}): {of_the_pool} | on the side true: {on_true}", flush=True)
    print(f"eligible (the pretrain half, on the side true): {len(candidates)} | chosen {len(chosen)}: {dict(by_kind)}", flush=True)
    store, wanted = arguments.candidates_store, set(chosen)
    with (store / POOL_FILE).open(encoding="utf-8") as handle:
        pool = wanted_rows(handle, wanted)
    with (store / PROOF_FILES[0]).open(encoding="utf-8") as published, (store / PROOF_FILES[1]).open(encoding="utf-8") as renamed:
        found, unmatched = certified(published, renamed, pool, wanted)
    without = sorted(unmatched)
    print(f"with the certificate the pool build verified (matched by the hash of its check): {len(found)} of {len(chosen)} | without: {len(without)}"
          + "".join(f"\n  {key}: {unmatched[key]}" for key in without[:SHOWN]), flush=True)
    if len(without) > own["maximum_share_without_a_match"] * len(chosen):
        raise ValueError(f"{len(without)} of the {len(chosen)} chosen problems have no certificate whose check has the hash the pool names, more than "
                         f"{own['maximum_share_without_a_match']:.0%}: the hash is not being rebuilt the way the pool build made it. Nothing was written "
                         f"(first: {without[0]}: {unmatched[without[0]]})")
    by_source = Counter(found[key].source for key in chosen if key in found)
    renamed_ones = sorted(key for key in found if found[key].renamed)
    other_statement = sorted(key for key in found if candidates[key]["statement"] != pool[key]["statement"])
    other_source = sorted(key for key in found if found[key].source != pool[key]["certificate_source"] or list(found[key].renamed) != list(pool[key]["certificate_renamed"]))
    rows, left_out = pretraining_rows(chosen, found, candidates, pool, half_seed, own["seed"], own["maximum_characters"])
    print(f"matched by source: {dict(by_source)} | of them a renamed certificate: {len(renamed_ones)}", flush=True)
    print(f"statements that are not the pool row's: {len(other_statement)} | certificates whose source or renamed names are not the pool row's: {len(other_source)}", flush=True)
    for rule, what in (("statement_is_not_the_certificates_theorem", "the statement is not the theorem its certificate proves"),
                       ("forbidden_token", "the proof holds a forbidden token"),
                       ("too_long", f"statement and proof are over {own['maximum_characters']} characters")):
        print(f"left out, {what}: {len(left_out[rule])}" + (f" (first: {', '.join(left_out[rule][:SHOWN])})" if left_out[rule] else ""), flush=True)
    print(f"rows: {len(rows)} of {len(chosen)} chosen: {dict(Counter(row['kind'] for row in rows))}", flush=True)
    lengths = sorted(row["proof_lines"] for row in rows)
    if rows:
        print(f"proof lines: median {lengths[len(lengths) // 2]}, mean {sum(lengths) / len(lengths):.2f}, longest {lengths[-1]} | by group {_groups(rows)}", flush=True)
    if arguments.dry_run:
        print("dry run: nothing was sent and nothing was written", flush=True)
        return 0
    out = arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    out.write_bytes(content)
    summary = {
        "spec": "docs/spec/ladder-loop.spec.md, L4: the loop from a model pretrained on published proofs (the pretraining file)",
        "file": L4_PRETRAIN, "rows": len(rows), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
        "half_seed": half_seed, "seed": own["seed"], "limit": arguments.limit, "settings": dict(own),
        "candidates_by_half": of_the_pool, "candidates_on_the_side_true_by_half": on_true, "eligible": len(candidates), "chosen": len(chosen),
        "chosen_by_kind": dict(by_kind),
        "the_proof": "the certificate the pool build verified, taken from the problem's published proofs by the hash of its check (pool.jsonl): no Lean check was made",
        "matched": len(found), "matched_by_source": dict(by_source), "matched_with_a_renamed_certificate": len(renamed_ones),
        "without_a_match": {"problems": len(without), "first": {key: unmatched[key] for key in without[:SHOWN]}},
        "statements_that_are_not_the_pool_rows": len(other_statement), "certificates_whose_source_or_renamed_names_are_not_the_pool_rows": len(other_source),
        "left_out": {rule: {"problems": len(left_out[rule]), "first": left_out[rule][:SHOWN]} for rule in LEFT_OUT},
        "rows_by_kind": dict(Counter(row["kind"] for row in rows)), "rows_by_half": dict(Counter(row["half"] for row in rows)),
        "rows_by_certificate_source": dict(Counter(row["certificate_source"] for row in rows)),
        "rows_with_a_renamed_certificate": sum(bool(row["certificate_renamed"]) for row in rows),
        "proof_lines": {"by_length_group": _groups(rows), "median": lengths[len(lengths) // 2] if lengths else None,
                        "mean": round(sum(lengths) / len(lengths), 2) if lengths else None, "longest": lengths[-1] if lengths else None},
        "longest_row_characters": max((len(row["statement"]) + len(row["proof"]) for row in rows), default=None),
    }
    _write_json(summary_path(out), summary)
    print(f"written: {len(rows)} rows to {out} ({len(content)} bytes) and {summary_path(out)}. THE CONTROL IS NOT MADE YET: --control {own['control']['rows']}", flush=True)
    return 0


def control_sample(rows: Sequence[Mapping], size: int, seed: int) -> list[Mapping]:
    """A seeded sample of `size` rows of the file (all of them when it holds no more), in the file's order."""
    picked = sorted(random.Random(seed).sample(range(len(rows)), min(size, len(rows))))
    return [rows[index] for index in picked]


def control(arguments, config: dict, check: Callable[[Sequence[tuple[str, str, str, str]]], Sequence[Mapping]] | None = None) -> int:
    """THE CONTROL of L4's pretraining file, by Lean: a seeded sample of the WRITTEN file's rows, each built as a
    solver's attempt and judged as one. What verified is printed and stored beside the file; the exit code is 1
    when less than `control.minimum_share_verified` of the sample verified. `check`: what answers the checks
    (default: the Lean pool, as background work; a test hands in its own)."""
    own = config["ladder_loop"]["l4"]["pretraining_file"]["control"]
    out, size = arguments.out, arguments.control or own["rows"]
    content = out.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    sample = control_sample(rows, size, own["seed"])
    if not sample:
        raise ValueError(f"{out} holds no row: there is nothing to check")
    snippets = [(row["problem_id"], row["problem_id"], row["proof"], imports_first(build_proof_source(row["statement"], row["proof"]))) for row in sample]
    print(f"the control: {len(sample)} of the {len(rows)} rows of {out} (seed {own['seed']}), each checked as a solver's attempt is", flush=True)
    if arguments.dry_run:
        print("dry run: nothing was sent and nothing was written", flush=True)
        return 0
    if check is None:
        settings = lean_check_settings(arguments, config, own["lean_seconds"])
        print(f"the Lean pool: {in_flight_text(settings)}, priority {settings.priority}, Lean timeout {settings.lean_timeout_seconds} s", flush=True)
        results = send(settings, snippets)
    else:
        results = check(snippets)
    outcomes = [classify_check_result(name, result, failed_warning_is_error=False) for (name, _, _, _), result in zip(snippets, results)]
    statuses = Counter(outcome.status.value for outcome in outcomes)
    verified = sum(outcome.is_verified for outcome in outcomes)
    not_verified = [{"problem_id": row["problem_id"], "status": outcome.status.value, "kind": row["kind"], "certificate_source": row.get("certificate_source"),
                     "certificate_renamed": row.get("certificate_renamed"), "proof_lines": row["proof_lines"]}
                    for row, outcome in zip(sample, outcomes) if not outcome.is_verified]
    share, minimum = verified / len(sample), own["minimum_share_verified"]
    passes = share >= minimum
    _write_json(control_path(out), {
        "spec": "docs/spec/ladder-loop.spec.md, L4: the loop from a model pretrained on published proofs (the pretraining file: a control, by Lean)",
        "file": L4_PRETRAIN, "control_of": out.name, "sha256": hashlib.sha256(content).hexdigest(), "rows_of_the_file": len(rows), "sampled": len(sample),
        "seed": own["seed"], "lean_seconds": own["lean_seconds"], "checked_as": "a solver's attempt: imports_first(build_proof_source(statement, proof)), judged as one under v4.27",
        "verified": verified, "share_verified": round(share, 5), "minimum_share_verified": minimum, "passes": passes, "statuses": dict(statuses),
        "not_verified": not_verified})
    print(f"check statuses: {dict(statuses)}", flush=True)
    for entry in not_verified:
        print(f"  not verified: {entry['problem_id']}: {entry['status']} ({entry['certificate_source']}, {entry['proof_lines']} lines"
              + (f", renamed {entry['certificate_renamed']}" if entry["certificate_renamed"] else "") + ")", flush=True)
    print(f"THE CONTROL {'PASSES' if passes else 'FAILS'}: {verified} of {len(sample)} verified ({share:.1%}); at least {minimum:.0%} is asked"
          + ("" if passes else ". THE FILE IS NOT TO BE USED") + f". Stored in {control_path(out)}", flush=True)
    return 0 if passes else 1


def build(arguments, config: dict, check: Callable[[Sequence[tuple[str, str, str, str]]], Sequence[Mapping]] | None = None) -> int:
    """The whole build. `check`: what answers the checks (default: the Lean pool; a test hands in its own)."""
    if getattr(arguments, "file", CEILING_FILE) == L4_PRETRAIN:      # L4's file is made with no Lean check; its control is one
        return control(arguments, config, check) if getattr(arguments, "control", None) is not None else build_pretraining(arguments, config)
    ceiling = config["ladder_loop"]["ceiling"]
    own, (small, total) = ceiling["training_file"], ceiling["checkpoints"]
    predicted = {row["problem_id"]: row["predicted_rate"] for row in read_jsonl(arguments.scores)}
    candidates = eligible(read_jsonl(POOL_CANDIDATES), predicted, own["below_predicted_rate"])
    refuse_heldout(candidates, (row["problem_id"] for row in read_jsonl(HELDOUT)))
    workbook, stp, chosen = choose(candidates, own["stp_sample"], own["seed"], arguments.limit)
    print(f"eligible: {len(workbook)} Lean Workbook, {len(stp)} STP | chosen {len(chosen)}", flush=True)

    def stored_lines():
        for name in PROOF_FILES:
            with (arguments.candidates_store / name).open(encoding="utf-8") as handle:
                yield handle

    proofs, too_long = published_proofs(stored_lines(), set(chosen), candidates, own["maximum_characters"])
    print("problems with a published proof to try:", sum(1 for key in chosen if proofs.get(key)), "| proofs:", sum(len(proofs.get(key, ())) for key in chosen), flush=True)
    print(f"published proofs not tried for their length (statement and proof over {own['maximum_characters']} characters): {too_long}", flush=True)
    snippets = checks(chosen, proofs, candidates, own["proofs_tried"])
    print("checks to send:", len(snippets), flush=True)
    if arguments.dry_run:
        print("dry run: nothing was sent and nothing was written", flush=True)
        return 0
    if check is None:
        settings = lean_check_settings(arguments, config, own["lean_seconds"])
        print(f"the Lean pool: {in_flight_text(settings)}, priority {settings.priority}, Lean timeout {settings.lean_timeout_seconds} s", flush=True)
        results = send(settings, snippets)
    else:
        results = check(snippets)
    verified, statuses = read_answers(snippets, results)
    print("check statuses:", dict(statuses), flush=True)
    book, other, kept = written(training_rows(chosen, verified, candidates, predicted), total, own["seed"])
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    lengths = sorted(row["proof_lines"] for row in kept)
    print(f"verified: {len(book)} Lean Workbook, {len(other)} STP | written {len(kept)} ({sum(row['kind'] == LEAN_WORKBOOK for row in kept)} Lean Workbook) "
          f"to {arguments.out} ({arguments.out.stat().st_size} bytes)", flush=True)
    if kept:
        print(f"proof lines: median {lengths[len(lengths) // 2]}, mean {sum(lengths) / len(lengths):.1f}, p90 {lengths[int(.9 * len(lengths))]} | by group {_groups(kept)}",
              flush=True)
        first = kept[:small]
        print(f"the first {small}: {sum(row['kind'] == LEAN_WORKBOOK for row in first)} Lean Workbook | by group {_groups(first)}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--file", choices=(CEILING_FILE, L4_PRETRAIN), default=CEILING_FILE,
                        help="which file is written: the ceiling's training file (the default), or L4's pretraining file")
    parser.add_argument("--scores", type=Path, default=None,
                        help="the pass rate the challenger predicted for the base, by problem: candidate_scores_r1.jsonl of L2's seed 0 (read only); "
                             "the ceiling's file needs it, L4's pretraining file reads no score")
    parser.add_argument("--candidates-store", type=Path, default=None,
                        help="the pool build's steps directory (<store>/steps of tools/ladder_pool): candidates.jsonl and renamed.jsonl, and for L4's "
                             "file pool.jsonl, read only")
    parser.add_argument("--control", type=int, nargs="?", const=0, default=None, metavar="ROWS",
                        help="L4's file only: do not write it; check a seeded sample of this many of its rows (--out) by Lean, as solver's attempts, "
                             "store what verified beside it and exit non-zero under the share asked (with no number: ladder_loop.l4.pretraining_file.control.rows)")
    parser.add_argument("--out", type=Path, default=None, help="the training file to write (src/rlvr_lean/data/ladder_ceiling/training.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="a trial: only this many of the chosen problems (half from the front, the rest from the back)")
    parser.add_argument("--api-key-file", type=Path, help="the pool's key; read, never printed")
    parser.add_argument("--ca-file", type=Path, help="the pool authority's certificate")
    parser.add_argument("--in-flight", type=in_flight_argument, default=None,
                        help="requests in flight: a number (default: lean.pool.concurrent_requests), or `auto` to follow the size the pool states. "
                             "The checks are background work: the pool takes them only when no solver's check is waiting")
    parser.add_argument("--dry-run", action="store_true", help="select, read the proofs and count the checks; send nothing, write nothing")
    arguments = parser.parse_args(argv)
    if arguments.file == CEILING_FILE and arguments.scores is None:
        parser.error("the ceiling's file needs --scores")
    if arguments.control is not None:
        if arguments.file != L4_PRETRAIN or arguments.control < 0 or arguments.out is None:
            parser.error("--control is of L4's file (--file l4_pretrain), takes a number of rows and needs the written file (--out)")
        if not arguments.dry_run and (arguments.api_key_file is None or arguments.ca_file is None):
            parser.error("without --dry-run the control needs --api-key-file and --ca-file")
        return build(arguments, yaml.safe_load(arguments.config.read_text()))
    if arguments.candidates_store is None:
        parser.error("the following arguments are required: --candidates-store")
    if arguments.file == L4_PRETRAIN:           # no Lean check is made for this file: it needs no key
        if not arguments.dry_run and arguments.out is None:
            parser.error("without --dry-run the tool needs --out")
        return build(arguments, yaml.safe_load(arguments.config.read_text()))
    if not arguments.dry_run and (arguments.out is None or arguments.api_key_file is None or arguments.ca_file is None):
        parser.error("without --dry-run the tool needs --out, --api-key-file and --ca-file")
    return build(arguments, yaml.safe_load(arguments.config.read_text()))


if __name__ == "__main__":
    sys.exit(main())
