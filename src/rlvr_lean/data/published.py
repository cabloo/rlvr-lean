"""The published answers the ladder loop's pool is built from: where they are pinned, how they are read, and
which problems they answer. Spec: docs/spec/ladder-loop.spec.md, L0 and "A problem, and the pool".

Four files of three publishers (config `ladder_loop.sources`), each pinned by repository revision AND by
SHA-256: a file with another hash is refused. `build_candidates` turns them into CANDIDATES (a statement, the
side its published answer takes, its certificates in the order they are tried) and counts everything it leaves
out, by reason. Nothing here asks Lean anything.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from rlvr_lean.domain.conjecturing.dedup import content_id, normalize_statement, strip_lean_comments
from rlvr_lean.domain.conjecturing.parsing import rename_theorem
from rlvr_lean.domain.problem_pool.certificates import (
    FALSE_SIDE,
    GOEDEL,
    HAS_HYPOTHESES,
    INTERNLM_PROOFS,
    INTERNLM_ROWS,
    STP,
    TRUE_SIDE,
    UNPARSED,
    Certificate,
    hypotheses_reading,
    negation_certificate,
    negation_of,
    proof_from_rows,
    published_file_proof,
    stp_proof,
    stp_statement,
    tactic_block,
    workbook_statement,
)
from rlvr_lean.domain.problem_pool.selection import LEAN_WORKBOOK, STP_CONJECTURE, Candidate, rank
from rlvr_lean.domain.verification.lean_source import find_forbidden_token, theorem_name_of
from rlvr_lean.domain.verification.pin import LeanPin

# Why a statement with a published answer is not a candidate.
PUBLISHED_BOTH_WAYS = "published_both_ways"                 # proved in one file, disproved in another: it says nothing
DISPROOF_WITH_HYPOTHESES = "disproof_with_hypotheses"       # the published disproof is not of the exact negation
DISPROOF_UNPARSED = "disproof_unparsed"                     # its statement could not be split into binders and type
NEGATION_WRAPPER_NOT_BUILT = "negation_wrapper_not_built"   # variables that are not explicit `(x : T)` binders
ROWS_NOT_ONE_CHAIN = "rows_not_one_chain"                   # a published disproof whose rows are not one chain
STATEMENT_REJECTED_LEXICALLY = "statement_rejected_lexically"
NO_USABLE_CERTIFICATE = "no_usable_certificate"             # every published proof was unreadable or filtered


@dataclass(frozen=True)
class PublishedFile:
    source: str
    repo: str
    revision: str
    path: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{self.path}"

    def local(self, directory: Path) -> Path:
        return directory / self.source / self.revision[:8] / self.path.replace("/", "__")


def published_files(config: dict) -> list[PublishedFile]:
    return [PublishedFile(source, entry["repo"], entry["revision"], path, sha256)
            for source, entry in config["ladder_loop"]["sources"].items() for path, sha256 in entry["files"].items()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(published: PublishedFile, directory: Path, download: bool = True) -> Path:
    """The local copy of a pinned file, its SHA-256 checked. Missing: downloaded from the pinned revision
    (or refused, with `download=False`). A copy with another hash is refused and left where it is, so it can
    be looked at; nothing is ever read from it."""
    path = published.local(directory)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing: run `ladder_pool fetch` first")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(published.url, temporary)
        temporary.replace(path)
    found = file_sha256(path)
    if found != published.sha256:
        raise ValueError(f"{path} has SHA-256 {found}, and {published.repo}@{published.revision[:8]}:{published.path} is "
                         f"pinned to {published.sha256}: refused. Delete the file to download the pinned one again.")
    return path


def _parquet():
    import pyarrow.parquet as parquet       # the `data` dependency group; nothing else in the package needs it

    return parquet


def read_parquet(path: Path, columns: list[str]) -> list[dict]:
    return _parquet().read_table(path, columns=columns).to_pylist()


# ------------------------------------------------------------------------------------------------ the build
@dataclass
class Build:
    candidates: list[Candidate]
    set_aside: list[dict]                          # the present holdout's statements with no published answer
    excluded: list[dict]                           # {"problem_id", "reason"}
    counts: dict
    stp_index: list[dict]                          # every STP conjecture: id, rank, where its shortest proofs are
    present_holdout: set[str] = field(default_factory=set)


def _group(natural_language: str) -> str:
    return "nl_" + hashlib.sha256(natural_language.strip().encode()).hexdigest()[:16]


def _shortest(proofs: list[str], limit: int) -> list[str]:
    return sorted(set(proofs), key=lambda proof: (len(proof), proof))[:limit]


def _stp_tag(tag: str) -> str:
    return tag.strip("[]'\" ")


def _stp_shards(files: dict[str, Path]) -> list[Path]:
    return [path for name, path in sorted(files.items()) if name.startswith("stp:")]


def scan_stp(shards: list[Path], workbook_by_text: dict[str, list[str]], minif2f_texts: set[str], per_source: int,
             counts: Counter, log: Callable[[str], None]) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    """Pass 1 over STP: for every Lean Workbook problem it proves and every conjecture, where its
    `per_source` shortest proofs are, as (proof length, shard, row). No proof text is kept."""
    import pyarrow.compute as compute

    references: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    key_of_prompt: dict[str, tuple[str, str] | None] = {}
    for shard_number, shard in enumerate(shards):
        table = _parquet().read_table(shard, columns=["prompt", "target", "tag"])
        lengths = compute.utf8_length(table.column("target")).to_pylist()
        prompts, tags = table.column("prompt").to_pylist(), table.column("tag").to_pylist()
        del table
        for row_number, (prompt, tag, length) in enumerate(zip(prompts, tags, lengths)):
            tag = _stp_tag(tag)
            counts[f"stp_rows_{tag}"] += 1
            if tag not in ("statement", "conjecture"):
                continue
            if prompt in key_of_prompt:
                key = key_of_prompt[prompt]
            else:
                statement = stp_statement(prompt)
                if statement is None:
                    key = None
                    counts[f"stp_prompts_unreadable_{tag}"] += 1
                else:
                    text = normalize_statement(statement)
                    if text in workbook_by_text:
                        # Two workbook names can state one text: the proof belongs to the name STP gave it.
                        names, own = workbook_by_text[text], theorem_name_of(statement)
                        key = (LEAN_WORKBOOK, own if own in names else names[0])
                    elif tag == "statement":
                        key = None                   # miniF2F- and ProofNet-style names: the pool does not use them
                        counts["stp_statements_not_lean_workbook"] += 1
                    elif text in minif2f_texts:
                        key = None
                        counts["stp_conjectures_equal_to_minif2f"] += 1
                    else:
                        key = (STP_CONJECTURE, content_id(statement))
                key_of_prompt[prompt] = key
            if key is None:
                continue
            kept = references[key]
            kept.append((length or 0, shard_number, row_number))
            if len(kept) > per_source:
                kept.sort()
                del kept[per_source:]
        log(f"STP shard {shard_number + 1}/{len(shards)} scanned: {len(references)} problems with a proof so far")
    return references


def read_stp_rows(shards: list[Path], wanted: dict[tuple[int, int], tuple[str, str]]) -> Iterator[tuple[tuple[str, str], str, str]]:
    """Pass 2 over STP: (key, prompt, target) for the rows `wanted` names by (shard, row)."""
    by_shard: dict[int, list[int]] = defaultdict(list)
    for shard_number, row_number in wanted:
        by_shard[shard_number].append(row_number)
    for shard_number, rows in sorted(by_shard.items()):
        table = _parquet().read_table(shards[shard_number], columns=["prompt", "target"]).take(sorted(rows))
        for row_number, row in zip(sorted(rows), table.to_pylist()):
            yield wanted[(shard_number, row_number)], row["prompt"], row["target"]


def build_candidates(files: dict[str, Path], settings: dict, pin: LeanPin, minif2f_texts: set[str],
                     present_holdout: set[str], log: Callable[[str], None] = print) -> Build:
    """Every candidate the published files give, in the pool's fixed order (Lean Workbook in file order, then
    STP conjectures in the seeded order), and everything left out, by reason.

    `files` maps `internlm_workbook`, `internlm_rows`, `goedel` and `stp:<n>` to verified local paths.
    """
    counts: Counter = Counter()
    per_source = settings["certificates"]["per_source"]
    source_order = {source: position for position, source in enumerate(settings["certificates"]["source_order"])}
    seed = settings["heldout"]["seed"]

    # ---- the Lean Workbook's statements, and its own proofs
    statements: dict[str, str] = {}
    groups: dict[str, str] = {}
    proofs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))     # name -> source -> proof blocks
    for row in json.loads(files["internlm_workbook"].read_text()):
        statement = workbook_statement(row["formal_statement"])
        if statement is None:
            counts["workbook_rows_not_a_theorem"] += 1
            continue
        name = theorem_name_of(statement)
        if name in statements:
            counts["workbook_rows_repeating_a_name"] += 1
            continue
        statements[name] = statement
        groups[name] = _group(row["natural_language_statement"])
        for proof in row.get("proof") or []:
            block = tactic_block(strip_lean_comments(proof))
            if block.strip():
                proofs[name][INTERNLM_PROOFS].append(block)
    counts["workbook_statements"] = len(statements)
    workbook_by_text: dict[str, list[str]] = defaultdict(list)
    for name, statement in statements.items():
        workbook_by_text[normalize_statement(statement)].append(name)

    # ---- InternLM's tactic rows: proved and disproved
    rows_of: dict[str, list[dict]] = defaultdict(list)
    for row in read_parquet(files["internlm_rows"], ["id", "status", "tactic", "state_before", "state_after"]):
        rows_of[row["id"]].append(row)
    disproved: dict[str, list[dict]] = {}
    published_true: set[str] = set(proofs)
    for name, rows in rows_of.items():
        if name not in statements:
            counts["internlm_rows_for_an_unknown_statement"] += 1
            continue
        status = {row["status"] for row in rows}
        if status == {"disproved"}:
            disproved[name] = rows
        elif status == {"proved"}:
            published_true.add(name)
            proof = proof_from_rows(rows)
            if proof is None:
                counts["internlm_proved_rows_not_one_chain"] += 1
            else:
                proofs[name][INTERNLM_ROWS].append(proof)
        else:
            counts["internlm_rows_of_mixed_status"] += 1

    # ---- Goedel-Prover's files
    for row in read_parquet(files["goedel"], ["problem_id", "full_proof"]):
        name = row["problem_id"]
        if name not in statements:
            counts["goedel_proofs_for_an_unknown_statement"] += 1
            continue
        published_true.add(name)
        read = published_file_proof(row["full_proof"], name)
        if read is None or normalize_statement(read[0]) != normalize_statement(statements[name]):
            counts["goedel_proofs_unreadable_or_of_another_statement"] += 1
            continue
        proofs[name][GOEDEL].append(read[1])

    # ---- STP: proofs of Lean Workbook statements, and its own conjectures
    shards = _stp_shards(files)
    references = scan_stp(shards, workbook_by_text, minif2f_texts, per_source, counts, log)
    published_true.update(key[1] for key in references if key[0] == LEAN_WORKBOOK)
    conjectures = sorted((key[1] for key in references if key[0] == STP_CONJECTURE),
                         key=lambda identifier: rank(seed, "stp_order", identifier))
    counts["stp_conjectures_with_a_published_proof"] = len(conjectures)
    selected = set(conjectures[:settings["certificates"]["stp_conjectures_checked"]])
    stp_index = [{"problem_id": f"stp_{identifier}", "position": position,
                  "proofs": [[shard, row, length] for length, shard, row in references[(STP_CONJECTURE, identifier)]]}
                 for position, identifier in enumerate(conjectures)]
    wanted = {(shard, row): key for key, kept in references.items()
              if key[0] == LEAN_WORKBOOK or key[1] in selected for _, shard, row in kept}
    conjecture_statement: dict[str, str] = {}
    conjecture_proofs: dict[str, list[str]] = defaultdict(list)
    for (kind, key), prompt, target in read_stp_rows(shards, wanted):
        proof = stp_proof(target)
        if proof is None:
            counts["stp_proofs_empty"] += 1
        elif kind == LEAN_WORKBOOK:
            proofs[key][STP].append(proof)
        else:
            conjecture_statement.setdefault(key, stp_statement(prompt))
            conjecture_proofs[key].append(proof)
    log(f"STP: {len(conjectures)} conjectures with a published proof, the first {len(selected)} of the seeded order kept")

    # ---- candidates
    candidates: list[Candidate] = []
    excluded: list[dict] = []

    def leave_out(problem_id: str, reason: str) -> None:
        excluded.append({"problem_id": problem_id, "reason": reason})
        counts[f"excluded_{reason}"] += 1

    def certificates_for(problem_id: str, side: str, theorem: str, by_source: dict[str, list[str]]) -> list[Certificate]:
        kept = []
        for source in sorted(by_source, key=lambda name: source_order[name]):
            for proof in _shortest(by_source[source], per_source):
                if find_forbidden_token(proof) is not None:
                    counts["certificates_rejected_lexically"] += 1
                    continue
                rewritten_proof, places = pin.rewrite(proof)
                kept.append(Certificate(problem_id, side, source, theorem, rewritten_proof, places))
        return kept

    for name, statement in statements.items():
        is_true, is_false = name in published_true, name in disproved
        if not is_true and not is_false:
            continue
        counts["workbook_published_proved"] += is_true
        counts["workbook_published_disproved"] += is_false
        if is_true and is_false:
            leave_out(name, PUBLISHED_BOTH_WAYS)
            continue
        if find_forbidden_token(statement) is not None:
            leave_out(name, STATEMENT_REJECTED_LEXICALLY)
            continue
        at_pin, places = pin.rewrite(statement)
        if is_true:
            certificates = certificates_for(name, TRUE_SIDE, at_pin, proofs.get(name, {}))
        else:
            rows = disproved[name]
            reading = hypotheses_reading(statement, rows[0]["state_before"])
            if reading in (HAS_HYPOTHESES, UNPARSED):
                leave_out(name, DISPROOF_WITH_HYPOTHESES if reading == HAS_HYPOTHESES else DISPROOF_UNPARSED)
                continue
            published = proof_from_rows(rows)
            if published is None:
                leave_out(name, ROWS_NOT_ONE_CHAIN)
                continue
            built = negation_certificate(at_pin, published, negation_of(name))
            if built is None:
                leave_out(name, NEGATION_WRAPPER_NOT_BUILT)
                continue
            counts["negations_with_variables"] += "published_for_every_value" in built[1]
            certificates = certificates_for(name, FALSE_SIDE, built[0], {INTERNLM_ROWS: [built[1]]})
        if not certificates:
            leave_out(name, NO_USABLE_CERTIFICATE)
            continue
        candidates.append(Candidate(name, LEAN_WORKBOOK, TRUE_SIDE if is_true else FALSE_SIDE, at_pin, statement, places,
                                    groups[name], tuple(certificates)))
    counts["candidates_lean_workbook"] = len(candidates)

    for identifier in conjectures:
        if identifier not in selected:
            continue
        problem_id = f"stp_{identifier}"
        statement = conjecture_statement.get(identifier)
        if statement is None:
            leave_out(problem_id, NO_USABLE_CERTIFICATE)
            continue
        if find_forbidden_token(statement) is not None:
            leave_out(problem_id, STATEMENT_REJECTED_LEXICALLY)
            continue
        statement = rename_theorem(statement, problem_id)
        at_pin, places = pin.rewrite(statement)
        certificates = certificates_for(problem_id, TRUE_SIDE, at_pin, {STP: conjecture_proofs[identifier]})
        if not certificates:
            leave_out(problem_id, NO_USABLE_CERTIFICATE)
            continue
        candidates.append(Candidate(problem_id, STP_CONJECTURE, TRUE_SIDE, at_pin, statement, places, problem_id, tuple(certificates)))
    counts["candidates_stp_conjectures"] = len(candidates) - counts["candidates_lean_workbook"]
    counts["candidates_known_true"] = sum(candidate.side == TRUE_SIDE for candidate in candidates)
    counts["candidates_known_false"] = sum(candidate.side == FALSE_SIDE for candidate in candidates)
    counts["candidates_with_a_rewritten_statement"] = sum(candidate.rewritten > 0 for candidate in candidates)
    counts["certificates"] = sum(len(candidate.certificates) for candidate in candidates)

    # ---- the present holdout: what of it has no published answer is set aside for the novel stage
    set_aside = []
    for name in sorted(present_holdout):
        if name not in statements:
            counts["present_holdout_not_in_the_workbook"] += 1
        elif name not in published_true and name not in disproved:
            at_pin, places = pin.rewrite(statements[name])
            set_aside.append({"statement_id": name, "statement": at_pin, "statement_published": statements[name],
                              "rewritten": places, "group": groups[name]})
    counts["present_holdout"] = len(present_holdout)
    counts["present_holdout_set_aside_no_published_answer"] = len(set_aside)
    counts["present_holdout_candidates"] = sum(candidate.problem_id in present_holdout for candidate in candidates)
    return Build(candidates=candidates, set_aside=set_aside, excluded=excluded, counts=dict(sorted(counts.items())),
                 stp_index=stp_index, present_holdout=set(present_holdout))


# ------------------------------------------------------------------------------------------- stored candidates
def candidate_row(candidate: Candidate) -> dict:
    return {"problem_id": candidate.problem_id, "kind": candidate.kind, "side": candidate.side,
            "statement": candidate.statement, "statement_published": candidate.statement_published,
            "rewritten": candidate.rewritten, "group": candidate.group,
            "certificates": [certificate_entry(certificate) for certificate in candidate.certificates]}


def certificate_entry(certificate: Certificate) -> dict:
    """One certificate as stored. `renamed` is written only for a renamed certificate, so the rows of a first
    pass read the same before and after the renaming pass existed."""
    entry = {"source": certificate.source, "theorem": certificate.theorem, "proof": certificate.proof, "rewritten": certificate.rewritten}
    return {**entry, "renamed": list(certificate.renamed)} if certificate.renamed else entry


def certificate_from_entry(problem_id: str, side: str, entry: dict) -> Certificate:
    return Certificate(problem_id, side, entry["source"], entry["theorem"], entry["proof"], entry["rewritten"],
                       tuple(entry.get("renamed", ())))


def candidate_from_row(row: dict) -> Candidate:
    certificates = tuple(certificate_from_entry(row["problem_id"], row["side"], entry) for entry in row["certificates"])
    return Candidate(row["problem_id"], row["kind"], row["side"], row["statement"], row["statement_published"],
                     row["rewritten"], row["group"], certificates)
