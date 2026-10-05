"""What a ROUND of the ladder loop needs, as files that travel with the code beside L0's.
Spec: docs/spec/ladder-loop.spec.md, "A round", "The challenger in this stage", L1.

A stage on the GPU box sees neither the dev machine's pool store nor an earlier task's output directory, so
`python -m rlvr_lean.tools.ladder_round export` writes a data directory of the package (committed with the code):

  candidates.jsonl       the pool problems the challenger may choose from: a seeded draw of the pool that holds
                         no problem of H, none of the base-map sample and none set aside (L1: 20,000 of them;
                         L2, `--candidates all`: every one). Statement, side, and the published FACTS the
                         challenger reads (source and length of the published proof, STP's round number). No
                         proof text
  base_map_facts.jsonl   the same facts for the base-map sample (its statements are in L0's data directory)
  base_results.jsonl     the base model's k of n on every problem of H and of the base-map sample: L0's GPU half
  summary.json           rows and SHA-256 of each file, and the hashes of the L0 data files they belong to

A published proof is a certificate only: its LENGTH is a fact the challenger may read, its text never leaves the
dev machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rlvr_lean.domain.problem_pool.selection import rank

CANDIDATES_FILE, FACTS_FILE, RESULTS_FILE, SUMMARY_FILE = "candidates.jsonl", "base_map_facts.jsonl", "base_results.jsonl", "summary.json"
ROUND_FILES = (CANDIDATES_FILE, FACTS_FILE, RESULTS_FILE)
FACT_KEYS = ("certificate_source", "published_proof_chars", "stp_round")
CANDIDATE_KEYS = ("problem_id", "kind", "side", "statement", "rewritten", *FACT_KEYS)
RESULT_KEYS = ("problem_id", "set", "episodes", "resolved", "resolved_by_statement", "resolved_by_negation", "sides")
HELDOUT_SET, BASE_MAP_SET = "heldout", "base_map"
CANDIDATES_LABEL = "round_candidates"       # the seeded order the candidates are drawn in


def round_candidates(pool_rows: Sequence[Mapping], barred_ids: Iterable[str], seed: int, wanted: int | None) -> list[Mapping]:
    """A seeded random draw of the pool for the challenger to choose from: the first `wanted` of a content-hash
    order of the pool rows that are not barred (H, the base-map sample, the statements set aside). `wanted`
    None is the WHOLE pool, in that order (L2: "Candidates: the whole pool")."""
    barred = set(barred_ids)
    free = [row for row in pool_rows if row["problem_id"] not in barred]
    return sorted(free, key=lambda row: rank(seed, CANDIDATES_LABEL, row["problem_id"]))[:wanted]


def facts_of(problem_id: str, certificate_sources: Mapping[str, str], proof_chars: Mapping[str, int], stp_rounds: Mapping[str, int]) -> dict:
    return {"certificate_source": certificate_sources.get(problem_id), "published_proof_chars": proof_chars.get(problem_id),
            "stp_round": stp_rounds.get(problem_id)}


def certificate_lengths(candidate_rows: Iterable[Mapping], renamed_rows: Iterable[Mapping], pool_rows: Sequence[Mapping]) -> dict[str, int]:
    """The length in characters of the certificate that verified each pool problem: the published proof whose
    Lean file has the hash the pool row names (`certificate_sha`), among the problem's published and renamed
    certificates."""
    from rlvr_lean.data.published import candidate_from_row, certificate_from_entry
    from rlvr_lean.domain.problem_pool.certificates import TRUE_SIDE, certificate_source, sha_of

    wanted = {row["problem_id"]: row["certificate_sha"] for row in pool_rows}
    renamed = {row["problem_id"]: row["certificates"] for row in renamed_rows}
    lengths = {}
    for row in candidate_rows:
        if row["problem_id"] not in wanted:
            continue
        candidate = candidate_from_row(row)
        extra = [certificate_from_entry(candidate.problem_id, candidate.side, entry) for entry in renamed.get(candidate.problem_id, ())]
        for certificate in (*candidate.certificates, *extra):
            if sha_of(certificate_source(certificate, with_fingerprint=candidate.side == TRUE_SIDE)) == wanted[candidate.problem_id]:
                lengths[candidate.problem_id] = len(certificate.proof)
                break
    return lengths


def stp_rounds(stp_index_rows: Iterable[Mapping], shards: Sequence[Path], wanted_ids: Iterable[str]) -> dict[str, int]:
    """STP's round number (`iteration`) for each wanted conjecture: the earliest among its kept published proofs."""
    wanted = set(wanted_ids)
    rows_of: dict[int, dict[int, list[str]]] = {}
    for entry in stp_index_rows:
        if entry["problem_id"] in wanted:
            for shard, row, _length in entry["proofs"]:
                rows_of.setdefault(shard, {}).setdefault(row, []).append(entry["problem_id"])
    rounds: dict[str, int] = {}
    if not rows_of:
        return rounds
    import pyarrow.parquet as parquet       # the `data` group; read only when a conjecture's rows are wanted

    for shard, by_row in sorted(rows_of.items()):
        numbers = sorted(by_row)
        column = parquet.read_table(shards[shard], columns=["iteration"]).take(numbers).column("iteration").to_pylist()
        for number, value in zip(numbers, column):
            try:
                iteration = int(value)
            except (TypeError, ValueError):
                continue
            for problem_id in by_row[number]:
                rounds[problem_id] = min(iteration, rounds.get(problem_id, iteration))
    return rounds


def exported_candidate(row: Mapping, facts: Mapping) -> dict:
    return {**{key: row[key] for key in CANDIDATE_KEYS if key not in FACT_KEYS}, **{key: facts.get(key) for key in FACT_KEYS}}


def exported_result(row: Mapping, set_name: str) -> dict:
    return {**{key: row[key] for key in RESULT_KEYS if key != "set"}, "set": set_name}


def _encoded(rows: Sequence[Mapping]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def write_round_export(directory: Path, candidates: Sequence[Mapping], base_map_facts: Sequence[Mapping],
                       base_results: Sequence[Mapping], summary: Mapping) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in ((CANDIDATES_FILE, candidates), (FACTS_FILE, base_map_facts), (RESULTS_FILE, base_results)):
        data = _encoded(rows)
        (directory / name).write_bytes(data)
        files[name] = {"rows": len(rows), "sha256": hashlib.sha256(data).hexdigest()}
    full = {**summary, "files": files}
    (directory / SUMMARY_FILE).write_bytes((json.dumps(full, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    return full


def load_round_data(directory: Path, l0_summary: Mapping) -> tuple[list[dict], list[dict], list[dict], dict]:
    """(candidates, base-map facts, base results, summary), refused unless every file is what the export
    recorded AND the export was made for the L0 data directory in use (its files' hashes)."""
    summary = json.loads((directory / SUMMARY_FILE).read_text())
    rows = {}
    for name in ROUND_FILES:
        recorded = summary["files"][name]
        data = (directory / name).read_bytes()
        found = hashlib.sha256(data).hexdigest()
        if found != recorded["sha256"]:
            raise ValueError(f"{directory / name} has SHA-256 {found} and the export recorded {recorded['sha256']}: refused")
        rows[name] = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(rows[name]) != recorded["rows"]:
            raise ValueError(f"{directory / name} has {len(rows[name])} rows and the export recorded {recorded['rows']}")
    made_for = summary.get("l0_files") or {}
    in_use = {name: entry["sha256"] for name, entry in l0_summary["files"].items()}
    if made_for != in_use:
        raise ValueError(f"{directory} was exported for another L0 data directory (its files: {made_for}; in use: {in_use}): refused")
    return rows[CANDIDATES_FILE], rows[FACTS_FILE], rows[RESULTS_FILE], summary


def check_round_data(heldout: Sequence[Mapping], base_map: Sequence[Mapping], candidates: Sequence[Mapping],
                     base_map_facts: Sequence[Mapping], base_results: Sequence[Mapping], heldout_episodes: int) -> dict:
    """Refuse a round's data that breaks the held-out rule or does not fit together (spec fixture 6): no
    problem of H is a candidate or in the base map's results as a pool problem; no base-map problem is a
    candidate (the solver has attempted it already); every problem of H and of the base map has its result,
    H's with the placing number of episodes."""
    held = {row["problem_id"] for row in heldout}
    mapped = {row["problem_id"] for row in base_map}
    candidate_ids = [row["problem_id"] for row in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("a candidate appears twice in the round's data")
    for name, others in (("H", held), ("the base-map sample", mapped)):
        shared = sorted(set(candidate_ids) & others)
        if shared:
            raise ValueError(f"{len(shared)} candidates are in {name} (first: {shared[0]}): refused")
    results: dict[str, dict[str, Mapping]] = {HELDOUT_SET: {}, BASE_MAP_SET: {}}
    for row in base_results:
        if row["set"] not in results or row["problem_id"] in results[row["set"]]:
            raise ValueError(f"the base result of {row['problem_id']} names the set {row['set']!r} or appears twice")
        if not 0 <= row["resolved"] <= row["episodes"]:
            raise ValueError(f"the base result of {row['problem_id']} is not k of n: {row['resolved']} of {row['episodes']}")
        results[row["set"]][row["problem_id"]] = row
    for name, wanted, found in (("H", held, set(results[HELDOUT_SET])), ("the base-map sample", mapped, set(results[BASE_MAP_SET]))):
        if wanted != found:
            raise ValueError(f"the base results do not cover {name}: {len(wanted - found)} problems without a result, "
                             f"{len(found - wanted)} results for problems that are not in it")
    wrong = sorted(key for key, row in results[HELDOUT_SET].items() if row["episodes"] != heldout_episodes)
    if wrong:
        raise ValueError(f"{len(wrong)} problems of H were placed with another number of episodes than {heldout_episodes} (first: {wrong[0]})")
    facts = {row["problem_id"] for row in base_map_facts}
    if facts != mapped:
        raise ValueError(f"the base-map facts do not cover the base-map sample ({len(mapped - facts)} missing, {len(facts - mapped)} extra)")
    return {"candidates": len(candidate_ids), "heldout": len(held), "base_map": len(mapped)}
