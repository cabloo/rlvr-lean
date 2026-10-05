"""The dev-machine side of a ladder-loop ROUND: the data a round needs, written where a stage on the GPU box can read it.
Spec: docs/spec/ladder-loop.spec.md, "A round", L1. See `rlvr_lean/data/ladder_round_export.py`.

  export   <out>/: candidates.jsonl, base_map_facts.jsonl, base_results.jsonl, summary.json
           from  --store       the pool store of `ladder_pool` (pool.jsonl, candidates.jsonl, stp_index.jsonl, ...)
                 --l0-results  the directory that holds L0's GPU results as they were collected from the GPU box
                               (experiments/rlvr_lean/<the ladder_l0b task>/steps: episodes_heldout_problems.jsonl
                               and episodes_base_map_problems.jsonl)
                 --l0-data     L0's data directory (default: the package's, src/rlvr_lean/data/ladder_l0)
                 --candidates  how many pool problems the challenger may choose from: a number (default:
                               `ladder_loop.challenger.candidates`, L1's draw of 20,000) or `all` (L2: the
                               whole pool, less H and the base map)
           Commit <out>: it travels with the code.

    python -m rlvr_lean.tools.ladder_round export --store experiments/rlvr_lean/ladder_l0 \
        --l0-results experiments/rlvr_lean/ladder_l0b_r1/steps --out src/rlvr_lean/data/ladder_l1
    python -m rlvr_lean.tools.ladder_round export --store experiments/rlvr_lean/ladder_l0 \
        --l0-results experiments/rlvr_lean/ladder_l0b_r2/steps --out src/rlvr_lean/data/ladder_l2 --candidates all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import yaml

from rlvr_lean.data.ladder_round_export import (
    BASE_MAP_SET,
    HELDOUT_SET,
    certificate_lengths,
    check_round_data,
    exported_candidate,
    exported_result,
    facts_of,
    round_candidates,
    stp_rounds,
    write_round_export,
)

CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
PACKAGE_L0_DATA = Path(__file__).resolve().parents[1] / "data" / "ladder_l0"
STP_KIND = "stp_conjecture"
ALL_CANDIDATES = "all"          # `--candidates all`: every pool problem that is free to be a candidate (L2)


def log(message: str) -> None:
    print(f"[ladder_round {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _rows(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def candidates_wanted(argument: str | int | None, configured: int) -> int | None:
    """How many candidates an export draws: the config's number, the number given, or None for `all`."""
    if argument is None:
        return configured
    if argument == ALL_CANDIDATES:
        return None
    if not str(argument).isdigit() or int(argument) < 1:
        raise SystemExit(f"--candidates must be a positive number or `{ALL_CANDIDATES}`, got {argument!r}")
    return int(argument)


def export(arguments, config: dict) -> int:
    from rlvr_lean.gpu.ladder_loop import load_problems
    from rlvr_lean.tools.ladder_pool import local_files

    settings = config["ladder_loop"]
    wanted = candidates_wanted(getattr(arguments, "candidates", None), settings["challenger"]["candidates"])
    steps = arguments.store / "steps"
    if not (steps / "pool.jsonl").exists():
        raise SystemExit(f"{steps / 'pool.jsonl'} does not exist: `ladder_pool pool` has not run")
    heldout, base_map, l0_summary = load_problems(arguments.l0_data)
    pool_rows = list(_rows(steps / "pool.jsonl"))
    by_id = {row["problem_id"]: row for row in pool_rows}
    held = {row["problem_id"] for row in heldout}
    mapped = [row["problem_id"] for row in base_map]
    in_the_pool_and_held_out = sorted(held & set(by_id))
    if in_the_pool_and_held_out:
        raise SystemExit(f"{len(in_the_pool_and_held_out)} problems of H are in the pool file (first: {in_the_pool_and_held_out[0]}): refused")
    missing = [problem_id for problem_id in mapped if problem_id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} base-map problems are not in the pool file (first: {missing[0]}): this store is not the one L0's data came from")
    set_aside = {row["statement_id"] for row in _rows(steps / "set_aside.jsonl")}
    candidates = round_candidates(pool_rows, held | set(mapped) | set_aside, settings["round"]["candidates_seed"], wanted)
    described = [*candidates, *(by_id[problem_id] for problem_id in mapped)]
    renamed = list(_rows(steps / "renamed.jsonl")) if (steps / "renamed.jsonl").exists() else []
    lengths = certificate_lengths(_rows(steps / "candidates.jsonl"), renamed, described)
    shards = [path for name, path in sorted(local_files(config, arguments.store, download=False).items()) if name.startswith("stp:")]
    rounds = stp_rounds(_rows(steps / "stp_index.jsonl"), shards, (row["problem_id"] for row in described if row["kind"] == STP_KIND))
    sources = {row["problem_id"]: row["certificate_source"] for row in described}
    exported = [exported_candidate(row, facts_of(row["problem_id"], sources, lengths, rounds)) for row in candidates]
    facts = [{"problem_id": problem_id, **facts_of(problem_id, sources, lengths, rounds)} for problem_id in mapped]
    results = [exported_result(row, HELDOUT_SET) for row in _rows(arguments.l0_results / "episodes_heldout_problems.jsonl")]
    results += [exported_result(row, BASE_MAP_SET) for row in _rows(arguments.l0_results / "episodes_base_map_problems.jsonl")]
    counts = check_round_data(heldout, base_map, exported, facts, results, settings["goal"]["base_episodes"])
    without_a_length = sum(row["published_proof_chars"] is None for row in exported) + sum(row["published_proof_chars"] is None for row in facts)
    stp_described = sum(row["kind"] == STP_KIND for row in described)
    summary = write_round_export(arguments.out, exported, facts, results, {
        "fixture": False, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "lean_pin": settings["lean_pin"],
        "l0_files": {name: entry["sha256"] for name, entry in l0_summary["files"].items()},
        "candidates_wanted": ALL_CANDIDATES if wanted is None else wanted, "candidates_seed": settings["round"]["candidates_seed"],
        "pool_problems": len(pool_rows), "pool_problems_free_to_be_candidates": len(pool_rows) - len(set(mapped)),
        "counts": counts, "problems_without_a_published_proof_length": without_a_length,
        "stp_conjectures_described": stp_described, "stp_conjectures_with_a_round": sum(row["problem_id"] in rounds for row in described),
        "l0_results": str(arguments.l0_results)})
    log(f"exported to {arguments.out}: {json.dumps(summary['files'])}"
        + ("" if wanted is None or len(exported) == wanted else f"; the pool gives only {len(exported)} of the {wanted} candidates wanted")
        + (f"; {without_a_length} problems have no published-proof length (their certificate was not found by its hash)" if without_a_length else ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export"])
    parser.add_argument("--store", type=Path, required=True, help="the pool store of `ladder_pool` (experiments/rlvr_lean/ladder_l0)")
    parser.add_argument("--l0-results", type=Path, required=True, help="the steps directory of the finished ladder_l0b task")
    parser.add_argument("--l0-data", type=Path, default=PACKAGE_L0_DATA)
    parser.add_argument("--out", type=Path, required=True, help="the data directory to write (src/rlvr_lean/data/ladder_l1)")
    parser.add_argument("--candidates", default=None, metavar="N|all",
                        help=f"pool problems exported as candidates: a number (default: ladder_loop.challenger.candidates) or `{ALL_CANDIDATES}` (L2: the whole pool)")
    parser.add_argument("--config", type=Path, default=CONFIG)
    arguments = parser.parse_args()
    return export(arguments, yaml.safe_load(arguments.config.read_text()))


if __name__ == "__main__":
    sys.exit(main())
