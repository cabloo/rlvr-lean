"""What the ladder loop's GPU steps need of the pool, as files that travel WITH the code.
Spec: docs/spec/ladder-loop.spec.md, L0. A stage on the GPU box cannot see the dev machine's `experiments/` directory, so
`ladder_pool export` writes H and the base-map sample into a data directory of the package, which is committed
with the code. The files hold statements and ids. They hold no published proof: a certificate is never a
training target, and nothing that reaches the GPU box could make it one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rlvr_lean.domain.problem_pool.selection import rank

HELDOUT_FILE, BASE_MAP_FILE, SUMMARY_FILE = "heldout.jsonl", "base_map.jsonl", "summary.json"
PROBLEM_KEYS = ("problem_id", "kind", "side", "statement", "rewritten")      # everything a problem carries to the GPU box


def exported_problem(row: dict, **extra) -> dict:
    """One problem as it travels: what it states, where it is from, which side its certificate proves."""
    return {**{key: row[key] for key in PROBLEM_KEYS}, **extra}


def base_map_sample(pool_rows: list[dict], seed: int, problems: int) -> list[dict]:
    """A seeded random draw of the pool: the first `problems` of a content-hash order."""
    return sorted(pool_rows, key=lambda row: rank(seed, "base_map", row["problem_id"]))[:problems]


def write_export(directory: Path, heldout: list[dict], base_map: list[dict], summary: dict) -> dict:
    """Write the two problem files and `summary.json`, which records each file's rows and SHA-256: the GPU
    step refuses a data directory whose files do not match what the export recorded."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in ((HELDOUT_FILE, heldout), (BASE_MAP_FILE, base_map)):
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        (directory / name).write_bytes(data)
        files[name] = {"rows": len(rows), "sha256": hashlib.sha256(data).hexdigest()}
    full = {**summary, "files": files}
    (directory / SUMMARY_FILE).write_bytes((json.dumps(full, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    return full
