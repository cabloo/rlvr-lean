"""How long is the shortest published proof of each held-out problem? (spec ladder-loop, L3c: the read by length).

    python -m rlvr_lean.data.heldout_proof_lines <candidates.jsonl of the pool build> <heldout.jsonl> <out.jsonl>

Writes one row a held-out problem: the line count of its shortest published proof (non-empty lines that are not
comments), how many published proofs it has, and the length group the reports use. The candidates file is the
pool build's (`ladder_pool build`): every published proof by problem. The shipped copy is
`data/ladder_l0/heldout_proof_lines.jsonl`. Reports read it; no prompt and no rule does.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

LENGTH_GROUPS = ("1", "2-3", "4-7", "8+")


def proof_lines(proof: str) -> int:
    return len([line for line in proof.strip("\n").split("\n") if line.strip() and not line.strip().startswith("--")])


def length_group(lines: int) -> str:
    return "1" if lines == 1 else "2-3" if lines <= 3 else "4-7" if lines <= 7 else "8+"


def build(candidates: Path, heldout: Path, out: Path) -> Counter:
    with heldout.open() as handle:
        order = [json.loads(line)["problem_id"] for line in handle if line.strip()]
    wanted, found = set(order), {}
    with candidates.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["problem_id"] not in wanted:
                continue
            lengths = [proof_lines(certificate["proof"]) for certificate in row.get("certificates", []) if certificate.get("proof")]
            found[row["problem_id"]] = {"problem_id": row["problem_id"], "shortest_published_proof_lines": min(lengths),
                                        "published_proofs": len(lengths), "length_group": length_group(min(lengths))}
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"{len(missing)} held-out problems have no published proof in {candidates}")
    out.write_text("".join(json.dumps(found[key], ensure_ascii=False) + "\n" for key in order))
    return Counter(found[key]["length_group"] for key in order)


if __name__ == "__main__":
    print(dict(build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))))
