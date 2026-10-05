"""Load and split the statement sets of spec §2. Deterministic: every split is a function of a content
hash and the config's split seed, so a rerun reproduces it exactly.

- Lean Workbook (`internlm/Lean-Workbook`, pinned revision): one formalization per natural-language
  problem (as STP de-duplicated it: 89,221 problems), split by a hash of the NATURAL-LANGUAGE statement,
  so two formalizations of one problem can never land in different sets.
- miniF2F: the copy in the DeepSeek-Prover-V1.5 repo (`datasets/minif2f.jsonl`, pinned commit).
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.conjecturing.dedup import normalize_statement

_TRAILING_PROOF_STUB = re.compile(r":=\s*by\s*sorry\s*$")


@dataclass(frozen=True)
class Statement:
    statement_id: str
    statement: str           # the theorem up to and including `:= by` and a newline
    natural_language: str
    source: str


def _download(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    return target


def _rank(seed: int, label: str, text: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{text}".encode()).hexdigest()


def load_lean_workbook(data_directory: Path, revision: str) -> list[Statement]:
    path = _download(f"https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/{revision}/lean_workbook.json",
                     data_directory / f"lean_workbook_{revision[:8]}.json")
    statements, seen_problems = [], set()
    for row in json.loads(path.read_text()):
        problem = row["natural_language_statement"].strip()
        body = _TRAILING_PROOF_STUB.sub("", row["formal_statement"]).rstrip()
        if problem in seen_problems or not body.startswith("theorem "):
            continue
        seen_problems.add(problem)
        name = body.split()[1]
        statements.append(Statement(statement_id=name, statement=body + " := by\n", natural_language=problem,
                                    source=row["split"]))
    return statements


def load_minif2f(data_directory: Path, commit: str, split: str) -> list[Statement]:
    path = _download(f"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/{commit}/datasets/minif2f.jsonl",
                     data_directory / "minif2f.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted((Statement(statement_id=row["name"], statement=row["formal_statement"],
                             natural_language=row.get("informal_prefix", ""), source=f"minif2f_{row['split']}")
                   for row in rows if row["split"] == split), key=lambda statement: statement.statement_id)


def all_minif2f_normalized(data_directory: Path, commit: str) -> set[str]:
    return {normalize_statement(statement.statement)
            for split in ("valid", "test") for statement in load_minif2f(data_directory, commit, split)}


def ranked_workbook_candidates(statements: list[Statement], seed: int, excluded_normalized: set[str]) -> list[Statement]:
    """Every workbook problem in a seeded hash order, minus anything equal to a miniF2F statement."""
    eligible = [statement for statement in statements if normalize_statement(statement.statement) not in excluded_normalized]
    return sorted(eligible, key=lambda statement: _rank(seed, "workbook", statement.natural_language))


def reward_half(statement: Statement, seed: int) -> str:
    """Reward-pool statements split 50/50 by statement: 'gradient' (scoring) or 'validation' (the holdout)."""
    return "gradient" if int(_rank(seed, "reward", statement.natural_language)[:8], 16) % 2 == 0 else "validation"


def is_conjecture_holdout(conjecture_id: str, seed: int, fraction: float) -> bool:
    return int(_rank(seed, "conjecture_holdout", conjecture_id)[:8], 16) / 0xFFFFFFFF < fraction


@dataclass(frozen=True)
class EarlierStatements:
    """An earlier Lean pin's seed and reward statements, to be filtered again by compiling each under another
    pin (the OEIS Open spec, O2a item 9d). The sets only shrink: a statement that no longer
    compiles is dropped, never replaced, and a survivor stays in the set (and the reward half) it was in, so
    runs under the two pins are about the same statements."""

    seeds: list[Statement]
    reward: list[Statement]

    @property
    def candidates(self) -> list[Statement]:
        return self.seeds + self.reward

    def survivors(self, compiled: list[Statement]) -> tuple[list[Statement], list[Statement]]:
        """(seed, reward) statements among `compiled`, each set in its earlier order."""
        compiled_ids = {statement.statement_id for statement in compiled}
        return ([statement for statement in self.seeds if statement.statement_id in compiled_ids],
                [statement for statement in self.reward if statement.statement_id in compiled_ids])


def _stored_statement_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: this Lean pin filters the earlier pin's statements again "
                                "(the OEIS Open spec, O2a item 9d), so that pin's prepare_data must have run for this profile")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_earlier_statements(run_directory: Path, workbook: list[Statement]) -> EarlierStatements:
    """The seed and reward statements a finished run stored in `run_directory` (read, never written), as the
    workbook's own `Statement`s. A stored statement the workbook does not hold, or holds with another text, is
    a data-revision mismatch and is refused: the two pins would not be judging the same statements."""
    by_id = {statement.statement_id: statement for statement in workbook}

    def statements(name: str) -> list[Statement]:
        found = []
        for row in _stored_statement_rows(run_directory / name):
            statement = by_id.get(row["statement_id"])
            if statement is None or statement.statement != row["statement"]:
                raise ValueError(f"{row['statement_id']} in {run_directory / name} is not the workbook's statement of that name")
            found.append(statement)
        return found

    return EarlierStatements(seeds=statements("statements_seed.jsonl"), reward=statements("statements_reward.jsonl"))
