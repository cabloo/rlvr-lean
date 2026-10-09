"""The read over the seeds of L4's arm (the loop from a model PRETRAINED ON PUBLISHED PROOFS), from the pulled task
directories. Spec: docs/spec/ladder-loop.spec.md, "L4: the loop from a model pretrained on published proofs",
"Further seeds of the arm, made exact before they run" ("The read over the three seeds, fixed now"). No GPU, no Lean and
nothing sent anywhere: files are read, lines are printed. The read itself is `reporting.ladder_l4.over_seeds`.

    PYTHONPATH=src python -m rlvr_lean.tools.ladder_l4_seeds <the task directory of seed 0> <of seed 1> <of seed 2> [--out <read.json>]

A task directory is what was pulled for one task of the stage `ladder_l4` (`experiments/rlvr_lean/ladder_l4_seed<N>_r<M>`;
its `steps` directory may be named in its place). Read from each, and only read:

  report_ladder_l4.json                      the seed, whether the report can be read, its branch and its primary
  l4_goal_set_again.jsonl                    G': the ids the primary is read on (the pretraining's, the same at every seed)
  episodes_l3d2_more_with_problems.jsonl     `with` of that seed on the SECOND sampling of G, problem by problem
  l4_stored_pre_more.jsonl                   `pre` on it (the pretraining run's rows, the same at every seed)

Printed: each seed's primary (computed again from the rows and held to its report's figure), the pooled read (each problem's
mean over the seeds of its difference, 95% bootstrap over the problems of G'), each seed's sign and the seeds whose own
interval is clear of zero. A seed whose report is INCONCLUSIVE or not to be read is named and nothing is pooled. No verdict
is named: the spec's branches stand. Refused, with what is wrong: a directory without one of the four files, two directories
of one seed, seeds that do not hold one G' or one `pre`, rows that do not give the report's primary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml

from rlvr_lean.domain.ladder_round.l3d import WITH
from rlvr_lean.domain.ladder_round.l3d2 import L3D2
from rlvr_lean.domain.ladder_round.l4 import PRE
from rlvr_lean.gpu.ladder_ceiling import MORE, goal_set, stored_file
from rlvr_lean.gpu.ladder_l4 import AGAIN_FILE, ARM_REPORT_FILE, L4
from rlvr_lean.reporting.ladder_l4 import over_seeds

CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
WITH_FILE = f"episodes_{goal_set(MORE, WITH, L3D2)}_problems.jsonl"        # `with` on the second sampling of G, as the arm's measure step stored it
PRE_FILE = stored_file(PRE, MORE, L4)                                      # `pre` on it, as the arm's prepare step copied it from the pretraining run
FILES = (ARM_REPORT_FILE, AGAIN_FILE, WITH_FILE, PRE_FILE)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def steps_directory(directory: Path) -> Path:
    """Where a pulled task's step files are: its `steps` directory, or the directory itself when that is what was named."""
    return directory / "steps" if (directory / "steps").is_dir() else directory


def read_seed(directory: Path) -> tuple[int, dict]:
    """(the seed, what `over_seeds` reads of it) from one pulled task directory. A directory without one of the four
    files is refused, naming them: a task that did not reach its report delivered no read."""
    steps = steps_directory(directory)
    lacking = [name for name in FILES if not (steps / name).is_file()]
    if lacking:
        raise ValueError(f"{steps} does not hold {lacking}: it is not the pulled directory of a task of the stage `ladder_l4` that wrote its report")
    report = json.loads((steps / ARM_REPORT_FILE).read_text(encoding="utf-8"))
    if report.get("stage") != L4 or not isinstance(report.get("seed"), int):
        raise ValueError(f"{steps / ARM_REPORT_FILE} is not a report of L4's arm with its seed")
    return report["seed"], {"report": report, "again": [row["problem_id"] for row in _rows(steps / AGAIN_FILE)], "with": _rows(steps / WITH_FILE),
                            "pre": _rows(steps / PRE_FILE), "directory": str(steps)}


def read_seeds(directories: Sequence[Path]) -> dict[int, dict]:
    by_seed: dict[int, dict] = {}
    for directory in directories:
        seed, entry = read_seed(directory)
        if seed in by_seed:
            raise ValueError(f"{entry['directory']} and {by_seed[seed]['directory']} are both of seed {seed}: one task directory a seed")
        by_seed[seed] = entry
    return by_seed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, help="the pulled task directory of each seed of the arm (or its steps directory); only read")
    parser.add_argument("--config", type=Path, default=CONFIG, help="where the bootstrap's settings are read (evaluation.bootstrap_resamples, .bootstrap_seed): the runs' own")
    parser.add_argument("--out", type=Path, default=None, help="a JSON file to write the read to; nothing is written without it")
    arguments = parser.parse_args(argv)
    evaluation = yaml.safe_load(arguments.config.read_text())["evaluation"]
    try:
        by_seed = read_seeds(arguments.directories)
        read = over_seeds(by_seed, evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr, flush=True)
        return 2
    read["directories"] = {str(seed): entry["directory"] for seed, entry in sorted(by_seed.items())}
    for line in read["lines"]:
        print(line, flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(read, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
