"""The read over the seeds of L5 (do DEEP solves teach what ordinary solves do not?), from the pulled task directories.
Spec: docs/spec/ladder-loop.spec.md, "L5, further seeds" ("The reads, fixed before they run") and its amendment
("where the lead sits, and a read of reach"). No GPU, no Lean and nothing sent anywhere: files are read, lines are printed.
The read itself is `reporting.ladder_l5.over_seeds`.

    PYTHONPATH=src python -m rlvr_lean.tools.ladder_l5_seeds <the task directory of seed 0> [<of seed 1> <of seed 2>] [--out <read.json>]

A task directory is what was pulled for one task of the stage `ladder_l5_deep` (`experiments/rlvr_lean/ladder_l5_deep_seed<N>_r<M>`;
its `steps` directory may be named in its place). Read from each, and only read:

  report_ladder_l5.json                         the seed, whether the report can be read, its branches, its primary, the start
                                                model, how G was sampled, each measured set's attempts and timeouts, and, of a
                                                further seed, whose search it read
  l5_search_problems.jsonl, l5_deep_solves.jsonl   the ONE search: its problems, and by problem it solved its k, its n and the place
                                                of its first success (never a proof's text)
  l5_held_aside_problems.jsonl                  the ONE held-aside set
  l4_heldout_groups.jsonl, l4_goal_set_again.jsonl   the goal set and G'
  episodes_l5_held_aside_<model>_problems.jsonl each trained model on the held-aside set, problem by problem; the start model's
                                                too at the seed that made the search, and at a further seed the rows that seed's
                                                first step copied from it (l5_stored_<start>_held_aside.jsonl)
  episodes_l4_<sampling>_<model>_problems.jsonl each trained model on each sampling of G
  l4_stored_<start>_<sampling>.jsonl            the start model on it (its own run's rows, the same at every seed)

and, from the code's own data, the length group of each goal problem's shortest published proof (the stage's by-length read
uses it; no proof text is read). Printed: each seed's own figures (its primary computed again and held to its report's); the
primary on the held-aside set over the three seeds; ON THE NEW SEEDS ALONE the goal set's lead (confirmed or not), its split and
the reach on the long problems; the spread of each kind of model; the section's branch for what follows; where two models'
shares of timed-out attempts differ by more than a point. It works on fewer seeds than three and says which reads it could not
make. A seed whose report is INCONCLUSIVE or not to be read is named and nothing is pooled. Refused, with what is wrong: a
directory without one of its files, two directories of one seed, seeds that do not share one search, one held-aside set, one
measurement of the start model or one goal set, rows that do not give a report's primary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml

from rlvr_lean.domain.ladder_round.l5 import MODELS
from rlvr_lean.gpu.ladder_ceiling import goal_set, stored_file
from rlvr_lean.gpu.ladder_l3c import proof_lengths
from rlvr_lean.gpu.ladder_l4 import AGAIN_FILE, GROUPS_FILE, L4
from rlvr_lean.gpu.ladder_l5 import HELD_ASIDE_FILE, HELD_ASIDE_PART, L5, REPORT_FILE, SEARCH_PROBLEMS_FILE, SOLVES_FILE, held_aside_set
from rlvr_lean.reporting.ladder_l5 import _not_read, over_seeds

CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"
FILES = (SEARCH_PROBLEMS_FILE, SOLVES_FILE, HELD_ASIDE_FILE, GROUPS_FILE, AGAIN_FILE)      # what every seed that is read holds, whoever made the search


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def steps_directory(directory: Path) -> Path:
    """Where a pulled task's step files are: its `steps` directory, or the directory itself when that is what was named."""
    return directory / "steps" if (directory / "steps").is_dir() else directory


def results_file(set_name: str) -> str:
    """A measured set's per-problem rows, as its episode step stored them."""
    return f"episodes_{set_name}_problems.jsonl"


def read_seed(directory: Path) -> tuple[int, dict]:
    """(the seed, what `over_seeds` reads of it) from one pulled task directory. A directory without its report, or
    (when that report is read) without one of the files the read stands on, is refused, naming them. Of a report that is
    INCONCLUSIVE or not to be read the report alone is handed on: nothing else is said of such a run."""
    steps = steps_directory(directory)
    if not (steps / REPORT_FILE).is_file():
        raise ValueError(f"{steps} does not hold {REPORT_FILE}: it is not the pulled directory of a task of the stage `ladder_l5_deep` that wrote its report")
    report = json.loads((steps / REPORT_FILE).read_text(encoding="utf-8"))
    if report.get("check") != "L5" or not isinstance(report.get("seed"), int) or not isinstance(report.get("start"), str):
        raise ValueError(f"{steps / REPORT_FILE} is not a report of L5 with its seed and its start model")
    seed, start = report["seed"], report["start"]
    if _not_read(report) is not None:
        return seed, {"report": report, "directory": str(steps)}
    samplings = [sampling["name"] for sampling in report["sizes"]["goal_samplings"]]
    # The start model on the held-aside set: measured by the seed that made the search; at a further seed, the rows its first step copied from that run.
    of_the_start = stored_file(start, HELD_ASIDE_PART, L5) if report.get("a_further_seed") else results_file(held_aside_set(start))
    held_aside = {start: of_the_start, **{name: results_file(held_aside_set(name)) for name in MODELS}}
    goal = {start: [stored_file(start, sampling, L4) for sampling in samplings], **{name: [results_file(goal_set(sampling, name, L4)) for sampling in samplings] for name in MODELS}}
    lacking = [name for name in (*FILES, *held_aside.values(), *(name for names in goal.values() for name in names)) if not (steps / name).is_file()]
    if lacking:
        raise ValueError(f"{steps} does not hold {lacking}: it is not the pulled directory of a task of the stage `ladder_l5_deep` that wrote its report")
    return seed, {"report": report, "directory": str(steps), "searched": _rows(steps / SEARCH_PROBLEMS_FILE),
                  "solves": {row["problem_id"]: [row["k"], row["n"], row["first_success"]] for row in _rows(steps / SOLVES_FILE)},      # never a proof's text
                  "aside": _rows(steps / HELD_ASIDE_FILE), "groups": _rows(steps / GROUPS_FILE), "again": [row["problem_id"] for row in _rows(steps / AGAIN_FILE)],
                  "held_aside": {name: _rows(steps / file) for name, file in held_aside.items()},
                  "goal": {name: [_rows(steps / file) for file in files] for name, files in goal.items()}}


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
    parser.add_argument("directories", nargs="+", type=Path, help="the pulled task directory of each seed of L5 that is here (or its steps directory); only read")
    parser.add_argument("--config", type=Path, default=CONFIG, help="where the bootstrap's settings are read (evaluation.bootstrap_resamples, .bootstrap_seed): the runs' own")
    parser.add_argument("--lengths", type=Path, default=None, help="the length group of each held-out problem's shortest published proof; the code's own data without it")
    parser.add_argument("--out", type=Path, default=None, help="a JSON file to write the read to; nothing is written without it")
    arguments = parser.parse_args(argv)
    evaluation = yaml.safe_load(arguments.config.read_text())["evaluation"]
    try:
        by_seed = read_seeds(arguments.directories)
        read = over_seeds(by_seed, proof_lengths(arguments.lengths), evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"])
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
