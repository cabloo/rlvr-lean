"""The read over the fresh seeds of L5d (the three kinds again, over fresh draws, measured on the goal set's first sampling),
from the pulled task directories. Spec: docs/spec/ladder-loop.spec.md, "L5d: the three kinds again, over fresh
draws, measured on the goal set's first sampling" ("The reads, fixed before any fresh seed runs"). No GPU, no Lean and nothing
sent anywhere: files are read, lines are printed. The read itself is `reporting.ladder_l5_draws.read_over_the_seeds`, the one a
seed's own report step makes on the box from the runs that are there.

    PYTHONPATH=src python -m rlvr_lean.tools.ladder_l5_draws <the task directory of each fresh seed that is here> [--out <read.json>]

A task directory is what was pulled for one task of the stage `ladder_l5_draws` (`experiments/rlvr_lean/ladder_l5_draws_seed<N>_r<M>`;
its `steps` directory may be named in its place). Read from each, and only read, its `report_ladder_l5_draws.json`: what that
report says of its own seed (its recipe, whether its checks passed, how it was measured, its three models' figures and counts);
what it read of L5's own seeds on the same sampling, for the exploratory lines; and what is said beside every read when L5n did
not name the recipe. The seeds the reads are over are the setting's (`ladder_loop.l5_draws`: `seeds`, and `more_seeds` once the
approval is recorded). A seed of that number whose directory is not given makes the read NOT READ, naming it; a seed whose checks
failed makes it INCONCLUSIVE; seeds of two recipes are not pooled. Refused, with what is wrong: a directory without its report,
a report that is not L5d's, two directories of one seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import yaml

from rlvr_lean.domain.ladder_round.l5_draws import fixed_seeds
from rlvr_lean.gpu.ladder_l5_draws import REPORT_FILE, the_settings
from rlvr_lean.reporting.ladder_l5_draws import lines_over_the_seeds, read_over_the_seeds

CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"


def steps_directory(directory: Path) -> Path:
    """Where a pulled task's step files are: its `steps` directory, or the directory itself when that is what was named."""
    return directory / "steps" if (directory / "steps").is_dir() else directory


def read_seed(directory: Path) -> tuple[int, dict, dict]:
    """(the seed, what the read over the seeds takes of it, its whole report) from one pulled task directory. A directory
    without its report, or with a report that is not one of L5d with what it says of its own seed, is refused."""
    steps = steps_directory(directory)
    if not (steps / REPORT_FILE).is_file():
        raise ValueError(f"{steps} does not hold {REPORT_FILE}: it is not the pulled directory of a task of the stage `ladder_l5_draws` that wrote its report")
    report = json.loads((steps / REPORT_FILE).read_text(encoding="utf-8"))
    entry = report.get("of_this_seed")
    if report.get("check") != "L5d" or not isinstance(entry, dict) or isinstance(entry.get("seed"), bool) or not isinstance(entry.get("seed"), int):
        raise ValueError(f"{steps / REPORT_FILE} is not a report of L5d with what it says of its own seed")
    return entry["seed"], {**entry, "there": True, "run": str(steps)}, report


def read_seeds(directories: Sequence[Path]) -> tuple[dict[int, dict], dict[int, dict] | None, str | None]:
    """(by seed, what the read takes of it; L5's own seeds as the first report that holds them all read them, or as the
    last report that read any says they are not there, or None; what is said beside every read, when a report says it)."""
    by_seed: dict[int, dict] = {}
    of_l5, beside = None, None
    for directory in directories:
        seed, entry, report = read_seed(directory)
        if seed in by_seed:
            raise ValueError(f"{entry['run']} and {by_seed[seed]['run']} are both of seed {seed}: one task directory a seed")
        by_seed[seed] = entry
        theirs = {int(key): own for key, own in (report.get("the_seeds_of_l5") or {}).items()}
        if theirs and (of_l5 is None or not all(own.get("there") for own in of_l5.values())):
            of_l5 = theirs
        beside = beside or (report.get("the_recipe") or {}).get("said_beside_every_read")
    return by_seed, of_l5, beside


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path, help="the pulled task directory of each fresh seed of L5d that is here (or its steps directory); only read")
    parser.add_argument("--config", type=Path, default=CONFIG, help="where the seeds the reads are over are read (ladder_loop.l5_draws): the runs' own")
    parser.add_argument("--out", type=Path, default=None, help="a JSON file to write the read to; nothing is written without it")
    arguments = parser.parse_args(argv)
    try:
        draws = the_settings(yaml.safe_load(arguments.config.read_text()))
        fixed = fixed_seeds(draws.seeds, draws.more, draws.approved)
        by_seed, of_l5, beside = read_seeds(arguments.directories)
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr, flush=True)
        return 2
    given = {str(seed): entry["run"] for seed, entry in sorted(by_seed.items())}
    for seed in fixed:
        by_seed.setdefault(seed, {"there": False, "why": "its pulled task directory was not given"})
    read = read_over_the_seeds(by_seed, fixed, of_l5)
    read["lines"], read["directories"] = lines_over_the_seeds(read, beside), given
    for line in read["lines"]:
        print(line, flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(read, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
