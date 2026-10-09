"""The harvest H0: assembled proofs of pool problems that no stored attempt ever verified. Spec:
docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches" ("The harvest H0", "A harvested proof is
minimised"). No GPU and no generation: only the Lean pool is used, as BACKGROUND work, and only pool statements, stored
model attempts at them and mechanical recombinations of those are sent.

  read       every training-side attempt the ladder runs stored: under `--runs-root`, the task directories whose
             name starts `ladder_l` (not a smoke or a pilot run), in the order of their names, each run once however
             often it was pulled; their `steps/episodes_*_attempts_*.jsonl`. A file whose rows are held-out problems, or
             are not pool candidates, is left at its first such row: a held-out problem is never read
  problems   the pool problems that NO stored attempt verified, on either side
  attempts   each side's stored failed attempts (status `lean_error`), in the order run, file, row; a text once; none
             with a forbidden token; at most `--attempts` (48)
  re-check   every such attempt is checked again, for Lean's error positions (the stored rows hold none)
  replay     attempt 1, 2, ...: the accumulating episode's Lean-only part (`domain/repair/replay.py`: the harvest of
             the failed attempt into the pool, the pool's check, the kept closers after it), with L3c's rules and
             sizes. A closer that verifies after the pool is an ASSEMBLED proof: the problem is resolved
  minimise   every assembled proof: its pool blocks taken out one at a time, from the last to the first, a block
             staying out when the proof still verifies; the text that is left checked once more; if that fails, the
             proof as assembled
  written    `--out` (`src/rlvr_lean/data/ladder_l3d/harvest_h0.jsonl`): one row a problem, a training example as a
             round stores one (`problem_id`, `side`, `theorem`, `completion`) with where it came from; and beside it
             `harvest_h0.summary.json`. While it runs, what is resolved so far is kept in `<out>.partial.json`

A stored failure that verifies on the re-check is counted apart and is no assembled proof: it is not written. A proof
that verifies on the side a problem's published answer rules out is a SOUNDNESS ALARM: the tool stops (exit 3).

The job is hours long and keeps nothing of its own between runs: the pool caches by a file's text, so a rerun sends
the same files and is answered from the cache up to where the last run stopped. `--dry-run` reads, selects and counts.

    PYTHONPATH=src python -m rlvr_lean.tools.ladder_harvest --runs-root <the pulled runs> \\
        --out src/rlvr_lean/data/ladder_l3d/harvest_h0.jsonl \\
        --api-key-file <the pool's key file> --ca-file <the pool authority's certificate> [--in-flight 24 | auto] [--limit N] [--attempts M]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from rlvr_lean.domain.conjecturing.negation import negate_statement
from rlvr_lean.domain.problem_pool.certificates import negation_of
from rlvr_lean.domain.problem_pool.episodes import STATEMENT_SIDE, contradicted_side
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm
from rlvr_lean.domain.repair import cut as rules
from rlvr_lean.domain.repair import replay
from rlvr_lean.domain.repair.accumulate import proof_line_count
from rlvr_lean.domain.repair.alternate import checked_text
from rlvr_lean.domain.verification.lean_source import build_proof_source, find_forbidden_token
from rlvr_lean.infrastructure.verification_service import LeanCheckPool
from rlvr_lean.tools.ladder_ceiling_set import lean_check_settings as background_check_settings
from rlvr_lean.tools.ladder_pool import in_flight_argument, in_flight_text

PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / "config" / "experiment.yaml"
POOL_CANDIDATES = PACKAGE / "data" / "ladder_l2" / "candidates.jsonl"       # the whole pool as L2 reads it: ids, kinds, sides, statements
HELDOUT = PACKAGE / "data" / "ladder_l0" / "heldout.jsonl"
STORED_TASKS = "ladder_l"                                                    # the task directories read: the ladder loop's own runs
LEFT_OUT = ("smoke", "pilot")                                                # ... but not these
REPORTED_AFTER = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48)                        # the attempts after which the count so far is printed
KEPT_EVERY = 4                                                               # attempts between two writes of what is resolved so far
SOUNDNESS_ALARM_EXIT = 3                                                     # as `gpu.__main__`


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ------------------------------------------------------------------------------------------------- stored
def stored_attempts(runs_root: Path, statements: Mapping[str, str], heldout: set[str]) -> tuple[dict, set[str], list[str]]:
    """(each (problem, side)'s stored failed attempts as (run, file, row, completion); the problems some stored
    attempt verified; the task directories read). A run pulled twice (`<run>_r1`, `<run>_r2`) gives each file once.
    A file is left at the first row that is a held-out problem or is not a pool candidate."""
    stored, solved, seen_files, tasks = defaultdict(list), set(), set(), []
    for task in sorted(os.listdir(runs_root)):
        if not task.startswith(STORED_TASKS) or any(word in task for word in LEFT_OUT):
            continue
        run = re.sub(r"_r\d+$", "", task)
        for path in sorted(glob.glob(f"{runs_root}/{task}/steps/episodes_*_attempts_*.jsonl")):
            name = os.path.basename(path)
            if (run, name) in seen_files:
                continue
            seen_files.add((run, name))
            if task not in tasks:
                tasks.append(task)
            with open(path, encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    if row["problem_id"] in heldout or row["problem_id"] not in statements:
                        break
                    if row["status"] == "verified":
                        solved.add(row["problem_id"])
                    elif row["status"] == "lean_error" and row.get("completion"):
                        stored[(row["problem_id"], row["side"])].append((run, name, index, row["completion"]))
    return stored, solved, tasks


def episodes_of(stored: Mapping, solved: set[str], statements: Mapping[str, str], attempts: int, limit: int | None = None) -> list[dict]:
    """One episode for each side of each problem no stored attempt verified, in the order of (problem, side): its
    distinct failed attempts in the order run, file, row (a text once, as it stands in the checked file; none with a
    forbidden token; at most `attempts`), and the statement of that side (the exact negation for the negation)."""
    episodes = []
    for (problem, side), rows in sorted(stored.items()):
        if problem in solved:
            continue
        texts, chain = set(), {}
        for run, _, _, completion in sorted(rows):
            text = checked_text(completion)
            if text in texts or find_forbidden_token(completion) is not None:
                continue
            texts.add(text)
            chain[len(chain) + 1] = {"completion": completion, "from": run}
            if len(chain) == attempts:
                break
        statement = statements[problem] if side == STATEMENT_SIDE else negate_statement(statements[problem], negation_of(problem))
        episodes.append({**replay.new_episode(statement), "problem_id": problem, "side": side, "chain": chain, "stored_attempts": len(rows),
                         "resolved_at": None, "assembled_proof": None, "closer": None})
    return episodes[:limit] if limit else episodes


# ------------------------------------------------------------------------------------------------- replay
def raise_on_the_ruled_out_side(episode: Mapping, published_side: str, what: str) -> None:
    """A proof Lean verified on the side the problem's published answer rules out is a proof of both sides."""
    if episode["side"] == contradicted_side(published_side):
        raise SoundnessAlarm(f"{episode['problem_id']} is known {published_side} (a published proof our Lean verified) and {what} verified its "
                             f"{episode['side']}: a proof of both sides")


def replay_stored(episodes: Sequence[dict], check: Callable[[dict[str, str]], Mapping[str, Mapping]], sizes: Mapping, attempts: int, sides: Mapping[str, str],
                  stats: Counter, after_attempt: Callable[[int, int, int], None] | None = None) -> None:
    """Re-check every attempt for Lean's errors, then replay attempt 1, 2, ... of every episode that is not resolved:
    `replay.harvested`, the pool checks, `replay.closers_to_send`, the closer checks, `replay.first_verified`. An
    episode's `resolved_at` is the attempt after which an assembled proof verified (with `assembled_proof`, `closer`
    and the pool it stood on); MINUS the attempt for a stored failure that verified on the re-check (counted apart:
    never an assembled proof). `sides`: each problem's published side, for the alarm."""
    sources = {}
    for episode in episodes:
        for row in episode["chain"].values():
            source = build_proof_source(episode["statement"], row["completion"])
            row["file"] = replay.name_of(source)
            sources[row["file"]] = source
    print("re-checks for error positions:", len(sources), flush=True)
    answers = check(sources)
    for episode in episodes:
        for generation, row in episode["chain"].items():
            verdict = replay.judged(row["file"], answers[row["file"]])
            stats["recheck:" + verdict.status.value] += 1
            row["errors"] = None if verdict.is_verified else rules.errors_of(answers[row["file"]])
            if verdict.is_verified and episode["resolved_at"] is None:        # stored as an error, verified now: counted apart, never as assembly
                raise_on_the_ruled_out_side(episode, sides[episode["problem_id"]], "a stored attempt, checked again,")
                episode["resolved_at"], episode["closer"] = -generation, None
    print("re-checks done:", dict(stats), flush=True)
    for generation in range(1, attempts + 1):
        holders, requests = [], {}
        for episode in episodes:
            row = episode["chain"].get(generation)
            if episode["resolved_at"] is not None or row is None or not row.get("errors"):
                continue
            after = replay.harvested(episode, row["completion"], row["errors"], generation, sizes["pool_blocks"], stats)
            if after is None:
                continue
            holders.append((episode, after))
            if after["pool"] is not None:
                requests[after["pool"]["file"]] = after["pool"]["source"]
        if not holders:
            continue
        answers = check(requests)
        sources = {}
        for episode, after in holders:
            sources.update(replay.closers_to_send(episode, after, answers, sizes["kept_closers"], stats))
        answers = check(sources)
        for episode, after in holders:
            won = replay.first_verified(episode, after, answers, stats)
            if won is not None and episode["resolved_at"] is None:
                raise_on_the_ruled_out_side(episode, sides[episode["problem_id"]], "an assembled proof")
                episode["resolved_at"], (episode["assembled_proof"], episode["closer"]) = generation, won
        if after_attempt is not None:
            after_attempt(generation, len(requests), len(sources))


# ---------------------------------------------------------------------------------------------------- rows
def assembled(episodes: Sequence[Mapping]) -> list[dict]:
    """What is minimised: each episode an assembled proof resolved, with the pool and the closer it was made of."""
    return [{"episode": episode, "statement": episode["statement"], "blocks": list(episode["pool"]), "closer": episode["closer"],
             "assembled": episode["assembled_proof"]} for episode in episodes if episode["resolved_at"] is not None and episode["resolved_at"] > 0]


def harvest_rows(proofs: Sequence[Mapping], kinds: Mapping[str, tuple[str, str]]) -> list[dict]:
    """One row a problem, in the order of the problems: a training example as a round stores one (`problem_id`,
    `side`, `theorem`, `completion`), and where it came from. Two proofs of one problem would be proofs of both of
    its sides, which the alarm has stopped before this: refused."""
    rows = []
    for proof in proofs:
        episode, blocks = proof["episode"], proof["blocks"]
        kind, published_side = kinds[episode["problem_id"]]
        pooled = sorted({block["generation"] for block in blocks} | {proof["closer"]["generation"]})
        rows.append({"problem_id": episode["problem_id"], "side": episode["side"], "theorem": episode["statement"], "completion": proof["completion"],
                     "kind": kind, "published_side": published_side, "assembled": True,
                     # The attempt after which the proof verified, of how many distinct failed attempts replayed and how many stored.
                     "resolved_after_attempt": episode["resolved_at"], "attempts_replayed": len(episode["chain"]), "stored_attempts": episode["stored_attempts"],
                     # The attempts whose lemmas the pool held and whose closing step closed it, and the runs they were stored by.
                     "pooled_attempts": pooled, "runs": sorted({episode["chain"][generation]["from"] for generation in pooled}),
                     "blocks_before": len(blocks), "blocks_after": len(proof["kept"]), "lines_before": proof_line_count(proof["assembled"]),
                     "lines_after": proof_line_count(proof["completion"]), "minimise_checks": proof["minimise_checks"], "minimised": proof["minimised"]})
    twice = sorted(problem for problem, count in Counter(row["problem_id"] for row in rows).items() if count > 1)
    if twice:
        raise SoundnessAlarm(f"{len(twice)} problems have an assembled proof on both sides (first: {twice[0]}): a proof of both sides")
    return rows


def _resolved(episodes: Sequence[Mapping], kinds: Mapping[str, tuple[str, str]]) -> list[dict]:
    return [{**{key: episode[key] for key in ("problem_id", "side", "resolved_at", "assembled_proof", "closer", "stored_attempts")},
             "pool_blocks": len(episode["pool"]), "kind": kinds[episode["problem_id"]][0], "published_side": kinds[episode["problem_id"]][1]}
            for episode in episodes if episode["resolved_at"] is not None]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def summary_path(out: Path) -> Path:
    return out.with_name(out.stem + ".summary.json")


def partial_path(out: Path) -> Path:
    return out.with_name(out.name + ".partial.json")


# ---------------------------------------------------------------------------------------------- the tool
def lean_check_settings(arguments, config: dict):
    """The pool's settings as `tools/ladder_pool` builds them for its bulk check (the pin's endpoint and limits, the
    requests in flight asked for, BACKGROUND priority, a wait that covers the pool's queue), with an episode's own
    Lean limit for one proof (`ladder_loop.episode.lean_seconds`: 30 s)."""
    return background_check_settings(arguments, config, config["ladder_loop"]["episode"]["lean_seconds"])


def pool_check(pool) -> Callable[[dict[str, str]], dict[str, dict]]:
    """Lean's raw answer for every file of a batch, through ONE client: the files go in the order of their names, each
    as the ladder loop's pin sends a file (its imports first)."""
    def check(sources: dict[str, str]) -> dict[str, dict]:
        if not sources:
            return {}
        return {raw["id"]: raw for raw in pool.submit_sources({name: sources[name] for name in sorted(sources)}).result()}
    return check


def build(arguments, config: dict, check: Callable[[dict[str, str]], Mapping[str, Mapping]] | None = None) -> int:
    """The whole harvest. `check`: what answers a batch of Lean files (default: the pool; a test hands in its own)."""
    ladder = config["ladder_loop"]
    sizes = {"pool_blocks": ladder["accumulate"]["pool_blocks"], "kept_closers": ladder["accumulate"]["kept_closers"]}
    attempts = arguments.attempts or ladder["l3d"]["harvest"]["attempts"]
    pool_rows = read_jsonl(POOL_CANDIDATES)
    statements = {row["problem_id"]: row["statement"] for row in pool_rows}
    kinds = {row["problem_id"]: (row["kind"], row["side"]) for row in pool_rows}
    heldout = {row["problem_id"] for row in read_jsonl(HELDOUT)}
    stored, solved, tasks = stored_attempts(arguments.runs_root, statements, heldout)
    episodes = episodes_of(stored, solved, statements, attempts, arguments.limit)
    problems = {episode["problem_id"] for episode in episodes}
    used = sorted(len(episode["chain"]) for episode in episodes)
    print(f"pool problems no stored attempt verified: {len(problems)} | sides to replay: {len(episodes)} | distinct failed attempts used: {sum(used)}"
          f" (at most {attempts} a side) | a side's median {used[len(used) // 2] if used else 0}", flush=True)
    if arguments.dry_run:
        print(f"dry run: {len(tasks)} task directories read, {sum(used)} re-checks to send first; nothing was sent and nothing was written", flush=True)
        return 0
    stats: Counter = Counter()
    sides = {problem: side for problem, (_, side) in kinds.items()}
    out = arguments.out

    def after_attempt(generation: int, pool_checks: int, closer_checks: int) -> None:
        won = {episode["problem_id"] for episode in episodes if episode["resolved_at"] is not None and episode["resolved_at"] > 0}
        if generation in REPORTED_AFTER or generation == attempts:
            print(f"after attempt {generation}: problems resolved by an assembled proof {len(won)} | pool checks {pool_checks}, closer checks {closer_checks}", flush=True)
        if generation % KEPT_EVERY == 0 or generation == attempts:          # kept as it goes: a long job
            _write(partial_path(out), {"attempts_done": generation, "problems": len(problems), "stats": dict(stats), "resolved": _resolved(episodes, kinds)})

    def run(ask: Callable) -> list[dict]:
        try:
            replay_stored(episodes, ask, sizes, attempts, sides, stats, after_attempt)
        except SoundnessAlarm:
            _write(partial_path(out), {"soundness_alarm": True, "problems": len(problems), "stats": dict(stats), "resolved": _resolved(episodes, kinds)})
            raise
        proofs = assembled(episodes)
        print(f"assembled proofs to minimise: {len(proofs)} ({sum(len(proof['blocks']) for proof in proofs)} pool blocks)", flush=True)
        replay.minimise(proofs, ask, stats)
        return proofs

    if check is None:
        settings = lean_check_settings(arguments, config)
        print(f"the Lean pool: {in_flight_text(settings)}, priority {settings.priority}, Lean timeout {settings.lean_timeout_seconds} s", flush=True)
        pool = LeanCheckPool(settings)
        try:
            proofs = run(pool_check(pool))
        finally:
            pool.close()
    else:
        proofs = run(check)
    rows = harvest_rows(proofs, kinds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    again = sorted(episode["problem_id"] for episode in episodes if episode["resolved_at"] is not None and episode["resolved_at"] < 0)
    lines = sorted(row["lines_after"] for row in rows)
    summary = {
        "spec": "docs/spec/ladder-loop.spec.md, L3d: train on what the episode reaches (the harvest H0)",
        "rows": len(rows), "problems_no_stored_attempt_verified": len(problems), "sides_replayed": len(episodes), "attempts_replayed": sum(used),
        "attempts_a_side_at_most": attempts, "limit": arguments.limit, "sizes": sizes,
        "a_pool_stood_on": len({episode["problem_id"] for episode in episodes if episode["pool"]}),
        "resolved_by_an_assembled_proof": len(rows),
        "a_stored_failure_verified_on_the_recheck": {"problems": len(set(again)), "what": "counted apart: no assembled proof, not written", "problem_ids": sorted(set(again))},
        "by_side": dict(Counter(row["side"] for row in rows)), "by_kind": dict(Counter(row["kind"] for row in rows)),
        "resolved_after_attempt": {str(key): count for key, count in sorted(Counter(row["resolved_after_attempt"] for row in rows).items())},
        "minimisation": {"blocks_before": sum(row["blocks_before"] for row in rows), "blocks_after": sum(row["blocks_after"] for row in rows),
                         "proofs_made_shorter": sum(row["blocks_after"] < row["blocks_before"] for row in rows),
                         "proofs_used_as_assembled": sum(not row["minimised"] for row in rows), "checks": sum(row["minimise_checks"] for row in rows)},
        "proof_lines": {"median": lines[len(lines) // 2] if lines else None, "longest": lines[-1] if lines else None},
        "task_directories_read": tasks, "check_statuses": dict(sorted(stats.items())),
        "timeouts": {key: count for key, count in sorted(stats.items()) if "timeout" in key},
    }
    _write(summary_path(out), summary)
    partial_path(out).unlink(missing_ok=True)
    print(f"of {len(problems)} pool problems no stored attempt verified: a pool stood on {summary['a_pool_stood_on']}, assembly resolves {len(rows)}"
          f" | a stored failure verified on the re-check: {len(set(again))}", flush=True)
    print("stats:", dict(stats), flush=True)
    print(f"wrote {out} ({len(rows)} rows) and {summary_path(out)}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--runs-root", type=Path, required=True, help="the directory the ladder runs' outputs were pulled to (one directory a task; read only)")
    parser.add_argument("--out", type=Path, default=None, help="the harvest to write (src/rlvr_lean/data/ladder_l3d/harvest_h0.jsonl); its summary is written beside it")
    parser.add_argument("--limit", type=int, default=None, help="a trial: only the first this many sides")
    parser.add_argument("--attempts", type=int, default=None, help="at most this many distinct failed attempts a side (default: ladder_loop.l3d.harvest.attempts)")
    parser.add_argument("--api-key-file", type=Path, help="the pool's key; read, never printed")
    parser.add_argument("--ca-file", type=Path, help="the pool authority's certificate")
    parser.add_argument("--in-flight", type=in_flight_argument, default=None,
                        help="requests in flight: a number (default: lean.pool.concurrent_requests), or `auto` to follow the size the pool states. "
                             "The checks are background work: the pool takes them only when no solver's check is waiting")
    parser.add_argument("--dry-run", action="store_true", help="read what is stored, select and count; send nothing, write nothing")
    arguments = parser.parse_args(argv)
    if not arguments.dry_run and (arguments.out is None or arguments.api_key_file is None or arguments.ca_file is None):
        parser.error("without --dry-run the tool needs --out, --api-key-file and --ca-file")
    try:
        return build(arguments, yaml.safe_load(arguments.config.read_text()))
    except SoundnessAlarm as alarm:
        print(f"SOUNDNESS ALARM: {alarm}", file=sys.stderr, flush=True)
        return SOUNDNESS_ALARM_EXIT


if __name__ == "__main__":
    sys.exit(main())
