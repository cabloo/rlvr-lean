"""A model's stored one-shot attempts at the goal set, as episodes of 8 with the episode's Lean-only assembly added.
Spec: docs/spec/ladder-loop.spec.md, "L3d: train on what the episode reaches" ("The aim, as a count"; "Measured":
"After the run, Lean only: each model's stored goal attempts cut into episodes of 8 in the order drawn, with assembly,
for the reliable count"). No GPU and no generation: only the Lean pool is used, as BACKGROUND work, and only held-out
statements, stored model attempts at them and mechanical recombinations of those are sent. Nothing here is trained on.

  replay   For ONE model: its stored attempts at the goal problems (`--attempts <steps directory> <set name>`, once
           for each stored sampling, in the order given: say `<the L1 run>/steps reach_base` and `<the L2 run>/steps
           control`), each side's cut into episodes of 8 in the order drawn (the last few are left over). An episode
           ends at its first verified attempt. Every failed attempt before that is checked again for Lean's error
           positions (the stored rows hold none) and replayed as `tools/ladder_harvest` replays one
           (`domain/repair/replay.py`: the harvest into the pool, the pool's check, the kept closers), with L3c's
           rules and sizes. Writes `--out`: one row an episode (`blind_at`: the attempt that verified; `resolved_at`
           and `how`: the attempt after which it was resolved, by an `attempt` or an `assembled` proof) and a summary:
           the episodes resolved by attempts alone and with assembly, and the goal problems solved at least once, in a
           quarter, in half ("reliably") and in nine tenths of their episodes, both ways, by the length group of the
           shortest published proof. `--problems`: a run's `problems.jsonl` that holds the goal problems with their
           groups, statements and exact negations (a stored L3c run's). `--dry-run` reads, cuts and counts
  report   prints the summaries of several replays (their `--out` files) side by side

An assembled proof on the side a problem's published answer rules out is a SOUNDNESS ALARM: the tool stops (exit 3),
with nothing written. The pool caches by a file's text: a rerun is answered from the cache.

    PYTHONPATH=src python -m rlvr_lean.tools.ladder_goal_assembly replay --model base --problems <an L3c run>/steps/problems.jsonl \\
        --attempts <the L1 run>/steps reach_base --attempts <the L2 run>/steps control --out <base.json> \\
        --api-key-file <the pool's key file> --ca-file <the pool authority's certificate> [--in-flight 24 | auto] [--limit N]
    PYTHONPATH=src python -m rlvr_lean.tools.ladder_goal_assembly report <base.json> <without.json> <with.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml

from rlvr_lean.data.heldout_proof_lines import LENGTH_GROUPS
from rlvr_lean.domain.ladder_round.read import GOAL
from rlvr_lean.domain.ladder_round.reliable import EPISODE_ATTEMPTS, reliability
from rlvr_lean.domain.problem_pool.episodes import NEGATION_SIDE, VERIFIED
from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm
from rlvr_lean.domain.repair import cut as rules
from rlvr_lean.domain.repair import replay
from rlvr_lean.domain.verification.lean_source import build_proof_source, find_forbidden_token
from rlvr_lean.infrastructure.verification_service import LeanCheckPool
from rlvr_lean.tools.ladder_harvest import (CONFIG, PACKAGE, SOUNDNESS_ALARM_EXIT, in_flight_argument, in_flight_text, lean_check_settings, pool_check,
                                            raise_on_the_ruled_out_side, read_jsonl)

PROOF_LINES = PACKAGE / "data" / "ladder_l0" / "heldout_proof_lines.jsonl"      # each held-out problem's shortest published proof: the summary's groups
ATTEMPT, ASSEMBLED = "attempt", "assembled"                                      # how an episode was resolved
FOUR_PLUS, ALL, UNKNOWN = "4_or_more", "all", "not_known"
COUNTS = ("solved_at_least_once", "in_a_quarter", "reliably", "in_nine_tenths")


# ------------------------------------------------------------------------------------------------ episodes
def goal_episodes(problems: Mapping[str, Mapping], samplings: Sequence[tuple[str, Sequence[Mapping]]], limit: int | None = None) -> list[dict]:
    """Every (goal problem, side)'s stored attempts of each sampling, cut into episodes of 8 in the order drawn.
    `samplings`: (the set's name, its stored attempt rows in the files' order), in the order given. An episode holds
    its 8 rows as `chain` (attempt 1 to 8) and `blind_at`: the first attempt that verified, or None."""
    episodes = []
    for set_name, attempts in samplings:
        rows = defaultdict(list)
        for row in attempts:
            if row["problem_id"] in problems and problems[row["problem_id"]]["group"] == GOAL:
                rows[(row["problem_id"], row["side"])].append(row)
        for (problem, side), found in sorted(rows.items()):
            found.sort(key=lambda row: row["episode"])
            statement = problems[problem]["negation"] if side == NEGATION_SIDE else problems[problem]["statement"]
            for start in range(0, len(found) - (EPISODE_ATTEMPTS - 1), EPISODE_ATTEMPTS):
                chain = {index + 1: row for index, row in enumerate(found[start:start + EPISODE_ATTEMPTS])}
                episodes.append({**replay.new_episode(statement), "problem_id": problem, "side": side, "sampling": set_name, "episode": start // EPISODE_ATTEMPTS,
                                 "chain": chain, "blind_at": next((generation for generation in sorted(chain) if chain[generation]["status"] == VERIFIED), None),
                                 "resolved_at": None, "how": None, "assembled_proof": None, "published_side": problems[problem].get("side")})
    return episodes[:limit] if limit else episodes


def rechecks(episodes: Sequence[dict]) -> dict[str, str]:
    """The Lean files of the first check: every failed attempt (status `lean_error`, no forbidden token) that came
    before its episode's first verified one, checked again for Lean's error positions. Each such row gets its `file`."""
    sources = {}
    for episode in episodes:
        for generation, row in episode["chain"].items():
            if episode["blind_at"] is not None and generation >= episode["blind_at"]:
                continue                                              # the episode ended there: nothing after it is replayed
            if row["status"] == "lean_error" and row.get("completion") and find_forbidden_token(row["completion"]) is None:
                source = build_proof_source(episode["statement"], row["completion"])
                row["file"] = replay.name_of(source)
                sources[row["file"]] = source
    return sources


def replay_goal(episodes: Sequence[dict], check: Callable[[dict[str, str]], Mapping[str, Mapping]], sizes: Mapping, stats: Counter) -> None:
    """Re-check, then replay attempt 1 to 8 of every episode: it is resolved by its first verified `attempt`, or
    earlier by an `assembled` proof (a kept closer that verifies after the pool its failed attempts left)."""
    sources = rechecks(episodes)
    print("re-checks for error positions:", len(sources), flush=True)
    answers = check(sources)
    for episode in episodes:
        for row in episode["chain"].values():
            if "file" in row:
                verdict = replay.judged(row["file"], answers[row["file"]])
                stats["recheck:" + verdict.status.value] += 1
                row["errors"] = None if verdict.is_verified else rules.errors_of(answers[row["file"]])
                if verdict.is_verified and episode["published_side"] is not None:      # stored as an error, verified now: no attempt's success, but the alarm
                    raise_on_the_ruled_out_side(episode, episode["published_side"], "a stored attempt, checked again,")
    print("re-checks done:", dict(stats), flush=True)
    for generation in range(1, EPISODE_ATTEMPTS + 1):
        holders, requests = [], {}
        for episode in episodes:
            if episode["resolved_at"] is not None:
                continue
            row = episode["chain"][generation]
            if row["status"] == VERIFIED:
                episode["resolved_at"], episode["how"] = generation, ATTEMPT
                continue
            if not row.get("errors"):
                continue
            after = replay.harvested(episode, row["completion"], row["errors"], generation, sizes["pool_blocks"], stats)
            if after is None:
                continue
            holders.append((episode, after))
            if after["pool"] is not None:
                requests[after["pool"]["file"]] = after["pool"]["source"]
        answers = check(requests)
        stats["pool_checks"] += len(requests)
        sources = {}
        for episode, after in holders:
            sources.update(replay.closers_to_send(episode, after, answers, sizes["kept_closers"], stats))
        answers = check(sources)
        stats["closer_checks"] += len(sources)
        for episode, after in holders:
            won = replay.first_verified(episode, after, answers, stats)
            if won is not None and episode["resolved_at"] is None:
                if episode["published_side"] is not None:             # a proof of the side the published answer rules out: the alarm
                    raise_on_the_ruled_out_side(episode, episode["published_side"], "an assembled proof")
                episode["resolved_at"], episode["how"], episode["assembled_proof"] = generation, ASSEMBLED, won[0]
        print(f"within {generation}: with assembly {sum(episode['resolved_at'] is not None for episode in episodes)} | attempts alone "
              f"{sum(episode['blind_at'] is not None and episode['blind_at'] <= generation for episode in episodes)}"
              f" | pool checks {len(requests)}, closer checks {len(sources)}", flush=True)


# ------------------------------------------------------------------------------------------------- summary
def episode_rows(episodes: Sequence[Mapping]) -> list[dict]:
    return [{**{key: episode[key] for key in ("problem_id", "side", "sampling", "episode", "blind_at", "resolved_at", "how", "assembled_proof")},
             "pool_blocks": len(episode["pool"])} for episode in episodes]


def problem_episodes(rows: Sequence[Mapping], key: str) -> dict[str, list[bool]]:
    """Each problem's episodes (one for each sampling and place, whichever side: an episode is one attempt at each
    side a problem is attempted on) and whether each was resolved: by `key`, `blind_at` (attempts alone) or
    `resolved_at` (with assembly), on either side."""
    found: dict[str, dict[tuple[str, int], bool]] = {}
    for row in rows:
        place = found.setdefault(row["problem_id"], {})
        place[(row["sampling"], row["episode"])] = place.get((row["sampling"], row["episode"]), False) or row[key] is not None
    return {problem: [resolved for _, resolved in sorted(places.items())] for problem, places in found.items()}


def summary_of(rows: Sequence[Mapping], lengths: Mapping[str, Mapping]) -> dict:
    """The replay's read: episodes (a side's 8 attempts) resolved by attempts alone and with assembly; the goal
    problems solved at least once, in a quarter, in half ("reliably") and in nine tenths of their episodes, both ways,
    on all of them and by the length group of the shortest published proof."""
    problems = sorted({row["problem_id"] for row in rows})
    group_of = {problem: (lengths.get(problem) or {}).get("length_group") or UNKNOWN for problem in problems}
    groups = (*LENGTH_GROUPS, UNKNOWN) if UNKNOWN in group_of.values() else LENGTH_GROUPS
    sets = {ALL: problems, **{group: [problem for problem in problems if group_of[problem] == group] for group in groups},
            FOUR_PLUS: [problem for problem in problems if group_of[problem] in ("4-7", "8+")]}
    ways = {"by_attempts_alone": problem_episodes(rows, "blind_at"), "with_assembly": problem_episodes(rows, "resolved_at")}
    proofs = sorted(len([line for line in row["assembled_proof"].strip("\n").split("\n") if line.strip()]) for row in rows if row["how"] == ASSEMBLED)
    return {"episodes_of_one_side": len(rows), "resolved_by_attempts_alone": sum(row["blind_at"] is not None for row in rows),
            "resolved_with_assembly": sum(row["resolved_at"] is not None for row in rows),
            "resolved_by_an_assembled_proof": len(proofs), "how": dict(Counter(str(row["how"]) for row in rows)),
            "assembled_proof_lines": {"median": proofs[len(proofs) // 2] if proofs else None, "longest": proofs[-1] if proofs else None},
            "goal_problems": {way: {name: reliability(episodes, ids) for name, ids in sets.items()} for way, episodes in ways.items()}}


def report_lines(outputs: Sequence[Mapping]) -> list[str]:
    """Several replays side by side: a line a model for all of the goal set, then one for each length group."""
    lines = ["goal problems solved at least once / in a quarter / in half (reliably) / in nine tenths of their episodes of 8: by attempts alone -> with assembly"]
    for output in outputs:
        summary = output["summary"]
        alone, assembly = summary["goal_problems"]["by_attempts_alone"], summary["goal_problems"]["with_assembly"]
        lines.append(f"{output['model']}: {alone[ALL]['problems']} problems, {alone[ALL]['episodes_a_problem']} episodes a problem; episodes resolved "
                     f"{summary['resolved_by_attempts_alone']} -> {summary['resolved_with_assembly']} of {summary['episodes_of_one_side']} "
                     f"({summary['resolved_by_an_assembled_proof']} by an assembled proof)")
        for name in alone:
            lines.append(f"    {name:>9}: " + " / ".join(f"{alone[name][count]} -> {assembly[name][count]}" for count in COUNTS) + f"   of {alone[name]['problems']}")
    return lines


# ---------------------------------------------------------------------------------------------- the tool
def run_replay(arguments, config: dict, check: Callable[[dict[str, str]], Mapping[str, Mapping]] | None = None) -> int:
    ladder = config["ladder_loop"]
    sizes = {"pool_blocks": ladder["accumulate"]["pool_blocks"], "kept_closers": ladder["accumulate"]["kept_closers"]}
    problems = {row["problem_id"]: row for row in read_jsonl(arguments.problems)}
    samplings = []
    for directory, set_name in arguments.attempts:
        paths = sorted(glob.glob(f"{directory}/episodes_{set_name}_attempts_*.jsonl"))
        if not paths:
            raise ValueError(f"{directory} holds no stored attempts of the set {set_name} (episodes_{set_name}_attempts_*.jsonl): refused, nothing was sent")
        samplings.append((set_name, [row for path in paths for row in read_jsonl(Path(path))]))
    episodes = goal_episodes(problems, samplings, arguments.limit)
    print(f"{arguments.model}: goal problems {len({episode['problem_id'] for episode in episodes})} | episodes of {EPISODE_ATTEMPTS} attempts {len(episodes)} | "
          f"resolved by an attempt: {sum(episode['blind_at'] is not None for episode in episodes)}", flush=True)
    if arguments.dry_run:
        print(f"dry run: {len(rechecks(episodes))} re-checks to send first; nothing was sent and nothing was written", flush=True)
        return 0
    stats: Counter = Counter()
    if check is None:
        settings = lean_check_settings(arguments, config)
        print(f"the Lean pool: {in_flight_text(settings)}, priority {settings.priority}, Lean timeout {settings.lean_timeout_seconds} s", flush=True)
        pool = LeanCheckPool(settings)
        try:
            replay_goal(episodes, pool_check(pool), sizes, stats)
        finally:
            pool.close()
    else:
        replay_goal(episodes, check, sizes, stats)
    rows = episode_rows(episodes)
    output = {"model": arguments.model, "sources": [{"steps": str(directory), "set": set_name} for directory, set_name in arguments.attempts],
              "sizes": sizes, "stats": dict(stats), "summary": summary_of(rows, {row["problem_id"]: row for row in read_jsonl(PROOF_LINES)}), "episodes": rows}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print("stats:", dict(stats), flush=True)
    print("how resolved:", dict(Counter(row["how"] for row in rows)), flush=True)
    for line in report_lines([output]):
        print(line, flush=True)
    print("wrote", arguments.out, flush=True)
    return 0


def run_report(arguments) -> int:
    for line in report_lines([json.loads(Path(path).read_text(encoding="utf-8")) for path in arguments.outputs]):
        print(line, flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("replay", help="one model's stored goal attempts, as episodes of 8 with assembly")
    one.add_argument("--config", type=Path, default=CONFIG)
    one.add_argument("--model", required=True, help="what the model is called in the output (base, without, with, ...)")
    one.add_argument("--problems", type=Path, required=True, help="a run's problems.jsonl with the goal problems: group, statement, negation (read only)")
    one.add_argument("--attempts", nargs=2, action="append", required=True, metavar=("STEPS_DIRECTORY", "SET"),
                     help="a stored sampling of the goal set: a pulled run's steps directory and the set's name; once for each sampling, in order")
    one.add_argument("--out", type=Path, default=None, help="the JSON to write: the summary and one row an episode")
    one.add_argument("--limit", type=int, default=None, help="a trial: only the first this many episodes")
    one.add_argument("--api-key-file", type=Path, help="the pool's key; read, never printed")
    one.add_argument("--ca-file", type=Path, help="the pool authority's certificate")
    one.add_argument("--in-flight", type=in_flight_argument, default=None, help="requests in flight: a number, or `auto` (as tools/ladder_pool)")
    one.add_argument("--dry-run", action="store_true", help="read, cut into episodes and count; send nothing, write nothing")
    several = commands.add_parser("report", help="the summaries of several replays, side by side")
    several.add_argument("outputs", nargs="+", type=Path, help="the --out files of replays")
    arguments = parser.parse_args(argv)
    if arguments.command == "report":
        return run_report(arguments)
    if not arguments.dry_run and (arguments.out is None or arguments.api_key_file is None or arguments.ca_file is None):
        parser.error("without --dry-run a replay needs --out, --api-key-file and --ca-file")
    try:
        return run_replay(arguments, yaml.safe_load(arguments.config.read_text()))
    except SoundnessAlarm as alarm:
        print(f"SOUNDNESS ALARM: {alarm}", file=sys.stderr, flush=True)
        return SOUNDNESS_ALARM_EXIT


if __name__ == "__main__":
    sys.exit(main())
